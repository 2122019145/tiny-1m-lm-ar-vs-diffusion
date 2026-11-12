"""Unified trainer for the AR and diffusion models (CPU-friendly)."""
import argparse
import json
import math
import os
import time

import numpy as np
import torch

from model import ARModel, Config, DiffusionLM


def load_meta(path="data/meta.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class TokenDataset:
    """Random fixed-length windows over a flat token array."""

    def __init__(self, path, seq_len, seed=0):
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        self.seq_len = seq_len
        self.n = len(self.data)
        self.rng = np.random.default_rng(seed)

    def sample(self, batch_size, generator=None):
        hi = self.n - self.seq_len - 1
        if generator is not None:
            # torch generator for reproducibility across models
            starts = torch.randint(0, hi, (batch_size,), generator=generator).numpy()
        else:
            starts = self.rng.integers(0, hi, size=batch_size)
        idx = np.stack([self.data[s:s + self.seq_len].astype(np.int64) for s in starts])
        tgt = np.stack([self.data[s + 1:s + self.seq_len + 1].astype(np.int64) for s in starts])
        return torch.from_numpy(idx), torch.from_numpy(tgt)


def lr_at(step, args):
    if step < args.warmup:
        return args.lr * (step + 1) / max(1, args.warmup)
    p = (step - args.warmup) / max(1, args.max_steps - args.warmup)
    p = min(1.0, max(0.0, p))
    return args.lr_min + 0.5 * (args.lr - args.lr_min) * (1 + math.cos(math.pi * p))


@torch.no_grad()
def eval_ar(model, ds, args, n_batches=8, generator=None):
    model.eval()
    tot, cnt = 0.0, 0
    for _ in range(n_batches):
        x, y = ds.sample(args.eval_batch, generator)
        _, loss = model(x, y)
        tot += loss.item()
        cnt += 1
    model.train()
    return tot / cnt


@torch.no_grad()
def eval_diffusion_ce(model, ds, args, n_batches=8, generator=None):
    """Average denoising cross-entropy at random mask ratios (training objective).

    The denoising target is the sequence itself: position i must be recovered
    from its context, so we pass x as both input and target.
    """
    model.eval()
    tot, cnt = 0.0, 0
    for _ in range(n_batches):
        x, _ = ds.sample(args.eval_batch, generator)
        _, loss = model(x, x, generator=generator)
        tot += loss.item()
        cnt += 1
    model.train()
    return tot / cnt


@torch.no_grad()
def eval_diffusion_elbo_bpc(model, ds, args, n_batches=4, n_mc=64, generator=None):
    """Monte-Carlo estimate of the absorbing-diffusion NELBO, in bits/char.

    NLL <= E_t [ (1/t) * (1/T) * sum_{i masked} -log p(x_i | x_masked) ],  t~U(0,1).

    The plain 1/t form has heavy tails (a single masked token at t=0.01 gets
    weight 100), so we use the algebraically identical ratio estimator

        (1/T) * (1/t) * sum_masked = (n_masked / (T*t)) * mean_masked(-log p)

    where n_masked/(T*t) is close to 1 for every t; this keeps the estimate
    stable instead of dominated by one unlucky draw.
    """
    model.eval()
    T = args.seq_len
    M = model.cfg.mask_token_id
    tot, cnt = 0.0, 0
    for _ in range(n_batches):
        x, _ = ds.sample(args.eval_batch, generator)
        B = x.size(0)
        acc = 0.0
        for k in range(n_mc):
            t = (k + 0.5) / n_mc                 # stratified t over (0,1)
            mask = torch.rand(B, T, generator=generator) < t
            mask[:, 0] = mask[:, 0] | ~mask.any(dim=1)
            noisy = torch.where(mask, torch.full_like(x, M), x)
            logits, _ = model(noisy)
            logp = torch.log_softmax(logits.float(), dim=-1)
            nll = -logp.gather(-1, x.unsqueeze(-1)).squeeze(-1)      # (B,T)
            n_masked = mask.sum(dim=1).clamp(min=1)
            mean_masked = (nll * mask).sum(dim=1) / n_masked         # (B,)
            # (n_masked/(T*t)) * mean_masked == (1/t) * sum_masked / T
            acc += float((mean_masked * (n_masked / (T * t))).mean())
        tot += acc / n_mc
        cnt += 1
    model.train()
    return tot / cnt / math.log(2)


def build_model(kind, meta, args):
    cfg = Config(vocab_size=meta["vocab_size"], n_layer=args.n_layer,
                 n_head=args.n_head, d_model=args.d_model, d_ff=args.d_ff,
                 max_seq_len=args.seq_len, dropout=args.dropout)
    cfg.mask_token_id = meta["mask_id"]
    cfg.eos_id = meta["eos_id"]
    return ARModel(cfg) if kind == "ar" else DiffusionLM(cfg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["ar", "diffusion"], required=True)
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--seq_len", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--eval_batch", type=int, default=16)
    ap.add_argument("--n_layer", type=int, default=4)
    ap.add_argument("--n_head", type=int, default=4)
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--d_ff", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--lr_min", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--max_steps", type=int, default=3000)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--log_every", type=int, default=25)
    ap.add_argument("--eval_every", type=int, default=250)
    ap.add_argument("--ckpt_every", type=int, default=500)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    if args.threads > 0:
        torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    os.makedirs(args.run_dir, exist_ok=True)

    meta = load_meta(os.path.join(args.data_dir, "meta.json"))
    train_ds = TokenDataset(os.path.join(args.data_dir, "train.bin"), args.seq_len, seed=args.seed)
    val_ds = TokenDataset(os.path.join(args.data_dir, "val.bin"), args.seq_len, seed=args.seed + 1)

    model = build_model(args.arch, meta, args)
    n_all = sum(p.numel() for p in model.parameters())
    n_emb = model.tok_emb.weight.numel()
    print(f"[model] arch={args.arch} params={n_all:,} (non-embedding {n_all-n_emb:,})")
    print(f"[data]  train_tokens={train_ds.n:,} val_tokens={val_ds.n:,} "
          f"vocab={meta['vocab_size']} seq_len={args.seq_len}")
    print(f"[cfg]   batch={args.batch_size} steps={args.max_steps} "
          f"tokens/step={args.batch_size*args.seq_len:,} "
          f"total_tokens={args.batch_size*args.seq_len*args.max_steps:,}")

    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if p.ndim >= 2:
            decay.append(p)
        else:
            no_decay.append(p)
    opt = torch.optim.AdamW([
        {"params": decay, "weight_decay": args.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ], lr=args.lr, betas=(0.9, 0.95), eps=1e-8)

    gen = torch.Generator().manual_seed(args.seed)
    start_step = 0
    history = []
    ckpt_path = os.path.join(args.run_dir, "latest.pt")
    if args.resume and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start_step = ck["step"]
        history = ck.get("history", [])
        print(f"[resume] from step {start_step}")

    t0 = time.time()
    tokens_done = 0
    model.train()
    for step in range(start_step, args.max_steps):
        lr = lr_at(step, args)
        for g in opt.param_groups:
            g["lr"] = lr
        x, y = train_ds.sample(args.batch_size, gen)
        if args.arch == "ar":
            _, loss = model(x, y)
        else:
            # absorbing diffusion denoises x itself: the target at a masked
            # position i is x[i], NOT the next token y[i].  Training on y let the
            # model read x[i] from the (visible) neighbour at i+1, a degenerate
            # objective whose loss looked good but produced garbage samples.
            _, loss = model(x, x, generator=gen)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        tokens_done += args.batch_size * args.seq_len

        if step % args.log_every == 0 or step == args.max_steps - 1:
            dt = time.time() - t0
            tps = tokens_done / max(dt, 1e-6)
            bpc = loss.item() / math.log(2)
            print(f"step {step:5d}/{args.max_steps} loss {loss.item():.4f} "
                  f"bpc {bpc:.4f} lr {lr:.2e} gnorm {float(gn):.2f} "
                  f"{tps/1000:.1f}k tok/s eta {((args.max_steps-step)*args.batch_size*args.seq_len/max(tps,1))/60:.1f}m",
                  flush=True)

        if (step + 1) % args.eval_every == 0 or step == args.max_steps - 1:
            if args.arch == "ar":
                v = eval_ar(model, val_ds, args)
                entry = {"step": step + 1, "val_loss": v, "val_bpc": v / math.log(2)}
            else:
                v = eval_diffusion_ce(model, val_ds, args)
                b = eval_diffusion_elbo_bpc(model, val_ds, args)
                entry = {"step": step + 1, "val_ce": v, "val_ce_bpc": v / math.log(2),
                         "val_elbo_bpc": b}
            entry["train_loss"] = loss.item()
            entry["lr"] = lr
            entry["elapsed_s"] = round(time.time() - t0, 1)
            history.append(entry)
            print(f"[eval] {json.dumps(entry)}", flush=True)
            with open(os.path.join(args.run_dir, "history.json"), "w") as f:
                json.dump(history, f, indent=1)

        if (step + 1) % args.ckpt_every == 0 or step == args.max_steps - 1:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "step": step + 1, "args": vars(args), "history": history},
                       ckpt_path)
            torch.save({"model": model.state_dict(), "args": vars(args),
                        "meta": meta, "history": history},
                       os.path.join(args.run_dir, "model.pt"))

    total = time.time() - t0
    print(f"[done] {total/60:.1f} min, {tokens_done/1e6:.1f}M tokens")
    with open(os.path.join(args.run_dir, "summary.json"), "w") as f:
        json.dump({"arch": args.arch, "params": n_all, "params_non_embedding": n_all - n_emb,
                   "tokens": tokens_done, "steps": args.max_steps,
                   "minutes": round(total / 60, 2), "history": history}, f, indent=1)


if __name__ == "__main__":
    main()
