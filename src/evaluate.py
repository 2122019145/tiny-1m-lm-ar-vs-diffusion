"""Final evaluation of the two checkpoints on the held-out validation split.

AR        : exact negative log-likelihood under teacher forcing -> bits/char.
Diffusion : absorbing-state masked diffusion, two quantities are reported.

  (a) denoise_ce_bpc -- E_{t~U(0,1)} E_{mask} [ mean_{i masked} -log p(x0_i | xt) ].
      A stable number, but it is NOT an upper bound on the NLL, so it must not
      be compared to the AR bpc as if it were one.

  (b) elbo_bpc -- the variational upper bound on the NLL,
          NLL <= E_t E_{xt} [ (1/t) * (1/T) * sum_{i masked} -log p(x0_i | xt) ].
      Computed as (n_masked/(T*t)) * mean_masked, which equals the textbook
      (1/t) * sum_masked / T but keeps the per-t weight near 1 so a single
      unlucky small-t draw cannot dominate. THIS is the number comparable to the
      AR bits/char: it bounds the same quantity, so elbo_bpc >= ar_bpc must hold
      for a correct implementation (up to estimator noise).
"""
import argparse
import json
import math
import os

import numpy as np
import torch

from model import ARModel, Config, DiffusionLM


class TokenDataset:
    def __init__(self, path, seq_len, seed=0):
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        self.seq_len = seq_len

    def sample(self, batch_size, generator=None):
        hi = len(self.data) - self.seq_len - 1
        starts = torch.randint(0, hi, (batch_size,), generator=generator).tolist()
        x = torch.stack([torch.from_numpy(
            np.array(self.data[s:s + self.seq_len], dtype=np.int64)) for s in starts])
        y = torch.stack([torch.from_numpy(
            np.array(self.data[s + 1:s + 1 + self.seq_len], dtype=np.int64)) for s in starts])
        return x, y


def ckpt_step(ck):
    """model.pt has no 'step' key; fall back to the last history entry."""
    if ck.get("step") is not None:
        return ck["step"]
    hist = ck.get("history") or []
    return hist[-1].get("step") if hist else None


def build_from_ckpt(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    a, meta = ck["args"], ck["meta"]
    cfg = Config(vocab_size=meta["vocab_size"], n_layer=a["n_layer"], n_head=a["n_head"],
                 d_model=a["d_model"], d_ff=a["d_ff"], max_seq_len=a["seq_len"],
                 dropout=a.get("dropout", 0.0))
    cfg.mask_token_id = meta["mask_id"]
    cfg.eos_id = meta["eos_id"]
    model = ARModel(cfg) if a["arch"] == "ar" else DiffusionLM(cfg)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, meta, ck


@torch.no_grad()
def ar_bpc(model, ds, n_batches, batch, generator):
    """Exact NLL in bits per character (teacher forcing, causal)."""
    tot_nll, tot_tok = 0.0, 0
    for _ in range(n_batches):
        x, y = ds.sample(batch, generator)
        logits, _ = model(x)
        nll = torch.nn.functional.cross_entropy(
            logits.float().view(-1, logits.size(-1)), y.reshape(-1), reduction="sum")
        tot_nll += float(nll)
        tot_tok += y.numel()
    return tot_nll / tot_tok / math.log(2)


@torch.no_grad()
def diffusion_metrics(model, ds, n_batches, batch, n_mc, generator):
    """Return (denoise_ce_bpc, elbo_bpc) for the masked-diffusion model."""
    T = ds.seq_len
    M = model.cfg.mask_token_id
    ce_acc, elbo_acc, cnt = 0.0, 0.0, 0

    for _ in range(n_batches):
        x, _ = ds.sample(batch, generator)
        B = x.size(0)
        ce_sum, elbo_sum = 0.0, 0.0
        for k in range(n_mc):
            t = (k + 0.5) / n_mc                        # stratified t over (0,1)
            mask = torch.rand(B, T, generator=generator) < t
            mask[:, 0] |= ~mask.any(dim=1)              # guarantee >= 1 masked token
            noisy = torch.where(mask, torch.full_like(x, M), x)
            logits, _ = model(noisy)
            logp = torch.log_softmax(logits.float(), dim=-1)
            tok_nll = -logp.gather(-1, x.unsqueeze(-1)).squeeze(-1)   # (B,T)
            n_masked = mask.sum(dim=1).clamp(min=1)
            masked_nll = (tok_nll * mask).sum(dim=1)                  # (B,)
            mean_masked = masked_nll / n_masked                       # (B,)
            ce_sum += float(mean_masked.mean())                       # mean over masked
            # (n_masked/(T*t)) * mean_masked == (1/t) * sum_masked / T, but keeps
            # the per-t weight near 1 so one unlucky small-t draw cannot dominate.
            elbo_sum += float((mean_masked * (n_masked / (T * t))).mean())
        ce_acc += ce_sum / n_mc
        elbo_acc += elbo_sum / n_mc
        cnt += 1

    return ce_acc / cnt / math.log(2), elbo_acc / cnt / math.log(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="../data")
    ap.add_argument("--ar_ckpt", default="../runs/ar/model.pt")
    ap.add_argument("--diff_ckpt", default="../runs/diffusion/model.pt")
    ap.add_argument("--out", default="../out/eval.json")
    ap.add_argument("--seq_len", type=int, default=256)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--n_batches", type=int, default=16)
    ap.add_argument("--n_mc", type=int, default=128)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    g = torch.Generator().manual_seed(args.seed)
    ds = TokenDataset(os.path.join(args.data_dir, "val.bin"), args.seq_len)
    meta = json.load(open(os.path.join(args.data_dir, "meta.json"), encoding="utf-8"))
    result = {"val_tokens": meta["val_tokens"], "vocab_size": meta["vocab_size"],
              "eval_batches": args.n_batches, "eval_batch": args.batch,
              "tokens_evaluated": args.n_batches * args.batch * args.seq_len}

    if os.path.exists(args.ar_ckpt):
        m, _, ck = build_from_ckpt(args.ar_ckpt)
        bpc = ar_bpc(m, ds, args.n_batches, args.batch, g)
        result["ar"] = {"params": sum(p.numel() for p in m.parameters()),
                        "step": ckpt_step(ck), "val_bpc": bpc,
                        "val_loss_nats": bpc * math.log(2),
                        "perplexity_per_char": 2.0 ** bpc}
        print(f"[AR]        step={ckpt_step(ck)}  val_bpc={bpc:.4f}  ppl/char={2.0**bpc:.2f}")

    if os.path.exists(args.diff_ckpt):
        m, _, ck = build_from_ckpt(args.diff_ckpt)
        ce_bpc, elbo_bpc = diffusion_metrics(m, ds, args.n_batches, args.batch,
                                             args.n_mc, g)
        result["diffusion"] = {"params": sum(p.numel() for p in m.parameters()),
                               "step": ckpt_step(ck),
                               "val_denoise_ce_bpc": ce_bpc,
                               "val_elbo_bpc": elbo_bpc,
                               # the NLL bound is the ELBO, not the denoising CE
                               "val_loss_nats": elbo_bpc * math.log(2),
                               "perplexity_per_char": 2.0 ** elbo_bpc}
        print(f"[Diffusion] step={ckpt_step(ck)}  denoise_ce_bpc={ce_bpc:.4f}  "
              f"elbo_bpc={elbo_bpc:.4f}")

    if "ar" in result and "diffusion" in result:
        # only the ELBO bounds the same quantity as the AR NLL, so this is the
        # honest gap; the denoising CE is reported separately and is not a bound.
        gap = result["diffusion"]["val_elbo_bpc"] - result["ar"]["val_bpc"]
        result["bpc_gap_diffusion_minus_ar"] = gap
        result["bpc_gap_denoise_ce_minus_ar"] = (
            result["diffusion"]["val_denoise_ce_bpc"] - result["ar"]["val_bpc"])
        print(f"[gap]       diffusion ELBO - AR = {gap:+.4f} bits/char "
              f"(denoise-CE gap, NOT a bound: "
              f"{result['bpc_gap_denoise_ce_minus_ar']:+.4f})")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
