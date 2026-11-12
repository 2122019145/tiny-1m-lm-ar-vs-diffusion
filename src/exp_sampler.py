"""Compare masked-diffusion decoding strategies on the current checkpoint.

The naive scheme (unmask the k most confident positions each step, never revisit)
locks in repetition: at step 0 every position is masked, so the model outputs the
same marginal everywhere, the k most confident positions all get the same
high-frequency token, and that block is frozen forever.

ReMask-style decoding fixes this: each step we re-score *every* position (masked
ones by the probability of their new prediction, already-decoded ones by the
probability the model now assigns to their current token), then keep only the
globally most confident N and re-mask the rest. Bad early guesses can be revoked.
"""
import argparse
import json
import sys

import torch
import torch.nn.functional as F

from model import DiffusionLM, Config


def load(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    a, meta = ck["args"], ck["meta"]
    cfg = Config(vocab_size=meta["vocab_size"], n_layer=a["n_layer"], n_head=a["n_head"],
                 d_model=a["d_model"], d_ff=a["d_ff"], max_seq_len=a["seq_len"],
                 dropout=a.get("dropout", 0.0))
    cfg.mask_token_id = meta["mask_id"]
    cfg.eos_id = meta["eos_id"]
    m = DiffusionLM(cfg)
    m.load_state_dict(ck["model"])
    m.eval()
    return m, meta


def decode(ids, vocab, eos_id):
    out = []
    for i in ids:
        i = int(i)
        if i == eos_id:
            break
        if i < 3:
            continue
        out.append(vocab[i])
    return "".join(out)


@torch.no_grad()
def sample_naive(model, seq_len, steps, temperature, top_k, generator, prompt_ids=None):
    M = model.cfg.mask_token_id
    x = torch.full((1, seq_len), M, dtype=torch.long)
    if prompt_ids:
        x[0, :len(prompt_ids)] = torch.as_tensor(prompt_ids, dtype=torch.long)
    for s in range(steps):
        logits, _ = model(x)
        logits = logits / max(temperature, 1e-5)
        if top_k:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1)
            logits[logits < v[..., [-1]]] = -float("inf")
        probs = F.softmax(logits, dim=-1)
        pred = torch.multinomial(probs.view(-1, probs.size(-1)), 1,
                                 generator=generator).view(1, seq_len)
        conf = probs.gather(-1, pred.unsqueeze(-1)).squeeze(-1)
        conf = torch.where(x == M, conf, torch.full_like(conf, -1.0))
        n_left = int((x == M).sum().item())
        k = max(1, -(-n_left // (steps - s)))
        order = torch.argsort(conf[0], descending=True)[:k]
        x[0, order] = pred[0, order]
        if not (x == M).any():
            break
    return x


@torch.no_grad()
def sample_remask(model, seq_len, steps, temperature, top_k, generator,
                  prompt_ids=None, remask=True):
    """ReMask / ReMDM-style decoding: re-score all positions, keep top-N."""
    M = model.cfg.mask_token_id
    x = torch.full((1, seq_len), M, dtype=torch.long)
    n_prompt = 0
    if prompt_ids:
        n_prompt = len(prompt_ids)
        x[0, :n_prompt] = torch.as_tensor(prompt_ids, dtype=torch.long)
    fixed = torch.zeros(seq_len, dtype=torch.bool)
    fixed[:n_prompt] = True

    for s in range(steps):
        logits, _ = model(x)
        logits = logits / max(temperature, 1e-5)
        if top_k:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1)
            logits[logits < v[..., [-1]]] = -float("inf")
        probs = F.softmax(logits, dim=-1)
        pred = torch.multinomial(probs.view(-1, probs.size(-1)), 1,
                                 generator=generator).view(1, seq_len)
        # confidence of the *new* prediction at masked spots
        conf_new = probs.gather(-1, pred.unsqueeze(-1)).squeeze(-1)[0]
        # confidence of the *current* token at already-decoded spots
        conf_cur = probs.gather(-1, x.unsqueeze(-1)).squeeze(-1)[0]
        conf = torch.where(x[0] == M, conf_new, conf_cur)

        # how many positions should be committed after this step
        frac = (s + 1) / steps
        n_keep = int(round(seq_len * frac))
        n_keep = min(max(n_keep, n_prompt + 1), seq_len)

        order = torch.argsort(conf, descending=True)
        keep = torch.zeros(seq_len, dtype=torch.bool)
        keep[order[:n_keep]] = True
        keep |= fixed
        if not remask:
            keep |= (x[0] != M)
        newx = torch.where(keep, torch.where(x[0] == M, pred[0], x[0]),
                           torch.full_like(x[0], M))
        newx[fixed] = x[0, fixed]
        x[0] = newx
        if not (x == M).any():
            break
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="../runs/diffusion/model.pt")
    ap.add_argument("--seq_len", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    model, meta = load(args.ckpt)
    vocab = meta["vocab"]
    stoi = {c: i for i, c in enumerate(vocab)}
    prompt = "从前，有一个小女孩"
    pids = [stoi[c] for c in prompt if c in stoi]

    configs = [
        ("naive steps=64", lambda g: sample_naive(model, args.seq_len, 64, args.temperature, args.top_k, g)),
        ("naive steps=200", lambda g: sample_naive(model, args.seq_len, 200, args.temperature, args.top_k, g)),
        ("remask steps=64", lambda g: sample_remask(model, args.seq_len, 64, args.temperature, args.top_k, g, remask=True)),
        ("remask steps=128", lambda g: sample_remask(model, args.seq_len, 128, args.temperature, args.top_k, g, remask=True)),
        ("remask+prompt 128", lambda g: sample_remask(model, args.seq_len, 128, args.temperature, args.top_k, g, prompt_ids=pids, remask=True)),
    ]
    for name, fn in configs:
        print("=" * 70)
        print(name)
        for seed in (7, 11):
            g = torch.Generator().manual_seed(seed)
            out = fn(g)
            txt = decode(out[0].tolist(), vocab, model.cfg.eos_id)
            print(f"  [{seed}] {txt.strip()[:170]}")


if __name__ == "__main__":
    main()
