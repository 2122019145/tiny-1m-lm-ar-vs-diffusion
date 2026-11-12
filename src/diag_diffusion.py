"""Diagnose the FIXED diffusion checkpoint: loss by noise level + sampler comparison."""
import json
import math
import os
import sys

import numpy as np
import torch

from model import DiffusionLM
from sample import build_from_ckpt, decode

torch.set_num_threads(8)
ROOT = r"D:\dsh1\modeltest"


class TokenDataset:
    def __init__(self, path, seq_len):
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        self.seq_len = seq_len

    def sample(self, batch_size, generator=None):
        hi = len(self.data) - self.seq_len - 1
        starts = torch.randint(0, hi, (batch_size,), generator=generator).tolist()
        x = torch.stack([torch.from_numpy(np.array(self.data[s:s + self.seq_len], dtype=np.int64))
                         for s in starts])
        return x


@torch.no_grad()
def loss_by_t(model, ds, ts, n_batch=8, batch=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    T = ds.seq_len
    M = model.cfg.mask_token_id
    out = []
    for t in ts:
        tot, n = 0.0, 0
        for _ in range(n_batch):
            x = ds.sample(batch, g)
            mask = torch.rand(x.size(0), T, generator=g) < t
            mask[:, 0] |= ~mask.any(dim=1)
            noisy = torch.where(mask, torch.full_like(x, M), x)
            logits, _ = model(noisy)
            logp = torch.log_softmax(logits.float(), dim=-1)
            nll = -logp.gather(-1, x.unsqueeze(-1)).squeeze(-1)
            nm = mask.sum(dim=1).clamp(min=1)
            tot += float(((nll * mask).sum(dim=1) / nm).mean())
            n += 1
        out.append((t, tot / n / math.log(2)))
    return out


def sample_naive(model, seq_len, steps, temperature, top_k, generator, prompt_ids=None):
    M = model.cfg.mask_token_id
    dev = next(model.parameters()).device
    B = 1
    x = torch.full((B, seq_len), M, dtype=torch.long, device=dev)
    n_prompt = 0
    if prompt_ids is not None:
        p = torch.as_tensor(prompt_ids, dtype=torch.long, device=dev).unsqueeze(0)
        n_prompt = min(p.size(1), seq_len)
        x[:, :n_prompt] = p[:, :n_prompt]
    for s in range(steps):
        still = (x == M)
        n_left = int(still.sum(dim=1).max())
        if n_left == 0:
            break
        logits, _ = model(x)
        logits = logits.float() / max(temperature, 1e-6)
        if top_k:
            kth = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1).values[..., -1:]
            logits = logits.masked_fill(logits < kth, float("-inf"))
        probs = torch.softmax(logits, dim=-1)
        conf, pred = probs.max(dim=-1)
        conf = conf.masked_fill(~still, float("-inf"))
        k = max(1, math.ceil(n_left / (steps - s)))
        idx = torch.topk(conf, min(k, seq_len), dim=-1).indices
        for b in range(B):
            x[b, idx[b]] = pred[b, idx[b]]
        if n_prompt:
            x[:, :n_prompt] = torch.as_tensor(prompt_ids, dtype=torch.long,
                                              device=dev)[:n_prompt]
    return x


def sample_remask(model, seq_len, steps, temperature, top_k, generator, prompt_ids=None,
                  unmask_frac=0.85):
    """ReMDM/ReMask: keep only the most confident tokens, re-mask the rest."""
    M = model.cfg.mask_token_id
    dev = next(model.parameters()).device
    x = torch.full((1, seq_len), M, dtype=torch.long, device=dev)
    n_prompt = 0
    if prompt_ids is not None:
        p = torch.as_tensor(prompt_ids, dtype=torch.long, device=dev).unsqueeze(0)
        n_prompt = min(p.size(1), seq_len)
        x[:, :n_prompt] = p[:, :n_prompt]
    for s in range(steps):
        logits, _ = model(x)
        logits = logits.float() / max(temperature, 1e-6)
        if top_k:
            kth = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1).values[..., -1:]
            logits = logits.masked_fill(logits < kth, float("-inf"))
        probs = torch.softmax(logits, dim=-1)
        conf, pred = probs.max(dim=-1)
        # score every position: masked -> new prediction prob, decoded -> prob of current tok
        score = torch.where(x == M, conf, probs.gather(-1, x.unsqueeze(-1)).squeeze(-1))
        score[:, :n_prompt] = float("inf")
        n_keep = max(1, round(seq_len * (s + 1) / steps))
        keep = torch.topk(score, min(n_keep, seq_len), dim=-1).indices
        newx = torch.full_like(x, M)
        newx.scatter_(1, keep, torch.where(x.gather(1, keep) == M, pred.gather(1, keep),
                                           x.gather(1, keep)))
        x = newx
        if n_prompt:
            x[:, :n_prompt] = torch.as_tensor(prompt_ids, dtype=torch.long,
                                              device=dev)[:n_prompt]
    return x


def main():
    ck = os.path.join(ROOT, "runs", "diffusion", "model.pt")
    model, meta, ckpt = build_from_ckpt(ck)
    model.eval()
    vocab, eos = meta["vocab"], meta["eos_id"]
    stoi = {c: i for i, c in enumerate(vocab)}
    ds = TokenDataset(os.path.join(ROOT, "data", "val.bin"), 256)

    print("=== loss by noise level (bits/char) ===")
    print("uniform baseline = %.3f" % (math.log(len(vocab)) / math.log(2)))
    for t, b in loss_by_t(model, ds, [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.85, 0.95, 0.99]):
        print(f"  t={t:<5} bpc={b:.3f}")

    prompts = ["从前，有一个小女孩", "一天，小兔子在森林里"]
    print("\n=== naive sampler (steps=64) ===")
    for p in prompts:
        ids = [stoi[c] for c in p if c in stoi]
        g = torch.Generator().manual_seed(3)
        out = sample_naive(model, 256, 64, 0.9, 50, g, ids)
        print(f"[{p}] -> {decode(out[0].tolist(), vocab, eos)[:160]}")

    print("\n=== remask sampler (steps=64) ===")
    for p in prompts:
        ids = [stoi[c] for c in p if c in stoi]
        g = torch.Generator().manual_seed(3)
        out = sample_remask(model, 256, 64, 0.9, 50, g, ids)
        print(f"[{p}] -> {decode(out[0].tolist(), vocab, eos)[:160]}")

    print("\n=== naive, more steps (128) + higher temp ===")
    for p in prompts:
        ids = [stoi[c] for c in p if c in stoi]
        g = torch.Generator().manual_seed(3)
        out = sample_naive(model, 256, 128, 1.0, 0, g, ids)
        print(f"[{p}] -> {decode(out[0].tolist(), vocab, eos)[:160]}")


if __name__ == "__main__":
    main()
