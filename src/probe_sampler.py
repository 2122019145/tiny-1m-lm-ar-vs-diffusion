"""Probe why diffusion samples repeat: does it predict the mask token? Is the
error accumulation from early unmasking, or is the model simply weak?"""
import math
import os

import numpy as np
import torch

from model import DiffusionLM
from sample import build_from_ckpt, decode

torch.set_num_threads(14)
ROOT = r"D:\dsh1\modeltest"


class DS:
    def __init__(self, path, seq_len):
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        self.seq_len = seq_len

    def sample(self, b, g=None):
        hi = len(self.data) - self.seq_len - 1
        st = torch.randint(0, hi, (b,), generator=g).tolist()
        return torch.stack([torch.from_numpy(np.array(self.data[s:s + self.seq_len],
                                                    dtype=np.int64)) for s in st])


def main():
    model, meta, _ = build_from_ckpt(os.path.join(ROOT, "runs", "diffusion", "model.pt"))
    model.eval()
    vocab, eos = meta["vocab"], meta["eos_id"]
    M = meta["mask_id"]
    stoi = {c: i for i, c in enumerate(vocab)}
    ds = DS(os.path.join(ROOT, "data", "val.bin"), 256)
    g = torch.Generator().manual_seed(0)

    print("=== does it ever predict the mask token? ===")
    for t in (0.1, 0.5, 0.9):
        x = ds.sample(8, g)
        mask = torch.rand(8, 256, generator=g) < t
        mask[:, 0] |= ~mask.any(dim=1)
        noisy = torch.where(mask, torch.full_like(x, M), x)
        with torch.no_grad():
            logits, _ = model(noisy)
        pred = logits.argmax(-1)
        pm = (pred[mask] == M).float().mean().item()
        acc = (pred[mask] == x[mask]).float().mean().item()
        print(f"  t={t}: pred==MASK {pm:.4f}   masked-token top1 acc {acc:.4f}")

    print("\n=== reconstruction quality by number of oracle-visible tokens ===")
    x = ds.sample(4, g)
    for keep_frac in (0.1, 0.3, 0.5, 0.7, 0.9):
        mask = torch.rand(4, 256, generator=g) > keep_frac
        mask[:, 0] |= ~mask.any(dim=1)
        noisy = torch.where(mask, torch.full_like(x, M), x)
        with torch.no_grad():
            logits, _ = model(noisy)
        pred = logits.argmax(-1)
        acc = (pred[mask] == x[mask]).float().mean().item()
        print(f"  keep {keep_frac:.0%} visible -> masked top1 acc {acc:.4f}")

    print("\n=== greedy full reconstruction of a REAL val sequence (oracle) ===")
    x = ds.sample(1, g)
    print("  TRUE :", decode(x[0].tolist(), vocab, eos)[:200])

    # mask everything, then unmask in 64 steps excluding the mask token from preds
    def sample(exclude_mask=True, steps=64, temp=1.0, top_k=50, seed=0):
        gg = torch.Generator().manual_seed(seed)
        xx = torch.full((1, 256), M, dtype=torch.long)
        for s in range(steps):
            still = xx == M
            n_left = int(still.sum().item())
            if n_left == 0:
                break
            with torch.no_grad():
                lg, _ = model(xx)
            lg = lg.float() / temp
            if exclude_mask:
                lg[..., M] = -float("inf")
            if top_k:
                v = torch.topk(lg, top_k, dim=-1).values[..., -1:]
                lg = lg.masked_fill(lg < v, float("-inf"))
            probs = torch.softmax(lg, dim=-1)
            conf, pred = probs.max(dim=-1)
            conf = conf.masked_fill(~still, -1e9)
            k = max(1, math.ceil(n_left / (steps - s)))
            idx = torch.topk(conf, min(k, 256), dim=-1).indices
            xx[0, idx[0]] = pred[0, idx[0]]
        return xx

    print("\n  exclude MASK from preds:")
    for sd in (0, 1):
        print("   ->", decode(sample(True, 64, 1.0, 50, sd)[0].tolist(), vocab, eos)[:200])
    print("\n  allow MASK in preds (old behaviour):")
    for sd in (0, 1):
        print("   ->", decode(sample(False, 64, 1.0, 50, sd)[0].tolist(), vocab, eos)[:200])


if __name__ == "__main__":
    main()
