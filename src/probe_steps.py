"""Does step count fix the repetition? Test 256-step decoding and a low-temp
variant, and check whether repetition comes from the model or the schedule."""
import math
import os
import re

import torch

from sample import build_from_ckpt, decode
from tune_sampler import repetition_score, sample_conf, sample_remask

ROOT = r"D:\dsh1\modeltest"
torch.set_num_threads(14)


def main():
    model, meta, _ = build_from_ckpt(os.path.join(ROOT, "runs", "diffusion", "model.pt"))
    model.eval()
    vocab, eos = meta["vocab"], meta["eos_id"]
    stoi = {c: i for i, c in enumerate(vocab)}
    prompts = ["从前，有一个小女孩", "一天，小兔子在森林里", "妈妈对小熊说："]
    pids = [[stoi[c] for c in p if c in stoi] for p in prompts]

    print("=== step count sweep (remask, temp 1.0, top_k 50) ===")
    for steps in (128, 192, 256):
        reps, txts = [], []
        for i, pid in enumerate(pids):
            g = torch.Generator().manual_seed(5 + i)
            out = sample_remask(model, 256, steps, 1.0, 50, g, pid)
            t = decode(out[0].tolist(), vocab, eos)
            reps.append(repetition_score(t))
            txts.append(t)
        print(f"  steps={steps}: mean rep={sum(reps)/3:.3f}")
        print(f"    {txts[0][:180].replace(chr(10),' ')}")

    print("\n=== low temperature (remask, 128 steps, top_k 20) ===")
    for temp in (0.6, 0.75, 0.9):
        reps, txts = [], []
        for i, pid in enumerate(pids):
            g = torch.Generator().manual_seed(5 + i)
            out = sample_remask(model, 256, 128, temp, 20, g, pid)
            t = decode(out[0].tolist(), vocab, eos)
            reps.append(repetition_score(t))
            txts.append(t)
        print(f"  temp={temp}: mean rep={sum(reps)/3:.3f}")
        print(f"    {txts[0][:180].replace(chr(10),' ')}")


if __name__ == "__main__":
    main()
