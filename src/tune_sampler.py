"""Compare decoders on the final diffusion checkpoint and pick the best one.

The training objective is fine (NELBO 3.91 bpc); degenerate-looking samples are
usually a decoding artifact.  We compare the standard confidence-based unmasker
against a ReMDM/ReMask variant that can revise early guesses, over a small grid
of step counts and temperatures, and score each with a repetition metric.
"""
import argparse
import json
import os

import torch

from decoders import repetition_score, sample_conf, sample_remask
from sample import build_from_ckpt, decode

ROOT = r"D:\dsh1\modeltest"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "runs", "diffusion", "model.pt"))
    ap.add_argument("--seq_len", type=int, default=256)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--threads", type=int, default=14)
    ap.add_argument("--out", default=os.path.join(ROOT, "out", "sampler_grid.json"))
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    model, meta, _ = build_from_ckpt(args.ckpt)
    model.eval()
    vocab, eos = meta["vocab"], meta["eos_id"]
    stoi = {c: i for i, c in enumerate(vocab)}
    prompts = ["从前，有一个小女孩", "一天，小兔子在森林里", "妈妈对小熊说："]
    pids = [[stoi[c] for c in p if c in stoi] for p in prompts]

    rows = []
    grid = []
    for steps in (32, 64, 128):
        for temp in (0.8, 1.0):
            grid.append(("conf", steps, temp, 50))
            grid.append(("remask", steps, temp, 50))
    grid.append(("remask", 64, 1.0, 0))
    grid.append(("remask", 128, 1.0, 0))

    for kind, steps, temp, top_k in grid:
        scores, sample_txt = [], []
        for i, pid in enumerate(pids):
            g = torch.Generator().manual_seed(args.seed + i)
            fn = sample_conf if kind == "conf" else sample_remask
            out = fn(model, args.seq_len, steps, temp, top_k, g, pid)
            txt = decode(out[0].tolist(), vocab, eos)
            scores.append(repetition_score(txt))
            if i == 0:
                sample_txt.append(txt[:110].replace("\n", " "))
        rep = sum(scores) / len(scores)
        rows.append({"decoder": kind, "steps": steps, "temp": temp, "top_k": top_k,
                     "repetition": round(rep, 4), "example": sample_txt[0]})
        print(f"{kind:7s} steps={steps:<4d} temp={temp:<4} top_k={top_k:<4d} "
              f"rep={rep:.3f}  {sample_txt[0][:70]}")

    rows.sort(key=lambda r: r["repetition"])
    print("\n=== best by repetition ===")
    for r in rows[:3]:
        print(r["decoder"], r["steps"], r["temp"], r["top_k"], r["repetition"])

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
