"""Export the torch checkpoints into numpy .npz + model_info.json.

Used once at packaging time; the shipped app needs no torch.

    python src/export_npz.py --out build\models
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sample import build_from_ckpt, ckpt_step            # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def export(ckpt_path, out_dir, name):
    model, meta, ck = build_from_ckpt(ckpt_path)
    sd = {k: v.detach().cpu().numpy().astype(np.float32)
          for k, v in model.state_dict().items()}
    # lm_head is tied to tok_emb -> drop the duplicate
    sd.pop("lm_head.weight", None)

    os.makedirs(out_dir, exist_ok=True)
    npz = os.path.join(out_dir, f"{name}.npz")
    np.savez(npz, **sd)

    info = {
        "arch": ck["args"]["arch"],
        "step": ckpt_step(ck),
        "params": int(sum(p.numel() for p in model.parameters())),
        "vocab": meta["vocab"],
        "eos_id": meta["eos_id"],
        "mask_id": meta["mask_id"],
        "pad_id": meta["pad_id"],
        "cfg": {"n_layer": ck["args"]["n_layer"], "n_head": ck["args"]["n_head"],
                "d_model": ck["args"]["d_model"], "d_ff": ck["args"]["d_ff"],
                "max_seq_len": ck["args"]["seq_len"],
                "vocab_size": meta["vocab_size"]},
    }
    with open(os.path.join(out_dir, f"{name}_info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False)
    mb = os.path.getsize(npz) / 1e6
    print(f"[export] {name}: {len(sd)} tensors, {mb:.1f} MB, "
          f"{info['params']:,} params, step {info['step']}")
    return npz


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ar_ckpt", default=os.path.join(ROOT, "runs", "ar_long", "model.pt"))
    ap.add_argument("--diff_ckpt", default=os.path.join(ROOT, "runs", "diffusion_long", "model.pt"))
    ap.add_argument("--out", default=os.path.join(ROOT, "build", "models"))
    args = ap.parse_args()
    export(args.ar_ckpt, args.out, "ar")
    export(args.diff_ckpt, args.out, "diffusion")


if __name__ == "__main__":
    main()
