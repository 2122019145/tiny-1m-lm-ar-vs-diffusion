"""Assert the numpy engine reproduces the torch model's logits.

This is the safety net for the shipped package: if the two implementations ever
drift apart, the packaged app would generate different text than the one we
benchmarked, so the difference is checked numerically here.

    python src/check_numpy.py
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from numpy_engine import NumpyModel                 # noqa: E402
from sample import build_from_ckpt                  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(ROOT, "build", "models")


def compare(name, ckpt, causal, seq_len=24):
    torch.manual_seed(0)
    ids = torch.randint(3, 3490, (1, seq_len), dtype=torch.long)

    tmodel, _meta, _ck = build_from_ckpt(ckpt)
    tmodel.eval()
    with torch.no_grad():
        tlogits, _ = tmodel(ids)
    tlogits = tlogits[0].numpy()

    nmodel = NumpyModel(os.path.join(MODELS, f"{name}.npz"),
                        os.path.join(MODELS, f"{name}_info.json"))
    nlogits = nmodel.forward(ids.numpy(), causal=causal)[0]

    assert tlogits.shape == nlogits.shape, f"{name}: {tlogits.shape} vs {nlogits.shape}"
    diff = np.abs(tlogits - nlogits)
    rel = diff.max() / (np.abs(tlogits).max() + 1e-12)
    print(f"[{name:9s}] max|torch-numpy| = {diff.max():.3e}   rel = {rel:.3e}")
    assert rel < 2e-4, f"{name}: numpy/torch mismatch, rel={rel:.2e}"


def main():
    torch.set_num_threads(8)
    compare("ar", os.path.join(ROOT, "runs", "ar_long", "model.pt"), causal=True)
    compare("diffusion", os.path.join(ROOT, "runs", "diffusion_long", "model.pt"),
            causal=False)
    print("[ok] numpy engine matches torch")


if __name__ == "__main__":
    main()
