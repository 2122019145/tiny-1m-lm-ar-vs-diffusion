"""Independent end-to-end verification of the two trained checkpoints.

Checks, for each run:
  * the checkpoint loads and rebuilds the right architecture,
  * the parameter count is the intended ~1M,
  * a short generation actually runs and produces non-degenerate text.

Run:  python src/verify.py
"""
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evaluate import build_from_ckpt, ckpt_step          # noqa: E402
from decoders import sample_conf                          # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
RUNS = os.path.join(ROOT, "runs")

PROMPTS = ["从前，有一个小女孩", "一天，小兔子在森林里", "妈妈对小熊说："]


def decode(ids, vocab, eos_id):
    return "".join(vocab[i] for i in ids if i >= 3 and i != eos_id)


def repetition_score(text, n=4):
    """Fraction of duplicated n-char shingles; 0 = diverse, 1 = degenerate."""
    if len(text) < n:
        return 0.0
    sh = [text[i:i + n] for i in range(len(text) - n + 1)]
    return 1.0 - len(set(sh)) / len(sh)


def main():
    torch.set_num_threads(16)
    meta = json.load(open(os.path.join(DATA, "meta.json"), encoding="utf-8"))
    vocab, eos_id = meta["vocab"], meta["eos_id"]

    rows = []
    for name, kind in [("ar_long", "ar"), ("diffusion_long", "diffusion")]:
        path = os.path.join(RUNS, name, "model.pt")
        assert os.path.exists(path), f"missing checkpoint {path}"
        model, _mt, ck = build_from_ckpt(path)
        n = sum(p.numel() for p in model.parameters())
        step = ckpt_step(ck)
        assert model.name == kind, f"{name}: arch mismatch {model.name} != {kind}"
        print(f"[{kind:9s}] params={n:,} step={step} arch={ck['args']['arch']} "
              f"layers={ck['args']['n_layer']} d_model={ck['args']['d_model']}")
        assert 900_000 < n < 1_100_000, f"{name}: param count {n} not ~1M"

        # actually generate, so a broken checkpoint cannot pass on metadata alone
        torch.manual_seed(11)
        if kind == "ar":
            ids = [vocab.index(c) for c in PROMPTS[0]]
            out = model.generate(torch.tensor([ids]), max_new_tokens=80,
                                 temperature=0.85, top_k=40, eos_id=eos_id)
            text = decode(out[0].tolist(), vocab, eos_id)
        else:
            p = torch.tensor([[vocab.index(c) for c in PROMPTS[0]]])
            out = sample_conf(model, seq_len=96, steps=32, temperature=1.0,
                              top_k=50, generator=torch.Generator().manual_seed(11),
                              prompt_ids=p)
            text = decode(out[0].tolist(), vocab, eos_id)

        rep = repetition_score(text)
        print(f"           rep={rep:.3f}  sample={text[:60]!r}")
        assert len(text) > 40, f"{name}: generation too short ({len(text)} chars)"
        assert rep < 0.6, f"{name}: degenerate repetition {rep:.2f}"
        rows.append((kind, n, step, rep, text))

    print("\n[ok] both checkpoints verified end to end")


if __name__ == "__main__":
    main()
