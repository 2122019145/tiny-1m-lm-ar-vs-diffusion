"""Sample from the trained checkpoints and build a side-by-side report."""
import argparse
import json
import os
import time

import torch

from decoders import sample_conf, sample_remask
from model import ARModel, Config, DiffusionLM

# A few Chinese story openings (they are all in-distribution for TinyStories-zh).
PROMPTS = [
    "从前，有一个小女孩",
    "一天，小兔子在森林里",
    "妈妈对小熊说：",
]


def build_from_ckpt(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    a = ck["args"]
    meta = ck["meta"]
    cfg = Config(vocab_size=meta["vocab_size"], n_layer=a["n_layer"], n_head=a["n_head"],
                 d_model=a["d_model"], d_ff=a["d_ff"], max_seq_len=a["seq_len"],
                 dropout=a.get("dropout", 0.0))
    cfg.mask_token_id = meta["mask_id"]
    cfg.eos_id = meta["eos_id"]
    model = ARModel(cfg) if a["arch"] == "ar" else DiffusionLM(cfg)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, meta, ck


def decode(ids, vocab, eos_id):
    out = []
    for i in ids:
        i = int(i)
        if i == eos_id:
            break
        if i < 3:            # specials: eos / mask / pad
            continue
        out.append(vocab[i])
    return "".join(out)


def ckpt_step(ck):
    """model.pt carries no explicit step, so fall back to the eval history."""
    if "step" in ck:
        return ck["step"]
    h = ck.get("history") or []
    return h[-1]["step"] if h else "?"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ar_ckpt", default="../runs/ar/model.pt")
    ap.add_argument("--diff_ckpt", default="../runs/diffusion/model.pt")
    ap.add_argument("--out", default="../out/samples.md")
    ap.add_argument("--n_samples", type=int, default=3)
    ap.add_argument("--ar_len", type=int, default=200)
    ap.add_argument("--diff_len", type=int, default=200)
    ap.add_argument("--diff_steps", type=int, default=64)
    ap.add_argument("--diff_decoder", default="conf", choices=["conf", "remask"],
                    help="conf = plain mask-predict, remask = ReMDM-style revocable decoding")
    ap.add_argument("--temperature", type=float, default=0.9,
                    help="sampling temperature for BOTH models")
    ap.add_argument("--ar_temperature", type=float, default=None,
                    help="override --temperature for the AR model")
    ap.add_argument("--diff_temperature", type=float, default=None,
                    help="override --temperature for the diffusion model")
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    ar_temp = args.temperature if args.ar_temperature is None else args.ar_temperature
    df_temp = args.temperature if args.diff_temperature is None else args.diff_temperature
    torch.manual_seed(args.seed)
    lines = ["# 中文小故事生成：自回归 vs 扩散\n"]

    if os.path.exists(args.ar_ckpt):
        ar, ar_meta, ck = build_from_ckpt(args.ar_ckpt)
        vocab, stoi = ar_meta["vocab"], {c: i for i, c in enumerate(ar_meta["vocab"])}
        lines.append(f"\n## 自回归模型 (AR)\n")
        lines.append(f"参数量 {sum(p.numel() for p in ar.parameters()):,} · "
                     f"训练 {ckpt_step(ck)} 步 · temp {ar_temp} · top_k {args.top_k}\n")
        t0 = time.time()
        for pi, prompt in enumerate(PROMPTS[:args.n_samples]):
            pids = [stoi[c] for c in prompt if c in stoi]
            idx = torch.tensor([pids], dtype=torch.long)
            out = ar.generate(idx, args.ar_len, temperature=ar_temp,
                              top_k=args.top_k, eos_id=ar.cfg.eos_id)
            txt = decode(out[0].tolist(), vocab, ar.cfg.eos_id)
            lines.append(f"**{pi+1}. 提示：`{prompt}`**\n\n```\n{txt.strip()}\n```\n")
        lines.append(f"_AR 生成耗时 {time.time()-t0:.1f}s_\n")

    if os.path.exists(args.diff_ckpt):
        dm, dm_meta, ck = build_from_ckpt(args.diff_ckpt)
        vocab = dm_meta["vocab"]
        stoi = {c: i for i, c in enumerate(vocab)}
        lines.append(f"\n## 扩散模型 (masked diffusion)\n")
        lines.append(f"参数量 {sum(p.numel() for p in dm.parameters()):,} · "
                     f"训练 {ckpt_step(ck)} 步 · {args.diff_steps} 步去噪 "
                     f"({args.diff_decoder}) · "
                     f"temp {df_temp} · top_k {args.top_k}\n")
        t0 = time.time()
        dec = sample_remask if args.diff_decoder == "remask" else sample_conf
        for pi, prompt in enumerate(PROMPTS[:args.n_samples]):
            pids = [stoi[c] for c in prompt if c in stoi]
            g = torch.Generator().manual_seed(args.seed + pi)
            out = dec(dm, args.diff_len, args.diff_steps, df_temp,
                      args.top_k, g, pids)
            txt = decode(out[0].tolist(), vocab, dm.cfg.eos_id)
            lines.append(f"**{pi+1}. 提示：`{prompt}`**\n\n```\n{txt.strip()}\n```\n")
        lines.append(f"_扩散生成耗时 {time.time()-t0:.1f}s_\n")

        # unconditional samples too, to show the pure generative behaviour
        lines.append("\n### 无条件生成（全掩码起步）\n")
        for s in range(2):
            g = torch.Generator().manual_seed(args.seed + 100 + s)
            out = dec(dm, args.diff_len, args.diff_steps, df_temp,
                      args.top_k, g)
            txt = decode(out[0].tolist(), vocab, dm.cfg.eos_id)
            lines.append(f"**Sample {s+1}**\n\n```\n{txt.strip()}\n```\n")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"wrote {args.out}")
    print("\n".join(lines)[:4000])


if __name__ == "__main__":
    main()

