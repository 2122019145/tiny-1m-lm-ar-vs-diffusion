"""Interactive playground for the two trained ~1M-param Chinese models.

Run it, type a Chinese story opening, press Enter, and both models continue it.
Type /help inside the prompt for the tunable knobs, /quit to exit.

    .\\.venv\\Scripts\\python.exe src\\chat.py

Everything runs on CPU; both checkpoints are ~4 MB and load in well under a second.
"""
import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from decoders import repetition_score, sample_conf, sample_remask   # noqa: E402
from sample import build_from_ckpt, ckpt_step, decode              # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

HELP = """
命令（在提示符下输入）：
  <任意中文>        两个模型各续写一段，并排显示
  /ar <中文>        只用自回归模型
  /diff <中文>      只用扩散模型
  /model ar|diff|both   切换默认用哪个模型（当前: {model}）
  /len <N>          续写长度，字符数（当前: {length}）
  /temp <X>         AR 温度，越小越保守（当前: {ar_temp}）
  /dtemp <X>        扩散温度（当前: {diff_temp}）
  /topk <N>         top-k 采样（当前: {top_k}）
  /steps <N>        扩散去噪步数，越多越慢越准（当前: {steps}）
  /decoder conf|remask  扩散解码器（当前: {decoder}）
  /seed <N>         随机种子；同一提示词+同一种子可复现
  /help             显示本帮助
  /quit 或 Ctrl-C   退出
""".strip()


def encode(prompt, stoi):
    """Char-level encode; returns (ids, dropped_chars)."""
    ids, dropped = [], []
    for c in prompt:
        if c in stoi:
            ids.append(stoi[c])
        else:
            dropped.append(c)
    return ids, dropped


def show(tag, text, secs, rep=None):
    extra = f" · 复读 {rep:.2f}" if rep is not None else ""
    print(f"\n\033[1m[{tag}]\033[0m {secs:.2f}s{extra}")
    print("─" * 68)
    for line in text.strip().splitlines():
        print("  " + line if line.strip() else "")
    print("─" * 68)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ar_ckpt", default=os.path.join(ROOT, "runs", "ar_long", "model.pt"))
    ap.add_argument("--diff_ckpt", default=os.path.join(ROOT, "runs", "diffusion_long", "model.pt"))
    ap.add_argument("--length", type=int, default=160, help="续写字符数")
    ap.add_argument("--ar_temp", type=float, default=0.85)
    ap.add_argument("--diff_temp", type=float, default=1.0)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--steps", type=int, default=64, help="扩散去噪步数")
    ap.add_argument("--decoder", default="conf", choices=["conf", "remask"])
    ap.add_argument("--model", default="both", choices=["ar", "diff", "both"])
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    torch.set_num_threads(args.threads or min(16, os.cpu_count() or 8))

    for path in (args.ar_ckpt, args.diff_ckpt):
        if not os.path.exists(path):
            sys.exit(f"找不到权重文件: {path}")

    print("正在加载模型 ...")
    ar, meta, ar_ck = build_from_ckpt(args.ar_ckpt)
    dm, _, dm_ck = build_from_ckpt(args.diff_ckpt)
    vocab, eos_id = meta["vocab"], meta["eos_id"]
    stoi = {c: i for i, c in enumerate(vocab)}
    n_ar = sum(p.numel() for p in ar.parameters())
    n_dm = sum(p.numel() for p in dm.parameters())
    print(f"  自回归  {n_ar:,} 参数 · {ckpt_step(ar_ck)} 步  ({os.path.basename(args.ar_ckpt)})")
    print(f"  扩散    {n_dm:,} 参数 · {ckpt_step(dm_ck)} 步  ({os.path.basename(args.diff_ckpt)})")
    print(f"  词表 {len(vocab)} · 字符级 · 设备 CPU")
    print("\n输入中文开头试试，例如：从前，有一个小女孩")
    print("输入 /help 看命令，/quit 退出。")

    run_seed = args.seed
    while True:
        try:
            raw = input("\n提示> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见。")
            return
        if not raw:
            continue

        low = raw.lower()
        if low in ("/quit", "/exit", "/q"):
            print("再见。")
            return
        if low in ("/help", "/h", "/?"):
            print(HELP.format(model=args.model, length=args.length, ar_temp=args.ar_temp,
                              diff_temp=args.diff_temp, top_k=args.top_k,
                              steps=args.steps, decoder=args.decoder))
            continue

        # ---- parameter commands -------------------------------------------------
        if low.startswith("/"):
            cmd, _, rest = raw.partition(" ")
            rest = rest.strip()
            cmd = cmd.lower()

            # single-model shortcuts that also carry a prompt: /ar <text>, /diff <text>
            if cmd in ("/ar", "/diff") and rest:
                only = "ar" if cmd == "/ar" else "diff"
                saved, args.model = args.model, only
                try:
                    _generate(rest, args, ar, dm, vocab, stoi, eos_id, run_seed)
                finally:
                    args.model = saved
                continue

            try:
                if cmd == "/model" and rest in ("ar", "diff", "both"):
                    args.model = rest
                elif cmd == "/len":
                    args.length = max(8, int(rest))
                elif cmd == "/temp":
                    args.ar_temp = float(rest)
                elif cmd == "/dtemp":
                    args.diff_temp = float(rest)
                elif cmd == "/topk":
                    args.top_k = max(1, int(rest))
                elif cmd == "/steps":
                    args.steps = max(1, int(rest))
                elif cmd == "/decoder" and rest in ("conf", "remask"):
                    args.decoder = rest
                elif cmd == "/seed":
                    run_seed = int(rest)
                else:
                    print("未知命令，输入 /help 查看。")
                    continue
                print(f"已设置 {cmd} = {rest}")
            except ValueError:
                print("参数格式不对，例如 /len 160")
            continue

        # ---- generation ---------------------------------------------------------
        _generate(raw, args, ar, dm, vocab, stoi, eos_id, run_seed)


def _generate(raw, args, ar, dm, vocab, stoi, eos_id, run_seed):
    """Encode the prompt and run whichever model(s) args.model selects."""
    ids, dropped = encode(raw, stoi)
    if not ids:
        print("提示里没有任何词表内的字符，换一句试试。")
        return
    if dropped:
        print(f"  (提示中有 {len(dropped)} 个字符不在词表内，已跳过: {''.join(dropped[:8])})")

    if args.model in ("ar", "both"):
        if run_seed is not None:
            torch.manual_seed(run_seed)
        t0 = time.time()
        out = ar.generate(torch.tensor([ids], dtype=torch.long), args.length,
                          temperature=args.ar_temp, top_k=args.top_k, eos_id=eos_id)
        text = decode(out[0].tolist(), vocab, eos_id)
        show("自回归", text, time.time() - t0, repetition_score(text))

    if args.model in ("diff", "both"):
        gen = torch.Generator().manual_seed(run_seed if run_seed is not None else 0)
        dec = sample_remask if args.decoder == "remask" else sample_conf
        seq_len = min(ar.cfg.max_seq_len, len(ids) + args.length)
        t0 = time.time()
        out = dec(dm, seq_len, args.steps, args.diff_temp, args.top_k, gen,
                  prompt_ids=ids)
        text = decode(out[0].tolist(), vocab, eos_id)
        show("扩散", text, time.time() - t0, repetition_score(text))


if __name__ == "__main__":
    main()
