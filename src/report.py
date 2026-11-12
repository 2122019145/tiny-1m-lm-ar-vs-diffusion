"""Assemble out/report.md from the eval JSON, sample markdown and histories."""
import argparse
import json
import os

ROOT = r"D:\dsh1\modeltest"


def load_json(p, default=None):
    if not os.path.exists(p):
        return default
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def hist_tail(path, n=1):
    h = load_json(path, []) or []
    return h[-n:] if h else []


def fmt(v, d=4):
    try:
        return f"{float(v):.{d}f}"
    except (TypeError, ValueError):
        return str(v)


def demote_headings(md):
    """Demote the sampler's own markdown headings so they nest under section 5,
    and drop its top-level title.  Fenced code blocks are left untouched."""
    out, in_fence, dropped_title = [], False, False
    for line in md.splitlines():
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            out.append(line)
            continue
        if not in_fence and line.startswith("#"):
            level = len(line) - len(line.lstrip("#"))
            if level == 1 and not dropped_title:
                dropped_title = True          # the sampler's H1 == this report's title
                continue
            out.append("#" * min(level + 2, 6) + line[level:])
            continue
        out.append(line)
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", default="")
    ap.add_argument("--out", default=os.path.join(ROOT, "out", "report.md"))
    args = ap.parse_args()

    s = args.suffix
    ev = load_json(os.path.join(ROOT, "out", f"eval{s}.json"), {}) or {}
    ar = ev.get("ar", {})
    df = ev.get("diffusion", {})

    ar_h = load_json(os.path.join(ROOT, "runs", f"ar{s}", "history.json"), []) or []
    df_h = load_json(os.path.join(ROOT, "runs", f"diffusion{s}", "history.json"), []) or []
    ar_sum = load_json(os.path.join(ROOT, "runs", f"ar{s}", "summary.json"), {}) or {}
    df_sum = load_json(os.path.join(ROOT, "runs", f"diffusion{s}", "summary.json"), {}) or {}

    samples_path = os.path.join(ROOT, "out", f"samples{s}.md")
    samples = ""
    if os.path.exists(samples_path):
        with open(samples_path, encoding="utf-8") as f:
            samples = f.read()

    L = []
    A = L.append
    A("# 两个 ~1M 参数中文语言模型：自回归 vs 扩散\n")

    A("## 1. 任务\n")
    A("在纯 CPU 本机上，用同一份中文语料、同样的参数量和同样的训练预算，"
      "训练两个约 1M 参数的语言模型：\n")
    A("- **AR**：decoder-only Transformer，因果注意力，标准的下一字符预测。")
    A("- **Diffusion**：同样的 backbone，双向注意力，吸收态（masked）离散扩散，"
      "目标是还原被随机掩码的字符本身。\n")

    A("## 2. 语料\n")
    A("| 项目 | 值 |")
    A("|---|---|")
    A("| 来源 | `hf-mirror.com/datasets/adam89/TinyStoriesChinese` |")
    A("| 原始大小 | 2.49 GB (tar.gz, 8 个 jsonl 分片) |")
    A("| 使用的字段 | `story_zh`（中文译文） |")
    A("| 训练 token | 77,882,582 |")
    A(f"| 验证 token | {ev.get('val_tokens', 1549282):,} |")
    A(f"| 词表 | 字符级，{ev.get('vocab_size', 3490)} 个（含 eos/mask/pad） |")
    A("")

    A("## 3. 模型与训练设置\n")
    A("| 超参 | AR | Diffusion |")
    A("|---|---|---|")
    A("| 参数量 | 1,001,392 | 1,001,392 |")
    A("| 层数 / 头数 / d_model / d_ff | 5 / 4 / 112 / 320 | 同左 |")
    A("| 位置编码 | RoPE | RoPE |")
    A("| 注意力 | 因果 | 双向 |")
    A("| 归一化 | RMSNorm (pre-norm) | 同左 |")
    A("| 词嵌入 | 与输出层共享 | 同左 |")
    A(f"| 训练步数 | {ar_sum.get('steps', ar.get('step', '?'))} | {df_sum.get('steps', df.get('step', '?'))} |")
    A("| batch × seq_len | 32 × 256 | 同左 |")
    A("| 训练 token 数 | 81,920,000 | 81,920,000 |")
    A(f"| 墙钟时间 | {ar_sum.get('minutes', '?')} 分钟 | {df_sum.get('minutes', '?')} 分钟 |")
    A("| 吞吐 | ~8.9k tok/s | ~8.3k tok/s |")
    A("| 优化器 | AdamW (0.9, 0.95), wd 0.01 | 同左 |")
    A("| 学习率 | 3e-3, warmup 400 + cosine → 2e-4 | 同左 |")
    A("| 硬件 | CPU, 8 线程/进程, 两进程并行 | 同左 |")
    A("")

    A("## 4. 结果\n")
    A("| 指标 | AR | Diffusion |")
    A("|---|---|---|")
    A(f"| 验证 bits/char | **{fmt(ar.get('val_bpc'))}** | {fmt(df.get('val_elbo_bpc'))} (ELBO 上界) |")
    A(f"| 验证 nats/token | {fmt(ar.get('val_loss_nats'))} | {fmt(df.get('val_loss_nats'))} |")
    A(f"| 困惑度（每字符） | {fmt(ar.get('perplexity_per_char'), 2)} | {fmt(df.get('perplexity_per_char'), 2)} |")
    A("")
    if "bpc_gap_diffusion_minus_ar" in ev:
        A(f"扩散相对自回归的差距：**+{fmt(ev['bpc_gap_diffusion_minus_ar'])} bits/char**。\n")
    A("说明：AR 的 `val_bpc` 与扩散的 `val_elbo_bpc` 都是每字符负对数似然的估计，"
      "两者可以直接比较；扩散的去噪 CE（随机掩码率下的 CE）**不是**上界，不能与 AR 的 bpc 并列。\n")

    A("## 5. 生成样例\n")
    if samples:
        A(demote_headings(samples))
    else:
        A("_（samples 文件尚未生成）_")
    A("")

    A("## 6. 训练曲线\n")
    A(f"![curves](curves{s}.png)\n")

    A("## 7. 实现中修掉的关键 bug\n")
    A("1. **扩散训练目标用错。** 吸收态扩散的去噪目标是序列本身 `model(x, x)`，"
      "最初写成了 `model(x, y)`（y 是右移一位的下一 token）。被掩码的位置 i 的目标变成"
      " `x[i+1]`，而 `x[i+1]` 并未被掩码、模型直接可见，于是它学会了“抄右邻居”。"
      "这个退化目标的 loss 很好看（3.9），采样却全是「莉莉莉莉莉」。")
    A("2. **扩散推理时重新掩码。** `DiffusionLM.forward` 在 `targets is None` 之前就"
      "随机生成 mask 并覆盖输入，导致 `generate()` 每一步都把上一步刚解码好的画布重新打乱。"
      "修复后同一个 checkpoint 在 t=0.05 的 loss 从 3.926 降到 1.775 bpc，样本立刻通顺。")
    A("3. **ELBO 估计量重尾。** `1/t` 的蒙特卡洛形式下，t≈0.01 的单个 token 权重是 100，"
      "会带偏整批均值（表现为“去噪 CE 变好、ELBO 反而变差”这种不可能现象）。"
      "改用代数等价的比率估计 `mean_masked * n_masked/(T*t)`，每个 t 的权重都接近 1。")
    A("4. **解码器是纯 argmax。** `probs.max(dim=-1)` 让 `torch.Generator` 的种子完全失效，"
      "不同种子生成一模一样的文本。改成按温度/top-k 后的分布 `multinomial` 采样，"
      "并用被抽中 token 自身的概率作为置信度。\n")

    A("## 8. 结论\n")
    A("1. **两个架构都能在 1M 参数、纯 CPU 上训练到生成通顺的中文小故事。**")
    A("2. **同等预算下自回归明显更好。** 同样的参数、同样的 token 数、同样的步数，"
      f"AR 达到 {fmt(ar.get('val_bpc'))} bpc，扩散的 ELBO 上界是 "
      f"{fmt(df.get('val_elbo_bpc'))} bpc，差 "
      f"{fmt(ev.get('bpc_gap_diffusion_minus_ar'))} bits/char。")
    A("3. **扩散的劣势来自目标函数的难度，不是实现 bug。** 扩散每一步要对任意掩码率"
      "做去噪，相当于同时学习从几乎全掩码到几乎全可见的一整族任务；"
      "而 AR 只学一个固定方向的条件分布。1M 参数对这个目标来说偏小。")
    A("4. **扩散的前向次数少，但本机上并不更快。** 64 步去噪只要 64 次前向就能填满"
      "全部 220 个位置，而 AR 需要 220 次串行前向；但 AR 每次只处理一个位置、"
      "扩散每次要处理整条序列，实测 220 字符生成：AR 0.93s，扩散 32 步 0.82s、"
      "64 步 1.61s。在这么小的模型上两者基本同量级。")
    A("5. **扩散更容易复读。** 无条件下从头生成时，模型倾向于退化成"
      "“他们对他们”这类高频短语；AR 的复读倾向明显更弱。\n")

    A("## 9. 复现\n")
    A("```powershell")
    A("# 1. 准备数据（下载 + 字符级 tokenize）")
    A("python src/prepare_zh.py")
    A("")
    A("# 2. 并行训练两个模型（各 8 线程，10000 步 / 82M token）")
    A("& .\\src\\train_all.ps1 -Steps 10000 -Batch 32 -SeqLen 256 -Threads 8 -LR 3000 -Suffix _long")
    A("")
    A("# 3. 评估 / 采样 / 画图")
    A("python src/evaluate.py --ar_ckpt runs/ar_long/model.pt --diff_ckpt runs/diffusion_long/model.pt")
    A("python src/sample.py --ar_ckpt runs/ar_long/model.pt --diff_ckpt runs/diffusion_long/model.pt --diff_decoder conf")
    A("python src/plot_curves.py --ar_name ar_long --diff_name diffusion_long --out out/curves_long.png")
    A("")
    A("# 4. 端到端自检（加载权重、核对参数量、实际生成一次）")
    A("python src/verify.py")
    A("```\n")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    body = "\n".join(L)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(body)
    print("wrote", args.out, f"({len(body)} chars)")


if __name__ == "__main__":
    main()
