"""Plot the training curves for the AR and diffusion runs side by side."""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

CANDIDATES = [
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\msyhbd.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\simsun.ttc",
    r"C:\Windows\Fonts\Deng.ttf",
]


def pick_font():
    for p in CANDIDATES:
        if os.path.exists(p):
            try:
                font_manager.fontManager.addfont(p)
                name = font_manager.FontProperties(fname=p).get_name()
                return name
            except Exception:
                continue
    return None


def load(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="../runs")
    ap.add_argument("--ar_name", default="ar")
    ap.add_argument("--diff_name", default="diffusion")
    ap.add_argument("--out", default="../out/curves.png")
    args = ap.parse_args()

    font = pick_font()
    if font:
        plt.rcParams["font.sans-serif"] = [font]
    plt.rcParams["axes.unicode_minus"] = False

    ar = load(os.path.join(args.runs, args.ar_name, "history.json"))
    df = load(os.path.join(args.runs, args.diff_name, "history.json"))
    if not ar and not df:
        raise SystemExit("no history yet")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    if ar:
        ax.plot([h["step"] for h in ar], [h["val_bpc"] for h in ar],
                "o-", label="AR 验证 bpc", color="#1f77b4")
    if df:
        ax.plot([h["step"] for h in df], [h["val_ce_bpc"] for h in df],
                "s-", label="Diffusion 去噪 CE (bpc)", color="#d62728")
        ax.plot([h["step"] for h in df], [h["val_elbo_bpc"] for h in df],
                "^--", label="Diffusion ELBO 上界 (bpc)", color="#ff7f0e", alpha=0.8)
    ax.set_xlabel("训练步数")
    ax.set_ylabel("bits per character (越低越好)")
    ax.set_title("验证集 bits-per-character")
    ax.grid(alpha=0.3)
    ax.legend()

    ax = axes[1]
    if ar:
        ax.plot([h["step"] for h in ar], [h["train_loss"] for h in ar],
                "o-", label="AR 训练 loss", color="#1f77b4")
    if df:
        ax.plot([h["step"] for h in df], [h["train_loss"] for h in df],
                "s-", label="Diffusion 训练 loss", color="#d62728")
    ax.set_xlabel("训练步数")
    ax.set_ylabel("cross-entropy (nats/token)")
    ax.set_title("训练损失")
    ax.grid(alpha=0.3)
    ax.legend()

    fig.suptitle("两个 ~1M 参数中文语言模型：自回归 vs 扩散", fontsize=13)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print("wrote", args.out, "font:", font)


if __name__ == "__main__":
    main()
