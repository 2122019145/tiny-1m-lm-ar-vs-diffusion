# 两个 ~1M 参数中文语言模型的对照实验：自回归 vs 离散扩散

在同一份语料、同等参数量、同等训练预算下，训练并对比两种目标函数：
**因果自回归**（next-token）与 **双向吸收态离散扩散**（masked diffusion）。
全程在**纯 CPU** 笔记本上完成，单个模型约 2.5 小时。

> 目的不是把模型做大，而是做一次**控制变量**的实验：
> 只改训练目标，看"理解语言"这件事对结构有多敏感。

---

## 结果

| 指标 | AR（因果） | Diffusion（双向） |
|---|---|---|
| 参数量 | 1,001,392 | 1,001,392 |
| 验证 bits/char | **2.3506** | 3.5114（ELBO 上界） |
| nats/token | 1.6293 | 2.4339 |
| 困惑度（每字符） | **5.10** | 11.40 |
| 墙钟时间（8 线程 CPU） | 153 min | 158 min |
| 吞吐 | ~8.9k tok/s | ~8.3k tok/s |

**扩散相对自回归落后 +1.16 bits/char。**

作为参照：词表有 3490 个字符，**完全随机猜**需要 `log2(3490) ≈ 11.77 bits/char`。
所以两个模型都把不确定性压掉了七成以上，只是 AR 压得更狠。

> ⚠️ 口径说明：AR 的 `val_bpc` 与扩散的 `val_elbo_bpc` 都是每字符负对数似然的估计，
> **可以直接比较**；扩散的**去噪交叉熵不是上界**，不能与 AR 的 bpc 并列。

---

## 结构

```
src/
  model.py        两个模型共用的 Transformer 骨干（RMSNorm + RoPE + SDPA）
                  ARModel：因果注意力；DiffusionLM：双向 + 掩码去噪
  train.py        统一训练器（学习率 warmup + cosine、验证、ELBO 估计）
  evaluate.py     评测：bpc / ELBO / 困惑度
  sample.py       AR 自回归采样 / 扩散迭代并行解码
  decoders.py     两种解码策略
  prepare_zh.py   中文语料清洗与分词（字符级，3490 词表）
  numpy_engine.py  纯 NumPy 推理引擎（不依赖 PyTorch 即可跑导出权重）
  report.py       自动生成实验报告
models/
  ar.npz / diffusion.npz   导出权重（NumPy，可直接加载推理）
out/
  report.md       完整实验报告（含生成样例）
  eval*.json      评测原始数据
  curves*.png     训练曲线
```

## 数据

- 来源：`hf-mirror.com/datasets/adam89/TinyStoriesChinese`（中文译文，字符级）
- 训练 token：77,882,582；验证：1,549,282
- 词表：3490（含 `<|eos|> <|mask|> <|pad|>`）

## 关键设计

1. **同骨干对照**：两个模型共用 `Block`，只改两处——
   注意力掩码方向（causal / bidirectional）与训练目标（预测下一个 vs 还原被掩码的位置）。
2. **吸收态离散扩散**：文本是离散的，加不了高斯噪声，所以用 `<|mask|>` 做"吸收态"，
   训练时按 `t ~ U(0,1)` 随机遮挡，只对遮挡位置计算交叉熵。
3. **ELBO 的稳定估计**：直接做蒙特卡洛 `(1/t)·Σ` 会有重尾（`t=0.01` 时权重 100），
   改用代数等价的比率估计 `(n_masked/(T·t))·mean` 把估计压稳。
4. **纯 CPU 可复现**：8 线程、`batch 32 × seq 256`、1 万步收敛，不需要显卡。

## 复现

```bash
pip install torch numpy matplotlib
python src/prepare_zh.py          # 生成 train.bin / val.bin
python src/train.py --model ar
python src/train.py --model diffusion
python src/evaluate.py
python src/report.py              # 产出 out/report.md
```

## 环境

Python 3.10+ / PyTorch 2.x（CPU 即可）/ NumPy。
训练不需要 GPU；`numpy_engine.py` 提供无需 PyTorch 的推理路径。

## License

MIT
