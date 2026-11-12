"""Pure-numpy inference for the two trained models.

No torch: the shipped package only needs numpy, so the whole thing is a few tens
of MB instead of the ~680 MB a torch venv would cost.

The numerics are written to match src/model.py exactly (RMSNorm eps 1e-5,
interleaved RoPE, SDPA scale 1/sqrt(head_dim), tanh-approximate GELU, tied
embedding).  src/export_npz.py dumps the torch weights into .npz, and
src/check_numpy.py asserts this engine reproduces the torch logits.
"""
import json
import math
import os

import numpy as np


def _rms_norm(x, w, eps=1e-5):
    return w * x / np.sqrt((x * x).mean(-1, keepdims=True) + eps)


def _gelu_tanh(x):
    # matches F.gelu(x, approximate="tanh")
    return 0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))


def _rope_cache(seq_len, head_dim, base=10000.0):
    inv_freq = 1.0 / (base ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
    t = np.arange(seq_len, dtype=np.float32)
    freqs = np.outer(t, inv_freq).astype(np.float32)      # (T, Dh/2)
    return np.cos(freqs).astype(np.float32), np.sin(freqs).astype(np.float32)


def _apply_rope(x, cos, sin):
    """x: (B, H, T, Dh); rotate interleaved (even, odd) pairs."""
    T = x.shape[-2]
    c = cos[:T][None, None, :, :]
    s = sin[:T][None, None, :, :]
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = np.empty_like(x)
    out[..., 0::2] = x1 * c - x2 * s
    out[..., 1::2] = x1 * s + x2 * c
    return out


def _softmax(x):
    x = x - x.max(-1, keepdims=True)
    np.exp(x, out=x)
    x /= x.sum(-1, keepdims=True)
    return x


class NumpyModel:
    """Loads one .npz + its model_info.json and runs forward passes."""

    def __init__(self, npz_path, info_path):
        z = np.load(npz_path, allow_pickle=False)
        self.w = {k: z[k] for k in z.files}
        with open(info_path, encoding="utf-8") as f:
            info = json.load(f)
        self.cfg = info["cfg"]
        self.vocab = info["vocab"]
        self.eos_id = info["eos_id"]
        self.mask_id = info["mask_id"]
        self.arch = info["arch"]
        self.n_layer = self.cfg["n_layer"]
        self.n_head = self.cfg["n_head"]
        self.d_model = self.cfg["d_model"]
        self.head_dim = self.d_model // self.n_head
        self.max_seq_len = self.cfg["max_seq_len"]
        self._rope = {}
        self.stoi = {c: i for i, c in enumerate(self.vocab)}

    # ---- internals ---------------------------------------------------------
    def _get_rope(self, T):
        key = max(T, self.max_seq_len)
        if key not in self._rope:
            self._rope[key] = _rope_cache(key, self.head_dim)
        return self._rope[key]

    def forward(self, idx, causal):
        """idx: (B, T) int64 -> logits (B, T, vocab)."""
        B, T = idx.shape
        C = self.d_model
        cos, sin = self._get_rope(T)
        x = self.w["tok_emb.weight"][idx]                     # (B,T,C)
        scale = 1.0 / math.sqrt(self.head_dim)

        for li in range(self.n_layer):
            p = f"blocks.{li}."
            # --- attention (pre-norm) ---
            h = _rms_norm(x, self.w[p + "n1.weight"])
            qkv = h @ self.w[p + "attn.qkv.weight"].T          # (B,T,3C)
            qkv = qkv.reshape(B, T, 3, self.n_head, self.head_dim).transpose(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]                   # (B,H,T,Dh)
            q = _apply_rope(q, cos, sin)
            k = _apply_rope(k, cos, sin)

            att = (q @ k.transpose(0, 1, 3, 2)) * scale        # (B,H,T,T)
            if causal:
                mask = np.triu(np.ones((T, T), dtype=bool), k=1)
                att = np.where(mask[None, None, :, :], -np.inf, att)
            att = _softmax(att)
            y = att @ v                                        # (B,H,T,Dh)
            y = y.transpose(0, 2, 1, 3).reshape(B, T, C)
            x = x + y @ self.w[p + "attn.proj.weight"].T

            # --- MLP (pre-norm) ---
            h = _rms_norm(x, self.w[p + "n2.weight"])
            h = _gelu_tanh(h @ self.w[p + "mlp.fc.weight"].T)
            x = x + h @ self.w[p + "mlp.fc2.weight"].T

        x = _rms_norm(x, self.w["norm_f.weight"])
        # lm_head is tied to tok_emb
        return x @ self.w["tok_emb.weight"].T

    # ---- decoding ----------------------------------------------------------
    def decode_ids(self, ids):
        out = []
        for i in ids:
            i = int(i)
            if i == self.eos_id:
                break
            if i < 3:
                continue
            out.append(self.vocab[i])
        return "".join(out)

    def encode(self, text):
        ids, dropped = [], []
        for c in text:
            if c in self.stoi:
                ids.append(self.stoi[c])
            else:
                dropped.append(c)
        return ids, dropped

    def _sample(self, probs, rng, top_k):
        if top_k:
            k = min(top_k, probs.size)
            keep = np.argpartition(probs, -k)[-k:]
            p = np.zeros_like(probs)
            p[keep] = probs[keep]
            s = p.sum()
            probs = p / s if s > 0 else probs
        return int(rng.choice(probs.size, p=probs))

    def generate_ar(self, prompt_ids, max_new_tokens, temperature=0.85, top_k=50,
                    rng=None):
        rng = rng or np.random.default_rng(0)
        idx = list(prompt_ids)
        for _ in range(max_new_tokens):
            ctx = np.array([idx[-self.max_seq_len:]], dtype=np.int64)
            logits = self.forward(ctx, causal=True)[0, -1]
            logits = logits / max(temperature, 1e-5)
            if top_k:
                k = min(top_k, logits.size)
                cut = np.partition(logits, -k)[-k]
                logits = np.where(logits < cut, -np.inf, logits)
            probs = _softmax(logits.astype(np.float64))
            nxt = self._sample(probs, rng, None)
            idx.append(nxt)
            if nxt == self.eos_id:
                break
        return idx

    def generate_diffusion(self, prompt_ids, seq_len, steps=64, temperature=1.0,
                           top_k=50, rng=None, decoder="conf"):
        rng = rng or np.random.default_rng(0)
        M = self.mask_id
        x = np.full((1, seq_len), M, dtype=np.int64)
        n_p = min(len(prompt_ids), seq_len)
        if n_p:
            x[0, :n_p] = np.array(prompt_ids[:n_p], dtype=np.int64)
        pin = x[0, :n_p].copy() if n_p else None
        steps = max(1, min(steps, seq_len))

        for s in range(steps):
            logits = self.forward(x, causal=False)[0] / max(temperature, 1e-5)
            if top_k:
                k = min(top_k, logits.shape[-1])
                cut = np.partition(logits, -k, axis=-1)[..., -k][:, None]
                logits = np.where(logits < cut, -np.inf, logits)
            probs = _softmax(logits.astype(np.float64))

            if decoder == "remask":
                cur_p = np.take_along_axis(probs, np.clip(x[0], 0, None)[:, None], -1)[:, 0]
                pred = np.array([self._sample(probs[i], rng, None) for i in range(seq_len)])
                conf = probs[np.arange(seq_len), pred]
                score = np.where(x[0] == M, conf, cur_p)
                if pin is not None:
                    score[:n_p] = np.inf
                n_keep = max(1, round(seq_len * (s + 1) / steps))
                keep = np.argsort(-score)[:n_keep]
                newx = np.full(seq_len, M, dtype=np.int64)
                for j in keep:
                    newx[j] = pred[j] if x[0, j] == M else x[0, j]
                x[0] = newx
            else:
                pred = np.array([self._sample(probs[i], rng, None) for i in range(seq_len)])
                conf = probs[np.arange(seq_len), pred]
                still = x[0] == M
                conf = np.where(still, conf, -1e9)
                n_left = int(still.sum())
                if n_left == 0:
                    break
                k = max(1, math.ceil(n_left / (steps - s)))
                idx = np.argsort(-conf)[:k]
                x[0, idx] = pred[idx]
            if pin is not None:
                x[0, :n_p] = pin
        return x[0]
