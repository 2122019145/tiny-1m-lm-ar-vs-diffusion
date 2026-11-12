"""Two ~1M-parameter language models on a shared Transformer block.

  * ARModel      : causal (autoregressive) decoder, next-token prediction.
  * DiffusionLM  : non-causal (bidirectional) encoder trained with masked
                   discrete diffusion (absorbing-state / mask-predict, a la
                   MDLM / LLaDA), iterative parallel decoding at inference.

Both share `Block`, so the parameter counts stay comparable.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Config:
    def __init__(self, vocab_size, n_layer=4, n_head=4, d_model=128,
                 d_ff=512, max_seq_len=256, dropout=0.0, tie_embedding=True):
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.d_model = d_model
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.tie_embedding = tie_embedding

    def to_dict(self):
        return dict(self.__dict__)


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return self.weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


def build_rope_cache(seq_len, head_dim, base=10000.0, device=None, dtype=torch.float32):
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)           # (T, head_dim/2)
    return torch.cos(freqs).to(dtype), torch.sin(freqs).to(dtype)


def apply_rope(x, cos, sin):
    """x: (B, H, T, Dh)"""
    T = x.size(-2)
    cos = cos[:T].unsqueeze(0).unsqueeze(0)
    sin = sin[:T].unsqueeze(0).unsqueeze(0)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.d_model % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.head_dim = cfg.d_model // cfg.n_head
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.dropout = cfg.dropout

    def forward(self, x, cos, sin, causal):
        B, T, C = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n_head, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]              # (B,H,T,Dh)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        y = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=causal,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.fc2 = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.fc2(F.gelu(self.fc(x), approximate="tanh")))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n1 = RMSNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.n2 = RMSNorm(cfg.d_model)
        self.mlp = MLP(cfg)

    def forward(self, x, cos, sin, causal):
        x = x + self.attn(self.n1(x), cos, sin, causal)
        x = x + self.mlp(self.n2(x))
        return x


class _Backbone(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.norm_f = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embedding:
            self.lm_head.weight = self.tok_emb.weight
        self.apply(self._init)
        # scaled init for residual projections
        for name, p in self.named_parameters():
            if name.endswith("proj.weight") or name.endswith("fc2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))
        self._rope = {}

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def _get_rope(self, T, device, dtype):
        key = (T, device, dtype)
        if key not in self._rope:
            self._rope[key] = build_rope_cache(
                max(T, self.cfg.max_seq_len), self.cfg.d_model // self.cfg.n_head,
                device=device, dtype=dtype)
        return self._rope[key]

    def n_params(self, non_embedding=False):
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.tok_emb.weight.numel()
            if not self.cfg.tie_embedding:
                n -= self.lm_head.weight.numel()
        return n

    def body(self, idx, causal):
        B, T = idx.shape
        x = self.drop(self.tok_emb(idx))
        cos, sin = self._get_rope(T, idx.device, x.dtype)
        for blk in self.blocks:
            x = blk(x, cos, sin, causal)
        return self.norm_f(x)


class ARModel(_Backbone):
    """Causal decoder: p(x_t | x_<t)."""
    name = "ar"

    def forward(self, idx, targets=None):
        h = self.body(idx, causal=True)
        logits = self.lm_head(h)
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1),
                               ignore_index=-100)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, eos_id=None):
        self.eval()
        for _ in range(max_new_tokens):
            ctx = idx[:, -self.cfg.max_seq_len:]
            logits, _ = self(ctx)
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, nxt], dim=1)
            if eos_id is not None and (nxt == eos_id).all():
                break
        return idx


class DiffusionLM(_Backbone):
    """Bidirectional encoder trained with absorbing-state masked diffusion.

    Training  : sample mask ratio t ~ U(0,1); mask each token w.p. t;
                predict the original tokens at masked positions only.
    Inference : start from fully masked sequence, repeatedly unmask the
                most confident positions (mask-predict / LLaDA-style).
    """
    name = "diffusion"

    def forward(self, idx, targets=None, mask_ratio=None, generator=None):
        B, T = idx.shape
        if targets is None:
            # INFERENCE: the caller already built the canvas (decoded tokens plus
            # explicit mask tokens).  Do NOT re-mask it here -- doing so would
            # re-corrupt every already-decoded position at each sampling step.
            h = self.body(idx, causal=False)
            return self.lm_head(h), None
        if mask_ratio is None:
            t = torch.rand(B, 1, device=idx.device, generator=generator)
        else:
            t = torch.full((B, 1), float(mask_ratio), device=idx.device)
        # keep at least one masked position per sequence
        t = t.clamp(min=1.0 / T, max=1.0)
        mask = torch.rand(B, T, device=idx.device, generator=generator) < t
        mask[:, 0] = mask[:, 0] | ~mask.any(dim=1)   # guarantee non-empty
        noisy = torch.where(mask, torch.full_like(idx, self.cfg.mask_token_id), idx)
        h = self.body(noisy, causal=False)
        logits = self.lm_head(h)
        tgt = torch.where(mask, targets, torch.full_like(targets, -100))
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), tgt.reshape(-1),
                               ignore_index=-100)
        return logits, loss

    @torch.no_grad()
    def generate(self, batch_size, seq_len, steps=32, temperature=1.0,
                 device="cpu", generator=None, top_k=None, prompt_ids=None):
        """Iterative parallel decoding: unmask the k most confident tokens per step.

        `prompt_ids` (list[int]) is clamped into the front of the canvas and kept
        fixed, turning the unconditional sampler into a continuation sampler.
        """
        self.eval()
        M = self.cfg.mask_token_id
        x = torch.full((batch_size, seq_len), M, dtype=torch.long, device=device)
        n_prompt = 0
        if prompt_ids:
            p = torch.as_tensor(prompt_ids, dtype=torch.long, device=device)
            n_prompt = min(len(p), seq_len)
            x[:, :n_prompt] = p[:n_prompt]
        steps = max(1, min(steps, seq_len))
        for s in range(steps):
            logits, _ = self(x)
            logits = logits / max(temperature, 1e-5)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1)
                logits[logits < v[..., [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            pred = torch.multinomial(probs.view(-1, probs.size(-1)), 1,
                                     generator=generator).view(batch_size, seq_len)
            conf = probs.gather(-1, pred.unsqueeze(-1)).squeeze(-1)  # (B,T)
            conf = torch.where(x == M, conf, torch.full_like(conf, -1.0))
            n_left = (x == M).sum(dim=1)
            k = torch.clamp((n_left + (steps - s) - 1) // (steps - s), min=1)
            k_max = int(k.max().item())
            # positions to fill this step: top-k per row among still-masked
            order = torch.argsort(conf, dim=1, descending=True)[:, :k_max]
            fill = torch.zeros_like(x, dtype=torch.bool)
            for i in range(batch_size):
                fill[i, order[i, :int(k[i].item())]] = True
            x = torch.where(fill, pred, x)
            if not (x == M).any():
                break
        return x