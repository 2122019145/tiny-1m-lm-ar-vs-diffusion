"""Diffusion decoders for absorbing-state masked diffusion LMs.

Both start from a canvas that is fully <|mask|> except for an optional fixed
prompt prefix.  They differ in whether an early guess can be revised:

  sample_conf   -- standard mask-predict: unmask the k most confident positions
                   each step and never revisit them.
  sample_remask -- ReMDM/ReMask style: re-score EVERY position at every step,
                   keep the globally best n_keep = seq_len*(s+1)/steps and
                   re-mask the rest, so a bad early guess is revocable.

`repetition_score` is the fraction of repeated 4-char shingles in a decoded
string: 0 is fully diverse, 1 is degenerate (a single shingle repeated).
"""
import math
import re

import torch


def _prep(prompt_ids, seq_len, device, M):
    """Build a fully-masked canvas with an optional fixed prompt prefix.

    prompt_ids may be a list, a 1-D tensor, or None -- `if prompt_ids:` would
    raise "Boolean value of Tensor with more than one value is ambiguous" on a
    tensor, so test for None explicitly.
    """
    x = torch.full((1, seq_len), M, dtype=torch.long, device=device)
    if prompt_ids is None:
        return x, 0
    p = torch.as_tensor(prompt_ids, dtype=torch.long, device=device).reshape(-1)
    n = min(int(p.numel()), seq_len)
    if n > 0:
        x[:, :n] = p[:n]
    return x, n


def _logits(model, x, temperature, top_k):
    logits, _ = model(x)
    logits = logits.float() / max(temperature, 1e-6)
    if top_k:
        v = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1).values[..., -1:]
        logits = logits.masked_fill(logits < v, float("-inf"))
    return torch.softmax(logits, dim=-1)


def _pick(probs, generator, stochastic):
    """Return (chosen_token, its_probability).

    Greedy argmax makes every seed produce the same story, so when a generator is
    supplied we actually draw from the (temperature/top-k shaped) distribution and
    use the drawn token's own probability as its confidence score.
    """
    if stochastic and generator is not None:
        flat = probs.reshape(-1, probs.size(-1))
        tok = torch.multinomial(flat, 1, generator=generator).view(probs.shape[:-1])
        p = probs.gather(-1, tok.unsqueeze(-1)).squeeze(-1)
        return tok, p
    p, tok = probs.max(dim=-1)
    return tok, p


def sample_conf(model, seq_len, steps, temperature, top_k, generator, prompt_ids=None,
                stochastic=True):
    """Standard mask-predict: unmask the k most confident, never revisit."""
    M = model.cfg.mask_token_id
    dev = next(model.parameters()).device
    x, n_p = _prep(prompt_ids, seq_len, dev, M)
    pin = x[0, :n_p].clone() if n_p else None
    for s in range(steps):
        still = (x == M)
        n_left = int(still.sum().item())
        if n_left == 0:
            break
        probs = _logits(model, x, temperature, top_k)
        pred, conf = _pick(probs, generator, stochastic)
        conf = conf.masked_fill(~still, -1e9)
        k = max(1, math.ceil(n_left / (steps - s)))
        idx = torch.topk(conf, min(k, seq_len), dim=-1).indices
        x[0, idx[0]] = pred[0, idx[0]]
        if pin is not None:
            x[0, :n_p] = pin
    return x


def sample_remask(model, seq_len, steps, temperature, top_k, generator, prompt_ids=None,
                  final_only=False, stochastic=True):
    """ReMDM: re-score every position each step and keep only the top n_keep."""
    M = model.cfg.mask_token_id
    dev = next(model.parameters()).device
    x, n_p = _prep(prompt_ids, seq_len, dev, M)
    pin = x[0, :n_p].clone() if n_p else None
    for s in range(steps):
        probs = _logits(model, x, temperature, top_k)
        pred, conf = _pick(probs, generator, stochastic)
        cur_p = probs.gather(-1, x.clamp(min=0).unsqueeze(-1)).squeeze(-1)
        score = torch.where(x == M, conf, cur_p)
        if pin is not None:
            score[0, :n_p] = float("inf")
        n_keep = max(1, round(seq_len * (s + 1) / steps))
        keep = torch.topk(score, min(n_keep, seq_len), dim=-1).indices
        newx = torch.full_like(x, M)
        cur = x.gather(1, keep)
        newx.scatter_(1, keep, torch.where(cur == M, pred.gather(1, keep), cur))
        x = newx
        if pin is not None:
            x[0, :n_p] = pin
    return x


def repetition_score(text):
    """Fraction of 4-char shingles that repeat -- 0 is diverse, 1 is degenerate."""
    t = re.sub(r"\s+", "", text)
    if len(t) < 8:
        return 1.0
    sh = [t[i:i + 4] for i in range(len(t) - 3)]
    if not sh:
        return 1.0
    return 1.0 - len(set(sh)) / len(sh)
