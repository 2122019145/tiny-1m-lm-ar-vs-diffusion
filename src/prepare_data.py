"""Prepare character-level TinyStories data.

Builds:
  data/meta.json          -- vocab, special ids, sizes
  data/train.bin          -- uint16 token ids, concatenated documents
  data/val.bin            -- uint16 token ids
Documents are joined with a single <eos> token (id 0); no BOS needed because
both models learn document boundaries from <eos>.
"""
import argparse
import json
import os
import time

EOS = "<|eos|>"
MASK = "<|mask|>"
PAD = "<|pad|>"
SPECIALS = [EOS, MASK, PAD]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_file", default="data/TinyStoriesV2-GPT4-train.txt")
    ap.add_argument("--val_file", default="data/TinyStoriesV2-GPT4-valid.txt")
    ap.add_argument("--out_dir", default="data")
    ap.add_argument("--max_train_chars", type=int, default=60_000_000)
    ap.add_argument("--max_val_chars", type=int, default=2_000_000)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    t0 = time.time()

    # Pass 1: learn the vocabulary from a prefix of the training text.
    with open(args.train_file, "r", encoding="utf-8") as f:
        head = f.read(args.max_train_chars)
    chars = sorted(set(head))
    # Keep printable chars only; drop rare control chars.
    chars = [c for c in chars if c.isprintable() or c in "\n\t "]
    vocab = SPECIALS + chars
    stoi = {c: i for i, c in enumerate(vocab)}
    # <|endoftext|> markers in the raw text map to EOS
    doc_sep = "<|endoftext|>"
    stoi[doc_sep] = 0
    print(f"[vocab] {len(vocab)} tokens from {len(head)/1e6:.1f}M chars in {time.time()-t0:.1f}s")

    def encode_file(path, max_chars, out_path):
        n_chars = 0
        n_tokens = 0
        skipped = 0
        eos_id = 0
        with open(path, "r", encoding="utf-8") as f, open(out_path, "wb") as g:
            buf = []
            while True:
                chunk = f.read(8 << 20)
                if not chunk:
                    break
                if max_chars and n_chars >= max_chars:
                    break
                if max_chars:
                    chunk = chunk[: max_chars - n_chars]
                n_chars += len(chunk)
                # split out document separators first
                for i, part in enumerate(chunk.split(doc_sep)):
                    if i > 0:
                        buf.append(eos_id)
                        n_tokens += 1
                    ids = []
                    for ch in part:
                        j = stoi.get(ch)
                        if j is None:
                            skipped += 1
                            continue
                        ids.append(j)
                    buf.extend(ids)
                    n_tokens += len(ids)
                if len(buf) > 4_000_000:
                    g.write(_pack(buf))
                    buf = []
            if buf:
                g.write(_pack(buf))
        return n_chars, n_tokens, skipped

    def _pack(ids):
        import numpy as np
        return np.asarray(ids, dtype=np.uint16).tobytes()

    tr_chars, tr_tokens, tr_skip = encode_file(args.train_file, args.max_train_chars,
                                               os.path.join(args.out_dir, "train.bin"))
    print(f"[train] {tr_chars/1e6:.1f}M chars -> {tr_tokens/1e6:.2f}M tokens "
          f"(skipped {tr_skip}) in {time.time()-t0:.1f}s")
    va_chars, va_tokens, va_skip = encode_file(args.val_file, args.max_val_chars,
                                               os.path.join(args.out_dir, "val.bin"))
    print(f"[val]   {va_chars/1e6:.1f}M chars -> {va_tokens/1e6:.2f}M tokens "
          f"(skipped {va_skip}) in {time.time()-t0:.1f}s")

    meta = {
        "vocab": vocab,
        "vocab_size": len(vocab),
        "eos_id": 0,
        "mask_id": 1,
        "pad_id": 2,
        "train_tokens": tr_tokens,
        "val_tokens": va_tokens,
        "char_level": True,
    }
    with open(os.path.join(args.out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    print(f"[done] vocab_size={len(vocab)} total {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
