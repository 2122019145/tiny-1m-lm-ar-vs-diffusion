"""Prepare a Chinese character-level corpus from the TinyStories-zh archive.

Input : data/zh.tar.gz  (jsonl records with a `story_zh` field)
Output: data/meta.json, data/train.bin, data/val.bin  (uint16 token ids)

Vocabulary = specials + the most frequent CJK characters and punctuation.
Characters outside the vocabulary are dropped (they are rare).
Documents are joined with a single <eos> token.
"""
import argparse
import collections
import json
import os
import tarfile
import time

import numpy as np

SPECIALS = ["<|eos|>", "<|mask|>", "<|pad|>"]
EOS, MASK, PAD = 0, 1, 2
JSONL_BATCH = 2_000_000   # bytes of jsonl consumed per parse() call


def iter_jsonl_lines(tar_path):
    """Stream every line of every *.jsonl member of the archive."""
    with tarfile.open(tar_path, "r:gz") as tf:
        for m in tf:
            if not m.isfile() or not m.name.endswith(".jsonl"):
                continue
            fh = tf.extractfile(m)
            if fh is None:
                continue
            pending = b""
            while True:
                chunk = fh.read(JSONL_BATCH)
                if not chunk:
                    break
                buf = pending + chunk
                cut = buf.rfind(b"\n")
                if cut < 0:
                    pending = buf
                    continue
                for line in buf[:cut].split(b"\n"):
                    if line.strip():
                        yield line
                pending = buf[cut + 1:]
            if pending.strip():
                yield pending


def iter_texts(tar_path, max_docs=None, skip_docs=0):
    n = 0
    seen = 0
    for raw in iter_jsonl_lines(tar_path):
        seen += 1
        if seen <= skip_docs:
            continue
        try:
            rec = json.loads(raw.decode("utf-8", errors="ignore"))
        except Exception:
            continue
        txt = rec.get("story_zh") or ""
        if not txt:
            continue
        yield txt
        n += 1
        if max_docs and n >= max_docs:
            return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/zh.tar.gz")
    ap.add_argument("--out_dir", default="data")
    ap.add_argument("--vocab_size", type=int, default=4000)
    ap.add_argument("--max_train_docs", type=int, default=300_000)
    ap.add_argument("--max_val_docs", type=int, default=6_000)
    ap.add_argument("--scan_docs", type=int, default=40_000,
                    help="docs scanned for the character frequency table")
    args = ap.parse_args()

    t0 = time.time()
    os.makedirs(args.out_dir, exist_ok=True)

    # Pass 1: character frequency over a sample of documents
    freq = collections.Counter()
    n_scan = 0
    for doc in iter_texts(args.src, max_docs=args.scan_docs):
        freq.update(doc)
        n_scan += 1
    if not freq:
        raise SystemExit(f"no documents parsed from {args.src}")
    print(f"[scan] {n_scan} docs, {sum(freq.values())/1e6:.1f}M chars, "
          f"{len(freq)} distinct in {time.time()-t0:.1f}s")

    keep = [c for c, _ in freq.most_common() if c.isprintable() or c in "\n\t "]
    keep = keep[: args.vocab_size]
    vocab = SPECIALS + keep
    stoi = {c: i for i, c in enumerate(vocab) if i >= len(SPECIALS)}
    print(f"[vocab] {len(vocab)} entries; top: {''.join(keep[:30])!r}")

    def encode(docs, out_path, limit_docs):
        buf = []
        n_chars = 0
        n_docs = 0
        dropped = 0
        with open(out_path, "wb") as g:
            for doc in docs:
                if limit_docs and n_docs >= limit_docs:
                    break
                doc = doc.strip()
                if not doc:
                    continue
                n_docs += 1
                n_chars += len(doc)
                for ch in doc:
                    j = stoi.get(ch)
                    if j is None:
                        dropped += 1
                        continue
                    buf.append(j)
                buf.append(EOS)
                if len(buf) > 2_000_000:
                    g.write(np.asarray(buf, dtype=np.uint16).tobytes())
                    buf = []
            if buf:
                g.write(np.asarray(buf, dtype=np.uint16).tobytes())
        return n_chars, n_docs, dropped

    # train: first N docs
    tr_c, tr_d, tr_drop = encode(iter_texts(args.src), os.path.join(args.out_dir, "train.bin"),
                                 args.max_train_docs)
    print(f"[train] {tr_d} docs, {tr_c/1e6:.1f}M chars, dropped {tr_drop} "
          f"({100*tr_drop/max(1,tr_c+tr_drop):.2f}%) in {time.time()-t0:.1f}s")

    # val: held-out docs after the training range
    va_c, va_d, va_drop = encode(
        iter_texts(args.src, skip_docs=args.max_train_docs),
        os.path.join(args.out_dir, "val.bin"), args.max_val_docs)
    print(f"[val]   {va_d} docs, {va_c/1e6:.1f}M chars in {time.time()-t0:.1f}s")

    tr_bytes = os.path.getsize(os.path.join(args.out_dir, "train.bin"))
    va_bytes = os.path.getsize(os.path.join(args.out_dir, "val.bin"))
    meta = {
        "vocab": vocab,
        "vocab_size": len(vocab),
        "eos_id": EOS, "mask_id": MASK, "pad_id": PAD,
        "char_level": True, "language": "zh",
        "train_chars": tr_c, "val_chars": va_c,
        "train_tokens": tr_bytes // 2, "val_tokens": va_bytes // 2,
    }
    with open(os.path.join(args.out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    print(f"[done] vocab_size={len(vocab)} train_tokens={tr_bytes//2/1e6:.2f}M "
          f"val_tokens={va_bytes//2/1e6:.2f}M {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
