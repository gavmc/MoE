"""Pack smol-smoltalk for SFT (train.md sec 9).

Unlike prepare.py this cannot read a flat `text` column -- smol-smoltalk is
`messages: list<struct<content, role>>`. Each conversation is rendered with a
plain-text chat template (the tokenizer has no chat special tokens and adding any
would resize the tied embedding and break the base checkpoint), then tokenized
with the SAME tokenizer.json the base model was trained on.

Two parallel arrays are written per shard:
  *.bin   uint16 token ids
  *.mask  uint8, 1 where loss should be taken (assistant turns only)

Effort conditioning (sec 9, K3 sec 4.1.2): each example is tagged by the length
band of its assistant turns, stated in natural language before the first user
turn. At inference the tag is the only thing that varies, so output length can be
shown to respond to it.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer

EOS = "<|endoftext|>"
FLUSH = 4_000_000

EFFORT = {
    "low": "Answer briefly.",
    "medium": "Answer at moderate length.",
    "high": "Answer in detail.",
}


def parquet_files(src):
    f = sorted(Path(src).glob("**/*.parquet"))
    if not f:
        raise SystemExit(f"no .parquet under {src}")
    return f


def segments(msgs, band):
    """Yield (text, is_assistant). Assistant turns are the only trained tokens."""
    out, first_user = [], True
    for m in msgs:
        role, content = m["role"], (m["content"] or "").strip()
        if not content:
            continue
        if role == "system":
            out.append((f"System: {content}\n", False))
        elif role == "user":
            if first_user:
                out.append((f"{EFFORT[band]}\n", False))
                first_user = False
            out.append((f"User: {content}\nAssistant:", False))
        elif role == "assistant":
            out.append((f" {content}", True))
    return out


def assistant_chars(msgs):
    return sum(len(m["content"] or "") for m in msgs if m["role"] == "assistant")


def band_edges(files, sample=40_000):
    """Tertiles of assistant length, so the three bands are equally populated."""
    lens, n = [], 0
    for p in files:
        for rb in pq.ParquetFile(p).iter_batches(batch_size=1000, columns=["messages"]):
            for msgs in rb.column("messages").to_pylist():
                lens.append(assistant_chars(msgs))
                n += 1
                if n >= sample:
                    a = np.array(lens)
                    return float(np.quantile(a, 1 / 3)), float(np.quantile(a, 2 / 3))
    a = np.array(lens)
    return float(np.quantile(a, 1 / 3)), float(np.quantile(a, 2 / 3))


def pack(tok, files, out_dir, eos_id, lo, hi, shard_tokens=100_000_000):
    tokbuf, maskbuf = [], []
    shards, idx, written = [], 0, 0
    counts = {"low": 0, "medium": 0, "high": 0}
    trained = total = 0

    def flush(final=False):
        nonlocal tokbuf, maskbuf, idx, written
        if not tokbuf:
            return
        name = f"sft_{idx:04d}"
        t = np.asarray(tokbuf, dtype=np.uint16)
        m = np.asarray(maskbuf, dtype=np.uint8)
        tmp_t, tmp_m = out_dir / f"{name}.bin.partial", out_dir / f"{name}.mask.partial"
        tmp_t.write_bytes(t.tobytes())
        tmp_m.write_bytes(m.tobytes())
        tmp_t.replace(out_dir / f"{name}.bin")
        tmp_m.replace(out_dir / f"{name}.mask")
        shards.append({"file": f"{name}.bin", "mask": f"{name}.mask", "tokens": int(t.size)})
        written += t.size
        idx += 1
        tokbuf, maskbuf = [], []

    for p in files:
        for rb in pq.ParquetFile(p).iter_batches(batch_size=500, columns=["messages"]):
            rows = rb.column("messages").to_pylist()
            bands, segs, spans = [], [], []
            for msgs in rows:
                c = assistant_chars(msgs)
                band = "low" if c <= lo else ("medium" if c <= hi else "high")
                bands.append(band)
                s = segments(msgs, band)
                spans.append(len(s))
                segs.extend(s)
            encs = tok.encode_batch([t for t, _ in segs])
            k = 0
            for row_i, nseg in enumerate(spans):
                ids, mask = [], []
                for j in range(nseg):
                    e = encs[k + j]
                    ids.extend(e.ids)
                    mask.extend([1 if segs[k + j][1] else 0] * len(e.ids))
                k += nseg
                if not any(mask):
                    continue
                ids.append(eos_id)
                mask.append(1)              # learn to stop
                tokbuf.extend(ids)
                maskbuf.extend(mask)
                counts[bands[row_i]] += 1
                trained += sum(mask)
                total += len(ids)
            if len(tokbuf) >= shard_tokens:
                flush()
    flush(final=True)
    return shards, written, counts, trained, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=r"D:\ml\raw\smoltalk")
    ap.add_argument("--tokenizer", default=r"D:\ml\data\tokenizer.json")
    ap.add_argument("--out", default=r"D:\ml\data-sft")
    ap.add_argument("--holdout-shards", type=int, default=1)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tok = Tokenizer.from_file(args.tokenizer)
    eos_id = tok.token_to_id(EOS)
    assert eos_id is not None, "tokenizer has no EOS -- wrong tokenizer.json?"

    files = parquet_files(args.source)
    print(f"{len(files)} parquet files, tokenizer vocab {tok.get_vocab_size()}")

    t0 = time.time()
    lo, hi = band_edges(files)
    print(f"effort bands by assistant chars: low<={lo:.0f}  medium<={hi:.0f}  high>")

    shards, written, counts, trained, total = pack(tok, files, out, eos_id, lo, hi)
    dt = time.time() - t0

    hold = [s["file"] for s in shards[-args.holdout_shards:]] if len(shards) > 1 else []
    manifest = {
        "tokenizer": "tokenizer.json",
        "vocab_size": tok.get_vocab_size(),
        "eos_id": eos_id,
        "dtype": "uint16",
        "masked": True,
        "effort_bands": {"low_max_chars": lo, "medium_max_chars": hi},
        "effort_prompts": EFFORT,
        "examples_per_band": counts,
        "shards": shards,
        "holdout": hold,
        "total_tokens": int(written),
        "trained_tokens": int(trained),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    # the SFT loader needs the tokenizer beside the data, same as prepare.py
    if not (out / "tokenizer.json").exists():
        (out / "tokenizer.json").write_bytes(Path(args.tokenizer).read_bytes())

    print(f"\n{len(shards)} shards, {written/1e6:.1f}M tokens in {dt:.0f}s")
    print(f"  trained (assistant) tokens {trained/1e6:.1f}M = {trained/max(1,total):.1%}")
    print(f"  examples per band: {counts}")
    print(f"  holdout: {hold or 'none'}")


if __name__ == "__main__":
    main()
