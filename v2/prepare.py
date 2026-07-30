import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

EOS = "<|endoftext|>"
FLUSH_TOKENS = 4_000_000


def parquet_files(src):
    files = sorted(Path(src).glob("**/*.parquet"))
    if not files:
        raise SystemExit(f"no .parquet under {src}")
    return files


def iter_texts(path, column, batch_size=1000):
    f = pq.ParquetFile(path)
    for rb in f.iter_batches(batch_size=batch_size, columns=[column]):
        yield [t for t in rb.column(column).to_pylist() if t]


def train_tokenizer(files, column, vocab_size, sample_docs, out_path):
    def sample():
        n = 0
        for path in files:
            for texts in iter_texts(path, column):
                for t in texts:
                    yield t
                    n += 1
                    if n >= sample_docs:
                        return

    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=[EOS],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    tok.train_from_iterator(sample(), trainer=trainer)
    tok.save(str(out_path))
    return tok


def pack_shard(tok, src, column, out_path, eos_id):
    buf, written = [], 0
    tmp = out_path.with_suffix(".partial")
    with open(tmp, "wb") as f:
        for texts in iter_texts(src, column):
            for enc in tok.encode_batch(texts):
                buf.extend(enc.ids)
                buf.append(eos_id)
            if len(buf) >= FLUSH_TOKENS:
                a = np.asarray(buf, dtype=np.uint16)
                f.write(a.tobytes())
                written += a.size
                buf = []
        if buf:
            a = np.asarray(buf, dtype=np.uint16)
            f.write(a.tobytes())
            written += a.size
    tmp.replace(out_path)
    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--column", default="text")
    ap.add_argument("--vocab-size", type=int, default=32000)
    ap.add_argument("--tokenizer-sample", type=int, default=400_000)
    ap.add_argument("--holdout-shards", type=int, default=2)
    ap.add_argument("--limit-files", type=int, default=0)
    args = ap.parse_args()

    if args.vocab_size > 65536:
        raise SystemExit("vocab_size must fit in uint16")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.json"
    tok_path = out / "tokenizer.json"

    files = parquet_files(args.source)
    if args.limit_files:
        files = files[: args.limit_files]

    if tok_path.exists():
        tok = Tokenizer.from_file(str(tok_path))
        print(f"tokenizer: reusing {tok_path}")
    else:
        print(f"tokenizer: training on <={args.tokenizer_sample} docs")
        t0 = time.time()
        tok = train_tokenizer(files, args.column, args.vocab_size, args.tokenizer_sample, tok_path)
        print(f"tokenizer: {tok.get_vocab_size()} tokens in {time.time()-t0:.0f}s")

    eos_id = tok.token_to_id(EOS)
    assert eos_id is not None

    manifest = {
        "tokenizer": tok_path.name,
        "vocab_size": tok.get_vocab_size(),
        "eos_id": eos_id,
        "dtype": "uint16",
        "seq_source": str(Path(args.source).resolve()),
        "shards": [],
        "holdout": [],
    }
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    done = {s["source"]: s for s in manifest["shards"]}

    for i, src in enumerate(files):
        name = f"shard_{i:04d}.bin"
        path = out / name
        if src.name in done and path.exists():
            print(f"[{i+1}/{len(files)}] {name} exists, skipping")
            continue
        t0 = time.time()
        n = pack_shard(tok, src, args.column, path, eos_id)
        dt = time.time() - t0
        manifest["shards"] = [s for s in manifest["shards"] if s["source"] != src.name]
        manifest["shards"].append({"file": name, "source": src.name, "tokens": n})
        manifest["shards"].sort(key=lambda s: s["file"])
        manifest_path.write_text(json.dumps(manifest, indent=2))
        print(f"[{i+1}/{len(files)}] {name}  {n/1e6:9.1f}M tokens  {dt:6.0f}s  {n/dt/1e3:6.0f}k tok/s")

    shards = manifest["shards"]
    k = min(args.holdout_shards, max(0, len(shards) - 1))
    manifest["holdout"] = [s["file"] for s in shards[-k:]] if k else []
    total = sum(s["tokens"] for s in shards)
    hold = sum(s["tokens"] for s in shards if s["file"] in set(manifest["holdout"]))
    manifest["total_tokens"] = total
    manifest["holdout_tokens"] = hold
    manifest["train_tokens"] = total - hold
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(f"\n{len(shards)} shards, {total/1e9:.3f}B tokens")
    print(f"  train   {(total-hold)/1e9:.3f}B")
    print(f"  holdout {hold/1e9:.3f}B  ({', '.join(manifest['holdout']) or 'none'})")
    print(f"  on disk {total*2/1e9:.1f} GB")


if __name__ == "__main__":
    main()
