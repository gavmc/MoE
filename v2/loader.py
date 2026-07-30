import json
from pathlib import Path

import numpy as np
import torch


class PackedLoader:
    def __init__(self, shard_paths, seq_len, seed=0, pos=0):
        self.paths = [str(Path(p)) for p in shard_paths]
        self.shards = [np.memmap(p, dtype=np.uint16, mode="r") for p in self.paths]
        self.seq_len = seq_len
        self.seed = seed

        counts = [max(0, (len(a) - 1) // seq_len) for a in self.shards]
        self.starts = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
        self.total = int(self.starts[-1])
        if self.total == 0:
            raise ValueError(f"no full sequences of length {seq_len} in {self.paths}")

        self.epoch = -1
        self.pos = pos
        self._ensure_perm()

    def _ensure_perm(self):
        e = self.pos // self.total
        if e != self.epoch:
            self.epoch = e
            self.perm = np.random.default_rng(self.seed + e).permutation(self.total)

    def _locate(self, g):
        s = int(np.searchsorted(self.starts, g, side="right") - 1)
        return s, int(g - self.starts[s]) * self.seq_len

    def batch(self, n, device="cuda"):
        out = np.empty((n, self.seq_len + 1), dtype=np.int64)
        for j in range(n):
            self._ensure_perm()
            s, o = self._locate(self.perm[self.pos % self.total])
            out[j] = self.shards[s][o : o + self.seq_len + 1]
            self.pos += 1
        t = torch.from_numpy(out).to(device, non_blocking=True)
        return t[:, :-1].contiguous(), t[:, 1:].contiguous()

    def warm(self):
        for a in self.shards:
            for i in range(0, len(a), 1 << 22):
                _ = a[i]

    @property
    def tokens_seen(self):
        return self.pos * self.seq_len

    def state_dict(self):
        return {"pos": self.pos, "seed": self.seed, "seq_len": self.seq_len, "paths": self.paths}

    def load_state_dict(self, sd):
        if sd["paths"] != self.paths or sd["seq_len"] != self.seq_len:
            raise ValueError("loader state does not match this dataset")
        self.seed, self.pos = sd["seed"], sd["pos"]
        self._ensure_perm()


def load_manifest(data_dir):
    return json.loads((Path(data_dir) / "manifest.json").read_text())


def make_loaders(data_dir, seq_len, seed=0):
    root = Path(data_dir)
    m = load_manifest(root)
    hold = set(m["holdout"])
    train = [root / s["file"] for s in m["shards"] if s["file"] not in hold]
    val = [root / f for f in m["holdout"]]
    if not train:
        raise ValueError("no training shards")
    train_loader = PackedLoader(train, seq_len, seed)
    val_loader = PackedLoader(val, seq_len, seed + 1) if val else None
    return train_loader, val_loader, m


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--seq-len", type=int, default=2048)
    args = ap.parse_args()

    from tokenizers import Tokenizer

    train, val, m = make_loaders(args.data, args.seq_len)
    tok = Tokenizer.from_file(str(Path(args.data) / m["tokenizer"]))

    print(f"vocab       {m['vocab_size']}  eos={m['eos_id']}")
    print(f"train       {train.total:,} seqs  ({train.total*args.seq_len/1e9:.3f}B tokens)")
    if val:
        print(f"holdout     {val.total:,} seqs  ({val.total*args.seq_len/1e9:.3f}B tokens)")
    print(f"holdout files {m['holdout']}")

    x, y = train.batch(2, device="cpu")
    print(f"batch       x{tuple(x.shape)} y{tuple(y.shape)} dtype={x.dtype}")
    print(f"shift ok    {torch.equal(x[0, 1:], y[0, :-1])}")
    print(f"in range    {int(x.max()) < m['vocab_size']}")
    print(f"\nsample: {tok.decode(x[0, :64].tolist())!r}")
