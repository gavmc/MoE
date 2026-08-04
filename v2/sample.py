"""Generate samples and measure effort conditioning (train.md sec 9).

The effort tag is the ONLY thing that varies between the three runs of each
prompt, so a difference in mean output length is attributable to the tag.
Writes a markdown report you can read without a GPU.
"""

import argparse
import json
import statistics
from pathlib import Path

import torch
from tokenizers import Tokenizer

from config import Config, build_model

PROMPTS = [
    "Summarize the following in your own words: The library closed early because "
    "the heating system failed, and staff asked visitors to return on Monday.",
    "Rewrite this sentence to be more formal: we couldn't fix the thing so we gave up.",
    "Write a short product description for a stainless steel water bottle.",
    "Explain what a lighthouse is for.",
    "List three reasons someone might learn to cook.",
    "Write a short story about a cat who finds a key.",
    "Turn this into a bulleted list: bread milk eggs coffee and apples.",
    "What is the difference between weather and climate?",
]


def build_prompt(effort_line, user):
    return f"{effort_line}\nUser: {user}\nAssistant:"


@torch.no_grad()
def generate(model, tok, text, eos_id, max_new=220, temperature=0.8, device="cuda"):
    ids = torch.tensor([tok.encode(text).ids], device=device)
    n_in = ids.shape[1]
    logits, state = model(ids, use_cache=True)
    out = []
    for _ in range(max_new):
        probs = (logits[:, -1].float() / temperature).softmax(-1)
        nxt = torch.multinomial(probs, 1)
        t = int(nxt.item())
        if t == eos_id:
            break
        out.append(t)
        logits, state = model(nxt, state, use_cache=True)
    return tok.decode(out), len(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=r"D:\ml\runs\sft\ckpt_last.pt")
    ap.add_argument("--data", default=r"D:\ml\data-sft")
    ap.add_argument("--out", default=r"D:\ml\runs\sft\samples.md")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--max-new", type=int, default=220)
    args = ap.parse_args()

    device = "cuda"
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = Config(**ck["config"])
    model = build_model(cfg, ck["vocab_size"]).to(device)
    model.load_state_dict(ck["model"])
    model.eval()

    man = json.loads((Path(args.data) / "manifest.json").read_text())
    tok = Tokenizer.from_file(str(Path(args.data) / "tokenizer.json"))
    eos_id = man["eos_id"]
    efforts = man["effort_prompts"]          # {"low": "Answer briefly.", ...}

    lines = [f"# SFT samples", "",
             f"checkpoint: `{args.ckpt}` (step {ck['step']:,})",
             f"temperature {args.temperature}, max_new {args.max_new}", ""]

    lengths = {k: [] for k in efforts}
    for i, p in enumerate(PROMPTS):
        lines.append(f"## {i+1}. {p}")
        lines.append("")
        for band in ("low", "medium", "high"):
            text, n = generate(model, tok, build_prompt(efforts[band], p), eos_id,
                               args.max_new, args.temperature, device)
            lengths[band].append(n)
            lines.append(f"**{band}** (`{efforts[band]}`) — {n} tokens")
            lines.append("")
            lines.append("> " + text.strip().replace("\n", "\n> "))
            lines.append("")

    lines += ["## Effort conditioning", "",
              "Mean output tokens per tag, same prompts, tag is the only variable:", "",
              "| effort | mean tokens | median |", "|---|---|---|"]
    for band in ("low", "medium", "high"):
        L = lengths[band]
        lines.append(f"| {band} | {statistics.mean(L):.1f} | {statistics.median(L):.0f} |")
    lo, hi = statistics.mean(lengths["low"]), statistics.mean(lengths["high"])
    lines += ["", f"high/low ratio: **{hi/max(lo,1e-9):.2f}x** "
                  f"({'modulates' if hi > lo * 1.15 else 'NO clear effect'})"]

    Path(args.out).write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[-10:]))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
