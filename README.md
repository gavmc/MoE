# Mixture of Experts (MoE) — GPT-2 Style Language Model

A PyTorch implementation of a **Mixture of Experts** (MoE) system built on a GPT-2-style transformer architecture. Each expert is a specialized language model trained on a distinct domain, enabling focused, high-quality generation per topic.

## Architecture

Each expert is an independent decoder-only transformer with the following specs:

| Parameter        | Value  |
|------------------|--------|
| Vocab Size       | 50,258 |
| d_model          | 512    |
| Heads            | 8      |
| FFN dim          | 2,048  |
| Layers           | 6      |
| Dropout          | 0.1    |
| Max seq length   | 512    |

### Training Strategy

- **Expert 3** (web_samples_v1) is the **base expert** — trained first with all layers unfrozen
- **Other experts** load the base expert's embedding + positional encoding weights and **freeze** them, then train only their own transformer layers
- AdamW optimizer with cosine warmup (2,000 steps) → constant LR (3e-4)
- Gradient clipping at 1.0

## Experts & Data Sources

| Expert | Domain              | CosmoPedia Collections                   | Stop Step |
|--------|---------------------|------------------------------------------|-----------|
| 0      | Math                | auto_math_text                           | 150,000   |
| 1      | Physics / Academic  | stanford                                 | 150,000   |
| 2      | Creative Writing    | stories                                  | 150,000   |
| 3      | **Base (General)**  | web_samples_v1                           | 300,000   |
| 4      | Web / General       | web_samples_v2                           | 150,000   |
| 5      | Educational / How-to| khanacademy, openstax, wikihow           | 150,000   |

## Project Structure

```
MoE/
├── main.py            # Training loop for all 6 experts
├── data_loader.py     # Dataset loading with GPT-2 tokenizer (CosmoPedia)
├── merge.py           # Simple Moving Average (SMA) checkpoint merging
├── test_experts.py    # Inference / text generation test harness
├── cosmopedia/        # CosmoPedia dataset (parquet files)
├── checkpoints/       # Per-expert checkpoint directories (step_N.pt)
├── models/            # Final merged expert weights (.pt)
├── .gitignore
└── README.md
```

## Requirements

```text
torch>=2.0
transformers
datasets
```

## Usage

### 1. Train All Experts

Runs sequentially through all 6 experts. Expert 3 (base) trains first.

```bash
python main.py
```

### 2. Merge Checkpoints

Averages the last 10 checkpoints per expert using Simple Moving Average (SMA).

```bash
python merge.py
```

### 3. Test / Generate Text

Loads merged expert models and runs domain-specific prompts through each one.

```bash
python test_experts.py
```

## Notes

- Checkpoints are saved every **10,000 steps** after warmup (expert 3 only during training)
- Final model weights are saved to `models/expert_N_final.pt` after each expert finishes
- Merged weights (SMA of last 10 checkpoints) are saved to `models/expert_N_merged.pt`
- Embedding layer sharing between experts reduces parameter redundancy and promotes consistent token representations
- CosmoPedia dataset is loaded in **streaming mode** for memory efficiency
