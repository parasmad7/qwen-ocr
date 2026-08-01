# Qwen3.5-4B Handwriting OCR Lab

This repository contains scripts to fine-tune the **Qwen3.5-4B** multimodal model for **line-level**, **full-page**, **selective** (ignoring strikeouts), and **real-world** handwriting recognition.

## Project Structure
The project is split into four specialized pipelines:
- `scripts/line/`: Line-level OCR training and evaluation on the IAM-line dataset.
- `scripts/page/`: Full-page OCR training, synthetic data generation, and evaluation.
- `scripts/selective/`: Selective OCR (ignoring strikeouts) training and evaluation.
- `scripts/real/`: Real-world handwriting OCR - training, evaluation, error analysis, and checkpoint merging for handwritten C programming notebook pages.

### Key Features
- **Unified Evaluation**: A single `evaluate.py` script per pipeline that handles both base and fine-tuned models.
- **Dual CER Metrics** (real pipeline): Strict CER (raw text, layout-aware) and Content CER (normalized, whitespace-collapsed) to separate formatting errors from reading errors.
- **Error Analysis**: Automated error categorization (`analyze_errors.py`) flags page number mismatches, flowchart representation issues, hallucinations, and other failure modes.
- **Robust Metrics**: CER computed via character-level Levenshtein edit distance with NFKC Unicode normalization.
- **Outlier Analysis**: Evaluation scripts automatically identify and report samples with CER > 10% for easy debugging.
- **Deterministic Inference**: Generation is locked to `do_sample=False` and `temperature=0` for stable OCR output.

## Setup
1. Install **uv** (if not already installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. Initialize and setup project:
   ```bash
   # Automatically syncs dependencies AND installs performance kernels
   uv run setup
   ```





3. Prepare the dataset:
   ```bash
   uv run scripts/prepare_data.py
   ```

## Training
To start training on A100 GPUs:
```bash
# Line-level training
CUDA_VISIBLE_DEVICES=0 uv run scripts/line/train.py

# Full-page training
CUDA_VISIBLE_DEVICES=0 uv run scripts/page/train.py

# Selective OCR training
CUDA_VISIBLE_DEVICES=0 uv run scripts/selective/train.py
```

The scripts are heavily optimized using **Unsloth** and **QLoRA** (4-bit), utilizing `finetune_vision_layers` and custom DataCollators to mask prompts and prevent "prompt washing".

## Evaluation & Inference
**Performance (Fine-Tuned vs. Zero-Shot Baseline):**
| Task | Dataset | Baseline CER | Fine-Tuned CER | Baseline WER | Fine-Tuned WER |
|------|---------|-------------|----------------|-------------|----------------|
| **Line-Level** | IAM Handwriting | 0.0370 (3.7%) | 0.0236 (2.36%) | 0.1520 (15.2%) | 0.0697 (6.97%) |
| **Full-Page** | Synthetic | 0.0377 (3.77%) | 0.0028 (0.28%) | 0.2233 (22.33%) | 0.0057 (0.57%) |
| **Selective OCR** | Synthetic | 0.1783 (17.8%) | 0.0052 (0.52%) | 0.3460 (34.6%) | 0.0255 (2.55%) |

**Real-World Performance (284 handwritten notebook pages):**
| Model | Epoch | Strict CER | Content CER |
|-------|-------|------------|-------------|
| Base Qwen3.5-4B (no finetune) | - | 93.61% | 90.64% |
| Checkpoint-146 | 1 | 45.25% | 32.24% |
| **Checkpoint-292 (best)** | **2** | **31.81%** | **16.70%** |
| Final merged | 3 | 37.16% | 18.75% |

### Run Evaluation
```bash
# Line-level (Default to Qwen/Qwen3.5-4B base model)
uv run scripts/line/evaluate.py --num_samples 100

# Fine-tuned Line Model
uv run scripts/line/evaluate.py --model "outputs/line/checkpoint-300"

# Full-page Model
uv run scripts/page/evaluate.py --model "outputs/page/qwen3.5-4B" --batch_size 16

# Selective OCR Model
uv run scripts/selective/evaluate.py --model "outputs/selective/qwen3.5-4B" --batch_size 16

# Real-world Model
uv run scripts/real/evaluate.py --model "outputs/real/checkpoint-292" --batch_size 16

# Error analysis
uv run scripts/real/analyze_errors.py outputs/real/results_checkpoint-292.json

# Merge a LoRA checkpoint into standalone model
uv run scripts/real/merge_checkpoint.py --checkpoint outputs/real/checkpoint-292 --output outputs/real/qwen3.5-4B-real-v2
```

## Hardware Requirements
- **GPU:** NVIDIA A100 (40GB or 80GB recommended). 45GB+ VRAM allows for batch_size 16 evaluation at 1024x1024 res.
- **RAM:** 32GB+.
- **Storage:** 20GB+ for model and data.
