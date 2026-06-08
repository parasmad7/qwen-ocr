# Qwen3.5-4B Handwriting OCR Lab

This repository contains scripts to fine-tune the **Qwen3.5-4B** multimodal model for **line-level**, **full-page**, and **selective** (ignoring strikeouts) handwriting recognition.

## Project Structure
The project is split into three specialized pipelines:
- `scripts/line/`: Scripts for line-level OCR training and evaluation on the IAM-line dataset.
- `scripts/page/`: Scripts for full-page OCR training, synthetic data generation, and evaluation. 
- `scripts/selective/`: Scripts for selective OCR (ignoring strikeouts) training and evaluation.

### Key Features
- **Unified Evaluation**: A single `evaluate.py` script per pipeline that handles both base and fine-tuned models.
- **Robust Metrics**: Character and Word Error Rates (CER/WER) are calculated after NFKC Unicode normalization and whitespace collapsing.
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

Evaluation scripts identify "Outliers" (samples with CER > 10%) to help pinpoint failure modes.

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
```


```

## Hardware Requirements
- **GPU:** NVIDIA A100 (40GB or 80GB recommended). 45GB+ VRAM allows for batch_size 16 evaluation at 1024x1024 res.
- **RAM:** 32GB+.
- **Storage:** 20GB+ for model and data.
