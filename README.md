# Qwen3.5-4B Handwriting OCR Lab

This repository contains scripts to fine-tune the **Qwen3.5-4B** multimodal model for both **line-level** and **full-page** handwriting recognition.

## Project Structure
The project is split into two specialized pipelines:
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

2. Create environment and install dependencies:
   ```bash
   uv venv
   source .venv/bin/activate
   uv pip install -r requirements.txt
   ```

3. Prepare the dataset:
   ```bash
   python scripts/prepare_data.py
   ```

## Training
To start training on A100 GPUs (requires ~35GB VRAM for full-page):
```bash
# Line-level training
CUDA_VISIBLE_DEVICES=0 python scripts/line/train.py

# Full-page training
CUDA_VISIBLE_DEVICES=0 python scripts/page/train.py
```
The scripts are heavily optimized using **Unsloth** and **QLoRA** (4-bit), utilizing `finetune_vision_layers` and custom DataCollators to mask prompts and prevent "prompt washing".

## Evaluation & Inference
**Performance:** The full-page model achieves **CER: 0.0028** (0.28%) and **WER: 0.0057** (0.57%) on the hold-out synthetic test set.

Evaluation scripts identify "Outliers" (samples with CER > 10%) to help pinpoint failure modes.

### Run Evaluation
```bash
# Line-level (Default to Qwen/Qwen3.5-4B base model)
python scripts/line/evaluate.py --num_samples 100

# Fine-tuned Line Model
python scripts/line/evaluate.py --model "outputs/line/checkpoint-300"

# Full-page Model
python scripts/page/evaluate.py --model "outputs/page/qwen3.5-4B" --batch_size 4
```

### Run Single Inference
```bash
python scripts/line/inference.py --model "outputs/line/checkpoint-300" --image "test_sample.png"
```

## Hardware Requirements
- **GPU:** NVIDIA A100 (40GB or 80GB recommended). 
- **RAM:** 32GB+.
- **Storage:** 20GB+ for model and data.
