# Qwen3.5-4B Handwriting OCR Lab

This repository contains scripts to fine-tune the **Qwen3.5-4B** multimodal model for both **line-level** and **full-page** handwriting recognition.

## Project Structure
The project is split into two specialized pipelines:
- `scripts/line/`: Scripts for line-level OCR training and evaluation on the IAM-line dataset.
- `scripts/page/`: Scripts for full-page OCR training, synthetic data generation, and evaluation. 

Key files in each directory:
- `generate_synthetic.py` (page only): Generates high-fidelity full-page synthetic handwriting.
- `train.py`: Unsloth-optimized QLoRA fine-tuning script.
- `inference.py`: Single-image inference.
- `evaluate.py`: Calculates Character Error Rate (CER) and Word Error Rate (WER).
- `evaluate_final.py`: Production-grade evaluation logic with strict stop-strings.

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

Test the model and calculate CER/WER:
```bash
python scripts/page/evaluate.py --model "outputs/page/qwen3.5-4B" --data_dir "data/synthetic_test" --batch_size 4
```

## Hardware Requirements
- **GPU:** NVIDIA A100 (40GB or 80GB recommended). 
- **RAM:** 32GB+.
- **Storage:** 20GB+ for model and data.
