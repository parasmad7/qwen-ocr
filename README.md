# Qwen3.5-4B Handwriting OCR Lab

This repository contains scripts to fine-tune the **Qwen3.5-4B** multimodal model on the **IAM Handwriting Database**.

## Project Structure
- `scripts/prepare_data.py`: Downloads the IAM-line dataset from Hugging Face and formats it for Qwen3.5.
- `scripts/train.py`: Fine-tunes the model using LoRA on A100 GPUs.
- `scripts/inference.py`: Runs inference on a fine-tuned model checkpoint.
- `requirements.txt`: Python dependencies.

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
To start training on A100 GPUs using **accelerate**:
```bash
accelerate launch scripts/train.py
```
The script is configured for **LoRA** in **bf16** precision, which is optimal for A100 performance.

## Inference
After training, you can test the model:
```bash
python scripts/inference.py --model_path outputs/qwen3.5-iam-lora --image_path path/to/handwriting.png
```

## Hardware Requirements
- **GPU:** NVIDIA A100 (40GB or 80GB recommended).
- **RAM:** 32GB+.
- **Storage:** 20GB+ for model and data.
