# Selective OCR Module

This module provides a complete pipeline for training and evaluating a handwriting OCR model that can **ignore crossed-out text** (selective transcription).

## 1. Data Generation
Generate a full-page dataset with synthetic "cuttings" (strikeouts).
```bash
python scripts/selective/generate_synthetic.py --num_train 5000 --num_test 500 --strikeout_prob 0.2
```
**Strikeout Styles included:** Single line, double line, wavy scribble, cross-hatching, and dense obscuring scribble.

## 2. Baseline Evaluation
Evaluate how the base model (e.g., Qwen3.5-4B or your previously trained clean-page model) performs on the selective test set. It will likely fail to ignore strikeouts, leading to high error rates.
```bash
python scripts/selective/evaluate.py --model Qwen/Qwen3.5-4B --data_dir data/selective/test
```

## 3. Fine-Tuning
Train the model to recognize and ignore the strikeout patterns using the generated synthetic training set.
```bash
python scripts/selective/train.py
```
*Note: This script requires a GPU with ~24GB+ VRAM (A100 recommended) and uses Unsloth for efficiency.*

## 4. Final Evaluation
Evaluate the fine-tuned model on the same test set to verify the improvement in selective transcription.
```bash
python scripts/selective/evaluate.py --model outputs/selective/qwen3.5-4B --data_dir data/selective/test
```

## Directory Structure
- `generate_synthetic.py`: Synthetic page generator with word-level strikeout logic.
- `train.py`: Unsloth-based fine-tuning script.
- `evaluate.py`: CER/WER calculation script for the selective task.
- `view_synthetic.py`: Utility to visualize generated samples and their filtered ground truth.
