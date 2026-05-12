# Project Learnings: Qwen3.5 Handwriting Fine-Tuning

## 🏆 Key Milestones & Results

### Selective OCR (Ongoing - 2026-05-12)
- **Phase 1 Result:** Achieved **CER: 4.11** / **WER: 4.58** on the initial 500-step fine-tune.
- **Baseline Comparison:** Prior to selective training, the model failed completely (**CER: 5.04**), proving that specialized fine-tuning is required to override the "transcribe-everything" bias.
- **Synthetic Data Edge Case:** Discovered that strikeouts must pass through the vertical center of characters (`textbbox` centering) to be recognized as deletions rather than overlines.

### Full-Page OCR (Completed - 2026-05-10)
- **Fine-Tuned Result:** Achieved **CER: 0.0028** (0.28%) and **WER: 0.0057** (0.57%) on hold-out synthetic data.
- **Baseline:** Zero-shot performance on synthetic data showed significant structure failure prior to fine-tuning.
- **Scaling:** Successfully scaled to 5,000 synthetic samples with mixed paper styles and marginalia.

### Line-Level OCR (Completed - 2026-05-06)
- **Fine-Tuned Result:** Achieved **CER: 0.045** and **WER: 0.082** on the IAM-line test set.
- **Baseline:** Zero-shot Qwen3.5-4B on IAM-line: **CER: 0.1264** / **WER: 0.2452**.
- **Insight:** Fine-tuning on domain-specific handwriting (IAM) reduced the error rate by over 60%.

---

## 💡 Technical Insights

### Selective OCR Module
- **High-Fidelity Synthetic Messiness:** Developed a synthetic generation pipeline (`generate_synthetic.py`) that replicates 5 distinct strikeout styles (double-line, wavy, cross-hatch, etc.).
- **Overcoming Transcription Bias:** Learned that the model has a strong habit of turning visual noise into garbage text. Fine-tuning with a mix of clean (30%) and messy (70%) pages is required to teach "selective skipping."
- **Inference Speed Optimization:** Discovered that full-page inference was extremely slow (45s/image) due to visual token explosion. Capping resolution at `1024x1024` pixels (`max_pixels`) provided a ~4x speedup.
- **Batch Scaling:** Successfully scaled evaluation to `batch_size=16` on 45GB VRAM by combining resolution capping with BF16 precision.

### Training & Memory Management
- **Batching (Full Page, 35GB VRAM):** `per_device_train_batch_size=4` with `gradient_accumulation_steps=16` at `max_pixels=1344x1344` maximizes GPU utilization without OOM errors.
- **Batching (Line Level):** `per_device_train_batch_size=2` with `gradient_accumulation_steps=8` provides stable line-level convergence.
- **Optimization:** Consistently used `adamw_8bit` and `gradient_checkpointing="unsloth"` to stabilize high-resolution multimodal training.

---

## 🛠️ Infrastructure & Tooling
- **Unified Evaluation Pipeline:** Consolidated disparate scripts into a single, CLI-driven `evaluate.py` for each module.
- **Robust Metric Normalization:** Implemented `NFKC` Unicode normalization and whitespace collapsing prior to CER/WER calculation.
- **Outlier Analysis:** Added automatic outlier detection (CER > 10%) to quickly identify model failure modes.
- **Deterministic Inference:** Standardized `do_sample=False` and `temperature=0` across all inference paths.

---

## 📝 Model Specifications
- **Model:** Qwen3.5-4B (Multimodal)
- **Architecture:** Gated Delta Networks + Gated Attention.
- **Release Date:** March 2, 2026.
- **Data Format:** Conversational JSONL with `<|vision_start|><|image_pad|><|vision_end|>` markers.
- **Fine-Tuning Method:** LoRA (Low-Rank Adaptation) using Unsloth.
