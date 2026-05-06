# Project Learnings: Qwen3.5 Handwriting Fine-Tuning

## Model Details
- **Model:** Qwen3.5-4B (Multimodal)
- **Architecture:** Gated Delta Networks + Gated Attention.
- **Release Date:** March 2, 2026.
- **Capabilities:** Strong zero-shot OCR; fine-tuning on IAM improves domain-specific handwriting recognition.

## Fine-Tuning Strategy
- **Method:** LoRA (Low-Rank Adaptation).
- **Optimization:** BF16 precision for A100 GPUs.
- **Data Format:** Conversational JSONL with `<|vision_start|><|image_pad|><|vision_end|>` markers.
- **Batching:** `per_device_train_batch_size=2` with `gradient_accumulation_steps=8` (Effective Batch Size = 16) provides a good balance between stability and speed.

## Milestones & Results (2026-05-06)
- **Fine-Tuning Accuracy (Checkpoint 300):** Achieved a clean **CER: 0.0236** and **WER: 0.0697** on the IAM-line test set.
- **Rambling Mitigation:** Identified that Qwen3.5 can "ramble" (repeat system/user tokens) after transcribing. Resolved this by using `stop_strings=["\n", "user"]` and post-processing.
- **Inference Stability:** Confirmed that for single-line OCR, forcing a split on the first newline/user keyword is a robust way to filter out post-prediction noise.

## Milestones & Results (2026-05-03)
- **Baseline Benchmarked:** Established a zero-shot baseline of **CER: 0.0378** and **WER: 0.2406** on the IAM-line dataset.
- **Pipeline Hardened:** Successfully configured a DDP (Distributed Data Parallel) training script for 2x A100-80GB GPUs using Unsloth.
- **Accuracy Breakthrough:** Fine-tuned the model for 500 steps (~37 mins), resulting in a **90% reduction in Word Error Rate**.
    - **Final CER:** 0.0058 (vs 0.0378 baseline)
    - **Final WER:** 0.0251 (vs 0.2406 baseline)

## Key Technical Takeaways
- **Direct OCR Behavior:** For instruct-tuned Qwen models, setting `enable_thinking=False` and using a strict system prompt is critical to suppress "chain-of-thought" chatter during transcription.
- **Multi-Modal Data Collators:** Standard `SFTTrainer` defaults fail for Vision-Language models because they don't know how to batch images. A custom `DataCollator` that utilizes the model's `processor` to interleave text tokens and image tensors is mandatory.
- **Path Resolution in Unsloth:** When loading local adapters or merged models, using `os.path.abspath()` is the most reliable way to prevent the loader from confusing local directories with Hugging Face Hub IDs.
- **Efficiency:** Training a 4B model with LoRA on A100s is extremely efficient; 5 epochs were completed in under 40 minutes while achieving production-grade accuracy.

## Future Extensions
- **Synthetic Data Augmentation:** Implementing random noise, shadows, and perspective transforms to handle non-white backgrounds.
- **Full Form OCR:** Transitioning from line-level to paragraph-level to leverage Qwen's vision-encoder spatial awareness.
- **Multi-GPU Scaling:** Exploring FSDP (Fully Sharded Data Parallel) for larger 72B Qwen models.
