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
- **Batching (Full Page, 35GB VRAM):** `per_device_train_batch_size=4` with `gradient_accumulation_steps=16` (Effective Batch Size = 64) at `max_pixels=1344x1344` maximizes GPU utilization while preventing OOM errors on large context lengths (4096 tokens).
- **Batching (Line Level):** `per_device_train_batch_size=2` with `gradient_accumulation_steps=8` (Effective Batch Size = 16) provides a good balance between stability and speed.

## Milestones & Results (2026-05-10)
- **Full-Page Training Completion:** Successfully fine-tuned Qwen3.5-4B on 5,000 synthetic full-page images. The model was trained with vision layers unfrozen (`finetune_vision_layers=True`) to learn spatial layouts and merged into a standalone safetensors checkpoint.
- **VRAM Optimization:** Maxed out a 35GB VRAM GPU by capping `max_pixels` at `1344x1344` (generating exactly 2,304 image tokens) and pushing sequence length to 4096, which fits perfectly with a physical batch size of 4.
- **Dataset Serialization Fix:** Bypassed a severe Hugging Face Datasets bug where lazy-loading PIL images via `.map()` causes `AttributeError: 'dict' object has no attribute 'convert'` due to improper Arrow serialization. Solution: Store absolute string paths in the dataset and lazily open the `Image.open(path)` directly inside the DataCollator.

## Milestones & Results (2026-05-09)
- **Synthetic Layout Generation:** Built a robust data generator for full-page OCR that correctly mimics Qwen's top-to-bottom, left-to-right reading order. Implemented a proportional spacing algorithm to explicitly inject space tokens into the ground truth where large physical gaps exist (e.g., between main text and marginalia).
- **Decoupled Evaluation:** Updated `evaluate.py` to run a whitespace normalization pass before calculating CER and WER. This allows us to train the model on complex spatial layouts (heavy spacing) without whitespace mismatches artificially destroying transcription accuracy metrics.

## Milestones & Results (2026-05-06)
- **Fine-Tuning Accuracy (Checkpoint 300):** Achieved a clean **CER: 0.0236** and **WER: 0.0697** on the IAM-line test set.
- **Rambling Mitigation:** Identified that Qwen3.5 can "ramble" (repeat system/user tokens) after transcribing. Resolved this by using `stop_strings=["\n", "user"]` and post-processing.
- **Inference Stability:** Confirmed that for single-line OCR, forcing a split on the first newline/user keyword is a robust way to filter out post-prediction noise.
- **Token Slicing Logic:** Identified that for decoder-only models (like Qwen/Llama), `model.generate()` returns the entire sequence including the prompt. Manual slicing using `len(input_ids)` is required to isolate the predicted transcription from the system and user instructions.
- **Full-Page Transition:** Initiated the scaling up phase. Created a dedicated pipeline in `scripts/full_page/` that enables vision-layer fine-tuning and supports up to 2048 tokens for document-level transcription.

## Milestones & Results (2026-05-03)
- **Baseline Benchmarked:** Established a zero-shot baseline of **CER: 0.0378** and **WER: 0.2406** on the IAM-line dataset.
- **Pipeline Hardened:** Successfully configured a DDP (Distributed Data Parallel) training script for 2x A100-80GB GPUs using Unsloth.

## Key Technical Takeaways
- **Direct OCR Behavior:** For instruct-tuned Qwen models, setting `enable_thinking=False` and using a strict system prompt is critical to suppress "chain-of-thought" chatter during transcription.
- **Multi-Modal Data Collators:** Standard `SFTTrainer` defaults fail for Vision-Language models because they don't know how to batch images. A custom `DataCollator` that utilizes the model's `processor` to interleave text tokens and image tensors is mandatory.
- **Path Resolution in Unsloth:** When loading local adapters or merged models, using `os.path.abspath()` is the most reliable way to prevent the loader from confusing local directories with Hugging Face Hub IDs.
- **Efficiency:** Training a 4B model with LoRA on A100s is extremely efficient; 5 epochs were completed in under 40 minutes while achieving production-grade accuracy.
- **Vision Layer Criticality:** For full-page OCR, enabling `finetune_vision_layers=True` in LoRA is essential. The vision encoder needs to adapt to the spatial complexity and density of full pages.
- **Sequence Length Scaling:** Full-page transcription requires significantly higher `max_seq_length` (2048+) compared to line-level (512), impacting VRAM and requiring gradient accumulation.
- **VLM Spatial Bias:** Vision-Language Models process images via raster scan (top-bottom, left-right). Creating ground truth text that forces "semantic reading" (e.g., placing right-margin notes at the end of the text) actively fights this bias and leads to hallucinations and slow convergence.
- **Explicit Layout Formatting:** Pre-trained VLMs naturally collapse large white space. To teach the model to preserve layout structure, horizontal gaps must be explicitly mapped to proportional space tokens during fine-tuning.
- **Transcription vs. Layout Metrics:** Standard Levenshtein-based metrics (CER/WER) harshly penalize whitespace variations. Whitespace must be normalized prior to evaluation to accurately measure word-level reading capability independently of formatting.

## Future Extensions
- **Synthetic Data Augmentation:** Implementing random noise, shadows, and perspective transforms to handle non-white backgrounds.
- **Full Form OCR:** Transitioning from line-level to paragraph-level to leverage Qwen's vision-encoder spatial awareness.
- **Multi-GPU Scaling:** Exploring FSDP (Fully Sharded Data Parallel) for larger 72B Qwen models.
