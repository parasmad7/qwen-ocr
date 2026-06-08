import os
import torch
import unicodedata
from unsloth import FastVisionModel
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from PIL import Image

# Configuration
MODEL_ID = "Qwen/Qwen3.5-4B"
OUTPUT_DIR = "outputs/line/qwen3.5-4B-iam-lora"

# Avoid memory fragmentation
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

class QwenDataCollator:
    """
    Custom Data Collator for Qwen-VL that handles multimodal inputs (text + images)
    and masks labels for completion-only training (prompt-masking).
    """
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, examples):
        # 1. Extract messages and images from the batch
        texts = [example["prompt_text"] for example in examples]
        images = [example["image"].convert("RGB") for example in examples]

        # 2. Process all together
        # This turns text + images into input_ids, pixel_values, etc.
        batch = self.processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
        )

        # 3. Create labels (cloned from input_ids)
        labels = batch["input_ids"].clone()
        
        # 4. Define our markers
        # Primary: End of thinking block
        thought_end_marker = self.processor.tokenizer("</think>\n\n", add_special_tokens=False)["input_ids"]
        # Fallback: Standard assistant header
        assistant_header = self.processor.tokenizer("<|im_start|>assistant\n", add_special_tokens=False)["input_ids"]
        
        for i in range(labels.shape[0]):
            input_ids = batch["input_ids"][i].tolist()
            
            # Step A: Try to find the end of the thought block
            found_thought_end = False
            for j in range(len(input_ids) - len(thought_end_marker) + 1):
                if input_ids[j : j + len(thought_end_marker)] == thought_end_marker:
                    # Mask EVERYTHING up to and including the thought end marker
                    labels[i, : j + len(thought_end_marker)] = -100
                    found_thought_end = True
                    break
            
            # Step B: Fallback to assistant header if thought block isn't present
            if not found_thought_end:
                for j in range(len(input_ids) - len(assistant_header) + 1):
                    if input_ids[j : j + len(assistant_header)] == assistant_header:
                        labels[i, : j + len(assistant_header)] = -100
                        break
        
        # 5. Mask padding tokens as well
        if self.processor.tokenizer.pad_token_id is not None:
            labels[labels == self.processor.tokenizer.pad_token_id] = -100
            
        batch["labels"] = labels
        
        return batch

def train():
    # 0. GPU Diagnostics
    if torch.cuda.is_available():
        free_gpu_mem, total_gpu_mem = torch.cuda.mem_get_info()
        print(f"\n" + "="*40)
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Free Memory: {free_gpu_mem / 1024**3:.2f} GiB")
        print(f"Total Memory: {total_gpu_mem / 1024**3:.2f} GiB")
        print("="*40 + "\n")
    else:
        print("No CUDA GPU detected!")

    # 1. Load Model & Processor
    model, processor = FastVisionModel.from_pretrained(
        model_name = MODEL_ID,
        load_in_4bit = True,
        torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        trust_remote_code = True,
    )

    # 2. Add LoRA Adapters
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers     = False,
        finetune_language_layers   = True,
        finetune_attention_modules = True,
        finetune_mlp_modules       = True,
        r = 16,
        lora_alpha = 32,
        lora_dropout = 0, # Fast-path
        bias = "none",
        random_state = 3407,
    )
    
    # 3. Load Dataset
    print("Loading IAM-line dataset...")
    dataset = load_dataset("Teklia/IAM-line", split="train")

    # 4. Preparation Function
    def format_dataset(example):
        system_prompt = "You are a specialized OCR engine. Output ONLY the transcribed text from the image without any explanation or reasoning."
        
        # Apply NFKC Unicode normalization to standardize characters
        normalized_text = unicodedata.normalize("NFKC", example["text"])
        
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "image", "image": example["image"]}, {"type": "text", "text": "Transcribe the text in this image."}]},
            {"role": "assistant", "content": [{"type": "text", "text": normalized_text}]}
        ]
        # We store the applied template text in a new column
        example["prompt_text"] = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, enable_thinking=False)
        return example

    dataset = dataset.map(format_dataset)

    # 5. Training Arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=4,  # Increased to 4 for faster throughput
        gradient_accumulation_steps=16, # Effective batch size = 64
        warmup_steps=50,
        max_steps=500,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=5,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=3407,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        report_to="none",
        gradient_checkpointing="unsloth",
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
    )

    # 6. Initialize Trainer with Custom Collator
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=QwenDataCollator(processor),
        max_seq_length=512, # Reduced to 512 for speed (Safe for line OCR)
    )

    # 7. SANITY CHECK: Visualize Decoded Labels
    print("\n" + "="*60)
    print("SANITY CHECK: DECODED LABELS (Ground Truth)")
    print("Tokens marked -100 will show as the Pad Token (usually <|endoftext|>)")
    print("="*60)
    
    # Get one batch from the dataloader
    example_batch = next(iter(trainer.get_train_dataloader()))
    example_labels = example_batch["labels"].clone()
    
    # Temporarily replace -100 with pad_id so we can decode it
    pad_id = processor.tokenizer.pad_token_id if processor.tokenizer.pad_token_id is not None else 0
    example_labels[example_labels == -100] = pad_id
    
    decoded_labels = processor.batch_decode(example_labels, skip_special_tokens=False)
    
    for i in range(min(2, len(decoded_labels))):
        print(f"\n[Sample {i} Decoded GT]:\n{decoded_labels[i]}")
    print("\n" + "="*60 + "\n")

    # 8. Start Training
    print("Starting training...")
    trainer.train()
    
    # 9. Save Final Model
    print(f"Saving model to {OUTPUT_DIR}...")
    model.save_pretrained_merged(OUTPUT_DIR, processor, save_method = "lora",)
    print("Training complete!")

if __name__ == "__main__":
    train()
