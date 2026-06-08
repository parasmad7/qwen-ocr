import os
import torch
import unicodedata
from unsloth import FastVisionModel
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from PIL import Image

# Configuration
MODEL_ID = "outputs/page/qwen3.5-4B-lora"
OUTPUT_DIR = "outputs/selective/qwen3.5-4B-selective"
DATA_DIR = "data/selective/train"

# Avoid memory fragmentation
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

class QwenDataCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, examples):
        texts = [example["prompt_text"] for example in examples]
        images = [Image.open(example["image_path"]).convert("RGB") for example in examples]

        batch = self.processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
            min_pixels=512*512,
            max_pixels=1344*1344,
        )

        labels = batch["input_ids"].clone()
        assistant_header = self.processor.tokenizer("<|im_start|>assistant\n", add_special_tokens=False)["input_ids"]
        
        for i in range(labels.shape[0]):
            input_ids = batch["input_ids"][i].tolist()
            for j in range(len(input_ids) - len(assistant_header) + 1):
                if input_ids[j : j + len(assistant_header)] == assistant_header:
                    labels[i, : j + len(assistant_header)] = -100
                    break
        
        if self.processor.tokenizer.pad_token_id is not None:
            labels[labels == self.processor.tokenizer.pad_token_id] = -100
            
        batch["labels"] = labels
        return batch

def train():
    if not torch.cuda.is_available():
        print("No CUDA GPU detected!")
        return

    model, processor = FastVisionModel.from_pretrained(
        model_name = MODEL_ID,
        load_in_4bit = True,
    )

    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers     = True,
        finetune_language_layers   = True,
        finetune_attention_modules = True,
        finetune_mlp_modules       = True,
        r = 32,
        lora_alpha = 64,
        lora_dropout = 0,
        bias = "none",
        random_state = 3407,
    )
    
    metadata_path = os.path.join(DATA_DIR, "metadata.jsonl")
    if not os.path.exists(metadata_path):
        print(f"Error: {metadata_path} not found. Generate data first.")
        return

    dataset = load_dataset("json", data_files=metadata_path, split="train")
    dataset = dataset.map(lambda x: {"image_path": os.path.join(DATA_DIR, x["image"])})

    def format_dataset(example):
        system_prompt = "You are a specialized Selective OCR engine. Transcribe the handwriting while ignoring any crossed-out or struck-through text."
        
        # Apply NFKC Unicode normalization to standardize characters
        normalized_text = unicodedata.normalize("NFKC", example["text"])
        
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "image", "image": example["image_path"]}, {"type": "text", "text": "Transcribe the text, ignoring strikeouts."}]},
            {"role": "assistant", "content": [{"type": "text", "text": normalized_text}]}
        ]
        example["prompt_text"] = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, enable_thinking=False)
        return example

    dataset = dataset.map(format_dataset)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=4, # Effective batch size 64
        warmup_steps=50,
        max_steps=500,
        learning_rate=1e-4,
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

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=QwenDataCollator(processor),
        max_seq_length=4096,
    )

    print("Starting Selective OCR training...")
    trainer.train()
    model.save_pretrained_merged(OUTPUT_DIR, processor, save_method = "lora")
    print("Training complete!")

if __name__ == "__main__":
    train()
