import os
import json
import torch
import unicodedata
from unsloth import FastVisionModel
from torch.utils.data import Dataset as TorchDataset
from transformers import Trainer, TrainingArguments
from PIL import Image

MODEL_ID = "Qwen/Qwen3.5-4B"
OUTPUT_DIR = "outputs/real/qwen3.5-4B-real"
TRAIN_JSON = "data/real/train.json"
IMAGES_DIR = "data/real/images"

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

SYSTEM_PROMPT = "You are a specialized Selective OCR engine. Transcribe the handwriting while ignoring any crossed-out or struck-through text."
USER_PROMPT = "Transcribe the text, ignoring strikeouts."


class RealOCRDataset(TorchDataset):
    def __init__(self, records, processor):
        self.records = records
        self.processor = processor

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        entry = self.records[idx]
        normalized_text = unicodedata.normalize("NFKC", entry["ground_truth"])
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [{"type": "image", "image": entry["image_path"]}, {"type": "text", "text": USER_PROMPT}]},
            {"role": "assistant", "content": [{"type": "text", "text": normalized_text}]},
        ]
        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False, enable_thinking=False
        )
        return {"prompt": prompt, "image_path": entry["image_path"]}


class QwenDataCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, examples):
        texts = [ex["prompt"] for ex in examples]
        images = [Image.open(ex["image_path"]).convert("RGB") for ex in examples]

        batch = self.processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
        )

        labels = batch["input_ids"].clone()
        assistant_header = self.processor.tokenizer(
            "<|im_start|>assistant\n", add_special_tokens=False
        )["input_ids"]

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

    free_mem, total_mem = torch.cuda.mem_get_info()
    print(f"\n{'=' * 40}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Free Memory: {free_mem / 1024**3:.2f} GiB")
    print(f"Total Memory: {total_mem / 1024**3:.2f} GiB")
    print(f"{'=' * 40}\n")

    model, processor = FastVisionModel.from_pretrained(
        model_name=MODEL_ID,
        load_in_4bit=True,
    )

    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=False,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=16,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        random_state=3407,
    )

    with open(TRAIN_JSON) as f:
        raw = json.load(f)

    records = []
    for entry in raw:
        img_path = os.path.join(IMAGES_DIR, entry["filename"])
        if os.path.exists(img_path):
            records.append({"image_path": img_path, "ground_truth": entry["text"]})

    dataset = RealOCRDataset(records, processor)
    print(f"Training samples: {len(dataset)}")

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_ratio=0.05,
        num_train_epochs=3,
        learning_rate=1e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=5,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=3407,
        save_strategy="epoch",
        save_total_limit=3,
        report_to="none",
        gradient_checkpointing="unsloth",
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=QwenDataCollator(processor),
    )

    # Sanity check
    print(f"\n{'=' * 60}")
    print("SANITY CHECK: DECODED LABELS")
    print(f"{'=' * 60}")
    example_batch = next(iter(trainer.get_train_dataloader()))
    example_labels = example_batch["labels"].clone()
    pad_id = processor.tokenizer.pad_token_id or 0
    example_labels[example_labels == -100] = pad_id
    decoded = processor.batch_decode(example_labels, skip_special_tokens=False)
    for i in range(min(2, len(decoded))):
        print(f"\n[Sample {i}]:\n{decoded[i]}")
    print(f"\n{'=' * 60}\n")

    print("Starting real-world selective OCR training...")
    trainer.train()

    print(f"Saving model to {OUTPUT_DIR}...")
    model.save_pretrained_merged(OUTPUT_DIR, processor, save_method="lora")
    print("Training complete!")


if __name__ == "__main__":
    train()
