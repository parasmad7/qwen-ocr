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

SYSTEM_PROMPT = "You are an expert OCR assistant. Extract ALL visible text from this image."
USER_PROMPT = """The black-filled regions are intentionally hidden - do not mention them or describe them.

Formatting rules:
1. If text appears in a data table (rows and columns of information), render it as a clean Markdown table (pipes and dashes) containing only the actual cell content - never copy the source image's border characters, grid lines, or box-drawing symbols into your output. Exception: division methods, long division, repeated division, conversion tables, and columnar calculations follow rules 15-20 instead.
2. Preserve all line breaks exactly as they appear in the source document.
3. Preserve indentation and spatial layout as much as possible.
4. Output ONLY the extracted text - no preamble, no explanation, no commentary.

Diagrams, flowcharts, decision trees, and process/flow charts:
5. Extract the text inside every node/box and convert the diagram into a vertical flow representation.
6. Preserve the actual execution/order of the flow shown by the arrows.
7. For sequential flow, place each step on a new line separated by: ↓

   Example:
   Start
   ↓
   Input N
   ↓
   Process
   ↓
   End

8. For decisions, preserve branches using:

   Condition ?
   Yes ↓
   Next Step
   No ↓
   Alternative Step

9. For nested decisions, continue the same structure recursively.
10. Preserve loops by returning to the relevant earlier step using the same flow notation.
11. Do not reproduce boxes, diamonds, circles, borders, connector lines, dashed lines, arrowheads, or any other visual diagram elements.
12. Extract only visible text from nodes.
13. Hidden text must be omitted completely. Do not guess, reconstruct, or use placeholders.
14. Output only the meaningful flow content in reading/execution order.

Tables, division methods, long division, repeated division, conversion tables, and columnar calculations:
15. Preserve the original row-by-row structure.
16. Extract each row as a separate line.
17. Preserve column ordering from left to right.
18. Do not convert tabular work into equations or prose.
19. Do not reproduce borders, grid lines, brackets, or separators.
20. Maintain spacing between columns where possible.

Example:

Image:

2 | 145 | 1
2 |  72 | 0
2 |  36 | 0
2 |  18 | 0
2 |   9 | 1
2 |   4 | 0
2 |   2 | 0
      1

Output:

2  145  1
2   72  0
2   36  0
2   18  0
2    9  1
2    4  0
2    2  0
     1"""


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
        img_path = os.path.join(IMAGES_DIR, entry["file_name"])
        if os.path.exists(img_path):
            records.append({"image_path": img_path, "ground_truth": entry["text"]})

    dataset = RealOCRDataset(records, processor)
    print(f"Training samples: {len(dataset)}")

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=4,
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
