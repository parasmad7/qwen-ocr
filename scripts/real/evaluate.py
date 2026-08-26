import torch
import os
import json
import re
import unicodedata
import argparse
from tqdm import tqdm
from PIL import Image
from unsloth import FastVisionModel


SYSTEM_PROMPT = "You are an OCR assistant. Transcribe all handwritten text from this image, ignoring any crossed-out or struck-through text."
USER_PROMPT = "Transcribe the text in this image. Preserve line breaks, spatial layout, and structure. For flowcharts and diagrams, extract the text from each node in execution order using the arrow symbol to show flow. For data tables, render as clean markdown tables. Output only the transcribed text."
DEFAULT_MODEL = "outputs/real/qwen3.5-4B-real"


def edit_distance(s1, s2):
    m, n = len(s1), len(s2)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if s1[i - 1] == s2[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]


def normalize_strict(text):
    if not text:
        return ""
    return text


def normalize_content(text):
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = re.sub(r"[ \t]+([,.!?;:])", r"\1", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_data(annotations_path, images_dir):
    with open(annotations_path) as f:
        raw = json.load(f)
    data = []
    for entry in raw:
        img_path = os.path.join(images_dir, entry["file_name"])
        if os.path.exists(img_path):
            data.append({"image": img_path, "text": entry["text"], "file_name": entry["file_name"]})
    return data


def run_inference(model, processor, data, batch_size):
    processor.tokenizer.padding_side = "left"

    predictions = []

    for i in tqdm(range(0, len(data), batch_size), desc="Inference"):
        batch = data[i : i + batch_size]

        texts = []
        images_list = []

        for entry in batch:
            image = Image.open(entry["image"]).convert("RGB")
            messages = [
                {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": USER_PROMPT}]},
            ]
            prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            texts.append(prompt)
            images_list.append(image)

        inputs = processor(
            text=texts,
            images=images_list,
            return_tensors="pt",
            padding=True,
        ).to(model.device)

        output_ids = model.generate(
            **inputs,
            max_new_tokens=2048,
            use_cache=True,
            do_sample=False,
            temperature=0,
            eos_token_id=processor.tokenizer.eos_token_id,
            stop_strings=["<|im_end|>", "<|endoftext|>", "<|im_start|>", "\nuser"],
            tokenizer=processor.tokenizer,
        )

        input_len = inputs.input_ids.shape[1]
        for j in range(len(batch)):
            generated_ids = output_ids[j][input_len:]
            pred = processor.decode(generated_ids, skip_special_tokens=True).strip()

            if "</think>" in pred:
                pred = pred.split("</think>")[-1].strip()
            elif "<think>" in pred:
                pred = pred.split("<think>")[0].strip()

            pred = pred.split("\nuser")[0].split("<|im_start|>")[0].strip()
            predictions.append(pred)

    return predictions


def compute_metrics(references, predictions, normalize_fn):
    norm_refs = [normalize_fn(r) for r in references]
    norm_preds = [normalize_fn(p) for p in predictions]

    valid = [(r, p) for r, p in zip(norm_refs, norm_preds) if r]
    if not valid:
        return {"cer": float("nan"), "n": 0, "per_sample_cer": []}

    refs, preds = zip(*valid)

    total_edits = 0
    total_chars = 0
    per_sample = []
    for r, p in zip(refs, preds):
        ed = edit_distance(r, p)
        total_edits += ed
        total_chars += len(r)
        per_sample.append(ed / len(r))

    return {
        "cer": total_edits / total_chars,
        "n": len(refs),
        "per_sample_cer": per_sample,
    }


def print_results(strict, content, model_id, data):
    print(f"\n{'=' * 60}")
    print("REAL-WORLD EVALUATION RESULTS")
    print(f"{'=' * 60}")
    print(f"Model:    {model_id}")
    print(f"Samples:  {strict['n']}")
    print(f"\n  Strict CER (layout-aware):     {strict['cer']:.4f} ({strict['cer']*100:.2f}%)")
    print(f"  Content CER (whitespace-flat):  {content['cer']:.4f} ({content['cer']*100:.2f}%)")
    print(f"{'=' * 60}")

    for label, metrics in [("STRICT", strict), ("CONTENT", content)]:
        outlier_count = sum(1 for c in metrics["per_sample_cer"] if c > 0.1)
        print(f"\n--- {label} Outliers (CER > 10%): {outlier_count} / {metrics['n']} ---")
        shown = 0
        for i, sample_cer in enumerate(metrics["per_sample_cer"]):
            if sample_cer > 0.1:
                shown += 1
                gt = normalize_strict(data[i]["text"])[:60]
                print(f"  [{data[i]['file_name']}] CER={sample_cer:.4f} | GT: '{gt}...'")
                if shown >= 20:
                    remaining = sum(1 for c in metrics["per_sample_cer"][i + 1 :] if c > 0.1)
                    if remaining:
                        print(f"  ... and {remaining} more outliers")
                    break


def main():
    parser = argparse.ArgumentParser(description="Evaluate selective OCR on real handwritten data")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Model path or ID")
    parser.add_argument("--annotations", type=str, default="data/real/test.json")
    parser.add_argument("--images_dir", type=str, default="data/real/images")
    parser.add_argument("--num_samples", type=int, default=0, help="Number of samples (0 = all)")
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    model_id = args.model

    if os.path.exists(model_id):
        model_id = os.path.abspath(model_id)
    elif model_id.count("/") > 1 or model_id.startswith((".", "/", "\\")):
        raise ValueError(f"Local model path '{model_id}' does not exist.")

    data = load_data(args.annotations, args.images_dir)

    if args.num_samples > 0:
        data = data[: args.num_samples]

    print(f"Loaded {len(data)} samples from {args.annotations}")

    print(f"Loading model: {model_id}")
    model, processor = FastVisionModel.from_pretrained(
        model_name=model_id,
        load_in_4bit=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    )
    FastVisionModel.for_inference(model)

    predictions = run_inference(model, processor, data, args.batch_size)

    references = [d["text"] for d in data]
    strict = compute_metrics(references, predictions, normalize_strict)
    content = compute_metrics(references, predictions, normalize_content)

    print_results(strict, content, model_id, data)

    print("\n--- Sample Predictions ---")
    for i in range(min(3, len(data))):
        print(f"\n[{data[i]['file_name']}]")
        print(f"GT:   {data[i]['text'][:150]}")
        print(f"PRED: {predictions[i][:150]}")
        print("-" * 40)

    model_slug = os.path.basename(model_id.rstrip("/"))
    results_path = f"outputs/real/results_{model_slug}.json"
    results = {
        "model_id": model_id,
        "strict_cer": strict["cer"],
        "content_cer": content["cer"],
        "n": strict["n"],
        "samples": [
            {
                "file_name": d["file_name"],
                "reference": d["text"],
                "prediction": p,
                "strict_cer": sc,
                "content_cer": cc,
            }
            for d, p, sc, cc in zip(data, predictions, strict["per_sample_cer"], content["per_sample_cer"])
        ],
    }
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
