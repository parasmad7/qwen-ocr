import torch
import os
import json
from tqdm import tqdm
from jiwer import cer, wer
from PIL import Image
from unsloth import FastVisionModel
import re

import unicodedata

def normalize_text(text):
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def evaluate(model_id, data_dir, num_samples, batch_size):
    metadata_path = os.path.join(data_dir, "metadata.jsonl")
    
    print(f"Loading model: {model_id}")
    model, processor = FastVisionModel.from_pretrained(
        model_name = model_id,
        load_in_4bit = True,
        torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    )
    FastVisionModel.for_inference(model)
    
    if not os.path.exists(metadata_path):
        print(f"Error: {metadata_path} not found.")
        return

    data = []
    with open(metadata_path, "r") as f:
        for line in f:
            data.append(json.loads(line))
            
    test_data = data[:num_samples] if num_samples > 0 else data
    
    print(f"Evaluating {len(test_data)} samples from {data_dir}...")
    
    predictions = []
    references = []
    system_prompt = "You are a specialized Selective OCR engine. Transcribe the handwriting while ignoring any crossed-out or struck-through text."
    
    processor.tokenizer.padding_side = "left"
    
    for i in tqdm(range(0, len(test_data), batch_size)):
        batch = test_data[i:i+batch_size]
        texts = []
        images_list = []
        batch_gts = []
        
        for entry in batch:
            img_path = os.path.join(data_dir, entry["image"])
            batch_gts.append(entry["text"])
            image = Image.open(img_path).convert("RGB")
            
            messages = [
                {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
                {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": "Transcribe the text, ignoring strikeouts."}]},
            ]
            
            prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            texts.append(prompt)
            images_list.append(image)
            
        inputs = processor(
            text=texts, 
            images=images_list, 
            return_tensors="pt", 
            padding=True,
            min_pixels=256*256,
            max_pixels=1024*1024 # Optimized for speed
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
            
            # 1. Capture full output for observation
            full_pred = pred
            
            # 2. Strip thinking blocks for metrics (safety)
            if "</think>" in pred:
                pred = pred.split("</think>")[-1].strip()
            elif "<think>" in pred:
                pred = pred.split("<think>")[0].strip()
                
            # 3. Cleanup template markers
            pred = pred.split("\nuser")[0].split("<|im_start|>")[0].strip()
            
            predictions.append(pred)
            references.append(batch_gts[j].strip())
            
            # 4. Observe a few samples
            if i == 0 and j < 3:
                print(f"\n--- SAMPLE {j+1} ---")
                print(f"GROUND TRUTH:\n{batch_gts[j].strip()}")
                print(f"\nPREDICTION (Filtered):\n{pred}")
                if "<think>" in full_pred:
                     # Get the thinking content to show the user
                     thought = full_pred.split("<think>")[-1].split("</think>")[0]
                     print(f"\nTHINKING: {thought[:200]}...") 
                print("-" * 20)
            
    # 4. Calculate Metrics
    norm_references = [normalize_text(r) for r in references]
    norm_predictions = [normalize_text(p) for p in predictions]
    
    total_cer = cer(norm_references, norm_predictions)
    total_wer = wer(norm_references, norm_predictions)
    
    print("\n" + "="*40)
    print("SELECTIVE OCR EVALUATION")
    print("="*40)
    print(f"Model: {model_id}")
    print(f"CER: {total_cer:.4f}")
    print(f"WER: {total_wer:.4f}")
    print("="*40)
    
    # Outlier Analysis
    print("\n--- Outlier Analysis (CER > 10%) ---")
    per_sample_cer = [cer(r, p) for r, p in zip(norm_references, norm_predictions)]
    outliers_found = False
    for i, (r, p, c) in enumerate(zip(norm_references, norm_predictions, per_sample_cer)):
        if c > 0.1:
            outliers_found = True
            print(f"Sample {i+1} (CER: {c:.4f}) | GT: '{r[:50]}...' | Pred: '{p[:50]}...'")
    if not outliers_found:
        print("No major outliers found.")
    
    # Save results
    results_path = "evaluation_results_selective.json"
    with open(results_path, "w") as f:
        json.dump({
            "model_id": model_id,
            "cer": total_cer,
            "wer": total_wer,
            "predictions": predictions,
            "references": references
        }, f, indent=4)
    print(f"Detailed results saved to {results_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="outputs/page/qwen3.5-4B")
    parser.add_argument("--data_dir", type=str, default="data/selective/test")
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()
    
    evaluate(args.model, args.data_dir, args.num_samples, args.batch_size)
