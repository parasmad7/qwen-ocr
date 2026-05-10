import torch
import os
from unsloth import FastVisionModel
from datasets import load_dataset
from jiwer import cer, wer
from tqdm import tqdm
from PIL import Image
import argparse

def evaluate_final(adapter_path, num_samples=50):
    # 1. Load Model and Processor
    abs_path = os.path.abspath(adapter_path)
    print(f"Loading fine-tuned model from: {abs_path}")
    
    model, processor = FastVisionModel.from_pretrained(
        model_name = abs_path,
        load_in_4bit = True,
    )
    FastVisionModel.for_inference(model)
    
    # 2. Load IAM-line test set (using same split as baseline)
    print("Loading IAM-line dataset...")
    # We use the 'test' split to evaluate generalization
    dataset = load_dataset("Teklia/IAM-line", split="test[:50]") 
    
    predictions = []
    references = []
    
    print(f"Running evaluation on {len(dataset)} samples...")
    for i, example in enumerate(tqdm(dataset)):
        image = example["image"].convert("RGB")
        ground_truth = example["text"]
        
        # Same prompt and settings as baseline
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a specialized OCR engine. Output ONLY the transcribed text from the image without any explanation or reasoning."}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "Transcribe the text in this image."}
                ]
            }
        ]
        
        text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = processor(text=[text_prompt], images=[image], return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=128,
                use_cache=True,
                # Force the model to stop when it tries to ramble
                eos_token_id = processor.tokenizer.eos_token_id,
                stop_strings = ["\n", "user", "<|im_start|>", "<|im_end|>", "<think>"],
                tokenizer = processor.tokenizer,
            )
        
        # Decode only the new tokens
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        prediction = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]
        
        # Post-process: Take only the first line and remove any leftover garbage
        prediction = prediction.split("\n")[0].split("user")[0].split("<think>")[0].strip()
        
        predictions.append(prediction)
        references.append(ground_truth)
        
        # DEBUG: Print first 5 samples to see formatting
        if i < 5:
            print(f"\n--- Sample {i} ---")
            print(f"Ground Truth: '{ground_truth}'")
            print(f"Prediction:   '{prediction}'")
        
    # 3. Calculate Metrics
    # We calculate per-sample CER to find the outliers
    per_sample_cer = [cer(r, p) for r, p in zip(references, predictions)]
    final_cer = cer(references, predictions)
    final_wer = wer(references, predictions)
    
    print("\n" + "="*30)
    print("FINAL EVALUATION RESULTS")
    print("="*30)
    print(f"Model: {adapter_path}")
    print(f"Aggregate CER: {final_cer:.4f}")
    print(f"Aggregate WER: {final_wer:.4f}")
    print("="*30)

    # 4. Analyze Outliers (The "Why is my CER 61%?" section)
    print("\n--- Outlier Analysis (Samples with CER > 10%) ---")
    outliers_found = False
    for i, (r, p, c) in enumerate(zip(references, predictions, per_sample_cer)):
        if c > 0.1:
            outliers_found = True
            print(f"\nSample {i} (CER: {c:.4f})")
            print(f"GT:   '{r}'")
            print(f"Pred: '{p}'")
    
    if not outliers_found:
        print("No major outliers found in these samples.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_path", type=str, default="outputs/qwen3.5-iam-lora/checkpoint-300")
    parser.add_argument("--num_samples", type=str, default=50)
    args = parser.parse_args()
    
    evaluate_final(args.adapter_path, int(args.num_samples))
