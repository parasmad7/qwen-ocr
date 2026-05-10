import torch
import os
import json
from tqdm import tqdm
from jiwer import cer, wer
from PIL import Image
from unsloth import FastVisionModel
import re

def normalize_text(t):
    return re.sub(r'\s+', ' ', t).strip()

def evaluate(model_id, data_dir, num_samples, batch_size):
    metadata_path = os.path.join(data_dir, "metadata.jsonl")
    
    # Validate if model_id is a local path
    if os.path.exists(model_id):
        model_id = os.path.abspath(model_id)
    elif model_id.count("/") > 1 or model_id.startswith((".", "/", "\\")):
        raise ValueError(f"Error: The local model path '{model_id}' does not exist. Please check the path.")

    # 1. Load Model and Processor
    print(f"Loading Unsloth model for evaluation: {model_id}")
    model, processor = FastVisionModel.from_pretrained(
        model_name = model_id,
        load_in_4bit = True,
    )
    FastVisionModel.for_inference(model)
    
    # 2. Load Metadata
    if not os.path.exists(metadata_path):
        print(f"Error: {metadata_path} not found. Please run scripts/page/generate_synthetic.py first.")
        return

    data = []
    with open(metadata_path, "r") as f:
        for line in f:
            data.append(json.loads(line))
            
    # Limit to requested samples
    test_data = data[:num_samples] if num_samples > 0 else data
    
    # 3. Evaluate
    print(f"Running batched inference on {len(test_data)} images (batch_size={batch_size})...")
    
    predictions = []
    references = []
    
    system_prompt = "You are a specialized full-page OCR engine. Output ONLY the transcribed text from the image maintaining line breaks and structure, without any explanation or reasoning."
    
    # Ensure left padding for batched generation
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
                {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": "Transcribe the text in this full-page image."}]},
            ]
            
            prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            texts.append(prompt)
            images_list.append(image)
            
        inputs = processor(text=texts, images=images_list, return_tensors="pt", padding=True).to("cuda")
        
        output_ids = model.generate(
            **inputs,
            max_new_tokens=2048, # Increased to allow for full page + possible thinking
            use_cache=True,
            eos_token_id=processor.tokenizer.eos_token_id,
            stop_strings=["<|im_end|>", "<|endoftext|>", "<|im_start|>", "\nuser"],
            tokenizer=processor.tokenizer,
        )
        
        # Decode batched outputs
        input_len = inputs.input_ids.shape[1]
        for j in range(len(batch)):
            generated_ids = output_ids[j][input_len:]
            pred = processor.decode(generated_ids, skip_special_tokens=True).strip()
            
            # Remove reasoning blocks to isolate the final transcription
            if "</think>" in pred:
                pred = pred.split("</think>")[-1].strip()
            elif "<think>" in pred:
                pred = pred.split("<think>")[0].strip()
                
            # Post-process: remove any leftover template garbage (similar to line/evaluate_final.py)
            pred = pred.split("\nuser")[0].split("<|im_start|>")[0].strip()
            
            predictions.append(pred)
            references.append(batch_gts[j].strip())
            
            if i == 0 and j < 3:
                print(f"\n--- SAMPLE {j+1} ---")
                print(f"GROUND TRUTH:\n{batch_gts[j].strip()}")
                print(f"\nPREDICTION:\n{pred}")
                print("-" * 20)
            
    # 4. Calculate Metrics
    norm_references = [normalize_text(r) for r in references]
    norm_predictions = [normalize_text(p) for p in predictions]
    
    total_cer = cer(norm_references, norm_predictions)
    total_wer = wer(norm_references, norm_predictions)
    
    print("\n" + "="*40)
    print("EVALUATION RESULTS")
    print("="*40)
    print(f"Model: {model_id}")
    print(f"Samples: {len(test_data)}")
    print(f"Character Error Rate (CER): {total_cer:.4f}")
    print(f"Word Error Rate (WER): {total_wer:.4f}")
    print("="*40)
    
    # Save results
    results_path = "evaluation_results.json"
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
    parser.add_argument("--model", type=str, default="Qwen/Qwen3.5-4B", help="Model path or ID")
    parser.add_argument("--data_dir", type=str, default="data/synthetic_test", help="Directory containing metadata.jsonl and images")
    parser.add_argument("--num_samples", type=int, default=500, help="Number of samples to evaluate (0 for all)")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for inference")
    args = parser.parse_args()
    
    evaluate(args.model, args.data_dir, args.num_samples, args.batch_size)
