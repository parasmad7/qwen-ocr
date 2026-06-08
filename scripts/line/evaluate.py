import torch
import os
import argparse
import re
import unicodedata
from tqdm import tqdm
from PIL import Image
from datasets import load_dataset
from jiwer import cer, wer
from unsloth import FastVisionModel

def normalize_text(text):
    """
    Standardizes text by normalizing Unicode, converting to lowercase, 
    and removing spaces around punctuation/brackets for fair comparison.
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    
    # Remove spaces before punctuation (e.g., "word ." -> "word.")
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)
    # Remove spaces inside parentheses/brackets
    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r"\s+\)", ")", text)
    
    # Collapse multiple whitespaces
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def evaluate(model_path, num_samples, output_file, batch_size):
    """
    Runs batched inference on the IAM-line dataset and calculates CER/WER metrics.
    """
    # 1. Load Model and Processor
    print(f"Loading model from: {model_path}")
    
    # Resolve local paths to absolute paths for consistency
    if os.path.exists(model_path):
        model_name = os.path.abspath(model_path)
    else:
        model_name = model_path

    model, processor = FastVisionModel.from_pretrained(
        model_name = model_name,
        load_in_4bit = True,
        torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    )
    FastVisionModel.for_inference(model)
    
    # MUST use left-padding for batched generation so that the last token 
    # of each sequence aligns, allowing the model to predict the next token correctly.
    processor.tokenizer.padding_side = "left"
    
    # 2. Load IAM-line test set
    print(f"Loading first {num_samples} samples from IAM-line test set...")
    dataset = load_dataset("Teklia/IAM-line", split=f"test[:{num_samples}]") 
    
    predictions = []
    references = []
    
    # 3. Inference Loop
    print(f"Running evaluation on {len(dataset)} samples (batch_size={batch_size})...")
    for i in tqdm(range(0, len(dataset), batch_size)):
        batch = dataset[i : i + batch_size]
        
        # Temporary containers for the current batch
        batch_prompts = []
        batch_images = []
        batch_gts = []
        
        # Prepare each sample in the batch
        for example in batch:
            image = example["image"].convert("RGB")
            ground_truth = example["text"]
            
            # Format the multimodal conversation
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
            
            # Apply the chat template to create a single string prompt
            text_prompt = processor.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True, 
                enable_thinking=False)
            
            batch_prompts.append(text_prompt)
            batch_images.append(image)
            batch_gts.append(ground_truth)
        
        # 4. Batch Processing
        # images are resized/normalized and text is tokenized with padding
        inputs = processor(
            text=batch_prompts, 
            images=batch_images, 
            return_tensors="pt",
            padding=True).to(model.device)
        
        # Perform deterministic generation
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=64, # 64 is sufficient for single-line OCR
                use_cache=True,
                do_sample=False,
                temperature=0,
                eos_token_id = processor.tokenizer.eos_token_id,
                stop_strings = ["\n", "user", "<|im_start|>", "<|im_end|>", "<think>"],
                tokenizer = processor.tokenizer,
            )
        
        # 5. Decode and Post-process
        input_len = inputs.input_ids.shape[1] # Use the length of the padded input batch
        for j in range(len(batch)):
            # Trim the prompt tokens from the generated output
            out_ids = generated_ids[j][input_len:]
            prediction = processor.decode(out_ids, skip_special_tokens=True).strip()
            
            # Aggressive cleaning: line-level OCR should not contain line breaks or "user" tokens
            prediction = prediction.split("\n")[0].split("user")[0].split("<think>")[0].strip()
            
            predictions.append(prediction)
            references.append(batch_gts[j])
            
            if i == 0 and j < 3: # Print first few for visual check
                print(f"\nSample {j}:")
                print(f"GT: '{batch_gts[j]}'")
                print(f"PR: '{prediction}'")

    # 6. Calculate Metrics
    norm_references = [normalize_text(r) for r in references]
    norm_predictions = [normalize_text(p) for p in predictions]
    
    per_sample_cer = [cer(r, p) for r, p in zip(norm_references, norm_predictions)]
    final_cer = cer(norm_references, norm_predictions)
    final_wer = wer(norm_references, norm_predictions)
    
    # 7. Report and Save
    print("\n" + "="*30)
    print("EVALUATION RESULTS")
    print("="*30)
    print(f"Model: {model_path}")
    print(f"Aggregate CER: {final_cer:.4f}")
    print(f"Aggregate WER: {final_wer:.4f}")
    print("="*30)
    
    # Outlier Analysis
    print("\n--- Outlier Analysis (CER > 10%) ---")
    outliers_found = False
    for i, (r, p, c) in enumerate(zip(norm_references, norm_predictions, per_sample_cer)):
        if c > 0.1:
            outliers_found = True
            print(f"Sample {i} (CER: {c:.4f}) | GT: '{r}' | Pred: '{p}'")
    if not outliers_found:
        print("No major outliers found.")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        f.write(f"Model: {model_path}\n")
        f.write(f"Aggregate CER: {final_cer:.4f}\n")
        f.write(f"Aggregate WER: {final_wer:.4f}\n")
        f.write("\nSample Predictions:\n")
        for r, p in zip(references[:10], predictions[:10]):
            f.write(f"GT: {r}\nPR: {p}\n---\n")
    print(f"\nFull results saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Qwen-VL model on IAM-line dataset.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3.5-4B", help="Model ID or path to adapter.")
    parser.add_argument("--num_samples", type=int, default=50, help="Number of samples to evaluate.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for inference.")
    parser.add_argument("--output", type=str, default="outputs/line/evaluation_results.txt", help="Path to save results.")
    args = parser.parse_args()
    
    evaluate(args.model, args.num_samples, args.output, args.batch_size)
