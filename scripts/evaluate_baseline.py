import torch
from unsloth import FastVisionModel
from datasets import load_dataset
from jiwer import cer, wer
from tqdm import tqdm
from PIL import Image
import os

def evaluate_baseline(model_id="Qwen/Qwen3.5-4B", num_samples=50):
    print(f"Loading base model for baseline evaluation: {model_id}")
    
    # 1. Load Model and Processor
    model, processor = FastVisionModel.from_pretrained(
        model_name = model_id,
        load_in_4bit = True,
    )
    FastVisionModel.for_inference(model)
    
    # 2. Load IAM-line test set
    print("Loading IAM-line dataset...")
    dataset = load_dataset("Teklia/IAM-line", split="test[:50]") # Using first 50 as a sample test
    
    predictions = []
    references = []
    
    print(f"Running zero-shot inference on {len(dataset)} samples...")
    for i, example in enumerate(tqdm(dataset)):
        image = example["image"].convert("RGB")
        ground_truth = example["text"]
        
        # Prepare input
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
        
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt"
        ).to(model.device)
        
        # Generate
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=64)
        
        # Trim and Decode
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()
        
        predictions.append(output_text)
        references.append(ground_truth)
        
        if i < 3: # Print first few for visual check
            print(f"\nSample {i}:")
            print(f"GT: {ground_truth}")
            print(f"PR: {output_text}")

    # 3. Calculate Metrics
    final_cer = cer(references, predictions)
    final_wer = wer(references, predictions)
    
    print("\n" + "="*30)
    print("BASELINE ZERO-SHOT PERFORMANCE")
    print("="*30)
    print(f"Character Error Rate (CER): {final_cer:.4f}")
    print(f"Word Error Rate (WER):      {final_wer:.4f}")
    print("="*30)
    
    # Save results
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/baseline_results.txt", "w") as f:
        f.write(f"Model: {model_id}\n")
        f.write(f"CER: {final_cer:.4f}\n")
        f.write(f"WER: {final_wer:.4f}\n")
        f.write("\nSample Predictions:\n")
        for r, p in zip(references[:10], predictions[:10]):
            f.write(f"GT: {r}\nPR: {p}\n---\n")

if __name__ == "__main__":
    evaluate_baseline()
