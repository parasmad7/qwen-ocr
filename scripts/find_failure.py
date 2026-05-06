import torch
from unsloth import FastVisionModel
from datasets import load_dataset
from PIL import Image
import os

def find_failure(model_id="Qwen/Qwen3.5-4B", num_to_check=50):
    print(f"Loading base model to find a failure case...")
    model, processor = FastVisionModel.from_pretrained(
        model_name = model_id,
        load_in_4bit = True,
    )
    FastVisionModel.for_inference(model)
    
    print("Searching IAM dataset for a sample the base model gets WRONG...")
    dataset = load_dataset("Teklia/IAM-line", split="train", streaming=True)
    
    for i, sample in enumerate(dataset):
        if i >= num_to_check: break
        
        image = sample["image"].convert("RGB")
        gt_text = sample["text"]
        
        # Inference
        messages = [
            {"role": "system", "content": [{"type": "text", "text": "You are a specialized OCR engine. Output ONLY the transcribed text."}]},
            {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": "Transcribe the text."}]}
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=64)
        
        pred = processor.batch_decode(generated_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0].strip()
        
        if pred.lower() != gt_text.lower():
            print(f"\nFOUND FAILURE at index {i}!")
            print(f"Ground Truth: {gt_text}")
            print(f"Base Model:   {pred}")
            image.save("failure_sample.png")
            print("Saved image to failure_sample.png")
            return
            
    print("Could not find a failure in the first 50 samples. Try checking more.")

if __name__ == "__main__":
    find_failure()
