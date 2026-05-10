import torch
from unsloth import FastVisionModel
from datasets import load_dataset
from PIL import Image

def quick_test():
    model_id = "Qwen/Qwen3.5-4B"
    print(f"--- Quick Test: Verifying {model_id} with Unsloth ---")
    
    # 1. Load Model and Processor using Unsloth
    try:
        model, processor = FastVisionModel.from_pretrained(
            model_name = model_id,
            load_in_4bit = True,
        )
        FastVisionModel.for_inference(model) # Enable native 2x faster inference
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 2. Load one sample from IAM
    print("Fetching one sample from Teklia/IAM-line test set...")
    dataset = load_dataset("Teklia/IAM-line", split="test", streaming=True)
    sample = next(iter(dataset))
    
    image = sample["image"].convert("RGB")
    image.save("test_sample.png")
    print("Saved test image to 'test_sample.png'")
    ground_truth = sample["text"]
    
    # 3. Prepare Input
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
    
    # 4. Generate
    print("Generating...")
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=64)
        
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()
    
    print("\n" + "="*40)
    print(f"GROUND TRUTH: {ground_truth}")
    print(f"PREDICTION:   {output_text}")
    print("="*40)
    print("\nTest passed! The pipeline is ready for fine-tuning.")

if __name__ == "__main__":
    quick_test()
