import torch
import os
from unsloth import FastVisionModel
from PIL import Image
import argparse

def run_inference(model_path, image_path):
    # 1. Load Model and Processor
    # Ensure we use the absolute path to avoid HF Repo ID confusion
    abs_model_path = os.path.abspath(model_path)
    print(f"Loading model and adapters from: {abs_model_path}")
    
    model, processor = FastVisionModel.from_pretrained(
        model_name = abs_model_path,
        load_in_4bit = True,
    )
    FastVisionModel.for_inference(model) # Enable 2x faster inference

    # 2. Prepare Input
    print(f"Processing image: {image_path}")
    image = Image.open(image_path).convert("RGB")
    
    # Use the same prompt and system message as training/baseline
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
    
    # Disable thinking to ensure direct output
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    
    inputs = processor(
        text=[text],
        images=[image],
        padding=True,
        return_tensors="pt"
    ).to(model.device)
    
    # 3. Generate
    print("Generating transcription...")
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=64)
        
    # Trim the input tokens from the output
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()
    
    print("\n" + "="*40)
    print(f"TRANSCRIPTION: {output_text}")
    print("="*40 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on a single image using fine-tuned Qwen-VL LoRA adapters.")
    parser.add_argument("--model_path", type=str, default="outputs/qwen3.5-iam-lora", help="Path to the saved LoRA adapters.")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the image to transcribe.")
    
    args = parser.parse_args()
    run_inference(args.model_path, args.image_path)
