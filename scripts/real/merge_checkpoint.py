import argparse
import torch
from unsloth import FastVisionModel


def main():
    parser = argparse.ArgumentParser(description="Merge a LoRA checkpoint into a standalone model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint directory")
    parser.add_argument("--output", type=str, required=True, help="Output directory for merged model")
    args = parser.parse_args()

    print(f"Loading checkpoint: {args.checkpoint}")
    model, processor = FastVisionModel.from_pretrained(
        model_name=args.checkpoint,
        load_in_4bit=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    )

    print(f"Saving merged model to: {args.output}")
    model.save_pretrained_merged(args.output, processor, save_method="lora")
    print("Done.")


if __name__ == "__main__":
    main()
