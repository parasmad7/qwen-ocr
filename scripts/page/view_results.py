import json
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default="evaluation_results.json", help="Path to results file")
    parser.add_argument("--num", type=int, default=5, help="Number of samples to view")
    args = parser.parse_args()

    try:
        with open(args.file, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find {args.file}")
        return

    print(f"Viewing {args.num} samples from {args.file}")
    print(f"Metrics - CER: {data.get('cer', 'N/A'):.4f} | WER: {data.get('wer', 'N/A'):.4f}")

    for i, (ref, pred) in enumerate(zip(data["references"], data["predictions"])):
        if i >= args.num:
            break
            
        print(f"\n{'='*60}")
        print(f"SAMPLE {i+1}")
        print(f"{'='*60}")
        
        print("\n--- GROUND TRUTH ---")
        print(ref)
        
        print("\n--- PREDICTION ---")
        print(pred)

if __name__ == "__main__":
    main()
