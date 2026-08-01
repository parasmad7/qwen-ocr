import json
import random
import argparse


def main():
    parser = argparse.ArgumentParser(description="Split annotations into train/test sets")
    parser.add_argument("--annotations", type=str, default="data/real/annotations.json")
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.annotations) as f:
        data = json.load(f)

    random.seed(args.seed)
    random.shuffle(data)

    split_idx = int(len(data) * (1 - args.test_ratio))
    train = data[:split_idx]
    test = data[split_idx:]

    train_path = args.annotations.replace("annotations.json", "train.json")
    test_path = args.annotations.replace("annotations.json", "test.json")

    with open(train_path, "w") as f:
        json.dump(train, f, indent=2)
    with open(test_path, "w") as f:
        json.dump(test, f, indent=2)

    print(f"Total: {len(data)} | Train: {len(train)} | Test: {len(test)}")
    print(f"Saved to {train_path} and {test_path}")


if __name__ == "__main__":
    main()
