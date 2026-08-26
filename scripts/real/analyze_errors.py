import json
import re
import unicodedata
import argparse


def normalize_strict(text):
    if not text:
        return ""
    return text


def normalize_content(text):
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = re.sub(r"[ \t]+([,.!?;:])", r"\1", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def edit_distance(s1, s2):
    m, n = len(s1), len(s2)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if s1[i - 1] == s2[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]


def categorize_error(gt, pred, sample_cer):
    categories = []

    gt_n = normalize_strict(gt)
    pred_n = normalize_strict(pred)

    if not gt_n:
        return ["empty_gt"]

    gt_len = len(gt_n)
    pred_len = len(pred_n)
    ratio = pred_len / gt_len if gt_len > 0 else float("inf")

    if gt_len < 30:
        categories.append("short_gt")

    gt_has_page = bool(re.match(r"^\d+/16", gt_n))
    pred_has_page = bool(re.match(r"^\d+/16", pred_n))
    if gt_has_page != pred_has_page:
        categories.append("page_number_mismatch")

    if gt_n.count("|") > 3 or pred_n.count("|") > 3:
        categories.append("table_content")

    arrow_chars = set("→↓↑←⇒⇐⇑⇓")
    if any(c in gt_n or c in pred_n for c in arrow_chars):
        categories.append("has_arrows_symbols")

    if ratio > 1.5:
        categories.append("model_overgenerated")
    elif ratio < 0.5:
        categories.append("model_undergenerated")

    gt_stripped = re.sub(r"^\d+/16\s*\n*", "", gt_n).strip()
    pred_stripped = re.sub(r"^\d+/16\s*\n*", "", pred_n).strip()
    if gt_stripped and pred_stripped:
        stripped_ed = edit_distance(gt_stripped, pred_stripped)
        stripped_cer = stripped_ed / len(gt_stripped)
        if stripped_cer < sample_cer * 0.5:
            categories.append("page_number_is_main_error")

    if not categories:
        categories.append("content_error")

    return categories


def compute_cer(samples, normalize_fn):
    per_sample = []
    total_ed = 0
    total_chars = 0
    for s in samples:
        gt_n = normalize_fn(s["reference"])
        pred_n = normalize_fn(s["prediction"])
        if gt_n:
            ed = edit_distance(gt_n, pred_n)
            total_ed += ed
            total_chars += len(gt_n)
            per_sample.append(ed / len(gt_n))
        else:
            per_sample.append(0.0)
    micro = total_ed / total_chars if total_chars else float("nan")
    return micro, per_sample


def print_section(label, micro_cer, per_sample, samples, cer_threshold, show_samples):
    total = len(samples)

    ranges = [
        ("Perfect (0%)", 0.0, 0.001),
        ("Near-perfect (0-1%)", 0.001, 0.01),
        ("Good (1-5%)", 0.01, 0.05),
        ("Fair (5-10%)", 0.05, 0.10),
        ("Poor (10-20%)", 0.10, 0.20),
        ("Bad (20-50%)", 0.20, 0.50),
        ("Very bad (50-100%)", 0.50, 1.00),
        ("Catastrophic (>100%)", 1.00, float("inf")),
    ]

    print(f"\n{'=' * 70}")
    print(f"{label} EVALUATION")
    print(f"{'=' * 70}")
    print(f"Micro CER:  {micro_cer:.4f} ({micro_cer * 100:.2f}%)")
    macro = sum(per_sample) / len(per_sample)
    print(f"Macro CER:  {macro:.4f} ({macro * 100:.2f}%)")

    print(f"\n{'CER Range':<25} {'Count':>6} {'Pct':>7}")
    print("-" * 40)
    for range_label, lo, hi in ranges:
        count = sum(1 for c in per_sample if lo <= c < hi)
        print(f"{range_label:<25} {count:>6} {count / total * 100:>6.1f}%")

    outliers = [
        (s, cer_val)
        for s, cer_val in zip(samples, per_sample)
        if cer_val >= cer_threshold
    ]
    outliers.sort(key=lambda x: x[1], reverse=True)

    print(f"\n--- Outlier Categories ({len(outliers)} samples with CER >= {cer_threshold * 100:.0f}%) ---")

    category_counts = {}
    category_samples = {}
    for s, cer_val in outliers:
        cats = categorize_error(s["reference"], s["prediction"], cer_val)
        for cat in cats:
            category_counts[cat] = category_counts.get(cat, 0) + 1
            if cat not in category_samples:
                category_samples[cat] = []
            category_samples[cat].append((s, cer_val))

    print(f"\n{'Category':<30} {'Count':>6} {'Pct of outliers':>16}")
    print("-" * 55)
    for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
        print(f"{cat:<30} {count:>6} {count / len(outliers) * 100:>15.1f}%")

    if show_samples > 0:
        print(f"\n--- Sample Errors ---")
        for cat in sorted(category_counts, key=lambda c: -category_counts[c]):
            print(f"\n  {cat} ({category_counts[cat]} samples):")
            for s, cer_val in category_samples[cat][:show_samples]:
                gt_n = normalize_strict(s["reference"])
                pred_n = normalize_strict(s["prediction"])
                print(f"    [{s['file_name'][:55]}] CER={cer_val:.4f}")
                print(f"    GT   ({len(gt_n):>4} chars): {gt_n[:100]}")
                print(f"    PRED ({len(pred_n):>4} chars): {pred_n[:100]}")

    perfect = sum(1 for c in per_sample if c < 0.001)
    under_5 = sum(1 for c in per_sample if c < 0.05)
    under_10 = sum(1 for c in per_sample if c < 0.10)

    print(f"\n  Perfect: {perfect}/{total}  |  <5%: {under_5}/{total}  |  <10%: {under_10}/{total}  |  Outliers: {len(outliers)}/{total}")


def main():
    parser = argparse.ArgumentParser(description="Analyze OCR evaluation errors")
    parser.add_argument("results", type=str, help="Path to results JSON from evaluate.py")
    parser.add_argument("--cer_threshold", type=float, default=0.1, help="CER threshold for outliers")
    parser.add_argument("--show_samples", type=int, default=2, help="Samples to show per category")
    args = parser.parse_args()

    with open(args.results) as f:
        data = json.load(f)

    samples = data["samples"]

    print("=" * 70)
    print("ERROR ANALYSIS REPORT")
    print("=" * 70)
    print(f"Model:      {data['model_id']}")
    print(f"Samples:    {len(samples)}")

    strict_cer, strict_per = compute_cer(samples, normalize_strict)
    content_cer, content_per = compute_cer(samples, normalize_content)

    print_section("STRICT (layout-aware)", strict_cer, strict_per, samples, args.cer_threshold, args.show_samples)
    print_section("CONTENT (whitespace-flat)", content_cer, content_per, samples, args.cer_threshold, args.show_samples)

    print(f"\n{'=' * 70}")
    print("COMPARISON")
    print(f"{'=' * 70}")
    print(f"  Strict CER:   {strict_cer:.4f} ({strict_cer * 100:.2f}%)")
    print(f"  Content CER:  {content_cer:.4f} ({content_cer * 100:.2f}%)")
    print(f"  Delta:        {(strict_cer - content_cer) * 100:.2f}pp attributed to whitespace/layout differences")


if __name__ == "__main__":
    main()
