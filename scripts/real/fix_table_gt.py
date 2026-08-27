"""
Normalize division/conversion tables in GT to use consistent pipe format.

The GT has two conventions for the same visual pattern (hand-drawn division tables):
  Plain:  2 145  1       (60 entries)
  Pipes:  | 2 | 145 | 1 | (44 entries)

Both represent the same thing in the images (numbers separated by hand-drawn vertical lines).
This script converts all plain-spacing division tables to pipe format for consistency.

Only converts contiguous blocks of 3+ lines matching the division pattern (prevents
false positives on isolated lines like "2 2 2.0" in code context).

Usage:
    python scripts/real/fix_table_gt.py                    # dry run
    python scripts/real/fix_table_gt.py --apply            # write changes
"""

import json
import re
import argparse


def is_division_line(line):
    """Division table row: a single-digit divisor, then a quotient number, optionally a remainder."""
    return bool(re.match(r"^\s*2\s+\d+(\s+\d+)?\s*$", line))


def is_terminal_line(line):
    """Terminal row of a division table: just a final quotient (0 or 1), optionally with remainder."""
    stripped = line.strip()
    return bool(re.match(r"^\d+(\s+\d+)?\s*$", stripped)) and len(stripped) <= 5


def convert_division_line(line):
    parts = line.strip().split()
    return "| " + " | ".join(parts) + " |"


def convert_terminal_line(line):
    parts = line.strip().split()
    if len(parts) == 1:
        return "| | " + parts[0] + " | |"
    elif len(parts) == 2:
        return "| | " + parts[0] + " | " + parts[1] + " |"
    return "| " + " | ".join(parts) + " |"


def has_pipe_tables(text):
    return bool(re.search(r"\|\s*2\s*\|\s*\d+\s*\|", text))


def find_division_blocks(lines):
    """Find contiguous blocks of 3+ division lines. Returns list of (start, end) indices."""
    blocks = []
    i = 0
    while i < len(lines):
        if is_division_line(lines[i]):
            start = i
            while i < len(lines) and is_division_line(lines[i]):
                i += 1
            if i < len(lines) and is_terminal_line(lines[i]):
                i += 1
            end = i
            if (end - start) >= 3:
                blocks.append((start, end))
        else:
            i += 1
    return blocks


def convert_entry(text):
    if has_pipe_tables(text):
        return text, 0

    lines = text.split("\n")
    blocks = find_division_blocks(lines)

    if not blocks:
        return text, 0

    changes = 0
    for start, end in blocks:
        for i in range(start, end):
            if is_division_line(lines[i]):
                lines[i] = convert_division_line(lines[i])
                changes += 1
            elif is_terminal_line(lines[i]):
                lines[i] = convert_terminal_line(lines[i])
                changes += 1

    return "\n".join(lines), changes


def main():
    parser = argparse.ArgumentParser(description="Normalize division table GT to pipe format")
    parser.add_argument("--input", default="data/real/annotations.json")
    parser.add_argument("--apply", action="store_true", help="Write changes (default is dry run)")
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    modified_entries = []
    total_lines_changed = 0

    for entry in data:
        text = entry.get("text", "")
        converted, num_changed = convert_entry(text)
        if num_changed > 0:
            modified_entries.append({
                "file_name": entry["file_name"],
                "lines_changed": num_changed,
                "before": text,
                "after": converted,
            })
            total_lines_changed += num_changed
            if args.apply:
                entry["text"] = converted

    print(f"Entries modified: {len(modified_entries)}")
    print(f"Total lines converted: {total_lines_changed}")
    print()

    for m in modified_entries[:8]:
        print(f"=== {m['file_name'][:55]} ({m['lines_changed']} lines) ===")
        b_lines = m["before"].split("\n")
        a_lines = m["after"].split("\n")
        print("  BEFORE:")
        for l in b_lines:
            if re.match(r"^\s*2\s+\d+", l) or (l.strip() and re.match(r"^\d+$", l.strip()) and len(l.strip()) <= 3):
                print(f"    {l}")
        print("  AFTER:")
        for l in a_lines:
            if "|" in l and re.search(r"\d", l):
                print(f"    {l}")
        print()

    if args.apply:
        with open(args.input, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Written to {args.input}")
    else:
        print("DRY RUN - use --apply to write changes")


if __name__ == "__main__":
    main()
