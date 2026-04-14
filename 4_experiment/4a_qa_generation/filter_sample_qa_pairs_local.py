#!/usr/bin/env python3
"""
Filter and sample QA pairs from qa_pairs_local.jsonl.

Steps:
  1. Keep only records with verification_choice == "A".
  2. Randomly sample N records from the filtered set.
  3. Assign a sequential question_id (0, 1, 2, …) to each sampled record.
  4. Write the result to:
       <same dir as qa_pairs_local.jsonl>/qa_pairs_local_filtered_sampled_<N>.jsonl
"""

import json
import random
from pathlib import Path

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
N = 2000  # Number of QA pairs to sample

SEED = 42  # Set to None for a non-deterministic run

# Paths (resolved relative to this script's location)
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT   = (_SCRIPT_DIR / ".." / "..").resolve()

INPUT_FILE = (
    _REPO_ROOT
    / "dataset"
    / "4_experiment"
    / "4a_qa_generation"
    / "4a_ii_qa_pairs"
    / "qa_pairs_local.jsonl"
)

OUTPUT_FILE = INPUT_FILE.parent / f"qa_pairs_local_filtered_sampled_{N}.jsonl"

# ──────────────────────────────────────────────
# Main logic
# ──────────────────────────────────────────────
def main() -> None:
    if SEED is not None:
        random.seed(SEED)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    # 1. Load and filter
    filtered: list[dict] = []
    total_lines = 0
    with INPUT_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total_lines += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[WARN] Skipping malformed JSON line {total_lines}: {e}")
                continue
            if record.get("verification_choice") == "A":
                filtered.append(record)

    print(f"Total lines read   : {total_lines}")
    print(f"Filtered (choice A): {len(filtered)}")

    # 2. Sample
    if len(filtered) < N:
        raise ValueError(
            f"Cannot sample {N} records — only {len(filtered)} records "
            f"have verification_choice == 'A'."
        )

    sampled = random.sample(filtered, N)

    # 3. Write output (prepend question_id to each record)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("w", encoding="utf-8") as out:
        for question_id, record in enumerate(sampled):
            record_with_id = {"question_id": question_id, **record}
            out.write(json.dumps(record_with_id, ensure_ascii=False) + "\n")

    print(f"Sampled            : {N}")
    print(f"Output written to  : {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
