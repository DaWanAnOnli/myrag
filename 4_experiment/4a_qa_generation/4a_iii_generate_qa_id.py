#!/usr/bin/env python3
import json
import os
import random  # NEW
from pathlib import Path

import dotenv

env_file_path = Path("../../.env")

# Load the .env file
if not env_file_path.exists():
    raise(ImportError(f"Error: .env file not found at {env_file_path}"))

dotenv.load_dotenv(env_file_path)

is_sample = os.getenv('IS_SAMPLE', '').lower()

if is_sample == "true":
    IS_SAMPLE = True
elif is_sample == "false":
    IS_SAMPLE = False
else:
    raise(ValueError(f"Wrong configuration of IS_SAMPLE in .env file: {is_sample}"))

if IS_SAMPLE:
    SOURCE_FILE = "../../dataset/samples/4_experiment/4a_qa_generation/4a_ii_qa_pairs/qa_pairs_combined.jsonl"
    DEST_FILE   = "../../dataset/samples/4_experiment/4a_qa_generation/4a_iii_qa_pairs_with_id/qa_pairs.jsonl"
else:
    SOURCE_FILE = "../../dataset/4_experiment/4a_qa_generation/4a_ii_qa_pairs/qa_pairs_combined.jsonl"
    DEST_FILE   = "../../dataset/4_experiment/4a_qa_generation/4a_iii_qa_pairs_with_id/qa_pairs.jsonl"

# -----------------------------
# Sampling configuration (edit here)
# -----------------------------
N_SAMPLE = 5000          # set to desired sample size; set to 0 or None to keep all filtered lines
RANDOM_SEED = 42        # set to None for different samples each run
if RANDOM_SEED is not None:
    random.seed(RANDOM_SEED)

def is_A(record) -> bool:
    val = str(record.get("verification_choice", "")).strip().upper()
    return val == "A"

def add_ids_after_filter(source_file: str, destination_file: str, n_sample: int = N_SAMPLE):
    os.makedirs(os.path.dirname(destination_file), exist_ok=True)

    total_lines = 0
    parsed_ok = 0
    filtered_count = 0

    # Reservoir holds tuples: (filtered_index, record)
    reservoir = []

    # First pass: filter and (optionally) sample
    with open(source_file, "r", encoding="utf-8") as infile:
        for line in infile:
            total_lines += 1
            s = line.strip()
            if not s:
                continue
            try:
                data = json.loads(s)
                parsed_ok += 1
            except Exception:
                # Skip malformed JSON lines silently (or print a warning if you prefer)
                continue

            if not is_A(data):
                continue

            # We only count index among filtered items
            filtered_index = filtered_count
            filtered_count += 1

            # Ensure we don't carry over any pre-existing id (if present)
            if "id" in data:
                del data["id"]

            # No sampling requested: keep all filtered items
            if n_sample is None or n_sample <= 0:
                reservoir.append((filtered_index, data))
                continue

            # Reservoir sampling for uniform sample of size n_sample
            k = n_sample
            if len(reservoir) < k:
                reservoir.append((filtered_index, data))
            else:
                # j is a random int in [0, filtered_count-1]
                j = random.randint(0, filtered_count - 1)
                if j < k:
                    reservoir[j] = (filtered_index, data)

    # Preserve original order of appearance among filtered items
    reservoir.sort(key=lambda t: t[0])

    kept = len(reservoir)

    # Second pass: write sampled items with fresh sequential ids
    with open(destination_file, "w", encoding="utf-8") as outfile:
        for new_id, (_, rec) in enumerate(reservoir):
            row = {"id": new_id, **rec}
            json.dump(row, outfile, ensure_ascii=False)
            outfile.write("\n")

    print(f"Read lines: {total_lines}")
    print(f"Valid JSON lines: {parsed_ok}")
    print(f"Filtered (verification_choice == 'A'): {filtered_count}")
    if n_sample is None or n_sample <= 0:
        print(f"Kept: {kept} (no sampling)")
    else:
        print(f"Sampling: requested {n_sample}, kept {kept}")
    print(f"Wrote: {destination_file}")

if __name__ == "__main__":
    add_ids_after_filter(SOURCE_FILE, DEST_FILE)