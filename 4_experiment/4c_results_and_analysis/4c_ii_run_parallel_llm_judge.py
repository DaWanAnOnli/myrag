#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Runs multiple instances of llm_judge.py in parallel with different Google API keys,
splits the input JSONL equally (round-robin) among processes using shard flags,
and merges per-process CSVs into a single output sorted by id.

- Reads API keys GOOGLE_API_KEY_1, GOOGLE_API_KEY_2, ... from ../../.env
- Uses the latest input file matched by llm_judge.INPUT_PATTERN
- Ensures consistent CSV columns via a shared labels.json
"""

import os
import re
import sys
import glob
import json
import csv
import subprocess
from pathlib import Path
from typing import List, Dict, Any

try:
    from dotenv import load_dotenv
except ImportError:
    print("Missing dependency: python-dotenv. Install with: pip install python-dotenv", file=sys.stderr)
    sys.exit(1)

# Reuse constants/utilities from llm_judge
try:
    import llm_judge as judge
except Exception as e:
    print(f"Failed to import llm_judge: {e}", file=sys.stderr)
    sys.exit(1)

SCRIPT_DIR = Path(__file__).resolve().parent
RUN_TS = judge.now_timestamp()


def load_api_keys_from_dotenv() -> List[str]:
    env_path = (SCRIPT_DIR / "../../.env").resolve()
    load_dotenv(dotenv_path=env_path)
    # Collect GOOGLE_API_KEY_1, GOOGLE_API_KEY_2, ...
    keys = []
    pattern = re.compile(r"^GOOGLE_API_KEY_(\d+)$")
    found = []
    for k, v in os.environ.items():
        m = pattern.match(k)
        if m and v.strip():
            found.append((int(m.group(1)), v.strip()))
    found.sort(key=lambda x: x[0])
    keys = [v for _, v in found]
    # Optional fallback to single key if numbered not present
    if not keys and os.getenv("GOOGLE_API_KEY"):
        keys = [os.getenv("GOOGLE_API_KEY")]
    return keys


def find_latest_input_file() -> Path:
    base_dir = judge.INPUT_DIR
    pattern = str(base_dir / judge.INPUT_PATTERN)
    matches = glob.glob(pattern)
    if not matches:
        print(f"No input files found for pattern: {pattern}", file=sys.stderr)
        sys.exit(1)
    matches.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return Path(matches[0]).resolve()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                # match judge.read_jsonl behavior: skip malformed
                continue
    return items


def stable_union_labels(items: List[Dict[str, Any]]) -> List[str]:
    return judge.detect_all_labels(items)


def sortable_id(value: Any):
    if value is None or str(value).strip() == "":
        return (1, 2, "")  # Missing IDs go last
    s = str(value).strip()
    try:
        return (0, 0, int(s))
    except Exception:
        pass
    try:
        return (0, 1, float(s))
    except Exception:
        pass
    return (0, 2, s)  # Lexicographic fallback


def merge_csvs(part_paths: List[Path], output_path: Path) -> None:
    rows = []
    header = None
    for p in part_paths:
        if not p.exists() or p.stat().st_size == 0:
            continue
        with p.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if header is None:
                header = reader.fieldnames
            else:
                if reader.fieldnames != header:
                    # Make a superset header to reconcile minor differences
                    merged = list(dict.fromkeys((header or []) + (reader.fieldnames or [])))
                    header = merged
            for row in reader:
                rows.append(row)

    if header is None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8", newline="") as f:
            f.write("")
        return

    rows.sort(key=lambda r: sortable_id(r.get("id", "")))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header, extrasaction="ignore", quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for r in rows:
            for col in header:
                r.setdefault(col, "")
            writer.writerow(r)


def main():
    api_keys = load_api_keys_from_dotenv()
    if not api_keys:
        print("No GOOGLE_API_KEY_n entries (or GOOGLE_API_KEY) found in ../../.env", file=sys.stderr)
        sys.exit(1)

    n_proc = len(api_keys)
    print(f"Discovered {n_proc} API key(s). Launching {n_proc} parallel judge process(es).")

    input_path = find_latest_input_file()
    print(f"Using input dataset: {input_path}")

    # Compute consistent label set across the whole dataset
    all_items = read_jsonl(input_path)
    all_labels = stable_union_labels(all_items)
    print(f"Detected {len(all_labels)} label(s): {', '.join(all_labels) if all_labels else '(none)'}")

    # Write labels file to enforce consistent CSV columns
    run_dir = (judge.OUTPUT_DIR / f"parallel_run_{RUN_TS}").resolve()
    labels_path = run_dir / "labels.json"
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    with labels_path.open("w", encoding="utf-8") as f:
        json.dump(all_labels, f, ensure_ascii=False, indent=2)

    # Launch subprocesses: each process handles a round-robin shard
    processes = []
    part_csvs: List[Path] = []

    for i, key in enumerate(api_keys):
        shard_idx = i + 1  # 1-based shard index
        suffix = f"part_{shard_idx}"
        env = os.environ.copy()
        env["GOOGLE_API_KEY"] = key  # llm_judge.py will read this

        part_csv = judge.OUTPUT_DIR / f"{judge.OUTPUT_CSV_BASENAME}_{RUN_TS}__{judge.normalize_label(suffix)}.csv"
        part_csvs.append(part_csv)

        cmd = [
            sys.executable,
            str((SCRIPT_DIR / "llm_judge.py").resolve()),
            "--input-file", str(input_path),
            "--shard-index", str(shard_idx),
            "--num-shards", str(n_proc),
            "--run-ts", RUN_TS,
            "--output-suffix", suffix,
            "--labels-file", str(labels_path),
            "--force-include-id",
        ]

        print(f"[P{shard_idx}] Starting: {' '.join(cmd)}")
        p = subprocess.Popen(cmd, env=env)
        processes.append(p)

    # Wait for all
    exit_codes = []
    for i, p in enumerate(processes):
        rc = p.wait()
        exit_codes.append(rc)
        print(f"[P{i+1}] Exit code: {rc}")

    if any(rc != 0 for rc in exit_codes):
        print("Warning: one or more judge processes reported a non-zero exit. Check per-process logs.", file=sys.stderr)

    # Merge per-process CSVs into final sorted CSV
    final_csv = judge.OUTPUT_DIR / f"{judge.OUTPUT_CSV_BASENAME}_{RUN_TS}.csv"
    merge_csvs(part_csvs, final_csv)
    print(f"Merged CSV written to: {final_csv}")
    print("Parallel judging complete.")


if __name__ == "__main__":
    main()