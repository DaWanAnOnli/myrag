#!/usr/bin/env python3
"""
Distribute JSON files from an input directory into N balanced batches by size.

Behavior:
- Warns if the output directory exists and is non-empty; requires confirmation before deleting it.
- Creates exactly N batch folders (batch-001, batch-002, ...).
- Distributes JSON files to balance total bytes across batches using a greedy LPT algorithm.
- Copies files by default (does not modify the source). Toggle MOVE_FILES to move instead.
- Prints a summary: number of folders, files per folder, and total size per folder.

Adjustments:
- Change BATCH_COUNT below to control how many folders are created.
- Change INPUT_DIR and OUTPUT_DIR to point to your desired locations.
"""

import os
import sys
import shutil
import heapq
from pathlib import Path
from typing import List, Tuple

import dotenv

# ----------------------------
# Configuration (edit as needed)
# ----------------------------

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

# Number of output folders (batches) to create
BATCH_COUNT = 36  # <-- change this number as needed

# Input and output directories
# INPUT_DIR is relative to this script's location; OUTPUT_DIR is absolute as requested.
SCRIPT_DIR = Path(__file__).resolve().parent

if IS_SAMPLE:
    INPUT_DIR = (SCRIPT_DIR / "../../dataset/samples/2_extract_text_results").resolve()
    OUTPUT_DIR = Path("../../dataset/samples/4_experiment/4a_qa_generation/4a_i_json_batches_for_qa").resolve()
else:
    INPUT_DIR = (SCRIPT_DIR / "../../dataset/2_extract_text_results").resolve()
    OUTPUT_DIR = Path("../../dataset/4_experiment/4a_qa_generation/4a_i_json_batches_for_qa").resolve()

# Whether to move files instead of copying (default False)
MOVE_FILES = False

# Folder naming
BATCH_PREFIX = "batch-"
# ----------------------------


def abort(msg: str, code: int = 1) -> None:
    print(f"Error: {msg}", file=sys.stderr)
    sys.exit(code)


def human_size(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    size = float(n)
    for i, u in enumerate(units):
        if size < 1024 or i == len(units) - 1:
            if u == "B":
                return f"{int(size)} {u}"
            return f"{size:.2f} {u}"
        size /= 1024.0
    return f"{n} B"


def confirm_overwrite(path: Path) -> None:
    print(f"Warning: Output directory '{path}' exists and is not empty.")
    print("This action will permanently delete its contents before creating new batches.")
    resp = input("Type 'yes' to continue, or anything else to abort: ").strip().lower()
    if resp not in ("y", "yes"):
        print("Aborted by user.")
        sys.exit(1)


def collect_json_files(input_dir: Path) -> List[Path]:
    if not input_dir.exists() or not input_dir.is_dir():
        abort(f"Input directory does not exist or is not a directory: {input_dir}")
    # Collect only top-level .json files (case-insensitive)
    files = [p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() == ".json"]
    return sorted(files, key=lambda p: p.name)


def prepare_output_dir(output_dir: Path) -> None:
    if output_dir.exists():
        non_empty = any(output_dir.iterdir())
        if non_empty:
            confirm_overwrite(output_dir)
            shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def balance_files_by_size(files: List[Path], batch_count: int) -> List[Tuple[int, List[Path]]]:
    """
    Greedy LPT: sort files by descending size, then push each into the bin
    with the current smallest total size. Returns list of tuples:
      [(total_size_bytes, [paths]), ...] of length = batch_count
    """
    if batch_count <= 0:
        abort("BATCH_COUNT must be >= 1")
    # Initialize batches
    batches = [(0, []) for _ in range(batch_count)]  # list of (size, files)
    # Min-heap of (current_size, batch_index)
    heap = [(0, i) for i in range(batch_count)]
    heapq.heapify(heap)

    # Sort files by size descending
    sized_files = [ (f.stat().st_size, f) for f in files ]
    sized_files.sort(key=lambda t: t[0], reverse=True)

    for size, path in sized_files:
        cur_size, idx = heapq.heappop(heap)
        total_size, plist = batches[idx]
        total_size += size
        plist.append(path)
        batches[idx] = (total_size, plist)
        heapq.heappush(heap, (total_size, idx))

    return batches


def copy_or_move(src: Path, dst: Path, move: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        # Avoid overwriting: add numeric suffix
        stem, ext = src.stem, src.suffix
        counter = 1
        new_dst = dst
        while new_dst.exists():
            new_dst = dst.with_name(f"{stem}_{counter}{ext}")
            counter += 1
        dst = new_dst
    if move:
        shutil.move(str(src), str(dst))
    else:
        shutil.copy2(str(src), str(dst))


def compute_dir_size(path: Path) -> int:
    total = 0
    for p in path.glob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total


def main() -> None:
    # Basic sanity checks
    if OUTPUT_DIR.resolve().is_relative_to(INPUT_DIR.resolve()) or INPUT_DIR.resolve().is_relative_to(OUTPUT_DIR.resolve()):
        abort("Input and output directories must not be nested within each other.")

    files = collect_json_files(INPUT_DIR)
    total_input_size = sum(f.stat().st_size for f in files)

    # Prepare output directory (with confirmation if needed)
    prepare_output_dir(OUTPUT_DIR)

    # Create batch folders
    batch_dirs = []
    for i in range(1, BATCH_COUNT + 1):
        d = OUTPUT_DIR / f"{BATCH_PREFIX}{i:03d}"
        d.mkdir(parents=True, exist_ok=True)
        batch_dirs.append(d)

    if not files:
        print("No JSON files found. Created empty batch folders.")
        print(f"Input directory:  {INPUT_DIR}")
        print(f"Output directory: {OUTPUT_DIR}")
        print(f"Number of folders: {BATCH_COUNT}")
        for i, d in enumerate(batch_dirs, start=1):
            print(f"- {d.name}: 0 files, 0 B")
        return

    # Balance files across batches by size
    batches = balance_files_by_size(files, BATCH_COUNT)

    # Distribute files
    for idx, (_, plist) in enumerate(batches):
        out_dir = batch_dirs[idx]
        for src in plist:
            dst = out_dir / src.name
            copy_or_move(src, dst, MOVE_FILES)

    # Compute summary from actual output
    print(f"Completed distribution ({'moved' if MOVE_FILES else 'copied'} files).")
    print(f"Input directory:  {INPUT_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Total JSON files: {len(files)} ({human_size(total_input_size)})")
    print(f"Number of folders: {BATCH_COUNT}")

    for d in batch_dirs:
        files_in_d = [p for p in d.glob("*.json") if p.is_file()]
        size_in_d = compute_dir_size(d)
        print(f"- {d.name}: {len(files_in_d)} files, {human_size(size_in_d)}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nAborted by user (Ctrl+C).")
        sys.exit(130)