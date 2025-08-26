#!/usr/bin/env python3
"""
Distribute .pkl files into N folders such that total sizes are as balanced as possible.

Hardcoded settings below:
- NUM_FOLDERS: how many subfolders to create in the destination (named 1..N)
- SOURCE_DIR:  where to read .pkl files from
- DEST_DIR:    where to create numbered subfolders and copy files to

Behavior:
- If DEST_DIR is not empty, the script warns and asks for confirmation.
  On confirmation, it deletes EVERYTHING inside DEST_DIR (recursive) and recreates it.
- Files are copied using a greedy bin-packing strategy (largest-first to the lightest folder).
- Skips the file named exactly: all_langchain_documents.pkl
"""

import heapq
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# -------------------
# Hardcoded parameters
# -------------------
NUM_FOLDERS = 8  # Change this to your desired number of folders
SOURCE_DIR = (Path(__file__).resolve().parent / "../dataset/samples/langchain-results/").resolve()
DEST_DIR = (Path(__file__).resolve().parent / "../dataset/samples/langchain-batches/").resolve()
# SOURCE_DIR = (Path(__file__).resolve().parent / "../dataset/langchain-results/").resolve()
# DEST_DIR = (Path(__file__).resolve().parent / "../dataset/langchain-batches/").resolve()
FILE_PATTERN = "*.pkl"  # Only .pkl files are processed
SKIP_FILES = {"all_langchain_documents.pkl"}  # Exact filenames to skip


def human_bytes(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(n)
    for u in units:
        if size < 1024 or u == units[-1]:
            return f"{size:.2f} {u}"
        size /= 1024


def ensure_dirs(dest_root: Path, n: int) -> List[Path]:
    dest_root.mkdir(parents=True, exist_ok=True)
    subdirs = []
    for i in range(1, n + 1):
        d = dest_root / str(i)
        d.mkdir(parents=True, exist_ok=True)
        subdirs.append(d)
    return subdirs


def is_relative_to(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except Exception:
        return False


def confirm_and_clean_destination(src: Path, dest_root: Path) -> None:
    # Safety checks to avoid destructive mistakes
    if dest_root == Path("/") or str(dest_root).strip() in ("", "."):
        print(f"Refusing to operate on suspicious destination path: {dest_root}", file=sys.stderr)
        sys.exit(2)

    if src.resolve() == dest_root.resolve():
        print("Source and destination directories must be different.", file=sys.stderr)
        sys.exit(2)

    if is_relative_to(src, dest_root) or is_relative_to(dest_root, src):
        print("Refusing to delete because source and destination overlap.", file=sys.stderr)
        print(f"Source: {src}\nDestination: {dest_root}", file=sys.stderr)
        sys.exit(2)

    if dest_root.exists():
        try:
            non_empty = any(dest_root.iterdir())
        except PermissionError as e:
            print(f"Cannot inspect destination directory: {e}", file=sys.stderr)
            sys.exit(2)
        if non_empty:
            # Count items for an informative prompt
            try:
                item_count = sum(1 for _ in dest_root.rglob("*"))
            except Exception:
                item_count = -1
            print("WARNING: Destination directory is not empty and will be fully overwritten.")
            print(f"Destination: {dest_root}")
            if item_count >= 0:
                print(f"Contents to remove: ~{item_count} item(s)")
            ans = input("Type YES to confirm deletion of ALL contents and continue: ").strip()
            if ans != "YES":
                print("Aborted by user.")
                sys.exit(1)
            try:
                shutil.rmtree(dest_root)
            except Exception as e:
                print(f"Failed to remove destination directory: {e}", file=sys.stderr)
                sys.exit(2)
    # Recreate clean destination root
    dest_root.mkdir(parents=True, exist_ok=True)


def main() -> int:
    n = int(NUM_FOLDERS)
    if n <= 0:
        print("Error: NUM_FOLDERS must be a positive integer.", file=sys.stderr)
        return 2

    src: Path = SOURCE_DIR
    dest_root: Path = DEST_DIR

    if not src.exists() or not src.is_dir():
        print(f"Error: Source directory does not exist or is not a directory: {src}", file=sys.stderr)
        return 2

    # Collect .pkl files and their sizes, skipping specified filenames
    files = [p for p in src.glob(FILE_PATTERN) if p.is_file() and p.name not in SKIP_FILES]

    skipped_present = any((src / name).exists() for name in SKIP_FILES)
    if skipped_present:
        print(f"Note: Skipping files: {', '.join(sorted(SKIP_FILES))}")

    if not files:
        print(f"No .pkl files to process in: {src} (after skipping)")
        return 0

    sized_files: List[Tuple[int, Path]] = []
    total_bytes = 0
    for p in files:
        try:
            sz = p.stat().st_size
        except Exception as e:
            print(f"Warning: skipping {p} (cannot stat: {e})", file=sys.stderr)
            continue
        sized_files.append((sz, p))
        total_bytes += sz

    if not sized_files:
        print(f"No readable .pkl files found in: {src}")
        return 0

    # Confirm and clean destination if needed
    confirm_and_clean_destination(src, dest_root)

    # Prepare destination folders
    subdirs = ensure_dirs(dest_root, n)

    # Greedy bin packing: largest-first into the lightest bin
    sized_files.sort(key=lambda t: t[0], reverse=True)

    # Min-heap of (current_total_size, folder_index starting at 1)
    heap: List[Tuple[int, int]] = [(0, i) for i in range(1, n + 1)]
    heapq.heapify(heap)

    allocations: Dict[int, List[Tuple[int, Path]]] = {i: [] for i in range(1, n + 1)}

    for sz, path in sized_files:
        current_total, idx = heapq.heappop(heap)
        allocations[idx].append((sz, path))
        heapq.heappush(heap, (current_total + sz, idx))

    # Copy files according to allocations
    copied = 0
    folder_totals: Dict[int, int] = {i: 0 for i in range(1, n + 1)}
    for i in range(1, n + 1):
        target_dir = subdirs[i - 1]
        for sz, src_path in allocations[i]:
            dest_path = target_dir / src_path.name
            try:
                shutil.copy2(src_path, dest_path)
                folder_totals[i] += sz
                copied += 1
            except Exception as e:
                print(f"Error copying {src_path} -> {dest_path}: {e}", file=sys.stderr)

    # Summary
    print(f"Copied {copied} .pkl files from {src} to {dest_root} across {n} folders.")
    print(f"Total size: {human_bytes(total_bytes)}")
    for i in range(1, n + 1):
        print(f"  Folder {i}: {human_bytes(folder_totals[i])} in {len(allocations[i])} files")

    if folder_totals:
        sizes = list(folder_totals.values())
        diff = max(sizes) - min(sizes)
        print(f"Balance spread (max - min): {human_bytes(diff)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())