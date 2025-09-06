#!/usr/bin/env python3
"""
Distribute per-document LangChain .pkl files into N folders such that
the total CHUNKS per folder are as balanced as possible.

Hardcoded settings below:
- NUM_FOLDERS: how many subfolders to create in the destination (named 1..N)
- SOURCE_DIR:  where to read .pkl files from
- DEST_DIR:    where to create numbered subfolders and copy files to

Behavior:
- Skips the file named exactly: all_langchain_documents.pkl (the combined file).
- Counts chunks by unpickling each .pkl and taking len(obj). This assumes the
  environment can unpickle the file (e.g., LangChain's Document class available).
  If unpickling fails or the object has no length, defaults to chunk count = 1 (warns).
- Greedy bin-packing by chunk count: largest chunk-count files first, always placed
  into the currently lightest (by chunk count) folder.
- If DEST_DIR is not empty, the script warns and asks for confirmation.
  On confirmation, it deletes EVERYTHING inside DEST_DIR (recursive) and recreates it.
- After copying, it prints for each folder:
  - total size (bytes, human readable)
  - total number of files in the folder (on disk)
  - number of langchain files (.pkl) in the folder
  - total chunks contained (sum of chunks across those .pkl files)

Note:
- This script balances by chunk count, not bytes. Sizes are still reported for visibility.
"""

import heapq
import os
import pickle
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import dotenv

# -------------------
# Hardcoded parameters
# -------------------
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

NUM_FOLDERS = 12  # Change this to your desired number of folders
FILE_PATTERN = "*.pkl"  # Only .pkl files are processed
SKIP_FILES = {"all_langchain_documents.pkl"}  # Exact filenames to skip


if IS_SAMPLE:
    SOURCE_DIR = (Path(__file__).resolve().parent / "../../dataset/samples/3_indexing/3a_langchain_results/").resolve()
    DEST_DIR = (Path(__file__).resolve().parent / "../../dataset/samples/3_indexing/3b_langchain_batches/").resolve()
else:
    SOURCE_DIR = (Path(__file__).resolve().parent / "../../dataset/3_indexing/3a_langchain_results/").resolve()
    DEST_DIR = (Path(__file__).resolve().parent / "../../dataset/3_indexing/3b_langchain_batches/").resolve()


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


def count_chunks_in_pkl(pkl_path: Path) -> int:
    """
    Manually count chunks in a LangChain .pkl by unpickling and taking len(obj).
    If unpickling fails or the object has no length, default to 1 and warn.
    """
    try:
        with open(pkl_path, "rb") as f:
            obj = pickle.load(f)
        try:
            count = len(obj)  # type: ignore
            if isinstance(count, int) and count >= 0:
                return count
        except Exception:
            pass
        print(f"Warning: {pkl_path.name} loaded but has no length; defaulting chunk count to 1", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Warning: could not unpickle {pkl_path.name} ({e}); defaulting chunk count to 1", file=sys.stderr)
        return 1


def count_all_files_in_dir(dir_path: Path) -> int:
    """Count all regular files (recursively) under dir_path."""
    try:
        return sum(1 for p in dir_path.rglob("*") if p.is_file())
    except Exception:
        return 0


def main() -> int:
    n = int(NUM_FOLDERS)
    if n <= 0:
        print("Error: NUM_FOLDERS must be a positive integer.", file=sys.stderr)
        return 2

    src: Path = SOURCE_DIR
    dest_root: Path = DEST_DIR

    print(f"Source directory: {src}")
    print(f"Destination directory: {dest_root}")
    print(f"Requested folders: {n}")
    print("Distribution metric: chunk count per .pkl (manual unpickling)")

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

    print(f"Found {len(files)} .pkl file(s) to distribute.")

    # Build list of (chunk_count, size_bytes, path) by manually counting chunks
    entries: List[Tuple[int, int, Path]] = []
    total_bytes = 0
    total_chunks = 0

    for p in files:
        try:
            sz = p.stat().st_size
        except Exception as e:
            print(f"Warning: skipping {p} (cannot stat: {e})", file=sys.stderr)
            continue

        chunks = count_chunks_in_pkl(p)

        entries.append((chunks, sz, p))
        total_bytes += sz
        total_chunks += chunks

    if not entries:
        print(f"No readable .pkl files found in: {src}")
        return 0

    print(f"Total chunks (sum across files): {total_chunks}")
    print(f"Total bytes (before distribution): {human_bytes(total_bytes)}")

    # Confirm and clean destination if needed
    confirm_and_clean_destination(src, dest_root)

    # Prepare destination folders
    subdirs = ensure_dirs(dest_root, n)

    # Greedy bin packing by chunk count: largest-first into the lightest bin
    entries.sort(key=lambda t: (t[0], t[1]), reverse=True)  # sort by chunk_count desc, then size desc

    # Min-heap of (current_total_chunks, folder_index starting at 1)
    heap: List[Tuple[int, int]] = [(0, i) for i in range(1, n + 1)]
    heapq.heapify(heap)

    # Allocations: folder_idx -> list[(chunk_count, size_bytes, path)]
    allocations: Dict[int, List[Tuple[int, int, Path]]] = {i: [] for i in range(1, n + 1)}

    print("Allocating files to folders (largest chunk-count first)...")
    for chunk_count, size_bytes, path in entries:
        current_chunks, idx = heapq.heappop(heap)
        allocations[idx].append((chunk_count, size_bytes, path))
        heapq.heappush(heap, (current_chunks + chunk_count, idx))

    # Copy files according to allocations and accumulate stats
    print("Copying files to destination folders...")
    copied = 0
    folder_bytes: Dict[int, int] = {i: 0 for i in range(1, n + 1)}
    folder_chunks: Dict[int, int] = {i: 0 for i in range(1, n + 1)}
    folder_pkl_files: Dict[int, int] = {i: 0 for i in range(1, n + 1)}

    for i in range(1, n + 1):
        target_dir = subdirs[i - 1]
        for chunk_count, size_bytes, src_path in allocations[i]:
            dest_path = target_dir / src_path.name
            try:
                shutil.copy2(src_path, dest_path)
                folder_bytes[i] += size_bytes
                folder_chunks[i] += chunk_count
                folder_pkl_files[i] += 1
                copied += 1
            except Exception as e:
                print(f"Error copying {src_path} -> {dest_path}: {e}", file=sys.stderr)

    # Summary (preserving and extending original logging)
    print(f"Copied {copied} .pkl files from {src} to {dest_root} across {n} folders.")
    print(f"Total size: {human_bytes(total_bytes)}")
    print(f"Total chunks: {total_chunks}")

    for i in range(1, n + 1):
        # Original-style summary line (size + count of files)
        print(f"  Folder {i}: {human_bytes(folder_bytes[i])} in {len(allocations[i])} files")
        # Extended details required: total files, langchain files, chunks
        folder_dir = subdirs[i - 1]
        all_files_count = count_all_files_in_dir(folder_dir)
        print(
            f"    Details: {all_files_count} file(s) on disk | "
            f"{folder_pkl_files[i]} langchain file(s) | "
            f"{folder_chunks[i]} chunk(s)"
        )

    if folder_bytes:
        sizes = list(folder_bytes.values())
        diff = max(sizes) - min(sizes)
        # Original label for size spread
        print(f"Balance spread (max - min): {human_bytes(diff)}")

    if folder_chunks:
        chunk_sizes = list(folder_chunks.values())
        chunk_diff = max(chunk_sizes) - min(chunk_sizes)
        print(f"Chunk balance spread (max - min): {chunk_diff} chunk(s)")

    return 0


if __name__ == "__main__":
    sys.exit(main())