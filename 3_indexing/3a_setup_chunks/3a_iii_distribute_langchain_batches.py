#!/usr/bin/env python3
"""
Distribute per-document LangChain .pkl files into N folders such that
the total CHUNKS per folder are as balanced as possible.

What's new:
- Files can now be split: if a .pkl contains many chunks (list of LangChain Documents),
  it will be sliced into multiple .pkl parts and distributed across different folders.
- Greedy balancing at CHUNK level to a per-folder target = ceil(total_chunks / N).
- Un-sliceable files (cannot unpickle or not a list) are still placed as-whole.

Behavior:
- Skips the file named exactly: all_langchain_documents.pkl (the combined file).
- For splittable files (a pickled list), we slice contiguous ranges and write new .pkl parts.
- For unsplittable files, we copy the .pkl as-is to the current lightest folder.
- If DEST_DIR is not empty, the script warns and asks for confirmation, then clears it.

Reporting per folder:
  - total size (bytes, human readable)
  - total number of files (recursively)
  - number of langchain files (.pkl) in the folder
  - total chunks contained (sum of chunks across those .pkl files)
"""

import heapq
import math
import os
import pickle
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import dotenv

# -------------------
# Hardcoded parameters
# -------------------
env_file_path = Path("../../.env")

# Load the .env file
if not env_file_path.exists():
    raise (ImportError(f"Error: .env file not found at {env_file_path}"))

dotenv.load_dotenv(env_file_path)

is_sample = os.getenv('IS_SAMPLE', '').lower()

if is_sample == "true":
    IS_SAMPLE = True
elif is_sample == "false":
    IS_SAMPLE = False
else:
    raise (ValueError(f"Wrong configuration of IS_SAMPLE in .env file: {is_sample}"))

NUM_FOLDERS = 34  # Change this to your desired number of folders
FILE_PATTERN = "*.pkl"  # Only .pkl files are processed
SKIP_FILES = {"all_langchain_documents.pkl"}  # Exact filenames to skip

# Splitting behavior
# We always allow splitting when a file is a pickled list. Parts are sized adaptively to
# fill the current lightest folder up to the per-folder target chunk count.
MAX_PART_SIZE_HINT = None  # Optional hard cap for part size in chunks (None = no extra cap)


# naive
BATCH_NO = "1"

# flash
# BATCH_NO = "1" # started running on 12/9/25 23.00 - finished
# BATCH_NO = "2" # started running on 13/9/25 23.00 - finished
# BATCH_NO = "3"

# pro
# BATCH_NO = "1" 
# BATCH_NO = "2"
# BATCH_NO = "3"
# BATCH_NO = "4"
# BATCH_NO = "5"
# BATCH_NO = "6"

if IS_SAMPLE:
    SOURCE_DIR = (Path(__file__).resolve().parent / f"../../dataset/samples/3_indexing/3a1_langchain_batches_days/{BATCH_NO}").resolve()
    DEST_DIR = (Path(__file__).resolve().parent / "../../dataset/samples/3_indexing/3b_langchain_batches/").resolve()
else:
    SOURCE_DIR = (Path(__file__).resolve().parent / f"../../dataset/3_indexing/3a1_langchain_batches_days/{BATCH_NO}").resolve()
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


def safe_base_name(path: Path) -> str:
    stem = path.stem
    return "".join(c for c in stem if c.isalnum() or c in ('-', '_')).rstrip() or "doc"


def analyze_pkl(pkl_path: Path) -> Tuple[bool, int, Optional[List]]:
    """
    Try to unpickle a .pkl file and decide if it's splittable.

    Returns:
      (splittable, chunk_count, obj_list_or_none)

    - splittable=True if object is a list (of LangChain Documents) -> sliceable
    - chunk_count = len(obj) if possible, else 1
    - obj_list_or_none = the loaded list if splittable, else None
    """
    try:
        with open(pkl_path, "rb") as f:
            obj = pickle.load(f)
        # Prefer list (typical for LangChain Document arrays)
        if isinstance(obj, list):
            return True, len(obj), obj
        # Not a list: still try len for chunk count, but mark unsplittable
        try:
            count = len(obj)  # type: ignore
            if isinstance(count, int) and count >= 0:
                return False, count, None
        except Exception:
            pass
        # Fallback
        return False, 1, None
    except Exception:
        # Could not unpickle
        return False, 1, None


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
    print("Distribution metric: chunk count (files may be split into parts)")

    if not src.exists() or not src.is_dir():
        print(f"Error: Source directory does not exist or is not a directory: {src}", file=sys.stderr)
        return 2

    # Collect .pkl files, skipping specified filenames
    files = [p for p in src.glob(FILE_PATTERN) if p.is_file() and p.name not in SKIP_FILES]

    skipped_present = any((src / name).exists() for name in SKIP_FILES)
    if skipped_present:
        print(f"Note: Skipping files: {', '.join(sorted(SKIP_FILES))}")

    if not files:
        print(f"No .pkl files to process in: {src} (after skipping)")
        return 0

    print(f"Found {len(files)} .pkl file(s) to distribute.")

    # Analyze files: which are splittable lists vs unsplittable
    splittable_entries: List[Tuple[int, int, Path, List]] = []  # (chunk_count, size_bytes, path, obj_list)
    unsplittable_entries: List[Tuple[int, int, Path]] = []      # (chunk_count, size_bytes, path)
    total_bytes_src = 0
    total_chunks = 0
    num_split_candidates = 0

    for p in files:
        try:
            sz = p.stat().st_size
        except Exception as e:
            print(f"Warning: skipping {p} (cannot stat: {e})", file=sys.stderr)
            continue

        splittable, chunk_count, obj_list = analyze_pkl(p)

        total_bytes_src += sz
        total_chunks += chunk_count

        if splittable and chunk_count > 0 and isinstance(obj_list, list):
            splittable_entries.append((chunk_count, sz, p, obj_list))
            num_split_candidates += 1
        else:
            unsplittable_entries.append((chunk_count, sz, p))

    if not splittable_entries and not unsplittable_entries:
        print(f"No readable .pkl files found in: {src}")
        return 0

    print(f"Total chunks (sum across files): {total_chunks}")
    print(f"Total bytes (before distribution): {human_bytes(total_bytes_src)}")
    print(f"Splittable files: {len(splittable_entries)} | Un-splittable files: {len(unsplittable_entries)}")

    # Confirm and clean destination if needed
    confirm_and_clean_destination(src, dest_root)

    # Prepare destination folders
    subdirs = ensure_dirs(dest_root, n)

    # Target chunks per folder for balancing
    target_per_folder = math.ceil(total_chunks / n)
    print(f"Per-folder target chunks: {target_per_folder}")

    # Min-heap of (current_total_chunks, folder_index starting at 1)
    heap: List[Tuple[int, int]] = [(0, i) for i in range(1, n + 1)]
    heapq.heapify(heap)

    # Stats
    copied = 0
    split_files_count = 0
    split_parts_written = 0

    folder_bytes: Dict[int, int] = {i: 0 for i in range(1, n + 1)}
    folder_chunks: Dict[int, int] = {i: 0 for i in range(1, n + 1)}
    folder_pkl_files: Dict[int, int] = {i: 0 for i in range(1, n + 1)}

    # 1) Place unsplittable entries first (largest-first) so we know the baseline load
    unsplittable_entries.sort(key=lambda t: (t[0], t[1]), reverse=True)
    for chunk_count, size_bytes, path in unsplittable_entries:
        curr, idx = heapq.heappop(heap)
        target_dir = subdirs[idx - 1]
        dest_path = target_dir / path.name
        try:
            shutil.copy2(path, dest_path)
            folder_bytes[idx] += size_bytes
            folder_chunks[idx] += chunk_count
            folder_pkl_files[idx] += 1
            copied += 1
        except Exception as e:
            print(f"Error copying {path} -> {dest_path}: {e}", file=sys.stderr)
        heapq.heappush(heap, (curr + chunk_count, idx))

    # 2) Distribute splittable entries by slicing into parts to fill folders up to target
    splittable_entries.sort(key=lambda t: (t[0], t[1]), reverse=True)

    print("Allocating slices from splittable files to folders...")
    for chunk_count, size_bytes, path, obj_list in splittable_entries:
        base = safe_base_name(path)
        remaining = chunk_count
        start = 0
        part_no = 1
        file_was_split = False

        while remaining > 0:
            curr, idx = heapq.heappop(heap)
            target_dir = subdirs[idx - 1]

            # Compute desired slice size for this folder
            capacity = max(0, target_per_folder - curr)
            if capacity <= 0:
                # Folder already at/over target; still put some, but not too large.
                # Heuristic: a slice up to target, but no more than remaining.
                capacity = min(remaining, target_per_folder)

            if MAX_PART_SIZE_HINT is not None:
                capacity = min(capacity, MAX_PART_SIZE_HINT)

            # Always at least 1 chunk
            take = max(1, min(remaining, capacity))

            end = start + take
            slice_obj = obj_list[start:end]

            # Name parts so you can trace origin and range
            part_name = f"{base}.part{part_no:03d}_idx{start}-{end-1}.pkl"
            dest_path = target_dir / part_name

            try:
                with open(dest_path, "wb") as f:
                    pickle.dump(slice_obj, f)
                sz_written = dest_path.stat().st_size
                folder_bytes[idx] += sz_written
                folder_chunks[idx] += take
                folder_pkl_files[idx] += 1
                split_parts_written += 1
                file_was_split = file_was_split or (take != chunk_count)
            except Exception as e:
                print(f"Error writing slice {dest_path}: {e}", file=sys.stderr)
                # If writing failed, push heap entry back unchanged and abort this file gracefully
                heapq.heappush(heap, (curr, idx))
                break

            # Update heap and counters
            heapq.heappush(heap, (curr + take, idx))
            start = end
            remaining -= take
            part_no += 1

        if file_was_split:
            split_files_count += 1

        # free memory
        del obj_list

    # Summary (preserving and extending original logging)
    total_dest_bytes = sum(folder_bytes.values())
    print(f"Copied {copied} unsplittable .pkl file(s) as-is.")
    print(f"Wrote {split_parts_written} part file(s) from {split_files_count} splittable source file(s).")
    print(f"Total size written: {human_bytes(total_dest_bytes)}")
    print(f"Total chunks distributed: {sum(folder_chunks.values())}")

    for i in range(1, n + 1):
        print(f"  Folder {i}: {human_bytes(folder_bytes[i])} in {folder_pkl_files[i]} .pkl file(s)")
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
        print(f"Balance spread (size, max - min): {human_bytes(diff)}")

    if folder_chunks:
        chunk_sizes = list(folder_chunks.values())
        chunk_diff = max(chunk_sizes) - min(chunk_sizes)
        print(f"Chunk balance spread (max - min): {chunk_diff} chunk(s)")

    return 0


if __name__ == "__main__":
    sys.exit(main())