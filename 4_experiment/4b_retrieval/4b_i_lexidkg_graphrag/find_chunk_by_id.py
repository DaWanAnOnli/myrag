#!/usr/bin/env python3
"""
find_chunk_by_id.py

Purpose:
  Given a LangChain chunk_id, print the full content (page_content) of that chunk.
  Useful sanity check to confirm your retrieval stack is pointing at the right pickles.

What it does:
  - Scans one or more directories for *.pkl files (recursively).
  - Unpickles each file (expects lists of langchain.docstore.document.Document or dict-like objects).
  - Looks for a Document whose metadata['chunk_id'] matches the input.
  - Prints the content and (optionally) the metadata and file path.

Usage:
  python find_chunk_by_id.py CHUNK_ID [-d /path/to/langchain_dir ...] [--all] [--json] [--show-meta] [-q]

Notes:
  - Requires that your environment can unpickle LangChain Document objects.
    pip install langchain tiktoken python-dotenv
  - You can provide multiple -d/--dir arguments; all will be scanned.
  - By default, stops at the first match. Use --all to print every match found.
"""

import argparse
import os
import sys
import pickle
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

# Try to make unpickling of LangChain Documents work if the package is present
try:
    from langchain.docstore.document import Document  # type: ignore
except Exception:
    Document = None  # type: ignore

# Optionally load .env if python-dotenv is available (to pick up LANGCHAIN_DIR, etc.)
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass


def _ts() -> str:
    t = time.time()
    base = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t))
    ms = int((t % 1) * 1000)
    return f"{base}.{ms:03d}"


def log(msg: str) -> None:
    sys.stderr.write(f"[{_ts()}] {msg}\n")
    sys.stderr.flush()


def norm_id(x: Any) -> str:
    return str(x).strip() if x is not None else ""


def iter_pickle_files(roots: List[Path]) -> Iterator[Path]:
    """Yield all *.pkl files under the given roots (recursively)."""
    seen: set[Path] = set()
    for root in roots:
        if not root.exists() or not root.is_dir():
            continue
        for p in root.rglob("*.pkl"):
            # Avoid following symlinks endlessly or scanning duplicates
            rp = p.resolve()
            if rp in seen:
                continue
            seen.add(rp)
            yield rp


def safe_unpickle(pkl_path: Path) -> Optional[Any]:
    try:
        with open(pkl_path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        log(f"[WARN] Could not unpickle {pkl_path.name}: {e}")
        return None


def iter_docs(obj: Any) -> Iterator[Tuple[Dict[str, Any], str]]:
    """
    Yield tuples of (metadata, page_content) from a loaded pickle object.

    Supports:
      - list[Document]
      - list[dict] with keys 'metadata' and 'page_content'
      - single Document
      - single dict with 'metadata' and 'page_content'
    """
    def as_pair(x: Any) -> Optional[Tuple[Dict[str, Any], str]]:
        # LangChain Document style
        if hasattr(x, "metadata") and hasattr(x, "page_content"):
            md = getattr(x, "metadata", None)
            pc = getattr(x, "page_content", None)
            if isinstance(md, dict) and isinstance(pc, str):
                return (md, pc)
        # Dict-like style
        if isinstance(x, dict):
            md = x.get("metadata")
            pc = x.get("page_content")
            if isinstance(md, dict) and isinstance(pc, str):
                return (md, pc)
        return None

    if isinstance(obj, list):
        for item in obj:
            pair = as_pair(item)
            if pair:
                yield pair
    else:
        pair = as_pair(obj)
        if pair:
            yield pair


def find_chunks_by_id(
    chunk_id: str,
    roots: List[Path],
    first_only: bool = True,
    quiet: bool = False
) -> List[Tuple[Path, Dict[str, Any], str]]:
    """
    Search for chunk_id in the given roots.
    Returns a list of (pkl_path, metadata, page_content) matches.
    """
    target = norm_id(chunk_id)
    matches: List[Tuple[Path, Dict[str, Any], str]] = []
    scanned_files = 0

    for pkl_path in iter_pickle_files(roots):
        scanned_files += 1
        if not quiet and scanned_files % 50 == 1:
            log(f"Scanning file {scanned_files}: {pkl_path}")

        obj = safe_unpickle(pkl_path)
        if obj is None:
            continue

        for md, pc in iter_docs(obj):
            cid = norm_id(md.get("chunk_id"))
            if cid == target:
                matches.append((pkl_path, md, pc))
                if first_only:
                    return matches

    if not quiet:
        log(f"Scan complete. Files scanned: {scanned_files}. Matches: {len(matches)}.")
    return matches


def guess_default_dirs() -> List[Path]:
    """
    Try to guess reasonable defaults for where the pickles might live.
    Priority:
      1) LANGCHAIN_DIR env
      2) Common sample/non-sample defaults used in your pipeline
      3) Current working directory
    """
    env_dir = os.getenv("LANGCHAIN_DIR")
    dirs: List[Path] = []
    if env_dir:
        d = Path(env_dir).resolve()
        if d.exists():
            dirs.append(d)

    # Sample and non-sample defaults from your provided scripts
    candidates = [
        Path("../../../dataset/samples/3_indexing/3a_langchain_results"),
        Path("../../../dataset/3_indexing/3a_langchain_results"),
        Path("../../../dataset/samples/3_indexing/3a1_langchain_batches_days"),
        Path("../../../dataset/3_indexing/3a1_langchain_batches_days"),
        Path("../../../dataset/samples/3_indexing/3b_langchain_batches"),
        Path("../../../dataset/3_indexing/3b_langchain_batches"),
        Path("."),
    ]

    for c in candidates:
        try:
            rc = c.resolve()
        except Exception:
            continue
        if rc.exists() and rc.is_dir() and rc not in dirs:
            dirs.append(rc)

    # Deduplicate while preserving order
    out: List[Path] = []
    seen: set[Path] = set()
    for d in dirs:
        if d not in seen:
            out.append(d)
            seen.add(d)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Find and print the content of a LangChain chunk by chunk_id.")
    parser.add_argument("chunk_id", type=str, help="The chunk_id to search for.")
    parser.add_argument(
        "-d", "--dir", dest="dirs", action="append", default=None,
        help="Directory to scan for *.pkl (recursively). You can pass multiple --dir. If omitted, tries LANGCHAIN_DIR env and common defaults."
    )
    parser.add_argument("--all", action="store_true", help="Show all matches (default: stop at first match).")
    parser.add_argument("--json", action="store_true", help="Output JSON instead of plain text.")
    parser.add_argument("--show-meta", action="store_true", help="Also print metadata for the match.")
    parser.add_argument("-q", "--quiet", action="store_true", help="Reduce progress logging.")
    args = parser.parse_args()

    roots: List[Path]
    if args.dirs:
        roots = [Path(d).resolve() for d in args.dirs]
    else:
        roots = guess_default_dirs()

    if not roots:
        log("No directories to scan. Provide --dir or set LANGCHAIN_DIR.")
        return 2

    if not args.quiet:
        log(f"Search roots: {', '.join(str(r) for r in roots)}")

    matches = find_chunks_by_id(
        chunk_id=args.chunk_id,
        roots=roots,
        first_only=not args.all,
        quiet=args.quiet
    )

    if not matches:
        if args.json:
            print(json.dumps({"found": False, "chunk_id": args.chunk_id}, ensure_ascii=False, indent=2))
        else:
            print(f"(No chunk found with chunk_id={args.chunk_id})")
        return 1

    # Print matches
    out_items = []
    for pkl_path, md, pc in matches:
        if args.json:
            out_items.append({
                "found": True,
                "file": str(pkl_path),
                "chunk_id": md.get("chunk_id"),
                "document_id": md.get("document_id"),
                "metadata": md if args.show_meta else None,
                "page_content": pc
            })
        else:
            print(f"=== Match in: {pkl_path} ===")
            print(f"chunk_id:   {md.get('chunk_id')}")
            print(f"document_id:{md.get('document_id')}")
            if args.show_meta:
                print("\n--- Metadata ---")
                try:
                    print(json.dumps(md, ensure_ascii=False, indent=2))
                except Exception:
                    print(str(md))
            print("\n--- Content ---")
            print(pc)
            print("\n")

    if args.json:
        print(json.dumps(out_items, ensure_ascii=False, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())