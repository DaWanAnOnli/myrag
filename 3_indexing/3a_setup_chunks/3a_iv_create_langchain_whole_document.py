#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Full-document LangChain generator with size cap and equal distribution splitting.

What it does:
- Loads JSON files that contain pages with 'text' and 'ocr' fields.
- For each JSON, combines each page's text and OCR (same as your pipeline),
  then joins pages with the exact PAGE_BREAK used in your current code.
- Produces one or more LangChain Document objects per source JSON:
  - If the whole combined text fits under the token cap (approx via 3.5 chars/token),
    produce exactly one Document.
  - If it exceeds, split into the minimum number of documents so each is under 750k tokens,
    distributing text roughly equally across the splits (page-aware, with optional within-page fallback).
- Saves each generated Document as its own pickle file with a unique file_id in the filename.
- Also writes a combined pickle with all Documents, a CSV summary, processing stats, and a sample JSON.
"""

import os
import json
import uuid
import math
import pickle
import csv
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

import dotenv
from langchain.docstore.document import Document

# =============================================================================
# CONFIGURATION CONSTANTS
# =============================================================================

env_file_path = Path("../../.env")

# Load the .env file
if not env_file_path.exists():
    raise ImportError(f"Error: .env file not found at {env_file_path}")

dotenv.load_dotenv(env_file_path)

is_sample = os.getenv('IS_SAMPLE', '').lower()
if is_sample == "true":
    IS_SAMPLE = True
elif is_sample == "false":
    IS_SAMPLE = False
else:
    raise ValueError(f"Wrong configuration of IS_SAMPLE in .env file: {is_sample}")

if IS_SAMPLE:
    INPUT_DIR = "../../dataset/samples/2_extract_text_results"
    OUTPUT_DIR = "../../dataset/samples/3_indexing/3f_langchain_whole_document"
else:
    INPUT_DIR = "../../dataset/2_extract_text_results"
    OUTPUT_DIR = "../../dataset/3_indexing/3f_langchain_whole_document"

# Heuristic token model
MAX_TOKENS_PER_DOC = 750_000
AVG_CHARS_PER_TOKEN = 3.5
MAX_CHARS_PER_DOC = int(MAX_TOKENS_PER_DOC * AVG_CHARS_PER_TOKEN)

# Execution options
LIMIT_FILES: Optional[int] = None   # None = all files
FORCE_OVERWRITE: bool = False       # Delete output dir contents without prompt

# Page separation
PAGE_BREAK = "\n--- PAGE BREAK ---\n"

# =============================================================================
# HELPERS
# =============================================================================

def estimate_tokens_from_chars(n_chars: int) -> int:
    """Approximate token count using 3.5 chars/token heuristic."""
    return math.ceil(n_chars / AVG_CHARS_PER_TOKEN)

def safe_stem(name: str) -> str:
    """Sanitize a name for filesystem use."""
    stem = "".join(c for c in Path(name).stem if c.isalnum() or c in ('-', '_')).rstrip()
    return stem or "document"

def check_and_prepare_output_directory(output_dir: str, force: bool = FORCE_OVERWRITE) -> bool:
    """Check if output directory exists and handle accordingly."""
    output_path = Path(output_dir)

    if output_path.exists():
        if any(output_path.iterdir()):
            if not force:
                print(f"⚠️  Warning: Output directory '{output_dir}' exists and is not empty!")
                print("Contents will be overwritten.")

                while True:
                    response = input("Do you want to continue? (y/n): ").lower().strip()
                    if response in ['y', 'yes']:
                        break
                    elif response in ['n', 'no']:
                        print("Operation cancelled.")
                        return False
                    else:
                        print("Please enter 'y' for yes or 'n' for no.")

            import shutil
            shutil.rmtree(output_path)

    output_path.mkdir(parents=True, exist_ok=True)
    return True

def combine_text_and_ocr(page: Dict[str, Any]) -> str:
    """Combine text and OCR content from a page (same behavior as your pipeline)."""
    text_content = str(page.get('text', '') or '').strip()
    ocr_content = str(page.get('ocr', '') or '').strip()

    combined_parts = []
    if text_content:
        combined_parts.append(text_content)
    if ocr_content:
        combined_parts.append(ocr_content)
    return "\n".join(combined_parts)

def build_sorted_pages(document_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return pages sorted by page_number."""
    pages = document_data.get('pages', []) or []
    return sorted(pages, key=lambda x: x['page_number'])

def build_per_page_texts(pages: List[Dict[str, Any]]) -> List[Tuple[int, str]]:
    """Return list of (page_number, page_text) using combine_text_and_ocr."""
    result: List[Tuple[int, str]] = []
    for p in pages:
        page_text = combine_text_and_ocr(p)
        result.append((p['page_number'], page_text))
    return result

def compute_full_doc_char_count(page_texts: List[Tuple[int, str]]) -> int:
    """Compute total characters including PAGE_BREAK between pages (no trailing at end)."""
    if not page_texts:
        return 0
    total = sum(len(t) for _, t in page_texts)
    total += (len(page_texts) - 1) * len(PAGE_BREAK)
    return total

def split_oversized_page_into_segments(page_text: str) -> List[str]:
    """Split a single page into <= MAX_CHARS_PER_DOC segments (rare fallback)."""
    segs: List[str] = []
    n = len(page_text)
    if n <= MAX_CHARS_PER_DOC:
        return [page_text]
    for i in range(0, n, MAX_CHARS_PER_DOC):
        segs.append(page_text[i:i + MAX_CHARS_PER_DOC])
    return segs

def build_units_within_limit(page_texts: List[Tuple[int, str]]) -> List[Dict[str, Any]]:
    """
    Transform pages into 'units' that individually fit MAX_CHARS_PER_DOC.
    - Normally one unit per page.
    - Rarely splits a single extremely large page into multiple units.
    Each unit dict includes: page_number, text, char_len, segment_index, segment_total.
    """
    units: List[Dict[str, Any]] = []
    for page_number, text in page_texts:
        segs = split_oversized_page_into_segments(text)
        if len(segs) == 1:
            units.append({
                "page_number": page_number,
                "text": segs[0],
                "char_len": len(segs[0]),
                "segment_index": 0,
                "segment_total": 1,
            })
        else:
            for idx, seg in enumerate(segs):
                units.append({
                    "page_number": page_number,
                    "text": seg,
                    "char_len": len(seg),
                    "segment_index": idx,
                    "segment_total": len(segs),
                })
    return units

def split_into_equal_parts(units: List[Dict[str, Any]], total_chars_full: int) -> List[Dict[str, Any]]:
    """
    Split units into the minimum number of parts so each part <= MAX_CHARS_PER_DOC,
    distributing text roughly equally among parts.

    Strategy:
      - Compute minimal parts N = ceil(total_chars_full / MAX_CHARS_PER_DOC).
      - For part j, target size = ceil(remaining_chars / remaining_parts).
      - Add units sequentially:
         * Add PAGE_BREAK between units whose page_number differs (no break within segments of same page).
         * Never exceed MAX_CHARS_PER_DOC.
         * Stop a part once we reach/exceed target (except allow slight underfill to keep later parts feasible).
      - On the final part, just add remaining units respecting MAX_CHARS_PER_DOC.
    """
    if not units:
        return []

    sep_len = len(PAGE_BREAK)
    N = max(1, math.ceil(total_chars_full / MAX_CHARS_PER_DOC))

    parts: List[Dict[str, Any]] = []
    idx = 0
    consumed_chars = 0

    while idx < len(units):
        remaining_units = len(units) - idx

        # Recompute remaining based on what we've actually emitted so far
        remaining_parts = max(1, N - len(parts))
        # Approximation for remaining characters: we use the original full total minus consumed so far.
        # Note: this slightly overestimates because we don't carry separators across part boundaries,
        # but it keeps us conservative.
        remaining_chars_est = max(0, total_chars_full - consumed_chars)
        target_for_this_part = math.ceil(remaining_chars_est / remaining_parts)

        current_text_parts: List[str] = []
        current_pages: List[int] = []
        current_chars = 0
        prev_page_num: Optional[int] = None

        # Fill this part
        while idx < len(units):
            u = units[idx]
            add_len = u["char_len"]
            # Add separator only if we had a previous unit and the page_number changes
            if prev_page_num is not None and u["page_number"] != prev_page_num:
                add_len += sep_len

            # If adding this unit would exceed the hard char limit, we stop (unless nothing is in this part yet)
            if current_chars > 0 and current_chars + add_len > MAX_CHARS_PER_DOC:
                break

            # If we have nothing yet and even a single unit exceeds the hard limit (shouldn't happen; units are capped),
            # we still add it to ensure progress.
            if current_chars == 0 and add_len > MAX_CHARS_PER_DOC:
                # This should be impossible due to unit capping, but we guard anyway.
                pass

            # Add the unit
            if prev_page_num is not None and u["page_number"] != prev_page_num:
                current_text_parts.append(PAGE_BREAK)
                current_chars += sep_len
            current_text_parts.append(u["text"])
            current_chars += u["char_len"]

            if not current_pages or current_pages[-1] != u["page_number"]:
                current_pages.append(u["page_number"])
            prev_page_num = u["page_number"]
            idx += 1

            # Stop if we've reached our target size and we still have parts to fill
            still_have_parts_after_this = (len(parts) + 1) < N
            if still_have_parts_after_this and current_chars >= target_for_this_part:
                break

        # Commit current part
        part_text = "".join(current_text_parts)
        part_chars = len(part_text)
        consumed_chars += part_chars

        parts.append({
            "text": part_text,
            "pages": current_pages,
            "char_count": part_chars,
            "approx_tokens": estimate_tokens_from_chars(part_chars),
        })

        # If we expected N parts but the last part would overflow the hard limit, we will naturally
        # produce more than N parts (extremely rare; generally only if a page was split into many segments).
        # That's acceptable because "minimum number" under the hard cap takes precedence.

    return parts

def create_document_metadata(
    document_data: Dict[str, Any],
    base_document_id: str,
    split_index: int,
    split_total: int,
    pages_spanned: List[int],
    approx_token_count: int,
    char_count: int,
    file_id: str
) -> Dict[str, Any]:
    """Create metadata for a full/split document, keeping compatibility with your fields."""
    doc_metadata = document_data.get('metadata', {}) or {}

    return {
        # Identifiers
        "chunk_id": str(uuid.uuid4()),           # unique per LangChain Document
        "document_id": base_document_id,         # shared across splits from the same source JSON
        "file_id": file_id,                      # unique per generated pickle file

        # Positioning
        "pages": pages_spanned,
        "chunk_index": split_index,              # 0-based index among splits
        "total_chunks": split_total,             # total number of splits for this source doc

        # Searchable fields (consistent with your pipeline)
        "uu_number": doc_metadata.get('﻿UU_Number'),
        "title": doc_metadata.get('Title'),
        "tanggal_berlaku": doc_metadata.get('Tanggal_Berlaku'),

        # Stats
        "token_count": approx_token_count,       # approximate per heuristic
        "char_count": char_count,
        "token_count_method": "chars_per_token_heuristic",
        "avg_chars_per_token": AVG_CHARS_PER_TOKEN,
        "max_tokens_per_document": MAX_TOKENS_PER_DOC,

        # Provenance
        "source_json_file": document_data.get('source_json_file') or document_data.get('filename'),
        "original_pdf_filename": document_data.get('filename'),
        "source_type": "legal_document_full",
    }

def process_single_json_file_to_full_documents(json_file_path: str) -> Tuple[List[Document], Optional[Dict[str, Any]]]:
    """Process one JSON into one or more full-document LangChain Documents."""
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            document_data = json.load(f)

        if not isinstance(document_data, dict) or 'filename' not in document_data:
            return [], None

        base_document_id = str(uuid.uuid4())
        doc_metadata = document_data.get('metadata', {}) or {}
        sorted_pages = build_sorted_pages(document_data)
        if not sorted_pages:
            return [], None

        # Compute per-page combined strings (same behavior as your pipeline)
        page_texts = build_per_page_texts(sorted_pages)

        # Character stats
        total_text_length = sum(p.get('text_length', 0) for p in sorted_pages)
        total_ocr_length = sum(p.get('ocr_length', 0) for p in sorted_pages)
        full_doc_chars = compute_full_doc_char_count(page_texts)
        full_doc_tokens_approx = estimate_tokens_from_chars(full_doc_chars)

        # Build units (splitting any single oversized page into segments if needed)
        units = build_units_within_limit(page_texts)

        # Split into minimal number of parts under char cap, distributing roughly equally
        parts = split_into_equal_parts(units, total_chars_full=full_doc_chars)
        split_total = max(1, len(parts))

        # Build LangChain Documents
        documents: List[Document] = []
        for split_index, part in enumerate(parts):
            file_id = str(uuid.uuid4())
            metadata = create_document_metadata(
                document_data=document_data,
                base_document_id=base_document_id,
                split_index=split_index,
                split_total=split_total,
                pages_spanned=part["pages"],
                approx_token_count=part["approx_tokens"],
                char_count=part["char_count"],
                file_id=file_id
            )
            documents.append(Document(page_content=part["text"], metadata=metadata))

        # Build doc summary for CSV and stats
        doc_summary = {
            "source_json_file": os.path.basename(json_file_path),
            "original_pdf_filename": document_data.get('filename'),
            "title": doc_metadata.get('Title', 'Unknown'),
            "uu_number": doc_metadata.get('﻿UU_Number', 'Unknown'),
            "subject": doc_metadata.get('Subject', 'N/A'),
            "tanggal_penetapan": doc_metadata.get('Tanggal_Penetapan', 'N/A'),
            "tanggal_pengundangan": doc_metadata.get('Tanggal_Pengundangan', 'N/A'),
            "tanggal_berlaku": doc_metadata.get('Tanggal_Berlaku', 'N/A'),
            "total_pages_in_pdf": document_data.get('total_pages_in_pdf', 0),
            "pages_processed": document_data.get('pages_processed', 0),

            # Char/token stats
            "combined_content_length": full_doc_chars,
            "combined_content_characters": len("".join(t for _, t in page_texts).replace(' ', '').replace('\n', '')),
            "total_approx_tokens": full_doc_tokens_approx,

            # Splitting summary
            "num_full_documents_created": split_total,
            "base_document_id": base_document_id
        }

        # Attach a field the saver can use (just for convenience)
        document_data['source_json_file'] = os.path.basename(json_file_path)

        return documents, doc_summary

    except Exception as e:
        print(f"Error processing {json_file_path}: {e}")
        return [], None

def save_single_document_pickle(
    doc: Document,
    output_dir: Path,
    base_safe_name: str,
    split_index: int,
    split_total: int
) -> str:
    """Save a single LangChain Document to a unique pickle filename including file_id."""
    file_id = doc.metadata.get("file_id") or str(uuid.uuid4())
    short_id = file_id.split("-")[0]

    if split_total == 1:
        candidate = f"{base_safe_name}__full__{short_id}.pkl"
    else:
        candidate = f"{base_safe_name}__part-{split_index+1}-of-{split_total}__{short_id}.pkl"

    langchain_filepath = output_dir / candidate
    i = 1
    while langchain_filepath.exists():
        # Extremely unlikely since file_id is unique, but just in case
        if split_total == 1:
            candidate = f"{base_safe_name}__full__{short_id}__{i}.pkl"
        else:
            candidate = f"{base_safe_name}__part-{split_index+1}-of-{split_total}__{short_id}__{i}.pkl"
        langchain_filepath = output_dir / candidate
        i += 1

    with open(langchain_filepath, 'wb') as f:
        pickle.dump([doc], f)  # store as a single-item list for easy loading patterns

    return candidate

def process_json_directory() -> Dict[str, Any]:
    """Process all JSON files and write outputs."""
    input_path = Path(INPUT_DIR)
    if not input_path.exists():
        raise ValueError(f"Input directory '{INPUT_DIR}' does not exist")

    json_files = list(input_path.glob("*.json"))
    if not json_files:
        raise ValueError(f"No JSON files found in '{INPUT_DIR}'")

    if LIMIT_FILES is not None:
        json_files = json_files[:LIMIT_FILES]

    print("=== Full-Document LangChain Generator ===")
    print(f"Input directory: {INPUT_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"File limit: {'None (process all)' if LIMIT_FILES is None else LIMIT_FILES}")
    print(f"Token cap per document (approx): {MAX_TOKENS_PER_DOC} (≈ {MAX_CHARS_PER_DOC} chars)")
    print(f"PAGE_BREAK literal: {repr(PAGE_BREAK)}")
    print()

    if not check_and_prepare_output_directory(OUTPUT_DIR):
        return {"status": "cancelled"}

    output_path = Path(OUTPUT_DIR)
    processing_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    all_documents: List[Document] = []
    csv_rows: List[Dict[str, Any]] = []

    stats = {
        "total_files": len(json_files),
        "successful_files": 0,
        "failed_files": 0,
        "total_full_documents_generated": 0,
        "files_processed": [],
        "files_failed": [],
        "config_used": {
            "input_dir": INPUT_DIR,
            "output_dir": OUTPUT_DIR,
            "limit_files": LIMIT_FILES,
            "page_break": PAGE_BREAK,
            "token_cap": MAX_TOKENS_PER_DOC,
            "avg_chars_per_token": AVG_CHARS_PER_TOKEN,
            "max_chars_per_doc": MAX_CHARS_PER_DOC,
        }
    }

    print(f"Processing {len(json_files)} JSON files...\n")

    for i, json_file in enumerate(json_files, 1):
        print(f"[{i}/{len(json_files)}] {json_file.name}...", end=" ")
        try:
            documents, doc_summary = process_single_json_file_to_full_documents(str(json_file))
            if not documents or not doc_summary:
                stats["failed_files"] += 1
                stats["files_failed"].append(json_file.name)
                print("✗ Failed")
                continue

            base_name = safe_stem(doc_summary['source_json_file'])
            saved_filenames: List[str] = []

            for doc in documents:
                saved_name = save_single_document_pickle(
                    doc=doc,
                    output_dir=output_path,
                    base_safe_name=base_name,
                    split_index=doc.metadata["chunk_index"],
                    split_total=doc.metadata["total_chunks"]
                )
                saved_filenames.append(saved_name)
                all_documents.append(doc)

                # CSV row per generated file
                csv_rows.append({
                    "processing_timestamp": processing_timestamp,
                    "source_json_file": doc_summary['source_json_file'],
                    "generated_langchain_file": saved_name,
                    "file_id": doc.metadata.get("file_id"),
                    "original_pdf_filename": doc_summary['original_pdf_filename'],
                    "title": doc_summary['title'],
                    "uu_number": doc_summary['uu_number'],
                    "subject": doc_summary['subject'],
                    "tanggal_penetapan": doc_summary['tanggal_penetapan'],
                    "tanggal_pengundangan": doc_summary['tanggal_pengundangan'],
                    "tanggal_berlaku": doc_summary['tanggal_berlaku'],
                    "total_pages_in_pdf": doc_summary['total_pages_in_pdf'],
                    "pages_processed": doc_summary['pages_processed'],
                    "split_index": doc.metadata["chunk_index"],
                    "split_total": doc.metadata["total_chunks"],
                    "approx_token_count": doc.metadata.get("token_count", 0),
                    "char_count": doc.metadata.get("char_count", 0),
                    "base_document_id": doc_summary["base_document_id"],
                    "source_type": doc.metadata.get("source_type", "legal_document_full"),
                })

            stats["successful_files"] += 1
            stats["total_full_documents_generated"] += len(documents)
            stats["files_processed"].append({
                "filename": json_file.name,
                "generated_files": saved_filenames,
                "document_title": doc_summary.get('title', 'Unknown'),
                "uu_number": doc_summary.get('uu_number', 'Unknown'),
                "splits": len(documents)
            })

            print(f"✓ {len(documents)} document(s)")

        except Exception as e:
            stats["failed_files"] += 1
            stats["files_failed"].append(json_file.name)
            print(f"✗ Error: {e}")

    # Save combined documents pickle
    combined_pickle = output_path / "all_full_langchain_documents.pkl"
    with open(combined_pickle, 'wb') as f:
        pickle.dump(all_documents, f)
    print(f"\n💾 Saved all {len(all_documents)} Documents to {combined_pickle}")

    # Save CSV summary
    csv_file = output_path / "full_documents_summary.csv"
    if csv_rows:
        fieldnames = list(csv_rows[0].keys())
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"📊 Saved CSV summary with {len(csv_rows)} rows to {csv_file}")

    # Save processing stats
    stats_file = output_path / "processing_stats.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"📈 Saved processing statistics to {stats_file}")

    # Save sample preview
    if all_documents:
        sample = all_documents[0]
        sample_file = output_path / "sample_full_document.json"
        with open(sample_file, 'w', encoding='utf-8') as f:
            json.dump({
                "page_content": sample.page_content[:500] + ("..." if len(sample.page_content) > 500 else ""),
                "metadata": sample.metadata
            }, f, ensure_ascii=False, indent=2)
        print(f"🔍 Saved sample document preview to {sample_file}")

    # Print summary
    print("\n=== Summary ===")
    print(f"Files processed successfully: {stats['successful_files']}")
    print(f"Files failed: {stats['failed_files']}")
    print(f"Total LangChain documents generated: {stats['total_full_documents_generated']}")
    if stats['failed_files']:
        print(f"Failed files: {', '.join(stats['failed_files'])}")

    stats["status"] = "completed"
    return stats

# =============================================================================
# MAIN
# =============================================================================

def main():
    try:
        process_json_directory()
        print("\n✅ Done! One (or more) LangChain Document(s) per source JSON generated with ID-tagged filenames.")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Please check configuration and inputs, then try again.")

if __name__ == "__main__":
    main()