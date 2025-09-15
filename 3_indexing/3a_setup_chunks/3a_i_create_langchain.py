import dotenv
from langchain.docstore.document import Document
import json
import tiktoken
from typing import List, Dict, Any, Optional, Tuple
import uuid
import os
from pathlib import Path
import pickle
import csv
from datetime import datetime

# =============================================================================
# CONFIGURATION CONSTANTS
# =============================================================================

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
    INPUT_DIR = f"../../dataset/samples/2_extract_text_results"
    OUTPUT_DIR = "../../dataset/samples/3_indexing/3a_langchain_results"
else:
    INPUT_DIR = f"../../dataset/2_extract_text_results"
    OUTPUT_DIR = "../../dataset/3_indexing/3a_langchain_results"


LIMIT_FILES = None  # Set to integer to limit files processed, None for all files
MIN_CHUNK_SIZE = 400   # tokens
MAX_CHUNK_SIZE = 800   # tokens
CHUNK_OVERLAP = 100    # tokens
FORCE_OVERWRITE = False  # Skip user confirmation for directory overwrite

PAGE_BREAK = "\n\n--- PAGE BREAK ---\n\n"

# =============================================================================
# TOKENIZATION HELPERS
# =============================================================================

class Tokenizer:
    def __init__(self, encoding_name: str = "cl100k_base"):
        self.encoding = tiktoken.get_encoding(encoding_name)

    def encode(self, text: str) -> List[int]:
        return self.encoding.encode(text)

    def decode(self, tokens: List[int]) -> str:
        return self.encoding.decode(tokens)

    def token_len(self, text: str) -> int:
        return len(self.encode(text))


# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def combine_text_and_ocr(page: Dict[str, Any]) -> str:
    """Combine text and OCR content from a page."""
    text_content = page.get('text', '').strip()
    ocr_content = page.get('ocr', '').strip()

    combined = []
    if text_content:
        combined.append(text_content)
    if ocr_content:
        combined.append(ocr_content)

    return '\n\n'.join(combined)


def build_combined_text_and_token_map(
    pages: List[Dict[str, Any]],
    tokenizer: Tokenizer
) -> Tuple[str, List[int], List[Dict[str, Any]]]:
    """
    Build the combined document text, its tokenization, and per-page token ranges.

    Returns:
      combined_text: the joined string with PAGE_BREAK markers
      combined_tokens: tokens of the combined_text
      page_token_ranges: list of dicts with page_number, start_token, end_token
    """
    page_texts = []
    for page in pages:
        page_text = combine_text_and_ocr(page)
        page_texts.append((page['page_number'], page_text))

    # Create combined text (used for character lengths and samples)
    combined_text = PAGE_BREAK.join(text for _, text in page_texts)

    # Tokenize page-by-page to derive accurate token ranges
    combined_tokens: List[int] = []
    page_token_ranges: List[Dict[str, Any]] = []
    break_tokens = tokenizer.encode(PAGE_BREAK)

    for idx, (page_number, text) in enumerate(page_texts):
        start = len(combined_tokens)
        page_tokens = tokenizer.encode(text)
        combined_tokens.extend(page_tokens)
        end = len(combined_tokens)
        page_token_ranges.append({
            "page_number": page_number,
            "start_token": start,
            "end_token": end
        })
        # append page break tokens between pages, not after last page
        if idx < len(page_texts) - 1:
            combined_tokens.extend(break_tokens)

    return combined_text, combined_tokens, page_token_ranges


def compute_token_windows(
    n_tokens: int,
    max_tokens: int,
    min_tokens: int,
    overlap_tokens: int
) -> List[Tuple[int, int]]:
    """
    Compute [start, end) token windows that:
    - Are within [min_tokens, max_tokens] for all chunks (last chunk may relax overlap).
    - Respect max_tokens strictly.
    - Use fixed stride (max_tokens - overlap_tokens) for all but possibly the last chunk.

    Returns list of (start, end) token indices.
    """
    if max_tokens <= 0:
        raise ValueError("MAX_CHUNK_SIZE must be > 0")
    if min_tokens <= 0 or min_tokens > max_tokens:
        raise ValueError("MIN_CHUNK_SIZE must be > 0 and <= MAX_CHUNK_SIZE")
    if overlap_tokens < 0 or overlap_tokens >= max_tokens:
        raise ValueError("CHUNK_OVERLAP must be >= 0 and < MAX_CHUNK_SIZE")

    if n_tokens == 0:
        return []

    # If the whole doc fits into one chunk
    if n_tokens <= max_tokens:
        return [(0, n_tokens)]

    stride = max_tokens - overlap_tokens
    starts: List[int] = []
    i = 0

    while True:
        remaining = n_tokens - i
        if remaining <= max_tokens:
            # Last chunk candidate
            if remaining < min_tokens and len(starts) > 0:
                # Compute feasible last start range so that size ∈ [min, max]
                min_start_for_max = n_tokens - max_tokens       # ensures size <= max
                max_start_for_min = n_tokens - min_tokens       # ensures size >= min

                # Prefer to keep at least the requested overlap from previous start
                preferred = max(min_start_for_max, starts[-1] + overlap_tokens)

                if preferred <= max_start_for_min:
                    last_start = preferred
                else:
                    # Relax overlap to satisfy min
                    last_start = max_start_for_min

                # Ensure strictly increasing
                if last_start <= starts[-1]:
                    last_start = min(max_start_for_min, starts[-1] + 1)

                starts.append(last_start)
            else:
                starts.append(i)
            break
        else:
            starts.append(i)
            i += stride

    windows: List[Tuple[int, int]] = []
    for s in starts:
        e = min(s + max_tokens, n_tokens)
        # guard: ensure each window obeys min except the single-chunk small-doc case
        if e - s < min_tokens and not (len(starts) == 1 and n_tokens < min_tokens):
            # This can only happen if the entire document is shorter than min, or constraints conflict.
            # As a fallback, set to at least min but never exceed n_tokens.
            e = min(n_tokens, s + max(min_tokens, 1))
        windows.append((s, e))

    return windows


def pages_spanned_by_window(
    start_token: int,
    end_token: int,
    page_token_ranges: List[Dict[str, Any]]
) -> List[int]:
    """Return the list of page numbers whose token ranges intersect [start_token, end_token)."""
    pages = []
    for p in page_token_ranges:
        if not (end_token <= p["start_token"] or start_token >= p["end_token"]):
            pages.append(p["page_number"])
    return pages if pages else [1]


def create_chunk_metadata(
    document_data: Dict[str, Any],
    document_id: str,
    chunk_info: Dict[str, Any],
) -> Dict[str, Any]:
    """Create chunk metadata with essential + searchable fields."""
    doc_metadata = document_data.get('metadata', {})

    return {
        # Core identifiers
        'chunk_id': chunk_info['chunk_id'],
        'document_id': document_id,
        'pages': chunk_info['pages'],
        'chunk_index': chunk_info['chunk_index'],

        # Business-critical searchable fields
        'uu_number': doc_metadata.get('﻿UU_Number'),
        'title': doc_metadata.get('Title'),
        'tanggal_berlaku': doc_metadata.get('Tanggal_Berlaku'),

        # Efficiency fields
        'token_count': 0,  # Set later after tokenization
        'source_type': 'legal_document'
    }


def process_single_json_file(json_file_path: str) -> tuple[List[Document], Optional[Dict[str, Any]]]:
    """Process a single JSON file into LangChain documents."""
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            document_data = json.load(f)

        if not isinstance(document_data, dict) or 'filename' not in document_data:
            return [], None

        document_id = str(uuid.uuid4())
        doc_metadata = document_data.get('metadata', {})

        doc_summary = {
            'source_json_file': os.path.basename(json_file_path),
            'original_pdf_filename': document_data['filename'],
            'title': doc_metadata.get('Title', 'Unknown'),
            'uu_number': doc_metadata.get('﻿UU_Number', 'Unknown'),
            'subject': doc_metadata.get('Subject', 'N/A'),
            'tanggal_penetapan': doc_metadata.get('Tanggal_Penetapan', 'N/A'),
            'tanggal_pengundangan': doc_metadata.get('Tanggal_Pengundangan', 'N/A'),
            'tanggal_berlaku': doc_metadata.get('Tanggal_Berlaku', 'N/A'),
            'total_pages_in_pdf': document_data.get('total_pages_in_pdf', 0),
            'pages_processed': document_data.get('pages_processed', 0),
            'document_id': document_id
        }

        if 'pages' not in document_data or not document_data['pages']:
            return [], None

        sorted_pages = sorted(document_data['pages'], key=lambda x: x['page_number'])

        # Gather plain character stats from the input
        total_text_length = 0
        total_ocr_length = 0
        for page in sorted_pages:
            total_text_length += page.get('text_length', 0)
            total_ocr_length += page.get('ocr_length', 0)

        tokenizer = Tokenizer("cl100k_base")

        # Build combined text and token mapping
        combined_text, combined_tokens, page_token_ranges = build_combined_text_and_token_map(sorted_pages, tokenizer)

        doc_summary.update({
            'total_text_length': total_text_length,
            'total_ocr_length': total_ocr_length,
            'combined_content_length': len(combined_text),
            'combined_content_characters': len(combined_text.replace(' ', '').replace('\n', ''))
        })

        total_tokens = len(combined_tokens)
        doc_summary['total_tokens'] = total_tokens

        # Compute token windows obeying min/max; relax overlap on the final chunk if needed
        windows = compute_token_windows(
            n_tokens=total_tokens,
            max_tokens=MAX_CHUNK_SIZE,
            min_tokens=MIN_CHUNK_SIZE,
            overlap_tokens=CHUNK_OVERLAP
        )

        # Convert windows to Documents
        processed_chunks: List[Document] = []
        chunk_token_counts: List[int] = []
        chunk_character_counts: List[int] = []

        for i, (start, end) in enumerate(windows):
            chunk_tokens = combined_tokens[start:end]
            chunk_text = tokenizer.decode(chunk_tokens)

            pages_spanned = pages_spanned_by_window(start, end, page_token_ranges)

            chunk_info = {
                'chunk_id': str(uuid.uuid4()),
                'chunk_index': i,
                'total_chunks': len(windows),
                'pages': pages_spanned
            }
            metadata = create_chunk_metadata(document_data, document_id, chunk_info)

            # Inject uu_number and title into the chunk content
            uu_number_val = metadata.get('uu_number')
            title_val = metadata.get('title')
            prefix_parts = []
            if uu_number_val:
                prefix_parts.append(f"UU Number: {uu_number_val}")
            if title_val:
                prefix_parts.append(f"Title: {title_val}")
            if prefix_parts:
                chunk_text = "\n".join(prefix_parts) + "\n\n" + chunk_text

            token_count = end - start
            character_count = len(chunk_text)
            metadata['token_count'] = token_count

            processed_chunks.append(Document(page_content=chunk_text, metadata=metadata))
            chunk_token_counts.append(token_count)
            chunk_character_counts.append(character_count)

        # Summaries and sanity checks
        avg_tokens = (sum(chunk_token_counts) / len(chunk_token_counts)) if chunk_token_counts else 0.0
        min_tokens = min(chunk_token_counts) if chunk_token_counts else 0
        max_tokens = max(chunk_token_counts) if chunk_token_counts else 0

        avg_chars = (sum(chunk_character_counts) / len(chunk_character_counts)) if chunk_character_counts else 0.0
        min_chars = min(chunk_character_counts) if chunk_character_counts else 0
        max_chars = max(chunk_character_counts) if chunk_character_counts else 0

        # Average can never exceed max; rounding won’t change that, but we guard anyway.
        if avg_tokens > max_tokens:
            avg_tokens = float(max_tokens)
        if avg_chars > max_chars:
            avg_chars = float(max_chars)

        doc_summary.update({
            'num_chunks_created': len(processed_chunks),
            'avg_tokens_per_chunk': avg_tokens,
            'min_tokens_per_chunk': min_tokens,
            'max_tokens_per_chunk': max_tokens,
            'avg_characters_per_chunk': avg_chars,
            'min_characters_per_chunk': min_chars,
            'max_characters_per_chunk': max_chars
        })

        return processed_chunks, doc_summary

    except Exception as e:
        print(f"Error processing {json_file_path}: {e}")
        return [], None


def save_individual_document_chunks(chunks: List[Document], doc_summary: Dict[str, Any], output_dir: Path) -> Optional[str]:
    """Save chunks for a single document as a separate pickle file, avoiding name collisions."""
    if not chunks:
        return None

    source_json = doc_summary['source_json_file']
    base_name = Path(source_json).stem
    safe_name = "".join(c for c in base_name if c.isalnum() or c in ('-', '_')).rstrip()

    # Start with the default candidate name
    candidate = f"{safe_name}_chunks.pkl" if safe_name else "doc_chunks.pkl"
    langchain_filepath = output_dir / candidate

    # If a file with that name exists, append an incrementing suffix
    i = 1
    while langchain_filepath.exists():
        candidate = f"{safe_name}__{i}_chunks.pkl" if safe_name else f"doc__{i}_chunks.pkl"
        langchain_filepath = output_dir / candidate
        i += 1

    with open(langchain_filepath, 'wb') as f:
        pickle.dump(chunks, f)

    return candidate


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


def process_json_directory() -> Dict[str, Any]:
    """Process all JSON files in the configured directory."""
    input_path = Path(INPUT_DIR)
    if not input_path.exists():
        raise ValueError(f"Input directory '{INPUT_DIR}' does not exist")

    json_files = list(input_path.glob("*.json"))
    if not json_files:
        raise ValueError(f"No JSON files found in '{INPUT_DIR}'")

    print(f"Found {len(json_files)} JSON files in {INPUT_DIR}")

    if LIMIT_FILES is not None:
        json_files = json_files[:LIMIT_FILES]
        print(f"Limited to processing {len(json_files)} files")

    if not check_and_prepare_output_directory(OUTPUT_DIR):
        return {"status": "cancelled"}

    all_chunks: List[Document] = []
    csv_summary_data: List[Dict[str, Any]] = []
    processing_stats = {
        "total_files": len(json_files),
        "successful_files": 0,
        "failed_files": 0,
        "total_chunks": 0,
        "files_processed": [],
        "files_failed": [],
        "config_used": {
            "input_dir": INPUT_DIR,
            "output_dir": OUTPUT_DIR,
            "limit_files": LIMIT_FILES,
            "min_chunk_size_tokens": MIN_CHUNK_SIZE,
            "max_chunk_size_tokens": MAX_CHUNK_SIZE,
            "chunk_overlap_tokens": CHUNK_OVERLAP,
            "metadata_fields": ["chunk_id", "document_id", "pages", "chunk_index", "uu_number", "title", "tanggal_berlaku", "token_count", "source_type"]
        }
    }

    print(f"\nProcessing {len(json_files)} JSON files...")
    print("Configuration:")
    print(f"  Chunk size: {MIN_CHUNK_SIZE}-{MAX_CHUNK_SIZE} tokens (last chunk may relax overlap to meet min)")
    print(f"  Chunk overlap: {CHUNK_OVERLAP} tokens")
    print(f"  Metadata fields: Essential + Searchable (9 fields)")
    print(f"  File limit: {'None (all files)' if LIMIT_FILES is None else LIMIT_FILES}")
    print()

    output_path = Path(OUTPUT_DIR)
    processing_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for i, json_file in enumerate(json_files, 1):
        print(f"[{i}/{len(json_files)}] Processing {json_file.name}...", end=" ")

        chunks, doc_summary = process_single_json_file(str(json_file))

        if chunks and doc_summary:
            langchain_filename = save_individual_document_chunks(chunks, doc_summary, output_path)

            all_chunks.extend(chunks)
            processing_stats["successful_files"] += 1
            processing_stats["total_chunks"] += len(chunks)
            processing_stats["files_processed"].append({
                "filename": json_file.name,
                "chunks_created": len(chunks),
                "document_title": doc_summary.get('title', 'Unknown'),
                "uu_number": doc_summary.get('uu_number', 'Unknown')
            })

            csv_entry = {
                'processing_timestamp': processing_timestamp,
                'source_json_file': doc_summary['source_json_file'],
                'generated_langchain_file': langchain_filename,
                'original_pdf_filename': doc_summary['original_pdf_filename'],
                'title': doc_summary['title'],
                'uu_number': doc_summary['uu_number'],
                'subject': doc_summary['subject'],
                'tanggal_penetapan': doc_summary['tanggal_penetapan'],
                'tanggal_pengundangan': doc_summary['tanggal_pengundangan'],
                'tanggal_berlaku': doc_summary['tanggal_berlaku'],
                'total_pages_in_pdf': doc_summary['total_pages_in_pdf'],
                'pages_processed': doc_summary['pages_processed'],
                'total_text_length': doc_summary['total_text_length'],
                'total_ocr_length': doc_summary['total_ocr_length'],
                'combined_content_length': doc_summary['combined_content_length'],
                'total_tokens': doc_summary['total_tokens'],
                'num_chunks_created': doc_summary['num_chunks_created'],
                'avg_tokens_per_chunk': round(doc_summary['avg_tokens_per_chunk'], 2),
                'min_tokens_per_chunk': doc_summary['min_tokens_per_chunk'],
                'max_tokens_per_chunk': doc_summary['max_tokens_per_chunk'],
                'avg_characters_per_chunk': round(doc_summary['avg_characters_per_chunk'], 2),
                'min_characters_per_chunk': doc_summary['min_characters_per_chunk'],
                'max_characters_per_chunk': doc_summary['max_characters_per_chunk'],
                'chunk_size_config': f"{MIN_CHUNK_SIZE}-{MAX_CHUNK_SIZE}",
                'chunk_overlap_config': CHUNK_OVERLAP,
                'metadata_approach': 'essential_searchable',
                'document_id': doc_summary['document_id']
            }
            csv_summary_data.append(csv_entry)

            print(f"✓ {len(chunks)} chunks")
        else:
            processing_stats["failed_files"] += 1
            processing_stats["files_failed"].append(json_file.name)
            print("✗ Failed")

    # Save outputs
    output_path.mkdir(parents=True, exist_ok=True)
    all_chunks_file = output_path / "all_langchain_documents.pkl"
    with open(all_chunks_file, 'wb') as f:
        pickle.dump(all_chunks, f)
    print(f"\n💾 Saved all {len(all_chunks)} chunks to {all_chunks_file}")

    # Save CSV summary
    csv_file = output_path / "langchain-generation-summary.csv"
    if csv_summary_data:
        fieldnames = list(csv_summary_data[0].keys())
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_summary_data)
        print(f"📊 Saved CSV summary with {len(csv_summary_data)} entries to {csv_file}")

    # Save processing stats
    stats_file = output_path / "processing_stats.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(processing_stats, f, ensure_ascii=False, indent=2)
    print(f"📈 Saved processing statistics to {stats_file}")

    # Save sample chunk
    if all_chunks:
        sample_file = output_path / "sample_chunk.json"
        sample_chunk = {
            "page_content": all_chunks[0].page_content[:500] + "...",
            "metadata": all_chunks[0].metadata
        }
        with open(sample_file, 'w', encoding='utf-8') as f:
            json.dump(sample_chunk, f, ensure_ascii=False, indent=2)
        print(f"🔍 Saved sample chunk to {sample_file}")

    # Print summary
    print(f"\n=== Processing Summary ===")
    print(f"Files processed successfully: {processing_stats['successful_files']}")
    print(f"Files failed: {processing_stats['failed_files']}")
    print(f"Total chunks created: {processing_stats['total_chunks']}")
    print(f"Average chunks per document: {processing_stats['total_chunks'] / max(processing_stats['successful_files'], 1):.1f}")

    if processing_stats['files_failed']:
        print(f"Failed files: {', '.join(processing_stats['files_failed'])}")

    processing_stats["all_chunks"] = all_chunks
    processing_stats["csv_summary_data"] = csv_summary_data
    processing_stats["status"] = "completed"

    return processing_stats


def load_processed_documents(langchain_dir: str = OUTPUT_DIR) -> List[Document]:
    """Load previously processed LangChain documents."""
    chunks_file = Path(langchain_dir) / "all_langchain_documents.pkl"

    if not chunks_file.exists():
        raise FileNotFoundError(f"No processed documents found at {chunks_file}")

    with open(chunks_file, 'rb') as f:
        chunks = pickle.load(f)

    print(f"Loaded {len(chunks)} chunks from {chunks_file}")
    return chunks

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_chunk_summary(chunk: Document) -> Dict[str, Any]:
    """Get summary of chunk information."""
    return {
        'chunk_id': chunk.metadata['chunk_id'],
        'document_id': chunk.metadata['document_id'],
        'uu_number': chunk.metadata['uu_number'],
        'title': chunk.metadata['title'],
        'tanggal_berlaku': chunk.metadata['tanggal_berlaku'],
        'pages': chunk.metadata['pages'],
        'chunk_position': f"{chunk.metadata['chunk_index'] + 1}",
        'token_count': chunk.metadata['token_count'],
        'source_type': chunk.metadata['source_type'],
        'content_preview': chunk.page_content[:200] + "..." if len(chunk.page_content) > 200 else chunk.page_content
    }


def filter_chunks_by_metadata(chunks: List[Document], **filters) -> List[Document]:
    """Filter chunks based on metadata criteria."""
    filtered_chunks = []

    for chunk in chunks:
        match = True
        for key, value in filters.items():
            chunk_value = chunk.metadata.get(key)

            if chunk_value is None and value is not None:
                match = False
                break
            elif isinstance(value, str) and isinstance(chunk_value, str):
                if value.lower() not in chunk_value.lower():
                    match = False
                    break
            elif chunk_value != value:
                match = False
                break

        if match:
            filtered_chunks.append(chunk)

    return filtered_chunks


def search_chunks_by_text_and_metadata(chunks: List[Document], text_query: str = None, **metadata_filters) -> List[Document]:
    """Search chunks by both text content and metadata."""
    # Filter by metadata first
    if metadata_filters:
        filtered_chunks = filter_chunks_by_metadata(chunks, **metadata_filters)
    else:
        filtered_chunks = chunks

    # Filter by text content if provided
    if text_query:
        text_filtered = []
        query_lower = text_query.lower()

        for chunk in filtered_chunks:
            if query_lower in chunk.page_content.lower():
                text_filtered.append(chunk)

        return text_filtered

    return filtered_chunks


def group_chunks_by_document(chunks: List[Document]) -> Dict[str, List[Document]]:
    """Group chunks by their source document."""
    grouped = {}

    for chunk in chunks:
        doc_id = chunk.metadata['document_id']
        if doc_id not in grouped:
            grouped[doc_id] = []
        grouped[doc_id].append(chunk)

    # Sort chunks within each document by chunk_index
    for doc_id in grouped:
        grouped[doc_id].sort(key=lambda x: x.metadata['chunk_index'])

    return grouped


def analyze_csv_summary(csv_file: str = None) -> None:
    """Analyze and display insights from CSV summary."""
    if csv_file is None:
        csv_file = Path(OUTPUT_DIR) / "langchain-generation-summary.csv"

    try:
        import pandas as pd
        df = pd.read_csv(csv_file)

        print("=== CSV Summary Analysis ===")
        print(f"Total documents processed: {len(df)}")
        print(f"Total chunks generated: {df['num_chunks_created'].sum()}")
        print(f"Average chunks per document: {df['num_chunks_created'].mean():.1f}")
        print(f"Average tokens per chunk (across all): {df['avg_tokens_per_chunk'].mean():.2f}")
        print(f"Total tokens processed: {df['total_tokens'].sum():,}")

        print(f"\nChunk statistics:")
        print(f"  Min chunks per doc: {df['num_chunks_created'].min()}")
        print(f"  Max chunks per doc: {df['num_chunks_created'].max()}")
        print(f"  Median chunks per doc: {df['num_chunks_created'].median()}")

        print(f"\nTop 5 documents by chunk count:")
        top_chunks = df.nlargest(5, 'num_chunks_created')[['title', 'uu_number', 'num_chunks_created']]
        for _, row in top_chunks.iterrows():
            print(f"  {row['uu_number']}: {row['num_chunks_created']} chunks")

    except ImportError:
        print("Install pandas for advanced CSV analysis: pip install pandas")
    except FileNotFoundError:
        print(f"CSV summary not found at {csv_file}")
    except Exception as e:
        print(f"Error analyzing CSV: {e}")


def get_processing_summary(stats_file: str = None) -> None:
    """Display summary of processing results."""
    if stats_file is None:
        stats_file = Path(OUTPUT_DIR) / "processing_stats.json"

    try:
        with open(stats_file, 'r', encoding='utf-8') as f:
            stats = json.load(f)

        print("=== Processing Summary ===")
        print("Configuration used:")
        config = stats.get('config_used', {})
        for key, value in config.items():
            print(f"  {key}: {value}")

        print("\nResults:")
        print(f"  Total files found: {stats['total_files']}")
        print(f"  Successfully processed: {stats['successful_files']}")
        print(f"  Failed: {stats['failed_files']}")
        print(f"  Total chunks created: {stats['total_chunks']}")
        print(f"  Average chunks per document: {stats['total_chunks'] / max(stats['successful_files'], 1):.1f}")

        if stats.get('files_processed'):
            print("\nProcessed documents:")
            for doc in stats['files_processed'][:5]:
                print(f"  {doc['filename']}: {doc['chunks_created']} chunks - {doc['uu_number']}")

            if len(stats['files_processed']) > 5:
                print(f"  ... and {len(stats['files_processed']) - 5} more documents")

        if stats.get('files_failed'):
            print(f"Failed files: {', '.join(stats['files_failed'])}")

    except FileNotFoundError:
        print(f"No processing stats found at {stats_file}")
    except Exception as e:
        print(f"Error reading stats: {e}")


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """Main function - orchestrates the document processing workflow."""
    print("=== Legal Document Processing Pipeline ===")
    print(f"Input directory: {INPUT_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"File limit: {'None (process all)' if LIMIT_FILES is None else LIMIT_FILES}")
    print(f"Chunk configuration: {MIN_CHUNK_SIZE}-{MAX_CHUNK_SIZE} tokens, overlap: {CHUNK_OVERLAP}")
    print("Metadata approach: Essential + Searchable (9 fields)")
    print()

    try:
        results = process_json_directory()

        if results["status"] == "completed":
            chunks = results["all_chunks"]

            print("\n=== Files Generated ===")
            print(f"📁 Individual chunk files: {results['successful_files']} files (*_chunks.pkl)")
            print("📁 Combined chunks file: all_langchain_documents.pkl")
            print("📊 CSV summary: langchain-generation-summary.csv")
            print("📈 Processing stats: processing_stats.json")
            print("🔍 Sample chunk: sample_chunk.json")

            print("\n=== Sample Analysis ===")
            if chunks:
                sample = get_chunk_summary(chunks[0])
                print("Sample chunk:")
                print(f"  UU Number: {sample['uu_number']}")
                print(f"  Title: {sample['title']}")
                print(f"  Effective Date: {sample['tanggal_berlaku']}")
                print(f"  Pages: {sample['pages']}")
                print(f"  Token count: {sample['token_count']}")
                print(f"  Content preview: {sample['content_preview']}")

                unique_uus = set(chunk.metadata.get('uu_number') for chunk in chunks)
                unique_uus.discard(None)
                print(f"\nUnique UU numbers found: {len(unique_uus)}")
                for uu in sorted(list(unique_uus))[:5]:
                    uu_chunks = filter_chunks_by_metadata(chunks, uu_number=uu)
                    print(f"  {uu}: {len(uu_chunks)} chunks")

                if len(unique_uus) > 5:
                    print(f"  ... and {len(unique_uus) - 5} more UU numbers")

            try:
                print("\n=== CSV Analysis ===")
                analyze_csv_summary()
            except:
                print("CSV analysis skipped (install pandas for detailed analysis)")

            print("\n✅ Processing completed successfully!")
            print(f"📁 Results saved to: {OUTPUT_DIR}")

        elif results["status"] == "cancelled":
            print("❌ Processing was cancelled by user")

    except Exception as e:
        print(f"❌ Error during processing: {str(e)}")
        print("Please check the configuration constants and try again.")


if __name__ == "__main__":
    main()