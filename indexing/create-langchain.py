from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import json
import tiktoken
from typing import List, Dict, Any, Union, Optional
import uuid
import os
import glob
from pathlib import Path
import pickle
import csv
from datetime import datetime

# =============================================================================
# CONFIGURATION CONSTANTS
# =============================================================================

INPUT_DIR = "../dataset/extract-results/samples"
OUTPUT_DIR = "../dataset/langchain-results/samples"

LIMIT_FILES = None  # Set to integer to limit files processed, None for all files
MIN_CHUNK_SIZE = 400
MAX_CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
FORCE_OVERWRITE = False  # Skip user confirmation for directory overwrite

# =============================================================================
# CORE CLASSES AND FUNCTIONS
# =============================================================================

class TiktokenTextSplitter(RecursiveCharacterTextSplitter):
    """Text splitter using tiktoken for accurate token counting."""
    
    def __init__(self, encoding_name: str = "cl100k_base", **kwargs):
        super().__init__(**kwargs)
        self.encoding = tiktoken.get_encoding(encoding_name)
    
    def _length_function(self, text: str) -> int:
        return len(self.encoding.encode(text))

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

def create_chunk_metadata(
    document_data: Dict[str, Any], 
    document_id: str, 
    chunk_info: Dict[str, Any],
    source_file: str
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

def determine_chunk_pages(chunk_text: str, page_mapping: List[Dict]) -> List[int]:
    """Determine which pages a chunk spans based on content analysis."""
    if '--- PAGE BREAK ---' not in chunk_text:
        # Single page chunk - find matching page
        for page_info in page_mapping:
            page_content = page_info['content'][:100]
            if page_content in chunk_text:
                return [page_info['page_number']]
        return [page_mapping[0]['page_number']] if page_mapping else [1]
    
    # Multi-page chunk
    chunk_parts = chunk_text.split('\n\n--- PAGE BREAK ---\n\n')
    pages = []
    
    for i, part in enumerate(chunk_parts):
        if part.strip() and i < len(page_mapping):
            pages.append(page_mapping[i]['page_number'])
    
    return pages if pages else [1]

def process_single_json_file(json_file_path: str) -> tuple[List[Document], Dict[str, Any]]:
    """Process a single JSON file into LangChain documents."""
    try:
        # Load and validate JSON file
        with open(json_file_path, 'r', encoding='utf-8') as f:
            document_data = json.load(f)
        
        if not isinstance(document_data, dict) or 'filename' not in document_data:
            return [], None
        
        document_id = str(uuid.uuid4())
        source_file = os.path.basename(json_file_path)
        
        # Prepare document summary for CSV
        doc_metadata = document_data.get('metadata', {})
        doc_summary = {
            'source_json_file': source_file,
            'original_pdf_filename': document_data['filename'],
            'title': doc_metadata.get('Title', 'Unknown'),
            'uu_number': doc_metadata.get('﻿UU_Number', 'Unknown'),
            'subject': doc_metadata.get('Subject', 'N/A'),
            'tanggal_penetapan': doc_metadata.get('Tanggal_Penetapan', 'N/A'),
            'tanggal_pengundangan': doc_metadata.get('Tanggal_Pengundangan', 'N/A'),
            'tanggal_berlaku': doc_metadata.get('Tanggal_Berlaku', 'N/A'),
            'total_pages_in_pdf': document_data['total_pages_in_pdf'],
            'pages_processed': document_data['pages_processed'],
            'document_id': document_id
        }
        
        # Process pages
        page_mapping = []
        full_document_text = []
        total_text_length = 0
        total_ocr_length = 0
        
        if 'pages' not in document_data or not document_data['pages']:
            return [], None
            
        sorted_pages = sorted(document_data['pages'], key=lambda x: x['page_number'])
        
        for page in sorted_pages:
            page_content = combine_text_and_ocr(page)
            total_text_length += page.get('text_length', 0)
            total_ocr_length += page.get('ocr_length', 0)
            
            if page_content.strip():
                page_mapping.append({
                    'page_number': page['page_number'],
                    'content': page_content
                })
                full_document_text.append(page_content)
        
        if not full_document_text:
            return [], None
        
        # Combine text and add content statistics
        combined_text = '\n\n--- PAGE BREAK ---\n\n'.join(full_document_text)
        doc_summary.update({
            'total_text_length': total_text_length,
            'total_ocr_length': total_ocr_length,
            'combined_content_length': len(combined_text),
            'combined_content_characters': len(combined_text.replace(' ', '').replace('\n', ''))
        })
        
        # Set up text splitter
        text_splitter = TiktokenTextSplitter(
            encoding_name="cl100k_base",
            chunk_size=MAX_CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=[
                '\n\n--- PAGE BREAK ---\n\n',  # Prefer page boundaries
                '\n\n', '\n', '. ', '! ', '? ', ' ', ''
            ]
        )
        
        # Calculate total tokens
        total_tokens = text_splitter._length_function(combined_text)
        doc_summary['total_tokens'] = total_tokens
        
        # Create and split document
        temp_doc = Document(
            page_content=combined_text,
            metadata={'document_id': document_id}
        )
        document_chunks = text_splitter.split_documents([temp_doc])
        
        # Process chunks with metadata
        processed_chunks = []
        chunk_token_counts = []
        chunk_character_counts = []
        
        for i, chunk in enumerate(document_chunks):
            chunk_pages = determine_chunk_pages(chunk.page_content, page_mapping)
            
            chunk_info = {
                'chunk_id': str(uuid.uuid4()),
                'chunk_index': i,
                'total_chunks': len(document_chunks),
                'pages': chunk_pages
            }
            
            chunk.metadata = create_chunk_metadata(document_data, document_id, chunk_info, source_file)
            
            # Calculate and set token count
            token_count = text_splitter._length_function(chunk.page_content)
            character_count = len(chunk.page_content)
            chunk.metadata['token_count'] = token_count
            
            chunk_token_counts.append(token_count)
            chunk_character_counts.append(character_count)
            processed_chunks.append(chunk)
        
        # Add chunk statistics to summary
        doc_summary.update({
            'num_chunks_created': len(processed_chunks),
            'avg_tokens_per_chunk': sum(chunk_token_counts) / len(chunk_token_counts) if chunk_token_counts else 0,
            'min_tokens_per_chunk': min(chunk_token_counts) if chunk_token_counts else 0,
            'max_tokens_per_chunk': max(chunk_token_counts) if chunk_token_counts else 0,
            'avg_characters_per_chunk': sum(chunk_character_counts) / len(chunk_character_counts) if chunk_character_counts else 0,
            'min_characters_per_chunk': min(chunk_character_counts) if chunk_character_counts else 0,
            'max_characters_per_chunk': max(chunk_character_counts) if chunk_character_counts else 0
        })
        
        return processed_chunks, doc_summary
        
    except Exception as e:
        return [], None

def save_individual_document_chunks(chunks: List[Document], doc_summary: Dict[str, Any], output_dir: Path) -> str:
    """Save chunks for a single document as separate pickle file."""
    if not chunks:
        return None
    
    # Create safe filename
    source_json = doc_summary['source_json_file']
    base_name = Path(source_json).stem
    safe_name = "".join(c for c in base_name if c.isalnum() or c in ('-', '_')).rstrip()
    
    langchain_filename = f"{safe_name}_chunks.pkl"
    langchain_filepath = output_dir / langchain_filename
    
    with open(langchain_filepath, 'wb') as f:
        pickle.dump(chunks, f)
    
    return langchain_filename

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
    # Validate input directory
    input_path = Path(INPUT_DIR)
    if not input_path.exists():
        raise ValueError(f"Input directory '{INPUT_DIR}' does not exist")
    
    # Find JSON files
    json_files = list(input_path.glob("*.json"))
    if not json_files:
        raise ValueError(f"No JSON files found in '{INPUT_DIR}'")
    
    print(f"Found {len(json_files)} JSON files in {INPUT_DIR}")
    
    # Apply file limit
    if LIMIT_FILES is not None:
        json_files = json_files[:LIMIT_FILES]
        print(f"Limited to processing {len(json_files)} files")
    
    if not check_and_prepare_output_directory(OUTPUT_DIR):
        return {"status": "cancelled"}
    
    # Initialize processing
    all_chunks = []
    csv_summary_data = []
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
            "min_chunk_size": MIN_CHUNK_SIZE,
            "max_chunk_size": MAX_CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "metadata_fields": ["chunk_id", "document_id", "pages", "chunk_index", "uu_number", "title", "tanggal_berlaku", "token_count", "source_type"]
        }
    }
    
    print(f"\nProcessing {len(json_files)} JSON files...")
    print(f"Configuration:")
    print(f"  Chunk size: {MIN_CHUNK_SIZE}-{MAX_CHUNK_SIZE} tokens")
    print(f"  Chunk overlap: {CHUNK_OVERLAP} tokens")
    print(f"  Metadata fields: Essential + Searchable (9 fields)")
    print(f"  File limit: {'None (all files)' if LIMIT_FILES is None else LIMIT_FILES}")
    print()
    
    output_path = Path(OUTPUT_DIR)
    processing_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Process each file
    for i, json_file in enumerate(json_files, 1):
        print(f"[{i}/{len(json_files)}] Processing {json_file.name}...", end=" ")
        
        chunks, doc_summary = process_single_json_file(str(json_file))
        
        if chunks and doc_summary:
            # Save individual document chunks
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
            
            # Prepare CSV entry
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
    all_chunks_file = output_path / "all_langchain_documents.pkl"
    with open(all_chunks_file, 'wb') as f:
        pickle.dump(all_chunks, f)
    print(f"\n💾 Saved all {len(all_chunks)} chunks to {all_chunks_file}")
    
    # Save CSV summary
    csv_file = output_path / "langchain-generation-summary.csv"
    if csv_summary_data:
        fieldnames = csv_summary_data[0].keys()
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
        print(f"Average tokens per chunk (across all): {df['avg_tokens_per_chunk'].mean():.1f}")
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
        print(f"Configuration used:")
        config = stats.get('config_used', {})
        for key, value in config.items():
            print(f"  {key}: {value}")
        
        print(f"\nResults:")
        print(f"  Total files found: {stats['total_files']}")
        print(f"  Successfully processed: {stats['successful_files']}")
        print(f"  Failed: {stats['failed_files']}")
        print(f"  Total chunks created: {stats['total_chunks']}")
        print(f"  Average chunks per document: {stats['total_chunks'] / max(stats['successful_files'], 1):.1f}")
        
        if stats.get('files_processed'):
            print(f"\nProcessed documents:")
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
    print(f"Metadata approach: Essential + Searchable (9 fields)")
    print()
    
    try:
        results = process_json_directory()
        
        if results["status"] == "completed":
            chunks = results["all_chunks"]
            
            print(f"\n=== Files Generated ===")
            print(f"📁 Individual chunk files: {results['successful_files']} files (*_chunks.pkl)")
            print(f"📁 Combined chunks file: all_langchain_documents.pkl")
            print(f"📊 CSV summary: langchain-generation-summary.csv")
            print(f"📈 Processing stats: processing_stats.json")
            print(f"🔍 Sample chunk: sample_chunk.json")
            
            print(f"\n=== Sample Analysis ===")
            if chunks:
                sample = get_chunk_summary(chunks[0])
                print(f"Sample chunk:")
                print(f"  UU Number: {sample['uu_number']}")
                print(f"  Title: {sample['title']}")
                print(f"  Effective Date: {sample['tanggal_berlaku']}")
                print(f"  Pages: {sample['pages']}")
                print(f"  Token count: {sample['token_count']}")
                print(f"  Content preview: {sample['content_preview']}")
                
                # Show unique UU numbers
                unique_uus = set(chunk.metadata.get('uu_number') for chunk in chunks)
                unique_uus.discard(None)
                print(f"\nUnique UU numbers found: {len(unique_uus)}")
                for uu in sorted(list(unique_uus))[:5]:
                    uu_chunks = filter_chunks_by_metadata(chunks, uu_number=uu)
                    print(f"  {uu}: {len(uu_chunks)} chunks")
                
                if len(unique_uus) > 5:
                    print(f"  ... and {len(unique_uus) - 5} more UU numbers")
            
            # Analyze CSV if possible
            try:
                print(f"\n=== CSV Analysis ===")
                analyze_csv_summary()
            except:
                print("CSV analysis skipped (install pandas for detailed analysis)")
            
            print(f"\n✅ Processing completed successfully!")
            print(f"📁 Results saved to: {OUTPUT_DIR}")
            
        elif results["status"] == "cancelled":
            print("❌ Processing was cancelled by user")
            
    except Exception as e:
        print(f"❌ Error during processing: {str(e)}")
        print("Please check the configuration constants and try again.")

if __name__ == "__main__":
    main()