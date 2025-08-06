from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import json
import tiktoken
from typing import List, Dict, Any
import uuid

class TiktokenTextSplitter(RecursiveCharacterTextSplitter):
    """Text splitter that uses tiktoken for accurate token counting."""
    
    def __init__(self, encoding_name: str = "cl100k_base", **kwargs):
        super().__init__(**kwargs)
        self.encoding = tiktoken.get_encoding(encoding_name)
    
    def _length_function(self, text: str) -> int:
        return len(self.encoding.encode(text))

def combine_text_and_ocr(page: Dict[str, Any]) -> str:
    """Combine text and OCR content from a page."""
    text_content = page.get('text', '').strip()
    ocr_content = page.get('ocr', '').strip()
    
    # Combine with double newline as separator
    combined = []
    if text_content:
        combined.append(text_content)
    if ocr_content:
        combined.append(ocr_content)
    
    return '\n\n'.join(combined)

def create_page_metadata(pdf_data: Dict[str, Any], page: Dict[str, Any]) -> Dict[str, Any]:
    """Create comprehensive metadata for a page."""
    return {
        # Document-level metadata
        'filename': pdf_data['filename'],
        'total_pages_in_pdf': pdf_data['total_pages_in_pdf'],
        'pages_processed': pdf_data['pages_processed'],
        
        # Page-level metadata
        'page_number': page['page_number'],
        'page_index': page['page_index'],
        'ocr_strategy': page['ocr_strategy'],
        'ocr_reason': page['ocr_reason'],
        'has_image_blocks': page['has_image_blocks'],
        'image_blocks_count': page['image_blocks_count'],
        'visual_complexity': page['visual_complexity'],
        'text_length': page['text_length'],
        'ocr_length': page['ocr_length'],
        
        # For tracking
        'source_type': 'pdf_page',
        'original_page_id': f"{pdf_data['filename']}_page_{page['page_number']}"
    }

def load_json_to_langchain_documents(
    json_file_path: str = None,
    json_data: Dict = None,
    chunk_size: int = 512,
    chunk_overlap: int = 128
) -> List[Document]:
    """
    Convert JSON data to LangChain documents with chunking.
    
    Args:
        json_file_path: Path to JSON file (if loading from file)
        json_data: JSON data dict (if passing data directly)
        chunk_size: Token-based chunk size
        chunk_overlap: Token-based chunk overlap
    
    Returns:
        List of chunked LangChain documents
    """
    
    # Load data
    if json_file_path:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    elif json_data:
        data = json_data
    else:
        raise ValueError("Either json_file_path or json_data must be provided")
    
    documents = []
    
    # Process each PDF
    for pdf_filename, pdf_data in data.items():
        # Create a single document per PDF by concatenating all pages
        full_pdf_text = []
        
        # Sort pages by page_number to ensure correct order
        sorted_pages = sorted(pdf_data['pages'], key=lambda x: x['page_number'])
        
        for page in sorted_pages:
            page_content = combine_text_and_ocr(page)
            if page_content.strip():
                full_pdf_text.append(page_content)
        
        if full_pdf_text:
            # Combine all pages with page breaks
            combined_text = '\n\n--- PAGE BREAK ---\n\n'.join(full_pdf_text)
            
            # Create document-level metadata
            doc_metadata = {
                'filename': pdf_data['filename'],
                'total_pages_in_pdf': pdf_data['total_pages_in_pdf'],
                'pages_processed': pdf_data['pages_processed'],
                'source_type': 'pdf_document',
                'document_id': str(uuid.uuid4()),
                # Store page details for reference
                'page_details': {
                    page['page_number']: {
                        'page_index': page['page_index'],
                        'ocr_strategy': page['ocr_strategy'],
                        'ocr_reason': page['ocr_reason'],
                        'has_image_blocks': page['has_image_blocks'],
                        'image_blocks_count': page['image_blocks_count'],
                        'visual_complexity': page['visual_complexity'],
                        'text_length': page['text_length'],
                        'ocr_length': page['ocr_length']
                    }
                    for page in sorted_pages
                }
            }
            
            # Create document
            doc = Document(
                page_content=combined_text,
                metadata=doc_metadata
            )
            documents.append(doc)
    
    # Set up tokenizer-aware text splitter
    text_splitter = TiktokenTextSplitter(
        encoding_name="cl100k_base",  # Good for Gemini and most modern LLMs
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            '\n\n--- PAGE BREAK ---\n\n',  # Try to keep pages together when possible
            '\n\n',
            '\n',
            '. ',
            ' ',
            ''
        ]
    )
    
    # Chunk the documents
    chunked_documents = []
    for doc in documents:
        chunks = text_splitter.split_documents([doc])
        
        # Add chunk-specific metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata.update({
                'chunk_id': str(uuid.uuid4()),
                'chunk_index': i,
                'total_chunks': len(chunks),
                'parent_document_id': doc.metadata['document_id']
            })
            
        chunked_documents.extend(chunks)
    
    return chunked_documents

# Utility function to find source document for a chunk
def find_source_document(chunk: Document, original_documents: List[Document]) -> Document:
    """Find the original document that a chunk came from."""
    parent_id = chunk.metadata.get('parent_document_id')
    if parent_id:
        for doc in original_documents:
            if doc.metadata.get('document_id') == parent_id:
                return doc
    return None

# Usage example
if __name__ == "__main__":
    # Load and process documents
    documents = load_json_to_langchain_documents(
        json_file_path="your_data.json",
        chunk_size=512,
        chunk_overlap=128
    )
    
    print(f"Created {len(documents)} chunks from JSON data")
    
    # Example: Show chunk tracking
    for i, doc in enumerate(documents[:3]):  # Show first 3 chunks
        print(f"\nChunk {i + 1}:")
        print(f"  Filename: {doc.metadata['filename']}")
        print(f"  Chunk ID: {doc.metadata['chunk_id']}")
        print(f"  Parent Document ID: {doc.metadata['parent_document_id']}")
        print(f"  Chunk {doc.metadata['chunk_index'] + 1} of {doc.metadata['total_chunks']}")
        print(f"  Content preview: {doc.page_content[:100]}...")