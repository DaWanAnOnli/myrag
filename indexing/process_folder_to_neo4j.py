import os, pickle
from pathlib import Path
from typing import List
from dotenv import load_dotenv
from langchain.docstore.document import Document

from kg_extractor import extract_triples_from_chunk
from kg_store import write_triples_to_neo4j

# Load .env from the parent directory of this file
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

# Default directory: ../dataset/langchain-results/samples relative to this file
DEFAULT_LANGCHAIN_DIR = (Path(__file__).resolve().parent / "../dataset/langchain-results/samples").resolve()
LANGCHAIN_DIR = Path(os.getenv("LANGCHAIN_DIR", str(DEFAULT_LANGCHAIN_DIR)))

# Files to skip explicitly
SKIP_FILES = {"all_langchain_documents.pkl"}

def find_chunk_pickles(dir_path: Path) -> List[Path]:
    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {dir_path}")
    files = [p for p in dir_path.glob("*.pkl") if p.name not in SKIP_FILES]
    return sorted(files)

def load_chunks_from_file(pkl_path: Path) -> List[Document]:
    with open(pkl_path, "rb") as f:
        chunks = pickle.load(f)
    if not isinstance(chunks, list):
        raise ValueError(f"Unexpected pickle content in {pkl_path}")
    print(f"  - {pkl_path.name}: {len(chunks)} chunks")
    return chunks

def run_kg_pipeline_over_folder(
    dir_path: Path,
    max_files: int | None = None,
    max_chunks_per_file: int | None = None
):
    pkls = find_chunk_pickles(dir_path)
    if max_files is not None:
        pkls = pkls[:max_files]

    print(f"Found {len(pkls)} pickle files in {dir_path} (skipping: {', '.join(SKIP_FILES) or 'none'})")

    total_triples = 0
    total_chunks = 0

    for idx, pkl in enumerate(pkls, 1):
        print(f"[{idx}/{len(pkls)}] Processing {pkl.name}")
        try:
            chunks: List[Document] = load_chunks_from_file(pkl)
        except Exception as e:
            print(f"    ! Failed to load {pkl.name}: {e}")
            continue

        use_chunks = chunks if max_chunks_per_file is None else chunks[:max_chunks_per_file]
        for j, chunk in enumerate(use_chunks, 1):
            meta = {
                "document_id": chunk.metadata.get("document_id"),
                "chunk_id": chunk.metadata.get("chunk_id"),
                "uu_number": chunk.metadata.get("uu_number"),
                "pages": chunk.metadata.get("pages"),
            }
            try:
                triples = extract_triples_from_chunk(chunk.page_content, meta)
                if triples:
                    write_triples_to_neo4j(triples)
                    total_triples += len(triples)
                total_chunks += 1
            except Exception as e:
                print(f"    ! Extraction/store failed on chunk {j}: {e}")

        print(f"    ✓ Done {pkl.name}: processed {len(use_chunks)} chunks")

    print(f"\nAll done. Chunks processed: {total_chunks}; Triples stored: {total_triples}")

if __name__ == "__main__":
    run_kg_pipeline_over_folder(
        LANGCHAIN_DIR,
        max_files=None,            # set to an int for quick testing
        max_chunks_per_file=None   # set to an int for quick testing
    )