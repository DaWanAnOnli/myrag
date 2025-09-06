import os
import json
import time
import uuid
from pathlib import Path
import fitz  # PyMuPDF
from paddleocr import PaddleOCR
from PIL import Image
import io
import numpy as np
from PyPDF2 import PdfReader
import difflib
import re
import paddle
import dotenv

env_file_path = Path("../.env")
    
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

# ======================
# OCR/Rendering Configuration (new)
# ======================
RENDER_DPI = 200  # 200–300 is typical for OCR
# Temporary directory name for PaddleOCR JSON outputs
JSON_TMP_DIRNAME = "_ocr_json_tmp"

# PaddleOCR options (aligned with your example script)
PADDLE_OCR_OPTIONS = dict(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
)

# Global temp dir for OCR JSON files (will be created in main)
_OCR_JSON_TMP_DIR = None


def setup_ocr_reader():
    """Initialize PaddleOCR engine."""
    t0 = time.perf_counter()
    ocr = PaddleOCR(**PADDLE_OCR_OPTIONS)
    t_init = time.perf_counter() - t0
    print(f"✅ PaddleOCR model initialized in {t_init:.2f}s")
    return ocr


def _ensure_ocr_json_tmp_dir():
    """Create and return the temp JSON directory used by PaddleOCR."""
    global _OCR_JSON_TMP_DIR
    if _OCR_JSON_TMP_DIR is None:
        _OCR_JSON_TMP_DIR = (Path.cwd() / JSON_TMP_DIRNAME).resolve()
        _OCR_JSON_TMP_DIR.mkdir(parents=True, exist_ok=True)
    return _OCR_JSON_TMP_DIR


def _render_matrix_for_dpi(dpi):
    """Return a fitz.Matrix scaled for desired DPI."""
    zoom = dpi / 72.0
    return fitz.Matrix(zoom, zoom)


def _paddle_predict_image_path(image_path, ocr_reader):
    """
    Run PaddleOCR on an image path using predict(...) and extract text lines
    by reading the JSON(s) written by save_to_json, as in your example.
    Cleans up intermediate JSON files afterwards.
    Returns a list of recognized text lines.
    """
    json_tmp_dir = _ensure_ocr_json_tmp_dir()
    page_texts = []

    # Run OCR
    result = ocr_reader.predict(str(image_path))

    # Save result objects to JSON using provided API
    saved_json_paths = []
    for res in result:
        # Writes JSON(s) to json_tmp_dir; PaddleOCR names files after image stem
        res.save_to_json(str(json_tmp_dir))

        # Find saved JSON files that start with the PNG's stem
        stem = Path(image_path).stem
        for candidate in json_tmp_dir.glob(f"{stem}*.json"):
            saved_json_paths.append(candidate)

    # Deduplicate and read
    saved_json_paths = sorted(set(saved_json_paths))
    for jp in saved_json_paths:
        try:
            data = json.loads(jp.read_text(encoding="utf-8"))
            rec_texts = data.get("rec_texts", [])
            if isinstance(rec_texts, list):
                page_texts.extend([t for t in rec_texts if isinstance(t, str)])
        finally:
            try:
                jp.unlink(missing_ok=True)
            except Exception as e:
                print(f"Warning: failed to delete JSON {jp.name}: {e}")

    return page_texts


def extract_pdf_metadata(pdf_path):
    """Extract metadata from PDF file"""
    try:
        print(f"  Extracting metadata from {os.path.basename(pdf_path)}...")
        
        with open(pdf_path, 'rb') as file:
            reader = PdfReader(file)
            metadata = reader.metadata
            
            if metadata:
                clean_metadata = {}
                for key, value in metadata.items():
                    if key.startswith('/Custom_'):
                        column_name = key.replace('/Custom_', '')
                        clean_metadata[column_name] = str(value) if value else ""
                        print(f"    Custom: {column_name} = {value}")
                    else:
                        clean_key = key.lstrip('/')
                        clean_metadata[clean_key] = str(value) if value else ""
                        print(f"    Standard: {clean_key} = {value}")
                
                print(f"  Found {len(clean_metadata)} metadata fields")
                return clean_metadata
            else:
                print(f"  No metadata found in {os.path.basename(pdf_path)}")
                return {}
                
    except Exception as e:
        print(f"  Error reading metadata from {pdf_path}: {e}")
        return {"error": str(e)}


def extract_text_from_page_only(page, page_num):
    """Extract only the regular text from this specific page"""
    try:
        text = page.get_text()
        print(f"    Page {page_num} regular text: {len(text)} characters")
        return text.strip()
    except Exception as e:
        print(f"    Error extracting text from page {page_num}: {e}")
        return ""


def get_image_blocks_info(page, page_num):
    """Get information about image blocks on this page"""
    try:
        blocks = page.get_text("dict")["blocks"]
        image_blocks = []
        
        for block_num, block in enumerate(blocks):
            if block.get("type") == 1:  # Type 1 = image block
                bbox = block["bbox"]  # [x0, y0, x1, y1]
                image_blocks.append({
                    "block_num": block_num,
                    "bbox": bbox,
                    "width": bbox[2] - bbox[0],
                    "height": bbox[3] - bbox[1]
                })
        
        print(f"    Page {page_num} has {len(image_blocks)} image blocks")
        for i, img_block in enumerate(image_blocks):
            print(f"      Block {i+1}: {img_block['width']:.0f}x{img_block['height']:.0f} at {img_block['bbox']}")
        
        return image_blocks
        
    except Exception as e:
        print(f"    Error getting image blocks from page {page_num}: {e}")
        return []


def extract_image_region(page, bbox, page_num, block_num):
    """Extract a specific image region from the page and return as PIL Image"""
    try:
        clip_rect = fitz.Rect(bbox)
        # Use DPI-based scale for better OCR quality
        mat = _render_matrix_for_dpi(RENDER_DPI)
        pix = page.get_pixmap(matrix=mat, clip=clip_rect)
        
        img_data = pix.tobytes("png")
        pil_image = Image.open(io.BytesIO(img_data))
        
        print(f"      Extracted region {block_num}: {pil_image.size}")
        
        pix = None
        return pil_image
        
    except Exception as e:
        print(f"      Error extracting image region {block_num}: {e}")
        return None


def perform_ocr_on_image_region(pil_image, ocr_reader, page_num, block_num):
    """
    Perform OCR on a specific image region using PaddleOCR.
    Implements the 'predict + save_to_json + read rec_texts' flow from your example.
    """
    try:
        img_array = np.array(pil_image)
        print(f"      Performing OCR on region {block_num} ({img_array.shape})...")
        
        # Save region to a temporary PNG
        tmp_dir = _ensure_ocr_json_tmp_dir()
        tmp_png = tmp_dir / f"region_p{page_num:04d}_b{block_num:04d}_{uuid.uuid4().hex[:8]}.png"
        pil_image.save(tmp_png)
        
        try:
            lines = _paddle_predict_image_path(tmp_png, ocr_reader)
            region_ocr = " ".join(lines).strip()
            print(f"      Region {block_num} OCR result: {len(region_ocr)} characters")
            return region_ocr
        finally:
            # Clean up temp image
            try:
                tmp_png.unlink(missing_ok=True)
            except Exception as e:
                print(f"Warning: failed to delete temp PNG {tmp_png.name}: {e}")
        
    except Exception as e:
        print(f"      Error performing OCR on region {block_num}: {e}")
        return ""


def extract_text_from_image_blocks_only(page, page_num, ocr_reader):
    """Extract text only from image blocks on the page (via PaddleOCR)"""
    try:
        print(f"    Extracting text from image blocks on page {page_num}...")
        
        image_blocks = get_image_blocks_info(page, page_num)
        if not image_blocks:
            print(f"    No image blocks found on page {page_num}")
            return ""
        
        all_ocr_texts = []
        for i, img_block in enumerate(image_blocks):
            block_num = i + 1
            bbox = img_block["bbox"]
            
            region_image = extract_image_region(page, bbox, page_num, block_num)
            if region_image is not None:
                region_text = perform_ocr_on_image_region(region_image, ocr_reader, page_num, block_num)
                if region_text:
                    all_ocr_texts.append(region_text)
        
        final_ocr_text = ' '.join(all_ocr_texts).strip()
        print(f"    Total OCR from {len(image_blocks)} image blocks: {len(final_ocr_text)} characters")
        return final_ocr_text
        
    except Exception as e:
        print(f"    Error extracting text from image blocks on page {page_num}: {e}")
        return ""


def perform_full_page_ocr(page, page_num, ocr_reader):
    """
    Perform OCR on the entire page (for scanned documents) using PaddleOCR.
    Follows the 'render to PNG -> predict -> save_to_json -> read rec_texts' flow.
    """
    try:
        print(f"    Performing full-page OCR on page {page_num}...")
        
        mat = _render_matrix_for_dpi(RENDER_DPI)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        
        tmp_dir = _ensure_ocr_json_tmp_dir()
        tmp_png = tmp_dir / f"full_p{page_num:04d}_{uuid.uuid4().hex[:8]}.png"
        pix.save(tmp_png)
        
        try:
            lines = _paddle_predict_image_path(tmp_png, ocr_reader)
            full_page_ocr = " ".join(lines).strip()
            print(f"    Full-page OCR result: {len(full_page_ocr)} characters")
            return full_page_ocr
        finally:
            try:
                tmp_png.unlink(missing_ok=True)
            except Exception as e:
                print(f"Warning: failed to delete temp PNG {tmp_png.name}: {e}")
        
    except Exception as e:
        print(f"    Error performing full-page OCR on page {page_num}: {e}")
        return ""


def analyze_page_content(page, page_num):
    """Analyze what type of content this page has"""
    try:
        regular_text = page.get_text().strip()
        text_length = len(regular_text)
        
        blocks = page.get_text("dict")["blocks"]
        image_blocks = [b for b in blocks if b.get("type") == 1]
        
        pix = page.get_pixmap(matrix=fitz.Matrix(0.5, 0.5))
        img_data = pix.tobytes("png")
        pil_image = Image.open(io.BytesIO(img_data))
        img_array = np.array(pil_image.convert('L'))
        
        visual_complexity = float(np.std(img_array))
        pix = None
        
        print(f"    Page {page_num} analysis:")
        print(f"      Text length: {text_length}")
        print(f"      Image blocks: {len(image_blocks)}")
        print(f"      Visual complexity: {visual_complexity:.1f}")
        
        return {
            "text_length": text_length,
            "has_image_blocks": len(image_blocks) > 0,
            "image_blocks_count": len(image_blocks),
            "visual_complexity": visual_complexity,
            "is_likely_scanned": text_length < 50 and visual_complexity > 40
        }
        
    except Exception as e:
        print(f"    Error analyzing page {page_num}: {e}")
        return {
            "text_length": 0,
            "has_image_blocks": False,
            "image_blocks_count": 0,
            "visual_complexity": 0.0,
            "is_likely_scanned": False
        }


def decide_ocr_strategy(page_analysis, page_num):
    """Decide OCR strategy based on page analysis"""
    if page_analysis["has_image_blocks"]:
        print(f"    Page {page_num}: OCR image blocks only - {page_analysis['image_blocks_count']} blocks found")
        return "image_blocks_only", "has_image_blocks"
    
    if page_analysis["is_likely_scanned"]:
        print(f"    Page {page_num}: Full-page OCR - appears to be scanned")
        return "full_page_ocr", "likely_scanned"
    
    print(f"    Page {page_num}: No OCR needed - regular text page")
    return "no_ocr", "regular_text_page"


def _normalize_for_similarity(text: str) -> str:
    """Normalize text to make semantic comparison more robust."""
    if not text:
        return ""
    # Remove soft hyphens and hyphenation at line breaks
    text = text.replace("\u00ad", "")
    text = re.sub(r"-\s*\n\s*", "", text)  # stitch hyphenated line breaks
    # Collapse whitespace and lowercase
    text = " ".join(text.split()).lower()
    return text


def _similarity_ratio(a: str, b: str) -> float:
    """Return a normalized similarity ratio between two strings."""
    na, nb = _normalize_for_similarity(a), _normalize_for_similarity(b)
    if not na and not nb:
        return 1.0
    return difflib.SequenceMatcher(None, na, nb).ratio()


def _looks_corrupted(text: str) -> bool:
    """
    Heuristic: is this text likely corrupted/low quality?
    Flags if:
      - Contains replacement chars,
      - Too many non-alnum+space,
      - Too many non-printables,
      - Lots of very long tokens (e.g., glued words).
    """
    if not text:
        return True
    total = len(text)
    printable = sum(ch.isprintable() for ch in text)
    alnum_space = sum(ch.isalnum() or ch.isspace() for ch in text)
    tokens = text.split()
    long_tokens = sum(1 for t in tokens if len(t) >= 30)

    has_replacement = "�" in text
    non_printable_ratio = 1 - (printable / max(total, 1))
    non_alnum_space_ratio = 1 - (alnum_space / max(total, 1))
    long_token_ratio = long_tokens / max(len(tokens), 1)

    return (
        has_replacement
        or non_printable_ratio > 0.10
        or non_alnum_space_ratio > 0.45
        or long_token_ratio > 0.05
    )


def process_single_page_isolated(doc, page_index, ocr_reader):
    """Process a single page with targeted OCR strategy"""
    page_num = page_index + 1
    print(f"\n  ========== PAGE {page_num} (index {page_index}) ==========")
    
    try:
        page = doc.load_page(page_index)
        
        regular_text = extract_text_from_page_only(page, page_num)
        page_analysis = analyze_page_content(page, page_num)
        ocr_strategy, ocr_reason = decide_ocr_strategy(page_analysis, page_num)
        
        ocr_text = ""
        if ocr_strategy == "image_blocks_only":
            ocr_text = extract_text_from_image_blocks_only(page, page_num, ocr_reader)
        elif ocr_strategy == "full_page_ocr":
            ocr_text = perform_full_page_ocr(page, page_num, ocr_reader)
        else:
            print(f"    Page {page_num}: Skipping OCR ({ocr_reason})")
        
        # ======= Deduplicate regular text vs OCR if they are extremely similar =======
        reg = regular_text or ""
        ocr = ocr_text or ""
        final_text_source = None
        duplicates_merged = False
        similarity = None

        # Decide which to keep
        if reg and ocr:
            similarity = _similarity_ratio(reg, ocr)
            len_close = abs(len(reg) - len(ocr)) / max(len(reg), 1) <= 0.05
            extremely_similar = similarity >= 0.90 or (similarity >= 0.80 and len_close)

            if extremely_similar:
                # Prefer regular PDF text unless it looks corrupted or the page is effectively scanned
                if _looks_corrupted(reg) and not _looks_corrupted(ocr):
                    final_text_source = "ocr"
                elif ocr_strategy == "full_page_ocr" and page_analysis["is_likely_scanned"]:
                    final_text_source = "ocr"
                else:
                    final_text_source = "regular"

                # Blank the unused text to avoid duplication downstream
                if final_text_source == "regular":
                    ocr = ""
                else:
                    reg = ""
                duplicates_merged = True

        elif reg and not ocr:
            final_text_source = "regular"
        elif ocr and not reg:
            final_text_source = "ocr"
        else:
            final_text_source = "none"

        final_text = reg if final_text_source == "regular" else (ocr if final_text_source == "ocr" else "")

        result = {
            "page_number": int(page_num),
            "page_index": int(page_index),
            "text": str(reg),  # possibly blanked if we chose OCR
            "ocr": str(ocr),   # possibly blanked if we chose regular
            "final_text": str(final_text),  # single field you can rely on
            "final_text_source": str(final_text_source),
            "duplicates_merged": bool(duplicates_merged),
            "text_similarity": float(similarity) if similarity is not None else None,
            "ocr_strategy": str(ocr_strategy),
            "ocr_reason": str(ocr_reason),
            "has_image_blocks": bool(page_analysis["has_image_blocks"]),
            "image_blocks_count": int(page_analysis["image_blocks_count"]),
            "visual_complexity": float(page_analysis["visual_complexity"]),
            "text_length": int(len(reg)),
            "ocr_length": int(len(ocr)),
            "final_text_length": int(len(final_text)),
        }

        print(f"    Page {page_num} FINAL:")
        print(f"      Regular text (kept): {len(reg)} chars")
        print(f"      OCR text (kept): {len(ocr)} chars")
        if similarity is not None:
            print(f"      Similarity: {similarity:.4f} | Duplicates merged: {duplicates_merged} | Used: {final_text_source}")
        print(f"      OCR strategy: {ocr_strategy}")
        
        page = None
        return result
        
    except Exception as e:
        print(f"    ERROR processing page {page_num}: {e}")
        return {
            "page_number": int(page_num),
            "page_index": int(page_index),
            "text": "",
            "ocr": "",
            "final_text": "",
            "final_text_source": "error",
            "duplicates_merged": False,
            "text_similarity": None,
            "ocr_strategy": "error",
            "ocr_reason": "error",
            "has_image_blocks": False,
            "image_blocks_count": 0,
            "visual_complexity": 0.0,
            "text_length": 0,
            "ocr_length": 0,
            "final_text_length": 0,
            "error": str(e)
        }


def process_pdf_with_metadata_and_ocr(pdf_path, ocr_reader, max_pages=None):
    """Process PDF with metadata extraction and targeted OCR per page"""
    try:
        print(f"\n{'='*60}")
        print(f"PROCESSING PDF: {os.path.basename(pdf_path)}")
        print(f"{'='*60}")
        
        pdf_metadata = extract_pdf_metadata(pdf_path)
        
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        
        pages_to_process = total_pages if max_pages is None else min(max_pages, total_pages)
        
        print(f"\nDocument has {total_pages} total pages")
        if max_pages is None:
            print(f"Processing ALL {pages_to_process} pages")
        else:
            print(f"Processing first {pages_to_process} pages (limited)")
        
        pages_data = []
        
        for page_index in range(pages_to_process):
            page_result = process_single_page_isolated(doc, page_index, ocr_reader)
            pages_data.append(page_result)
            
            if pages_to_process > 10 and (page_index + 1) % 10 == 0:
                print(f"\n  *** Progress: {page_index + 1}/{pages_to_process} pages completed ***")
        
        doc.close()
        
        print(f"\nCompleted processing {len(pages_data)} pages from {os.path.basename(pdf_path)}")
        return pages_data, total_pages, pdf_metadata
        
    except Exception as e:
        print(f"Error processing PDF {pdf_path}: {e}")
        return [], 0, {}


def create_json_filename(pdf_filename):
    """Create a JSON filename based on the PDF filename"""
    base_name = pdf_filename.replace('.pdf', '').replace('.PDF', '')
    return f"{base_name}.json"


def save_pdf_result_to_json(pdf_result, output_dir, pdf_filename):
    """Save a single PDF's result to its own JSON file"""
    try:
        json_filename = create_json_filename(pdf_filename)
        json_path = output_dir / json_filename
        
        print(f"  Saving results to: {json_path}")
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(pdf_result, f, indent=2, ensure_ascii=False)
        
        print(f"  ✅ Successfully saved: {json_filename}")
        return True
        
    except Exception as e:
        print(f"  ❌ Error saving {pdf_filename} results: {e}")
        return False


def check_gpu():
    if paddle.is_compiled_with_cuda():
        print("PaddlePaddle is compiled with CUDA support.")
        if paddle.get_device() == "gpu":
            print("PaddlePaddle is using the GPU.")
        else:
            print("PaddlePaddle is using the CPU.")
    else:
        print("PaddlePaddle is not compiled with CUDA support. Using CPU.")

check_gpu()

def main():
    # ======================
    # PROCESSING CONFIGURATION
    # ======================
    MAX_PDF_FILES = None  # Set to None to process ALL PDFs, or a number to limit
    MAX_PAGES_PER_PDF = None  # Set to None to process ALL pages, or a number to limit
    
    print("=" * 60)
    print("PDF OCR WITH METADATA EXTRACTION - INDIVIDUAL JSON FILES (PaddleOCR)")
    print("=" * 60)
    
    if MAX_PDF_FILES is None:
        print("Processing: ALL PDF files in directory")
    else:
        print(f"Processing: First {MAX_PDF_FILES} PDF file(s)")
        
    if MAX_PAGES_PER_PDF is None:
        print("Pages per PDF: ALL pages in each PDF")
    else:
        print(f"Pages per PDF: First {MAX_PAGES_PER_PDF} page(s)")
        
    print("=" * 60)
    print(f"OCR Rendering DPI: {RENDER_DPI}")

    if IS_SAMPLE:
        pdf_directory = Path("../dataset/samples/1_pdfs_with_metadata")
        output_directory = Path("../dataset/samples/2_extract_text_results")
    else:
        pdf_directory = Path("../dataset/1_pdfs_with_metadata")
        output_directory = Path("../dataset/2_extract_text_results")
    
    if not pdf_directory.exists():
        print(f"❌ ERROR: Input directory {pdf_directory} does not exist!")
        return
    
    try:
        output_directory.mkdir(parents=True, exist_ok=True)
        print(f"📁 Output directory: {output_directory}")
    except Exception as e:
        print(f"❌ ERROR: Could not create output directory {output_directory}: {e}")
        return
    
    # Prepare temp JSON dir for PaddleOCR
    tmp_dir = _ensure_ocr_json_tmp_dir()
    print(f"🗂️  Temp OCR JSON dir: {tmp_dir}")
    
    pdf_files = list(pdf_directory.glob("*.pdf"))
    if not pdf_files:
        print("❌ ERROR: No PDF files found in input directory!")
        return
    
    if MAX_PDF_FILES is None:
        files_to_process = pdf_files
        print(f"📄 Found {len(pdf_files)} PDFs - processing ALL of them")
    else:
        files_to_process = pdf_files[:MAX_PDF_FILES]
        print(f"📄 Found {len(pdf_files)} PDFs - processing first {len(files_to_process)}")
    
    print(f"📋 Files to process: {[f.name for f in files_to_process]}")
    
    print("\n🔧 Initializing OCR...")
    try:
        ocr_reader = setup_ocr_reader()
        print("✅ OCR ready!")
    except Exception as e:
        print(f"❌ Error initializing OCR: {e}")
        return
    
    total_files = len(files_to_process)
    successful_files = 0
    failed_files = 0
    total_pages_processed = 0
    
    print(f"\n🚀 Starting processing of {total_files} PDF files...")
    
    for file_index, pdf_file in enumerate(files_to_process, 1):
        print(f"\n{'='*20} PROCESSING {pdf_file.name} ({file_index}/{total_files}) {'='*20}")
        
        try:
            pages_data, total_pages, pdf_metadata = process_pdf_with_metadata_and_ocr(
                pdf_file, ocr_reader, MAX_PAGES_PER_PDF
            )
            
            pdf_result = {
                "filename": str(pdf_file.name),
                "total_pages_in_pdf": int(total_pages),
                "pages_processed": int(len(pages_data)),
                "metadata": pdf_metadata,
                "pages": pages_data
            }
            
            if save_pdf_result_to_json(pdf_result, output_directory, pdf_file.name):
                successful_files += 1
                total_pages_processed += len(pages_data)
            else:
                failed_files += 1
            
            print(f"{'='*20} FINISHED {pdf_file.name} ({file_index}/{total_files}) {'='*20}")
            
        except Exception as e:
            print(f"❌ ERROR processing {pdf_file.name}: {e}")
            
            error_result = {
                "filename": str(pdf_file.name),
                "error": str(e),
                "total_pages_in_pdf": 0,
                "pages_processed": 0,
                "metadata": {},
                "pages": []
            }
            
            if save_pdf_result_to_json(error_result, output_directory, pdf_file.name):
                print(f"  📝 Error info saved to JSON file")
            
            failed_files += 1
    
    print(f"\n{'='*60}")
    print("COMPREHENSIVE PROCESSING SUMMARY")
    print(f"{'='*60}")
    print(f"📁 Input directory: {pdf_directory}")
    print(f"📁 Output directory: {output_directory}")
    print(f"📄 Total PDFs found: {len(pdf_files)}")
    print(f"📄 PDFs processed: {len(files_to_process)}")
    print(f"✅ Successful: {successful_files}")
    print(f"❌ Failed: {failed_files}")
    print(f"📄 Total pages processed: {total_pages_processed}")
    print(f"💾 Individual JSON files created in: {output_directory}")
    
    # Try cleaning up temp JSON dir if empty
    try:
        if not any(tmp_dir.iterdir()):
            tmp_dir.rmdir()
    except Exception:
        pass
    
    if successful_files > 0:
        print(f"\n📋 Successfully created JSON files:")
        json_files = list(output_directory.glob("*.json"))
        for json_file in sorted(json_files):
            print(f"  • {json_file.name}")
    
    print(f"{'='*60}")


if __name__ == "__main__":
    main()