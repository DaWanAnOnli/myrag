import os
import json
from pathlib import Path
import fitz  # PyMuPDF
import easyocr
from PIL import Image
import io
import numpy as np
from PyPDF2 import PdfReader

def setup_ocr_reader():
    """Initialize EasyOCR reader"""
    return easyocr.Reader(['en'])

def extract_pdf_metadata(pdf_path):
    """Extract metadata from PDF file"""
    try:
        print(f"  Extracting metadata from {os.path.basename(pdf_path)}...")
        
        with open(pdf_path, 'rb') as file:
            reader = PdfReader(file)
            metadata = reader.metadata
            
            if metadata:
                # Convert metadata to a clean dictionary
                clean_metadata = {}
                
                for key, value in metadata.items():
                    # Handle custom metadata
                    if key.startswith('/Custom_'):
                        column_name = key.replace('/Custom_', '')
                        clean_metadata[column_name] = str(value) if value else ""
                        print(f"    Custom: {column_name} = {value}")
                    else:
                        # Handle standard metadata (remove leading slash)
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
    """Extract a specific image region from the page"""
    try:
        # Create a clip rectangle for this image region
        clip_rect = fitz.Rect(bbox)
        
        # Render only this specific region at high resolution
        mat = fitz.Matrix(2.0, 2.0)  # 2x resolution for better OCR
        pix = page.get_pixmap(matrix=mat, clip=clip_rect)
        
        # Convert to PIL Image
        img_data = pix.tobytes("png")
        pil_image = Image.open(io.BytesIO(img_data))
        
        print(f"      Extracted region {block_num}: {pil_image.size}")
        
        pix = None
        return pil_image
        
    except Exception as e:
        print(f"      Error extracting image region {block_num}: {e}")
        return None

def perform_ocr_on_image_region(pil_image, ocr_reader, page_num, block_num):
    """Perform OCR on a specific image region"""
    try:
        # Convert to numpy array for EasyOCR
        img_array = np.array(pil_image)
        
        # Skip very small images
        if img_array.shape[0] < 50 or img_array.shape[1] < 50:
            print(f"      Skipping small image region {block_num}: {img_array.shape}")
            return ""
        
        print(f"      Performing OCR on region {block_num} ({img_array.shape})...")
        
        # Perform OCR
        results = ocr_reader.readtext(img_array)
        
        # Extract text with confidence filtering
        region_texts = []
        for bbox, text, confidence in results:
            if confidence > 0.5:  # High confidence threshold for image regions
                region_texts.append(text)
                print(f"        Found: '{text[:50]}...' (conf: {confidence:.2f})")
        
        region_ocr = ' '.join(region_texts).strip()
        print(f"      Region {block_num} OCR result: {len(region_ocr)} characters")
        
        return region_ocr
        
    except Exception as e:
        print(f"      Error performing OCR on region {block_num}: {e}")
        return ""

def extract_text_from_image_blocks_only(page, page_num, ocr_reader):
    """Extract text only from image blocks on the page"""
    try:
        print(f"    Extracting text from image blocks on page {page_num}...")
        
        # Get image block information
        image_blocks = get_image_blocks_info(page, page_num)
        
        if not image_blocks:
            print(f"    No image blocks found on page {page_num}")
            return ""
        
        # Process each image block
        all_ocr_texts = []
        
        for i, img_block in enumerate(image_blocks):
            block_num = i + 1
            bbox = img_block["bbox"]
            
            # Extract the image region
            region_image = extract_image_region(page, bbox, page_num, block_num)
            
            if region_image is not None:
                # Perform OCR on this specific region
                region_text = perform_ocr_on_image_region(region_image, ocr_reader, page_num, block_num)
                
                if region_text:
                    all_ocr_texts.append(region_text)
        
        # Combine all OCR text from image blocks
        final_ocr_text = ' '.join(all_ocr_texts).strip()
        print(f"    Total OCR from {len(image_blocks)} image blocks: {len(final_ocr_text)} characters")
        
        return final_ocr_text
        
    except Exception as e:
        print(f"    Error extracting text from image blocks on page {page_num}: {e}")
        return ""

def perform_full_page_ocr(page, page_num, ocr_reader):
    """Perform OCR on the entire page (for scanned documents)"""
    try:
        print(f"    Performing full-page OCR on page {page_num}...")
        
        # Render the entire page at high resolution
        mat = fitz.Matrix(2.0, 2.0)
        pix = page.get_pixmap(matrix=mat)
        
        img_data = pix.tobytes("png")
        pil_image = Image.open(io.BytesIO(img_data))
        img_array = np.array(pil_image)
        
        # Perform OCR on full page
        results = ocr_reader.readtext(img_array)
        
        # Extract text with moderate confidence for scanned pages
        ocr_texts = []
        for bbox, text, confidence in results:
            if confidence > 0.3:  # Lower threshold for scanned content
                ocr_texts.append(text)
        
        full_page_ocr = ' '.join(ocr_texts).strip()
        
        pix = None
        
        print(f"    Full-page OCR result: {len(full_page_ocr)} characters")
        return full_page_ocr
        
    except Exception as e:
        print(f"    Error performing full-page OCR on page {page_num}: {e}")
        return ""

def analyze_page_content(page, page_num):
    """Analyze what type of content this page has"""
    try:
        # Get regular text
        regular_text = page.get_text().strip()
        text_length = len(regular_text)
        
        # Check for image blocks
        blocks = page.get_text("dict")["blocks"]
        image_blocks = [b for b in blocks if b.get("type") == 1]
        
        # Analyze visual complexity for scanned page detection
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
    
    # Strategy 1: Page has image blocks - OCR only the image blocks
    if page_analysis["has_image_blocks"]:
        print(f"    Page {page_num}: OCR image blocks only - {page_analysis['image_blocks_count']} blocks found")
        return "image_blocks_only", "has_image_blocks"
    
    # Strategy 2: Page appears to be scanned - full page OCR
    if page_analysis["is_likely_scanned"]:
        print(f"    Page {page_num}: Full-page OCR - appears to be scanned")
        return "full_page_ocr", "likely_scanned"
    
    # Strategy 3: Regular text page - no OCR needed
    print(f"    Page {page_num}: No OCR needed - regular text page")
    return "no_ocr", "regular_text_page"

def process_single_page_isolated(doc, page_index, ocr_reader):
    """Process a single page with targeted OCR strategy"""
    page_num = page_index + 1
    print(f"\n  ========== PAGE {page_num} (index {page_index}) ==========")
    
    try:
        # Load this specific page
        page = doc.load_page(page_index)
        
        # Step 1: Extract regular text
        regular_text = extract_text_from_page_only(page, page_num)
        
        # Step 2: Analyze page content
        page_analysis = analyze_page_content(page, page_num)
        
        # Step 3: Decide OCR strategy
        ocr_strategy, ocr_reason = decide_ocr_strategy(page_analysis, page_num)
        
        # Step 4: Apply OCR strategy
        ocr_text = ""
        if ocr_strategy == "image_blocks_only":
            ocr_text = extract_text_from_image_blocks_only(page, page_num, ocr_reader)
        elif ocr_strategy == "full_page_ocr":
            ocr_text = perform_full_page_ocr(page, page_num, ocr_reader)
        else:
            print(f"    Page {page_num}: Skipping OCR ({ocr_reason})")
        
        # Step 5: Create result
        result = {
            "page_number": int(page_num),
            "page_index": int(page_index),
            "text": str(regular_text),
            "ocr": str(ocr_text),
            "ocr_strategy": str(ocr_strategy),
            "ocr_reason": str(ocr_reason),
            "has_image_blocks": bool(page_analysis["has_image_blocks"]),
            "image_blocks_count": int(page_analysis["image_blocks_count"]),
            "visual_complexity": float(page_analysis["visual_complexity"]),
            "text_length": int(len(regular_text)),
            "ocr_length": int(len(ocr_text))
        }
        
        print(f"    Page {page_num} FINAL:")
        print(f"      Regular text: {len(regular_text)} chars")
        print(f"      OCR strategy: {ocr_strategy}")
        print(f"      OCR text: {len(ocr_text)} chars")
        
        # Clear page reference
        page = None
        return result
        
    except Exception as e:
        print(f"    ERROR processing page {page_num}: {e}")
        return {
            "page_number": int(page_num),
            "page_index": int(page_index),
            "text": "",
            "ocr": "",
            "ocr_strategy": "error",
            "ocr_reason": "error",
            "has_image_blocks": False,
            "image_blocks_count": 0,
            "visual_complexity": 0.0,
            "text_length": 0,
            "ocr_length": 0,
            "error": str(e)
        }

def process_pdf_with_metadata_and_ocr(pdf_path, ocr_reader, max_pages=None):
    """Process PDF with metadata extraction and targeted OCR per page"""
    try:
        print(f"\n{'='*60}")
        print(f"PROCESSING PDF: {os.path.basename(pdf_path)}")
        print(f"{'='*60}")
        
        # Step 1: Extract PDF metadata first
        pdf_metadata = extract_pdf_metadata(pdf_path)
        
        # Step 2: Open document for page processing
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        
        pages_to_process = total_pages if max_pages is None else min(max_pages, total_pages)
        
        print(f"\nDocument has {total_pages} total pages")
        if max_pages is None:
            print(f"Processing ALL {pages_to_process} pages")
        else:
            print(f"Processing first {pages_to_process} pages (limited)")
        
        # Step 3: Process each page
        pages_data = []
        
        for page_index in range(pages_to_process):
            page_result = process_single_page_isolated(doc, page_index, ocr_reader)
            pages_data.append(page_result)
            
            # Progress indicator for large documents
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
    # Remove .pdf extension and add .json
    base_name = pdf_filename.replace('.pdf', '').replace('.PDF', '')
    return f"{base_name}.json"

def save_pdf_result_to_json(pdf_result, output_dir, pdf_filename):
    """Save a single PDF's result to its own JSON file"""
    try:
        # Create the JSON filename
        json_filename = create_json_filename(pdf_filename)
        json_path = output_dir / json_filename
        
        print(f"  Saving results to: {json_path}")
        
        # Save the result
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(pdf_result, f, indent=2, ensure_ascii=False)
        
        print(f"  ✅ Successfully saved: {json_filename}")
        return True
        
    except Exception as e:
        print(f"  ❌ Error saving {pdf_filename} results: {e}")
        return False

def main():
    # ======================
    # PROCESSING CONFIGURATION
    # ======================
    
    # Set these to None for unlimited processing, or specific numbers for limits
    MAX_PDF_FILES = None  # Set to None to process ALL PDFs, or a number to limit
    MAX_PAGES_PER_PDF = None  # Set to None to process ALL pages, or a number to limit
    
    # Alternative: Use these for testing with limits
    # MAX_PDF_FILES = 2
    # MAX_PAGES_PER_PDF = 5
    
    print("=" * 60)
    print("PDF OCR WITH METADATA EXTRACTION - INDIVIDUAL JSON FILES")
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
    
    # Setup directories
    pdf_directory = Path("../dataset/pdfs-with-metadata/samples")
    output_directory = Path("../dataset/extract-results/samples")
    
    # Check input directory
    if not pdf_directory.exists():
        print(f"❌ ERROR: Input directory {pdf_directory} does not exist!")
        return
    
    # Create output directory if it doesn't exist
    try:
        output_directory.mkdir(parents=True, exist_ok=True)
        print(f"📁 Output directory: {output_directory}")
    except Exception as e:
        print(f"❌ ERROR: Could not create output directory {output_directory}: {e}")
        return
    
    # Get PDF files
    pdf_files = list(pdf_directory.glob("*.pdf"))
    if not pdf_files:
        print("❌ ERROR: No PDF files found in input directory!")
        return
    
    # Determine files to process
    if MAX_PDF_FILES is None:
        files_to_process = pdf_files
        print(f"📄 Found {len(pdf_files)} PDFs - processing ALL of them")
    else:
        files_to_process = pdf_files[:MAX_PDF_FILES]
        print(f"📄 Found {len(pdf_files)} PDFs - processing first {len(files_to_process)}")
    
    print(f"📋 Files to process: {[f.name for f in files_to_process]}")
    
    # Initialize OCR
    print("\n🔧 Initializing OCR...")
    try:
        ocr_reader = setup_ocr_reader()
        print("✅ OCR ready!")
    except Exception as e:
        print(f"❌ Error initializing OCR: {e}")
        return
    
    # Process files
    total_files = len(files_to_process)
    successful_files = 0
    failed_files = 0
    total_pages_processed = 0
    
    print(f"\n🚀 Starting processing of {total_files} PDF files...")
    
    for file_index, pdf_file in enumerate(files_to_process, 1):
        print(f"\n{'='*20} PROCESSING {pdf_file.name} ({file_index}/{total_files}) {'='*20}")
        
        try:
            # Process the PDF
            pages_data, total_pages, pdf_metadata = process_pdf_with_metadata_and_ocr(
                pdf_file, ocr_reader, MAX_PAGES_PER_PDF
            )
            
            # Create result structure for this PDF
            pdf_result = {
                "filename": str(pdf_file.name),
                "total_pages_in_pdf": int(total_pages),
                "pages_processed": int(len(pages_data)),
                "metadata": pdf_metadata,
                "pages": pages_data
            }
            
            # Save to individual JSON file
            if save_pdf_result_to_json(pdf_result, output_directory, pdf_file.name):
                successful_files += 1
                total_pages_processed += len(pages_data)
            else:
                failed_files += 1
            
            print(f"{'='*20} FINISHED {pdf_file.name} ({file_index}/{total_files}) {'='*20}")
            
        except Exception as e:
            print(f"❌ ERROR processing {pdf_file.name}: {e}")
            
            # Still try to save error info
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
    
    # Print comprehensive summary
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
    
    if successful_files > 0:
        print(f"\n📋 Successfully created JSON files:")
        json_files = list(output_directory.glob("*.json"))
        for json_file in sorted(json_files):
            print(f"  • {json_file.name}")
    
    print(f"{'='*60}")

if __name__ == "__main__":
    main()