# pip install paddleocr PyMuPDF
# If GPU PaddlePaddle isn't set up, PaddleOCR will use CPU.

import time
import json
from pathlib import Path
import fitz  # PyMuPDF
from paddleocr import PaddleOCR
import paddle

# -------------------- Configuration --------------------
PDF_PATH = "../dataset/pdfs/samples/UU Nomor 44 Tahun 2007.pdf"

# Limit pages processed from the start; set to None to process all pages.
MAX_PAGES = 5

# Rendering DPI for PDF -> PNG (200–300 is a good range)
RENDER_DPI = 200

# Directory to temporarily store JSON files (will be removed per file after reading)
JSON_TMP_DIRNAME = "_ocr_json_tmp"

# PaddleOCR options (as in your original approach)
OCR_OPTIONS = dict(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
)
# -------------------------------------------------------

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
    base_dir = Path(__file__).parent if "__file__" in globals() else Path.cwd()
    pdf_path = (base_dir / PDF_PATH).resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    output_txt = base_dir / f"{pdf_path.stem}_ocr.txt"
    json_tmp_dir = base_dir / JSON_TMP_DIRNAME
    json_tmp_dir.mkdir(exist_ok=True)

    # Initialize OCR
    t0 = time.perf_counter()
    ocr = PaddleOCR(**OCR_OPTIONS)
    t_init = time.perf_counter() - t0
    print(f"OCR model initialized in {t_init:.2f}s")

    # Open PDF
    doc = fitz.open(pdf_path)
    total_pages = doc.page_count
    pages_to_process = total_pages if MAX_PAGES is None else min(MAX_PAGES, total_pages)
    print(f"Processing {pages_to_process} page(s) from: {pdf_path.name}")

    # Prepare rendering scale
    zoom = RENDER_DPI / 72.0
    mat = fitz.Matrix(zoom, zoom)

    all_pages_text = []
    total_start = time.perf_counter()

    for page_index in range(pages_to_process):
        page_no = page_index + 1
        page = doc.load_page(page_index)

        # Temporary PNG in the same directory as the script
        png_path = base_dir / f"temp_page_{page_no:04d}.png"

        # Render to PNG
        t_render0 = time.perf_counter()
        pix = page.get_pixmap(matrix=mat, alpha=False)
        pix.save(png_path)
        render_time = time.perf_counter() - t_render0

        # OCR via predict(...) then save to JSON
        t_ocr0 = time.perf_counter()
        page_texts = []
        try:
            result = ocr.predict(str(png_path))
            # Save result(s) to JSON. There is typically one result per image,
            # but we'll handle multiple defensively.
            saved_json_paths = []
            for res in result:
                # save_to_json returns nothing; we derive the filename from the PNG stem
                res.save_to_json(str(json_tmp_dir))

                # Heuristic: PaddleOCR uses the input image's base name for the JSON file.
                # We'll look for files starting with the PNG stem.
                stem = png_path.stem  # e.g., "temp_page_0001"
                for candidate in json_tmp_dir.glob(f"{stem}*.json"):
                    saved_json_paths.append(candidate)

            # Deduplicate candidate list in case of overlap
            saved_json_paths = sorted(set(saved_json_paths))

            # Read JSON(s) and collect rec_texts
            for jp in saved_json_paths:
                try:
                    data = json.loads(jp.read_text(encoding="utf-8"))
                    # Primary extraction path
                    rec_texts = data.get("rec_texts", [])
                    if isinstance(rec_texts, list):
                        page_texts.extend([t for t in rec_texts if isinstance(t, str)])
                finally:
                    # Delete JSON after reading
                    try:
                        jp.unlink(missing_ok=True)
                    except Exception as e:
                        print(f"Warning: failed to delete JSON {jp.name}: {e}")

        finally:
            # Delete the PNG no matter what
            try:
                png_path.unlink(missing_ok=True)
            except Exception as e:
                print(f"Warning: failed to delete temp PNG {png_path.name}: {e}")

        ocr_time = time.perf_counter() - t_ocr0

        # Accumulate per-page text (join lines with newlines)
        if page_texts:
            all_pages_text.append("\n".join(page_texts))

        print(f"Page {page_no}/{pages_to_process} -> render {render_time:.2f}s, "
              f"OCR {ocr_time:.2f}s, lines {len(page_texts)}")

    # Write concatenated text for all pages
    final_text = "\n\n".join(all_pages_text)
    output_txt.write_text(final_text, encoding="utf-8")

    total_time = time.perf_counter() - total_start
    print(f"Done. Wrote {len(final_text):,} characters to {output_txt.name}")
    print(f"Total processing time (excluding model init): {total_time:.2f}s")

    # Optionally remove the temp JSON directory if empty
    try:
        if not any(json_tmp_dir.iterdir()):
            json_tmp_dir.rmdir()
    except Exception:
        pass


if __name__ == "__main__":
    main()