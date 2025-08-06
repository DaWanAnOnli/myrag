import fitz
import easyocr
import numpy as np
import time
from PIL import Image
import io

print("Initializing EasyOCR...")
reader = easyocr.Reader(['en'])

pdf_path = "C:/Users/Joel Rompis/Documents/Semester 7/TA/GraphRAG/myrag/dataset/pdfs/UU  Nomor 1 Tahun 2006.pdf"
output_file = "extracted_text.txt"

print(f"Processing PDF: {pdf_path}")
doc = fitz.open(pdf_path)
total_pages = len(doc)
print(f"Total pages: {total_pages}")

with open(output_file, 'w', encoding='utf-8') as f:
    for page_num in range(total_pages):
        print(f"\nProcessing page {page_num + 1}/{total_pages}...")
        
        page = doc[page_num]
        
        # Convert to image
        mat = fitz.Matrix(150/72, 150/72)
        pix = page.get_pixmap(matrix=mat)
        img_data = pix.tobytes("png")
        
        # Convert to numpy array
        img = Image.open(io.BytesIO(img_data))
        img_array = np.array(img)
        
        print(f"Page {page_num + 1} image size: {img_array.shape}")
        
        # Run OCR
        start_time = time.time()
        results = reader.readtext(img_array)
        processing_time = time.time() - start_time
        print(f"Page {page_num + 1} OCR took {processing_time:.2f} seconds")
        print(f"Found {len(results)} text elements on page {page_num + 1}")
        
        # Combine all text from this page into one line
        page_text = ""
        for bbox, text, confidence in results:
            page_text += text + " "
        
        # Write the page text followed by a newline
        f.write(page_text.strip() + "\n")
        
        print(f"Page {page_num + 1} text written to file")

doc.close()
print(f"\nAll pages processed. Text saved to: {output_file}")