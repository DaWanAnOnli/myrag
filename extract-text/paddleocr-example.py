from paddleocr import TextRecognition, PaddleOCR
import cv2
import fitz  # PyMuPDF
import os
import time

# Convert first page of PDF to temporary image file in same directory
def pdf_to_temp_image(pdf_path, page_num=0, dpi=200):
    """Convert specific page of PDF to temporary image file"""
    doc = fitz.open(pdf_path)
    page = doc[page_num]  # Get first page (0-indexed)
    
    # Convert page to image with RGB format
    mat = fitz.Matrix(dpi/72, dpi/72)  # Scale factor for DPI
    pix = page.get_pixmap(matrix=mat, alpha=False)  # alpha=False ensures RGB format
    
    # Create temporary file in same directory
    timestamp = str(int(time.time() * 1000))  # millisecond timestamp
    temp_filename = f"temp_page_{timestamp}.png"
    
    pix.save(temp_filename)
    
    doc.close()
    return temp_filename

# Hardcoded input file path
pdf_path = "../dataset/pdfs/samples/UU Nomor 44 Tahun 2007.pdf"  # Replace with your PDF file path
output_text_file = "extracted_text.txt"

temp_image_path = None

try:
    # Convert first page of PDF to temporary image file
    temp_image_path = pdf_to_temp_image(pdf_path)

    ocr = PaddleOCR(
        use_doc_orientation_classify=False, 
        use_doc_unwarping=False, 
        use_textline_orientation=False) # text detection + text recognition
    # ocr = PaddleOCR(use_doc_orientation_classify=True, use_doc_unwarping=True) # text image preprocessing + text detection + textline orientation classification + text recognition
    # ocr = PaddleOCR(use_doc_orientation_classify=False, use_doc_unwarping=False) # text detection + textline orientation classification + text recognition
    # ocr = PaddleOCR(
    #     text_detection_model_name="PP-OCRv5_mobile_det",
    #     text_recognition_model_name="PP-OCRv5_mobile_rec",
    #     use_doc_orientation_classify=False,
    #     use_doc_unwarping=False,
    #     use_textline_orientation=False) # Switch to PP-OCRv5_mobile models
    result = ocr.predict(temp_image_path)
    for res in result:
        res.print()
        res.save_to_img("output")
        res.save_to_json("output")

    
    # # Initialize PaddleOCR text recognition model
    # model = TextRecognition(model_name="PP-OCRv5_server_rec")
    
    # # Perform OCR on the image file
    # output = model.predict(input=temp_image_path, batch_size=1)

    # for res in output:
    #     res.print()
    
    # # Extract text and save to file
    # with open(output_text_file, 'w', encoding='utf-8') as f:
    #     for res in output:
    #         # Extract text from the result
    #         text = res.text if hasattr(res, 'text') else str(res)
    #         f.write(text + '\n')
    
    # print(f"Text extracted and saved to {output_text_file}")
    
except Exception as e:
    print(f"Error: {e}")

# finally:
#     # Clean up temporary file
#     if temp_image_path and os.path.exists(temp_image_path):
#         try:
#             os.remove(temp_image_path)
#         except PermissionError:
#             print(f"Could not delete temporary file: {temp_image_path}")