import os
import glob
import json

def verify_jsons(result_dir):
    json_files = glob.glob(os.path.join(result_dir, "*.json"))
    
    total_files = len(json_files)
    total_pages = 0
    mismatches = 0
    
    for filepath in json_files:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            continue
            
        pages = data.get("pages", [])
        total_pages += len(pages)
        
        for page in pages:
            source = page.get("final_text_source", "regular")
            
            # Using get() with default empty string in case the key is missing or None
            text_field = page.get("text") or ""
            ocr_field = page.get("ocr") or ""
            final_text_field = page.get("final_text") or ""
            
            len_text = len(text_field)
            len_ocr = len(ocr_field)
            len_final = len(final_text_field)
            
            if source == "regular":
                # As per description, final_text should match text
                expected_len = len_text
            elif source == "ocr":
                # final_text should match ocr
                expected_len = len_ocr
            elif source == "both":
                # Assuming simple concatenation or joining with newlines
                text_content = text_field.strip()
                ocr_content = ocr_field.strip()
                
                parts = []
                if text_content: parts.append(text_content)
                if ocr_content: parts.append(ocr_content)
                
                expected_len = len("\n\n".join(parts))
                
                # Check for naive combination (just in case) if first check fails
                naive_len = len_text + len_ocr
            else:
                print(f"Warning: Unknown final_text_source '{source}' in {os.path.basename(filepath)} - Page {page.get('page_number')}")
                continue
                
            # Perform the mismatch check
            is_mismatch = False
            
            if source in ["regular", "ocr"]:
                if len_final != expected_len and len_final != len(getattr(page, source, "").strip()):
                    # Sometimes final_text has trailing/leading whitespace stripped.
                    # We will flag as mismatch if even the stripped lengths don't match.
                    is_mismatch = True
            elif source == "both":
                if len_final != expected_len and len_final not in [naive_len, naive_len + 1, naive_len + 2]:
                    is_mismatch = True

            if is_mismatch:
                mismatches += 1
                print(f"Mismatch in {os.path.basename(filepath)} - Page {page.get('page_number')}:")
                print(f"  Source: {source}")
                print(f"  len(text): {len_text}, len(ocr): {len_ocr}")
                print(f"  Expected len: {expected_len}, Actual len(final_text): {len_final}")

    print("\n" + "="*40)
    print(" "*10 + "Verification Summary")
    print("="*40)
    print(f"Total files checked : {total_files}")
    print(f"Total pages checked : {total_pages}")
    print(f"Total mismatches    : {mismatches}")
    if mismatches == 0:
        print("✅ All pages have matching field lengths!")
    else:
        print("❌ Some pages contain mismatched field lengths.")

if __name__ == "__main__":
    result_directory = "../dataset/2_extract_text_results"
    
    if os.path.exists(result_directory):
        print(f"Verifying generated JSONs in {result_directory}...\n")
        verify_jsons(result_directory)
    else:
        print(f"Error: Directory {result_directory} does not exist. Please check the path.")
