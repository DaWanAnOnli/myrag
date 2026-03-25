import os
from pathlib import Path

# Paths based on project structure
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
PDF_DIR = PROJECT_DIR / "dataset" / "1_pdfs_with_metadata"
JSON_DIR = PROJECT_DIR / "dataset" / "2_extract_text_results"

def find_missing_jsons():
    print(f"Checking for missing JSON files...")
    print(f"PDF Directory: {PDF_DIR}")
    print(f"JSON Directory: {JSON_DIR}\n")

    if not PDF_DIR.exists():
        print(f"Error: PDF directory not found at {PDF_DIR}")
        return
    if not JSON_DIR.exists():
        print(f"Error: JSON directory not found at {JSON_DIR}")
        return

    # Get all PDF filenames without extension (case-insensitive)
    pdf_files = {f.stem for f in PDF_DIR.iterdir() if f.is_file() and f.suffix.lower() == '.pdf'}
    
    # Get all JSON filenames without extension (case-insensitive)
    json_files = {f.stem for f in JSON_DIR.iterdir() if f.is_file() and f.suffix.lower() == '.json'}
    
    # Find missing JSONs
    missing_jsons = pdf_files - json_files
    
    # Check for JSONs that don't have matching PDFs (just in case)
    orphaned_jsons = json_files - pdf_files
    
    print(f"Total PDFs found: {len(pdf_files)}")
    print(f"Total JSONs found: {len(json_files)}")
    print("-" * 40)
    
    if missing_jsons:
        print(f"Found {len(missing_jsons)} missing JSON files (PDF exists but no JSON):")
        for missing in sorted(missing_jsons):
            print(f"- {missing}.pdf")
    else:
        print("All PDFs have corresponding JSON files.")
        
    if orphaned_jsons:
        print("\nNote: Found JSON files without matching PDFs ('Orphaned' JSONs):")
        for orphaned in sorted(orphaned_jsons):
            print(f"- {orphaned}.json")

if __name__ == "__main__":
    find_missing_jsons()
