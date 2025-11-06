import requests
from bs4 import BeautifulSoup
import csv
import re
import time
import os
from pathlib import Path
from dotenv import load_dotenv
import dotenv  # pip install python-dotenv

env_file_path = Path("../.env")
    
# Load the .env file
if not env_file_path.exists():
    raise(ImportError(f"Error: .env file not found at {env_file_path}"))

dotenv.load_dotenv(env_file_path)

# ========== CONFIGURATION ==========
# Set this to a small number for testing (e.g., 5) or None for all pages
PAGE_LIMIT = None  # Change to None to scrape all pages

# Output folder for the CSV file (will be created if it doesn't exist)
OUTPUT_FOLDER = '../dataset/5_amendments/csv'  # Change to your desired folder path

# Output filename
OUTPUT_FILENAME = 'regulations_relationships.csv'
# ===================================

def extract_regulation_number(text):
    """
    Extract regulation number from text like:
    - "Undang-undang (UU) Nomor 1 Tahun 2025" -> "1_2025"
    - "Undang-undang (UU) No. 16 Tahun 2025" -> "16_2025"
    - "UU No. 16 Tahun 2025" -> "16_2025"
    """
    # Pattern for various formats
    patterns = [
        r'Nomor\s+(\d+)\s+Tahun\s+(\d+)',
        r'No\.\s+(\d+)\s+Tahun\s+(\d+)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return f"{match.group(1)}_{match.group(2)}"
    return None

def normalize_relationship(text):
    """
    Convert relationship text to lowercase with underscores
    "Diubah dengan" -> "diubah_dengan"
    """
    return text.strip().lower().replace(' ', '_')

def has_relationship_classes(tag):
    """Check if a tag has the relationship indicator classes"""
    if tag.name != 'div':
        return False
    classes = tag.get('class', [])
    return 'fw-bold' in classes and ('bg-light-primary' in classes or 'bg-light-danger' in classes)

def scrape_page(page_num):
    """
    Scrape a single page and return list of (source, relationship, target) tuples
    """
    url = f"https://peraturan.bpk.go.id/Search?keywords=&tentang=&nomor=&jenis=8&p={page_num}"
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        results = []
        
        # Find all card bodies which contain regulation entries
        card_bodies = soup.find_all('div', class_='card-body')
        
        for card_body in card_bodies:
            # Extract source regulation number from the title
            title_div = card_body.find('div', class_='fw-semibold')
            if not title_div or 'fs-5' not in title_div.get('class', []):
                continue
                
            source_text = title_div.get_text(strip=True)
            source = extract_regulation_number(source_text)
            
            if not source:
                continue
            
            # Check if there's a "Status Peraturan" section
            status_headers = card_body.find_all('div', class_='fw-bold')
            status_section = None
            
            for header in status_headers:
                if 'Status Peraturan' in header.get_text():
                    status_section = header.parent
                    break
            
            if not status_section:
                continue
            
            # Find all relationship type divs within this section
            relationship_divs = status_section.find_all(has_relationship_classes)
            
            for rel_type_div in relationship_divs:
                relationship = normalize_relationship(rel_type_div.get_text(strip=True))
                
                # Find the parent row and then the target column
                row = rel_type_div.parent.parent  # Go up to the row div
                target_col = row.find('div', class_='col-lg-10')
                
                if not target_col:
                    continue
                
                # Find all links within list items
                links = target_col.find_all('a', class_='text-danger')
                
                for link in links:
                    link_text = link.get_text(strip=True)
                    
                    # Only process if starts with "UU "
                    if not link_text.startswith('UU '):
                        continue
                    
                    target = extract_regulation_number(link_text)
                    
                    if target:
                        results.append((source, relationship, target))
                        print(f"    Found: {source} {relationship} {target}")
        
        return results
        
    except Exception as e:
        print(f"Error scraping page {page_num}: {e}")
        import traceback
        traceback.print_exc()
        return []

def main():
    # Create output folder if it doesn't exist
    if OUTPUT_FOLDER:
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)
        output_file = os.path.join(OUTPUT_FOLDER, OUTPUT_FILENAME)
    else:
        output_file = OUTPUT_FILENAME
    
    # Determine how many pages to scrape
    total_pages = 192  # Total pages available
    pages_to_scrape = PAGE_LIMIT if PAGE_LIMIT is not None else total_pages
    
    print(f"{'='*60}")
    print(f"Starting scraper...")
    print(f"Page limit: {pages_to_scrape} of {total_pages} total pages")
    print(f"Output file: {output_file}")
    print(f"{'='*60}\n")
    
    # Open CSV file for writing
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['source', 'relationship', 'target'])
        
        total_relationships = 0
        
        for page_num in range(1, pages_to_scrape + 1):
            print(f"Scraping page {page_num}/{pages_to_scrape}...")
            
            results = scrape_page(page_num)
            
            # Write results to CSV
            for result in results:
                writer.writerow(result)
            
            total_relationships += len(results)
            print(f"  Found {len(results)} relationships on page {page_num}")
            
            # Be respectful - add delay between requests
            time.sleep(2)
    
    print(f"\n{'='*60}")
    print(f"Scraping complete!")
    print(f"Total relationships found: {total_relationships}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()