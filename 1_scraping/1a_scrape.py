import os
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import re
from urllib.parse import urljoin, urlparse
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BPKScraper:
    def __init__(self, base_url="https://peraturan.bpk.go.id"):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
    def get_page(self, url, max_retries=3):
        """Get page content with retry logic"""
        for attempt in range(max_retries):
            try:
                response = self.session.get(url, timeout=30)
                response.raise_for_status()
                return response
            except requests.RequestException as e:
                logger.warning(f"Attempt {attempt + 1} failed for {url}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(5)
                else:
                    logger.error(f"Failed to get {url} after {max_retries} attempts")
                    return None
    
    def extract_list_links(self, list_url):
        """Extract main detail page links from the search results"""
        logger.info(f"Extracting links from: {list_url}")
        
        response = self.get_page(list_url)
        if not response:
            return []
        
        soup = BeautifulSoup(response.content, 'html.parser')
        links = []
        
        # Target the specific structure: div with class "col-lg-10 fs-2 fw-bold pe-4" containing the main links
        main_link_containers = soup.find_all('div', class_='col-lg-10 fs-2 fw-bold pe-4')
        
        for container in main_link_containers:
            link = container.find('a', href=True)
            if link:
                href = link.get('href')
                if href and '/Details/' in href:
                    full_url = urljoin(self.base_url, href)
                    if full_url not in links:
                        links.append(full_url)
                        logger.info(f"Found link: {href}")
        
        logger.info(f"Found {len(links)} unique detail links")
        return links
    
    def extract_metadata(self, soup):
        """Extract metadata from the specific div structure"""
        metadata = {}
        
        # Find the container with metadata
        metadata_container = soup.find('div', class_='container fs-6')
        
        if metadata_container:
            # Find all rows with the metadata structure
            rows = metadata_container.find_all('div', class_='row g-4 g-xl-9 ms-2')
            
            for row in rows:
                # Get the label (col-lg-3 fw-bold)
                label_elem = row.find('div', class_='col-lg-3 fw-bold')
                # Get the value (col-lg-9)
                value_elem = row.find('div', class_='col-lg-9')
                
                if label_elem and value_elem:
                    label = label_elem.get_text(strip=True)
                    value = value_elem.get_text(strip=True)
                    
                    if label and value:
                        # Clean up the label for use as dictionary key
                        clean_label = re.sub(r'[^\w\s]', '', label).strip().replace(' ', '_').lower()
                        metadata[clean_label] = value
        
        return metadata
    
    def extract_abstract_and_notes(self, soup):
        """Extract abstract and notes with enhanced debugging and multiple approaches"""
        abstract = 'N/A'
        notes = 'N/A'
        
        logger.info("=== ABSTRACT EXTRACTION DEBUG ===")
        
        # Step 1: Find the modal with id="abstrak"
        abstrak_modal = soup.find('div', id='abstrak')
        
        if abstrak_modal:
            logger.info("✓ Found modal with id='abstrak'")
            
            # Step 2: Find modal-content within the abstrak modal
            modal_content = abstrak_modal.find('div', class_='modal-content')
            
            if modal_content:
                logger.info("✓ Found modal-content within abstrak modal")
                
                # Step 3: Find modal-body
                modal_body = modal_content.find('div', class_='modal-body')
                
                if modal_body:
                    logger.info("✓ Found modal-body")
                    logger.info(f"Modal-body content preview: {str(modal_body)[:200]}...")
                    
                    # Step 4: Find the table within modal-body
                    table = modal_body.find('table')
                    
                    if table:
                        logger.info("✓ Found table within modal-body")
                        logger.info(f"Table content preview: {str(table)[:300]}...")
                        
                        # Let's see what's directly in the table
                        table_children = table.find_all(recursive=False)
                        logger.info(f"Direct table children: {[child.name for child in table_children if child.name]}")
                        
                        # Try multiple approaches to find rows
                        rows = []
                        
                        # Approach 1: Try to find tbody first
                        tbody = table.find('tbody')
                        if tbody:
                            logger.info("✓ Found tbody")
                            rows = tbody.find_all('tr')
                            logger.info(f"✓ Found {len(rows)} rows in tbody")
                        else:
                            logger.info("✗ No tbody found, trying direct tr search")
                            
                            # Approach 2: Look for tr elements directly in table
                            rows = table.find_all('tr')
                            logger.info(f"Found {len(rows)} tr elements directly in table")
                        
                        # Approach 3: If still no rows, look for any tr elements in the modal
                        if not rows:
                            logger.info("No rows found in table, searching entire modal for tr elements")
                            rows = modal_body.find_all('tr')
                            logger.info(f"Found {len(rows)} tr elements in entire modal body")
                        
                        # Process the rows
                        if rows:
                            logger.info(f"Processing {len(rows)} rows")
                            
                            for i, row in enumerate(rows):
                                cells = row.find_all('td')
                                logger.info(f"Row {i}: found {len(cells)} cells")
                                
                                if len(cells) >= 2:
                                    first_cell_text = cells[0].get_text(strip=True)
                                    logger.info(f"Row {i} first cell text: '{first_cell_text}'")
                                    
                                    # Check if this row contains abstract
                                    if first_cell_text == 'ABSTRAK:':
                                        logger.info(f"✓ Found ABSTRAK row at index {i}")
                                        abstract_cell = cells[1]
                                        
                                        # Extract text from ul.dash
                                        dash_list = abstract_cell.find('ul', class_='dash')
                                        if dash_list:
                                            logger.info("✓ Found ul.dash in abstract cell")
                                            abstract_items = []
                                            lis = dash_list.find_all('li')
                                            logger.info(f"✓ Found {len(lis)} li elements")
                                            
                                            for j, li in enumerate(lis):
                                                item_text = li.get_text(strip=True)
                                                if item_text:
                                                    abstract_items.append(item_text)
                                                    logger.info(f"✓ Added abstract item {j+1}: {item_text[:50]}...")
                                            
                                            if abstract_items:
                                                abstract = ' | '.join(abstract_items)
                                                logger.info(f"✓ Successfully extracted abstract with {len(abstract_items)} items")
                                        else:
                                            logger.warning("✗ No ul.dash found in abstract cell")
                                            # If no ul.dash, get all text from the cell
                                            abstract_text = abstract_cell.get_text(strip=True)
                                            if abstract_text:
                                                abstract = abstract_text
                                                logger.info("✓ Extracted abstract without ul.dash")
                                    
                                    # Check if this row contains notes/catatan
                                    elif first_cell_text == 'CATATAN:':
                                        logger.info(f"✓ Found CATATAN row at index {i}")
                                        notes_cell = cells[1]
                                        
                                        # Extract text from ul.dash
                                        dash_list = notes_cell.find('ul', class_='dash')
                                        if dash_list:
                                            logger.info("✓ Found ul.dash in notes cell")
                                            notes_items = []
                                            for li in dash_list.find_all('li'):
                                                item_text = li.get_text(strip=True)
                                                if item_text:
                                                    notes_items.append(item_text)
                                            
                                            if notes_items:
                                                notes = ' | '.join(notes_items)
                                                logger.info(f"✓ Successfully extracted notes with {len(notes_items)} items")
                                        else:
                                            logger.warning("✗ No ul.dash found in notes cell")
                                            notes_text = notes_cell.get_text(strip=True)
                                            if notes_text:
                                                notes = notes_text
                                                logger.info("✓ Extracted notes without ul.dash")
                                elif len(cells) == 1:
                                    # Single cell, might be header
                                    cell_text = cells[0].get_text(strip=True)
                                    logger.info(f"Row {i} single cell: '{cell_text}'")
                                else:
                                    logger.info(f"Row {i} has {len(cells)} cells (expected >= 2)")
                        else:
                            logger.error("✗ No rows found in table using any method")
                            
                            # Last resort: print the entire table HTML for debugging
                            logger.info("=== TABLE HTML DUMP ===")
                            logger.info(str(table))
                            logger.info("=== END TABLE HTML DUMP ===")
                    else:
                        logger.error("✗ No table found in modal-body")
                        # Let's see what's in modal-body
                        logger.info(f"Modal-body contents: {[child.name for child in modal_body.children if hasattr(child, 'name')]}")
                else:
                    logger.error("✗ No modal-body found in modal-content")
            else:
                logger.error("✗ No modal-content found in abstrak modal")
        else:
            logger.error("✗ No modal with id='abstrak' found")
            
            # Debug: Let's see what modals are available
            all_modals = soup.find_all('div', class_='modal')
            logger.info(f"Found {len(all_modals)} modals total")
            for modal in all_modals:
                modal_id = modal.get('id', 'no-id')
                modal_classes = modal.get('class', [])
                logger.info(f"  Modal ID: {modal_id}, Classes: {modal_classes}")
        
        logger.info(f"Final results - Abstract: {'Found' if abstract != 'N/A' else 'Not found'}, Notes: {'Found' if notes != 'N/A' else 'Not found'}")
        logger.info("=== END ABSTRACT EXTRACTION DEBUG ===")
        
        return abstract, notes
    
    def extract_download_links(self, soup):
        """Extract download links from the page"""
        download_links = []
        
        # Look for download links with the specific pattern from the HTML
        download_selectors = [
            'a.download-file',
            'a[href*="/Download/"]',
            'a[href*=".pdf"]',
            'a[href*=".doc"]',
            'a[href*="download"]',
            'a:contains("Download")',
            'a:contains("Unduh")'
        ]
        
        for selector in download_selectors:
            try:
                links = soup.select(selector)
                for link in links:
                    href = link.get('href')
                    if href:
                        full_link = urljoin(self.base_url, href)
                        if full_link not in download_links:
                            download_links.append(full_link)
            except Exception as e:
                logger.debug(f"Error with selector {selector}: {e}")
                continue
        
        return download_links
    
    def extract_detail_data(self, detail_url):
        """Extract data from a detail page"""
        logger.info(f"Scraping: {detail_url}")
        
        response = self.get_page(detail_url)
        if not response:
            return None
        
        soup = BeautifulSoup(response.content, 'html.parser')
        data = {'URL': detail_url}
        
        try:
            # Extract UU number from the header section
            uu_number_elem = soup.find('h4', class_='mb-8 text-white opacity-50')
            data['UU_Number'] = uu_number_elem.get_text(strip=True) if uu_number_elem else 'N/A'
            
            # Extract title from the h1 in the header section
            title_elem = soup.find('h1', class_='text-white')
            if not title_elem:
                # Fallback to any h1
                title_elem = soup.find('h1')
            
            data['Title'] = title_elem.get_text(strip=True) if title_elem else 'N/A'
            
            # Extract metadata using the specific structure
            metadata = self.extract_metadata(soup)
            
            # Add metadata to data
            data['Tipe_Dokumen'] = metadata.get('tipe_dokumen', 'N/A')
            data['Nomor'] = metadata.get('nomor', 'N/A')
            data['Bentuk'] = metadata.get('bentuk', 'N/A')
            data['Bentuk_Singkat'] = metadata.get('bentuk_singkat', 'N/A')
            data['Tahun'] = metadata.get('tahun', 'N/A')
            data['Tempat_Penetapan'] = metadata.get('tempat_penetapan', 'N/A')
            data['Tanggal_Penetapan'] = metadata.get('tanggal_penetapan', 'N/A')
            data['Tanggal_Pengundangan'] = metadata.get('tanggal_pengundangan', 'N/A')
            data['Tanggal_Berlaku'] = metadata.get('tanggal_berlaku', 'N/A')
            data['Sumber'] = metadata.get('sumber', 'N/A')
            data['Subjek'] = metadata.get('subjek', 'N/A')
            data['Status'] = metadata.get('status', 'N/A')
            data['Bahasa'] = metadata.get('bahasa', 'N/A')
            data['Lokasi'] = metadata.get('lokasi', 'N/A')
            data['Bidang'] = metadata.get('bidang', 'N/A')
            data['TEU'] = metadata.get('teu', 'N/A')
            
            # Extract abstract and notes
            abstract, notes = self.extract_abstract_and_notes(soup)
            data['Abstract'] = abstract
            data['Notes'] = notes
            
            # Extract download links
            download_links = self.extract_download_links(soup)
            data['Download_Links'] = '; '.join(download_links) if download_links else 'N/A'
            
            # Extract access count if available
            access_count_elem = soup.find(string=re.compile(r'Halaman ini telah diakses \d+ kali'))
            if access_count_elem:
                access_match = re.search(r'(\d+)', access_count_elem)
                data['Access_Count'] = access_match.group(1) if access_match else 'N/A'
            else:
                data['Access_Count'] = 'N/A'
            
            logger.info(f"Successfully extracted data from: {detail_url}")
            return data
            
        except Exception as e:
            logger.error(f"Error extracting data from {detail_url}: {e}")
            return None
    
    def scrape_all_pages(self, start_url, max_pages=None):
        """Scrape all pages starting from the given URL"""
        all_data = []
        page_num = 1
        
        while True:
            if max_pages and page_num > max_pages:
                break
                
            # Construct page URL
            current_url = re.sub(r'&p=\d+', f'&p={page_num}', start_url)
            if '&p=' not in current_url:
                current_url += f'&p={page_num}'
            
            logger.info(f"Processing page {page_num}: {current_url}")
            
            # Get links from current page
            detail_links = self.extract_list_links(current_url)
            
            if not detail_links:
                logger.info(f"No more links found on page {page_num}. Stopping.")
                break
            
            # Scrape each detail page
            for link in detail_links:
                data = self.extract_detail_data(link)
                if data:
                    all_data.append(data)
                
                # Be respectful - add delay between requests
                time.sleep(2)
            
            page_num += 1
            
            # Add delay between pages
            time.sleep(3)
        
        return all_data
    
    def save_to_csv(self, data, filename='bpk_undang_undang_data.csv'):
        """Save scraped data to CSV file with proper headers"""
        if not data:
            logger.warning("No data to save")
            return
        
        # Create the dataset directory path
        dataset_dir = os.path.join('..', 'dataset')
        os.makedirs(dataset_dir, exist_ok=True)
        
        # Create full file path
        filepath = os.path.join(dataset_dir, filename)
        
        # Define the column order with proper headers (UU_Number first)
        column_headers = [
            'UU_Number',  # Added as first column
            'Title',
            'Abstract', 
            'Notes',
            'Tipe_Dokumen',
            'Nomor',
            'Bentuk',
            'Bentuk_Singkat', 
            'Tahun',
            'Tempat_Penetapan',
            'Tanggal_Penetapan',
            'Tanggal_Pengundangan',
            'Tanggal_Berlaku',
            'Sumber',
            'Subjek',
            'Status',
            'Bahasa',
            'Lokasi',
            'Bidang',
            'TEU',
            'Download_Links',
            'Access_Count',
            'URL'
        ]
        
        df = pd.DataFrame(data)
        
        # Ensure all columns exist, add missing ones with 'N/A'
        for col in column_headers:
            if col not in df.columns:
                df[col] = 'N/A'
        
        # Reorder columns
        df = df[column_headers]
        
        # Save to CSV with UTF-8 encoding
        df.to_csv(filepath, index=False, encoding='utf-8-sig')
        logger.info(f"Data saved to {filepath} with {len(data)} records")
        
        # Print summary
        print(f"\nFile saved to: {filepath}")
        print(f"Columns saved: {list(df.columns)}")
        print(f"Total records: {len(data)}")

def main():
    # Initialize scraper
    scraper = BPKScraper()
    
    # URL to scrape
    start_url = "https://peraturan.bpk.go.id/Search?keywords=&tentang=&nomor=&jenis=9"
    
    # Scrape data (limit to first 1 page for testing)
    data = scraper.scrape_all_pages(start_url, max_pages=191)
    
    # Save to CSV
    scraper.save_to_csv(data, 'bpk_perpu_data.csv')
    
    print(f"\nScraping completed! Found {len(data)} records.")
    
    # Display sample of first record
    if data:
        print("\nSample of first record:")
        for key, value in data[0].items():
            if key == 'Abstract' and len(str(value)) > 200:
                print(f"{key}: {str(value)[:200]}...")
            elif len(str(value)) > 100:
                print(f"{key}: {str(value)[:100]}...")
            else:
                print(f"{key}: {value}")

if __name__ == "__main__":
    main()