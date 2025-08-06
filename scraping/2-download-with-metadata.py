import os
import csv
import requests
from urllib.parse import urlparse, unquote
import time
import json
from pathlib import Path
import PyPDF2
from PyPDF2 import PdfWriter, PdfReader
from datetime import datetime

# More comprehensive headers to mimic a real browser
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.9',
    'Accept-Encoding': 'gzip, deflate, br',
    'DNT': '1',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1',
    'Sec-Fetch-Dest': 'document',
    'Sec-Fetch-Mode': 'navigate',
    'Sec-Fetch-Site': 'none',
    'Sec-Fetch-User': '?1',
    'Cache-Control': 'max-age=0'
}

def add_metadata_to_pdf(input_path, output_path, metadata_row):
    """Add CSV row data as metadata to PDF file"""
    try:
        # Read the original PDF
        with open(input_path, 'rb') as input_file:
            reader = PdfReader(input_file)
            writer = PdfWriter()
            
            # Copy all pages
            for page in reader.pages:
                writer.add_page(page)
            
            # Prepare standard PDF metadata
            pdf_metadata = {
                '/Title': str(metadata_row.get('Title', '')),
                '/Subject': str(metadata_row.get('Abstract', ''))[:500] if metadata_row.get('Abstract') else '',
                '/Creator': 'BPK Undang-Undang Downloader',
                '/Producer': 'PyPDF2 with Metadata',
                '/CreationDate': datetime.now().strftime("D:%Y%m%d%H%M%S"),
            }
            
            # Add all CSV columns as custom metadata
            for column, value in metadata_row.items():
                if value and str(value).strip() and str(value) != 'nan':  # Skip empty/NaN values
                    # Create custom metadata key
                    custom_key = f'/Custom_{column}'
                    # Limit length to prevent PDF corruption
                    pdf_metadata[custom_key] = str(value)[:500]
            
            # Set metadata
            writer.add_metadata(pdf_metadata)
            
            # Write the new PDF with metadata
            with open(output_path, 'wb') as output_file:
                writer.write(output_file)
                
        return True
                
    except Exception as e:
        print(f"Warning: Could not add metadata to {output_path}: {e}")
        # If metadata addition fails, copy the original file
        try:
            with open(input_path, 'rb') as src, open(output_path, 'wb') as dst:
                dst.write(src.read())
            return True
        except Exception as e2:
            print(f"Error: Could not even copy file {input_path}: {e2}")
            return False

def get_unique_filename(output_dir, filename):
    """Generate a unique filename if a file with the same name already exists"""
    base_path = Path(output_dir) / filename
    
    if not base_path.exists():
        return str(base_path)
    
    # Split filename and extension
    stem = base_path.stem
    suffix = base_path.suffix
    counter = 1
    
    while True:
        new_filename = f"{stem}-{counter}{suffix}"
        new_path = Path(output_dir) / new_filename
        if not new_path.exists():
            return str(new_path)
        counter += 1

def download_file(session, url, output_dir, row_data, failed_downloads, max_retries=3, retry_delay=5):
    """Download a file from a given URL and save it to the output directory with metadata."""
    parsed_url = urlparse(url)
    # Decode the URL-encoded filename to get proper spaces and characters
    filename = unquote(os.path.basename(parsed_url.path))
    
    # Get unique filename to avoid overwriting
    final_output_path = get_unique_filename(output_dir, filename)
    temp_output_path = final_output_path + '.tmp'
    
    for attempt in range(max_retries):
        try:
            # Use session to maintain cookies and connection pooling
            response = session.get(url, headers=HEADERS, stream=True, timeout=30)
            response.raise_for_status()  # Raise an error for bad status codes
            
            # Download to temporary file first
            with open(temp_output_path, 'wb') as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
            
            # Add metadata to PDF and save to final location
            if add_metadata_to_pdf(temp_output_path, final_output_path, row_data):
                # Remove temporary file
                os.remove(temp_output_path)
                print(f"Downloaded with metadata: {os.path.basename(final_output_path)}")
                return True
            else:
                # If metadata addition failed, the file should still be copied
                if os.path.exists(temp_output_path):
                    os.remove(temp_output_path)
                print(f"Downloaded (metadata failed): {os.path.basename(final_output_path)}")
                return True
                
        except requests.exceptions.RequestException as e:
            # Clean up temporary file on failure
            if os.path.exists(temp_output_path):
                os.remove(temp_output_path)
                
            if attempt < max_retries - 1:  # Retry if not the last attempt
                print(f"Failed to download {url} (Attempt {attempt + 1}/{max_retries}): {e}. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print(f"Failed to download {url} after {max_retries} attempts: {e}")
                # Add complete row data to failed downloads
                failed_entry = dict(row_data)  # Copy all row data
                failed_entry['Failed_Download_Link'] = url  # Add which specific link failed
                failed_entry['Error_Message'] = str(e)
                failed_entry['Failed_Filename'] = filename
                failed_downloads.append(failed_entry)
                return False  # Failed
        except Exception as e:
            # Clean up temporary file on unexpected error
            if os.path.exists(temp_output_path):
                os.remove(temp_output_path)
            
            print(f"Unexpected error downloading {url}: {e}")
            # Add complete row data to failed downloads
            failed_entry = dict(row_data)  # Copy all row data
            failed_entry['Failed_Download_Link'] = url
            failed_entry['Error_Message'] = str(e)
            failed_entry['Failed_Filename'] = filename
            failed_downloads.append(failed_entry)
            return False

def parse_download_links(download_links_str):
    """Parse download links string that may contain multiple URLs separated by '; '"""
    if not download_links_str:
        return []
    
    # Split by '; ' and clean up each URL
    links = [link.strip() for link in download_links_str.split(';') if link.strip()]
    return links

def save_failed_downloads(failed_downloads, output_parent_dir, original_csv_fieldnames):
    """Save failed downloads with complete CSV row data to a CSV file."""
    if not failed_downloads:
        print("No failed downloads to save.")
        return
    
    failed_csv_path = os.path.join(output_parent_dir, 'failed_downloads.csv')
    
    # Create fieldnames: original CSV fields + additional failure info
    additional_fields = ['Failed_Download_Link', 'Error_Message', 'Failed_Filename']
    fieldnames = list(original_csv_fieldnames) + additional_fields
    
    with open(failed_csv_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        
        for failed_download in failed_downloads:
            writer.writerow(failed_download)
    
    print(f"\nSaved {len(failed_downloads)} failed downloads with complete CSV data to: {failed_csv_path}")
    print("The failed_downloads.csv contains all original CSV columns plus:")
    print("  - Failed_Download_Link: The specific URL that failed")
    print("  - Error_Message: The error that occurred")
    print("  - Failed_Filename: The filename that would have been saved")

def read_metadata_from_pdf(pdf_path):
    """Read metadata from a PDF file for testing purposes"""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PdfReader(file)
            metadata = reader.metadata
            
            if metadata:
                print(f"\nMetadata for {os.path.basename(pdf_path)}:")
                for key, value in metadata.items():
                    if key.startswith('/Custom_'):
                        column_name = key.replace('/Custom_', '')
                        print(f"  {column_name}: {value}")
                    else:
                        print(f"  {key}: {value}")
                return dict(metadata)
            else:
                print(f"No metadata found in {pdf_path}")
                return {}
                
    except Exception as e:
        print(f"Error reading metadata from {pdf_path}: {e}")
        return {}

def main():
    input_csv = os.path.join('..', 'dataset', 'bpk_undang_undang_data.csv')
    output_dir = os.path.join('..', 'dataset', 'pdfs-with-metadata')
    output_parent_dir = os.path.join('..', 'dataset')  # Parent directory for failed_downloads.csv
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Track failed downloads and CSV fieldnames
    failed_downloads = []
    successful_downloads = 0
    total_downloads = 0
    total_files = 0
    csv_fieldnames = []
    
    # Use a session to maintain cookies and connection pooling
    with requests.Session() as session:
        with open(input_csv, mode='r', newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            csv_fieldnames = reader.fieldnames  # Capture original CSV fieldnames
            
            for row_num, row in enumerate(reader, 1):
                download_links_str = row.get('Download_Links')
                if download_links_str:
                    # Parse multiple download links
                    download_links = parse_download_links(download_links_str)
                    total_downloads += 1
                    
                    print(f"\n{'='*60}")
                    print(f"Processing row {row_num}: {row.get('UU_Number', 'Unknown')}")
                    print(f"Title: {row.get('Title', 'No title')}")
                    print(f"Found {len(download_links)} download link(s)")
                    print(f"{'='*60}")
                    
                    row_successful = 0
                    for link_num, download_link in enumerate(download_links, 1):
                        total_files += 1
                        print(f"\nDownloading file {link_num}/{len(download_links)}: {download_link}")
                        
                        if download_file(session, download_link, output_dir, row, failed_downloads):
                            successful_downloads += 1
                            row_successful += 1
                        
                        # Add a delay between requests to be respectful to the server
                        time.sleep(1)  # 2 seconds between each file
                    
                    print(f"\nRow {row_num} summary: {row_successful}/{len(download_links)} files downloaded successfully")
    
    # Save failed downloads to CSV with complete row data
    save_failed_downloads(failed_downloads, output_parent_dir, csv_fieldnames)
    
    # Print summary
    print(f"\n" + "="*60)
    print(f"DOWNLOAD SUMMARY")
    print(f"="*60)
    print(f"Total CSV rows processed: {total_downloads}")
    print(f"Total files attempted: {total_files}")
    print(f"Successful file downloads: {successful_downloads}")
    print(f"Failed file downloads: {len(failed_downloads)}")
    print(f"Success rate: {(successful_downloads/total_files)*100:.1f}%" if total_files > 0 else "0%")
    
    # Show example of reading metadata from a downloaded file
    if successful_downloads > 0:
        # Find first successfully downloaded PDF
        for file in os.listdir(output_dir):
            if file.endswith('.pdf'):
                example_path = os.path.join(output_dir, file)
                print(f"\nExample metadata from downloaded file:")
                read_metadata_from_pdf(example_path)
                break

if __name__ == "__main__":
    main()