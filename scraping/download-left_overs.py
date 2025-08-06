import pandas as pd
import requests
import os
from pathlib import Path
from urllib.parse import urlparse, unquote
import time

def get_filename_from_url(url):
    """Extract filename from URL"""
    parsed_url = urlparse(url)
    filename = os.path.basename(parsed_url.path)
    # Decode URL-encoded characters
    filename = unquote(filename)
    
    # If no filename in URL, create one from the last part of path
    if not filename or filename == '/':
        path_parts = [part for part in parsed_url.path.split('/') if part]
        if path_parts:
            filename = unquote(path_parts[-1])
        else:
            filename = 'downloaded_file'
    
    # If still no extension, try to get from content-type or default to PDF
    if '.' not in filename:
        filename += '.pdf'
    
    return filename

def download_file(url, output_path, timeout=30):
    """Download file from URL to output_path"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=timeout, stream=True)
        response.raise_for_status()
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        return True, None
    except Exception as e:
        return False, str(e)

def handle_duplicate_filename(output_dir, filename):
    """Handle duplicate filenames by adding suffix"""
    base_name, ext = os.path.splitext(filename)
    output_path = os.path.join(output_dir, filename)
    
    # If file doesn't exist, use original filename
    if not os.path.exists(output_path):
        return output_path
    
    # If file exists, find next available number
    counter = 1
    while True:
        new_filename = f"{base_name}-{counter}{ext}"
        new_output_path = os.path.join(output_dir, new_filename)
        if not os.path.exists(new_output_path):
            return new_output_path
        counter += 1

def main():
    # Paths relative to script location
    script_dir = Path(__file__).parent
    csv_path = script_dir.parent / "dataset" / "pdfs" / "failed_downloads.csv"
    output_dir = script_dir.parent / "dataset" / "pdfs" / "leftovers"
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Read CSV file
    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded CSV with {len(df)} rows")
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_path}")
        return
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return
    
    # Check if download_link column exists
    if 'download_link' not in df.columns:
        print("Error: 'download_link' column not found in CSV file")
        print(f"Available columns: {list(df.columns)}")
        return
    
    # Track download statistics
    total_downloads = 0
    successful_downloads = 0
    failed_downloads = []
    
    print(f"Starting downloads to {output_dir}")
    print("-" * 60)
    
    # Process each row
    for index, row in df.iterrows():
        download_links = row['download_link']
        
        # Skip if download_links is NaN or empty
        if pd.isna(download_links) or not str(download_links).strip():
            continue
        
        # Split multiple links by '; '
        links = [link.strip() for link in str(download_links).split(';') if link.strip()]
        
        for link in links:
            total_downloads += 1
            
            try:
                # Get filename from URL
                filename = get_filename_from_url(link)
                
                # Handle duplicate filenames
                output_path = handle_duplicate_filename(output_dir, filename)
                
                print(f"[{total_downloads}] Downloading: {link}")
                print(f"    Saving as: {os.path.basename(output_path)}")
                
                # Download file
                success, error = download_file(link, output_path)
                
                if success:
                    successful_downloads += 1
                    file_size = os.path.getsize(output_path)
                    print(f"    ✓ Downloaded successfully ({file_size:,} bytes)")
                else:
                    failed_downloads.append((link, error))
                    print(f"    ✗ Download failed: {error}")
                
                print("-" * 60)
                
                # Small delay to be respectful to the server
                time.sleep(0.5)
                
            except Exception as e:
                failed_downloads.append((link, str(e)))
                print(f"    ✗ Error processing {link}: {e}")
                print("-" * 60)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"DOWNLOAD SUMMARY")
    print(f"{'='*60}")
    print(f"Total files attempted: {total_downloads}")
    print(f"Successful downloads: {successful_downloads}")
    print(f"Failed downloads: {len(failed_downloads)}")
    print(f"Success rate: {successful_downloads/total_downloads*100:.1f}%" if total_downloads > 0 else "No downloads attempted")
    
    if failed_downloads:
        print(f"\nFailed downloads:")
        for i, (link, error) in enumerate(failed_downloads, 1):
            print(f"  {i}. {link}")
            print(f"     Error: {error}")

if __name__ == "__main__":
    main()