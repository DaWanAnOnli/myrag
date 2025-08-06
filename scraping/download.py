import os
import csv
import requests
from urllib.parse import urlparse, unquote
import time

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

def download_file(session, url, output_dir, row_data, failed_downloads, max_retries=3, retry_delay=5):
    """Download a file from a given URL and save it to the output directory."""
    parsed_url = urlparse(url)
    # Decode the URL-encoded filename to get proper spaces and characters
    filename = unquote(os.path.basename(parsed_url.path))
    output_path = os.path.join(output_dir, filename)
    
    for attempt in range(max_retries):
        try:
            # Use session to maintain cookies and connection pooling
            response = session.get(url, headers=HEADERS, stream=True, timeout=30)
            response.raise_for_status()  # Raise an error for bad status codes
            
            with open(output_path, 'wb') as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
            
            print(f"Downloaded: {filename}")
            return True  # Success
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:  # Retry if not the last attempt
                print(f"Failed to download {url} (Attempt {attempt + 1}/{max_retries}): {e}. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print(f"Failed to download {url} after {max_retries} attempts: {e}")
                # Add to failed downloads list
                failed_downloads.append({
                    'filename': filename,
                    'download_link': url,
                    'error': str(e),
                    'row_data': row_data
                })
                return False  # Failed

def save_failed_downloads(failed_downloads, output_dir):
    """Save failed downloads to a CSV file."""
    if not failed_downloads:
        print("No failed downloads to save.")
        return
    
    failed_csv_path = os.path.join(output_dir, 'failed_downloads.csv')
    
    with open(failed_csv_path, mode='w', newline='', encoding='utf-8') as file:
        fieldnames = ['filename', 'download_link', 'error']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        
        for failed_download in failed_downloads:
            writer.writerow({
                'filename': failed_download['filename'],
                'download_link': failed_download['download_link'],
                'error': failed_download['error']
            })
    
    print(f"\nSaved {len(failed_downloads)} failed downloads to: {failed_csv_path}")

def main():
    input_csv = os.path.join('..', 'dataset', 'bpk_undang_undang_data.csv')
    output_dir = os.path.join('..', 'dataset', 'pdfs')
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Track failed downloads
    failed_downloads = []
    successful_downloads = 0
    total_downloads = 0
    
    # Use a session to maintain cookies and connection pooling
    with requests.Session() as session:
        with open(input_csv, mode='r', newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                download_link = row.get('Download_Links')
                if download_link:
                    total_downloads += 1
                    print(f"\nProcessing download {total_downloads}: {download_link}")
                    
                    if download_file(session, download_link, output_dir, row, failed_downloads):
                        successful_downloads += 1
                    
                    # Add a delay between requests to be respectful to the server
                    time.sleep(2)  # Increased to 2 seconds
    
    # Save failed downloads to CSV
    save_failed_downloads(failed_downloads, output_dir)
    
    # Print summary
    print(f"\n" + "="*50)
    print(f"DOWNLOAD SUMMARY")
    print(f"="*50)
    print(f"Total downloads attempted: {total_downloads}")
    print(f"Successful downloads: {successful_downloads}")
    print(f"Failed downloads: {len(failed_downloads)}")
    print(f"Success rate: {(successful_downloads/total_downloads)*100:.1f}%" if total_downloads > 0 else "0%")

if __name__ == "__main__":
    main()