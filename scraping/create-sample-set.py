import os
import shutil
import random
import sys

def main():
    # Define directories
    source_dir = '../dataset/pdfs-with-metadata'
    dest_dir = '../dataset/pdfs-with-metadata/samples'
    
    # Check if source directory exists
    if not os.path.exists(source_dir):
        print(f"Error: Source directory '{source_dir}' does not exist")
        return
    
    # Get all PDF files in source directory
    pdf_files = [f for f in os.listdir(source_dir) 
                if f.lower().endswith('.pdf') and 
                os.path.isfile(os.path.join(source_dir, f))]
    
    if not pdf_files:
        print(f"No PDF files found in '{source_dir}'")
        return
    
    total_files = len(pdf_files)
    sample_size = min(100, total_files)  # Handle case when fewer than 100 PDFs exist
    
    # Check if destination directory exists
    if os.path.exists(dest_dir):
        print(f"Warning: Destination directory '{dest_dir}' already exists!")
        response = input("Do you want to overwrite it? (yes/no): ").strip().lower()
        
        if response == 'yes':
            shutil.rmtree(dest_dir)
        else:
            print("Operation cancelled by user")
            return
    
    # Create destination directory
    os.makedirs(dest_dir)
    
    # Randomly sample files
    sampled_files = random.sample(pdf_files, sample_size)
    
    # Copy files to destination
    print(f"\nSampling {sample_size} of {total_files} PDF files...")
    for i, pdf_file in enumerate(sampled_files, 1):
        src_path = os.path.join(source_dir, pdf_file)
        dst_path = os.path.join(dest_dir, pdf_file)
        shutil.copy2(src_path, dst_path)
        print(f"Copied: {pdf_file} ({i}/{sample_size})")
    
    print(f"\nOperation completed! {sample_size} files copied to '{dest_dir}'")

if __name__ == "__main__":
    main()