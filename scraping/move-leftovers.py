import os
import shutil

# Define source and destination directories
source_dir = '../dataset/pdfs/leftovers'
dest_dir = '../dataset/pdfs'

# Create destination directory if it doesn't exist
os.makedirs(dest_dir, exist_ok=True)

# Move PDF files
for filename in os.listdir(source_dir):
    if filename.lower().endswith('.pdf'):
        source_path = os.path.join(source_dir, filename)
        dest_path = os.path.join(dest_dir, filename)
        
        try:
            shutil.move(source_path, dest_path)
            print(f"Moved: {filename}")
        except Exception as e:
            print(f"Error moving {filename}: {str(e)}")

print("PDF move operation completed!")