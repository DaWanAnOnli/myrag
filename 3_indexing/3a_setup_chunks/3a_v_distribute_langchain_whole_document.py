import os
import shutil
from pathlib import Path
from typing import List, Tuple

import dotenv

env_file_path = Path("../../.env")

# Load the .env file
if not env_file_path.exists():
    raise ImportError(f"Error: .env file not found at {env_file_path}")

dotenv.load_dotenv(env_file_path)

is_sample = os.getenv('IS_SAMPLE', '').lower()
if is_sample == "true":
    IS_SAMPLE = True
elif is_sample == "false":
    IS_SAMPLE = False
else:
    raise ValueError(f"Wrong configuration of IS_SAMPLE in .env file: {is_sample}")

if IS_SAMPLE:
    OUTPUT_DIR = "../../dataset/samples/3_indexing/3f1_langchain_whole_document_batches"
    INPUT_DIR = "../../dataset/samples/3_indexing/3f_langchain_whole_document"
else:
    OUTPUT_DIR = "../../dataset/3_indexing/3f1_langchain_whole_document_batches"
    INPUT_DIR = "../../dataset/3_indexing/3f_langchain_whole_document"


# ============================================================================
# CONFIGURATION - EDIT THESE PARAMETERS
# ============================================================================
N_FOLDERS = 15                                # Number of subfolders to create
SKIP_FILE = "all_full_langchain_documents.pkl"  # File to skip from distribution
# ============================================================================


def get_file_size(filepath: str) -> int:
    """Get file size in bytes."""
    return os.path.getsize(filepath)


def distribute_files_by_size(input_folder: str, output_folder: str, n: int, skip_file: str):
    """
    Distribute .pkl files from input_folder to n subfolders in output_folder,
    balanced by total size.
    
    Args:
        input_folder: Source folder containing .pkl files
        output_folder: Destination folder where numbered subfolders will be created
        n: Number of subfolders to create
        skip_file: Filename to skip from distribution
    """
    # Get all .pkl files from input folder
    input_path = Path(input_folder)
    pkl_files = []
    
    for file in input_path.glob("*.pkl"):
        if file.name != skip_file:
            size = get_file_size(file)
            pkl_files.append((file, size))
    
    if not pkl_files:
        print("No .pkl files found to distribute.")
        return
    
    # Sort files by size (descending) for better distribution
    pkl_files.sort(key=lambda x: x[1], reverse=True)
    
    # Create output folder if it doesn't exist
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create n subfolders and initialize tracking
    folder_sizes = {}
    folder_paths = {}
    
    for i in range(1, n + 1):
        folder_path = output_path / str(i)
        folder_path.mkdir(exist_ok=True)
        folder_paths[i] = folder_path
        folder_sizes[i] = 0
    
    # Distribute files using greedy algorithm
    # Always place the next file in the folder with smallest total size
    file_distribution = {i: [] for i in range(1, n + 1)}
    
    for file_path, file_size in pkl_files:
        # Find folder with minimum current size
        min_folder = min(folder_sizes, key=folder_sizes.get)
        
        # Copy file to that folder
        dest_path = folder_paths[min_folder] / file_path.name
        shutil.copy2(file_path, dest_path)
        
        # Update tracking
        folder_sizes[min_folder] += file_size
        file_distribution[min_folder].append((file_path.name, file_size))
    
    # Print distribution summary
    print(f"\nDistribution complete!")
    print(f"Total files distributed: {len(pkl_files)}")
    print(f"\nDistribution summary:")
    print("-" * 80)
    
    total_size = sum(folder_sizes.values())
    
    for i in range(1, n + 1):
        folder_size = folder_sizes[i]
        folder_size_mb = folder_size / (1024 * 1024)
        percentage = (folder_size / total_size * 100) if total_size > 0 else 0
        num_files = len(file_distribution[i])
        
        print(f"Folder {i}:")
        print(f"  Files: {num_files}")
        print(f"  Total size: {folder_size_mb:.2f} MB ({percentage:.1f}%)")
        
        if file_distribution[i]:
            print(f"  Files:")
            for filename, size in file_distribution[i]:
                print(f"    - {filename} ({size / (1024 * 1024):.2f} MB)")
        print()


def main():
    # Validate inputs
    if not os.path.exists(INPUT_DIR):
        print(f"Error: Input folder '{INPUT_DIR}' does not exist.")
        return
    
    if N_FOLDERS <= 0:
        print("Error: Number of folders must be positive.")
        return
    
    # Run distribution
    distribute_files_by_size(INPUT_DIR, OUTPUT_DIR, N_FOLDERS, SKIP_FILE)


if __name__ == "__main__":
    main()