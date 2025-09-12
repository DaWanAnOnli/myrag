import os
import json
import shutil
from pathlib import Path
from typing import List, Tuple

import dotenv

env_file_path = Path("../../.env")
    
# Load the .env file
if not env_file_path.exists():
    raise(ImportError(f"Error: .env file not found at {env_file_path}"))

dotenv.load_dotenv(env_file_path)

is_sample = os.getenv('IS_SAMPLE', '').lower()

if is_sample == "true":
    IS_SAMPLE = True
elif is_sample == "false":
    IS_SAMPLE = False
else:
    raise(ValueError(f"Wrong configuration of IS_SAMPLE in .env file: {is_sample}"))

# CONFIGURATION - modify this value as needed
N_FOLDERS = 1  # Number of folders to distribute files into

def get_file_size(file_path: Path) -> int:
    """Get file size in bytes."""
    return file_path.stat().st_size

def format_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"

def distribute_files_by_size(files_with_sizes: List[Tuple[Path, int]], n_folders: int) -> List[List[Tuple[Path, int]]]:
    """
    Distribute files into n folders trying to balance total size per folder.
    Uses a greedy algorithm: assign each file to the folder with smallest current total size.
    """
    # Initialize folders
    folders = [[] for _ in range(n_folders)]
    folder_sizes = [0] * n_folders
    
    # Sort files by size (largest first) for better distribution
    files_with_sizes.sort(key=lambda x: x[1], reverse=True)
    
    # Assign each file to the folder with smallest current total size
    for file_path, file_size in files_with_sizes:
        # Find folder with minimum total size
        min_folder_idx = folder_sizes.index(min(folder_sizes))
        
        # Add file to that folder
        folders[min_folder_idx].append((file_path, file_size))
        folder_sizes[min_folder_idx] += file_size
    
    return folders

def main():
    # Define paths
    if IS_SAMPLE:
        source_dir = Path("../../dataset/samples/2_extract_text_results")
        dest_dir = Path("../../dataset/samples/3_indexing/3a0_distributed_jsons")
    else:
        source_dir = Path("../../dataset/2_extract_text_results")
        dest_dir = Path("../../dataset/3_indexing/3a0_distributed_jsons")
    
    # Check if source directory exists
    if not source_dir.exists():
        print(f"Error: Source directory '{source_dir}' does not exist.")
        return
    
    # Get all JSON files from source directory
    json_files = list(source_dir.glob("*.json"))
    
    if not json_files:
        print(f"No JSON files found in '{source_dir}'.")
        return
    
    print(f"Found {len(json_files)} JSON files in source directory.")
    
    # Get file sizes
    files_with_sizes = []
    total_size = 0
    
    for file_path in json_files:
        try:
            size = get_file_size(file_path)
            files_with_sizes.append((file_path, size))
            total_size += size
        except OSError as e:
            print(f"Warning: Could not get size for {file_path}: {e}")
            continue
    
    if not files_with_sizes:
        print("No valid JSON files found.")
        return
    
    print(f"Total size of all files: {format_size(total_size)}")
    
    # Check destination directory
    if dest_dir.exists():
        if any(dest_dir.iterdir()):  # Directory is not empty
            print(f"\nWarning: Destination directory '{dest_dir}' exists and is not empty.")
            response = input("Do you want to proceed? This may overwrite existing files. (y/n): ").strip().lower()
            if response not in ['y', 'yes']:
                print("Operation cancelled.")
                return
    else:
        # Create destination directory
        dest_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created destination directory: {dest_dir}")
    
    # Distribute files
    print(f"\nDistributing files into {N_FOLDERS} folders...")
    distributed_folders = distribute_files_by_size(files_with_sizes, N_FOLDERS)
    
    # Create folders and copy files
    for i, folder_files in enumerate(distributed_folders, 1):
        folder_path = dest_dir / str(i)
        folder_path.mkdir(exist_ok=True)
        
        # Copy files to this folder
        for file_path, _ in folder_files:
            dest_file_path = folder_path / file_path.name
            try:
                shutil.copy2(file_path, dest_file_path)
            except OSError as e:
                print(f"Error copying {file_path} to {dest_file_path}: {e}")
    
    # Output statistics
    print("\n" + "="*60)
    print("DISTRIBUTION STATISTICS")
    print("="*60)
    print(f"Number of folders: {N_FOLDERS}")
    print(f"Total files distributed: {len(files_with_sizes)}")
    print(f"Total size: {format_size(total_size)}")
    print()
    
    total_folder_size = 0
    for i, folder_files in enumerate(distributed_folders, 1):
        folder_size = sum(size for _, size in folder_files)
        total_folder_size += folder_size
        percentage = (folder_size / total_size) * 100 if total_size > 0 else 0
        
        print(f"Folder {i}:")
        print(f"  - Number of files: {len(folder_files)}")
        print(f"  - Total size: {format_size(folder_size)} ({percentage:.1f}% of total)")
        print()
    
    # Verify total
    print(f"Verification - Total distributed size: {format_size(total_folder_size)}")
    
    # Show size distribution balance
    folder_sizes = [sum(size for _, size in folder_files) for folder_files in distributed_folders]
    min_size = min(folder_sizes)
    max_size = max(folder_sizes)
    size_diff = max_size - min_size
    
    print(f"\nDistribution balance:")
    print(f"  - Smallest folder: {format_size(min_size)}")
    print(f"  - Largest folder: {format_size(max_size)}")
    print(f"  - Size difference: {format_size(size_diff)}")
    print(f"  - Balance ratio: {(min_size/max_size)*100:.1f}%" if max_size > 0 else "N/A")

if __name__ == "__main__":
    main()