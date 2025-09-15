import os
import json
from pathlib import Path

def combine_jsonl_files():
    # Define the directory path
    input_dir = Path("../../dataset/4_experiment/4a_qa_generation/4a_ii_qa_pairs/")
    output_file = input_dir / "qa_pairs_combined.jsonl"
    
    # Get all .jsonl files in the directory
    jsonl_files = list(input_dir.glob("qa_pairs.batch-*"))
    print(input_dir)
    
    if not jsonl_files:
        print("No JSONL files found in the directory.")
        return
    
    print(f"Found {len(jsonl_files)} JSONL files:")
    for file in jsonl_files:
        print(f"  - {file.name}")
    
    # Combine all files
    total_lines = 0
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for jsonl_file in jsonl_files:
            print(f"Processing {jsonl_file.name}...")
            file_lines = 0
            
            with open(jsonl_file, 'r', encoding='utf-8') as infile:
                for line in infile:
                    line = line.strip()
                    if line:  # Skip empty lines
                        # Validate JSON (optional - will raise error if invalid)
                        try:
                            json.loads(line)
                            outfile.write(line + '\n')
                            file_lines += 1
                            total_lines += 1
                        except json.JSONDecodeError as e:
                            print(f"  Warning: Invalid JSON in {jsonl_file.name}: {e}")
                            continue
            
            print(f"  Added {file_lines} lines from {jsonl_file.name}")
    
    print(f"\nCombination complete!")
    print(f"Total lines written: {total_lines}")
    print(f"Output file: {output_file}")

if __name__ == "__main__":
    combine_jsonl_files()