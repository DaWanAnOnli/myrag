import json
import os
from pathlib import Path

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

if IS_SAMPLE:
    SOURCE_FOLDER = "../../dataset/samples/4_experiment/4a_qa_generation/4a_iv_complex_questions"
else:
    SOURCE_FOLDER = "../../dataset/4_experiment/4a_qa_generation/4a_iv_complex_questions"

def combine_json_to_jsonl(input_folder, output_file='combined_questions.jsonl'):
    """
    Combines all JSON files from a folder into a single JSONL file.
    
    Args:
        input_folder: Path to folder containing JSON files
        output_file: Output JSONL filename (default: combined_questions.jsonl)
    """
    global_id = 1
    all_records = []
    
    # Get all JSON files in the folder
    input_path = Path(input_folder)
    json_files = sorted(input_path.glob('*.json'))
    
    if not json_files:
        print(f"No JSON files found in {input_folder}")
        return
    
    print(f"Found {len(json_files)} JSON files")
    
    # Process each JSON file
    for json_file in json_files:
        print(f"Processing: {json_file.name}")
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Process each question in the file
            for item in data:
                record = {
                    'global_id': global_id,
                    'local_id': item['id'],
                    'source_file': json_file.name,
                    'question': item['pertanyaan']
                }
                all_records.append(record)
                global_id += 1
        
        except Exception as e:
            print(f"Error processing {json_file.name}: {e}")
            continue
    
    # Write to JSONL file
    with open(output_file, 'w', encoding='utf-8') as f:
        for record in all_records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    print(f"\nSuccessfully created {output_file}")
    print(f"Total questions: {len(all_records)}")

if __name__ == "__main__":
    # Specify your folder path here
    folder_path = SOURCE_FOLDER
    output_filename = "combined_questions.jsonl"
    
    combine_json_to_jsonl(folder_path, output_filename)