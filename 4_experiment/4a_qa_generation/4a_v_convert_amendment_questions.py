import csv
import json

def csv_to_jsonl(csv_file, jsonl_file):
    """
    Convert CSV file to JSONL format
    
    Args:
        csv_file: Path to input CSV file
        jsonl_file: Path to output JSONL file
    """
    with open(csv_file, 'r', encoding='utf-8') as f_in:
        # Read CSV file
        reader = csv.DictReader(f_in)
        
        processed_count = 0
        skipped_count = 0
        
        with open(jsonl_file, 'w', encoding='utf-8') as f_out:
            for row_num, row in enumerate(reader, start=2):  # start=2 because row 1 is header
                try:
                    # Skip rows with empty or whitespace-only "No" field
                    if not row["No"] or not row["No"].strip():
                        print(f"Warning: Skipping row {row_num} - empty 'No' field")
                        skipped_count += 1
                        continue
                    
                    # Create JSONL object with required fields
                    jsonl_obj = {
                        "id": int(row["No"].strip()),
                        "source_filename": row["UU (max 5 latest per year), no APBN amendments before 1996"].strip(),
                        "question": row["question"].strip(),
                        "answer": row["ground truth"].strip()
                    }
                    
                    # Write each object as a single line
                    f_out.write(json.dumps(jsonl_obj, ensure_ascii=False) + '\n')
                    processed_count += 1
                    
                except ValueError as e:
                    print(f"Warning: Skipping row {row_num} - invalid 'No' value: {row['No']}")
                    skipped_count += 1
                    continue
                except KeyError as e:
                    print(f"Warning: Skipping row {row_num} - missing column: {e}")
                    skipped_count += 1
                    continue
    
    print(f"\nConversion complete!")
    print(f"Processed: {processed_count} rows")
    print(f"Skipped: {skipped_count} rows")
    print(f"JSONL file saved as {jsonl_file}")

if __name__ == "__main__":
    # Specify your input and output file names
    csv_file = "../../dataset/4_experiment/4a_qa_generation/4a_v_amendment_questions/UU_Amendments.csv"  # Change to your CSV file name
    jsonl_file = "../../dataset/4_experiment/4a_qa_generation/4a_v_amendment_questions/amendment_questions.jsonl"  # Change to your desired output file name
    
    
    csv_to_jsonl(csv_file, jsonl_file)