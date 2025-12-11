import json

# Adjust these paths/names as needed
FIRST_JSONL = "../../dataset/4_experiment/4b_experiment_answers/combined_answers_20251130_194919_no_11_amendment_aware.jsonl"      # file with: id, question, ground_truth, 1_answer, 2_answer
SECOND_JSONL = "../../dataset/4_experiment/4a_qa_generation/4a_iii_qa_pairs_with_id/qa_pairs.jsonl"    # file with: id, source_filename, question, answer, outdated_answer
OUTPUT_JSONL = "../../dataset/4_experiment/4b_experiment_answers/combined_answers_20251130_183657_no_11_amendment_aware_with_outdated_answers.jsonl"

def main():
    # 1. Build a mapping from id -> outdated_answer from the second JSONL
    id_to_outdated = {}
    with open(SECOND_JSONL, "r", encoding="utf-8") as f2:
        for line in f2:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            _id = obj.get("id")
            if _id is not None and "outdated_answer" in obj:
                id_to_outdated[_id] = obj["outdated_answer"]

    # 2. Read the first JSONL, add outdated_answer (if found), and write to a new file
    with open(FIRST_JSONL, "r", encoding="utf-8") as f1, \
         open(OUTPUT_JSONL, "w", encoding="utf-8") as fout:

        for line in f1:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            _id = obj.get("id")

            if _id in id_to_outdated:
                obj["outdated_answer"] = id_to_outdated[_id]
            else:
                # If there is no matching id in the second file,
                # you can either skip adding or set it to None.
                # Here we simply leave it unchanged.
                pass

            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()