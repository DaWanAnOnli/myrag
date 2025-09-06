#!/usr/bin/env python3
"""
Join three JSONL sources by id and write a merged JSONL with fields:
- id
- question
- ground_truth
- agentic_lexidkg_graphrag_answer   (from 'generated_answer' in graph_rag files)
- agentic_naive_rag_answer          (as-is from naive_rag files)
- agentic_naivekg_graphrag_answer   (as-is from naivekg_graphrag files)
"""

import glob
import json
import os
from pathlib import Path
import sys
from datetime import datetime
from typing import Dict, Any, Iterable, List

import dotenv

# =========================
# CONFIG: Edit as needed
# =========================

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
    BASE_INPUT_DIRECTORY = "../../dataset/samples/4_experiment/4b_experiment_answers"
else:
    BASE_INPUT_DIRECTORY = "../../dataset/4_experiment/4b_experiment_answers"


LEXIDKG_GRAPHRAG_PATTERNS = [
    # graph_rag_answers_*.jsonl
    os.path.join(BASE_INPUT_DIRECTORY, "4b_i_lexidkg_graphrag", "graph_rag_answers_*.jsonl"),
]

NAIVE_RAG_PATTERNS = [
    # naive_rag_answers_*.jsonl
    os.path.join(BASE_INPUT_DIRECTORY, "4b_ii_naive_rag", "naive_rag_answers_*.jsonl"),
]

# NAIVEKG_GRAPHRAG_PATTERNS = [
#     # naivekg_graphrag_answers_*.jsonl
#     # Note: original message had a small typo ("base_input/directory"); using BASE_INPUT_DIRECTORY consistently here.
#     os.path.join(BASE_INPUT_DIRECTORY, "4b_iii_naivekg_graphrag", "naivekg_graphrag_answers_*.jsonl"),
# ]

# Output path with timestamp
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_PATH = os.path.join(BASE_INPUT_DIRECTORY, f"combined_answers_{TIMESTAMP}.jsonl")

# =========================
# End CONFIG
# =========================


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def expand_paths(patterns: List[str]) -> List[str]:
    files: List[str] = []
    for p in patterns:
        matched = glob.glob(p)
        if matched:
            files.extend(matched)
        elif os.path.exists(p):
            files.append(p)
        else:
            eprint(f"[warn] No files matched or found for pattern/path: {p}")
    # Deduplicate while preserving order
    seen = set()
    uniq_files = []
    for f in files:
        if f not in seen:
            uniq_files.append(f)
            seen.add(f)
    return uniq_files


def open_maybe_gzip(path: str):
    if path.endswith(".gz"):
        import gzip
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def iter_jsonl(paths: List[str]) -> Iterable[Dict[str, Any]]:
    for path in paths:
        try:
            with open_maybe_gzip(path) as f:
                for lineno, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError as ex:
                        eprint(f"[warn] JSON decode error in {path}:{lineno} -> {ex}")
        except OSError as ex:
            eprint(f"[error] Failed to open {path}: {ex}")


def load_group(paths: List[str],
               source_answer_field: str,
               dest_answer_field: str,
               include_question: bool = True,
               include_ground_truth: bool = True) -> Dict[str, Dict[str, Any]]:
    data: Dict[str, Dict[str, Any]] = {}
    for obj in iter_jsonl(paths):
        if "id" not in obj:
            eprint("[warn] Skipping object without 'id'")
            continue
        id_val = obj["id"]
        id_key = str(id_val)
        rec = data.get(id_key)
        if rec is None:
            rec = {"id": id_val}
        if include_question and "question" in obj and obj["question"] is not None:
            rec["question"] = obj["question"]
        if include_ground_truth and "ground_truth" in obj and obj["ground_truth"] is not None:
            rec["ground_truth"] = obj["ground_truth"]
        if source_answer_field in obj and obj[source_answer_field] is not None:
            rec[dest_answer_field] = obj[source_answer_field]
        data[id_key] = rec
    return data


def pick_first(*vals):
    for v in vals:
        if v is not None:
            return v
    return None


def sort_ids(id_keys: Iterable[str]) -> List[str]:
    def keyfunc(s: str):
        try:
            return (0, int(s))
        except ValueError:
            return (1, s)
    return sorted(id_keys, key=keyfunc)


def main():
    graph_files = expand_paths(LEXIDKG_GRAPHRAG_PATTERNS)
    naive_rag_files = expand_paths(NAIVE_RAG_PATTERNS)
    # naivekg_graphrag_files = expand_paths(NAIVEKG_GRAPHRAG_PATTERNS)

    if not graph_files and not naive_rag_files: #and not naivekg_graphrag_files:
        eprint("[error] No input files found for any group. Exiting.")
        sys.exit(1)

    # Load/normalize each group
    graph_data = load_group(
        graph_files,
        source_answer_field="generated_answer",
        dest_answer_field="agentic_lexidkg_graphrag_answer",
        include_question=True,
        include_ground_truth=True,
    )
    naive_rag_data = load_group(
        naive_rag_files,
        source_answer_field="agentic_naive_rag_answer",
        dest_answer_field="agentic_naive_rag_answer",
        include_question=True,
        include_ground_truth=True,
    )
    # naivekg_graphrag_data = load_group(
    #     naivekg_graphrag_files,
    #     source_answer_field="agentic_naivekg_graphrag_answer",
    #     dest_answer_field="agentic_naivekg_graphrag_answer",
    #     include_question=True,
    #     include_ground_truth=True,
    # )

    all_ids = set(graph_data.keys()) | set(naive_rag_data.keys()) # | set(naivekg_graphrag_data.keys())
    if not all_ids:
        eprint("[warn] No records found across inputs. Output will be empty.")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    written = 0
    with open(OUTPUT_PATH, "w", encoding="utf-8") as out_f:
        for id_key in sort_ids(all_ids):
            g = graph_data.get(id_key, {})
            n = naive_rag_data.get(id_key, {})
            # k = naivekg_graphrag_data.get(id_key, {})

            # Preserve original id type when possible
            id_val = pick_first(g.get("id"), n.get("id"))# , k.get("id"))
            if id_val is None:
                try:
                    id_val = int(id_key)
                except ValueError:
                    id_val = id_key

            merged = {
                "id": id_val,
                "question": pick_first(g.get("question"), n.get("question")), #k.get("question")),
                "ground_truth": pick_first(g.get("ground_truth"), n.get("ground_truth")), #k.get("ground_truth")),
                "agentic_lexidkg_graphrag_answer": g.get("agentic_lexidkg_graphrag_answer"),
                "agentic_naive_rag_answer": n.get("agentic_naive_rag_answer")
                # "agentic_naivekg_graphrag_answer": k.get("agentic_naivekg_graphrag_answer"),
            }

            out_f.write(json.dumps(merged, ensure_ascii=False) + "\n")
            written += 1

    eprint(f"[info] Wrote {written} record(s) to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()