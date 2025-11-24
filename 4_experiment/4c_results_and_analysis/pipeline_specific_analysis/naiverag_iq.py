#!/usr/bin/env python3
"""
Intermediate Question (IQ) Analysis Script
Extracts IQ counts from RAG log files, maps to LLM judge scores,
and generates analysis reports.
"""

import os
import re
import csv
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

# ============ CONFIGURATION ============
# Path to the LLM judge CSV file
CSV_FILE = Path("../../../dataset/4_experiment/4c_experiment_results/new/llm_judge_results_20251114-222201_no_5_naiverag_iq.csv")

# Base directory that contains the 4 experiment folders
LAYER1_FOLDER = Path("../../4b_retrieval/4b_ii_naiverag/question_terminal_logs_naive_over_graph/no_5_iq")

# Mapping of folder names to CSV column names
FOLDER_TO_COLUMN = {
    "no_5_naiverag_2_iq_1250": "naiverag_2_iq_answer score",
    "no_5_naiverag_3_iq_1250": "naiverag_3_iq_answer score",
    "no_5_naiverag_4_iq_1250": "naiverag_4_iq_answer score",
    "no_5_naiverag_5_iq_1250": "naiverag_5_iq_answer score",
}

# Output directory for results
OUTPUT_DIR = Path("result_5_")


# ============ HELPER FUNCTIONS ============
def extract_question_id_from_filename(filename: str) -> int:
    """
    Extract question ID from log filename.
    Example: q0909_wid8_id908_20250927-102245_According...
    Returns: 908
    """
    match = re.search(r'_id(\d+)_', filename)
    if match:
        return int(match.group(1))
    return -1


def extract_iqs_from_log(log_path: Path) -> int:
    """
    Extract the actual number of intermediate questions (IQs) used from a log file.
    Uses multiple methods with most reliable patterns preferred.
    """
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Method 1: Look for "[IQ Planner] Planned X IQ(s):"
        # This is the most reliable as it's logged right after generation
        match = re.search(r'\[IQ Planner\]\s+Planned\s+(\d+)\s+IQ\(s?\):', content)
        if match:
            return int(match.group(1))
        
        # Method 2: Look for "- IQs executed: X" in the summary section
        match = re.search(r'^-\s+IQs\s+executed:\s+(\d+)', content, re.MULTILINE)
        if match:
            return int(match.group(1))
        
        # Method 3: Count the logged IQ lines "  - IQX: ..."
        # These are logged right after "Planned X IQ(s):"
        matches = re.findall(r'^\s+-\s+IQ\d+:', content, re.MULTILINE)
        if matches:
            return len(matches)
        
        # Method 4: Look for "• [IQX]:" in summary
        matches = re.findall(r'^\s+•\s+\[IQ\d+\]', content, re.MULTILINE)
        if matches:
            return len(matches)
        
        # Method 5: Count "--- IQ X/Y ---" patterns (most specific to actual execution)
        matches = re.findall(r'^---\s+IQ\s+\d+/(\d+)\s+---', content, re.MULTILINE)
        if matches:
            # Return the maximum Y value (total IQs)
            return max(int(m) for m in matches)
            
    except Exception as e:
        print(f"Error reading {log_path}: {e}")
    
    return -1


def load_scores_from_csv(csv_path: Path) -> Dict[str, Dict[int, int]]:
    """
    Load scores from CSV file.
    Returns: {column_name: {question_id: score}}
    """
    scores = {}
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            # Initialize score dictionaries for each column
            for col_name in FOLDER_TO_COLUMN.values():
                scores[col_name] = {}
            
            row_count = 0
            for row in reader:
                row_count += 1
                try:
                    q_id = int(row['id'])
                    for col_name in FOLDER_TO_COLUMN.values():
                        try:
                            score_str = row[col_name].strip()
                            score = int(score_str)
                            scores[col_name][q_id] = score
                        except (ValueError, KeyError) as e:
                            scores[col_name][q_id] = -1  # Missing or invalid score
                except (ValueError, KeyError) as e:
                    print(f"Warning: Error processing row {row_count}: {e}")
                    continue
            
            print(f"Loaded {row_count} rows from CSV")
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        raise
    
    return scores


def process_experiment_folder(folder_path: Path, column_name: str, 
                              scores: Dict[int, int]) -> List[Tuple[int, int, int]]:
    """
    Process all log files in an experiment folder.
    Returns: List of (question_id, iqs, score) tuples
    """
    results = []
    failed_extractions = []
    
    # Find all .txt log files
    log_files = list(folder_path.glob("*.txt"))
    print(f"  Found {len(log_files)} log files in {folder_path.name}")
    
    for log_file in log_files:
        # Extract question ID from filename
        q_id = extract_question_id_from_filename(log_file.name)
        if q_id == -1:
            print(f"  Warning: Could not extract ID from {log_file.name}")
            continue
        
        # Extract IQs from log content
        iqs = extract_iqs_from_log(log_file)
        if iqs == -1:
            failed_extractions.append((q_id, log_file.name))
            continue
        
        # Get score from CSV
        score = scores.get(q_id, -1)
        if score == -1:
            print(f"  Warning: No score found for question ID {q_id}")
        
        results.append((q_id, iqs, score))
    
    if failed_extractions:
        print(f"  WARNING: Failed to extract IQs from {len(failed_extractions)} files:")
        for q_id, fname in failed_extractions[:5]:  # Show first 5
            print(f"    - Question {q_id}: {fname}")
        if len(failed_extractions) > 5:
            print(f"    ... and {len(failed_extractions) - 5} more")
    
    return results


def analyze_results(results: List[Tuple[int, int, int]], experiment_name: str) -> str:
    """
    Analyze the results and generate a report.
    Returns: Analysis text
    """
    lines = []
    lines.append("=" * 80)
    lines.append(f"ANALYSIS REPORT: {experiment_name}")
    lines.append("=" * 80)
    lines.append("")
    
    # Basic statistics
    lines.append(f"Total questions processed: {len(results)}")
    lines.append("")
    
    # Group by IQ count
    by_iqs = defaultdict(list)
    for q_id, iqs, score in results:
        by_iqs[iqs].append((q_id, score))
    
    # Overall score distribution
    lines.append("OVERALL SCORE DISTRIBUTION:")
    lines.append("-" * 40)
    score_counts = defaultdict(int)
    for _, _, score in results:
        if score >= 0:
            score_counts[score] += 1
    
    total_scored = sum(score_counts.values())
    for score in sorted(score_counts.keys()):
        count = score_counts[score]
        pct = (count / total_scored * 100) if total_scored > 0 else 0
        lines.append(f"  Score {score}: {count:4d} questions ({pct:5.2f}%)")
    
    if total_scored > 0:
        avg_score = sum(score * count for score, count in score_counts.items()) / total_scored
        lines.append(f"  Average score: {avg_score:.3f}")
    lines.append("")
    
    # IQ distribution
    lines.append("INTERMEDIATE QUESTION (IQ) DISTRIBUTION:")
    lines.append("-" * 40)
    for iqs in sorted(by_iqs.keys()):
        count = len(by_iqs[iqs])
        pct = (count / len(results) * 100) if len(results) > 0 else 0
        lines.append(f"  {iqs} IQ(s): {count:4d} questions ({pct:5.2f}%)")
    lines.append("")
    
    # Score distribution by IQ count
    lines.append("SCORE DISTRIBUTION BY IQ COUNT:")
    lines.append("=" * 80)
    
    for iqs in sorted(by_iqs.keys()):
        questions = by_iqs[iqs]
        lines.append("")
        lines.append(f"Questions requiring {iqs} IQ(s): {len(questions)} total")
        lines.append("-" * 40)
        
        # Count scores
        iq_score_counts = defaultdict(int)
        for _, score in questions:
            if score >= 0:
                iq_score_counts[score] += 1
        
        total_iq_scored = sum(iq_score_counts.values())
        if total_iq_scored > 0:
            for score in sorted(iq_score_counts.keys()):
                count = iq_score_counts[score]
                pct = (count / total_iq_scored * 100)
                lines.append(f"  Score {score}: {count:4d} questions ({pct:5.2f}%)")
            
            avg = sum(score * count for score, count in iq_score_counts.items()) / total_iq_scored
            lines.append(f"  Average score: {avg:.3f}")
        else:
            lines.append("  No valid scores for this IQ count")
        
        # Show some example question IDs
        example_ids = [q[0] for q in questions[:10]]
        lines.append(f"  Example question IDs: {example_ids}")
    
    lines.append("")
    lines.append("=" * 80)
    
    return "\n".join(lines)


def main():
    print("Starting Intermediate Question (IQ) analysis...")
    print(f"CSV file: {CSV_FILE}")
    print(f"Layer 1 folder: {LAYER1_FOLDER}")
    print(f"Output directory: {OUTPUT_DIR}")
    print()
    
    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    
    # Load scores from CSV
    print("Loading scores from CSV...")
    all_scores = load_scores_from_csv(CSV_FILE)
    print(f"Loaded scores for {len(all_scores)} experiments")
    print()
    
    # Process each experiment folder
    for folder_name, column_name in FOLDER_TO_COLUMN.items():
        print(f"Processing experiment: {folder_name}")
        print(f"  CSV column: {column_name}")
        
        folder_path = LAYER1_FOLDER / folder_name
        if not folder_path.exists():
            print(f"  ERROR: Folder not found: {folder_path}")
            print()
            continue
        
        # Get scores for this experiment
        scores = all_scores[column_name]
        
        # Process log files
        results = process_experiment_folder(folder_path, column_name, scores)
        print(f"  Processed {len(results)} questions successfully")
        
        # Save results to CSV
        results_file = OUTPUT_DIR / f"{folder_name}_results.csv"
        with open(results_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['question_id', 'intermediate_questions_required', 'llm_judge_score'])
            for q_id, iqs, score in sorted(results):
                writer.writerow([q_id, iqs, score])
        print(f"  Saved results to: {results_file}")
        
        # Generate and save analysis
        analysis = analyze_results(results, folder_name)
        analysis_file = OUTPUT_DIR / f"{folder_name}_analysis.txt"
        with open(analysis_file, 'w', encoding='utf-8') as f:
            f.write(analysis)
        print(f"  Saved analysis to: {analysis_file}")
        
        # Also print analysis to console
        print()
        print(analysis)
        print()
    
    print("Intermediate Question (IQ) analysis complete!")


if __name__ == "__main__":
    main()