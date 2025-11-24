import os
import re
import csv
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

# ============ CONFIGURATION ============
# Path to the CSV file with LLM judge results
CSV_FILE = Path("../../../dataset/4_experiment/4c_experiment_results/new/llm_judge_results_20251114-220509_no_6_graphrag_iq.csv")

# Layer 1 folder containing the layer 2 experiment folders
LAYER1_FOLDER = Path("../../4b_retrieval/4b_i_lexidkg_graphrag/question_terminal_logs/no_6_iq")

# Mapping of folder names to CSV column names
FOLDER_TO_COLUMN = {
    "no_6_lexidkg_2_iq_fix": "graphrag_2_iq_answer score",
    "no_6_lexidkg_3_iq_fix": "graphrag_3_iq_answer score",
    "no_6_lexidkg_4_iq_fix": "graphrag_4_iq_answer score",
    "no_6_lexidkg_5_iq_fix": "graphrag_5_iq_answer score",
}

# Output directory for results
OUTPUT_DIR = Path("results_6_iq")


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


def extract_iterations_from_log(log_path: Path) -> int:
    """
    Extract the number of iterations used from a log file.
    Looks for lines like:
    "- Iterations used: 2"
    or "- Number of IQs executed: 3"
    """
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Look for "Iterations used: X"
        match = re.search(r'-\s*Iterations used:\s*(\d+)', content)
        if match:
            return int(match.group(1))
        
        # Look for "Number of IQs executed: X"
        match = re.search(r'-\s*Number of IQs executed:\s*(\d+)', content)
        if match:
            return int(match.group(1))
        
        # Alternative: count how many "--- IQ X/Y" lines appear
        iteration_matches = re.findall(r'--- IQ (\d+)/(\d+)', content)
        if iteration_matches:
            # Get the maximum iteration number actually reached
            max_iter = max(int(match[0]) for match in iteration_matches)
            return max_iter
            
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
    Returns: List of (question_id, iterations, score) tuples
    """
    results = []
    
    # Find all .txt log files
    log_files = list(folder_path.glob("*.txt"))
    print(f"  Found {len(log_files)} log files in {folder_path.name}")
    
    for log_file in log_files:
        # Extract question ID from filename
        q_id = extract_question_id_from_filename(log_file.name)
        if q_id == -1:
            print(f"  Warning: Could not extract ID from {log_file.name}")
            continue
        
        # Extract iterations from log content
        iterations = extract_iterations_from_log(log_file)
        if iterations == -1:
            print(f"  Warning: Could not extract iterations from {log_file.name}")
            continue
        
        # Get score from CSV
        score = scores.get(q_id, -1)
        if score == -1:
            print(f"  Warning: No score found for question ID {q_id}")
        
        results.append((q_id, iterations, score))
    
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
    
    # Group by iteration count
    by_iterations = defaultdict(list)
    for q_id, iters, score in results:
        by_iterations[iters].append((q_id, score))
    
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
    
    # Iteration distribution
    lines.append("ITERATION DISTRIBUTION:")
    lines.append("-" * 40)
    for iters in sorted(by_iterations.keys()):
        count = len(by_iterations[iters])
        pct = (count / len(results) * 100) if len(results) > 0 else 0
        lines.append(f"  {iters} iteration(s): {count:4d} questions ({pct:5.2f}%)")
    lines.append("")
    
    # Score distribution by iteration count
    lines.append("SCORE DISTRIBUTION BY ITERATION COUNT:")
    lines.append("=" * 80)
    
    for iters in sorted(by_iterations.keys()):
        questions = by_iterations[iters]
        lines.append("")
        lines.append(f"Questions requiring {iters} iteration(s): {len(questions)} total")
        lines.append("-" * 40)
        
        # Count scores
        iter_score_counts = defaultdict(int)
        for _, score in questions:
            if score >= 0:
                iter_score_counts[score] += 1
        
        total_iter_scored = sum(iter_score_counts.values())
        if total_iter_scored > 0:
            for score in sorted(iter_score_counts.keys()):
                count = iter_score_counts[score]
                pct = (count / total_iter_scored * 100)
                lines.append(f"  Score {score}: {count:4d} questions ({pct:5.2f}%)")
            
            avg = sum(score * count for score, count in iter_score_counts.items()) / total_iter_scored
            lines.append(f"  Average score: {avg:.3f}")
        else:
            lines.append("  No valid scores for this iteration count")
        
        # Show some example question IDs
        lines.append(f"  Example question IDs: {[q[0] for q in questions[:10]]}")
    
    lines.append("")
    lines.append("=" * 80)
    
    return "\n".join(lines)


def main():
    print("Starting IQ iteration analysis...")
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
        print(f"  Processed {len(results)} questions")
        
        # Save results to CSV
        results_file = OUTPUT_DIR / f"{folder_name}_results.csv"
        with open(results_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['question_id', 'iterations_required', 'llm_judge_score'])
            for q_id, iters, score in sorted(results):
                writer.writerow([q_id, iters, score])
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
    
    print("IQ iteration analysis complete!")


if __name__ == "__main__":
    main()