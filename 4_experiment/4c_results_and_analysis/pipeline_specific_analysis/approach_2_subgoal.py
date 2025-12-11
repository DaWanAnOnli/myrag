#!/usr/bin/env python3
"""
UPDATED Supervisor Subgoal Analysis Script

Works with the current agentic_supervisor() implementation in multi_agent.py.

It:
- Reads supervisor log files (the ones like *-supervisor.txt).
- Extracts how many subgoals were executed for each question.
- Joins that with LLM judge scores from a CSV.
- Produces per-experiment CSV + text reports summarizing:
  - score distribution overall
  - subgoal-count distribution
  - score distribution by subgoal count
  - average score by subgoal count
"""

import re
import csv
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

# ============ CONFIGURATION ============

# Path to the CSV file with LLM judge results
CSV_FILE = Path(
    "../../../dataset/4_experiment/4c_experiment_results/new/llm_judge_results_20251128-072321_no_18_newest.csv"
)

# Layer 1 folder containing the layer 2 experiment folders
LAYER1_FOLDER = Path(
    "../../4b_retrieval/4b_iv_multi_agents/question_terminal_logs_multi_agent/no_18_subgoal_newest"
)

# Mapping of folder names (each experiment variant) to CSV column names
FOLDER_TO_COLUMN = {
    "no_18_both_2_subgoals_newest": "1_answer score",
    "no_18_both_3_subgoals_newest": "2_answer score",
    "no_18_both_4_subgoals_newest": "3_answer score",
    "no_18_both_5_subgoals_newest": "4_answer score",
}

# Output directory for results
OUTPUT_DIR = Path("results_18_subgoal_newest")


# ============ HELPER FUNCTIONS ============

def extract_question_id_from_filename(filename: str) -> int:
    """
    Extract question ID from log filename.

    Example filename:
        q0909_wid8_id908_20250927-102245_According...
                              ^^^
    Returns: 908, or -1 if no match.
    """
    match = re.search(r"_id(\d+)_", filename)
    if match:
        return int(match.group(1))
    return -1


def extract_subgoals_from_log(log_path: Path) -> int:
    """
    Extract the number of subgoals generated from a supervisor log file.

    Compatible with the current agentic_supervisor logging, which includes:

        [Supervisor] Executing 2 subgoal(s).
        ...
        === Supervisor summary ===
        - Subgoals executed: 2

    Strategy (most reliable first):
    1) "[Supervisor] Executing X subgoal(s)."
    2) "Subgoals executed: X"
    3) "[SubgoalGenerator] Produced X subgoal(s):"
    4) Fallback: count unique 'sg=...' contexts, filtered for SG-like IDs.
    """
    try:
        content = log_path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        print(f"Error reading {log_path}: {e}")
        return -1

    # Method 1: "[Supervisor] Executing X subgoal(s)."
    # Example log:
    #   [2025-..] [INFO] ... [Supervisor] Executing 2 subgoal(s).
    m = re.search(r"\[Supervisor\]\s+Executing\s+(\d+)\s+subgoal\(s\)", content)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            pass

    # Method 2: "Subgoals executed: X" in the final summary section
    # Example:
    #   === Supervisor summary ===
    #   - Subgoals executed: 2
    m = re.search(r"-\s+Subgoals\s+executed:\s+(\d+)", content)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            pass

    # Method 3: "[SubgoalGenerator] Produced X subgoal(s): ..."
    m = re.search(r"\[SubgoalGenerator\]\s+Produced\s+(\d+)\s+subgoal\(s\)", content)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            pass

    # Method 4: fallback – count unique sg=... contexts
    # During subgoal execution, you log contexts like set_log_context(f"sg={sg.get('id')}")
    # so the log lines contain something like " [sg=SG1]".
    sg_matches = re.findall(r"\[sg=([^\]]+)\]", content)
    if sg_matches:
        unique_sgs = set(sg_matches)
        # Heuristic: keep IDs that look like actual subgoal IDs, e.g. "SG1", "SG2"
        unique_sgs = {sg for sg in unique_sgs if sg.upper().startswith("SG") or sg.isdigit()}
        if unique_sgs:
            return len(unique_sgs)

    # If all methods fail:
    return -1


def load_scores_from_csv(csv_path: Path) -> Dict[str, Dict[int, int]]:
    """
    Load scores from CSV file.

    Returns:
        {column_name: {question_id: score_int}}

    The column names are the values in FOLDER_TO_COLUMN.
    """
    scores: Dict[str, Dict[int, int]] = {}

    # Initialize empty dicts for each relevant column
    for col_name in FOLDER_TO_COLUMN.values():
        scores[col_name] = {}

    try:
        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            row_count = 0
            for row in reader:
                row_count += 1
                try:
                    q_id = int(row["id"])
                except (KeyError, ValueError):
                    print(f"Warning: could not parse 'id' in row {row_count}")
                    continue

                for col_name in FOLDER_TO_COLUMN.values():
                    raw = (row.get(col_name) or "").strip()
                    try:
                        score_val = int(raw) if raw != "" else -1
                    except ValueError:
                        score_val = -1
                    scores[col_name][q_id] = score_val

            print(f"Loaded {row_count} rows from CSV")
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        raise

    return scores


def process_experiment_folder(
    folder_path: Path,
    column_name: str,
    scores_for_experiment: Dict[int, int],
) -> List[Tuple[int, int, int]]:
    """
    Process all log files in a single experiment folder.

    Args:
        folder_path: folder containing supervisor logs (*.txt).
        column_name: CSV column name (for logging only).
        scores_for_experiment: {question_id: score_int}

    Returns:
        List of (question_id, subgoals_count, score_int)
    """
    results: List[Tuple[int, int, int]] = []
    failed_extractions: List[Tuple[int, str]] = []

    # We assume each question has one top-level supervisor log (*.txt)
    log_files = sorted(folder_path.glob("*.txt"))
    print(f"  Found {len(log_files)} log files in {folder_path.name}")

    for log_file in log_files:
        q_id = extract_question_id_from_filename(log_file.name)
        if q_id == -1:
            print(f"  Warning: Could not extract ID from {log_file.name}")
            continue

        subgoals_count = extract_subgoals_from_log(log_file)
        if subgoals_count == -1:
            failed_extractions.append((q_id, log_file.name))
            continue

        score = scores_for_experiment.get(q_id, -1)
        if score == -1:
            print(f"  Warning: No score found for question ID {q_id} in column '{column_name}'")

        results.append((q_id, subgoals_count, score))

    if failed_extractions:
        print(f"  WARNING: Failed to extract subgoal counts from {len(failed_extractions)} files:")
        for q_id, fname in failed_extractions[:5]:
            print(f"    - Question {q_id}: {fname}")
        if len(failed_extractions) > 5:
            print(f"    ... and {len(failed_extractions) - 5} more")

    return results


def analyze_results(results: List[Tuple[int, int, int]], experiment_name: str) -> str:
    """
    Analyze the results and generate a text report.

    results: List of (question_id, subgoals_count, score_int)
    """
    lines: List[str] = []
    lines.append("=" * 80)
    lines.append(f"ANALYSIS REPORT: {experiment_name}")
    lines.append("=" * 80)
    lines.append("")

    total_q = len(results)
    lines.append(f"Total questions processed: {total_q}")
    lines.append("")

    # Group by subgoal count
    by_subgoals: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    for q_id, subg, score in results:
        by_subgoals[subg].append((q_id, score))

    # Overall score distribution
    lines.append("OVERALL SCORE DISTRIBUTION:")
    lines.append("-" * 40)
    score_counts: Dict[int, int] = defaultdict(int)
    for _, _, score in results:
        if score >= 0:
            score_counts[score] += 1

    total_scored = sum(score_counts.values())
    for s in sorted(score_counts.keys()):
        count = score_counts[s]
        pct = (count / total_scored * 100) if total_scored > 0 else 0.0
        lines.append(f"  Score {s}: {count:4d} questions ({pct:5.2f}%)")

    if total_scored > 0:
        avg_score = sum(s * c for s, c in score_counts.items()) / total_scored
        lines.append(f"  Average score: {avg_score:.3f}")
    else:
        lines.append("  No valid scores available.")
    lines.append("")

    # Subgoal distribution
    lines.append("SUBGOAL COUNT DISTRIBUTION:")
    lines.append("-" * 40)
    for subg in sorted(by_subgoals.keys()):
        count = len(by_subgoals[subg])
        pct = (count / total_q * 100) if total_q > 0 else 0.0
        lines.append(f"  {subg} subgoal(s): {count:4d} questions ({pct:5.2f}%)")
    lines.append("")

    # Score distribution by subgoal count
    lines.append("SCORE DISTRIBUTION BY SUBGOAL COUNT:")
    lines.append("=" * 80)
    for subg in sorted(by_subgoals.keys()):
        group = by_subgoals[subg]
        lines.append("")
        lines.append(f"Questions with {subg} subgoal(s): {len(group)} total")
        lines.append("-" * 40)

        subgoal_score_counts: Dict[int, int] = defaultdict(int)
        for _, score in group:
            if score >= 0:
                subgoal_score_counts[score] += 1

        total_subg_scored = sum(subgoal_score_counts.values())
        if total_subg_scored > 0:
            for s in sorted(subgoal_score_counts.keys()):
                count = subgoal_score_counts[s]
                pct = (count / total_subg_scored * 100)
                lines.append(f"  Score {s}: {count:4d} questions ({pct:5.2f}%)")
            avg = sum(s * c for s, c in subgoal_score_counts.items()) / total_subg_scored
            lines.append(f"  Average score for {subg} subgoal(s): {avg:.3f}")
        else:
            lines.append("  No valid scores for this subgoal count")

        example_ids = [q_id for q_id, _ in group[:10]]
        lines.append(f"  Example question IDs: {example_ids}")
    lines.append("")

    # Correlation-style summary
    lines.append("CORRELATION STYLE SUMMARY (average score by subgoal count):")
    lines.append("-" * 40)
    for subg in sorted(by_subgoals.keys()):
        group = by_subgoals[subg]
        vals = [score for _, score in group if score >= 0]
        if vals:
            avg = sum(vals) / len(vals)
            lines.append(f"  {subg} subgoal(s): avg score = {avg:.3f} (n={len(vals)})")
    lines.append("")
    lines.append("=" * 80)

    return "\n".join(lines)


# ============ MAIN ============

def main():
    print("Starting UPDATED Supervisor Subgoal analysis...")
    print(f"CSV file: {CSV_FILE}")
    print(f"Layer 1 folder: {LAYER1_FOLDER}")
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

    print("Loading scores from CSV...")
    all_scores = load_scores_from_csv(CSV_FILE)
    print(f"Loaded scores for {len(all_scores)} CSV columns.")
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

        scores_for_experiment = all_scores.get(column_name, {})
        if not scores_for_experiment:
            print(f"  WARNING: No scores loaded for column '{column_name}'.")

        print(f"  Scores available for {len(scores_for_experiment)} question IDs in this experiment.")

        # Parse logs for this experiment
        results = process_experiment_folder(folder_path, column_name, scores_for_experiment)
        print(f"  Processed {len(results)} questions successfully")

        # Save per-question results
        results_file = OUTPUT_DIR / f"{folder_name}_results.csv"
        with results_file.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["question_id", "subgoals_count", "llm_judge_score"])
            for q_id, subg, score in sorted(results):
                writer.writerow([q_id, subg, score])
        print(f"  Saved results to: {results_file}")

        # Generate analysis
        analysis_text = analyze_results(results, folder_name)
        analysis_file = OUTPUT_DIR / f"{folder_name}_analysis.txt"
        analysis_file.write_text(analysis_text, encoding="utf-8")
        print(f"  Saved analysis to: {analysis_file}")

        # Also print to console
        print()
        print(analysis_text)
        print()

    print("UPDATED Supervisor Subgoal analysis complete!")


if __name__ == "__main__":
    main()