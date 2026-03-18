#!/usr/bin/env python3
"""
Multi-Agent Answer Judge Iteration Analysis Script (Updated + Aggregator Decision Analysis)

Extracts iteration counts from the NEW multi-agent RAG log files produced by
`agentic_multi_iterative` in multi_agent.py, maps them to LLM judge scores,
and generates analysis reports.

New: also analyzes the distribution of Aggregator decisions across all
iterations and per iteration index.
"""

import os
import re
import csv
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

# ============ CONFIGURATION ============
# Path to the CSV file with LLM judge results
CSV_FILE = Path(
    "../../../dataset/4_experiment/4c_experiment_results/new/llm_judge_results_20251127-042826_no_17_newest_1.csv"
)

# Layer 1 folder containing the layer 2 experiment folders
# (each subfolder contains many .txt logs, one per question)
LAYER1_FOLDER = Path(
    "../../4b_retrieval/4b_iv_multi_agents/question_terminal_logs_multi_agent/no_17_answer_judge_1_newest"
)

# Mapping of folder names (each experiment setting) to CSV column names
# Adjust these if your CSV schema / experiment naming changes.
FOLDER_TO_COLUMN = {
    "no_17_both_2_answer_judge_1_newest": "1_answer score",
    "no_17_both_3_answer_judge_1_newest": "2_answer score",
    "no_17_both_4_answer_judge_1_newest": "3_answer score",
    "no_17_both_5_answer_judge_1_newest": "4_answer score",
}

# Output directory for results
OUTPUT_DIR = Path("results_17_answer_judge_newest_with_aggregator_analyses")


# ============ HELPER FUNCTIONS ============

def extract_question_id_from_filename(filename: str) -> int:
    """
    Extract question ID from log filename.

    Expected pattern (same as old script):
        q0909_wid8_id908_20250927-102245_According...
                              ^^^
    Returns the integer ID (e.g., 908) or -1 if not found.
    """
    match = re.search(r"_id(\d+)_", filename)
    if match:
        return int(match.group(1))
    return -1


def extract_iterations_from_log(log_path: Path) -> int:
    """
    Extract the actual number of iterations executed from a multi-agent log file
    produced by the *new* agentic_multi_iterative orchestrator.

    Priority:
    1. Summary line: "- Total iterations used: X"
    2. Highest iteration in headers: "========== [Iter X/Y] =========="
    3. Any remaining "[Iter X]" patterns (fallback)
    """
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading {log_path}: {e}")
        return -1

    # --- Method 1: Summary line (most reliable) ---
    m1 = re.search(
        r"^-+\s*Total\s+iterations\s+used:\s+(\d+)\s*$|^.*-+\s*Total\s+iterations\s+used:\s+(\d+)",
        content,
        flags=re.MULTILINE,
    )
    if m1:
        for g in m1.groups():
            if g is not None:
                try:
                    return int(g)
                except ValueError:
                    pass

    # Simpler / more direct pattern for the exact log line:
    m1b = re.search(
        r"-\s+Total\s+iterations\s+used:\s+(\d+)", content
    )
    if m1b:
        try:
            return int(m1b.group(1))
        except ValueError:
            pass

    # --- Method 2: Header lines "========== [Iter X/Y] ==========" ---
    iter_matches = re.findall(
        r"=+\s*\[Iter\s+(\d+)/\d+\]\s*=+", content
    )
    if iter_matches:
        try:
            return max(int(x) for x in iter_matches)
        except ValueError:
            pass

    # --- Method 3: Any "[Iter X]" pattern as last fallback ---
    loose_matches = re.findall(r"\[Iter\s+(\d+)\]", content)
    if loose_matches:
        try:
            return max(int(x) for x in loose_matches)
        except ValueError:
            pass

    # If all else fails
    return -1


def extract_aggregator_decisions_from_log(log_path: Path) -> List[Tuple[int, str]]:
    """
    Extract (iteration_number, aggregator_decision) pairs from a log file.

    We rely on:
      - Iteration headers: "========== [Iter 1/2] =========="
      - Aggregator logs:   "[Aggregator] Decision=choose_graphrag | Rationale: ..."

    If an Aggregator decision appears outside any recognized iteration header,
    we assign it to iteration 1 as a fallback.
    """
    decisions: List[Tuple[int, str]] = []

    try:
        with open(log_path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading {log_path}: {e}")
        return decisions

    # Split into lines for a simple sequential scan
    lines = content.splitlines()

    # Regex for iteration header and aggregator decision line
    iter_header_re = re.compile(r"=+\s*\[Iter\s+(\d+)/\d+\]\s*=+")
    # Decision line as logged in multi_agent.py:
    # log(f"[Aggregator] Decision={decision} | Rationale: {rationale[:160]}", level="INFO")
    decision_re = re.compile(r"\[Aggregator\]\s+Decision=([\w_]+)\b")

    current_iter = 1  # default fallback

    for line in lines:
        # Update current_iter when we see a new iteration header
        m_iter = iter_header_re.search(line)
        if m_iter:
            try:
                current_iter = int(m_iter.group(1))
            except ValueError:
                current_iter = current_iter  # keep previous
            continue

        # Look for decision lines
        m_dec = decision_re.search(line)
        if m_dec:
            decision = m_dec.group(1).strip()
            decisions.append((current_iter, decision))

    return decisions


def load_scores_from_csv(csv_path: Path) -> Dict[str, Dict[int, int]]:
    """
    Load scores from the LLM judge CSV file.

    Returns:
        {column_name: {question_id: score_int}}

    Uses the FOLDER_TO_COLUMN mapping to know which CSV columns are relevant.
    """
    scores: Dict[str, Dict[int, int]] = {}

    # Initialize mapping for each column of interest
    for col_name in FOLDER_TO_COLUMN.values():
        scores[col_name] = {}

    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            row_count = 0
            for row in reader:
                row_count += 1
                try:
                    q_id = int(row["id"])
                except (KeyError, ValueError):
                    print(f"Warning: could not read 'id' in row {row_count}")
                    continue

                for col_name in FOLDER_TO_COLUMN.values():
                    try:
                        raw = (row.get(col_name) or "").strip()
                        score = int(raw) if raw != "" else -1
                    except ValueError:
                        score = -1
                    scores[col_name][q_id] = score

            print(f"Loaded {row_count} rows from CSV")
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        raise

    return scores


def process_experiment_folder(
    folder_path: Path,
    column_name: str,
    scores_for_experiment: Dict[int, int],
) -> List[Tuple[int, int, int, List[Tuple[int, str]]]]:
    """
    Process all log files in an experiment folder.

    Args:
        folder_path: path to the folder containing .txt logs for this experiment
        column_name: CSV column describing which score set to use (for logging only)
        scores_for_experiment: {question_id: score} for this experiment

    Returns:
        List of (question_id, iterations, score, aggregator_decisions)
        where:
          - question_id: int
          - iterations: total iterations used (int)
          - score: LLM judge score (int)
          - aggregator_decisions: List[(iter_index, decision_str)]
    """
    results: List[Tuple[int, int, int, List[Tuple[int, str]]]] = []
    failed_extractions: List[Tuple[int, str]] = []

    log_files = sorted(folder_path.glob("*.txt"))
    print(f"  Found {len(log_files)} log files in {folder_path.name}")

    for log_file in log_files:
        q_id = extract_question_id_from_filename(log_file.name)
        if q_id == -1:
            print(f"  Warning: Could not extract ID from {log_file.name}")
            continue

        iterations = extract_iterations_from_log(log_file)
        if iterations == -1:
            failed_extractions.append((q_id, log_file.name))
            continue

        score = scores_for_experiment.get(q_id, -1)
        if score == -1:
            print(f"  Warning: No score found in column '{column_name}' for question ID {q_id}")

        agg_decisions = extract_aggregator_decisions_from_log(log_file)

        results.append((q_id, iterations, score, agg_decisions))

    if failed_extractions:
        print(f"  WARNING: Failed to extract iterations from {len(failed_extractions)} files:")
        for q_id, fname in failed_extractions[:5]:
            print(f"    - Question {q_id}: {fname}")
        if len(failed_extractions) > 5:
            print(f"    ... and {len(failed_extractions) - 5} more")

    return results


def analyze_results(results: List[Tuple[int, int, int, List[Tuple[int, str]]]], experiment_name: str) -> str:
    """
    Analyze the (question_id, iterations, score, aggregator_decisions) tuples and generate a text report.

    - Overall score distribution
    - Iteration count distribution
    - Score distribution conditional on iterations
    - Overall Aggregator decision distribution (over all iterations)
    - Aggregator decision distribution conditional on iteration index
    """
    lines: List[str] = []
    lines.append("=" * 80)
    lines.append(f"ANALYSIS REPORT: {experiment_name}")
    lines.append("=" * 80)
    lines.append("")

    total_q = len(results)
    lines.append(f"Total questions processed: {total_q}")
    lines.append("")

    # Group by iteration count (for questions)
    by_iterations: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    # Scoring stats
    score_counts: Dict[int, int] = defaultdict(int)

    # Aggregator decision stats
    # Overall counts (across all iterations, all questions)
    decision_counts: Dict[str, int] = defaultdict(int)
    # By iteration index
    #   iter_index -> decision -> count
    decision_by_iter: Dict[int, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for q_id, iters, score, agg_decisions in results:
        # For iteration-level grouping (per question)
        by_iterations[iters].append((q_id, score))

        # Score stats
        if score >= 0:
            score_counts[score] += 1

        # Aggregator stats
        for iter_idx, dec in agg_decisions:
            dec_key = dec or "UNKNOWN"
            decision_counts[dec_key] += 1
            decision_by_iter[iter_idx][dec_key] += 1

    # ========== OVERALL SCORE DISTRIBUTION ==========
    lines.append("OVERALL SCORE DISTRIBUTION:")
    lines.append("-" * 40)
    total_scored = sum(score_counts.values())
    for score in sorted(score_counts.keys()):
        count = score_counts[score]
        pct = (count / total_scored * 100) if total_scored > 0 else 0.0
        lines.append(f"  Score {score}: {count:4d} questions ({pct:5.2f}%)")

    if total_scored > 0:
        avg_score = sum(s * c for s, c in score_counts.items()) / total_scored
        lines.append(f"  Average score: {avg_score:.3f}")
    else:
        lines.append("  No valid scores available.")
    lines.append("")

    # ========== ITERATION DISTRIBUTION (PER QUESTION) ==========
    lines.append("ITERATION DISTRIBUTION (PER QUESTION):")
    lines.append("-" * 40)
    for iters in sorted(by_iterations.keys()):
        count = len(by_iterations[iters])
        pct = (count / total_q * 100) if total_q > 0 else 0.0
        lines.append(f"  {iters} iteration(s): {count:4d} questions ({pct:5.2f}%)")
    lines.append("")

    # ========== SCORE DISTRIBUTION BY ITERATION COUNT ==========
    lines.append("SCORE DISTRIBUTION BY ITERATION COUNT (PER QUESTION):")
    lines.append("=" * 80)

    for iters in sorted(by_iterations.keys()):
        group = by_iterations[iters]
        lines.append("")
        lines.append(f"Questions requiring {iters} iteration(s): {len(group)} total")
        lines.append("-" * 40)

        iter_score_counts: Dict[int, int] = defaultdict(int)
        for _, s in group:
            if s >= 0:
                iter_score_counts[s] += 1

        total_iter_scored = sum(iter_score_counts.values())
        if total_iter_scored > 0:
            for score in sorted(iter_score_counts.keys()):
                count = iter_score_counts[score]
                pct = (count / total_iter_scored * 100)
                lines.append(f"  Score {score}: {count:4d} questions ({pct:5.2f}%)")
            avg = sum(s * c for s, c in iter_score_counts.items()) / total_iter_scored
            lines.append(f"  Average score for {iters} iteration(s): {avg:.3f}")
        else:
            lines.append("  No valid scores for this iteration count")

        # Show example question IDs
        example_ids = [q_id for q_id, _ in group[:10]]
        lines.append(f"  Example question IDs: {example_ids}")

    lines.append("")
    lines.append("=" * 80)

    # ========== NEW: OVERALL AGGREGATOR DECISION DISTRIBUTION ==========
    total_decisions = sum(decision_counts.values())
    lines.append("")
    lines.append("AGGREGATOR DECISION DISTRIBUTION (ALL ITERATIONS, ALL QUESTIONS):")
    lines.append("-" * 80)
    lines.append(f"Total aggregator decisions observed: {total_decisions}")
    if total_decisions == 0:
        lines.append("  No aggregator decisions found in logs.")
    else:
        for dec in sorted(decision_counts.keys()):
            count = decision_counts[dec]
            pct = (count / total_decisions * 100.0)
            lines.append(f"  Decision '{dec}': {count:5d} times ({pct:5.2f}%)")
    lines.append("")
    lines.append("=" * 80)

    # ========== NEW: AGGREGATOR DECISION DISTRIBUTION BY ITERATION INDEX ==========
    lines.append("AGGREGATOR DECISION DISTRIBUTION BY ITERATION INDEX:")
    lines.append("=" * 80)

    if not decision_by_iter:
        lines.append("  No aggregator decisions found by iteration.")
    else:
        for iter_idx in sorted(decision_by_iter.keys()):
            dec_map = decision_by_iter[iter_idx]
            iter_total = sum(dec_map.values())
            lines.append("")
            lines.append(f"Iteration {iter_idx}: {iter_total} total decisions")
            lines.append("-" * 40)
            for dec in sorted(dec_map.keys()):
                count = dec_map[dec]
                pct = (count / iter_total * 100.0) if iter_total > 0 else 0.0
                lines.append(f"  Decision '{dec}': {count:5d} times ({pct:5.2f}%)")

    lines.append("")
    lines.append("=" * 80)

    return "\n".join(lines)


# ============ MAIN ============

def main():
    print("Starting UPDATED Multi-Agent Iteration analysis (with Aggregator decisions)...")
    print(f"CSV file: {CSV_FILE}")
    print(f"Layer 1 folder: {LAYER1_FOLDER}")
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

    print("Loading scores from CSV...")
    all_scores_by_column = load_scores_from_csv(CSV_FILE)
    print(f"Loaded score mappings for {len(all_scores_by_column)} CSV columns.")
    print()

    # Process each experiment folder configured in FOLDER_TO_COLUMN
    for folder_name, column_name in FOLDER_TO_COLUMN.items():
        print(f"Processing experiment: {folder_name}")
        print(f"  Using CSV column: {column_name}")

        folder_path = LAYER1_FOLDER / folder_name
        if not folder_path.exists():
            print(f"  ERROR: Folder not found: {folder_path}")
            print()
            continue

        # Scores for this experiment
        scores_for_experiment = all_scores_by_column.get(column_name, {})
        if not scores_for_experiment:
            print(f"  WARNING: No scores loaded for column '{column_name}'. All scores may be -1.")
        print(f"  Scores available for {len(scores_for_experiment)} question IDs in this experiment.")

        # Parse logs
        results = process_experiment_folder(folder_path, column_name, scores_for_experiment)
        print(f"  Processed {len(results)} questions successfully")

        # Save raw results (per-question)
        results_file = OUTPUT_DIR / f"{folder_name}_results.csv"
        with open(results_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            # For CSV we’ll store only per-question fields (aggregator decisions are per-iteration and
            # more complex; they go into the text analysis report)
            writer.writerow(["question_id", "iterations_used", "llm_judge_score"])
            for q_id, iterations, score, _ in sorted(results):
                writer.writerow([q_id, iterations, score])
        print(f"  Saved per-question results to: {results_file}")

        # Generate & save analysis report (includes aggregator decision analysis)
        analysis_text = analyze_results(results, folder_name)
        analysis_file = OUTPUT_DIR / f"{folder_name}_analysis.txt"
        with open(analysis_file, "w", encoding="utf-8") as f:
            f.write(analysis_text)
        print(f"  Saved analysis report to: {analysis_file}")

        # Also print analysis summary to console
        print()
        print(analysis_text)
        print()

    print("Updated Multi-Agent Iteration analysis complete!")


if __name__ == "__main__":
    main()