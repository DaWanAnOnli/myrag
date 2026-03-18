#!/usr/bin/env python3
"""
UPDATED IQ Orchestrator Analysis Script

Extracts IQ counts from IQ orchestrator RAG log files (from the current
multi_agent.py with iq_orchestrator), maps to LLM judge scores, and
generates analysis reports.
"""

import re
import csv
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

# ============ CONFIGURATION ============
# Path to the CSV file with LLM judge results
CSV_FILE = Path("../../../dataset/4_experiment/4c_experiment_results/new/llm_judge_results_20251128-152001_no_19_newest.csv")

# Layer 1 folder containing the layer 2 experiment folders
LAYER1_FOLDER = Path("../../4b_retrieval/4b_iv_multi_agents/question_terminal_logs_multi_agent/no_19_iq_newest")

# Mapping of folder names to CSV column names
FOLDER_TO_COLUMN = {
    "no_19_both_2_iq_newest": "1_answer score",
    "no_19_both_3_iq_newest": "2_answer score",
    "no_19_both_4_iq_newest": "3_answer score",
    "no_19_both_5_iq_newest": "4_answer score",
}

# Output directory for results
OUTPUT_DIR = Path("results_19_iq_updated")


# ============ HELPER FUNCTIONS ============

def extract_question_id_from_filename(filename: str) -> int:
    """
    Extract question ID from log filename.

    Example:
      q0909_wid8_id908_20250927-102245_According...
                           ^^^
    Returns: 908, or -1 if not found.
    """
    m = re.search(r"_id(\d+)_", filename)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            return -1
    return -1


def extract_iqs_from_log(log_path: Path) -> int:
    """
    Extract the number of intermediate questions (IQs) generated from an
    IQ orchestrator log file in the current implementation.

    It uses multiple patterns, most reliable first:

    1) "[IQ Orchestrator] Executing X IQ step(s)."
    2) "IQ steps executed: X" (final summary)
    3) "[IQGenerator] Produced X IQ(s):"
    4) Fallback: count unique IQ IDs in log context prefixes "[iq=IQx]"
       while ignoring "iq-runner" and "iq=fallback".
    """
    try:
        content = log_path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        print(f"Error reading {log_path}: {e}")
        return -1

    # Method 1: "[IQ Orchestrator] Executing X IQ step(s)."
    # Example:
    #   [2025-..] [INFO] ... [IQ Orchestrator] Executing 3 IQ step(s).
    m = re.search(r"\[IQ Orchestrator\]\s+Executing\s+(\d+)\s+IQ\s+step\(s\)", content)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            pass

    # Method 2: final summary line: "IQ steps executed: X"
    # Example:
    #   === IQ Orchestrator summary ===
    #   - IQ steps executed: 3
    m = re.search(r"-\s+IQ\s+steps\s+executed:\s+(\d+)", content)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            pass

    # Method 3: "[IQGenerator] Produced X IQ(s):"
    m = re.search(r"\[IQGenerator\]\s+Produced\s+(\d+)\s+IQ\(s\)", content)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            pass

    # Method 4: fallback: count unique IQ contexts from "[iq=...]"
    # In iq_orchestrator you use:
    #   set_log_context("iq-runner")      # top-level
    #   set_log_context(f"iq={sid}")      # per IQ (sid like "IQ1")
    #   log_context_prefix="iq=fallback"  # in fallback path
    iq_matches = re.findall(r"\[iq=([^\]]+)\]", content)
    if iq_matches:
        unique = set()
        for iq in iq_matches:
            iq_str = iq.strip()
            # Filter out non-IQ contexts
            if iq_str in ("iq-runner", "fallback", "iq=fallback"):
                continue
            # Accept typical IQ IDs: "IQ1", "IQ2", etc (case-insensitive)
            if re.match(r"(?i)^IQ\d+$", iq_str):
                unique.add(iq_str.upper())
        if unique:
            return len(unique)

    # If everything fails:
    return -1


def load_scores_from_csv(csv_path: Path) -> Dict[str, Dict[int, int]]:
    """
    Load LLM judge scores from CSV.

    Returns:
        {column_name: {question_id: score_int}}
    where column_name are taken from FOLDER_TO_COLUMN values.
    """
    scores: Dict[str, Dict[int, int]] = {}

    # prepare dicts for each experiment column
    for col in FOLDER_TO_COLUMN.values():
        scores[col] = {}

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

                for col in FOLDER_TO_COLUMN.values():
                    raw = (row.get(col) or "").strip()
                    if raw == "":
                        scores[col][q_id] = -1
                        continue
                    try:
                        scores[col][q_id] = int(raw)
                    except ValueError:
                        scores[col][q_id] = -1
            print(f"Loaded {row_count} rows from CSV")
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        raise

    return scores


def process_experiment_folder(
    folder_path: Path,
    column_name: str,
    scores: Dict[int, int],
) -> List[Tuple[int, int, int]]:
    """
    Process all IQ orchestrator supervisor logs in a given experiment folder.

    Args:
        folder_path: path to folder with *-iq.txt logs.
        column_name: CSV column name for this experiment (for logging only).
        scores: mapping {question_id: score_int}

    Returns:
        List of (question_id, iqs_count, score_int)
    """
    results: List[Tuple[int, int, int]] = []
    failed: List[Tuple[int, str]] = []

    # IQ orchestrator logs are *.txt, often ending with "-iq.txt"
    log_files = sorted(folder_path.glob("*.txt"))
    print(f"  Found {len(log_files)} log files in {folder_path.name}")

    for log_file in log_files:
        q_id = extract_question_id_from_filename(log_file.name)
        if q_id == -1:
            print(f"  Warning: Could not extract ID from {log_file.name}")
            continue

        iqs_count = extract_iqs_from_log(log_file)
        if iqs_count == -1:
            failed.append((q_id, log_file.name))
            continue

        score = scores.get(q_id, -1)
        if score == -1:
            print(f"  Warning: No score found for question ID {q_id} in column '{column_name}'")

        results.append((q_id, iqs_count, score))

    if failed:
        print(f"  WARNING: Failed to extract IQ counts from {len(failed)} files:")
        for q_id, fname in failed[:5]:
            print(f"    - Question {q_id}: {fname}")
        if len(failed) > 5:
            print(f"    ... and {len(failed) - 5} more")

    return results


def analyze_results(results: List[Tuple[int, int, int]], experiment_name: str) -> str:
    """
    Analyze the (question_id, iqs_count, score) tuples and generate a text report.
    """
    lines: List[str] = []
    lines.append("=" * 80)
    lines.append(f"ANALYSIS REPORT: {experiment_name}")
    lines.append("=" * 80)
    lines.append("")

    total_q = len(results)
    lines.append(f"Total questions processed: {total_q}")
    lines.append("")

    # Group by IQ count
    by_iqs: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    for q_id, iqs, score in results:
        by_iqs[iqs].append((q_id, score))

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
        pct = (count / total_scored * 100.0) if total_scored > 0 else 0.0
        lines.append(f"  Score {s}: {count:4d} questions ({pct:5.2f}%)")

    if total_scored > 0:
        avg_score = sum(s * c for s, c in score_counts.items()) / total_scored
        lines.append(f"  Average score: {avg_score:.3f}")
    else:
        lines.append("  No valid scores available.")
    lines.append("")

    # IQ count distribution
    lines.append("INTERMEDIATE QUESTION (IQ) COUNT DISTRIBUTION:")
    lines.append("-" * 40)
    for iqs in sorted(by_iqs.keys()):
        count = len(by_iqs[iqs])
        pct = (count / total_q * 100.0) if total_q > 0 else 0.0
        lines.append(f"  {iqs} IQ(s): {count:4d} questions ({pct:5.2f}%)")
    lines.append("")

    # Score distribution by IQ count
    lines.append("SCORE DISTRIBUTION BY IQ COUNT:")
    lines.append("=" * 80)
    for iqs in sorted(by_iqs.keys()):
        group = by_iqs[iqs]
        lines.append("")
        lines.append(f"Questions using {iqs} IQ(s): {len(group)} total")
        lines.append("-" * 40)

        iq_score_counts: Dict[int, int] = defaultdict(int)
        for _, score in group:
            if score >= 0:
                iq_score_counts[score] += 1

        total_iq_scored = sum(iq_score_counts.values())
        if total_iq_scored > 0:
            for s in sorted(iq_score_counts.keys()):
                count = iq_score_counts[s]
                pct = (count / total_iq_scored * 100.0)
                lines.append(f"  Score {s}: {count:4d} questions ({pct:5.2f}%)")
            avg = sum(s * c for s, c in iq_score_counts.items()) / total_iq_scored
            lines.append(f"  Average score for {iqs} IQ(s): {avg:.3f}")
        else:
            lines.append("  No valid scores for this IQ count")

        example_ids = [q_id for q_id, _ in group[:10]]
        lines.append(f"  Example question IDs: {example_ids}")
    lines.append("")

    # Correlation / trend analysis
    lines.append("CORRELATION ANALYSIS:")
    lines.append("-" * 40)
    lines.append("\nAverage score by IQ count:")
    avg_scores_by_iq: List[Tuple[int, float]] = []

    for iqs in sorted(by_iqs.keys()):
        group = by_iqs[iqs]
        scores = [score for _, score in group if score >= 0]
        if scores:
            avg = sum(scores) / len(scores)
            avg_scores_by_iq.append((iqs, avg))
            lines.append(f"  {iqs} IQ(s): avg score = {avg:.3f} (n={len(scores)})")

    lines.append("\nTrend Analysis:")
    if len(avg_scores_by_iq) > 1:
        improving = 0
        declining = 0
        for i in range(1, len(avg_scores_by_iq)):
            prev_iq, prev_avg = avg_scores_by_iq[i - 1]
            cur_iq, cur_avg = avg_scores_by_iq[i]
            if cur_avg > prev_avg:
                improving += 1
            elif cur_avg < prev_avg:
                declining += 1

        if improving > declining:
            lines.append("  Trend: Average scores tend to IMPROVE with more IQs.")
        elif declining > improving:
            lines.append("  Trend: Average scores tend to DECLINE with more IQs.")
        else:
            lines.append("  Trend: No clear monotonic correlation between IQ count and scores.")
    else:
        lines.append("  Not enough distinct IQ counts to infer a trend.")
    lines.append("")
    lines.append("=" * 80)

    return "\n".join(lines)


# ============ MAIN ============

def main():
    print("Starting UPDATED IQ Orchestrator analysis...")
    print(f"CSV file: {CSV_FILE}")
    print(f"Layer 1 folder: {LAYER1_FOLDER}")
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

    # Load scores
    print("Loading scores from CSV...")
    all_scores = load_scores_from_csv(CSV_FILE)
    print(f"Loaded scores for {len(all_scores)} experiment columns.")
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

        scores_for_exp = all_scores.get(column_name, {})
        if not scores_for_exp:
            print(f"  WARNING: No scores loaded for column '{column_name}'")

        print(f"  Scores available for {len(scores_for_exp)} question IDs in this experiment.")

        # Parse logs
        results = process_experiment_folder(folder_path, column_name, scores_for_exp)
        print(f"  Processed {len(results)} questions successfully")

        # Save raw results
        results_file = OUTPUT_DIR / f"{folder_name}_results.csv"
        with results_file.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["question_id", "iqs_count", "llm_judge_score"])
            for q_id, iqs, score in sorted(results):
                writer.writerow([q_id, iqs, score])
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

    print("UPDATED IQ Orchestrator analysis complete!")


if __name__ == "__main__":
    main()