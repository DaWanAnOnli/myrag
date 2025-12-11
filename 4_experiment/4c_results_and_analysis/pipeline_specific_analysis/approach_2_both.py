#!/usr/bin/env python3
"""
Multi-Agent Aggregator Decision Analysis Script (updated for new multi-agent RAG)

- Extracts aggregator decisions from new-style multi-agent RAG log files.
- Maps aggregator decisions to LLM judge scores from a CSV.
- Generates CSV + text analysis reports.

Assumptions for this experiment:
- Each question is run with AGENTIC_MAX_ITERS = 1 (no loops).
- We only care about the aggregator decision for that single iteration.
"""

import os
import re
import csv
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

# ============ CONFIGURATION ============
# Path to the CSV file with LLM judge results
CSV_FILE = Path("../../../dataset/4_experiment/4c_experiment_results/new/llm_judge_results_20251201-094737_11_vs_14_3rd.csv")

# Layer 1 folder containing the experiment folders with log files
LAYER1_FOLDER = Path("../../4b_retrieval/4b_iv_multi_agents/question_terminal_logs_multi_agent/no_14_both_1_answer_judge_3_newest")

# Mapping of folder name to CSV column name
FOLDER_TO_COLUMN = {
    "inner_folder": "2_answer score",
}

# Output directory for results
OUTPUT_DIR = Path("results_14_both")

# Valid aggregator decisions
VALID_DECISIONS = {"naive", "graphrag", "mixed"}


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


def extract_aggregator_decision_from_log(log_path: Path) -> str:
    """
    Extract the aggregator decision from a multi-agent RAG log file.

    New RAG script behavior to support:
    - Aggregator returns:
        {
            "decision": "choose_graphrag" | "choose_naiverag" | "merge",
            "final_answer": "...",
            "rationale": "...",
            "chosen": "graphrag" | "naive" | "mixed",
            "confidence": ...
        }
    - Logs:
        [Aggregator] Decision=choose_graphrag | Rationale: ...
      and/or
        "chosen": "graphrag"

    We want normalized decision in {"naive", "graphrag", "mixed"}.
    Returns "unknown" if not found.
    """
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading {log_path}: {e}")
        return "unknown"

    # ---- Method 1: New-style summary line ----
    # Example: [Aggregator] Decision=choose_graphrag | Rationale: ...
    m = re.search(r'\[Aggregator\]\s+Decision\s*=\s*(\w+)', content, re.IGNORECASE)
    if m:
        raw = m.group(1).strip().lower()
        # Map choose_* to base decision
        if raw == "choose_graphrag":
            return "graphrag"
        if raw == "choose_naiverag":
            return "naive"
        if raw == "merge":
            return "mixed"
        # If already in naive/graphrag/mixed form
        if raw in VALID_DECISIONS:
            return raw

    # ---- Method 2: JSON-like "chosen": "graphrag|naive|mixed" ----
    m = re.search(r'"chosen"\s*:\s*"(naive|graphrag|mixed)"', content, re.IGNORECASE)
    if m:
        decision = m.group(1).strip().lower()
        if decision in VALID_DECISIONS:
            return decision

    # ---- Method 3: Old-style logs (for backward compatibility) ----
    # 3a: [Aggregator] Decision: chosen=<decision>
    m = re.search(r'\[Aggregator\]\s+Decision:\s+chosen\s*=\s*(\w+)', content, re.IGNORECASE)
    if m:
        decision = m.group(1).strip().lower()
        if decision in VALID_DECISIONS:
            return decision

    # 3b: summary line "- Aggregator: chosen=<decision>"
    m = re.search(r'-\s+Aggregator:\s+chosen\s*=\s*(\w+)', content, re.IGNORECASE)
    if m:
        decision = m.group(1).strip().lower()
        if decision in VALID_DECISIONS:
            return decision

    # 3c: fallback-heuristic messages; try to see "chosen=<...>" near them
    if re.search(r'\[Aggregator\].*Fallback', content, re.IGNORECASE):
        m = re.search(r'chosen\s*=\s*["\']?(naive|graphrag|mixed)["\']?', content, re.IGNORECASE)
        if m:
            decision = m.group(1).strip().lower()
            if decision in VALID_DECISIONS:
                return decision

    return "unknown"


def load_scores_from_csv(csv_path: Path) -> Dict[str, Dict[int, int]]:
    """
    Load scores from CSV file.
    Returns: {column_name: {question_id: score}}
    """
    scores: Dict[str, Dict[int, int]] = {}

    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            # Initialize score dictionaries for each column
            for col_name in FOLDER_TO_COLUMN.values():
                scores[col_name] = {}

            row_count = 0
            for row in reader:
                row_count += 1
                try:
                    q_id = int(row["id"])
                except (ValueError, KeyError) as e:
                    print(f"Warning: Error processing row {row_count}: cannot read 'id': {e}")
                    continue

                for col_name in FOLDER_TO_COLUMN.values():
                    try:
                        score_str = (row.get(col_name, "") or "").strip()
                        score = int(score_str)
                    except (ValueError, KeyError):
                        score = -1  # missing or invalid
                    scores[col_name][q_id] = score

            print(f"Loaded {row_count} rows from CSV")
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        raise

    return scores


def process_experiment_folder(
    folder_path: Path,
    column_name: str,
    scores: Dict[int, int],
) -> List[Tuple[int, str, int]]:
    """
    Process all log files in an experiment folder.
    Returns: List of (question_id, aggregator_decision, score) tuples
    """
    results: List[Tuple[int, str, int]] = []
    failed_extractions: List[Tuple[int, str]] = []

    # Find all .txt log files
    log_files = list(folder_path.glob("*.txt"))
    print(f"  Found {len(log_files)} log files in {folder_path.name}")

    for log_file in log_files:
        # Extract question ID from filename
        q_id = extract_question_id_from_filename(log_file.name)
        if q_id == -1:
            print(f"  Warning: Could not extract ID from {log_file.name}")
            continue

        # Extract aggregator decision from log content
        decision = extract_aggregator_decision_from_log(log_file)
        if decision == "unknown":
            failed_extractions.append((q_id, log_file.name))
            continue

        # Get score from CSV
        score = scores.get(q_id, -1)
        if score == -1:
            print(f"  Warning: No score found for question ID {q_id}")

        results.append((q_id, decision, score))

    if failed_extractions:
        print(f"  WARNING: Failed to extract aggregator decision from {len(failed_extractions)} files:")
        for q_id, fname in failed_extractions[:5]:  # show first 5
            print(f"    - Question {q_id}: {fname}")
        if len(failed_extractions) > 5:
            print(f"    ... and {len(failed_extractions) - 5} more")

    return results


def analyze_results(results: List[Tuple[int, str, int]], experiment_name: str) -> str:
    """
    Analyze the results and generate a report.
    Returns: Analysis text.
    """
    lines: List[str] = []
    lines.append("=" * 80)
    lines.append(f"MULTI-AGENT AGGREGATOR ANALYSIS REPORT: {experiment_name}")
    lines.append("=" * 80)
    lines.append("")

    # Basic statistics
    lines.append(f"Total questions processed: {len(results)}")
    lines.append("")

    # Group by aggregator decision
    by_decision: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
    for q_id, decision, score in results:
        by_decision[decision].append((q_id, score))

    # Overall score distribution
    lines.append("OVERALL SCORE DISTRIBUTION:")
    lines.append("-" * 40)
    score_counts: Dict[int, int] = defaultdict(int)
    for _, _, score in results:
        if score >= 0:
            score_counts[score] += 1

    total_scored = sum(score_counts.values())
    for score in sorted(score_counts.keys()):
        count = score_counts[score]
        pct = (count / total_scored * 100) if total_scored > 0 else 0
        lines.append(f"  Score {score}: {count:4d} questions ({pct:5.2f}%)")

    if total_scored > 0:
        avg_score = sum(s * c for s, c in score_counts.items()) / total_scored
        lines.append(f"  Average score: {avg_score:.3f}")
    lines.append("")

    # Aggregator decision distribution
    lines.append("AGGREGATOR DECISION DISTRIBUTION:")
    lines.append("-" * 40)
    for decision in sorted(by_decision.keys()):
        count = len(by_decision[decision])
        pct = (count / len(results) * 100) if results else 0
        lines.append(f"  {decision.upper()}: {count:4d} questions ({pct:5.2f}%)")
    lines.append("")

    # Score distribution by aggregator decision
    lines.append("SCORE DISTRIBUTION BY AGGREGATOR DECISION:")
    lines.append("=" * 80)

    for decision in sorted(by_decision.keys()):
        questions = by_decision[decision]
        lines.append("")
        lines.append(f"Decision: {decision.upper()}")
        lines.append(f"Total questions: {len(questions)}")
        lines.append("-" * 40)

        decision_score_counts: Dict[int, int] = defaultdict(int)
        for _, score in questions:
            if score >= 0:
                decision_score_counts[score] += 1

        total_decision_scored = sum(decision_score_counts.values())
        if total_decision_scored > 0:
            for score in sorted(decision_score_counts.keys()):
                count = decision_score_counts[score]
                pct = (count / total_decision_scored * 100)
                lines.append(f"  Score {score}: {count:4d} questions ({pct:5.2f}%)")

            avg = sum(s * c for s, c in decision_score_counts.items()) / total_decision_scored
            lines.append(f"  Average score: {avg:.3f}")
        else:
            lines.append("  No valid scores for this decision")

        example_ids = [q[0] for q in questions[:10]]
        lines.append(f"  Example question IDs: {example_ids}")

    lines.append("")

    # Comparative analysis
    lines.append("COMPARATIVE ANALYSIS:")
    lines.append("-" * 40)

    # Average score per decision
    lines.append("\nAverage score by aggregator decision:")
    decision_avgs: List[Tuple[str, float, int]] = []
    for decision in sorted(by_decision.keys()):
        questions = by_decision[decision]
        d_scores = [score for _, score in questions if score >= 0]
        if d_scores:
            avg = sum(d_scores) / len(d_scores)
            decision_avgs.append((decision, avg, len(d_scores)))
            lines.append(f"  {decision.upper()}: avg score = {avg:.3f} (n={len(d_scores)})")

    # Identify best performing decision
    lines.append("\nBest performing aggregator decision:")
    if decision_avgs:
        best_decision, best_avg, best_n = max(decision_avgs, key=lambda x: x[1])
        lines.append(f"  {best_decision.upper()} with average score {best_avg:.3f} (n={best_n})")

    # Score 2 analysis
    lines.append("\nPerfect scores (score=2) by decision:")
    for decision in sorted(by_decision.keys()):
        questions = by_decision[decision]
        perfect = sum(1 for _, score in questions if score == 2)
        total = len(questions)
        pct = (perfect / total * 100) if total > 0 else 0
        lines.append(f"  {decision.upper()}: {perfect}/{total} ({pct:.2f}%)")

    # Score 0 analysis
    lines.append("\nIncorrect answers (score=0) by decision:")
    for decision in sorted(by_decision.keys()):
        questions = by_decision[decision]
        incorrect = sum(1 for _, score in questions if score == 0)
        total = len(questions)
        pct = (incorrect / total * 100) if total > 0 else 0
        lines.append(f"  {decision.upper()}: {incorrect}/{total} ({pct:.2f}%)")

    # Partial correctness (score=1)
    lines.append("\nPartially correct answers (score=1) by decision:")
    for decision in sorted(by_decision.keys()):
        questions = by_decision[decision]
        partial = sum(1 for _, score in questions if score == 1)
        total = len(questions)
        pct = (partial / total * 100) if total > 0 else 0
        lines.append(f"  {decision.upper()}: {partial}/{total} ({pct:.2f}%)")

    # Decision strategy insights
    lines.append("\nDECISION STRATEGY INSIGHTS:")
    lines.append("-" * 40)

    if decision_avgs:
        sorted_decisions = sorted(decision_avgs, key=lambda x: x[1], reverse=True)
        lines.append("Ranking by average score:")
        for i, (dec, avg, n) in enumerate(sorted_decisions, 1):
            lines.append(f"  {i}. {dec.upper()}: avg={avg:.3f} (n={n})")

        lines.append("\nDetailed insights:")

        # Mixed strategy behavior
        if "mixed" in by_decision:
            mixed_scores = [score for _, score in by_decision["mixed"] if score >= 0]
            if mixed_scores:
                mixed_avg = sum(mixed_scores) / len(mixed_scores)
                lines.append(f"  - MIXED strategy: Used in {len(mixed_scores)} cases, avg score = {mixed_avg:.3f}")
                lines.append("    This suggests the aggregator sometimes synthesizes answers.")

        # Naive vs GraphRAG comparison
        if "naive" in by_decision and "graphrag" in by_decision:
            naive_scores = [score for _, score in by_decision["naive"] if score >= 0]
            graphrag_scores = [score for _, score in by_decision["graphrag"] if score >= 0]
            if naive_scores and graphrag_scores:
                naive_avg = sum(naive_scores) / len(naive_scores)
                graphrag_avg = sum(graphrag_scores) / len(graphrag_scores)
                diff = abs(naive_avg - graphrag_avg)

                if graphrag_avg > naive_avg:
                    lines.append(f"  - GraphRAG outperforms Naive by {diff:.3f} points on average")
                    lines.append(f"    ({graphrag_avg:.3f} vs {naive_avg:.3f})")
                elif naive_avg > graphrag_avg:
                    lines.append(f"  - Naive outperforms GraphRAG by {diff:.3f} points on average")
                    lines.append(f"    ({naive_avg:.3f} vs {graphrag_avg:.3f})")
                else:
                    lines.append(f"  - Naive and GraphRAG perform similarly ({naive_avg:.3f})")

        # Success rate (score >= 1)
        lines.append("\nSuccess rate (score >= 1) by decision:")
        for decision in sorted(by_decision.keys()):
            questions = by_decision[decision]
            successes = [score for _, score in questions if score >= 1]
            total = len([score for _, score in questions if score >= 0])
            pct = (len(successes) / total * 100) if total > 0 else 0
            lines.append(f"  {decision.upper()}: {len(successes)}/{total} ({pct:.2f}%)")

    # Recommendation
    lines.append("\nRECOMMENDATION:")
    lines.append("-" * 40)
    if decision_avgs:
        sorted_decisions = sorted(decision_avgs, key=lambda x: x[1], reverse=True)
        best = sorted_decisions[0]
        lines.append(f"Based on average scores, {best[0].upper()} is the best performing strategy.")
        lines.append(f"Average score: {best[1]:.3f} across {best[2]} questions")

        if best[0] == "mixed":
            lines.append("\nThe MIXED strategy performs best, suggesting that:")
            lines.append("  - Combining/synthesizing information from both approaches adds value.")
            lines.append("  - The aggregator is effective at reconciling differences.")
        elif best[0] == "graphrag":
            lines.append("\nGraphRAG performs best, suggesting that:")
            lines.append("  - KG-enhanced retrieval provides higher quality context.")
            lines.append("  - Structured entity/triple reasoning helps for this dataset.")
        elif best[0] == "naive":
            lines.append("\nNaive RAG performs best, suggesting that:")
            lines.append("  - Simple vector search is already strong for this dataset.")
            lines.append("  - The added GraphRAG complexity may not always pay off here.")

        if len(sorted_decisions) >= 2:
            second = sorted_decisions[1]
            gap = best[1] - second[1]
            if gap > 0.2:
                lines.append(f"\nNote: Significant performance gap ({gap:.3f}) between best and second-best.")
                lines.append("      Worth investigating why this gap exists.")
            elif gap < 0.05:
                lines.append(f"\nNote: Small performance gap ({gap:.3f}) between top strategies.")
                lines.append("      Multiple strategies may be viable in practice.")

    lines.append("")
    lines.append("=" * 80)
    return "\n".join(lines)


def main():
    print("Starting Multi-Agent Aggregator Decision analysis (updated)...")
    print(f"CSV file: {CSV_FILE}")
    print(f"Layer 1 folder: {LAYER1_FOLDER}")
    print(f"Output directory: {OUTPUT_DIR}")
    print()

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
        scores = all_scores.get(column_name, {})
        if not scores:
            print(f"  WARNING: No scores loaded for CSV column '{column_name}'")
        # Process log files
        results = process_experiment_folder(folder_path, column_name, scores)
        print(f"  Processed {len(results)} questions successfully")

        # Save raw results to CSV
        results_file = OUTPUT_DIR / f"{folder_name}_aggregator_results.csv"
        with open(results_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["question_id", "aggregator_decision", "llm_judge_score"])
            for q_id, decision, score in sorted(results):
                writer.writerow([q_id, decision, score])
        print(f"  Saved results to: {results_file}")

        # Generate and save analysis
        analysis = analyze_results(results, folder_name)
        analysis_file = OUTPUT_DIR / f"{folder_name}_aggregator_analysis.txt"
        with open(analysis_file, "w", encoding="utf-8") as f:
            f.write(analysis)
        print(f"  Saved analysis to: {analysis_file}")

        # Also print analysis to console
        print()
        print(analysis)
        print()

    print("Multi-Agent Aggregator Decision analysis complete!")


if __name__ == "__main__":
    main()