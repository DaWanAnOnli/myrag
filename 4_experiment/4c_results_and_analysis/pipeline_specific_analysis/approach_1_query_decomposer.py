#!/usr/bin/env python3
"""
Multi-Agent Query Decomposer + Aggregator Decision Analysis Script
Extracts decomposer and aggregator decisions from multi-agent RAG log files,
maps to LLM judge scores, and generates comprehensive analysis reports.
"""

import os
import re
import csv
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

# ============ CONFIGURATION ============
# Path to the CSV file with LLM judge results
CSV_FILE = Path("../../../dataset/4_experiment/4c_experiment_results/new/Individual base multi-agent - approach_1_query_decomposer.csv")

# Layer 1 folder containing the layer 2 experiment folders
LAYER1_FOLDER = Path("../../4b_retrieval/4b_iv_multi_agents/question_terminal_logs_multi_agent/no_13_approach_1_query_decomposer")


# Mapping of folder name to CSV column name
FOLDER_TO_COLUMN = {
    "no_13_approach_1_query_decomposer_5_hops_1250": "approach_1_query_decomposer_answer score",
}

# Output directory for results
OUTPUT_DIR = Path("results_13_query_decomposer")

# Valid aggregator decisions
VALID_AGGREGATOR_DECISIONS = {"choose_graphrag", "choose_naiverag", "merge"}


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


def normalize_task_pattern(pattern: str) -> str:
    """
    Normalize task pattern by sorting components.
    E.g., "1N_primary, 1G_support" -> "1G_support, 1N_primary"
    
    This ensures that patterns with the same tasks but different order
    are grouped together.
    """
    if not pattern or pattern == "unknown":
        return pattern
    
    # Split by comma and strip whitespace
    components = [c.strip() for c in pattern.split(',')]
    
    # Sort components to ensure consistent ordering
    # Sort by: pipeline (G before N), then role (primary before support), then count
    def sort_key(comp):
        # Extract pipeline letter and role
        if 'G_' in comp:
            pipeline = 'G'
        elif 'N_' in comp:
            pipeline = 'N'
        else:
            pipeline = 'Z'  # Unknown goes last
        
        if '_primary' in comp:
            role = 'A_primary'  # A so it comes before support
        elif '_support' in comp:
            role = 'B_support'
        else:
            role = 'Z_unknown'
        
        # Extract count (number at the beginning)
        count_match = re.match(r'(\d+)', comp)
        count = int(count_match.group(1)) if count_match else 0
        
        return (pipeline, role, -count)  # Negative count so higher counts come first
    
    components.sort(key=sort_key)
    
    return ", ".join(components)


def extract_task_pattern(content: str) -> Optional[str]:
    """
    Extract the pattern of tasks from decomposer logs.
    E.g., "1G_primary, 1N_support" or "2G_primary, 1N_primary"
    Returns normalized pattern (sorted for consistency).
    """
    # Look for task log entries like:
    # [Decomposer] Task 1: pipeline=graphrag role=primary aspect=...
    task_matches = re.findall(
        r'\[Decomposer\]\s+Task\s+\d+:\s+pipeline\s*=\s*(\w+)\s+role\s*=\s*(\w+)',
        content,
        re.IGNORECASE
    )
    
    if not task_matches:
        return None
    
    # Count tasks by pipeline and role
    graphrag_primary = sum(1 for p, r in task_matches if p.lower() == "graphrag" and r.lower() == "primary")
    graphrag_support = sum(1 for p, r in task_matches if p.lower() == "graphrag" and r.lower() == "support")
    naiverag_primary = sum(1 for p, r in task_matches if p.lower() == "naiverag" and r.lower() == "primary")
    naiverag_support = sum(1 for p, r in task_matches if p.lower() == "naiverag" and r.lower() == "support")
    
    parts = []
    if graphrag_primary > 0:
        parts.append(f"{graphrag_primary}G_primary")
    if graphrag_support > 0:
        parts.append(f"{graphrag_support}G_support")
    if naiverag_primary > 0:
        parts.append(f"{naiverag_primary}N_primary")
    if naiverag_support > 0:
        parts.append(f"{naiverag_support}N_support")
    
    if parts:
        pattern = ", ".join(parts)
        return normalize_task_pattern(pattern)
    
    return None


def extract_decomposer_decision_from_log(log_path: Path) -> str:
    """
    Extract the query decomposer decision/strategy from a multi-agent RAG log file.
    
    Returns ONLY the normalized task pattern (e.g., "1G_support, 1N_primary").
    Does NOT include the strategy description.
    """
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract task pattern directly (this is what we want)
        tasks_info = extract_task_pattern(content)
        if tasks_info:
            return tasks_info
        
    except Exception as e:
        print(f"Error reading {log_path}: {e}")
    
    return "unknown"


def extract_aggregator_decision_from_log(log_path: Path) -> str:
    """
    Extract the aggregator decision from a multi-agent RAG log file.
    
    Returns one of: "choose_graphrag", "choose_naiverag", "merge", or "unknown"
    """
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Method 1: Look for "[Aggregator] Decision=<decision>"
        match = re.search(r'\[Aggregator\]\s+Decision\s*=\s*(\w+)', content, re.IGNORECASE)
        if match:
            decision = match.group(1).strip().lower()
            if decision in VALID_AGGREGATOR_DECISIONS:
                return decision
        
        # Method 2: Look for "- Aggregator decision: <decision>" in summary section
        match = re.search(r'-\s+Aggregator\s+decision:\s*(\w+)', content, re.IGNORECASE)
        if match:
            decision = match.group(1).strip().lower()
            if decision in VALID_AGGREGATOR_DECISIONS:
                return decision
        
        # Method 3: Look for '"decision": "<decision>"' in aggregator output
        match = re.search(r'"decision"\s*:\s*"(choose_graphrag|choose_naiverag|merge)"', 
                         content, re.IGNORECASE)
        if match:
            decision = match.group(1).strip().lower()
            if decision in VALID_AGGREGATOR_DECISIONS:
                return decision
        
        # Method 4: Look for fallback decision
        if re.search(r'\[Aggregator\].*Fallback.*decision', content, re.IGNORECASE):
            match = re.search(r'decision\s*=\s*["\']?(choose_graphrag|choose_naiverag|merge)["\']?', 
                             content, re.IGNORECASE)
            if match:
                return match.group(1).strip().lower()
            
    except Exception as e:
        print(f"Error reading {log_path}: {e}")
    
    return "unknown"


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
                              scores: Dict[int, int]) -> List[Tuple[int, str, str, int]]:
    """
    Process all log files in an experiment folder.
    Returns: List of (question_id, decomposer_decision, aggregator_decision, score) tuples
    """
    results = []
    failed_decomposer = []
    failed_aggregator = []
    
    # Find all .txt log files
    log_files = list(folder_path.glob("*.txt"))
    print(f"  Found {len(log_files)} log files in {folder_path.name}")
    
    for log_file in log_files:
        # Extract question ID from filename
        q_id = extract_question_id_from_filename(log_file.name)
        if q_id == -1:
            print(f"  Warning: Could not extract ID from {log_file.name}")
            continue
        
        # Extract decomposer decision (now just the normalized pattern)
        decomposer_dec = extract_decomposer_decision_from_log(log_file)
        if decomposer_dec == "unknown":
            failed_decomposer.append((q_id, log_file.name))
        
        # Extract aggregator decision
        aggregator_dec = extract_aggregator_decision_from_log(log_file)
        if aggregator_dec == "unknown":
            failed_aggregator.append((q_id, log_file.name))
        
        # Get score from CSV
        score = scores.get(q_id, -1)
        if score == -1:
            print(f"  Warning: No score found for question ID {q_id}")
        
        results.append((q_id, decomposer_dec, aggregator_dec, score))
    
    if failed_decomposer:
        print(f"  WARNING: Failed to extract decomposer decision from {len(failed_decomposer)} files")
        for q_id, fname in failed_decomposer[:3]:
            print(f"    - Question {q_id}: {fname}")
        if len(failed_decomposer) > 3:
            print(f"    ... and {len(failed_decomposer) - 3} more")
    
    if failed_aggregator:
        print(f"  WARNING: Failed to extract aggregator decision from {len(failed_aggregator)} files")
        for q_id, fname in failed_aggregator[:3]:
            print(f"    - Question {q_id}: {fname}")
        if len(failed_aggregator) > 3:
            print(f"    ... and {len(failed_aggregator) - 3} more")
    
    return results


def analyze_results(results: List[Tuple[int, str, str, int]], experiment_name: str) -> str:
    """
    Analyze the results and generate a comprehensive report.
    Returns: Analysis text
    """
    lines = []
    lines.append("=" * 100)
    lines.append(f"DECOMPOSER + AGGREGATOR ANALYSIS REPORT: {experiment_name}")
    lines.append("=" * 100)
    lines.append("")
    
    # Basic statistics
    lines.append(f"Total questions processed: {len(results)}")
    lines.append("")
    
    # Overall score distribution
    lines.append("OVERALL SCORE DISTRIBUTION:")
    lines.append("-" * 50)
    score_counts = defaultdict(int)
    for _, _, _, score in results:
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
    
    # Group by decomposer decision
    by_decomposer = defaultdict(list)
    for q_id, dec_d, dec_a, score in results:
        by_decomposer[dec_d].append((q_id, dec_a, score))
    
    # Group by aggregator decision
    by_aggregator = defaultdict(list)
    for q_id, dec_d, dec_a, score in results:
        by_aggregator[dec_a].append((q_id, dec_d, score))
    
    # Group by combination
    by_combination = defaultdict(list)
    for q_id, dec_d, dec_a, score in results:
        by_combination[(dec_d, dec_a)].append((q_id, score))
    
    # ========== DECOMPOSER DECISION DISTRIBUTION ==========
    lines.append("QUERY DECOMPOSER DECISION DISTRIBUTION:")
    lines.append("=" * 100)
    
    decomposer_sorted = sorted(by_decomposer.items(), key=lambda x: len(x[1]), reverse=True)
    for dec_d, items in decomposer_sorted:
        count = len(items)
        pct = (count / len(results) * 100) if len(results) > 0 else 0
        lines.append(f"\nDecomposer: {dec_d}")
        lines.append(f"  Count: {count:4d} questions ({pct:5.2f}%)")
        
        # Score distribution for this decomposer decision
        dec_score_counts = defaultdict(int)
        for _, _, score in items:
            if score >= 0:
                dec_score_counts[score] += 1
        
        total_dec_scored = sum(dec_score_counts.values())
        if total_dec_scored > 0:
            lines.append("  Score breakdown:")
            for score in sorted(dec_score_counts.keys()):
                cnt = dec_score_counts[score]
                pct_s = (cnt / total_dec_scored * 100)
                lines.append(f"    Score {score}: {cnt:4d} ({pct_s:5.2f}%)")
            
            avg = sum(score * cnt for score, cnt in dec_score_counts.items()) / total_dec_scored
            lines.append(f"  Average score: {avg:.3f}")
        
        # Show aggregator breakdown for this decomposer
        agg_breakdown = defaultdict(int)
        for _, dec_a, _ in items:
            agg_breakdown[dec_a] += 1
        lines.append("  Aggregator decisions within this decomposer pattern:")
        for agg_d, agg_cnt in sorted(agg_breakdown.items(), key=lambda x: x[1], reverse=True):
            lines.append(f"    {agg_d}: {agg_cnt}")
    
    lines.append("")
    
    # ========== AGGREGATOR DECISION DISTRIBUTION ==========
    lines.append("AGGREGATOR DECISION DISTRIBUTION:")
    lines.append("=" * 100)
    
    aggregator_sorted = sorted(by_aggregator.items(), key=lambda x: len(x[1]), reverse=True)
    for dec_a, items in aggregator_sorted:
        count = len(items)
        pct = (count / len(results) * 100) if len(results) > 0 else 0
        lines.append(f"\nAggregator: {dec_a}")
        lines.append(f"  Count: {count:4d} questions ({pct:5.2f}%)")
        
        # Score distribution for this aggregator decision
        agg_score_counts = defaultdict(int)
        for _, _, score in items:
            if score >= 0:
                agg_score_counts[score] += 1
        
        total_agg_scored = sum(agg_score_counts.values())
        if total_agg_scored > 0:
            lines.append("  Score breakdown:")
            for score in sorted(agg_score_counts.keys()):
                cnt = agg_score_counts[score]
                pct_s = (cnt / total_agg_scored * 100)
                lines.append(f"    Score {score}: {cnt:4d} ({pct_s:5.2f}%)")
            
            avg = sum(score * cnt for score, cnt in agg_score_counts.items()) / total_agg_scored
            lines.append(f"  Average score: {avg:.3f}")
        
        # Show decomposer breakdown for this aggregator
        dec_breakdown = defaultdict(int)
        for _, dec_d, _ in items:
            dec_breakdown[dec_d] += 1
        lines.append("  Decomposer patterns within this aggregator decision:")
        for dec_d, dec_cnt in sorted(dec_breakdown.items(), key=lambda x: x[1], reverse=True)[:5]:
            lines.append(f"    {dec_d}: {dec_cnt}")
        if len(dec_breakdown) > 5:
            lines.append(f"    ... and {len(dec_breakdown) - 5} more patterns")
    
    lines.append("")
    
    # ========== COMBINED DECOMPOSER + AGGREGATOR ANALYSIS ==========
    lines.append("COMBINED DECOMPOSER + AGGREGATOR DECISION ANALYSIS:")
    lines.append("=" * 100)
    
    combination_sorted = sorted(by_combination.items(), key=lambda x: len(x[1]), reverse=True)
    
    lines.append(f"\nTotal unique combinations: {len(combination_sorted)}")
    lines.append("\nTop 20 most frequent combinations:")
    lines.append("-" * 50)
    
    for idx, ((dec_d, dec_a), items) in enumerate(combination_sorted[:20], 1):
        count = len(items)
        pct = (count / len(results) * 100) if len(results) > 0 else 0
        
        # Calculate average score for this combination
        scores_list = [score for _, score in items if score >= 0]
        avg_score = sum(scores_list) / len(scores_list) if scores_list else 0.0
        
        # Score breakdown
        combo_score_counts = defaultdict(int)
        for _, score in items:
            if score >= 0:
                combo_score_counts[score] += 1
        
        score_dist = ", ".join([f"{s}:{combo_score_counts[s]}" for s in sorted(combo_score_counts.keys())])
        
        lines.append(f"\n{idx}. Decomposer: {dec_d}")
        lines.append(f"   Aggregator: {dec_a}")
        lines.append(f"   Count: {count} ({pct:.2f}%) | Avg score: {avg_score:.3f}")
        lines.append(f"   Score distribution: {score_dist}")
    
    lines.append("")
    
    # ========== COMPARATIVE ANALYSIS ==========
    lines.append("COMPARATIVE ANALYSIS:")
    lines.append("=" * 100)
    
    # Best decomposer patterns
    lines.append("\nTop 10 Decomposer patterns by average score:")
    lines.append("-" * 50)
    decomposer_avgs = []
    for dec_d, items in by_decomposer.items():
        scores = [score for _, _, score in items if score >= 0]
        if scores and len(scores) >= 5:  # Only consider patterns with at least 5 examples
            avg = sum(scores) / len(scores)
            decomposer_avgs.append((dec_d, avg, len(scores)))
    
    decomposer_avgs.sort(key=lambda x: x[1], reverse=True)
    for i, (dec, avg, n) in enumerate(decomposer_avgs[:10], 1):
        lines.append(f"  {i}. {dec}")
        lines.append(f"     Avg score: {avg:.3f} (n={n})")
    
    # Best aggregator decisions
    lines.append("\nAggregator decisions by average score:")
    lines.append("-" * 50)
    aggregator_avgs = []
    for dec_a, items in by_aggregator.items():
        scores = [score for _, _, score in items if score >= 0]
        if scores:
            avg = sum(scores) / len(scores)
            aggregator_avgs.append((dec_a, avg, len(scores)))
    
    aggregator_avgs.sort(key=lambda x: x[1], reverse=True)
    for i, (dec, avg, n) in enumerate(aggregator_avgs, 1):
        lines.append(f"  {i}. {dec}")
        lines.append(f"     Avg score: {avg:.3f} (n={n})")
    
    # Best combinations
    lines.append("\nTop 15 Decomposer+Aggregator combinations by average score (min 5 examples):")
    lines.append("-" * 50)
    combo_avgs = []
    for (dec_d, dec_a), items in by_combination.items():
        scores = [score for _, score in items if score >= 0]
        if scores and len(scores) >= 5:
            avg = sum(scores) / len(scores)
            combo_avgs.append((dec_d, dec_a, avg, len(scores)))
    
    combo_avgs.sort(key=lambda x: x[2], reverse=True)
    for i, (dec_d, dec_a, avg, n) in enumerate(combo_avgs[:15], 1):
        lines.append(f"  {i}. Decomposer: {dec_d}")
        lines.append(f"     Aggregator: {dec_a}")
        lines.append(f"     Avg score: {avg:.3f} (n={n})")
        lines.append("")
    
    # ========== SUCCESS RATES ==========
    lines.append("SUCCESS RATES (score >= 1):")
    lines.append("=" * 100)
    
    # By decomposer (top 10)
    lines.append("\nTop 10 Decomposer patterns by success rate (min 10 examples):")
    decomposer_success = []
    for dec_d, items in by_decomposer.items():
        scores = [score for _, _, score in items if score >= 0]
        if scores and len(scores) >= 10:
            success = sum(1 for s in scores if s >= 1)
            rate = success / len(scores)
            decomposer_success.append((dec_d, rate, success, len(scores)))
    
    decomposer_success.sort(key=lambda x: x[1], reverse=True)
    for i, (dec, rate, succ, total) in enumerate(decomposer_success[:10], 1):
        lines.append(f"  {i}. {dec}")
        lines.append(f"     Success rate: {rate*100:.1f}% ({succ}/{total})")
    
    # By aggregator
    lines.append("\nAggregator decisions by success rate:")
    aggregator_success = []
    for dec_a, items in by_aggregator.items():
        scores = [score for _, _, score in items if score >= 0]
        if scores:
            success = sum(1 for s in scores if s >= 1)
            rate = success / len(scores)
            aggregator_success.append((dec_a, rate, success, len(scores)))
    
    aggregator_success.sort(key=lambda x: x[1], reverse=True)
    for i, (dec, rate, succ, total) in enumerate(aggregator_success, 1):
        lines.append(f"  {i}. {dec}")
        lines.append(f"     Success rate: {rate*100:.1f}% ({succ}/{total})")
    
    # ========== PERFECT SCORES (score=2) ==========
    lines.append("\nPERFECT SCORES (score=2) ANALYSIS:")
    lines.append("=" * 100)
    
    # By decomposer (top 10)
    lines.append("\nTop 10 Decomposer patterns by perfect score rate (min 10 examples):")
    decomposer_perfect = []
    for dec_d, items in by_decomposer.items():
        scores = [score for _, _, score in items if score >= 0]
        if scores and len(scores) >= 10:
            perfect = sum(1 for s in scores if s == 2)
            rate = perfect / len(scores)
            decomposer_perfect.append((dec_d, rate, perfect, len(scores)))
    
    decomposer_perfect.sort(key=lambda x: x[1], reverse=True)
    for i, (dec, rate, perf, total) in enumerate(decomposer_perfect[:10], 1):
        lines.append(f"  {i}. {dec}")
        lines.append(f"     Perfect score rate: {rate*100:.1f}% ({perf}/{total})")
    
    # By aggregator
    lines.append("\nAggregator decisions by perfect score rate:")
    aggregator_perfect = []
    for dec_a, items in by_aggregator.items():
        scores = [score for _, _, score in items if score >= 0]
        if scores:
            perfect = sum(1 for s in scores if s == 2)
            rate = perfect / len(scores)
            aggregator_perfect.append((dec_a, rate, perfect, len(scores)))
    
    aggregator_perfect.sort(key=lambda x: x[1], reverse=True)
    for i, (dec, rate, perf, total) in enumerate(aggregator_perfect, 1):
        lines.append(f"  {i}. {dec}")
        lines.append(f"     Perfect score rate: {rate*100:.1f}% ({perf}/{total})")
    
    # ========== KEY INSIGHTS ==========
    lines.append("\nKEY INSIGHTS:")
    lines.append("=" * 100)
    
    if decomposer_avgs:
        best_dec = decomposer_avgs[0]
        lines.append(f"\n1. Best Decomposer Pattern (by avg score, min 5 examples):")
        lines.append(f"   {best_dec[0]}")
        lines.append(f"   Average score: {best_dec[1]:.3f} across {best_dec[2]} questions")
    
    if aggregator_avgs:
        best_agg = aggregator_avgs[0]
        lines.append(f"\n2. Best Aggregator Decision:")
        lines.append(f"   {best_agg[0]}")
        lines.append(f"   Average score: {best_agg[1]:.3f} across {best_agg[2]} questions")
    
    if combo_avgs:
        best_combo = combo_avgs[0]
        lines.append(f"\n3. Best Combination (min 5 examples):")
        lines.append(f"   Decomposer: {best_combo[0]}")
        lines.append(f"   Aggregator: {best_combo[1]}")
        lines.append(f"   Average score: {best_combo[2]:.3f} across {best_combo[3]} questions")
    
    # Analyze aggregator choice patterns
    lines.append("\n4. Aggregator Choice Patterns:")
    if aggregator_avgs:
        for dec_a, avg, n in aggregator_avgs:
            lines.append(f"   - {dec_a}: avg={avg:.3f}, used in {n} cases")
            if dec_a == "merge":
                lines.append(f"     The MERGE strategy suggests combining both pipelines adds value")
            elif dec_a == "choose_graphrag":
                lines.append(f"     GraphRAG was chosen as the better source")
            elif dec_a == "choose_naiverag":
                lines.append(f"     NaiveRAG was chosen as the better source")
    
    # Decomposition strategy insights
    lines.append("\n5. Decomposition Strategy Insights:")
    if decomposer_avgs:
        # Analyze patterns
        primary_both = [d for d, _, _ in decomposer_avgs if "primary" in d.lower() and "g_primary" in d.lower() and "n_primary" in d.lower()]
        graphrag_focused = [d for d, _, _ in decomposer_avgs if "g_primary" in d.lower() and "support" in d.lower()]
        naiverag_focused = [d for d, _, _ in decomposer_avgs if "n_primary" in d.lower() and "support" in d.lower()]
        
        if primary_both:
            lines.append(f"   - Patterns sending query to both pipelines as primary: {len(primary_both)} patterns")
        if graphrag_focused:
            lines.append(f"   - Patterns with GraphRAG primary + support: {len(graphrag_focused)} patterns")
        if naiverag_focused:
            lines.append(f"   - Patterns with NaiveRAG primary + support: {len(naiverag_focused)} patterns")
    
    lines.append("")
    lines.append("=" * 100)
    
    return "\n".join(lines)


def main():
    print("Starting Decomposer + Aggregator Decision analysis...")
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
        results_file = OUTPUT_DIR / f"{folder_name}_decomposer_aggregator_results.csv"
        with open(results_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['question_id', 'decomposer_decision', 'aggregator_decision', 'llm_judge_score'])
            for q_id, dec_d, dec_a, score in sorted(results):
                writer.writerow([q_id, dec_d, dec_a, score])
        print(f"  Saved results to: {results_file}")
        
        # Generate and save analysis
        analysis = analyze_results(results, folder_name)
        analysis_file = OUTPUT_DIR / f"{folder_name}_decomposer_aggregator_analysis.txt"
        with open(analysis_file, 'w', encoding='utf-8') as f:
            f.write(analysis)
        print(f"  Saved analysis to: {analysis_file}")
        
        # Also print analysis to console
        print()
        print(analysis)
        print()
    
    print("Decomposer + Aggregator Decision analysis complete!")


if __name__ == "__main__":
    main()