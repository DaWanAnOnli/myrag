#!/usr/bin/env python3
"""
Router Decision Analysis Script
Extracts router decisions from multi-agent RAG log files,
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
CSV_FILE = Path("../../../dataset/4_experiment/4c_experiment_results/new/Individual base multi-agent - approach_1_router.csv")

# Layer 1 folder containing the layer 2 experiment folders
LAYER1_FOLDER = Path("../../4b_retrieval/4b_iv_multi_agents/question_terminal_logs_multi_agent/no_12_approach_1_router")



# Mapping of folder name to CSV column name
FOLDER_TO_COLUMN = {
    "no_12_approach_1_router_5_hops_1250": "approach_1_router_answer score",
}

# Output directory for results
OUTPUT_DIR = Path("router_analysis_results")

# Valid router decisions
VALID_ROUTER_DECISIONS = {"graphrag", "naiverag"}


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


def extract_router_decision_from_log(log_path: Path) -> str:
    """
    Extract the router decision from a multi-agent RAG log file.
    
    Returns one of: "graphrag", "naiverag", or "unknown"
    """
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Method 1: Look for "[Router] Decision=<decision>"
        match = re.search(r'\[Router\]\s+Decision\s*=\s*(\w+)', content, re.IGNORECASE)
        if match:
            decision = match.group(1).strip().lower()
            if decision in VALID_ROUTER_DECISIONS:
                return decision
        
        # Method 2: Look for "- Router decision: <decision>" in summary section
        match = re.search(r'-\s+Router\s+decision:\s*(\w+)', content, re.IGNORECASE)
        if match:
            decision = match.group(1).strip().lower()
            if decision in VALID_ROUTER_DECISIONS:
                return decision
        
        # Method 3: Look for "[Router] Executing <Pipeline> pipeline."
        match = re.search(r'\[Router\]\s+Executing\s+(GraphRAG|NaiveRAG)\s+pipeline', content, re.IGNORECASE)
        if match:
            pipeline = match.group(1).strip().lower()
            if pipeline in VALID_ROUTER_DECISIONS:
                return pipeline
        
        # Method 4: Look in the router JSON output in the final result
        router_match = re.search(
            r'"router":\s*\{[^}]*"decision":\s*"(graphrag|naiverag)"',
            content,
            re.IGNORECASE
        )
        if router_match:
            return router_match.group(1).strip().lower()
        
        # Method 5: Look for fallback decision pattern
        if re.search(r'\[Router\].*Fallback.*decision', content, re.IGNORECASE):
            match = re.search(r'decision\s*=\s*["\']?(graphrag|naiverag)["\']?', 
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
                              scores: Dict[int, int]) -> List[Tuple[int, str, int]]:
    """
    Process all log files in an experiment folder.
    Returns: List of (question_id, router_decision, score) tuples
    """
    results = []
    failed_extractions = []
    
    # Find all .txt log files
    log_files = list(folder_path.glob("*.txt"))
    print(f"  Found {len(log_files)} log files in {folder_path.name}")
    
    # For debugging: show a few sample log entries
    if log_files:
        sample_file = log_files[0]
        print(f"\n  DEBUG: Checking sample file {sample_file.name}")
        try:
            with open(sample_file, 'r', encoding='utf-8') as f:
                content = f.read()
                # Show router-related log lines
                router_lines = [line for line in content.split('\n') if '[Router]' in line]
                if router_lines:
                    print("  Sample router log lines:")
                    for line in router_lines[:5]:
                        print(f"    {line[:150]}")
                else:
                    print("  WARNING: No [Router] lines found in sample file!")
        except Exception as e:
            print(f"  DEBUG ERROR: {e}")
        print()
    
    for log_file in log_files:
        # Extract question ID from filename
        q_id = extract_question_id_from_filename(log_file.name)
        if q_id == -1:
            print(f"  Warning: Could not extract ID from {log_file.name}")
            continue
        
        # Extract router decision
        decision = extract_router_decision_from_log(log_file)
        if decision == "unknown":
            failed_extractions.append((q_id, log_file.name))
        
        # Get score from CSV
        score = scores.get(q_id, -1)
        if score == -1:
            print(f"  Warning: No score found for question ID {q_id}")
        
        results.append((q_id, decision, score))
    
    if failed_extractions:
        print(f"  WARNING: Failed to extract router decision from {len(failed_extractions)} files")
        for q_id, fname in failed_extractions[:5]:
            print(f"    - Question {q_id}: {fname}")
        if len(failed_extractions) > 5:
            print(f"    ... and {len(failed_extractions) - 5} more")
        
        # Show content snippet for debugging the first failed file
        if failed_extractions:
            q_id, fname = failed_extractions[0]
            print(f"  Showing first failed file for debugging:")
            print(f"    Question {q_id}: {fname}")
            try:
                log_path = folder_path / fname
                with open(log_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    router_lines = [line for line in content.split('\n') if 'router' in line.lower()][:10]
                    if router_lines:
                        print("    Router-related lines from this file:")
                        for line in router_lines:
                            print(f"      {line[:150]}")
                    else:
                        # Try to find any pipeline decision
                        pipeline_lines = [line for line in content.split('\n') 
                                        if any(kw in line.lower() for kw in ['graphrag', 'naiverag', 'decision', 'pipeline'])][:10]
                        if pipeline_lines:
                            print("    Pipeline/decision-related lines from this file:")
                            for line in pipeline_lines:
                                print(f"      {line[:150]}")
            except Exception as e:
                print(f"    Could not read file for debugging: {e}")
    
    return results


def analyze_results(results: List[Tuple[int, str, int]], experiment_name: str) -> str:
    """
    Analyze the results and generate a comprehensive report.
    Returns: Analysis text
    """
    lines = []
    lines.append("=" * 100)
    lines.append(f"ROUTER DECISION ANALYSIS REPORT: {experiment_name}")
    lines.append("=" * 100)
    lines.append("")
    
    # Basic statistics
    lines.append(f"Total questions processed: {len(results)}")
    lines.append("")
    
    # Overall score distribution
    lines.append("OVERALL SCORE DISTRIBUTION:")
    lines.append("-" * 50)
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
    
    # Group by router decision
    by_decision = defaultdict(list)
    for q_id, decision, score in results:
        by_decision[decision].append((q_id, score))
    
    # ========== ROUTER DECISION DISTRIBUTION ==========
    lines.append("ROUTER DECISION DISTRIBUTION:")
    lines.append("=" * 100)
    
    decision_sorted = sorted(by_decision.items(), key=lambda x: len(x[1]), reverse=True)
    for decision, items in decision_sorted:
        count = len(items)
        pct = (count / len(results) * 100) if len(results) > 0 else 0
        lines.append(f"\nRouter Decision: {decision.upper()}")
        lines.append(f"  Count: {count:4d} questions ({pct:5.2f}%)")
        
        # Score distribution for this router decision
        dec_score_counts = defaultdict(int)
        for _, score in items:
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
            
            # Success rate (score >= 1)
            success = sum(cnt for score, cnt in dec_score_counts.items() if score >= 1)
            success_rate = (success / total_dec_scored * 100)
            lines.append(f"  Success rate (score >= 1): {success_rate:.2f}% ({success}/{total_dec_scored})")
            
            # Perfect score rate (score == 2)
            perfect = dec_score_counts.get(2, 0)
            perfect_rate = (perfect / total_dec_scored * 100)
            lines.append(f"  Perfect score rate (score == 2): {perfect_rate:.2f}% ({perfect}/{total_dec_scored})")
    
    lines.append("")
    
    # ========== COMPARATIVE ANALYSIS ==========
    lines.append("COMPARATIVE ANALYSIS:")
    lines.append("=" * 100)
    
    # Compare router decisions
    lines.append("\nRouter decisions ranked by average score:")
    lines.append("-" * 50)
    decision_avgs = []
    for decision, items in by_decision.items():
        scores = [score for _, score in items if score >= 0]
        if scores:
            avg = sum(scores) / len(scores)
            decision_avgs.append((decision, avg, len(scores)))
    
    decision_avgs.sort(key=lambda x: x[1], reverse=True)
    for i, (decision, avg, n) in enumerate(decision_avgs, 1):
        lines.append(f"  {i}. {decision.upper()}")
        lines.append(f"     Avg score: {avg:.3f} (n={n})")
    
    # Success rates comparison
    lines.append("\nRouter decisions ranked by success rate (score >= 1):")
    lines.append("-" * 50)
    decision_success = []
    for decision, items in by_decision.items():
        scores = [score for _, score in items if score >= 0]
        if scores:
            success = sum(1 for s in scores if s >= 1)
            rate = success / len(scores)
            decision_success.append((decision, rate, success, len(scores)))
    
    decision_success.sort(key=lambda x: x[1], reverse=True)
    for i, (decision, rate, succ, total) in enumerate(decision_success, 1):
        lines.append(f"  {i}. {decision.upper()}")
        lines.append(f"     Success rate: {rate*100:.1f}% ({succ}/{total})")
    
    # Perfect score rates comparison
    lines.append("\nRouter decisions ranked by perfect score rate (score == 2):")
    lines.append("-" * 50)
    decision_perfect = []
    for decision, items in by_decision.items():
        scores = [score for _, score in items if score >= 0]
        if scores:
            perfect = sum(1 for s in scores if s == 2)
            rate = perfect / len(scores)
            decision_perfect.append((decision, rate, perfect, len(scores)))
    
    decision_perfect.sort(key=lambda x: x[1], reverse=True)
    for i, (decision, rate, perf, total) in enumerate(decision_perfect, 1):
        lines.append(f"  {i}. {decision.upper()}")
        lines.append(f"     Perfect score rate: {rate*100:.1f}% ({perf}/{total})")
    
    # ========== KEY INSIGHTS ==========
    lines.append("\n")
    lines.append("KEY INSIGHTS:")
    lines.append("=" * 100)
    
    if decision_avgs:
        best_dec = decision_avgs[0]
        lines.append(f"\n1. Best Router Decision (by average score):")
        lines.append(f"   {best_dec[0].upper()}")
        lines.append(f"   Average score: {best_dec[1]:.3f} across {best_dec[2]} questions")
    
    if decision_success:
        best_success = decision_success[0]
        lines.append(f"\n2. Best Router Decision (by success rate):")
        lines.append(f"   {best_success[0].upper()}")
        lines.append(f"   Success rate: {best_success[1]*100:.1f}% ({best_success[2]}/{best_success[3]} questions)")
    
    if decision_perfect:
        best_perfect = decision_perfect[0]
        lines.append(f"\n3. Best Router Decision (by perfect score rate):")
        lines.append(f"   {best_perfect[0].upper()}")
        lines.append(f"   Perfect score rate: {best_perfect[1]*100:.1f}% ({best_perfect[2]}/{best_perfect[3]} questions)")
    
    # Detailed comparison if we have both GraphRAG and NaiveRAG
    if len(decision_avgs) >= 2:
        lines.append("\n4. Detailed Comparison:")
        for decision in ["graphrag", "naiverag"]:
            if decision in by_decision:
                items = by_decision[decision]
                scores = [score for _, score in items if score >= 0]
                if scores:
                    avg = sum(scores) / len(scores)
                    success = sum(1 for s in scores if s >= 1)
                    perfect = sum(1 for s in scores if s == 2)
                    fail = sum(1 for s in scores if s == 0)
                    
                    lines.append(f"\n   {decision.upper()}:")
                    lines.append(f"   - Questions: {len(scores)}")
                    lines.append(f"   - Average score: {avg:.3f}")
                    lines.append(f"   - Failed (score=0): {fail} ({(fail/len(scores)*100):.1f}%)")
                    lines.append(f"   - Partial (score=1): {success - perfect} ({((success-perfect)/len(scores)*100):.1f}%)")
                    lines.append(f"   - Perfect (score=2): {perfect} ({(perfect/len(scores)*100):.1f}%)")
                    lines.append(f"   - Success rate (score>=1): {(success/len(scores)*100):.1f}%")
    
    # Distribution comparison
    if len(by_decision) >= 2:
        lines.append("\n5. Score Distribution Comparison:")
        lines.append("   " + "-" * 80)
        lines.append(f"   {'Decision':<15} {'Score=0':<15} {'Score=1':<15} {'Score=2':<15} {'Avg':<10}")
        lines.append("   " + "-" * 80)
        
        for decision in sorted(by_decision.keys()):
            items = by_decision[decision]
            scores = [score for _, score in items if score >= 0]
            if scores:
                s0 = sum(1 for s in scores if s == 0)
                s1 = sum(1 for s in scores if s == 1)
                s2 = sum(1 for s in scores if s == 2)
                avg = sum(scores) / len(scores)
                
                s0_pct = (s0 / len(scores) * 100)
                s1_pct = (s1 / len(scores) * 100)
                s2_pct = (s2 / len(scores) * 100)
                
                lines.append(f"   {decision.upper():<15} "
                           f"{s0:4d} ({s0_pct:4.1f}%)  "
                           f"{s1:4d} ({s1_pct:4.1f}%)  "
                           f"{s2:4d} ({s2_pct:4.1f}%)  "
                           f"{avg:.3f}")
        lines.append("   " + "-" * 80)
    
    lines.append("")
    lines.append("=" * 100)
    
    return "\n".join(lines)


def main():
    print("Starting Router Decision analysis...")
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
        results_file = OUTPUT_DIR / f"{folder_name}_router_results.csv"
        with open(results_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['question_id', 'router_decision', 'llm_judge_score'])
            for q_id, decision, score in sorted(results):
                writer.writerow([q_id, decision, score])
        print(f"  Saved results to: {results_file}")
        
        # Generate and save analysis
        analysis = analyze_results(results, folder_name)
        analysis_file = OUTPUT_DIR / f"{folder_name}_router_analysis.txt"
        with open(analysis_file, 'w', encoding='utf-8') as f:
            f.write(analysis)
        print(f"  Saved analysis to: {analysis_file}")
        
        # Also print analysis to console
        print()
        print(analysis)
        print()
    
    print("Router Decision analysis complete!")


if __name__ == "__main__":
    main()