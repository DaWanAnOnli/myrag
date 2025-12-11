# analyze_llm_calls_iterative.py
"""
Analyzes log files from the Agentic Iterative Multi-Pipeline RAG.
Tracks LLM calls per query including iterations, judge decisions,
query modifications, and aggregator choices.

Updated for the current multi_agent.py implementation:
- Still keyed off the same log markers used in the orchestrator:
    - "Agentic Multi-Iteration RAG run started"
    - "Original Query:"
    - "=== Agentic Multi-Iteration Summary ==="
    - "[Aggregator] Decision=..."
    - "[AnswerJudge] Decision: accepted=..., verdict=..., confidence=..."
    - "[QueryModifier] Modified query:"
    - "[Agent 2 - Naive] Prompt:"
    - "[Agent 2 - GraphRAG] Prompt:"
    - "[Agent 1] Prompt:"
    - "[Agent 1b] Prompt:"
    - "[AnswerJudge] Prompt:"
    - "[Aggregator] Prompt:"
    - "[QueryModifier] Prompt:"
- Per-iteration aggregator decisions are explicitly parsed
  (chosen + confidence) and included both per-iteration and final.
"""

import re
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import csv
from collections import Counter

# -------------------------------------------------------------------
# Paths – adjust these to your current folders if needed
# -------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent

# Folder with *.txt logs from multi_agent.py (iterative mode)
LOG_FOLDER = SCRIPT_DIR / "../../4b_retrieval/4b_iv_multi_agents/question_terminal_logs_multi_agent/no_14_both_1_answer_judge_3_newest"

# Where to dump analysis results
OUTPUT_FOLDER = SCRIPT_DIR / "llm_calls_analysis_results_approach_2_both_1_answer_judge_3"


# -------------------------------------------------------------------
# Basic extractors (query, iterations, per-iteration blocks)
# -------------------------------------------------------------------

def extract_query_from_log(log_path: Path) -> str:
    """Extract the original query from the log file."""
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            for line in f:
                if "Original Query:" in line:
                    # line pattern: "[ts] ... Original Query: <query text>"
                    match = re.search(r'Original Query:\s*(.+)$', line)
                    if match:
                        return match.group(1).strip()
    except Exception as e:
        print(f"Error extracting query from {log_path.name}: {e}")
    return "Unknown Query"


def extract_iterations_used(log_path: Path) -> int:
    """Extract the number of iterations used from the summary section."""
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            content = f.read()

            # Preferred: summary line near the end:
            # "- Total iterations used: X"
            match = re.search(r'- Total iterations used:\s*(\d+)', content)
            if match:
                return int(match.group(1))

            # Fallback: count "========== [Iter i/N] ==========" blocks
            iter_markers = re.findall(r'========== \[Iter (\d+)/\d+\] ==========', content)
            if iter_markers:
                return max(int(i) for i in iter_markers)
    except Exception as e:
        print(f"Error extracting iterations from {log_path.name}: {e}")
    return 0


def extract_per_iteration_decisions(log_path: Path) -> List[Dict[str, Any]]:
    """
    Extract aggregator and judge decisions for each iteration.

    For each iteration i, we return:
      {
        'iteration': i,
        'aggregator_chosen': 'graphrag' | 'naive' | 'mixed' | None,
        'aggregator_confidence': float,
        'judge_accepted': bool | None,
        'judge_verdict': str | None,
        'judge_confidence': float,
        'judge_problems': List[str],
        'judge_recommendations': List[str],
        'query_modified': bool
      }
    """
    iterations: List[Dict[str, Any]] = []

    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            content = f.read()

            # Split by iteration markers.
            # re.split keeps the group, so pattern:
            #  [ "", "1", "<block1>", "2", "<block2>", ...]
            iter_blocks = re.split(r'========== \[Iter (\d+)/\d+\] ==========', content)

            for i in range(1, len(iter_blocks), 2):
                if i + 1 >= len(iter_blocks):
                    break

                iter_num_str = iter_blocks[i]
                iter_content = iter_blocks[i + 1]
                try:
                    iter_num = int(iter_num_str)
                except ValueError:
                    continue

                iteration_data = {
                    'iteration': iter_num,
                    'aggregator_chosen': None,
                    'aggregator_confidence': 0.0,
                    'judge_accepted': None,
                    'judge_verdict': None,
                    'judge_confidence': 0.0,
                    'judge_problems': [],
                    'judge_recommendations': [],
                    'query_modified': False,
                }

                # ---------------- Aggregator info ----------------
                # New logs from aggregator_agent:
                #   [Aggregator] Decision=choose_graphrag | Rationale: ...
                #   then later (for compatibility):
                #   chosen = "graphrag" | "naive" | "mixed"
                #
                # In agentic_multi_iterative we log:
                #   log(f"[Aggregator] Decision={decision} | Rationale: {rationale[:160]}", level="INFO")
                #
                # decision ∈ {choose_graphrag, choose_naiverag, merge}
                # We map:
                #   choose_graphrag -> graphrag
                #   choose_naiverag -> naive
                #   merge -> mixed
                agg_decision_match = re.search(
                    r'\[Aggregator\] Decision=([a-zA-Z_]+)',
                    iter_content
                )
                chosen_from_decision: Optional[str] = None
                if agg_decision_match:
                    raw_decision = agg_decision_match.group(1).strip().lower()
                    if raw_decision == "choose_graphrag":
                        chosen_from_decision = "graphrag"
                    elif raw_decision == "choose_naiverag":
                        chosen_from_decision = "naive"
                    elif raw_decision == "merge":
                        chosen_from_decision = "mixed"

                # Older / compatibility logs may also have:
                #   "[Aggregator] Decision: chosen=<label> | Rationale..."
                agg_chosen_match = re.search(
                    r'\[Aggregator\]\s+Decision:\s+chosen=([a-zA-Z_]+)',
                    iter_content
                )
                if agg_chosen_match:
                    iteration_data['aggregator_chosen'] = agg_chosen_match.group(1).strip()
                elif chosen_from_decision is not None:
                    iteration_data['aggregator_chosen'] = chosen_from_decision

                # Confidence:
                #   "[Aggregator] Decision: chosen=..., confidence=0.7"
                agg_conf_match = re.search(
                    r'\[Aggregator\].*?confidence=([\d.]+)',
                    iter_content
                )
                if agg_conf_match:
                    try:
                        iteration_data['aggregator_confidence'] = float(agg_conf_match.group(1))
                    except ValueError:
                        pass

                # ---------------- Answer Judge info ----------------
                # From answer_judge_agent:
                #   log(f"[AnswerJudge] Decision: accepted={accepted}, verdict={verdict}, confidence={conf:.2f}", ...)
                judge_dec_match = re.search(
                    r'\[AnswerJudge\]\s+Decision:\s+accepted=(\w+),\s+verdict=([a-zA-Z_]+),\s+confidence=([\d.]+)',
                    iter_content
                )
                if judge_dec_match:
                    accepted_str = judge_dec_match.group(1).strip().lower()
                    verdict = judge_dec_match.group(2).strip()
                    conf_str = judge_dec_match.group(3).strip()
                    iteration_data['judge_accepted'] = (accepted_str == 'true')
                    iteration_data['judge_verdict'] = verdict
                    try:
                        iteration_data['judge_confidence'] = float(conf_str)
                    except ValueError:
                        pass
                else:
                    # Fallback to earlier style if present:
                    judge_accept_match = re.search(
                        r'\[AnswerJudge\]\s+Decision:\s+accepted=(\w+)',
                        iter_content
                    )
                    if judge_accept_match:
                        iteration_data['judge_accepted'] = (
                            judge_accept_match.group(1).strip().lower() == 'true'
                        )
                    judge_verdict_match = re.search(
                        r'\[AnswerJudge\].*?verdict=([a-zA-Z_]+)',
                        iter_content
                    )
                    if judge_verdict_match:
                        iteration_data['judge_verdict'] = judge_verdict_match.group(1).strip()
                    judge_conf_match = re.search(
                        r'\[AnswerJudge\].*?confidence=([\d.]+)',
                        iter_content
                    )
                    if judge_conf_match:
                        try:
                            iteration_data['judge_confidence'] = float(judge_conf_match.group(1))
                        except ValueError:
                            pass

                # Problems:
                #   log(f"[AnswerJudge] Problems: {problems}", ...)
                problems_match = re.search(
                    r'\[AnswerJudge\]\s+Problems:\s*\[([^\]]*)\]',
                    iter_content
                )
                if problems_match:
                    problems_str = problems_match.group(1)
                    if problems_str.strip():
                        iteration_data['judge_problems'] = [
                            p.strip().strip("'\"")
                            for p in problems_str.split(',')
                            if p.strip()
                        ]

                # Recommendations:
                recs_match = re.search(
                    r'\[AnswerJudge\]\s+Recommendations:\s*\[([^\]]*)\]',
                    iter_content
                )
                if recs_match:
                    recs_str = recs_match.group(1)
                    if recs_str.strip():
                        iteration_data['judge_recommendations'] = [
                            r.strip().strip("'\"")
                            for r in recs_str.split(',')
                            if r.strip()
                        ]

                # ---------------- Query Modifier info ----------------
                # If not accepted and not last iteration, QueryModifier is called:
                #   log("[QueryModifier] Prompt:", ...)
                #   log(f"[QueryModifier] Modified query: {modified}", ...)
                if '[QueryModifier] Modified query:' in iter_content:
                    iteration_data['query_modified'] = True

                iterations.append(iteration_data)

    except Exception as e:
        print(f"Error extracting per-iteration decisions from {log_path.name}: {e}")

    return iterations


# -------------------------------------------------------------------
# Core log analysis per file
# -------------------------------------------------------------------

def analyze_log_file(log_path: Path) -> Dict[str, Any]:
    """
    Analyze a single log file and count LLM calls and embeddings.

    Notes:
      - Counts all LLM calls including all iterations.
      - Agent 1 / 1b are only counted for GraphRAG.
      - Per requirements, Agent 1/1b in Naive RAG (if they ever appear)
        are ignored by design; we only rely on explicit "[Agent X - Naive]"
        markers for Agent 2.
    """
    result: Dict[str, Any] = {
        'log_file': log_path.name,
        'query': '',
        'mode': 'unknown',

        'iterations_used': 0,

        'agent1_calls': 0,          # Agent 1 (GraphRAG entity extractor)
        'agent1b_calls': 0,         # Agent 1b (GraphRAG triple extractor)
        'agent2_naive_calls': 0,    # Agent 2 in Naive RAG
        'agent2_graphrag_calls': 0, # Agent 2 in GraphRAG
        'aggregator_calls': 0,
        'judge_calls': 0,
        'query_modifier_calls': 0,

        'total_llm_calls': 0,
        'embed_calls': 0,

        # Final decisions (from last iteration)
        'final_judge_accepted': None,
        'final_judge_verdict': None,
        'final_aggregator_chosen': None,
        'final_aggregator_confidence': 0.0,

        # Per-iteration breakdown
        'per_iteration_decisions': [],
    }

    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            content = f.read()

            # ---------------- Basic metadata ----------------
            result['query'] = extract_query_from_log(log_path)

            if "Agentic Multi-Iteration RAG run started" in content:
                result['mode'] = 'iterative'

            result['iterations_used'] = extract_iterations_used(log_path)

            # ---------------- Per-iteration decisions ----------------
            per_iter = extract_per_iteration_decisions(log_path)
            result['per_iteration_decisions'] = per_iter

            # Final decisions from the last iteration
            if per_iter:
                last_iter = per_iter[-1]
                result['final_judge_accepted'] = last_iter.get('judge_accepted')
                result['final_judge_verdict'] = last_iter.get('judge_verdict')
                result['final_aggregator_chosen'] = last_iter.get('aggregator_chosen')
                result['final_aggregator_confidence'] = last_iter.get('aggregator_confidence', 0.0)

            # ---------------- LLM call counting ----------------
            # Agent 1 / 1b (GraphRAG only)
            result['agent1_calls'] = len(re.findall(r'\[Agent 1\] Prompt:', content))
            result['agent1b_calls'] = len(re.findall(r'\[Agent 1b\] Prompt:', content))

            # Agent 2 in Naive vs GraphRAG
            agent2_naive = 0
            agent2_graph = 0
            for m in re.finditer(r'\[Agent 2 - (Naive|GraphRAG)\] Prompt:', content):
                pipeline = m.group(1)
                if pipeline == 'Naive':
                    agent2_naive += 1
                elif pipeline == 'GraphRAG':
                    agent2_graph += 1
            result['agent2_naive_calls'] = agent2_naive
            result['agent2_graphrag_calls'] = agent2_graph

            # Aggregator, Judge, Query Modifier prompts (one per call)
            result['aggregator_calls'] = len(re.findall(r'\[Aggregator\] Prompt:', content))
            result['judge_calls'] = len(re.findall(r'\[AnswerJudge\] Prompt:', content))
            result['query_modifier_calls'] = len(re.findall(r'\[QueryModifier\] Prompt:', content))

            # Total LLM calls
            result['total_llm_calls'] = (
                result['agent1_calls']
                + result['agent1b_calls']
                + result['agent2_naive_calls']
                + result['agent2_graphrag_calls']
                + result['aggregator_calls']
                + result['judge_calls']
                + result['query_modifier_calls']
            )

            # Embedding calls – any "[Embed]" log line
            result['embed_calls'] = len(re.findall(r'\[Embed\]', content))

    except Exception as e:
        print(f"Error analyzing {log_path.name}: {e}")

    return result


# -------------------------------------------------------------------
# Batch analysis
# -------------------------------------------------------------------

def analyze_all_logs() -> List[Dict[str, Any]]:
    """Analyze all .txt log files in LOG_FOLDER."""
    log_folder = LOG_FOLDER.resolve()
    if not log_folder.exists():
        print(f"Error: Log folder not found: {log_folder}")
        return []

    log_files = sorted(log_folder.glob("*.txt"))
    if not log_files:
        print(f"Warning: No .txt files found in {log_folder}")
        return []

    print(f"Found {len(log_files)} log files to analyze...")
    results: List[Dict[str, Any]] = []
    for i, log_file in enumerate(log_files, 1):
        print(f"Analyzing {i}/{len(log_files)}: {log_file.name}")
        results.append(analyze_log_file(log_file))
    return results


# -------------------------------------------------------------------
# Summaries and reports (mostly reused, with minor tweaks)
# -------------------------------------------------------------------

def create_iterative_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Create comprehensive summary statistics for iterative runs only."""
    iterative_results = [r for r in results if r['mode'] == 'iterative']

    summary: Dict[str, Any] = {
        'total_queries': len(iterative_results),
        'iteration_stats': {
            'total': 0,
            'mean': 0.0,
            'median': 0.0,
            'min': 0,
            'max': 0,
            'distribution': Counter()
        },
        'judge_verdicts': Counter(),
        'aggregator_choices': Counter(),
        'acceptance_by_iteration': {},
        'avg_llm_calls_per_query': 0.0,
        'avg_llm_calls_per_iteration': 0.0,
        'common_problems': Counter(),
        'common_recommendations': Counter(),
        'by_iterations': {}
    }

    if not iterative_results:
        return summary

    # Iteration statistics
    iteration_counts = [r['iterations_used'] for r in iterative_results if r['iterations_used'] > 0]
    if iteration_counts:
        import statistics
        summary['iteration_stats']['total'] = sum(iteration_counts)
        summary['iteration_stats']['mean'] = statistics.mean(iteration_counts)
        summary['iteration_stats']['median'] = statistics.median(iteration_counts)
        summary['iteration_stats']['min'] = min(iteration_counts)
        summary['iteration_stats']['max'] = max(iteration_counts)
        summary['iteration_stats']['distribution'] = Counter(iteration_counts)

    # Collect decisions
    for r in iterative_results:
        # Final decisions (from last iteration)
        if r.get('final_judge_verdict'):
            summary['judge_verdicts'][r['final_judge_verdict']] += 1
        if r.get('final_aggregator_chosen'):
            summary['aggregator_choices'][r['final_aggregator_chosen']] += 1

        # Per-iteration stats
        for iter_data in r.get('per_iteration_decisions', []):
            iter_num = iter_data['iteration']

            # Acceptance by iteration
            if iter_num not in summary['acceptance_by_iteration']:
                summary['acceptance_by_iteration'][iter_num] = {'accepted': 0, 'rejected': 0}
            if iter_data.get('judge_accepted'):
                summary['acceptance_by_iteration'][iter_num]['accepted'] += 1
            else:
                summary['acceptance_by_iteration'][iter_num]['rejected'] += 1

            # Problems / recommendations
            for prob in iter_data.get('judge_problems', []):
                if prob:
                    summary['common_problems'][prob] += 1
            for rec in iter_data.get('judge_recommendations', []):
                if rec:
                    summary['common_recommendations'][rec] += 1

        # Group by total iteration count
        iters = r['iterations_used']
        if iters not in summary['by_iterations']:
            summary['by_iterations'][iters] = {
                'count': 0,
                'avg_llm_calls': 0.0,
                'queries': []
            }
        summary['by_iterations'][iters]['count'] += 1
        summary['by_iterations'][iters]['queries'].append({
            'query': r['query'][:100],
            'llm_calls': r['total_llm_calls'],
            'final_accepted': r.get('final_judge_accepted'),
            'final_verdict': r.get('final_judge_verdict')
        })

    # Avg LLM calls
    if iterative_results:
        total_llm = sum(r['total_llm_calls'] for r in iterative_results)
        summary['avg_llm_calls_per_query'] = total_llm / len(iterative_results)
        total_iters = sum(r['iterations_used'] for r in iterative_results if r['iterations_used'] > 0)
        if total_iters > 0:
            summary['avg_llm_calls_per_iteration'] = total_llm / total_iters

    # Avg LLM calls by iteration count
    for iters, data in summary['by_iterations'].items():
        qs = data['queries']
        if qs:
            data['avg_llm_calls'] = sum(q['llm_calls'] for q in qs) / len(qs)

    return summary


def save_results(results: List[Dict[str, Any]]) -> None:
    """Save analysis results and summaries to OUTPUT_FOLDER."""
    output_folder = OUTPUT_FOLDER.resolve()
    output_folder.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1) Detailed CSV
    csv_path = output_folder / f"llm_calls_detailed_{ts}.csv"
    csv_fields = [
        'log_file', 'query', 'mode', 'iterations_used',
        'total_llm_calls', 'agent1_calls', 'agent1b_calls',
        'agent2_naive_calls', 'agent2_graphrag_calls',
        'aggregator_calls', 'judge_calls', 'query_modifier_calls',
        'final_judge_accepted', 'final_judge_verdict',
        'final_aggregator_chosen', 'final_aggregator_confidence',
        'embed_calls'
    ]
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(results)
    print(f"Detailed results saved to: {csv_path}")

    # 2) Full JSON (includes per-iteration decisions)
    json_path = output_folder / f"llm_calls_detailed_{ts}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"JSON results saved to: {json_path}")

    # 3) Iterative summary JSON
    iter_summary = create_iterative_summary(results)
    iter_summary_path = output_folder / f"iterative_summary_{ts}.json"
    with open(iter_summary_path, 'w', encoding='utf-8') as f:
        summary_copy = iter_summary.copy()
        summary_copy['iteration_stats']['distribution'] = dict(iter_summary['iteration_stats']['distribution'])
        summary_copy['judge_verdicts'] = dict(iter_summary['judge_verdicts'])
        summary_copy['aggregator_choices'] = dict(iter_summary['aggregator_choices'])
        summary_copy['common_problems'] = dict(iter_summary['common_problems'])
        summary_copy['common_recommendations'] = dict(iter_summary['common_recommendations'])
        json.dump(summary_copy, f, indent=2, ensure_ascii=False)
    print(f"Iterative summary saved to: {iter_summary_path}")

    # 4) Iterative stats CSV
    iter_stats_csv = output_folder / f"iterative_statistics_{ts}.csv"
    with open(iter_stats_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Value'])
        writer.writerow(['Total Queries', iter_summary['total_queries']])
        writer.writerow([''])
        writer.writerow(['Iteration Statistics:', ''])
        writer.writerow(['  Total Iterations', iter_summary['iteration_stats']['total']])
        writer.writerow(['  Mean', f"{iter_summary['iteration_stats']['mean']:.2f}"])
        writer.writerow(['  Median', f"{iter_summary['iteration_stats']['median']:.2f}"])
        writer.writerow(['  Min/Max', f"{iter_summary['iteration_stats']['min']}/{iter_summary['iteration_stats']['max']}"])
        writer.writerow([''])
        writer.writerow(['Iteration Distribution:', ''])
        for iters, count in sorted(iter_summary['iteration_stats']['distribution'].items()):
            writer.writerow([f'  {iters} iterations', count])
        writer.writerow([''])
        writer.writerow(['Judge Verdicts:', ''])
        for verdict, count in iter_summary['judge_verdicts'].items():
            writer.writerow([f'  {verdict}', count])
        writer.writerow([''])
        writer.writerow(['Aggregator Choices:', ''])
        for choice, count in iter_summary['aggregator_choices'].items():
            writer.writerow([f'  {choice}', count])
        writer.writerow([''])
        writer.writerow(['LLM Calls:', ''])
        writer.writerow(['  Avg per Query', f"{iter_summary['avg_llm_calls_per_query']:.2f}"])
        writer.writerow(['  Avg per Iteration', f"{iter_summary['avg_llm_calls_per_iteration']:.2f}"])
    print(f"Iterative statistics CSV saved to: {iter_stats_csv}")

    # 5) Human-readable text report
    readable_path = output_folder / f"iterative_readable_{ts}.txt"
    with open(readable_path, 'w', encoding='utf-8') as f:
        separator = "=" * 80
        title = "AGENTIC ITERATIVE RAG DECISIONS - HUMAN READABLE"
        f.write(separator + "\n")
        f.write(title + "\n")
        f.write(separator + "\n")

        # Summary
        f.write("SUMMARY STATISTICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total queries analyzed: {iter_summary['total_queries']}\n")
        f.write("Iteration Statistics:\n")
        f.write(f"  Total iterations: {iter_summary['iteration_stats']['total']}\n")
        f.write(f"  Average: {iter_summary['iteration_stats']['mean']:.2f}\n")
        f.write(f"  Median: {iter_summary['iteration_stats']['median']:.2f}\n")
        f.write(f"  Range: {iter_summary['iteration_stats']['min']}-{iter_summary['iteration_stats']['max']}\n")

        f.write("Iteration Distribution:\n")
        for iters, count in sorted(iter_summary['iteration_stats']['distribution'].items()):
            pct = (count / iter_summary['total_queries'] * 100) if iter_summary['total_queries'] > 0 else 0
            f.write(f"  {iters} iterations: {count} queries ({pct:.1f}%)\n")

        f.write("\nJudge Verdicts:\n")
        for verdict, count in iter_summary['judge_verdicts'].items():
            pct = (count / iter_summary['total_queries'] * 100) if iter_summary['total_queries'] > 0 else 0
            f.write(f"  {verdict}: {count} ({pct:.1f}%)\n")

        f.write("\nAggregator Choices:\n")
        for choice, count in iter_summary['aggregator_choices'].items():
            pct = (count / iter_summary['total_queries'] * 100) if iter_summary['total_queries'] > 0 else 0
            f.write(f"  {choice}: {count} ({pct:.1f}%)\n")

        f.write("\nAcceptance by Iteration:\n")
        for iter_num in sorted(iter_summary['acceptance_by_iteration'].keys()):
            data = iter_summary['acceptance_by_iteration'][iter_num]
            total = data['accepted'] + data['rejected']
            accept_pct = (data['accepted'] / total * 100) if total > 0 else 0
            f.write(f"  Iteration {iter_num}: {data['accepted']} accepted, {data['rejected']} rejected ({accept_pct:.1f}% accepted)\n")

        f.write(f"\nLLM Call Statistics:\n")
        f.write(f"  Average per query: {iter_summary['avg_llm_calls_per_query']:.2f}\n")
        f.write(f"  Average per iteration: {iter_summary['avg_llm_calls_per_iteration']:.2f}\n")

        f.write("\nMost Common Problems:\n")
        for prob, count in iter_summary['common_problems'].most_common(10):
            f.write(f"  {prob}: {count}\n")

        f.write("\nMost Common Recommendations:\n")
        for rec, count in iter_summary['common_recommendations'].most_common(10):
            f.write(f"  {rec}: {count}\n")

        f.write("\n" + separator + "\n")
        f.write("INDIVIDUAL QUERY DETAILS\n")
        f.write("-" * 80 + "\n")

        for i, r in enumerate(results, 1):
            if r.get('mode') == 'iterative':
                f.write(f"[{i}] {r['log_file']}\n")
                f.write(f"Query: {r['query']}\n")
                f.write(f"Iterations: {r['iterations_used']} | LLM Calls: {r['total_llm_calls']}\n")
                f.write(
                    f"Final: Judge={r['final_judge_verdict']} "
                    f"(accepted={r['final_judge_accepted']}), "
                    f"Agg={r['final_aggregator_chosen']} "
                    f"(conf={r.get('final_aggregator_confidence',0.0):.2f})\n"
                )

                if r.get('per_iteration_decisions'):
                    f.write("  Per-Iteration Decisions:\n")
                    for iter_data in r['per_iteration_decisions']:
                        f.write(
                            f"    Iter {iter_data['iteration']}: "
                            f"Agg={iter_data['aggregator_chosen']} "
                            f"({iter_data['aggregator_confidence']:.2f}), "
                            f"Judge={iter_data['judge_verdict']} "
                            f"(accepted={iter_data['judge_accepted']}, "
                            f"conf={iter_data['judge_confidence']:.2f})"
                        )
                        if iter_data.get('query_modified'):
                            f.write(", Query Modified")
                        f.write("\n")

                f.write("-" * 80 + "\n")
    print(f"Human-readable report saved to: {readable_path}")

    # 6) High-level summary
    summary_text = create_summary(results, iter_summary)
    summary_path = output_folder / f"llm_calls_summary_{ts}.txt"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(summary_text)
    print(f"Summary saved to: {summary_path}")
    print(summary_text)


def create_summary(results: List[Dict[str, Any]], iter_summary: Dict[str, Any]) -> str:
    """Create a concise text summary of the analysis."""
    if not results:
        return "No results to summarize."

    total_files = len(results)
    iter_results = [r for r in results if r['mode'] == 'iterative']
    other_results = [r for r in results if r['mode'] != 'iterative']

    total_llm_calls = sum(r['total_llm_calls'] for r in results)
    total_agent1 = sum(r['agent1_calls'] for r in results)
    total_agent1b = sum(r['agent1b_calls'] for r in results)
    total_agent2_naive = sum(r['agent2_naive_calls'] for r in results)
    total_agent2_graphrag = sum(r['agent2_graphrag_calls'] for r in results)
    total_aggregator = sum(r['aggregator_calls'] for r in results)
    total_judge = sum(r['judge_calls'] for r in results)
    total_modifier = sum(r['query_modifier_calls'] for r in results)
    total_embed_calls = sum(r['embed_calls'] for r in results)

    avg_llm_per_query = total_llm_calls / total_files if total_files > 0 else 0.0
    avg_embed_per_query = total_embed_calls / total_files if total_files > 0 else 0.0

    min_llm = min((r['total_llm_calls'] for r in results), default=0)
    max_llm = max((r['total_llm_calls'] for r in results), default=0)

    header = """
╔══════════════════════════════════════════════════════════════╗
║   AGENTIC ITERATIVE MULTI-PIPELINE RAG LLM CALL ANALYSIS    ║
╚══════════════════════════════════════════════════════════════╝
"""

    overall_section = f"""
Overall Statistics:
- Total log files analyzed: {total_files}
  - Iterative mode: {len(iter_results)}
  - Other modes: {len(other_results)}
- Total LLM calls: {total_llm_calls}
  - Agent 1 (entity extraction): {total_agent1}
  - Agent 1b (triple extraction): {total_agent1b}
  - Agent 2 Naive: {total_agent2_naive}
  - Agent 2 GraphRAG: {total_agent2_graphrag}
  - Aggregator: {total_aggregator}
  - Answer Judge: {total_judge}
  - Query Modifier: {total_modifier}
- Total embedding calls: {total_embed_calls}
"""

    averages_section = f"""
Per-Query Averages:
- Average LLM calls per query: {avg_llm_per_query:.2f}
- Average embedding calls per query: {avg_embed_per_query:.2f}
- Average iterations per query (iterative only): {iter_summary['iteration_stats']['mean']:.2f}
"""

    ranges_section = f"""
Ranges:
- Minimum LLM calls in a query: {min_llm}
- Maximum LLM calls in a query: {max_llm}
- Minimum iterations: {iter_summary['iteration_stats']['min']}
- Maximum iterations: {iter_summary['iteration_stats']['max']}
"""

    iter_section = ""
    if iter_results and iter_summary['total_queries'] > 0:
        iter_section = "\nIteration Analysis:\n"
        iter_section += "Iteration Distribution:\n"
        for iters, count in sorted(iter_summary['iteration_stats']['distribution'].items()):
            pct = (count / iter_summary['total_queries']) * 100
            bar = "█" * int(pct / 2)
            iter_section += f"  - {iters} iterations: {count} queries ({pct:.1f}%) {bar}\n"

        iter_section += f"""
Iteration Stats:
- Total iterations: {iter_summary['iteration_stats']['total']}
- Mean: {iter_summary['iteration_stats']['mean']:.2f}
- Median: {iter_summary['iteration_stats']['median']:.2f}
- Range: {iter_summary['iteration_stats']['min']}-{iter_summary['iteration_stats']['max']}

Judge Verdicts:
"""
        for verdict, count in iter_summary['judge_verdicts'].most_common():
            pct = (count / iter_summary['total_queries']) * 100
            iter_section += f"  - {verdict}: {count} ({pct:.1f}%)\n"

        iter_section += "\nAggregator Choices:\n"
        for choice, count in iter_summary['aggregator_choices'].most_common():
            pct = (count / iter_summary['total_queries']) * 100
            iter_section += f"  - {choice}: {count} ({pct:.1f}%)\n"

        iter_section += f"""
LLM Call Statistics:
- Average per query: {iter_summary['avg_llm_calls_per_query']:.2f}
- Average per iteration: {iter_summary['avg_llm_calls_per_iteration']:.2f}
"""

    return header + overall_section + averages_section + ranges_section + iter_section


def main() -> None:
    """Main entry point."""
    sep = "=" * 70
    title = "AGENTIC ITERATIVE MULTI-PIPELINE RAG LLM CALL ANALYZER"
    subtitle = "(Tracking iterations, judge decisions, query refinement, aggregator choices)"

    print(sep)
    print(title)
    print(subtitle)
    print(sep)
    print(f"Input folder:  {LOG_FOLDER.resolve()}")
    print(f"Output folder: {OUTPUT_FOLDER.resolve()}")
    print()

    results = analyze_all_logs()
    if not results:
        print("No results to save. Exiting.")
        return

    save_results(results)

    print(sep)
    print("Analysis complete.")
    print(sep)


if __name__ == "__main__":
    main()