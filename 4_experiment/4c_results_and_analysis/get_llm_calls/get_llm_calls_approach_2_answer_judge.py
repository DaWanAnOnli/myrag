# analyze_llm_calls_iterative.py
"""
Analyzes log files from the Agentic Iterative Multi-Pipeline RAG.
Tracks LLM calls per query including iterations, judge decisions, and query modifications.
Ignores Agent 1/1b calls in Naive RAG pipeline per user request.
"""

import re
import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import csv
from collections import Counter

# Relative paths from this script's location
SCRIPT_DIR = Path(__file__).resolve().parent
LOG_FOLDER = SCRIPT_DIR / "../../4b_retrieval/4b_iv_multi_agents/question_terminal_logs_multi_agent/approach_2_both_5_answer_judge_5_hops_1250"
OUTPUT_FOLDER = SCRIPT_DIR / "llm_calls_analysis_results_approach_2_both_5_answer_judge_5_hops_1250"


def extract_query_from_log(log_path: Path) -> str:
    """Extract the original query from the log file."""
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            for line in f:
                if "Original Query:" in line:
                    match = re.search(r'Original Query:\s*(.+)$', line)
                    if match:
                        return match.group(1).strip()
    except Exception as e:
        print(f"Error extracting query from {log_path.name}: {e}")
    return "Unknown Query"


def extract_iterations_used(log_path: Path) -> int:
    """Extract the number of iterations used."""
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
            # Look for the summary line
            match = re.search(r'- Total iterations used:\s*(\d+)', content)
            if match:
                return int(match.group(1))
            
            # Fallback: count iteration markers
            iter_markers = re.findall(r'========== \[Iter (\d+)/\d+\] ==========', content)
            if iter_markers:
                return max(int(i) for i in iter_markers)
    
    except Exception as e:
        print(f"Error extracting iterations from {log_path.name}: {e}")
    
    return 0


def extract_per_iteration_decisions(log_path: Path) -> List[Dict[str, any]]:
    """Extract aggregator and judge decisions for each iteration."""
    iterations = []
    
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
            # Find all iteration blocks
            iter_blocks = re.split(r'========== \[Iter (\d+)/\d+\] ==========', content)
            
            # Process pairs: (iteration_num, content)
            for i in range(1, len(iter_blocks), 2):
                if i + 1 >= len(iter_blocks):
                    break
                
                iter_num = int(iter_blocks[i])
                iter_content = iter_blocks[i + 1]
                
                iteration_data = {
                    'iteration': iter_num,
                    'aggregator_chosen': None,
                    'aggregator_confidence': 0.0,
                    'judge_accepted': None,
                    'judge_verdict': None,
                    'judge_confidence': 0.0,
                    'judge_problems': [],
                    'judge_recommendations': [],
                    'query_modified': False
                }
                
                # Extract aggregator decision
                agg_match = re.search(r'\[Aggregator\] Decision: chosen=(\w+)', iter_content)
                if agg_match:
                    iteration_data['aggregator_chosen'] = agg_match.group(1)
                
                agg_conf = re.search(r'\[Aggregator\].*?confidence=([\d.]+)', iter_content)
                if agg_conf:
                    iteration_data['aggregator_confidence'] = float(agg_conf.group(1))
                
                # Extract judge decision
                judge_accept = re.search(r'\[AnswerJudge\] Decision: accepted=(\w+)', iter_content)
                if judge_accept:
                    iteration_data['judge_accepted'] = judge_accept.group(1).lower() == 'true'
                
                judge_verdict = re.search(r'verdict=(\w+)', iter_content)
                if judge_verdict:
                    iteration_data['judge_verdict'] = judge_verdict.group(1)
                
                judge_conf = re.search(r'\[AnswerJudge\].*?confidence=([\d.]+)', iter_content)
                if judge_conf:
                    iteration_data['judge_confidence'] = float(judge_conf.group(1))
                
                # Extract problems and recommendations
                problems_match = re.search(r'\[AnswerJudge\] Problems:\s*\[([^\]]+)\]', iter_content)
                if problems_match:
                    problems_str = problems_match.group(1)
                    iteration_data['judge_problems'] = [p.strip().strip("'\"") for p in problems_str.split(',')]
                
                recs_match = re.search(r'\[AnswerJudge\] Recommendations:\s*\[([^\]]+)\]', iter_content)
                if recs_match:
                    recs_str = recs_match.group(1)
                    iteration_data['judge_recommendations'] = [r.strip().strip("'\"") for r in recs_str.split(',')]
                
                # Check if query was modified
                if '[QueryModifier] Modified query:' in iter_content:
                    iteration_data['query_modified'] = True
                
                iterations.append(iteration_data)
    
    except Exception as e:
        print(f"Error extracting per-iteration decisions from {log_path.name}: {e}")
    
    return iterations


def analyze_log_file(log_path: Path) -> Dict:
    """
    Analyze a single log file and count LLM calls.
    
    Returns dict with metrics, counting all LLM calls including iterations.
    Ignores Agent 1/1b in Naive RAG pipeline.
    """
    
    result = {
        'log_file': log_path.name,
        'query': '',
        'iterations_used': 0,
        'agent1_calls': 0,  # Agent 1 (only in GraphRAG)
        'agent1b_calls': 0,  # Agent 1b (only in GraphRAG)
        'agent2_naive_calls': 0,  # Agent 2 in Naive
        'agent2_graphrag_calls': 0,  # Agent 2 in GraphRAG
        'aggregator_calls': 0,  # Aggregator (once per iteration)
        'judge_calls': 0,  # Answer Judge (once per iteration)
        'query_modifier_calls': 0,  # Query Modifier (if not accepted)
        'total_llm_calls': 0,
        'embed_calls': 0,
        'final_judge_accepted': None,
        'final_judge_verdict': None,
        'final_aggregator_chosen': None,
        'per_iteration_decisions': [],
        'mode': 'unknown'
    }
    
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
            # Extract query
            result['query'] = extract_query_from_log(log_path)
            
            # Detect mode
            if "Agentic Multi-Iteration RAG run started" in content:
                result['mode'] = 'iterative'
            
            # Extract iterations used
            result['iterations_used'] = extract_iterations_used(log_path)
            
            # Extract per-iteration decisions
            result['per_iteration_decisions'] = extract_per_iteration_decisions(log_path)
            
            # Get final decisions (from last iteration)
            if result['per_iteration_decisions']:
                last_iter = result['per_iteration_decisions'][-1]
                result['final_judge_accepted'] = last_iter.get('judge_accepted')
                result['final_judge_verdict'] = last_iter.get('judge_verdict')
                result['final_aggregator_chosen'] = last_iter.get('aggregator_chosen')
            
            # Count Agent 1 prompts
            # Agent 1 only appears in GraphRAG (never in Naive RAG by design)
            agent1_matches = re.findall(r'\[Agent 1\] Prompt:', content)
            result['agent1_calls'] = len(agent1_matches)
            
            # Count Agent 1b prompts
            # Agent 1b only appears in GraphRAG (never in Naive RAG by design)
            agent1b_matches = re.findall(r'\[Agent 1b\] Prompt:', content)
            result['agent1b_calls'] = len(agent1b_matches)
            
            # Count Agent 2 prompts by explicit pipeline markers
            agent2_naive = 0
            agent2_graphrag = 0
            for match in re.finditer(r'\[Agent 2 - (Naive|GraphRAG)\] Prompt:', content):
                pipeline = match.group(1)
                if pipeline == 'Naive':
                    agent2_naive += 1
                elif pipeline == 'GraphRAG':
                    agent2_graphrag += 1
            
            result['agent2_naive_calls'] = agent2_naive
            result['agent2_graphrag_calls'] = agent2_graphrag
            
            # Count Aggregator prompts
            aggregator_matches = re.findall(r'\[Aggregator\] Prompt:', content)
            result['aggregator_calls'] = len(aggregator_matches)
            
            # Count Answer Judge prompts
            judge_matches = re.findall(r'\[AnswerJudge\] Prompt:', content)
            result['judge_calls'] = len(judge_matches)
            
            # Count Query Modifier prompts
            modifier_matches = re.findall(r'\[QueryModifier\] Prompt:', content)
            result['query_modifier_calls'] = len(modifier_matches)
            
            # Total LLM calls
            result['total_llm_calls'] = (
                result['agent1_calls'] +
                result['agent1b_calls'] +
                result['agent2_naive_calls'] +
                result['agent2_graphrag_calls'] +
                result['aggregator_calls'] +
                result['judge_calls'] +
                result['query_modifier_calls']
            )
            
            # Count embedding calls
            embed_matches = re.findall(r'\[Embed\]', content)
            result['embed_calls'] = len(embed_matches)
            
    except Exception as e:
        print(f"Error analyzing {log_path.name}: {e}")
    
    return result

def analyze_all_logs() -> List[Dict]:
    """Analyze all log files in the input folder."""
    
    log_folder = LOG_FOLDER.resolve()
    
    if not log_folder.exists():
        print(f"Error: Log folder not found: {log_folder}")
        return []
    
    log_files = sorted(log_folder.glob("*.txt"))
    
    if not log_files:
        print(f"Warning: No .txt files found in {log_folder}")
        return []
    
    print(f"Found {len(log_files)} log files to analyze...")
    
    results = []
    for i, log_file in enumerate(log_files, 1):
        print(f"Analyzing {i}/{len(log_files)}: {log_file.name}")
        analysis = analyze_log_file(log_file)
        results.append(analysis)
    
    return results


def create_iterative_summary(results: List[Dict]) -> Dict[str, any]:
    """Create comprehensive iterative RAG summary statistics."""
    
    iterative_results = [r for r in results if r['mode'] == 'iterative']
    
    summary = {
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
    
    # Collect decisions across all iterations
    for r in iterative_results:
        # Final decisions
        if r.get('final_judge_verdict'):
            summary['judge_verdicts'][r['final_judge_verdict']] += 1
        if r.get('final_aggregator_chosen'):
            summary['aggregator_choices'][r['final_aggregator_chosen']] += 1
        
        # Per-iteration analysis
        for iter_data in r.get('per_iteration_decisions', []):
            iter_num = iter_data['iteration']
            
            # Track acceptance by iteration
            if iter_num not in summary['acceptance_by_iteration']:
                summary['acceptance_by_iteration'][iter_num] = {'accepted': 0, 'rejected': 0}
            
            if iter_data.get('judge_accepted'):
                summary['acceptance_by_iteration'][iter_num]['accepted'] += 1
            else:
                summary['acceptance_by_iteration'][iter_num]['rejected'] += 1
            
            # Collect problems and recommendations
            for prob in iter_data.get('judge_problems', []):
                if prob:
                    summary['common_problems'][prob] += 1
            
            for rec in iter_data.get('judge_recommendations', []):
                if rec:
                    summary['common_recommendations'][rec] += 1
        
        # Group by iteration count
        iter_count = r['iterations_used']
        if iter_count not in summary['by_iterations']:
            summary['by_iterations'][iter_count] = {
                'count': 0,
                'avg_llm_calls': 0.0,
                'queries': []
            }
        
        summary['by_iterations'][iter_count]['count'] += 1
        summary['by_iterations'][iter_count]['queries'].append({
            'query': r['query'][:100],
            'llm_calls': r['total_llm_calls'],
            'final_accepted': r.get('final_judge_accepted'),
            'final_verdict': r.get('final_judge_verdict')
        })
    
    # Average LLM calls
    if iterative_results:
        total_llm = sum(r['total_llm_calls'] for r in iterative_results)
        summary['avg_llm_calls_per_query'] = total_llm / len(iterative_results)
        
        total_iters = sum(r['iterations_used'] for r in iterative_results if r['iterations_used'] > 0)
        if total_iters > 0:
            summary['avg_llm_calls_per_iteration'] = total_llm / total_iters
    
    # Average LLM calls by iteration count
    for iter_count, data in summary['by_iterations'].items():
        queries = data['queries']
        if queries:
            data['avg_llm_calls'] = sum(q['llm_calls'] for q in queries) / len(queries)
    
    return summary


def save_results(results: List[Dict]):
    """Save analysis results to multiple output files."""
    
    output_folder = OUTPUT_FOLDER.resolve()
    output_folder.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Save main detailed results as CSV
    csv_path = output_folder / f"llm_calls_detailed_{timestamp}.csv"
    csv_fields = [
        'log_file', 'query', 'mode', 'iterations_used',
        'total_llm_calls', 'agent1_calls', 'agent1b_calls',
        'agent2_naive_calls', 'agent2_graphrag_calls',
        'aggregator_calls', 'judge_calls', 'query_modifier_calls',
        'final_judge_accepted', 'final_judge_verdict',
        'final_aggregator_chosen', 'embed_calls'
    ]
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(results)
    
    print(f"Detailed results saved to: {csv_path}")
    
    # 2. Save full JSON (includes per-iteration decisions)
    json_path = output_folder / f"llm_calls_detailed_{timestamp}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"JSON results saved to: {json_path}")
    
    # 3. Create and save iterative summary
    iter_summary = create_iterative_summary(results)
    iter_summary_path = output_folder / f"iterative_summary_{timestamp}.json"
    with open(iter_summary_path, 'w', encoding='utf-8') as f:
        # Convert Counter to dict for JSON serialization
        summary_copy = iter_summary.copy()
        summary_copy['iteration_stats']['distribution'] = dict(iter_summary['iteration_stats']['distribution'])
        summary_copy['judge_verdicts'] = dict(iter_summary['judge_verdicts'])
        summary_copy['aggregator_choices'] = dict(iter_summary['aggregator_choices'])
        summary_copy['common_problems'] = dict(iter_summary['common_problems'])
        summary_copy['common_recommendations'] = dict(iter_summary['common_recommendations'])
        json.dump(summary_copy, f, indent=2, ensure_ascii=False)
    
    print(f"Iterative summary saved to: {iter_summary_path}")
    
    # 4. Save iterative statistics CSV
    iter_stats_csv = output_folder / f"iterative_statistics_{timestamp}.csv"
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
    
    # 5. Save human-readable report
    readable_path = output_folder / f"iterative_readable_{timestamp}.txt"
    with open(readable_path, 'w', encoding='utf-8') as f:
        separator = "=" * 80
        title = "AGENTIC ITERATIVE RAG DECISIONS - HUMAN READABLE"
        f.write(separator + "\n")
        f.write(title + "\n")
        f.write(separator + "\n")
        
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
        f.write("BY ITERATION COUNT\n")
        f.write("-" * 80 + "\n")
        
        for iter_count in sorted(iter_summary['by_iterations'].keys()):
            data = iter_summary['by_iterations'][iter_count]
            f.write(f"{iter_count} ITERATIONS ({data['count']} queries)\n")
            f.write(f"  Avg LLM calls: {data['avg_llm_calls']:.2f}\n")
            f.write("-" * 40 + "\n")
            
            for i, q in enumerate(data['queries'][:10], 1):
                f.write(f"{i}. {q['query']}\n")
                f.write(f"   LLM: {q['llm_calls']} | Accepted: {q['final_accepted']} | Verdict: {q['final_verdict']}\n")
            
            f.write("\n")
        
        f.write(separator + "\n")
        f.write("INDIVIDUAL QUERY DETAILS\n")
        f.write("-" * 80 + "\n")
        
        for i, r in enumerate(results, 1):
            if r.get('mode') == 'iterative':
                f.write(f"[{i}] {r['log_file']}\n")
                f.write(f"Query: {r['query']}\n")
                f.write(f"Iterations: {r['iterations_used']} | LLM Calls: {r['total_llm_calls']}\n")
                f.write(f"Final: Judge={r['final_judge_verdict']} (accepted={r['final_judge_accepted']}), Agg={r['final_aggregator_chosen']}\n")
                
                if r.get('per_iteration_decisions'):
                    f.write("  Per-Iteration Decisions:\n")
                    for iter_data in r['per_iteration_decisions']:
                        f.write(f"    Iter {iter_data['iteration']}: ")
                        f.write(f"Agg={iter_data['aggregator_chosen']} ({iter_data['aggregator_confidence']:.2f}), ")
                        f.write(f"Judge={iter_data['judge_verdict']} (accepted={iter_data['judge_accepted']}, conf={iter_data['judge_confidence']:.2f})")
                        if iter_data.get('query_modified'):
                            f.write(", Query Modified")
                        f.write("\n")
                
                f.write("-" * 80 + "\n")
    
    print(f"Human-readable report saved to: {readable_path}")
    
    # 6. Create summary
    summary = create_summary(results, iter_summary)
    summary_path = output_folder / f"llm_calls_summary_{timestamp}.txt"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(summary)
    
    print(f"Summary saved to: {summary_path}")
    print(summary)


def create_summary(results: List[Dict], iter_summary: Dict) -> str:
    """Create a text summary of the analysis."""
    
    if not results:
        return "No results to summarize."
    
    total_files = len(results)
    
    # Filter by mode
    iter_results = [r for r in results if r['mode'] == 'iterative']
    other_results = [r for r in results if r['mode'] != 'iterative']
    
    # Overall stats
    total_llm_calls = sum(r['total_llm_calls'] for r in results)
    total_agent1 = sum(r['agent1_calls'] for r in results)
    total_agent1b = sum(r['agent1b_calls'] for r in results)
    total_agent2_naive = sum(r['agent2_naive_calls'] for r in results)
    total_agent2_graphrag = sum(r['agent2_graphrag_calls'] for r in results)
    total_aggregator = sum(r['aggregator_calls'] for r in results)
    total_judge = sum(r['judge_calls'] for r in results)
    total_modifier = sum(r['query_modifier_calls'] for r in results)
    total_embed_calls = sum(r['embed_calls'] for r in results)
    
    avg_llm_per_query = total_llm_calls / total_files if total_files > 0 else 0
    avg_embed_per_query = total_embed_calls / total_files if total_files > 0 else 0
    
    min_llm = min(r['total_llm_calls'] for r in results) if results else 0
    max_llm = max(r['total_llm_calls'] for r in results) if results else 0
    
    # Build summary sections using docstrings
    header = """
╔══════════════════════════════════════════════════════════════╗
║   AGENTIC ITERATIVE MULTI-PIPELINE RAG LLM CALL ANALYSIS    ║
╚══════════════════════════════════════════════════════════════╝
"""
    
    overall_section = f"""
📊 Overall Statistics:
├─ Total log files analyzed: {total_files}
│  ├─ Iterative mode: {len(iter_results)}
│  └─ Other modes: {len(other_results)}
├─ Total LLM calls: {total_llm_calls}
│  ├─ Agent 1 (entity extraction): {total_agent1}
│  ├─ Agent 1b (triple extraction): {total_agent1b}
│  ├─ Agent 2 Naive: {total_agent2_naive}
│  ├─ Agent 2 GraphRAG: {total_agent2_graphrag}
│  ├─ Aggregator: {total_aggregator}
│  ├─ Answer Judge: {total_judge}
│  └─ Query Modifier: {total_modifier}
└─ Total embedding calls: {total_embed_calls}
"""
    
    averages_section = f"""
📈 Per-Query Averages:
├─ Average LLM calls per query: {avg_llm_per_query:.2f}
├─ Average embedding calls per query: {avg_embed_per_query:.2f}
└─ Average iterations per query: {iter_summary['iteration_stats']['mean']:.2f}
"""
    
    ranges_section = f"""
📉 Range:
├─ Minimum LLM calls in a query: {min_llm}
├─ Maximum LLM calls in a query: {max_llm}
├─ Minimum iterations: {iter_summary['iteration_stats']['min']}
└─ Maximum iterations: {iter_summary['iteration_stats']['max']}
"""
    
    if iter_results and iter_summary['total_queries'] > 0:
        iter_section = f"""
🔄 ITERATION ANALYSIS ({iter_summary['total_queries']} queries):

📊 Iteration Distribution:
"""
        for iters, count in sorted(iter_summary['iteration_stats']['distribution'].items()):
            percentage = (count / iter_summary['total_queries']) * 100
            bar = "█" * int(percentage / 2)
            iter_section += f"   ├─ {iters} iterations: {count} queries ({percentage:.1f}%) {bar}\n"
        
        iter_section += f"""
📈 Iteration Statistics:
├─ Total iterations: {iter_summary['iteration_stats']['total']}
├─ Mean: {iter_summary['iteration_stats']['mean']:.2f}
├─ Median: {iter_summary['iteration_stats']['median']:.2f}
└─ Range: {iter_summary['iteration_stats']['min']}-{iter_summary['iteration_stats']['max']}

🎯 Judge Verdicts:
"""
        for verdict, count in iter_summary['judge_verdicts'].most_common():
            percentage = (count / iter_summary['total_queries']) * 100
            iter_section += f"   ├─ {verdict}: {count} ({percentage:.1f}%)\n"
        
        iter_section += """
📋 Aggregator Choices:
"""
        for choice, count in iter_summary['aggregator_choices'].most_common():
            percentage = (count / iter_summary['total_queries']) * 100
            iter_section += f"   ├─ {choice}: {count} ({percentage:.1f}%)\n"
        
        iter_section += f"""
💻 LLM Call Statistics:
├─ Average per query: {iter_summary['avg_llm_calls_per_query']:.2f}
└─ Average per iteration: {iter_summary['avg_llm_calls_per_iteration']:.2f}
"""
    else:
        iter_section = ""
    
    pipeline_doc = """
💡 Pipeline Pattern (Agentic Iterative):
   └─ For each iteration (up to AGENTIC_MAX_ITERS=4):
      1. Run both pipelines:
         a) Naive RAG:
            - Embed query
            - Vector search
            - Agent 1/1b: called but ignored
            - Agent 2: answer (1 LLM call)
         
         b) GraphRAG:
            - Agent 1: entity extraction (1 LLM call)
            - Agent 1b: triple extraction (1 LLM call)
            - Graph retrieval
            - Agent 2: answer (1 LLM call)
      
      2. Aggregator: synthesize (1 LLM call)
      
      3. Answer Judge: evaluate (1 LLM call)
         - If accepted: STOP
         - If not accepted: continue
      
      4. Query Modifier: refine query (1 LLM call)
         - Next iteration uses modified query
   
   └─ LLM calls per iteration:
      - If accepted: 1 + 3 + 1 + 1 = 6
      - If not accepted: 1 + 3 + 1 + 1 + 1 = 7
   └─ Agent 1/1b in Naive RAG not counted
"""
    
    agent_breakdown = f"""
🔍 Agent Call Breakdown (across all queries):
├─ Naive RAG pipeline (per iteration):
│  ├─ Agent 1/1b: present but ignored in count
│  └─ Agent 2: {total_agent2_naive} calls
├─ GraphRAG pipeline (per iteration):
│  ├─ Agent 1 (entity extraction): {total_agent1}
│  ├─ Agent 1b (triple extraction): {total_agent1b}
│  └─ Agent 2: {total_agent2_graphrag} calls
├─ Aggregator (per iteration): {total_aggregator}
├─ Answer Judge (per iteration): {total_judge}
├─ Query Modifier (if not accepted): {total_modifier}
└─ Total LLM calls: {total_llm_calls}
"""
    
    acceptance_section = ""
    if iter_summary.get('acceptance_by_iteration'):
        acceptance_section = """
📊 Acceptance by Iteration:
"""
        for iter_num in sorted(iter_summary['acceptance_by_iteration'].keys()):
            data = iter_summary['acceptance_by_iteration'][iter_num]
            total = data['accepted'] + data['rejected']
            accept_pct = (data['accepted'] / total * 100) if total > 0 else 0
            acceptance_section += f"   ├─ Iteration {iter_num}: {data['accepted']} accepted, {data['rejected']} rejected ({accept_pct:.1f}%)\n"
    
    problems_section = ""
    if iter_summary.get('common_problems'):
        problems_section = """
⚠️  Most Common Problems:
"""
        for prob, count in iter_summary['common_problems'].most_common(5):
            problems_section += f"   ├─ {prob}: {count}\n"
    
    comparison_doc = """
📊 COMPARISON TO OTHER METHODS:
└─ Agentic Iterative characteristics:
   ├─ Iterative refinement with judge feedback
   ├─ Query modification based on retrieval quality
   ├─ Both pipelines run in each iteration
   ├─ Stops when judge accepts or max iterations reached
   ├─ Variable cost: 6-7 LLM calls per iteration
   ├─ Total cost: (6-7) * N iterations
   └─ Best for: queries requiring refinement to retrieve correct evidence
"""
    
    notes_doc = f"""
📝 Notes:
   - Iterative pattern with judge-driven refinement
   - Each iteration: Naive (1) + GraphRAG (3) + Aggregator (1) + Judge (1) + [Modifier (1)]
   - Agent 1/1b called in Naive RAG but not counted
   - Stops early if judge accepts answer
   - Max iterations: {iter_summary['iteration_stats']['max']}
   - Average iterations: {iter_summary['iteration_stats']['mean']:.2f}
   - Query modified between iterations if not accepted
   - Embedding calls: very high due to repeated pipeline runs
   - Most expensive approach but potentially highest quality
"""
    
    summary = (
        header + overall_section + averages_section + ranges_section + 
        iter_section + pipeline_doc + agent_breakdown + acceptance_section +
        problems_section + comparison_doc + notes_doc
    )
    
    return summary


def main():
    """Main execution function."""
    
    separator = "=" * 70
    title = "AGENTIC ITERATIVE MULTI-PIPELINE RAG LLM CALL ANALYZER"
    subtitle = "(Tracking iterations, judge decisions, and query refinement)"
    
    print(separator)
    print(title)
    print(subtitle)
    print(separator)
    print(f"Input folder: {LOG_FOLDER.resolve()}")
    print(f"Output folder: {OUTPUT_FOLDER.resolve()}")
    print()
    
    # Analyze all log files
    results = analyze_all_logs()
    
    if not results:
        print("No results to save. Exiting.")
        return
    
    # Save results
    save_results(results)
    
    print(separator)
    print("✓ Analysis complete!")
    print(separator)


if __name__ == "__main__":
    main()