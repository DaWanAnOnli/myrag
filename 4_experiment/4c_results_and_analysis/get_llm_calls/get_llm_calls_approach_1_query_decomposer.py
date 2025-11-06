# analyze_llm_calls_decomposer.py
"""
Analyzes log files from the Decomposer-based Multi-Agent RAG system.
The Query Decomposer splits queries into sub-queries for GraphRAG and/or NaiveRAG.
Each sub-query runs through its respective pipeline with Answer Judge + Query Modifier loop.
An Aggregator synthesizes the final answer from all sub-answers.
Tracks LLM calls per query including iterations, judge decisions, decomposer strategy, and aggregator choice.
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
LOG_FOLDER = SCRIPT_DIR / "../../4b_retrieval/4b_iv_multi_agents/question_terminal_logs_multi_agent/approach_1_query_decomposer_5_hops_1250"
OUTPUT_FOLDER = SCRIPT_DIR / "llm_calls_analysis_results_approach_1_query_decomposer_5_hops_1250"


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


def extract_decomposer_info(content: str) -> Dict[str, any]:
    """Extract decomposer strategy and task counts."""
    info = {
        'strategy': None,
        'total_tasks': 0,
        'graphrag_tasks': 0,
        'naiverag_tasks': 0
    }
    
    # Pattern: "[Decomposer] Planned X task(s) | strategy: ..."
    match = re.search(r'\[Decomposer\] Planned (\d+) task\(s\) \| strategy: (.+)$', content, re.MULTILINE)
    if match:
        info['total_tasks'] = int(match.group(1))
        info['strategy'] = match.group(2).strip()
    
    # Count individual task assignments
    # Pattern: "[Decomposer] Task X: pipeline=graphrag ..." or "pipeline=naiverag ..."
    g_tasks = len(re.findall(r'\[Decomposer\] Task \d+: pipeline=graphrag', content))
    n_tasks = len(re.findall(r'\[Decomposer\] Task \d+: pipeline=naiverag', content))
    
    info['graphrag_tasks'] = g_tasks
    info['naiverag_tasks'] = n_tasks
    
    return info


def extract_aggregator_decision(content: str) -> Optional[str]:
    """Extract the aggregator's final decision."""
    # Pattern: "[Aggregator] Decision=..." 
    match = re.search(r'\[Aggregator\] Decision=(\w+)', content)
    if match:
        return match.group(1)
    
    # Alternative: look in summary section
    match = re.search(r'- Aggregator decision:\s*(\w+)', content)
    if match:
        return match.group(1)
    
    return None


def extract_graphrag_iterations(content: str) -> List[Dict[str, any]]:
    """
    Extract GraphRAG iteration data across all tasks.
    Pattern: --- GraphRAG Iteration X/Y START --- ... DONE
    """
    iterations = []
    
    # Pattern for GraphRAG iterations
    start_pattern = r'--- GraphRAG Iteration (\d+)/(\d+) START ---'
    done_pattern = r'--- GraphRAG Iteration (\d+)/(\d+) DONE'
    
    start_matches = list(re.finditer(start_pattern, content))
    
    for start_match in start_matches:
        iter_num = int(start_match.group(1))
        max_iter = int(start_match.group(2))
        
        # Find corresponding DONE marker or next iteration
        done_search = re.search(
            rf'--- GraphRAG Iteration {iter_num}/{max_iter} DONE',
            content[start_match.end():]
        )
        
        if done_search:
            block_end_pos = start_match.end() + done_search.end()
        else:
            # Look for next iteration or end
            next_iter = re.search(start_pattern, content[start_match.end():])
            if next_iter:
                block_end_pos = start_match.end() + next_iter.start()
            else:
                block_end_pos = len(content)
        
        block = content[start_match.end():block_end_pos]
        
        iter_data = {
            'iteration': iter_num,
            'max_iterations': max_iter,
            'judge_decision': None,
            'query_modified': False
        }
        
        # Check for judge decisions in this block
        if re.search(r'\[GraphRAG\] Judge verdict acceptable=True', block):
            iter_data['judge_decision'] = 'acceptable'
        elif re.search(r'\[GraphRAG\] Stop condition reached.*acceptable=True', block):
            iter_data['judge_decision'] = 'acceptable'
        elif re.search(r'\[GraphRAG\] Stop condition reached.*i=\d+\)', block):
            # Reached max iterations
            if iter_num >= max_iter:
                iter_data['judge_decision'] = 'rejected_at_cap'
            else:
                iter_data['judge_decision'] = 'insufficient'
        elif re.search(r'\[GraphRAG\] Modified query:', block):
            iter_data['judge_decision'] = 'insufficient'
            iter_data['query_modified'] = True
        
        iterations.append(iter_data)
    
    # Remove duplicates and sort
    seen = set()
    unique_iters = []
    for it in iterations:
        if it['iteration'] not in seen:
            seen.add(it['iteration'])
            unique_iters.append(it)
    
    unique_iters.sort(key=lambda x: x['iteration'])
    return unique_iters


def extract_naiverag_iterations(content: str) -> List[Dict[str, any]]:
    """
    Extract NaiveRAG iteration data across all tasks.
    Pattern: --- NaiveRAG Iteration X/Y START ---
    """
    iterations = []
    
    # Pattern for NaiveRAG iterations
    iter_pattern = r'--- NaiveRAG Iteration (\d+)/(\d+) START ---'
    done_pattern = r'--- NaiveRAG Iteration (\d+)/(\d+) DONE'
    
    iter_matches = list(re.finditer(iter_pattern, content))
    
    for i, match in enumerate(iter_matches):
        iter_num = int(match.group(1))
        max_iter = int(match.group(2))
        
        # Find end of this iteration block
        done_search = re.search(
            rf'--- NaiveRAG Iteration {iter_num}/{max_iter} DONE',
            content[match.end():]
        )
        
        if done_search:
            block_end_pos = match.end() + done_search.end()
        else:
            # Look for next iteration or end
            if i + 1 < len(iter_matches):
                block_end_pos = iter_matches[i + 1].start()
            else:
                # Find end marker
                end_search = re.search(r'(===.*Multi-Agent RAG.*summary|Log file:|\[Aggregator\])', content[match.end():])
                block_end_pos = match.end() + end_search.start() if end_search else len(content)
        
        block = content[match.end():block_end_pos]
        
        iter_data = {
            'iteration': iter_num,
            'max_iterations': max_iter,
            'judge_decision': None,
            'query_modified': False
        }
        
        # Check for judge decisions
        if re.search(r'\[NaiveRAG\] Judge decision: ACCEPTABLE', block):
            iter_data['judge_decision'] = 'acceptable'
        elif re.search(r'\[NaiveRAG\] ACCEPTED at iteration', block):
            iter_data['judge_decision'] = 'acceptable'
        elif re.search(r'\[NaiveRAG\] Iteration limit reached', block):
            iter_data['judge_decision'] = 'stopped_at_limit'
        elif re.search(r'\[NaiveRAG\] Judge decision: INSUFFICIENT', block):
            iter_data['judge_decision'] = 'insufficient'
            if re.search(r'\[NaiveRAG\] Modified query', block):
                iter_data['query_modified'] = True
        
        iterations.append(iter_data)
    
    # Remove duplicates and sort
    seen = set()
    unique_iters = []
    for it in iterations:
        if it['iteration'] not in seen:
            seen.add(it['iteration'])
            unique_iters.append(it)
    
    unique_iters.sort(key=lambda x: x['iteration'])
    return unique_iters


def count_graphrag_agent_calls(content: str) -> Dict[str, int]:
    """Count LLM calls for GraphRAG pipeline (all tasks combined)."""
    counts = {
        'agent1': 0,
        'agent1b': 0,
        'agent2': 0,
        'judge': 0,
        'modifier': 0
    }
    
    # Find all GraphRAG sections (may have multiple tasks)
    # Look for sections between "[Run] GraphRAG Task" and next task or aggregator
    g_sections = []
    for match in re.finditer(r'\[Run\] GraphRAG Task \d+', content):
        start = match.start()
        # Find end (next task or aggregator or summary)
        end_search = re.search(r'(\[Run\] (GraphRAG|NaiveRAG) Task|\[Aggregator\]|===.*summary)', content[start + 100:])
        if end_search:
            end = start + 100 + end_search.start()
        else:
            end = len(content)
        g_sections.append(content[start:end])
    
    # If no task markers, fall back to looking for GraphRAG iterations
    if not g_sections:
        g_start = re.search(r'--- GraphRAG Iteration 1/', content)
        if g_start:
            g_end_search = re.search(r'(--- NaiveRAG Iteration 1/|\[Aggregator\]|===.*summary)', content[g_start.start():])
            if g_end_search:
                g_sections = [content[g_start.start():g_start.start() + g_end_search.start()]]
            else:
                g_sections = [content[g_start.start():]]
    
    # Count across all GraphRAG sections
    for section in g_sections:
        counts['agent1'] += len(re.findall(r'\[Agent 1\] Prompt:', section))
        counts['agent1b'] += len(re.findall(r'\[Agent 1b\] Prompt:', section))
        counts['agent2'] += len(re.findall(r'\[Agent 2\] Prompt:', section))
        # Judge and modifier are logged as results
        counts['judge'] += len(re.findall(r'\[GraphRAG\] Judge verdict acceptable=', section))
        counts['modifier'] += len(re.findall(r'\[GraphRAG\] Modified query:', section))
    
    return counts


def count_naiverag_agent_calls(content: str) -> Dict[str, int]:
    """Count LLM calls for NaiveRAG pipeline (excluding Agent 1/1b, all tasks combined)."""
    counts = {
        'agent2': 0,
        'judge': 0,
        'modifier': 0
    }
    
    # Find all NaiveRAG sections (may have multiple tasks)
    n_sections = []
    for match in re.finditer(r'\[Run\] NaiveRAG Task \d+', content):
        start = match.start()
        # Find end (next task or aggregator or summary)
        end_search = re.search(r'(\[Run\] (GraphRAG|NaiveRAG) Task|\[Aggregator\]|===.*summary)', content[start + 100:])
        if end_search:
            end = start + 100 + end_search.start()
        else:
            end = len(content)
        n_sections.append(content[start:end])
    
    # If no task markers, fall back to looking for NaiveRAG iterations
    if not n_sections:
        n_start = re.search(r'--- NaiveRAG Iteration 1/', content)
        if n_start:
            n_end_search = re.search(r'(\[Aggregator\]|===.*summary)', content[n_start.start():])
            if n_end_search:
                n_sections = [content[n_start.start():n_start.start() + n_end_search.start()]]
            else:
                n_sections = [content[n_start.start():]]
    
    # Count across all NaiveRAG sections
    # NOTE: Agent 1/1b are NOT counted even if present
    for section in n_sections:
        counts['agent2'] += len(re.findall(r'\[Agent 2\] Prompt:', section))
        # Judge and modifier are logged as results
        counts['judge'] += len(re.findall(r'\[NaiveRAG\] Judge decision:', section))
        counts['modifier'] += len(re.findall(r'\[NaiveRAG\] Modified query', section))
    
    return counts


def analyze_log_file(log_path: Path) -> Dict:
    """
    Analyze a single log file for the decomposer-based system.
    
    Returns dict with:
    - Decomposer info (strategy, task counts)
    - Per-pipeline LLM call counts (across all tasks)
    - Per-pipeline iteration data (across all tasks)
    - Aggregator decision
    - Total LLM calls
    """
    
    result = {
        'log_file': log_path.name,
        'query': '',
        
        # Decomposer
        'decomposer_strategy': None,
        'decomposer_total_tasks': 0,
        'decomposer_graphrag_tasks': 0,
        'decomposer_naiverag_tasks': 0,
        'decomposer_calls': 1,  # Always exactly 1
        
        # GraphRAG pipeline (across all tasks)
        'graphrag_iterations_used': 0,
        'graphrag_agent1_calls': 0,
        'graphrag_agent1b_calls': 0,
        'graphrag_agent2_calls': 0,
        'graphrag_judge_calls': 0,
        'graphrag_modifier_calls': 0,
        'graphrag_per_iteration': [],
        
        # NaiveRAG pipeline (across all tasks)
        'naiverag_iterations_used': 0,
        'naiverag_agent2_calls': 0,
        'naiverag_judge_calls': 0,
        'naiverag_modifier_calls': 0,
        'naiverag_per_iteration': [],
        
        # Aggregator
        'aggregator_calls': 1,  # Always exactly 1
        'aggregator_decision': None,
        
        # Totals
        'total_llm_calls': 0,
        'embed_calls': 0
    }
    
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract query
        result['query'] = extract_query_from_log(log_path)
        
        # Extract decomposer info
        dec_info = extract_decomposer_info(content)
        result['decomposer_strategy'] = dec_info['strategy']
        result['decomposer_total_tasks'] = dec_info['total_tasks']
        result['decomposer_graphrag_tasks'] = dec_info['graphrag_tasks']
        result['decomposer_naiverag_tasks'] = dec_info['naiverag_tasks']
        
        # Extract GraphRAG data (across all tasks)
        if dec_info['graphrag_tasks'] > 0:
            g_iterations = extract_graphrag_iterations(content)
            result['graphrag_iterations_used'] = len(g_iterations)
            result['graphrag_per_iteration'] = g_iterations
            
            g_counts = count_graphrag_agent_calls(content)
            result['graphrag_agent1_calls'] = g_counts['agent1']
            result['graphrag_agent1b_calls'] = g_counts['agent1b']
            result['graphrag_agent2_calls'] = g_counts['agent2']
            result['graphrag_judge_calls'] = g_counts['judge']
            result['graphrag_modifier_calls'] = g_counts['modifier']
        
        # Extract NaiveRAG data (across all tasks)
        if dec_info['naiverag_tasks'] > 0:
            n_iterations = extract_naiverag_iterations(content)
            result['naiverag_iterations_used'] = len(n_iterations)
            result['naiverag_per_iteration'] = n_iterations
            
            n_counts = count_naiverag_agent_calls(content)
            result['naiverag_agent2_calls'] = n_counts['agent2']
            result['naiverag_judge_calls'] = n_counts['judge']
            result['naiverag_modifier_calls'] = n_counts['modifier']
        
        # Extract aggregator decision
        result['aggregator_decision'] = extract_aggregator_decision(content)
        
        # Total LLM calls = Decomposer + all pipeline agents + Aggregator
        result['total_llm_calls'] = (
            result['decomposer_calls'] +
            result['graphrag_agent1_calls'] +
            result['graphrag_agent1b_calls'] +
            result['graphrag_agent2_calls'] +
            result['graphrag_judge_calls'] +
            result['graphrag_modifier_calls'] +
            result['naiverag_agent2_calls'] +
            result['naiverag_judge_calls'] +
            result['naiverag_modifier_calls'] +
            result['aggregator_calls']
        )
        
        # Count embedding calls
        result['embed_calls'] = len(re.findall(r'\[Embed\]', content))
        
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


def create_summary_statistics(results: List[Dict]) -> Dict[str, any]:
    """Create comprehensive summary statistics."""
    
    summary = {
        'total_queries': len(results),
        
        # Decomposer statistics
        'decomposer': {
            'avg_total_tasks': 0.0,
            'avg_graphrag_tasks': 0.0,
            'avg_naiverag_tasks': 0.0,
            'strategy_distribution': Counter(),
            'total_calls': len(results)  # Always 1 per query
        },
        
        # GraphRAG statistics (when used)
        'graphrag': {
            'queries_with_tasks': 0,
            'iteration_stats': {
                'mean': 0.0,
                'median': 0.0,
                'min': 0,
                'max': 0,
                'distribution': Counter()
            },
            'judge_decisions': Counter(),
            'avg_llm_calls_per_query': 0.0,
            'total_llm_calls': 0
        },
        
        # NaiveRAG statistics (when used)
        'naiverag': {
            'queries_with_tasks': 0,
            'iteration_stats': {
                'mean': 0.0,
                'median': 0.0,
                'min': 0,
                'max': 0,
                'distribution': Counter()
            },
            'judge_decisions': Counter(),
            'avg_llm_calls_per_query': 0.0,
            'total_llm_calls': 0
        },
        
        # Aggregator statistics
        'aggregator': {
            'decisions': Counter(),
            'total_calls': len(results)  # Always 1 per query
        },
        
        # Overall statistics
        'avg_total_llm_calls_per_query': 0.0,
        'avg_embed_calls_per_query': 0.0
    }
    
    if not results:
        return summary
    
    import statistics
    
    # Decomposer statistics
    total_tasks = [r['decomposer_total_tasks'] for r in results]
    g_tasks = [r['decomposer_graphrag_tasks'] for r in results]
    n_tasks = [r['decomposer_naiverag_tasks'] for r in results]
    
    if results:
        summary['decomposer']['avg_total_tasks'] = statistics.mean(total_tasks)
        summary['decomposer']['avg_graphrag_tasks'] = statistics.mean(g_tasks)
        summary['decomposer']['avg_naiverag_tasks'] = statistics.mean(n_tasks)
    
    for r in results:
        strategy = r.get('decomposer_strategy')
        if strategy:
            summary['decomposer']['strategy_distribution'][strategy] += 1
    
    # GraphRAG statistics (only queries that used GraphRAG)
    graphrag_results = [r for r in results if r['decomposer_graphrag_tasks'] > 0]
    summary['graphrag']['queries_with_tasks'] = len(graphrag_results)
    
    if graphrag_results:
        g_iters = [r['graphrag_iterations_used'] for r in graphrag_results if r['graphrag_iterations_used'] > 0]
        if g_iters:
            summary['graphrag']['iteration_stats']['mean'] = statistics.mean(g_iters)
            summary['graphrag']['iteration_stats']['median'] = statistics.median(g_iters)
            summary['graphrag']['iteration_stats']['min'] = min(g_iters)
            summary['graphrag']['iteration_stats']['max'] = max(g_iters)
            summary['graphrag']['iteration_stats']['distribution'] = Counter(g_iters)
        
        for r in graphrag_results:
            # Collect judge decisions
            for iter_data in r.get('graphrag_per_iteration', []):
                decision = iter_data.get('judge_decision')
                if decision:
                    summary['graphrag']['judge_decisions'][decision] += 1
            
            # GraphRAG LLM calls
            g_llm = (
                r['graphrag_agent1_calls'] + r['graphrag_agent1b_calls'] +
                r['graphrag_agent2_calls'] + r['graphrag_judge_calls'] +
                r['graphrag_modifier_calls']
            )
            summary['graphrag']['total_llm_calls'] += g_llm
        
        summary['graphrag']['avg_llm_calls_per_query'] = (
            summary['graphrag']['total_llm_calls'] / len(graphrag_results)
        )
    
    # NaiveRAG statistics (only queries that used NaiveRAG)
    naiverag_results = [r for r in results if r['decomposer_naiverag_tasks'] > 0]
    summary['naiverag']['queries_with_tasks'] = len(naiverag_results)
    
    if naiverag_results:
        n_iters = [r['naiverag_iterations_used'] for r in naiverag_results if r['naiverag_iterations_used'] > 0]
        if n_iters:
            summary['naiverag']['iteration_stats']['mean'] = statistics.mean(n_iters)
            summary['naiverag']['iteration_stats']['median'] = statistics.median(n_iters)
            summary['naiverag']['iteration_stats']['min'] = min(n_iters)
            summary['naiverag']['iteration_stats']['max'] = max(n_iters)
            summary['naiverag']['iteration_stats']['distribution'] = Counter(n_iters)
        
        for r in naiverag_results:
            # Collect judge decisions
            for iter_data in r.get('naiverag_per_iteration', []):
                decision = iter_data.get('judge_decision')
                if decision:
                    summary['naiverag']['judge_decisions'][decision] += 1
            
            # NaiveRAG LLM calls
            n_llm = (
                r['naiverag_agent2_calls'] + r['naiverag_judge_calls'] +
                r['naiverag_modifier_calls']
            )
            summary['naiverag']['total_llm_calls'] += n_llm
        
        summary['naiverag']['avg_llm_calls_per_query'] = (
            summary['naiverag']['total_llm_calls'] / len(naiverag_results)
        )
    
    # Aggregator decisions
    for r in results:
        decision = r.get('aggregator_decision')
        if decision:
            summary['aggregator']['decisions'][decision] += 1
    
    # Overall statistics
    total_llm = sum(r['total_llm_calls'] for r in results)
    total_embed = sum(r['embed_calls'] for r in results)
    
    if results:
        summary['avg_total_llm_calls_per_query'] = total_llm / len(results)
        summary['avg_embed_calls_per_query'] = total_embed / len(results)
    
    return summary


def save_results(results: List[Dict]):
    """Save analysis results to multiple output files."""
    
    output_folder = OUTPUT_FOLDER.resolve()
    output_folder.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. CSV
    csv_path = output_folder / f"decomposer_detailed_{timestamp}.csv"
    csv_fields = [
        'log_file', 'query',
        'decomposer_strategy', 'decomposer_total_tasks', 'decomposer_graphrag_tasks', 'decomposer_naiverag_tasks', 'decomposer_calls',
        'graphrag_iterations_used', 'graphrag_agent1_calls', 'graphrag_agent1b_calls',
        'graphrag_agent2_calls', 'graphrag_judge_calls', 'graphrag_modifier_calls',
        'naiverag_iterations_used', 'naiverag_agent2_calls',
        'naiverag_judge_calls', 'naiverag_modifier_calls',
        'aggregator_calls', 'aggregator_decision',
        'total_llm_calls', 'embed_calls'
    ]
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(results)
    
    print(f"Detailed results saved to: {csv_path}")
    
    # 2. JSON
    json_path = output_folder / f"decomposer_detailed_{timestamp}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"JSON results saved to: {json_path}")
    
    # 3. Summary JSON
    summary = create_summary_statistics(results)
    summary_path = output_folder / f"decomposer_summary_{timestamp}.json"
    summary_copy = json.loads(json.dumps(summary, default=dict))
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary_copy, f, indent=2, ensure_ascii=False)
    
    print(f"Summary statistics saved to: {summary_path}")
    
    # 4. Human-readable report
    readable_path = output_folder / f"decomposer_readable_{timestamp}.txt"
    with open(readable_path, 'w', encoding='utf-8') as f:
        separator = "=" * 80
        
        f.write(separator + "\n")
        f.write("DECOMPOSER-BASED MULTI-AGENT RAG ANALYSIS - HUMAN READABLE REPORT\n")
        f.write(separator + "\n")
        
        f.write("SUMMARY STATISTICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total queries analyzed: {summary['total_queries']}\n")
        
        # Decomposer section
        f.write("QUERY DECOMPOSER\n")
        f.write("-" * 40 + "\n")
        d = summary['decomposer']
        f.write(f"Total calls: {d['total_calls']} (1 per query)\n")
        f.write(f"Average total tasks per query: {d['avg_total_tasks']:.2f}\n")
        f.write(f"Average GraphRAG tasks per query: {d['avg_graphrag_tasks']:.2f}\n")
        f.write(f"Average NaiveRAG tasks per query: {d['avg_naiverag_tasks']:.2f}\n")
        f.write("\nStrategy Distribution (top 10):\n")
        for strategy, count in d['strategy_distribution'].most_common(10):
            pct = (count / summary['total_queries'] * 100) if summary['total_queries'] > 0 else 0
            # Truncate long strategies
            strat_display = strategy[:60] + "..." if len(strategy) > 60 else strategy
            f.write(f"  {strat_display}: {count} ({pct:.1f}%)\n")
        f.write("\n")
        
        # GraphRAG section
        f.write("GRAPHRAG PIPELINE (when used)\n")
        f.write("-" * 40 + "\n")
        g = summary['graphrag']
        f.write(f"Queries with GraphRAG tasks: {g['queries_with_tasks']}\n")
        if g['queries_with_tasks'] > 0:
            f.write("Iteration Statistics:\n")
            f.write(f"  Mean: {g['iteration_stats']['mean']:.2f}\n")
            f.write(f"  Median: {g['iteration_stats']['median']:.2f}\n")
            f.write(f"  Range: {g['iteration_stats']['min']}-{g['iteration_stats']['max']}\n")
            f.write("  Distribution:\n")
            for iters, count in sorted(g['iteration_stats']['distribution'].items()):
                pct = (count / g['queries_with_tasks'] * 100) if g['queries_with_tasks'] > 0 else 0
                f.write(f"    {iters} iterations: {count} queries ({pct:.1f}%)\n")
            
            f.write("\nJudge Decisions:\n")
            for decision, count in g['judge_decisions'].items():
                f.write(f"  {decision}: {count}\n")
            
            f.write("\nLLM Calls:\n")
            f.write(f"  Total: {g['total_llm_calls']}\n")
            f.write(f"  Average per query (when used): {g['avg_llm_calls_per_query']:.2f}\n")
        else:
            f.write("  (No queries used GraphRAG)\n")
        f.write("\n")
        
        # NaiveRAG section
        f.write("NAIVERAG PIPELINE (when used)\n")
        f.write("-" * 40 + "\n")
        n = summary['naiverag']
        f.write(f"Queries with NaiveRAG tasks: {n['queries_with_tasks']}\n")
        if n['queries_with_tasks'] > 0:
            f.write("Iteration Statistics:\n")
            f.write(f"  Mean: {n['iteration_stats']['mean']:.2f}\n")
            f.write(f"  Median: {n['iteration_stats']['median']:.2f}\n")
            f.write(f"  Range: {n['iteration_stats']['min']}-{n['iteration_stats']['max']}\n")
            f.write("  Distribution:\n")
            for iters, count in sorted(n['iteration_stats']['distribution'].items()):
                pct = (count / n['queries_with_tasks'] * 100) if n['queries_with_tasks'] > 0 else 0
                f.write(f"    {iters} iterations: {count} queries ({pct:.1f}%)\n")
            
            f.write("\nJudge Decisions:\n")
            for decision, count in n['judge_decisions'].items():
                f.write(f"  {decision}: {count}\n")
            
            f.write("\nLLM Calls:\n")
            f.write(f"  Total: {n['total_llm_calls']}\n")
            f.write(f"  Average per query (when used): {n['avg_llm_calls_per_query']:.2f}\n")
        else:
            f.write("  (No queries used NaiveRAG)\n")
        f.write("\n")
        
        # Aggregator section
        f.write("AGGREGATOR\n")
        f.write("-" * 40 + "\n")
        a = summary['aggregator']
        f.write(f"Total calls: {a['total_calls']} (1 per query)\n")
        f.write("Decisions:\n")
        for decision, count in a['decisions'].items():
            pct = (count / summary['total_queries'] * 100) if summary['total_queries'] > 0 else 0
            f.write(f"  {decision}: {count} ({pct:.1f}%)\n")
        
        f.write("\nOVERALL STATISTICS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Average total LLM calls per query: {summary['avg_total_llm_calls_per_query']:.2f}\n")
        f.write(f"Average embedding calls per query: {summary['avg_embed_calls_per_query']:.2f}\n")
        
        f.write("\n" + separator + "\n")
        f.write("INDIVIDUAL QUERY DETAILS\n")
        f.write("-" * 80 + "\n")
        
        for i, r in enumerate(results, 1):
            f.write(f"[{i}] {r['log_file']}\n")
            f.write(f"Query: {r['query']}\n")
            f.write(f"Decomposer: {r['decomposer_total_tasks']} tasks (G:{r['decomposer_graphrag_tasks']} N:{r['decomposer_naiverag_tasks']})\n")
            f.write(f"Strategy: {r['decomposer_strategy']}\n")
            
            if r['decomposer_graphrag_tasks'] > 0:
                f.write(f"GraphRAG: {r['graphrag_iterations_used']} iters | "
                       f"LLM calls: A1={r['graphrag_agent1_calls']} A1b={r['graphrag_agent1b_calls']} "
                       f"A2={r['graphrag_agent2_calls']} AJ={r['graphrag_judge_calls']} "
                       f"QM={r['graphrag_modifier_calls']}\n")
                
                if r.get('graphrag_per_iteration'):
                    f.write("  GraphRAG iterations:\n")
                    for iter_data in r['graphrag_per_iteration']:
                        f.write(f"    Iter {iter_data['iteration']}: "
                               f"judge={iter_data['judge_decision']} "
                               f"modified={iter_data['query_modified']}\n")
            
            if r['decomposer_naiverag_tasks'] > 0:
                f.write(f"NaiveRAG: {r['naiverag_iterations_used']} iters | "
                       f"LLM calls: A2={r['naiverag_agent2_calls']} "
                       f"AJ={r['naiverag_judge_calls']} QM={r['naiverag_modifier_calls']}\n")
                
                if r.get('naiverag_per_iteration'):
                    f.write("  NaiveRAG iterations:\n")
                    for iter_data in r['naiverag_per_iteration']:
                        f.write(f"    Iter {iter_data['iteration']}: "
                               f"judge={iter_data['judge_decision']} "
                               f"modified={iter_data['query_modified']}\n")
            
            f.write(f"Aggregator: {r['aggregator_decision']}\n")
            f.write(f"Total LLM: {r['total_llm_calls']} | Embeddings: {r['embed_calls']}\n")
            f.write("-" * 80 + "\n")
    
    print(f"Human-readable report saved to: {readable_path}")
    
    # 5. Text summary
    summary_text = create_text_summary(results, summary)
    summary_txt_path = output_folder / f"decomposer_summary_{timestamp}.txt"
    with open(summary_txt_path, 'w', encoding='utf-8') as f:
        f.write(summary_text)
    
    print(f"Summary saved to: {summary_txt_path}")
    print(summary_text)


def create_text_summary(results: List[Dict], summary: Dict) -> str:
    """Create formatted text summary matching the requested format."""
    
    if not results:
        return "No results to summarize."
    
    total_queries = len(results)
    
    # Calculate totals
    total_decomposer = sum(r['decomposer_calls'] for r in results)
    
    total_g_agent1 = sum(r['graphrag_agent1_calls'] for r in results)
    total_g_agent1b = sum(r['graphrag_agent1b_calls'] for r in results)
    total_g_agent2 = sum(r['graphrag_agent2_calls'] for r in results)
    total_g_judge = sum(r['graphrag_judge_calls'] for r in results)
    total_g_modifier = sum(r['graphrag_modifier_calls'] for r in results)
    
    total_n_agent2 = sum(r['naiverag_agent2_calls'] for r in results)
    total_n_judge = sum(r['naiverag_judge_calls'] for r in results)
    total_n_modifier = sum(r['naiverag_modifier_calls'] for r in results)
    
    total_aggregator = sum(r['aggregator_calls'] for r in results)
    total_llm = sum(r['total_llm_calls'] for r in results)
    total_embed = sum(r['embed_calls'] for r in results)
    
    min_llm = min(r['total_llm_calls'] for r in results) if results else 0
    max_llm = max(r['total_llm_calls'] for r in results) if results else 0
    
    header = """
╔══════════════════════════════════════════════════════════════╗
║    DECOMPOSER-BASED MULTI-AGENT RAG LLM CALL ANALYSIS      ║
║    (Decomposer → GraphRAG/NaiveRAG Tasks → Aggregator)     ║
╚══════════════════════════════════════════════════════════════╝
"""
    
    overall = f"""
📊 Overall Statistics:
├─ Total log files analyzed: {total_queries}
├─ Total LLM calls: {total_llm}
│  ├─ Query Decomposer: {total_decomposer}
│  ├─ GraphRAG pipeline (across all tasks):
│  │  ├─ Agent 1 (entity extraction): {total_g_agent1}
│  │  ├─ Agent 1b (triple extraction): {total_g_agent1b}
│  │  ├─ Agent 2 (answerer): {total_g_agent2}
│  │  ├─ Answer Judge: {total_g_judge}
│  │  └─ Query Modifier: {total_g_modifier}
│  ├─ NaiveRAG pipeline (across all tasks):
│  │  ├─ Agent 2 (answerer): {total_n_agent2}
│  │  ├─ Answer Judge: {total_n_judge}
│  │  └─ Query Modifier: {total_n_modifier}
│  └─ Aggregator (final synthesis): {total_aggregator}
└─ Total embedding calls: {total_embed}
"""
    
    avg_llm_per_query = total_llm / total_queries if total_queries > 0 else 0
    avg_embed_per_query = total_embed / total_queries if total_queries > 0 else 0
    avg_total_tasks = summary['decomposer']['avg_total_tasks']
    avg_g_tasks = summary['decomposer']['avg_graphrag_tasks']
    avg_n_tasks = summary['decomposer']['avg_naiverag_tasks']
    avg_g_iters = summary['graphrag']['iteration_stats']['mean']
    avg_n_iters = summary['naiverag']['iteration_stats']['mean']
    
    averages = f"""
📈 Per-Query Averages:
├─ Average LLM calls per query: {avg_llm_per_query:.2f}
├─ Average embedding calls per query: {avg_embed_per_query:.2f}
├─ Average total tasks per query: {avg_total_tasks:.2f}
│  ├─ Average GraphRAG tasks: {avg_g_tasks:.2f}
│  └─ Average NaiveRAG tasks: {avg_n_tasks:.2f}
├─ Average GraphRAG iterations per query (when used): {avg_g_iters:.2f}
└─ Average NaiveRAG iterations per query (when used): {avg_n_iters:.2f}
"""
    
    min_g_iter = summary['graphrag']['iteration_stats']['min'] if summary['graphrag']['queries_with_tasks'] > 0 else 0
    max_g_iter = summary['graphrag']['iteration_stats']['max'] if summary['graphrag']['queries_with_tasks'] > 0 else 0
    min_n_iter = summary['naiverag']['iteration_stats']['min'] if summary['naiverag']['queries_with_tasks'] > 0 else 0
    max_n_iter = summary['naiverag']['iteration_stats']['max'] if summary['naiverag']['queries_with_tasks'] > 0 else 0
    
    ranges = f"""
📉 Range:
├─ Minimum LLM calls in a query: {min_llm}
├─ Maximum LLM calls in a query: {max_llm}
├─ GraphRAG iterations range (when used): {min_g_iter}-{max_g_iter}
└─ NaiveRAG iterations range (when used): {min_n_iter}-{max_n_iter}
"""
    
    # Decomposer analysis
    decomposer_section = f"""
🎯 QUERY DECOMPOSER ANALYSIS ({total_queries} queries):

📋 Task Distribution:
├─ Queries with GraphRAG tasks: {summary['graphrag']['queries_with_tasks']} ({summary['graphrag']['queries_with_tasks']/total_queries*100:.1f}%)
├─ Queries with NaiveRAG tasks: {summary['naiverag']['queries_with_tasks']} ({summary['naiverag']['queries_with_tasks']/total_queries*100:.1f}%)
└─ Average tasks per query: {avg_total_tasks:.2f}

📊 Top Decomposition Strategies:
"""
    for strategy, count in summary['decomposer']['strategy_distribution'].most_common(5):
        percentage = (count / total_queries) * 100
        bar = "█" * int(percentage / 2)
        # Truncate long strategies for display
        strat_display = strategy[:50] + "..." if len(strategy) > 50 else strategy
        decomposer_section += f"   ├─ {strat_display}: {count} ({percentage:.1f}%) {bar}\n"
    
    decomposer_section += f"""
💻 LLM Call Statistics:
└─ Total calls: {total_decomposer} (1 per query)
"""
    
    # GraphRAG analysis (only if queries used it)
    if summary['graphrag']['queries_with_tasks'] > 0:
        g_iter_section = f"""
🔵 GRAPHRAG PIPELINE ANALYSIS ({summary['graphrag']['queries_with_tasks']} queries with GraphRAG tasks):

📊 Iteration Distribution:
"""
        g_dist = summary['graphrag']['iteration_stats']['distribution']
        g_queries = summary['graphrag']['queries_with_tasks']
        for iters in sorted(g_dist.keys()):
            count = g_dist[iters]
            percentage = (count / g_queries) * 100
            bar = "█" * int(percentage / 2)
            g_iter_section += f"   ├─ {iters} iterations: {count} queries ({percentage:.1f}%) {bar}\n"
        
        total_g_iters = int(summary['graphrag']['iteration_stats']['mean'] * g_queries) if summary['graphrag']['iteration_stats']['mean'] > 0 else 1
        avg_g_llm_per_iter = summary['graphrag']['total_llm_calls'] / total_g_iters if total_g_iters > 0 else 0
        
        g_iter_section += f"""
📈 Iteration Statistics:
├─ Total iterations (across all tasks): {total_g_iters}
├─ Mean: {summary['graphrag']['iteration_stats']['mean']:.2f}
├─ Median: {summary['graphrag']['iteration_stats']['median']:.2f}
└─ Range: {summary['graphrag']['iteration_stats']['min']}-{summary['graphrag']['iteration_stats']['max']}

🎯 Judge Decisions:
"""
        for decision, count in summary['graphrag']['judge_decisions'].most_common():
            percentage = (count / sum(summary['graphrag']['judge_decisions'].values())) * 100 if summary['graphrag']['judge_decisions'] else 0
            g_iter_section += f"   ├─ {decision}: {count} ({percentage:.1f}%)\n"
        
        g_iter_section += f"""
💻 LLM Call Statistics:
├─ Total LLM calls: {summary['graphrag']['total_llm_calls']}
├─ Average per query (when used): {summary['graphrag']['avg_llm_calls_per_query']:.2f}
└─ Average per iteration: {avg_g_llm_per_iter:.2f}
"""
    else:
        g_iter_section = """
🔵 GRAPHRAG PIPELINE ANALYSIS:
   └─ No queries used GraphRAG tasks
"""
    
    # NaiveRAG analysis (only if queries used it)
    if summary['naiverag']['queries_with_tasks'] > 0:
        n_iter_section = f"""
🟢 NAIVERAG PIPELINE ANALYSIS ({summary['naiverag']['queries_with_tasks']} queries with NaiveRAG tasks):

📊 Iteration Distribution:
"""
        n_dist = summary['naiverag']['iteration_stats']['distribution']
        n_queries = summary['naiverag']['queries_with_tasks']
        for iters in sorted(n_dist.keys()):
            count = n_dist[iters]
            percentage = (count / n_queries) * 100
            bar = "█" * int(percentage / 2)
            n_iter_section += f"   ├─ {iters} iterations: {count} queries ({percentage:.1f}%) {bar}\n"
        
        total_n_iters = int(summary['naiverag']['iteration_stats']['mean'] * n_queries) if summary['naiverag']['iteration_stats']['mean'] > 0 else 1
        avg_n_llm_per_iter = summary['naiverag']['total_llm_calls'] / total_n_iters if total_n_iters > 0 else 0
        
        n_iter_section += f"""
📈 Iteration Statistics:
├─ Total iterations (across all tasks): {total_n_iters}
├─ Mean: {summary['naiverag']['iteration_stats']['mean']:.2f}
├─ Median: {summary['naiverag']['iteration_stats']['median']:.2f}
└─ Range: {summary['naiverag']['iteration_stats']['min']}-{summary['naiverag']['iteration_stats']['max']}

🎯 Judge Decisions:
"""
        for decision, count in summary['naiverag']['judge_decisions'].most_common():
            percentage = (count / sum(summary['naiverag']['judge_decisions'].values())) * 100 if summary['naiverag']['judge_decisions'] else 0
            n_iter_section += f"   ├─ {decision}: {count} ({percentage:.1f}%)\n"
        
        n_iter_section += f"""
💻 LLM Call Statistics:
├─ Total LLM calls: {summary['naiverag']['total_llm_calls']}
├─ Average per query (when used): {summary['naiverag']['avg_llm_calls_per_query']:.2f}
└─ Average per iteration: {avg_n_llm_per_iter:.2f}
"""
    else:
        n_iter_section = """
🟢 NAIVERAG PIPELINE ANALYSIS:
   └─ No queries used NaiveRAG tasks
"""
    
    # Aggregator section
    aggregator_section = """
🎯 AGGREGATOR ANALYSIS:

📋 Decision Distribution:
"""
    for decision, count in summary['aggregator']['decisions'].most_common():
        percentage = (count / total_queries) * 100 if total_queries > 0 else 0
        bar = "█" * int(percentage / 2)
        aggregator_section += f"   ├─ {decision}: {count} ({percentage:.1f}%) {bar}\n"
    
    aggregator_section += f"""
💻 LLM Call Statistics:
├─ Total calls: {total_aggregator} (1 per query)
└─ Always runs after all tasks complete
"""
    
    # Pipeline pattern explanation
    pipeline_doc = """
💡 Pipeline Pattern (Decomposer-Based):
   └─ For each query:
      
      1️⃣  Query Decomposer (always runs):
         └─ Analyzes query and creates sub-queries (1 LLM call)
            ├─ May split into multiple aspects for different pipelines
            ├─ May prefer one pipeline with support from the other
            └─ May send same query to both pipelines
      
      2️⃣  Pipeline Execution (multiple tasks may run):
         
         GraphRAG tasks (if any):
         └─ Up to MAX_ANSWER_JUDGE_ITERS iterations per task
            Each iteration:
            a) Agent 1: entity extraction (1 LLM)
            b) Agent 1b: triple extraction (1 LLM)
            c) Graph retrieval (vector + expansion)
            d) Agent 2: answer generation (1 LLM)
            e) Answer Judge: evaluate (1 LLM)
            f) Query Modifier: refine if needed (1 LLM if not accepted)
            └─ Stop if accepted or max reached
         
         NaiveRAG tasks (if any):
         └─ Up to MAX_ANSWER_JUDGE_ITERS iterations per task
            Each iteration:
            a) Vector search over chunks
            b) Agent 2: answer generation (1 LLM)
            c) Answer Judge: evaluate (1 LLM)
            d) Query Modifier: refine if needed (1 LLM if not accepted)
            └─ Stop if accepted or max reached
      
      3️⃣  Aggregator (runs after all tasks):
         └─ Synthesize final answer from all sub-answers (1 LLM call)
   
   └─ LLM calls breakdown:
      ├─ Always: 1(Decomposer) + 1(Aggregator)
      ├─ GraphRAG per iteration per task:
      │  ├─ If accepted: 1(A1) + 1(A1b) + 1(A2) + 1(AJ) = 4
      │  └─ If not accepted: 1(A1) + 1(A1b) + 1(A2) + 1(AJ) + 1(QM) = 5
      ├─ NaiveRAG per iteration per task:
      │  ├─ If accepted: 1(A2) + 1(AJ) = 2
      │  └─ If not accepted: 1(A2) + 1(AJ) + 1(QM) = 3
      └─ Total = 1(Decomposer) + sum(all_task_costs) + 1(Aggregator)
   
   └─ Agent 1/1b in NaiveRAG not counted (present but ignored)
"""
    
    # Agent breakdown
    agent_breakdown = f"""
🔍 Agent Call Breakdown (across all {total_queries} queries):
├─ Query Decomposer (always runs): {total_decomposer}
│  └─ Called once per query to create sub-queries
├─ GraphRAG pipeline ({summary['graphrag']['queries_with_tasks']} queries):
│  ├─ Agent 1 (entity extraction): {total_g_agent1}
│  │  └─ Called once per GraphRAG task iteration
│  ├─ Agent 1b (triple extraction): {total_g_agent1b}
│  │  └─ Called once per GraphRAG task iteration
│  ├─ Agent 2 (answerer): {total_g_agent2}
│  │  └─ Called once per GraphRAG task iteration
│  ├─ Answer Judge: {total_g_judge}
│  │  └─ Called once per GraphRAG task iteration
│  └─ Query Modifier: {total_g_modifier}
│     └─ Called when judge rejects (not final iteration)
├─ NaiveRAG pipeline ({summary['naiverag']['queries_with_tasks']} queries):
│  ├─ Agent 1/1b: present but ignored in count
│  ├─ Agent 2 (answerer): {total_n_agent2}
│  │  └─ Called once per NaiveRAG task iteration
│  ├─ Answer Judge: {total_n_judge}
│  │  └─ Called once per NaiveRAG task iteration
│  └─ Query Modifier: {total_n_modifier}
│     └─ Called when judge rejects (not final iteration)
├─ Aggregator (final synthesis): {total_aggregator}
│  └─ Called exactly once per query
└─ Total LLM calls: {total_llm}
"""
    
    # Comparison section
    comparison = """
📊 COMPARISON TO OTHER METHODS:
└─ Decomposer-Based characteristics:
   ├─ Decomposer analyzes query and creates targeted sub-queries
   ├─ Can route different aspects to different pipelines
   ├─ Can leverage both pipelines for complementary perspectives
   ├─ Multiple tasks may run (one or more per pipeline)
   ├─ Each task has iterative refinement with judge feedback
   ├─ Variable cost: 1(Decomposer) + task_costs + 1(Aggregator)
   ├─ More flexible than Router (single choice)
   ├─ More focused than Parallel (always runs both)
   ├─ Aggregator synthesizes from multiple sub-answers
   ├─ Best for: Complex queries with multiple aspects
   └─ Trade-off: Higher cost but better aspect coverage
"""
    
    # Notes section
    notes = f"""
📝 Notes:
   - Query Decomposer always runs first (1 LLM call)
   - Creates targeted sub-queries for GraphRAG and/or NaiveRAG
   - Average tasks per query: {avg_total_tasks:.2f}
   - Multiple tasks may run per query (not just one pipeline)
   - Each task has Answer Judge + Query Modifier loop
   - GraphRAG per iteration: 1(A1) + 1(A1b) + 1(A2) + 1(AJ) + [0-1(QM)]
   - NaiveRAG per iteration: 1(A2) + 1(AJ) + [0-1(QM)]
   - Agent 1/1b present in NaiveRAG but not counted per user request
   - Aggregator always runs once after all tasks complete (1 LLM call)
   - Aggregator decisions: {dict(summary['aggregator']['decisions'])}
   - Average GraphRAG iterations (when used): {avg_g_iters:.2f}
   - Average NaiveRAG iterations (when used): {avg_n_iters:.2f}
   - More comprehensive than Router, more focused than Parallel
   - Cost between Router (cheapest) and Parallel (most expensive)
   - Embedding calls: high due to multiple retrieval operations
"""
    
    return (
        header + overall + averages + ranges + decomposer_section +
        g_iter_section + n_iter_section + aggregator_section +
        pipeline_doc + agent_breakdown + comparison + notes
    )


def main():
    """Main execution function."""
    
    separator = "=" * 70
    title = "DECOMPOSER-BASED MULTI-AGENT RAG LLM CALL ANALYZER"
    subtitle = "(Decomposer → GraphRAG/NaiveRAG Tasks → Aggregator)"
    
    print(separator)
    print(title)
    print(subtitle)
    print(separator)
    print(f"Input folder: {LOG_FOLDER.resolve()}")
    print(f"Output folder: {OUTPUT_FOLDER.resolve()}")
    print()
    
    results = analyze_all_logs()
    
    if not results:
        print("No results to save. Exiting.")
        return
    
    save_results(results)
    
    print(separator)
    print("✓ Analysis complete!")
    print(separator)


if __name__ == "__main__":
    main()