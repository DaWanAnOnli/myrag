# analyze_llm_calls_multi_agent.py
"""
Analyzes log files from the Multi-Agent RAG system with parallel GraphRAG and NaiveRAG pipelines.
Handles interleaved logs from parallel threads using [pipe=G] and [pipe=N] tags.
"""

import re
import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import csv
from collections import Counter

# Relative paths
SCRIPT_DIR = Path(__file__).resolve().parent
LOG_FOLDER = SCRIPT_DIR / "../../4b_retrieval/4b_iv_multi_agents/question_terminal_logs_multi_agent/no_11_approach_1_both_3rd"
OUTPUT_FOLDER = SCRIPT_DIR / "llm_calls_analysis_results_approach_1_both_3rd"


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


def filter_lines_by_pipe(content: str, pipe_tag: str) -> str:
    """
    Filter log lines by pipeline tag.
    pipe_tag: 'G' for GraphRAG, 'N' for NaiveRAG
    Returns concatenated lines that match the pipeline tag.
    """
    lines = content.split('\n')
    filtered = []
    
    tag_pattern = f"[pipe={pipe_tag}]"
    
    for line in lines:
        if tag_pattern in line:
            filtered.append(line)
    
    return '\n'.join(filtered)


def extract_iterations_from_filtered(filtered_content: str, pipeline: str) -> List[Dict[str, any]]:
    """
    Extract iteration data from filtered pipeline content.
    pipeline: 'G' for GraphRAG, 'N' for NaiveRAG
    """
    iterations = []
    
    if pipeline == 'G':
        # Pattern: "--- GraphRAG Iteration X/Y START ---" (with [pipe=G])
        pattern = r'--- GraphRAG Iteration (\d+)/(\d+) START ---'
        judge_accept = r'\[G:Loop\] Answer acceptable'
        judge_reject_cap = r'\[G:Loop\] Iteration cap reached'
        modifier = r'\[Agent QM\] Modified query'
    else:  # N
        # Pattern: "--- NaiveRAG Iteration X/Y ---" (with [pipe=N])
        pattern = r'--- NaiveRAG Iteration (\d+)/(\d+) ---'
        judge_accept = r'\[N:Iter \d+\] Judge deemed answer ACCEPTABLE'
        judge_insufficient = r'\[N:Iter \d+\] Judge indicates insufficiency'
        judge_limit = r'\[N:Iter \d+\] Iteration limit reached'
        modifier = r'\[N:Iter \d+\] Modified query'
    
    matches = re.findall(pattern, filtered_content)
    
    for iter_num_str, max_iter_str in matches:
        iter_num = int(iter_num_str)
        max_iter = int(max_iter_str)
        
        iter_data = {
            'iteration': iter_num,
            'max_iterations': max_iter,
            'judge_decision': None,
            'query_modified': False
        }
        
        # For judging decisions, we need to look at lines with this iteration context
        # Pattern: [pipe=X] [iter=Y/Z]
        if pipeline == 'G':
            iter_context = f"[pipe=G] [iter={iter_num}/{max_iter}]"
        else:
            iter_context = f"[pipe=N] [iter={iter_num}/{max_iter}]"
        
        # Get lines for this specific iteration
        iter_lines = [line for line in filtered_content.split('\n') if iter_context in line]
        iter_block = '\n'.join(iter_lines)
        
        if pipeline == 'G':
            if re.search(judge_accept, iter_block):
                iter_data['judge_decision'] = 'acceptable'
            elif re.search(judge_reject_cap, iter_block):
                iter_data['judge_decision'] = 'rejected_at_cap'
            elif re.search(modifier, iter_block):
                iter_data['judge_decision'] = 'insufficient'
                iter_data['query_modified'] = True
        else:  # N
            if re.search(judge_accept, iter_block):
                iter_data['judge_decision'] = 'acceptable'
            elif re.search(judge_limit, iter_block):
                iter_data['judge_decision'] = 'stopped_at_limit'
            elif re.search(judge_insufficient, iter_block):
                iter_data['judge_decision'] = 'insufficient'
                if re.search(modifier, iter_block):
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


def count_llm_calls_in_filtered(filtered_content: str, pipeline: str) -> Dict[str, int]:
    """
    Count LLM calls in filtered pipeline content.
    pipeline: 'G' for GraphRAG, 'N' for NaiveRAG
    """
    counts = {
        'agent1': 0,
        'agent1b': 0,
        'agent2': 0,
        'judge': 0,
        'modifier': 0
    }
    
    if not filtered_content:
        return counts
    
    if pipeline == 'G':
        counts['agent1'] = len(re.findall(r'\[Agent 1\] Prompt:', filtered_content))
        counts['agent1b'] = len(re.findall(r'\[Agent 1b\] Prompt:', filtered_content))
        counts['agent2'] = len(re.findall(r'\[Agent 2\] Prompt:', filtered_content))
        counts['judge'] = len(re.findall(r'\[Agent AJ\] Prompt:', filtered_content))
        counts['modifier'] = len(re.findall(r'\[Agent QM\] Prompt:', filtered_content))
    else:  # N
        # Ignore Agent 1/1b in NaiveRAG per user request
        counts['agent2'] = len(re.findall(r'\[Agent 2\] Prompt:', filtered_content))
        counts['judge'] = len(re.findall(r'\[N:Judge\] Prompt:', filtered_content))
        counts['modifier'] = len(re.findall(r'\[N:Modifier\] Prompt:', filtered_content))
    
    return counts


def extract_aggregator_decision(content: str) -> Optional[str]:
    """Extract the aggregator's final decision."""
    match = re.search(r'\[Aggregator\] Decision=(\w+)', content)
    if match:
        return match.group(1)
    
    match = re.search(r'- Aggregator decision:\s*(\w+)', content)
    if match:
        return match.group(1)
    
    return None


def analyze_log_file(log_path: Path) -> Dict:
    """Analyze a single log file."""
    
    result = {
        'log_file': log_path.name,
        'query': '',
        
        # GraphRAG pipeline
        'graphrag_iterations_used': 0,
        'graphrag_agent1_calls': 0,
        'graphrag_agent1b_calls': 0,
        'graphrag_agent2_calls': 0,
        'graphrag_judge_calls': 0,
        'graphrag_modifier_calls': 0,
        'graphrag_per_iteration': [],
        
        # NaiveRAG pipeline
        'naiverag_iterations_used': 0,
        'naiverag_agent2_calls': 0,
        'naiverag_judge_calls': 0,
        'naiverag_modifier_calls': 0,
        'naiverag_per_iteration': [],
        
        # Aggregator
        'aggregator_calls': 1,
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
        
        # Filter lines by pipeline tag
        graphrag_content = filter_lines_by_pipe(content, 'G')
        naiverag_content = filter_lines_by_pipe(content, 'N')
        
        # Extract iterations
        g_iterations = extract_iterations_from_filtered(graphrag_content, 'G')
        result['graphrag_iterations_used'] = len(g_iterations)
        result['graphrag_per_iteration'] = g_iterations
        
        n_iterations = extract_iterations_from_filtered(naiverag_content, 'N')
        result['naiverag_iterations_used'] = len(n_iterations)
        result['naiverag_per_iteration'] = n_iterations
        
        # Count LLM calls
        g_counts = count_llm_calls_in_filtered(graphrag_content, 'G')
        result['graphrag_agent1_calls'] = g_counts['agent1']
        result['graphrag_agent1b_calls'] = g_counts['agent1b']
        result['graphrag_agent2_calls'] = g_counts['agent2']
        result['graphrag_judge_calls'] = g_counts['judge']
        result['graphrag_modifier_calls'] = g_counts['modifier']
        
        n_counts = count_llm_calls_in_filtered(naiverag_content, 'N')
        result['naiverag_agent2_calls'] = n_counts['agent2']
        result['naiverag_judge_calls'] = n_counts['judge']
        result['naiverag_modifier_calls'] = n_counts['modifier']
        
        # Extract aggregator decision
        result['aggregator_decision'] = extract_aggregator_decision(content)
        
        # Total LLM calls
        result['total_llm_calls'] = (
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
        import traceback
        traceback.print_exc()
    
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


# Keep all the other functions from before (create_summary_statistics, save_results, create_text_summary, main)
# They don't need to change - just copy them from the previous version

def create_summary_statistics(results: List[Dict]) -> Dict[str, any]:
    """Create comprehensive summary statistics."""
    
    summary = {
        'total_queries': len(results),
        'graphrag': {
            'iteration_stats': {'mean': 0.0, 'median': 0.0, 'min': 0, 'max': 0, 'distribution': Counter()},
            'judge_decisions': Counter(),
            'avg_llm_calls_per_query': 0.0,
            'total_llm_calls': 0
        },
        'naiverag': {
            'iteration_stats': {'mean': 0.0, 'median': 0.0, 'min': 0, 'max': 0, 'distribution': Counter()},
            'judge_decisions': Counter(),
            'avg_llm_calls_per_query': 0.0,
            'total_llm_calls': 0
        },
        'aggregator': {
            'decisions': Counter(),
            'total_calls': len(results)
        },
        'avg_total_llm_calls_per_query': 0.0,
        'avg_embed_calls_per_query': 0.0
    }
    
    if not results:
        return summary
    
    import statistics
    
    # GraphRAG statistics
    g_iters = [r['graphrag_iterations_used'] for r in results if r['graphrag_iterations_used'] > 0]
    if g_iters:
        summary['graphrag']['iteration_stats']['mean'] = statistics.mean(g_iters)
        summary['graphrag']['iteration_stats']['median'] = statistics.median(g_iters)
        summary['graphrag']['iteration_stats']['min'] = min(g_iters)
        summary['graphrag']['iteration_stats']['max'] = max(g_iters)
        summary['graphrag']['iteration_stats']['distribution'] = Counter(g_iters)
    
    for r in results:
        for iter_data in r.get('graphrag_per_iteration', []):
            decision = iter_data.get('judge_decision')
            if decision:
                summary['graphrag']['judge_decisions'][decision] += 1
        
        g_llm = (
            r['graphrag_agent1_calls'] + r['graphrag_agent1b_calls'] +
            r['graphrag_agent2_calls'] + r['graphrag_judge_calls'] +
            r['graphrag_modifier_calls']
        )
        summary['graphrag']['total_llm_calls'] += g_llm
    
    if results:
        summary['graphrag']['avg_llm_calls_per_query'] = summary['graphrag']['total_llm_calls'] / len(results)
    
    # NaiveRAG statistics
    n_iters = [r['naiverag_iterations_used'] for r in results if r['naiverag_iterations_used'] > 0]
    if n_iters:
        summary['naiverag']['iteration_stats']['mean'] = statistics.mean(n_iters)
        summary['naiverag']['iteration_stats']['median'] = statistics.median(n_iters)
        summary['naiverag']['iteration_stats']['min'] = min(n_iters)
        summary['naiverag']['iteration_stats']['max'] = max(n_iters)
        summary['naiverag']['iteration_stats']['distribution'] = Counter(n_iters)
    
    for r in results:
        for iter_data in r.get('naiverag_per_iteration', []):
            decision = iter_data.get('judge_decision')
            if decision:
                summary['naiverag']['judge_decisions'][decision] += 1
        
        n_llm = (
            r['naiverag_agent2_calls'] + r['naiverag_judge_calls'] +
            r['naiverag_modifier_calls']
        )
        summary['naiverag']['total_llm_calls'] += n_llm
    
    if results:
        summary['naiverag']['avg_llm_calls_per_query'] = summary['naiverag']['total_llm_calls'] / len(results)
    
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
    csv_path = output_folder / f"multi_agent_detailed_{timestamp}.csv"
    csv_fields = [
        'log_file', 'query',
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
    json_path = output_folder / f"multi_agent_detailed_{timestamp}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"JSON results saved to: {json_path}")
    
    # 3. Summary JSON
    summary = create_summary_statistics(results)
    summary_path = output_folder / f"multi_agent_summary_{timestamp}.json"
    summary_copy = json.loads(json.dumps(summary, default=dict))
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary_copy, f, indent=2, ensure_ascii=False)
    
    print(f"Summary statistics saved to: {summary_path}")
    
    # 4. Human-readable report
    readable_path = output_folder / f"multi_agent_readable_{timestamp}.txt"
    with open(readable_path, 'w', encoding='utf-8') as f:
        separator = "=" * 80
        
        f.write(separator + "\n")
        f.write("MULTI-AGENT RAG ANALYSIS - HUMAN READABLE REPORT\n")
        f.write(separator + "\n")
        
        f.write("SUMMARY STATISTICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total queries analyzed: {summary['total_queries']}\n")
        
        # GraphRAG section
        f.write("GRAPHRAG PIPELINE\n")
        f.write("-" * 40 + "\n")
        g = summary['graphrag']
        f.write("Iteration Statistics:\n")
        f.write(f"  Mean: {g['iteration_stats']['mean']:.2f}\n")
        f.write(f"  Median: {g['iteration_stats']['median']:.2f}\n")
        f.write(f"  Range: {g['iteration_stats']['min']}-{g['iteration_stats']['max']}\n")
        f.write("  Distribution:\n")
        for iters, count in sorted(g['iteration_stats']['distribution'].items()):
            pct = (count / summary['total_queries'] * 100) if summary['total_queries'] > 0 else 0
            f.write(f"    {iters} iterations: {count} queries ({pct:.1f}%)\n")
        
        f.write("\nJudge Decisions:\n")
        for decision, count in g['judge_decisions'].items():
            f.write(f"  {decision}: {count}\n")
        
        f.write("\nLLM Calls:\n")
        f.write(f"  Total: {g['total_llm_calls']}\n")
        f.write(f"  Average per query: {g['avg_llm_calls_per_query']:.2f}\n")
        
        # NaiveRAG section
        f.write("NAIVERAG PIPELINE\n")
        f.write("-" * 40 + "\n")
        n = summary['naiverag']
        f.write("Iteration Statistics:\n")
        f.write(f"  Mean: {n['iteration_stats']['mean']:.2f}\n")
        f.write(f"  Median: {n['iteration_stats']['median']:.2f}\n")
        f.write(f"  Range: {n['iteration_stats']['min']}-{n['iteration_stats']['max']}\n")
        f.write("  Distribution:\n")
        for iters, count in sorted(n['iteration_stats']['distribution'].items()):
            pct = (count / summary['total_queries'] * 100) if summary['total_queries'] > 0 else 0
            f.write(f"    {iters} iterations: {count} queries ({pct:.1f}%)\n")
        
        f.write("\nJudge Decisions:\n")
        for decision, count in n['judge_decisions'].items():
            f.write(f"  {decision}: {count}\n")
        
        f.write("\nLLM Calls:\n")
        f.write(f"  Total: {n['total_llm_calls']}\n")
        f.write(f"  Average per query: {n['avg_llm_calls_per_query']:.2f}\n")
        
        # Aggregator section
        f.write("AGGREGATOR\n")
        f.write("-" * 40 + "\n")
        a = summary['aggregator']
        f.write(f"Total calls: {a['total_calls']}\n")
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
    
    # 5. Text summary - use the same create_text_summary function from before
    summary_text = create_text_summary(results, summary)
    summary_txt_path = output_folder / f"multi_agent_summary_{timestamp}.txt"
    with open(summary_txt_path, 'w', encoding='utf-8') as f:
        f.write(summary_text)
    
    print(f"Summary saved to: {summary_txt_path}")
    print(summary_text)


# (Keep the create_text_summary function from the previous version - it doesn't need changes)
# I'll include it here for completeness, but it's the same as before

def create_text_summary(results: List[Dict], summary: Dict) -> str:
    """Create formatted text summary."""
    
    if not results:
        return "No results to summarize."
    
    total_queries = len(results)
    
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
║       MULTI-AGENT RAG LLM CALL ANALYSIS SUMMARY             ║
║   (Parallel GraphRAG + NaiveRAG → Aggregator)               ║
╚══════════════════════════════════════════════════════════════╝
"""
    
    overall = f"""
📊 Overall Statistics:
├─ Total log files analyzed: {total_queries}
├─ Total LLM calls: {total_llm}
│  ├─ GraphRAG pipeline:
│  │  ├─ Agent 1 (entity extraction): {total_g_agent1}
│  │  ├─ Agent 1b (triple extraction): {total_g_agent1b}
│  │  ├─ Agent 2 (answerer): {total_g_agent2}
│  │  ├─ Answer Judge: {total_g_judge}
│  │  └─ Query Modifier: {total_g_modifier}
│  ├─ NaiveRAG pipeline:
│  │  ├─ Agent 2 (answerer): {total_n_agent2}
│  │  ├─ Answer Judge: {total_n_judge}
│  │  └─ Query Modifier: {total_n_modifier}
│  └─ Aggregator (final synthesis): {total_aggregator}
└─ Total embedding calls: {total_embed}
"""
    
    avg_llm_per_query = total_llm / total_queries if total_queries > 0 else 0
    avg_embed_per_query = total_embed / total_queries if total_queries > 0 else 0
    avg_g_iters = summary['graphrag']['iteration_stats']['mean']
    avg_n_iters = summary['naiverag']['iteration_stats']['mean']
    
    averages = f"""
📈 Per-Query Averages:
├─ Average LLM calls per query: {avg_llm_per_query:.2f}
├─ Average embedding calls per query: {avg_embed_per_query:.2f}
├─ Average GraphRAG iterations per query: {avg_g_iters:.2f}
└─ Average NaiveRAG iterations per query: {avg_n_iters:.2f}
"""
    
    min_g_iter = summary['graphrag']['iteration_stats']['min']
    max_g_iter = summary['graphrag']['iteration_stats']['max']
    min_n_iter = summary['naiverag']['iteration_stats']['min']
    max_n_iter = summary['naiverag']['iteration_stats']['max']
    
    ranges = f"""
📉 Range:
├─ Minimum LLM calls in a query: {min_llm}
├─ Maximum LLM calls in a query: {max_llm}
├─ GraphRAG iterations range: {min_g_iter}-{max_g_iter}
└─ NaiveRAG iterations range: {min_n_iter}-{max_n_iter}
"""
    
    # GraphRAG iteration analysis
    g_iter_section = f"""
🔵 GRAPHRAG PIPELINE ANALYSIS ({total_queries} queries):

📊 Iteration Distribution:
"""
    g_dist = summary['graphrag']['iteration_stats']['distribution']
    for iters in sorted(g_dist.keys()):
        count = g_dist[iters]
        percentage = (count / total_queries) * 100
        bar = "█" * int(percentage / 2)
        g_iter_section += f"   ├─ {iters} iterations: {count} queries ({percentage:.1f}%) {bar}\n"
    
    total_g_iters = int(summary['graphrag']['iteration_stats']['mean'] * total_queries)
    avg_g_llm_per_iter = summary['graphrag']['total_llm_calls'] / total_g_iters if total_g_iters > 0 else 0
    
    g_iter_section += f"""
📈 Iteration Statistics:
├─ Total iterations: {total_g_iters}
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
├─ Average per query: {summary['graphrag']['avg_llm_calls_per_query']:.2f}
└─ Average per iteration: {avg_g_llm_per_iter:.2f}
"""
    
    # NaiveRAG iteration analysis
    n_iter_section = f"""
🟢 NAIVERAG PIPELINE ANALYSIS ({total_queries} queries):

📊 Iteration Distribution:
"""
    n_dist = summary['naiverag']['iteration_stats']['distribution']
    for iters in sorted(n_dist.keys()):
        count = n_dist[iters]
        percentage = (count / total_queries) * 100
        bar = "█" * int(percentage / 2)
        n_iter_section += f"   ├─ {iters} iterations: {count} queries ({percentage:.1f}%) {bar}\n"
    
    total_n_iters = int(summary['naiverag']['iteration_stats']['mean'] * total_queries)
    avg_n_llm_per_iter = summary['naiverag']['total_llm_calls'] / total_n_iters if total_n_iters > 0 else 0
    
    n_iter_section += f"""
📈 Iteration Statistics:
├─ Total iterations: {total_n_iters}
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
├─ Average per query: {summary['naiverag']['avg_llm_calls_per_query']:.2f}
└─ Average per iteration: {avg_n_llm_per_iter:.2f}
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
└─ Always runs exactly once after both pipelines complete
"""
    
    pipeline_doc = """
💡 Pipeline Pattern (Parallel Multi-Agent):
   └─ For each query:
      
      ┌─ PARALLEL EXECUTION (both run simultaneously) ─┐
      │                                                 │
      │  1️⃣  GraphRAG Pipeline:                         │
      │     └─ Up to MAX_ANSWER_JUDGE_ITERS iterations │
      │        Each iteration:                          │
      │        a) Agent 1: entity extraction (1 LLM)    │
      │        b) Agent 1b: triple extraction (1 LLM)   │
      │        c) Graph retrieval (vector + expansion)  │
      │        d) Agent 2: answer generation (1 LLM)    │
      │        e) Answer Judge: evaluate (1 LLM)        │
      │        f) Query Modifier: refine if needed      │
      │           (1 LLM if not accepted)               │
      │        └─ Stop if accepted or max reached       │
      │                                                 │
      │  2️⃣  NaiveRAG Pipeline:                          │
      │     └─ Up to MAX_ANSWER_JUDGE_ITERS iterations │
      │        Each iteration:                          │
      │        a) Vector search over chunks             │
      │        b) Agent 2: answer generation (1 LLM)    │
      │        c) Answer Judge: evaluate (1 LLM)        │
      │        d) Query Modifier: refine if needed      │
      │           (1 LLM if not accepted)               │
      │        └─ Stop if accepted or max reached       │
      │                                                 │
      └─────────────────────────────────────────────────┘
      
      3️⃣  Aggregator (runs after both complete):
         └─ Synthesize final answer (1 LLM call)
   
   └─ LLM calls breakdown:
      ├─ GraphRAG per iteration:
      │  ├─ If accepted: 1(A1) + 1(A1b) + 1(A2) + 1(AJ) = 4
      │  └─ If not accepted: 1(A1) + 1(A1b) + 1(A2) + 1(AJ) + 1(QM) = 5
      ├─ NaiveRAG per iteration:
      │  ├─ If accepted: 1(A2) + 1(AJ) = 2
      │  └─ If not accepted: 1(A2) + 1(AJ) + 1(QM) = 3
      └─ Total per query = GraphRAG_total + NaiveRAG_total + 1(Aggregator)
   
   └─ Agent 1/1b in NaiveRAG not counted (present but ignored)
"""
    
    agent_breakdown = f"""
🔍 Agent Call Breakdown (across all {total_queries} queries):
├─ GraphRAG pipeline:
│  ├─ Agent 1 (entity extraction): {total_g_agent1}
│  │  └─ Called once per GraphRAG iteration
│  ├─ Agent 1b (triple extraction): {total_g_agent1b}
│  │  └─ Called once per GraphRAG iteration
│  ├─ Agent 2 (answerer): {total_g_agent2}
│  │  └─ Called once per GraphRAG iteration
│  ├─ Answer Judge: {total_g_judge}
│  │  └─ Called once per GraphRAG iteration
│  └─ Query Modifier: {total_g_modifier}
│     └─ Called when judge rejects (not final iteration)
├─ NaiveRAG pipeline:
│  ├─ Agent 1/1b: present but ignored in count
│  ├─ Agent 2 (answerer): {total_n_agent2}
│  │  └─ Called once per NaiveRAG iteration
│  ├─ Answer Judge: {total_n_judge}
│  │  └─ Called once per NaiveRAG iteration
│  └─ Query Modifier: {total_n_modifier}
│     └─ Called when judge rejects (not final iteration)
├─ Aggregator (final synthesis): {total_aggregator}
│  └─ Called exactly once per query
└─ Total LLM calls: {total_llm}
"""
    
    comparison = """
📊 COMPARISON TO OTHER METHODS:
└─ Multi-Agent Parallel characteristics:
   ├─ Two independent pipelines run simultaneously
   ├─ Each pipeline has iterative refinement with judge feedback
   ├─ Query modification within each pipeline independently
   ├─ Stops when respective judge accepts or max iterations reached
   ├─ Variable cost per query: depends on both pipeline iterations
   ├─ Total cost = GraphRAG_cost + NaiveRAG_cost + Aggregator(1)
   ├─ Benefit: Diverse retrieval strategies (graph + vector)
   ├─ Benefit: Cross-validation through aggregator synthesis
   └─ Best for: Complex queries needing multiple retrieval perspectives
"""
    
    notes = f"""
📝 Notes:
   - Parallel execution: both pipelines run independently in threads
   - Logs are interleaved; filtered by [pipe=G] and [pipe=N] tags
   - Each pipeline has its own Answer Judge + Query Modifier loop
   - GraphRAG per iteration: 1(A1) + 1(A1b) + 1(A2) + 1(AJ) + [0-1(QM)]
   - NaiveRAG per iteration: 1(A2) + 1(AJ) + [0-1(QM)]
   - Agent 1/1b present in NaiveRAG but not counted per user request
   - Aggregator always runs exactly once after both pipelines complete
   - Average GraphRAG iterations: {avg_g_iters:.2f}
   - Average NaiveRAG iterations: {avg_n_iters:.2f}
   - Pipelines can have different iteration counts independently
   - Embedding calls: very high due to parallel retrieval in both pipelines
   - Most comprehensive but also most expensive approach
   - Provides robustness through diverse retrieval strategies
"""
    
    return (
        header + overall + averages + ranges + 
        g_iter_section + n_iter_section + aggregator_section +
        pipeline_doc + agent_breakdown + comparison + notes
    )


def main():
    """Main execution function."""
    
    separator = "=" * 70
    title = "MULTI-AGENT RAG LLM CALL ANALYZER"
    subtitle = "(Parallel GraphRAG + NaiveRAG with Aggregator)"
    
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