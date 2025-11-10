# analyze_llm_calls_supervisor.py
"""
Analyzes log files from the Agentic Supervisor RAG system.
Tracks LLM calls including subgoal generation, per-subgoal pipelines, and final aggregation.
Counts subgoals generated and aggregator decisions at both subgoal and final levels.
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
LOG_FOLDER = SCRIPT_DIR / "../../4b_retrieval/4b_iv_multi_agents/question_terminal_logs_multi_agent/approach_2_both_2_subgoal_new_fix"
OUTPUT_FOLDER = SCRIPT_DIR / "llm_calls_analysis_results_approach_2_both_2_subgoal_new_fix"


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


def extract_subgoal_info(log_path: Path) -> tuple:
    """
    Extract subgoal count and subgoal details from the log.
    Returns: (num_subgoals, list_of_subgoal_info)
    """
    num_subgoals = 0
    subgoal_details = []
    
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
            # Extract subgoal count
            match = re.search(r'Produced (\d+) subgoal\(s\):', content)
            if match:
                num_subgoals = int(match.group(1))
            
            # If not found, try from summary
            if num_subgoals == 0:
                match = re.search(r'- Subgoals executed:\s*(\d+)', content)
                if match:
                    num_subgoals = int(match.group(1))
            
            # Extract per-subgoal aggregator decisions
            # Look for patterns like [sg=sg1] ... [Aggregator] Decision:
            subgoal_blocks = re.findall(
                r'\[sg=(sg\d+)\].*?\[Aggregator\] Decision: chosen=(\w+)',
                content,
                re.DOTALL
            )
            
            for sg_id, chosen in subgoal_blocks:
                subgoal_details.append({
                    'subgoal_id': sg_id,
                    'aggregator_chosen': chosen
                })
    
    except Exception as e:
        print(f"Error extracting subgoal info from {log_path.name}: {e}")
    
    return num_subgoals, subgoal_details


def extract_final_aggregator_decision(log_path: Path) -> Dict[str, any]:
    """Extract final aggregator decision from the log."""
    decision = {
        'confidence': 0.0,
        'rationale': ''
    }
    
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
            # Look for FinalAggregator confidence
            conf_match = re.search(r'\[FinalAggregator\] Confidence=([\d.]+)', content)
            if conf_match:
                decision['confidence'] = float(conf_match.group(1))
    
    except Exception as e:
        print(f"Error extracting final aggregator from {log_path.name}: {e}")
    
    return decision


def analyze_log_file(log_path: Path) -> Dict:
    """
    Analyze a single log file and count LLM calls.
    
    Returns dict with metrics, counting Subgoal Gen + per-subgoal pipelines + Final Agg.
    Ignores Agent 1/1b in Naive RAG pipeline.
    """
    
    result = {
        'log_file': log_path.name,
        'query': '',
        'num_subgoals': 0,
        'subgoal_gen_calls': 0,  # Subgoal Generator (1 JSON call)
        'agent1_calls': 0,  # Agent 1 (only in GraphRAG, across all subgoals)
        'agent1b_calls': 0,  # Agent 1b (only in GraphRAG, across all subgoals)
        'agent2_naive_calls': 0,  # Agent 2 in Naive (across all subgoals)
        'agent2_graphrag_calls': 0,  # Agent 2 in GraphRAG (across all subgoals)
        'per_subgoal_aggregator_calls': 0,  # Per-subgoal aggregator
        'final_aggregator_calls': 0,  # Final Aggregator (1 JSON call)
        'total_llm_calls': 0,
        'embed_calls': 0,
        'subgoal_aggregator_choices': [],  # List of choices per subgoal
        'final_aggregator_confidence': 0.0,
        'mode': 'unknown'
    }
    
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
            # Extract query
            result['query'] = extract_query_from_log(log_path)
            
            # Detect mode
            if "Agentic Supervisor run started" in content:
                result['mode'] = 'supervisor'
            
            # Extract subgoal info
            num_subgoals, subgoal_details = extract_subgoal_info(log_path)
            result['num_subgoals'] = num_subgoals
            result['subgoal_aggregator_choices'] = [s['aggregator_chosen'] for s in subgoal_details]
            
            # Extract final aggregator
            final_agg = extract_final_aggregator_decision(log_path)
            result['final_aggregator_confidence'] = final_agg['confidence']
            
            # Count Subgoal Generator prompts (should be 1)
            subgoal_gen_matches = re.findall(r'\[SubgoalGenerator\] Prompt:', content)
            result['subgoal_gen_calls'] = len(subgoal_gen_matches)
            
            # Count Agent 1 prompts (only in GraphRAG, never in Naive)
            agent1_matches = re.findall(r'\[Agent 1\] Prompt:', content)
            result['agent1_calls'] = len(agent1_matches)
            
            # Count Agent 1b prompts (only in GraphRAG, never in Naive)
            agent1b_matches = re.findall(r'\[Agent 1b\] Prompt:', content)
            result['agent1b_calls'] = len(agent1b_matches)
            
            # Count Agent 2 prompts by pipeline
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
            
            # Count per-subgoal Aggregator prompts
            # These are marked with [sg=...] context
            per_sg_agg = len(re.findall(r'\[sg=sg\d+\].*?\[Aggregator\] Prompt:', content, re.DOTALL))
            result['per_subgoal_aggregator_calls'] = per_sg_agg
            
            # Count Final Aggregator prompts (should be 1)
            final_agg_matches = re.findall(r'\[FinalAggregator\] Prompt:', content)
            result['final_aggregator_calls'] = len(final_agg_matches)
            
            # Total LLM calls
            result['total_llm_calls'] = (
                result['subgoal_gen_calls'] +
                result['agent1_calls'] +
                result['agent1b_calls'] +
                result['agent2_naive_calls'] +
                result['agent2_graphrag_calls'] +
                result['per_subgoal_aggregator_calls'] +
                result['final_aggregator_calls']
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


def create_supervisor_summary(results: List[Dict]) -> Dict[str, any]:
    """Create comprehensive supervisor RAG summary statistics."""
    
    supervisor_results = [r for r in results if r['mode'] == 'supervisor']
    
    summary = {
        'total_queries': len(supervisor_results),
        'subgoal_stats': {
            'total': 0,
            'mean': 0.0,
            'median': 0.0,
            'min': 0,
            'max': 0,
            'distribution': Counter()
        },
        'per_subgoal_aggregator_choices': Counter(),
        'avg_llm_calls_per_query': 0.0,
        'avg_llm_calls_per_subgoal': 0.0,
        'final_aggregator_confidence': {
            'mean': 0.0,
            'median': 0.0,
            'min': 0.0,
            'max': 0.0
        },
        'by_subgoal_count': {}
    }
    
    if not supervisor_results:
        return summary
    
    # Subgoal statistics
    subgoal_counts = [r['num_subgoals'] for r in supervisor_results if r['num_subgoals'] > 0]
    
    if subgoal_counts:
        import statistics
        summary['subgoal_stats']['total'] = sum(subgoal_counts)
        summary['subgoal_stats']['mean'] = statistics.mean(subgoal_counts)
        summary['subgoal_stats']['median'] = statistics.median(subgoal_counts)
        summary['subgoal_stats']['min'] = min(subgoal_counts)
        summary['subgoal_stats']['max'] = max(subgoal_counts)
        summary['subgoal_stats']['distribution'] = Counter(subgoal_counts)
    
    # Per-subgoal aggregator choices
    for r in supervisor_results:
        for choice in r.get('subgoal_aggregator_choices', []):
            if choice:
                summary['per_subgoal_aggregator_choices'][choice] += 1
    
    # Final aggregator confidence stats
    confidences = [r.get('final_aggregator_confidence', 0.0) for r in supervisor_results if r.get('final_aggregator_confidence', 0) > 0]
    
    if confidences:
        import statistics
        summary['final_aggregator_confidence']['mean'] = statistics.mean(confidences)
        summary['final_aggregator_confidence']['median'] = statistics.median(confidences)
        summary['final_aggregator_confidence']['min'] = min(confidences)
        summary['final_aggregator_confidence']['max'] = max(confidences)
    
    # Average LLM calls
    if supervisor_results:
        total_llm = sum(r['total_llm_calls'] for r in supervisor_results)
        summary['avg_llm_calls_per_query'] = total_llm / len(supervisor_results)
        
        total_subgoals = sum(r['num_subgoals'] for r in supervisor_results if r['num_subgoals'] > 0)
        if total_subgoals > 0:
            # Subtract overhead (subgoal gen + final agg) from total
            subgoal_llm = total_llm - (2 * len(supervisor_results))  # 2 = subgoal gen + final agg
            summary['avg_llm_calls_per_subgoal'] = subgoal_llm / total_subgoals if total_subgoals > 0 else 0.0
    
    # Group by subgoal count
    for r in supervisor_results:
        count = r['num_subgoals']
        if count not in summary['by_subgoal_count']:
            summary['by_subgoal_count'][count] = {
                'count': 0,
                'avg_llm_calls': 0.0,
                'queries': []
            }
        
        summary['by_subgoal_count'][count]['count'] += 1
        summary['by_subgoal_count'][count]['queries'].append({
            'query': r['query'][:100],
            'llm_calls': r['total_llm_calls'],
            'final_confidence': r.get('final_aggregator_confidence', 0.0)
        })
    
    # Calculate average LLM calls by subgoal count
    for count, data in summary['by_subgoal_count'].items():
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
        'log_file', 'query', 'mode', 'num_subgoals',
        'total_llm_calls', 'subgoal_gen_calls', 'agent1_calls', 'agent1b_calls',
        'agent2_naive_calls', 'agent2_graphrag_calls',
        'per_subgoal_aggregator_calls', 'final_aggregator_calls',
        'final_aggregator_confidence', 'embed_calls'
    ]
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(results)
    
    print(f"Detailed results saved to: {csv_path}")
    
    # 2. Save full JSON (includes subgoal choices)
    json_path = output_folder / f"llm_calls_detailed_{timestamp}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"JSON results saved to: {json_path}")
    
    # 3. Create and save supervisor summary
    supervisor_summary = create_supervisor_summary(results)
    summary_path = output_folder / f"supervisor_summary_{timestamp}.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        # Convert Counter to dict
        summary_copy = supervisor_summary.copy()
        summary_copy['subgoal_stats']['distribution'] = dict(supervisor_summary['subgoal_stats']['distribution'])
        summary_copy['per_subgoal_aggregator_choices'] = dict(supervisor_summary['per_subgoal_aggregator_choices'])
        json.dump(summary_copy, f, indent=2, ensure_ascii=False)
    
    print(f"Supervisor summary saved to: {summary_path}")
    
    # 4. Save supervisor statistics CSV
    stats_csv = output_folder / f"supervisor_statistics_{timestamp}.csv"
    with open(stats_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Value'])
        writer.writerow(['Total Queries', supervisor_summary['total_queries']])
        writer.writerow([''])
        writer.writerow(['Subgoal Statistics:', ''])
        writer.writerow(['  Total Subgoals', supervisor_summary['subgoal_stats']['total']])
        writer.writerow(['  Mean', f"{supervisor_summary['subgoal_stats']['mean']:.2f}"])
        writer.writerow(['  Median', f"{supervisor_summary['subgoal_stats']['median']:.2f}"])
        writer.writerow(['  Min/Max', f"{supervisor_summary['subgoal_stats']['min']}/{supervisor_summary['subgoal_stats']['max']}"])
        writer.writerow([''])
        writer.writerow(['Subgoal Distribution:', ''])
        for count, freq in sorted(supervisor_summary['subgoal_stats']['distribution'].items()):
            writer.writerow([f'  {count} subgoals', freq])
        writer.writerow([''])
        writer.writerow(['Per-Subgoal Aggregator Choices:', ''])
        for choice, count in supervisor_summary['per_subgoal_aggregator_choices'].items():
            writer.writerow([f'  {choice}', count])
        writer.writerow([''])
        writer.writerow(['LLM Calls:', ''])
        writer.writerow(['  Avg per Query', f"{supervisor_summary['avg_llm_calls_per_query']:.2f}"])
        writer.writerow(['  Avg per Subgoal', f"{supervisor_summary['avg_llm_calls_per_subgoal']:.2f}"])
        writer.writerow([''])
        writer.writerow(['Final Aggregator Confidence:', ''])
        writer.writerow(['  Mean', f"{supervisor_summary['final_aggregator_confidence']['mean']:.4f}"])
        writer.writerow(['  Median', f"{supervisor_summary['final_aggregator_confidence']['median']:.4f}"])
        writer.writerow(['  Min', f"{supervisor_summary['final_aggregator_confidence']['min']:.4f}"])
        writer.writerow(['  Max', f"{supervisor_summary['final_aggregator_confidence']['max']:.4f}"])
    
    print(f"Supervisor statistics CSV saved to: {stats_csv}")
    
    # 5. Save human-readable report
    readable_path = output_folder / f"supervisor_readable_{timestamp}.txt"
    with open(readable_path, 'w', encoding='utf-8') as f:
        separator = "=" * 80
        title = "AGENTIC SUPERVISOR RAG DECISIONS - HUMAN READABLE"
        f.write(separator + "\n")
        f.write(title + "\n")
        f.write(separator + "\n")
        
        f.write("SUMMARY STATISTICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total queries analyzed: {supervisor_summary['total_queries']}\n")
        
        f.write("Subgoal Statistics:\n")
        f.write(f"  Total subgoals: {supervisor_summary['subgoal_stats']['total']}\n")
        f.write(f"  Average: {supervisor_summary['subgoal_stats']['mean']:.2f}\n")
        f.write(f"  Median: {supervisor_summary['subgoal_stats']['median']:.2f}\n")
        f.write(f"  Range: {supervisor_summary['subgoal_stats']['min']}-{supervisor_summary['subgoal_stats']['max']}\n")
        
        f.write("Subgoal Distribution:\n")
        for count, freq in sorted(supervisor_summary['subgoal_stats']['distribution'].items()):
            pct = (freq / supervisor_summary['total_queries'] * 100) if supervisor_summary['total_queries'] > 0 else 0
            f.write(f"  {count} subgoals: {freq} queries ({pct:.1f}%)\n")
        
        f.write("\nPer-Subgoal Aggregator Choices (across all subgoals):\n")
        total_choices = sum(supervisor_summary['per_subgoal_aggregator_choices'].values())
        for choice, count in supervisor_summary['per_subgoal_aggregator_choices'].most_common():
            pct = (count / total_choices * 100) if total_choices > 0 else 0
            f.write(f"  {choice}: {count} ({pct:.1f}%)\n")
        
        f.write(f"\nLLM Call Statistics:\n")
        f.write(f"  Average per query: {supervisor_summary['avg_llm_calls_per_query']:.2f}\n")
        f.write(f"  Average per subgoal: {supervisor_summary['avg_llm_calls_per_subgoal']:.2f}\n")
        
        f.write(f"\nFinal Aggregator Confidence:\n")
        f.write(f"  Mean: {supervisor_summary['final_aggregator_confidence']['mean']:.4f}\n")
        f.write(f"  Median: {supervisor_summary['final_aggregator_confidence']['median']:.4f}\n")
        f.write(f"  Range: {supervisor_summary['final_aggregator_confidence']['min']:.4f} - {supervisor_summary['final_aggregator_confidence']['max']:.4f}\n")
        
        f.write("\n" + separator + "\n")
        f.write("BY SUBGOAL COUNT\n")
        f.write("-" * 80 + "\n")
        
        for count in sorted(supervisor_summary['by_subgoal_count'].keys()):
            data = supervisor_summary['by_subgoal_count'][count]
            f.write(f"{count} SUBGOALS ({data['count']} queries)\n")
            f.write(f"  Avg LLM calls: {data['avg_llm_calls']:.2f}\n")
            f.write("-" * 40 + "\n")
            
            for i, q in enumerate(data['queries'][:10], 1):
                f.write(f"{i}. {q['query']}\n")
                f.write(f"   LLM: {q['llm_calls']} | Final Conf: {q['final_confidence']:.2f}\n")
            
            f.write("\n")
        
        f.write(separator + "\n")
        f.write("INDIVIDUAL QUERY DETAILS\n")
        f.write("-" * 80 + "\n")
        
        for i, r in enumerate(results, 1):
            if r.get('mode') == 'supervisor':
                f.write(f"[{i}] {r['log_file']}\n")
                f.write(f"Query: {r['query']}\n")
                f.write(f"Subgoals: {r['num_subgoals']} | LLM Calls: {r['total_llm_calls']}\n")
                f.write(f"Final Aggregator Confidence: {r.get('final_aggregator_confidence', 0.0):.2f}\n")
                
                if r.get('subgoal_aggregator_choices'):
                    f.write(f"Per-Subgoal Aggregator Choices: {', '.join(r['subgoal_aggregator_choices'])}\n")
                
                f.write("-" * 80 + "\n")
    
    print(f"Human-readable report saved to: {readable_path}")
    
    # 6. Create summary
    summary = create_summary(results, supervisor_summary)
    summary_path = output_folder / f"llm_calls_summary_{timestamp}.txt"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(summary)
    
    print(f"Summary saved to: {summary_path}")
    print(summary)


def create_summary(results: List[Dict], supervisor_summary: Dict) -> str:
    """Create a text summary of the analysis."""
    
    if not results:
        return "No results to summarize."
    
    total_files = len(results)
    
    # Filter by mode
    supervisor_results = [r for r in results if r['mode'] == 'supervisor']
    other_results = [r for r in results if r['mode'] != 'supervisor']
    
    # Overall stats
    total_llm_calls = sum(r['total_llm_calls'] for r in results)
    total_subgoal_gen = sum(r['subgoal_gen_calls'] for r in results)
    total_agent1 = sum(r['agent1_calls'] for r in results)
    total_agent1b = sum(r['agent1b_calls'] for r in results)
    total_agent2_naive = sum(r['agent2_naive_calls'] for r in results)
    total_agent2_graphrag = sum(r['agent2_graphrag_calls'] for r in results)
    total_per_sg_agg = sum(r['per_subgoal_aggregator_calls'] for r in results)
    total_final_agg = sum(r['final_aggregator_calls'] for r in results)
    total_embed_calls = sum(r['embed_calls'] for r in results)
    
    avg_llm_per_query = total_llm_calls / total_files if total_files > 0 else 0
    avg_embed_per_query = total_embed_calls / total_files if total_files > 0 else 0
    
    min_llm = min(r['total_llm_calls'] for r in results) if results else 0
    max_llm = max(r['total_llm_calls'] for r in results) if results else 0
    
    # Build summary sections using docstrings
    header = """
╔══════════════════════════════════════════════════════════════╗
║    AGENTIC SUPERVISOR RAG (SUBGOALS) LLM CALL ANALYSIS      ║
╚══════════════════════════════════════════════════════════════╝
"""
    
    overall_section = f"""
📊 Overall Statistics:
├─ Total log files analyzed: {total_files}
│  ├─ Supervisor mode: {len(supervisor_results)}
│  └─ Other modes: {len(other_results)}
├─ Total LLM calls: {total_llm_calls}
│  ├─ Subgoal Generator: {total_subgoal_gen}
│  ├─ Agent 1 (entity extraction): {total_agent1}
│  ├─ Agent 1b (triple extraction): {total_agent1b}
│  ├─ Agent 2 Naive: {total_agent2_naive}
│  ├─ Agent 2 GraphRAG: {total_agent2_graphrag}
│  ├─ Per-Subgoal Aggregator: {total_per_sg_agg}
│  └─ Final Aggregator: {total_final_agg}
└─ Total embedding calls: {total_embed_calls}
"""
    
    averages_section = f"""
📈 Per-Query Averages:
├─ Average LLM calls per query: {avg_llm_per_query:.2f}
├─ Average embedding calls per query: {avg_embed_per_query:.2f}
└─ Average subgoals per query: {supervisor_summary['subgoal_stats']['mean']:.2f}
"""
    
    ranges_section = f"""
📉 Range:
├─ Minimum LLM calls in a query: {min_llm}
├─ Maximum LLM calls in a query: {max_llm}
├─ Minimum subgoals: {supervisor_summary['subgoal_stats']['min']}
└─ Maximum subgoals: {supervisor_summary['subgoal_stats']['max']}
"""
    
    if supervisor_results and supervisor_summary['total_queries'] > 0:
        supervisor_section = f"""
🎯 SUBGOAL ANALYSIS ({supervisor_summary['total_queries']} queries):

📊 Subgoal Distribution:
"""
        for count, freq in sorted(supervisor_summary['subgoal_stats']['distribution'].items()):
            percentage = (freq / supervisor_summary['total_queries']) * 100
            bar = "█" * int(percentage / 2)
            supervisor_section += f"   ├─ {count} subgoals: {freq} queries ({percentage:.1f}%) {bar}\n"
        
        supervisor_section += f"""
📈 Subgoal Statistics:
├─ Total subgoals: {supervisor_summary['subgoal_stats']['total']}
├─ Mean: {supervisor_summary['subgoal_stats']['mean']:.2f}
├─ Median: {supervisor_summary['subgoal_stats']['median']:.2f}
└─ Range: {supervisor_summary['subgoal_stats']['min']}-{supervisor_summary['subgoal_stats']['max']}

🎯 Per-Subgoal Aggregator Choices (across all subgoals):
"""
        total_sg_choices = sum(supervisor_summary['per_subgoal_aggregator_choices'].values())
        for choice, count in supervisor_summary['per_subgoal_aggregator_choices'].most_common():
            percentage = (count / total_sg_choices * 100) if total_sg_choices > 0 else 0
            supervisor_section += f"   ├─ {choice}: {count} ({percentage:.1f}%)\n"
        
        supervisor_section += f"""
💻 LLM Call Statistics:
├─ Average per query: {supervisor_summary['avg_llm_calls_per_query']:.2f}
└─ Average per subgoal: {supervisor_summary['avg_llm_calls_per_subgoal']:.2f}

📊 Final Aggregator Confidence:
├─ Mean: {supervisor_summary['final_aggregator_confidence']['mean']:.4f}
├─ Median: {supervisor_summary['final_aggregator_confidence']['median']:.4f}
└─ Range: {supervisor_summary['final_aggregator_confidence']['min']:.4f} - {supervisor_summary['final_aggregator_confidence']['max']:.4f}
"""
    else:
        supervisor_section = ""
    
    pipeline_doc = """
💡 Pipeline Pattern (Agentic Supervisor):
   └─ For each query:
      1. Subgoal Generator: decompose into N subgoals (1 LLM call - JSON)
      
      2. For each subgoal (parallel or sequential):
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
         
         c) Per-Subgoal Aggregator: synthesize (1 LLM call)
         └─ Total per subgoal: 5 LLM calls
      
      3. Final Aggregator: synthesize across all subgoals (1 LLM call - JSON)
   
   └─ Total LLM calls = 1 + (N × 5) + 1 = 2 + 5N
   └─ Where N = number of subgoals (1 to SUBGOAL_MAX_N)
   └─ Agent 1/1b in Naive RAG not counted
"""
    
    agent_breakdown = f"""
🔍 Agent Call Breakdown (across all queries):
├─ Subgoal Generator (per query): {total_subgoal_gen}
├─ Per subgoal (both pipelines):
│  ├─ Agent 1 (GraphRAG only): {total_agent1}
│  ├─ Agent 1b (GraphRAG only): {total_agent1b}
│  ├─ Agent 2 Naive: {total_agent2_naive}
│  ├─ Agent 2 GraphRAG: {total_agent2_graphrag}
│  └─ Per-Subgoal Aggregator: {total_per_sg_agg}
├─ Final Aggregator (per query): {total_final_agg}
└─ Total LLM calls: {total_llm_calls}
"""
    
    subgoal_examples = ""
    if supervisor_summary.get('by_subgoal_count'):
        subgoal_examples = """
📌 EXAMPLES BY SUBGOAL COUNT:
"""
        for count in sorted(supervisor_summary['by_subgoal_count'].keys())[:3]:
            data = supervisor_summary['by_subgoal_count'][count]
            subgoal_examples += f"\n{count} Subgoals ({data['count']} queries):\n"
            subgoal_examples += f"  Average LLM calls: {data['avg_llm_calls']:.2f}\n"
            for q in data['queries'][:3]:
                query_preview = q['query'][:55] + "..." if len(q['query']) > 55 else q['query']
                subgoal_examples += f"  - {query_preview}\n"
                subgoal_examples += f"    LLM: {q['llm_calls']} | Final Conf: {q['final_confidence']:.2f}\n"
    
    comparison_doc = """
📊 COMPARISON TO OTHER METHODS:
└─ Agentic Supervisor characteristics:
   ├─ Query decomposition into independent subgoals
   ├─ Each subgoal runs full multi-pipeline (GraphRAG + Naive + Aggregator)
   ├─ Subgoals can run in parallel (configurable)
   ├─ Final aggregation synthesizes across all subgoal answers
   ├─ Variable cost: 2 + 5N LLM calls (N = subgoals)
   ├─ Typical: 7-22 calls (1-4 subgoals)
   └─ Best for: complex multi-faceted queries benefiting from decomposition
"""
    
    notes_doc = f"""
📝 Notes:
   - Pattern: 1 (subgoal gen) + N × 5 (per subgoal) + 1 (final agg)
   - Subgoal Generator decides decomposition (1 to {supervisor_summary['subgoal_stats']['max']} subgoals)
   - Each subgoal: Naive (1) + GraphRAG (3) + Aggregator (1) = 5 calls
   - Agent 1/1b called in Naive RAG but not counted
   - Subgoals can run in parallel or sequential
   - Final Aggregator provides unified answer with coverage analysis
   - Most flexible but potentially highest cost
   - Average subgoals: {supervisor_summary['subgoal_stats']['mean']:.2f}
   - Average LLM calls per subgoal: {supervisor_summary['avg_llm_calls_per_subgoal']:.2f}
"""
    
    summary = (
        header + overall_section + averages_section + ranges_section + 
        supervisor_section + pipeline_doc + agent_breakdown + 
        subgoal_examples + comparison_doc + notes_doc
    )
    
    return summary


def main():
    """Main execution function."""
    
    separator = "=" * 70
    title = "AGENTIC SUPERVISOR RAG (SUBGOALS) LLM CALL ANALYZER"
    subtitle = "(Tracking subgoal decomposition and per-subgoal pipelines)"
    
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