# analyze_llm_calls_iq.py
"""
Analyzes log files from the Sequential IQ Orchestrator RAG system.
Tracks LLM calls including IQ generation, query modification, per-IQ pipelines.
Counts IQs generated vs executed and aggregator decisions per IQ.
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
LOG_FOLDER = SCRIPT_DIR / "../../4b_retrieval/4b_iv_multi_agents/question_terminal_logs_multi_agent/approach_2_both_5_iq_5_hops_1250"
OUTPUT_FOLDER = SCRIPT_DIR / "llm_calls_analysis_results_approach_2_both_5_iq_5_hops_1250"


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


def extract_iq_info(log_path: Path) -> tuple:
    """
    Extract IQ count (generated and executed) and per-IQ details from the log.
    Returns: (num_generated, num_executed, list_of_iq_details)
    """
    num_generated = 0
    num_executed = 0
    iq_details = []
    
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
            # Extract IQs generated
            gen_match = re.search(r'Produced (\d+) IQ\(s\):', content)
            if gen_match:
                num_generated = int(gen_match.group(1))
            
            # If not found, try from summary
            if num_generated == 0:
                gen_match = re.search(r'- IQ steps generated:\s*(\d+)', content)
                if gen_match:
                    num_generated = int(gen_match.group(1))
            
            # Extract IQs executed
            exec_match = re.search(r'- IQ steps executed:\s*(\d+)', content)
            if exec_match:
                num_executed = int(exec_match.group(1))
            
            # Extract per-IQ aggregator decisions
            # Look for patterns like [iq=iq1] ... [Aggregator] Decision: chosen=
            iq_blocks = re.findall(
                r'\[iq=(iq\d+)\].*?\[Aggregator\] Decision: chosen=(\w+).*?confidence=([\d.]+)',
                content,
                re.DOTALL
            )
            
            for iq_id, chosen, conf in iq_blocks:
                iq_details.append({
                    'iq_id': iq_id,
                    'aggregator_chosen': chosen,
                    'aggregator_confidence': float(conf)
                })
    
    except Exception as e:
        print(f"Error extracting IQ info from {log_path.name}: {e}")
    
    return num_generated, num_executed, iq_details


def analyze_log_file(log_path: Path) -> Dict:
    """
    Analyze a single log file and count LLM calls.
    
    Returns dict with metrics, counting IQ Gen + Query Mod + per-IQ pipelines.
    Ignores Agent 1/1b in Naive RAG pipeline.
    """
    
    result = {
        'log_file': log_path.name,
        'query': '',
        'num_iqs_generated': 0,
        'num_iqs_executed': 0,
        'iq_gen_calls': 0,  # IQ Generator (1 JSON call)
        'query_modifier_calls': 0,  # Query Modifier (per IQ except first)
        'agent1_calls': 0,  # Agent 1 (only in GraphRAG, across all IQs)
        'agent1b_calls': 0,  # Agent 1b (only in GraphRAG, across all IQs)
        'agent2_naive_calls': 0,  # Agent 2 in Naive (across all IQs)
        'agent2_graphrag_calls': 0,  # Agent 2 in GraphRAG (across all IQs)
        'per_iq_aggregator_calls': 0,  # Per-IQ aggregator
        'total_llm_calls': 0,
        'embed_calls': 0,
        'per_iq_aggregator_choices': [],  # List of choices per IQ
        'final_iq_aggregator_chosen': None,
        'final_iq_aggregator_confidence': 0.0,
        'mode': 'unknown'
    }
    
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
            # Extract query
            result['query'] = extract_query_from_log(log_path)
            
            # Detect mode
            if "IQ Orchestrator run started" in content:
                result['mode'] = 'iq-orchestrator'
            
            # Extract IQ info
            num_gen, num_exec, iq_details = extract_iq_info(log_path)
            result['num_iqs_generated'] = num_gen
            result['num_iqs_executed'] = num_exec
            result['per_iq_aggregator_choices'] = [iq['aggregator_chosen'] for iq in iq_details]
            
            # Get final IQ aggregator decision (last one)
            if iq_details:
                last_iq = iq_details[-1]
                result['final_iq_aggregator_chosen'] = last_iq['aggregator_chosen']
                result['final_iq_aggregator_confidence'] = last_iq['aggregator_confidence']
            
            # Count IQ Generator prompts (should be 1)
            iq_gen_matches = re.findall(r'\[IQGenerator\] Prompt:', content)
            result['iq_gen_calls'] = len(iq_gen_matches)
            
            # Count Query Modifier prompts (one per IQ except potentially the first)
            query_mod_matches = re.findall(r'\[QueryModifier\] Prompt:', content)
            result['query_modifier_calls'] = len(query_mod_matches)
            
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
            
            # Count per-IQ Aggregator prompts
            # These are marked with [iq=...] context
            per_iq_agg = len(re.findall(r'\[iq=iq\d+\].*?\[Aggregator\] Prompt:', content, re.DOTALL))
            result['per_iq_aggregator_calls'] = per_iq_agg
            
            # Total LLM calls
            result['total_llm_calls'] = (
                result['iq_gen_calls'] +
                result['query_modifier_calls'] +
                result['agent1_calls'] +
                result['agent1b_calls'] +
                result['agent2_naive_calls'] +
                result['agent2_graphrag_calls'] +
                result['per_iq_aggregator_calls']
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


def create_iq_summary(results: List[Dict]) -> Dict[str, any]:
    """Create comprehensive IQ orchestrator summary statistics."""
    
    iq_results = [r for r in results if r['mode'] == 'iq-orchestrator']
    
    summary = {
        'total_queries': len(iq_results),
        'iq_generated_stats': {
            'total': 0,
            'mean': 0.0,
            'median': 0.0,
            'min': 0,
            'max': 0,
            'distribution': Counter()
        },
        'iq_executed_stats': {
            'total': 0,
            'mean': 0.0,
            'median': 0.0,
            'min': 0,
            'max': 0,
            'distribution': Counter()
        },
        'per_iq_aggregator_choices': Counter(),
        'avg_llm_calls_per_query': 0.0,
        'avg_llm_calls_per_iq': 0.0,
        'final_iq_aggregator_confidence': {
            'mean': 0.0,
            'median': 0.0,
            'min': 0.0,
            'max': 0.0
        },
        'by_executed_count': {}
    }
    
    if not iq_results:
        return summary
    
    # IQ generated statistics
    generated_counts = [r['num_iqs_generated'] for r in iq_results if r['num_iqs_generated'] > 0]
    
    if generated_counts:
        import statistics
        summary['iq_generated_stats']['total'] = sum(generated_counts)
        summary['iq_generated_stats']['mean'] = statistics.mean(generated_counts)
        summary['iq_generated_stats']['median'] = statistics.median(generated_counts)
        summary['iq_generated_stats']['min'] = min(generated_counts)
        summary['iq_generated_stats']['max'] = max(generated_counts)
        summary['iq_generated_stats']['distribution'] = Counter(generated_counts)
    
    # IQ executed statistics
    executed_counts = [r['num_iqs_executed'] for r in iq_results if r['num_iqs_executed'] > 0]
    
    if executed_counts:
        import statistics
        summary['iq_executed_stats']['total'] = sum(executed_counts)
        summary['iq_executed_stats']['mean'] = statistics.mean(executed_counts)
        summary['iq_executed_stats']['median'] = statistics.median(executed_counts)
        summary['iq_executed_stats']['min'] = min(executed_counts)
        summary['iq_executed_stats']['max'] = max(executed_counts)
        summary['iq_executed_stats']['distribution'] = Counter(executed_counts)
    
    # Per-IQ aggregator choices
    for r in iq_results:
        for choice in r.get('per_iq_aggregator_choices', []):
            if choice:
                summary['per_iq_aggregator_choices'][choice] += 1
    
    # Final IQ aggregator confidence stats
    confidences = [r.get('final_iq_aggregator_confidence', 0.0) 
                   for r in iq_results 
                   if r.get('final_iq_aggregator_confidence', 0) > 0]
    
    if confidences:
        import statistics
        summary['final_iq_aggregator_confidence']['mean'] = statistics.mean(confidences)
        summary['final_iq_aggregator_confidence']['median'] = statistics.median(confidences)
        summary['final_iq_aggregator_confidence']['min'] = min(confidences)
        summary['final_iq_aggregator_confidence']['max'] = max(confidences)
    
    # Average LLM calls
    if iq_results:
        total_llm = sum(r['total_llm_calls'] for r in iq_results)
        summary['avg_llm_calls_per_query'] = total_llm / len(iq_results)
        
        total_executed = sum(r['num_iqs_executed'] for r in iq_results if r['num_iqs_executed'] > 0)
        if total_executed > 0:
            # Subtract IQ gen call (1) from total
            iq_llm = total_llm - len(iq_results)  # 1 IQ gen per query
            summary['avg_llm_calls_per_iq'] = iq_llm / total_executed if total_executed > 0 else 0.0
    
    # Group by executed count
    for r in iq_results:
        count = r['num_iqs_executed']
        if count not in summary['by_executed_count']:
            summary['by_executed_count'][count] = {
                'count': 0,
                'avg_llm_calls': 0.0,
                'avg_generated': 0.0,
                'queries': []
            }
        
        summary['by_executed_count'][count]['count'] += 1
        summary['by_executed_count'][count]['queries'].append({
            'query': r['query'][:100],
            'generated': r['num_iqs_generated'],
            'executed': r['num_iqs_executed'],
            'llm_calls': r['total_llm_calls'],
            'final_confidence': r.get('final_iq_aggregator_confidence', 0.0)
        })
    
    # Calculate averages by executed count
    for count, data in summary['by_executed_count'].items():
        queries = data['queries']
        if queries:
            data['avg_llm_calls'] = sum(q['llm_calls'] for q in queries) / len(queries)
            data['avg_generated'] = sum(q['generated'] for q in queries) / len(queries)
    
    return summary


def save_results(results: List[Dict]):
    """Save analysis results to multiple output files."""
    
    output_folder = OUTPUT_FOLDER.resolve()
    output_folder.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Save main detailed results as CSV
    csv_path = output_folder / f"llm_calls_detailed_{timestamp}.csv"
    csv_fields = [
        'log_file', 'query', 'mode',
        'num_iqs_generated', 'num_iqs_executed',
        'total_llm_calls', 'iq_gen_calls', 'query_modifier_calls',
        'agent1_calls', 'agent1b_calls',
        'agent2_naive_calls', 'agent2_graphrag_calls',
        'per_iq_aggregator_calls',
        'final_iq_aggregator_chosen', 'final_iq_aggregator_confidence',
        'embed_calls'
    ]
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(results)
    
    print(f"Detailed results saved to: {csv_path}")
    
    # 2. Save full JSON (includes per-IQ choices)
    json_path = output_folder / f"llm_calls_detailed_{timestamp}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"JSON results saved to: {json_path}")
    
    # 3. Create and save IQ summary
    iq_summary = create_iq_summary(results)
    summary_path = output_folder / f"iq_summary_{timestamp}.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        # Convert Counter to dict
        summary_copy = iq_summary.copy()
        summary_copy['iq_generated_stats']['distribution'] = dict(iq_summary['iq_generated_stats']['distribution'])
        summary_copy['iq_executed_stats']['distribution'] = dict(iq_summary['iq_executed_stats']['distribution'])
        summary_copy['per_iq_aggregator_choices'] = dict(iq_summary['per_iq_aggregator_choices'])
        json.dump(summary_copy, f, indent=2, ensure_ascii=False)
    
    print(f"IQ summary saved to: {summary_path}")
    
    # 4. Save IQ statistics CSV
    stats_csv = output_folder / f"iq_statistics_{timestamp}.csv"
    with open(stats_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Value'])
        writer.writerow(['Total Queries', iq_summary['total_queries']])
        writer.writerow([''])
        writer.writerow(['IQs Generated:', ''])
        writer.writerow(['  Total', iq_summary['iq_generated_stats']['total']])
        writer.writerow(['  Mean', f"{iq_summary['iq_generated_stats']['mean']:.2f}"])
        writer.writerow(['  Median', f"{iq_summary['iq_generated_stats']['median']:.2f}"])
        writer.writerow(['  Min/Max', f"{iq_summary['iq_generated_stats']['min']}/{iq_summary['iq_generated_stats']['max']}"])
        writer.writerow([''])
        writer.writerow(['IQs Executed:', ''])
        writer.writerow(['  Total', iq_summary['iq_executed_stats']['total']])
        writer.writerow(['  Mean', f"{iq_summary['iq_executed_stats']['mean']:.2f}"])
        writer.writerow(['  Median', f"{iq_summary['iq_executed_stats']['median']:.2f}"])
        writer.writerow(['  Min/Max', f"{iq_summary['iq_executed_stats']['min']}/{iq_summary['iq_executed_stats']['max']}"])
        writer.writerow([''])
        writer.writerow(['Generated Distribution:', ''])
        for count, freq in sorted(iq_summary['iq_generated_stats']['distribution'].items()):
            writer.writerow([f'  {count} IQs', freq])
        writer.writerow([''])
        writer.writerow(['Executed Distribution:', ''])
        for count, freq in sorted(iq_summary['iq_executed_stats']['distribution'].items()):
            writer.writerow([f'  {count} IQs', freq])
        writer.writerow([''])
        writer.writerow(['Per-IQ Aggregator Choices:', ''])
        for choice, count in iq_summary['per_iq_aggregator_choices'].items():
            writer.writerow([f'  {choice}', count])
        writer.writerow([''])
        writer.writerow(['LLM Calls:', ''])
        writer.writerow(['  Avg per Query', f"{iq_summary['avg_llm_calls_per_query']:.2f}"])
        writer.writerow(['  Avg per IQ', f"{iq_summary['avg_llm_calls_per_iq']:.2f}"])
        writer.writerow([''])
        writer.writerow(['Final IQ Aggregator Confidence:', ''])
        writer.writerow(['  Mean', f"{iq_summary['final_iq_aggregator_confidence']['mean']:.4f}"])
        writer.writerow(['  Median', f"{iq_summary['final_iq_aggregator_confidence']['median']:.4f}"])
        writer.writerow(['  Min', f"{iq_summary['final_iq_aggregator_confidence']['min']:.4f}"])
        writer.writerow(['  Max', f"{iq_summary['final_iq_aggregator_confidence']['max']:.4f}"])
    
    print(f"IQ statistics CSV saved to: {stats_csv}")
    
    # 5. Save human-readable report
    readable_path = output_folder / f"iq_readable_{timestamp}.txt"
    with open(readable_path, 'w', encoding='utf-8') as f:
        separator = "=" * 80
        title = "SEQUENTIAL IQ ORCHESTRATOR DECISIONS - HUMAN READABLE"
        f.write(separator + "\n")
        f.write(title + "\n")
        f.write(separator + "\n")
        
        f.write("SUMMARY STATISTICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total queries analyzed: {iq_summary['total_queries']}\n")
        
        f.write("IQs Generated:\n")
        f.write(f"  Total: {iq_summary['iq_generated_stats']['total']}\n")
        f.write(f"  Average: {iq_summary['iq_generated_stats']['mean']:.2f}\n")
        f.write(f"  Median: {iq_summary['iq_generated_stats']['median']:.2f}\n")
        f.write(f"  Range: {iq_summary['iq_generated_stats']['min']}-{iq_summary['iq_generated_stats']['max']}\n")
        
        f.write("Generated Distribution:\n")
        for count, freq in sorted(iq_summary['iq_generated_stats']['distribution'].items()):
            pct = (freq / iq_summary['total_queries'] * 100) if iq_summary['total_queries'] > 0 else 0
            f.write(f"  {count} IQs: {freq} queries ({pct:.1f}%)\n")
        
        f.write("\nIQs Executed:\n")
        f.write(f"  Total: {iq_summary['iq_executed_stats']['total']}\n")
        f.write(f"  Average: {iq_summary['iq_executed_stats']['mean']:.2f}\n")
        f.write(f"  Median: {iq_summary['iq_executed_stats']['median']:.2f}\n")
        f.write(f"  Range: {iq_summary['iq_executed_stats']['min']}-{iq_summary['iq_executed_stats']['max']}\n")
        
        f.write("Executed Distribution:\n")
        for count, freq in sorted(iq_summary['iq_executed_stats']['distribution'].items()):
            pct = (freq / iq_summary['total_queries'] * 100) if iq_summary['total_queries'] > 0 else 0
            f.write(f"  {count} IQs: {freq} queries ({pct:.1f}%)\n")
        
        f.write("\nPer-IQ Aggregator Choices (across all IQs):\n")
        total_choices = sum(iq_summary['per_iq_aggregator_choices'].values())
        for choice, count in iq_summary['per_iq_aggregator_choices'].most_common():
            pct = (count / total_choices * 100) if total_choices > 0 else 0
            f.write(f"  {choice}: {count} ({pct:.1f}%)\n")
        
        f.write(f"\nLLM Call Statistics:\n")
        f.write(f"  Average per query: {iq_summary['avg_llm_calls_per_query']:.2f}\n")
        f.write(f"  Average per IQ: {iq_summary['avg_llm_calls_per_iq']:.2f}\n")
        
        f.write(f"\nFinal IQ Aggregator Confidence:\n")
        f.write(f"  Mean: {iq_summary['final_iq_aggregator_confidence']['mean']:.4f}\n")
        f.write(f"  Median: {iq_summary['final_iq_aggregator_confidence']['median']:.4f}\n")
        f.write(f"  Range: {iq_summary['final_iq_aggregator_confidence']['min']:.4f} - {iq_summary['final_iq_aggregator_confidence']['max']:.4f}\n")
        
        f.write("\n" + separator + "\n")
        f.write("BY EXECUTED IQ COUNT\n")
        f.write("-" * 80 + "\n")
        
        for count in sorted(iq_summary['by_executed_count'].keys()):
            data = iq_summary['by_executed_count'][count]
            f.write(f"{count} IQs EXECUTED ({data['count']} queries)\n")
            f.write(f"  Avg LLM calls: {data['avg_llm_calls']:.2f}\n")
            f.write(f"  Avg generated: {data['avg_generated']:.2f}\n")
            f.write("-" * 40 + "\n")
            
            for i, q in enumerate(data['queries'][:10], 1):
                f.write(f"{i}. {q['query']}\n")
                f.write(f"   Gen: {q['generated']} | Exec: {q['executed']} | LLM: {q['llm_calls']} | Final Conf: {q['final_confidence']:.2f}\n")
            
            f.write("\n")
        
        f.write(separator + "\n")
        f.write("INDIVIDUAL QUERY DETAILS\n")
        f.write("-" * 80 + "\n")
        
        for i, r in enumerate(results, 1):
            if r.get('mode') == 'iq-orchestrator':
                f.write(f"[{i}] {r['log_file']}\n")
                f.write(f"Query: {r['query']}\n")
                f.write(f"Generated: {r['num_iqs_generated']} | Executed: {r['num_iqs_executed']}\n")
                f.write(f"LLM Calls: {r['total_llm_calls']}\n")
                f.write(f"Final IQ Aggregator: {r.get('final_iq_aggregator_chosen')} (conf={r.get('final_iq_aggregator_confidence', 0.0):.2f})\n")
                
                if r.get('per_iq_aggregator_choices'):
                    f.write(f"Per-IQ Aggregator Choices: {', '.join(r['per_iq_aggregator_choices'])}\n")
                
                f.write("-" * 80 + "\n")
    
    print(f"Human-readable report saved to: {readable_path}")
    
    # 6. Create summary
    summary = create_summary(results, iq_summary)
    summary_path = output_folder / f"llm_calls_summary_{timestamp}.txt"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(summary)
    
    print(f"Summary saved to: {summary_path}")
    print(summary)


def create_summary(results: List[Dict], iq_summary: Dict) -> str:
    """Create a text summary of the analysis."""
    
    if not results:
        return "No results to summarize."
    
    total_files = len(results)
    
    # Filter by mode
    iq_results = [r for r in results if r['mode'] == 'iq-orchestrator']
    other_results = [r for r in results if r['mode'] != 'iq-orchestrator']
    
    # Overall stats
    total_llm_calls = sum(r['total_llm_calls'] for r in results)
    total_iq_gen = sum(r['iq_gen_calls'] for r in results)
    total_query_mod = sum(r['query_modifier_calls'] for r in results)
    total_agent1 = sum(r['agent1_calls'] for r in results)
    total_agent1b = sum(r['agent1b_calls'] for r in results)
    total_agent2_naive = sum(r['agent2_naive_calls'] for r in results)
    total_agent2_graphrag = sum(r['agent2_graphrag_calls'] for r in results)
    total_per_iq_agg = sum(r['per_iq_aggregator_calls'] for r in results)
    total_embed_calls = sum(r['embed_calls'] for r in results)
    
    avg_llm_per_query = total_llm_calls / total_files if total_files > 0 else 0
    avg_embed_per_query = total_embed_calls / total_files if total_files > 0 else 0
    
    min_llm = min(r['total_llm_calls'] for r in results) if results else 0
    max_llm = max(r['total_llm_calls'] for r in results) if results else 0
    
    # Build summary sections using docstrings
    header = """
╔══════════════════════════════════════════════════════════════╗
║    SEQUENTIAL IQ ORCHESTRATOR RAG LLM CALL ANALYSIS         ║
╚══════════════════════════════════════════════════════════════╝
"""
    
    overall_section = f"""
📊 Overall Statistics:
├─ Total log files analyzed: {total_files}
│  ├─ IQ-orchestrator mode: {len(iq_results)}
│  └─ Other modes: {len(other_results)}
├─ Total LLM calls: {total_llm_calls}
│  ├─ IQ Generator: {total_iq_gen}
│  ├─ Query Modifier: {total_query_mod}
│  ├─ Agent 1 (entity extraction): {total_agent1}
│  ├─ Agent 1b (triple extraction): {total_agent1b}
│  ├─ Agent 2 Naive: {total_agent2_naive}
│  ├─ Agent 2 GraphRAG: {total_agent2_graphrag}
│  └─ Per-IQ Aggregator: {total_per_iq_agg}
└─ Total embedding calls: {total_embed_calls}
"""
    
    averages_section = f"""
📈 Per-Query Averages:
├─ Average LLM calls per query: {avg_llm_per_query:.2f}
├─ Average embedding calls per query: {avg_embed_per_query:.2f}
├─ Average IQs generated: {iq_summary['iq_generated_stats']['mean']:.2f}
└─ Average IQs executed: {iq_summary['iq_executed_stats']['mean']:.2f}
"""
    
    ranges_section = f"""
📉 Range:
├─ Minimum LLM calls in a query: {min_llm}
├─ Maximum LLM calls in a query: {max_llm}
├─ IQs generated range: {iq_summary['iq_generated_stats']['min']}-{iq_summary['iq_generated_stats']['max']}
└─ IQs executed range: {iq_summary['iq_executed_stats']['min']}-{iq_summary['iq_executed_stats']['max']}
"""
    
    if iq_results and iq_summary['total_queries'] > 0:
        iq_section = f"""
🔗 IQ ANALYSIS ({iq_summary['total_queries']} queries):

📊 IQs Generated Distribution:
"""
        for count, freq in sorted(iq_summary['iq_generated_stats']['distribution'].items()):
            percentage = (freq / iq_summary['total_queries']) * 100
            bar = "█" * int(percentage / 2)
            iq_section += f"   ├─ {count} IQs: {freq} queries ({percentage:.1f}%) {bar}\n"
        
        iq_section += """
📊 IQs Executed Distribution:
"""
        for count, freq in sorted(iq_summary['iq_executed_stats']['distribution'].items()):
            percentage = (freq / iq_summary['total_queries']) * 100
            bar = "█" * int(percentage / 2)
            iq_section += f"   ├─ {count} IQs: {freq} queries ({percentage:.1f}%) {bar}\n"
        
        iq_section += f"""
📈 IQ Statistics:
├─ Generated:
│  ├─ Total: {iq_summary['iq_generated_stats']['total']}
│  ├─ Mean: {iq_summary['iq_generated_stats']['mean']:.2f}
│  └─ Median: {iq_summary['iq_generated_stats']['median']:.2f}
└─ Executed:
   ├─ Total: {iq_summary['iq_executed_stats']['total']}
   ├─ Mean: {iq_summary['iq_executed_stats']['mean']:.2f}
   └─ Median: {iq_summary['iq_executed_stats']['median']:.2f}

🎯 Per-IQ Aggregator Choices (across all IQs):
"""
        total_iq_choices = sum(iq_summary['per_iq_aggregator_choices'].values())
        for choice, count in iq_summary['per_iq_aggregator_choices'].most_common():
            percentage = (count / total_iq_choices * 100) if total_iq_choices > 0 else 0
            iq_section += f"   ├─ {choice}: {count} ({percentage:.1f}%)\n"
        
        iq_section += f"""
💻 LLM Call Statistics:
├─ Average per query: {iq_summary['avg_llm_calls_per_query']:.2f}
└─ Average per IQ: {iq_summary['avg_llm_calls_per_iq']:.2f}

📊 Final IQ Aggregator Confidence:
├─ Mean: {iq_summary['final_iq_aggregator_confidence']['mean']:.4f}
├─ Median: {iq_summary['final_iq_aggregator_confidence']['median']:.4f}
└─ Range: {iq_summary['final_iq_aggregator_confidence']['min']:.4f} - {iq_summary['final_iq_aggregator_confidence']['max']:.4f}
"""
    else:
        iq_section = ""
    
    pipeline_doc = """
💡 Pipeline Pattern (Sequential IQ Orchestrator):
   └─ For each query:
      1. IQ Generator: decompose into N sequential IQs (1 LLM call - JSON)
      
      2. For each IQ (sequential, dependent):
         a) Query Modifier: enrich/complete IQ using prior answers (1 LLM call - JSON)
         
         b) Run multi-pipeline for this IQ:
            - Naive RAG:
              * Embed query
              * Vector search
              * Agent 1/1b: called but ignored
              * Agent 2: answer (1 LLM call)
            
            - GraphRAG:
              * Agent 1: entity extraction (1 LLM call)
              * Agent 1b: triple extraction (1 LLM call)
              * Graph retrieval
              * Agent 2: answer (1 LLM call)
            
            - Per-IQ Aggregator: synthesize (1 LLM call)
         
         └─ Total per IQ: 1 (modifier) + 1 (naive) + 3 (graphrag) + 1 (agg) = 6 LLM calls
      
      3. Final answer = answer from last executed IQ
   
   └─ Total LLM calls = 1 (IQ gen) + (N × 6) = 1 + 6N
   └─ Where N = number of IQs executed (≤ IQ_MAX_N)
   └─ Agent 1/1b in Naive RAG not counted
"""
    
    agent_breakdown = f"""
🔍 Agent Call Breakdown (across all queries):
├─ IQ Generator (per query): {total_iq_gen}
├─ Query Modifier (per IQ): {total_query_mod}
├─ Per IQ (both pipelines):
│  ├─ Agent 1 (GraphRAG only): {total_agent1}
│  ├─ Agent 1b (GraphRAG only): {total_agent1b}
│  ├─ Agent 2 Naive: {total_agent2_naive}
│  ├─ Agent 2 GraphRAG: {total_agent2_graphrag}
│  └─ Per-IQ Aggregator: {total_per_iq_agg}
└─ Total LLM calls: {total_llm_calls}
"""
    
    iq_examples = ""
    if iq_summary.get('by_executed_count'):
        iq_examples = """
📌 EXAMPLES BY EXECUTED IQ COUNT:
"""
        for count in sorted(iq_summary['by_executed_count'].keys())[:3]:
            data = iq_summary['by_executed_count'][count]
            iq_examples += f"\n{count} IQs Executed ({data['count']} queries):\n"
            iq_examples += f"  Average generated: {data['avg_generated']:.2f}\n"
            iq_examples += f"  Average LLM calls: {data['avg_llm_calls']:.2f}\n"
            for q in data['queries'][:3]:
                query_preview = q['query'][:55] + "..." if len(q['query']) > 55 else q['query']
                iq_examples += f"  - {query_preview}\n"
                iq_examples += f"    Gen: {q['generated']} | Exec: {q['executed']} | LLM: {q['llm_calls']} | Final Conf: {q['final_confidence']:.2f}\n"
    
    comparison_doc = """
📊 COMPARISON TO OTHER METHODS:
└─ Sequential IQ Orchestrator characteristics:
   ├─ Query decomposition into dependent sequential steps
   ├─ Each IQ depends on previous IQ answers
   ├─ Query Modifier enriches each next IQ with prior context
   ├─ Each IQ runs full multi-pipeline (GraphRAG + Naive + Aggregator)
   ├─ Sequential execution (not parallel)
   ├─ Final answer = answer from last IQ
   ├─ Variable cost: 1 + 6N LLM calls (N = IQs executed)
   ├─ Typical: 7-31 calls (1-5 IQs)
   └─ Best for: complex multi-step reasoning requiring sequential dependencies
"""
    
    notes_doc = f"""
📝 Notes:
   - Pattern: 1 (IQ gen) + N × 6 (per IQ) = 1 + 6N
   - IQ Generator creates up to {iq_summary['iq_generated_stats']['max']} sequential IQs
   - Query Modifier enriches each IQ with prior answers (1 call per IQ)
   - Each IQ: Modifier (1) + Naive (1) + GraphRAG (3) + Aggregator (1) = 6 calls
   - Agent 1/1b called in Naive RAG but not counted
   - IQs are dependent (sequential execution)
   - Early stop possible if Query Modifier signals redundancy
   - Average generated: {iq_summary['iq_generated_stats']['mean']:.2f}
   - Average executed: {iq_summary['iq_executed_stats']['mean']:.2f}
   - Average LLM calls per IQ: {iq_summary['avg_llm_calls_per_iq']:.2f}
"""
    
    summary = (
        header + overall_section + averages_section + ranges_section + 
        iq_section + pipeline_doc + agent_breakdown + 
        iq_examples + comparison_doc + notes_doc
    )
    
    return summary


def main():
    """Main execution function."""
    
    separator = "=" * 70
    title = "SEQUENTIAL IQ ORCHESTRATOR RAG LLM CALL ANALYZER"
    subtitle = "(Tracking IQ generation, execution, and sequential refinement)"
    
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