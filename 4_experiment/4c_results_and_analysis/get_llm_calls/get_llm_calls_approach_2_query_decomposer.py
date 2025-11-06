# analyze_llm_calls_decomposer.py
"""
Analyzes log files from the Query Decomposer + Aggregator RAG.
Tracks LLM calls per query including decomposition, multiple subqueries, and aggregation.
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
LOG_FOLDER = SCRIPT_DIR / "../../4b_retrieval/4b_iv_multi_agents/question_terminal_logs_multi_agent/approach_2_query_decomposer_1250"
OUTPUT_FOLDER = SCRIPT_DIR / "llm_calls_analysis_results_approach_2_query_decomposer_1250"


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


def extract_decomposer_decision(log_path: Path) -> Dict[str, any]:
    """Extract decomposer decision from the log."""
    decision = {
        'strategy': None,
        'confidence': 0.0,
        'rationale': '',
        'signals': [],
        'num_naive_subqs': 0,
        'num_graphrag_subqs': 0
    }
    
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
            # Extract strategy and counts
            strategy_match = re.search(r'Strategy=(\w+)\s*\|\s*naive_q=(\d+)\s*\|\s*graphrag_q=(\d+)', content)
            if strategy_match:
                decision['strategy'] = strategy_match.group(1)
                decision['num_naive_subqs'] = int(strategy_match.group(2))
                decision['num_graphrag_subqs'] = int(strategy_match.group(3))
            
            # Extract confidence
            dec_conf = re.search(r'\[Decomposer\].*?confidence=([\d.]+)', content)
            if dec_conf:
                decision['confidence'] = float(dec_conf.group(1))
            
            # Extract rationale from summary
            rat_match = re.search(r'- Decomposer strategy:\s*\w+\s*\(confidence=([\d.]+)\)', content)
            if rat_match:
                decision['confidence'] = float(rat_match.group(1))
    
    except Exception as e:
        print(f"Error extracting decomposer decision from {log_path.name}: {e}")
    
    return decision


def extract_aggregator_decision(log_path: Path) -> Dict[str, any]:
    """Extract aggregator decision from the log."""
    decision = {
        'chosen': None,
        'confidence': 0.0,
        'rationale': ''
    }
    
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
            # Extract aggregator decision
            agg_match = re.search(r'\[Aggregator\] Decision: chosen=(\w+)', content)
            if agg_match:
                decision['chosen'] = agg_match.group(1)
            
            # Extract confidence
            agg_conf = re.search(r'- Aggregator choice:\s*\w+\s*\(confidence=([\d.]+)\)', content)
            if agg_conf:
                decision['confidence'] = float(agg_conf.group(1))
    
    except Exception as e:
        print(f"Error extracting aggregator decision from {log_path.name}: {e}")
    
    return decision


def analyze_log_file(log_path: Path) -> Dict:
    """
    Analyze a single log file and count LLM calls.
    
    Returns dict with metrics, counting Decomposer + pipelines + Aggregator.
    Ignores Agent 1/1b in Naive RAG pipeline.
    """
    
    result = {
        'log_file': log_path.name,
        'query': '',
        'decomposer_calls': 0,
        'agent1_calls': 0,
        'agent1b_calls': 0,
        'agent2_naive_calls': 0,
        'agent2_graphrag_calls': 0,
        'aggregator_calls': 0,
        'total_llm_calls': 0,
        'embed_calls': 0,
        'chunks_retrieved': 0,
        'triples_retrieved': 0,
        'decomposer_strategy': None,
        'decomposer_confidence': 0.0,
        'num_naive_subqs': 0,
        'num_graphrag_subqs': 0,
        'aggregator_chosen': None,
        'aggregator_confidence': 0.0,
        'mode': 'unknown'
    }
    
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
            # Extract query
            result['query'] = extract_query_from_log(log_path)
            
            # Detect mode
            if "Multi-Agent RAG (Decomposer + Aggregator) run started" in content:
                result['mode'] = 'decomposer'
            
            # Extract decomposer decision
            dec_decision = extract_decomposer_decision(log_path)
            result['decomposer_strategy'] = dec_decision['strategy']
            result['decomposer_confidence'] = dec_decision['confidence']
            result['num_naive_subqs'] = dec_decision['num_naive_subqs']
            result['num_graphrag_subqs'] = dec_decision['num_graphrag_subqs']
            
            # Extract aggregator decision
            agg_decision = extract_aggregator_decision(log_path)
            result['aggregator_chosen'] = agg_decision['chosen']
            result['aggregator_confidence'] = agg_decision['confidence']
            
            # Count Decomposer prompts (should be 1)
            decomposer_matches = re.findall(r'\[Decomposer\] Prompt:', content)
            result['decomposer_calls'] = len(decomposer_matches)
            
            # Count Aggregator prompts (should be 1)
            aggregator_matches = re.findall(r'\[Aggregator\] Prompt:', content)
            result['aggregator_calls'] = len(aggregator_matches)
            
            # Better approach: Find all agent prompts with their positions
            # and determine context based on nearest section marker
            
            def find_nearest_section(pos: int) -> str:
                """Find which section a position belongs to by looking backwards."""
                before = content[:pos]
                # Find last occurrence of each section marker
                last_graphrag = before.rfind('=== GraphRAG')
                last_naive = before.rfind('=== Naive RAG')
                
                if last_graphrag == -1 and last_naive == -1:
                    return 'unknown'
                elif last_graphrag > last_naive:
                    return 'graphrag'
                elif last_naive > last_graphrag:
                    return 'naive'
                else:
                    return 'unknown'
            
            # Find all Agent 1 prompts and check their context
            agent1_count = 0
            for match in re.finditer(r'\[Agent 1\] Prompt:', content):
                section = find_nearest_section(match.start())
                if section == 'graphrag':
                    agent1_count += 1
            result['agent1_calls'] = agent1_count
            
            # Find all Agent 1b prompts and check their context
            agent1b_count = 0
            for match in re.finditer(r'\[Agent 1b\] Prompt:', content):
                section = find_nearest_section(match.start())
                if section == 'graphrag':
                    agent1b_count += 1
            result['agent1b_calls'] = agent1b_count
            
            # Find all Agent 2 prompts and check their context
            agent2_naive = 0
            agent2_graphrag = 0
            for match in re.finditer(r'\[Agent 2\] Prompt:', content):
                section = find_nearest_section(match.start())
                if section == 'naive':
                    agent2_naive += 1
                elif section == 'graphrag':
                    agent2_graphrag += 1
            
            result['agent2_naive_calls'] = agent2_naive
            result['agent2_graphrag_calls'] = agent2_graphrag
            
            # Total LLM calls
            result['total_llm_calls'] = (
                result['decomposer_calls'] +
                result['agent1_calls'] +
                result['agent1b_calls'] +
                result['agent2_naive_calls'] +
                result['agent2_graphrag_calls'] +
                result['aggregator_calls']
            )
            
            # Count embedding calls
            embed_matches = re.findall(r'\[Embed\]|Embedded', content)
            result['embed_calls'] = len(embed_matches)
            
            # Extract retrieval stats
            chunks = re.findall(r'Selected (\d+) chunks|returned (\d+) candidates', content)
            for match in chunks:
                count = int(match[0] or match[1] or 0)
                result['chunks_retrieved'] += count
            
            triples = re.findall(r'Selected (\d+) triples', content)
            for match in triples:
                result['triples_retrieved'] += int(match)
            
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


def create_decomposer_summary(results: List[Dict]) -> Dict[str, any]:
    """Create comprehensive decomposer decision summary statistics."""
    
    decomposer_results = [r for r in results if r['mode'] == 'decomposer']
    
    summary = {
        'total_queries': len(decomposer_results),
        'strategies': Counter(),
        'confidence_stats': {
            'mean': 0.0,
            'median': 0.0,
            'min': 0.0,
            'max': 0.0,
            'std_dev': 0.0
        },
        'subquery_stats': {
            'naive': {'total': 0, 'avg': 0.0, 'min': 0, 'max': 0},
            'graphrag': {'total': 0, 'avg': 0.0, 'min': 0, 'max': 0}
        },
        'by_strategy': {},
        'aggregator_choices': Counter()
    }
    
    if not decomposer_results:
        return summary
    
    # Count strategies and subqueries
    naive_counts = []
    graphrag_counts = []
    for r in decomposer_results:
        strategy = r.get('decomposer_strategy', 'unknown')
        summary['strategies'][strategy] += 1
        
        naive_count = r.get('num_naive_subqs', 0)
        graphrag_count = r.get('num_graphrag_subqs', 0)
        
        if naive_count > 0:
            naive_counts.append(naive_count)
        if graphrag_count > 0:
            graphrag_counts.append(graphrag_count)
        
        summary['subquery_stats']['naive']['total'] += naive_count
        summary['subquery_stats']['graphrag']['total'] += graphrag_count
        
        # Track aggregator choices
        agg_choice = r.get('aggregator_chosen', 'unknown')
        if agg_choice:
            summary['aggregator_choices'][agg_choice] += 1
        
        # By strategy
        if strategy not in summary['by_strategy']:
            summary['by_strategy'][strategy] = {
                'count': 0,
                'avg_confidence': 0.0,
                'avg_llm_calls': 0.0,
                'avg_naive_subqs': 0.0,
                'avg_graphrag_subqs': 0.0,
                'queries': []
            }
        
        summary['by_strategy'][strategy]['count'] += 1
        summary['by_strategy'][strategy]['queries'].append({
            'query': r['query'][:100],
            'confidence': r.get('decomposer_confidence', 0.0),
            'llm_calls': r.get('total_llm_calls', 0),
            'naive_subqs': naive_count,
            'graphrag_subqs': graphrag_count,
            'agg_chosen': agg_choice
        })
    
    # Subquery statistics
    if naive_counts:
        summary['subquery_stats']['naive']['avg'] = sum(naive_counts) / len(naive_counts)
        summary['subquery_stats']['naive']['min'] = min(naive_counts)
        summary['subquery_stats']['naive']['max'] = max(naive_counts)
    
    if graphrag_counts:
        summary['subquery_stats']['graphrag']['avg'] = sum(graphrag_counts) / len(graphrag_counts)
        summary['subquery_stats']['graphrag']['min'] = min(graphrag_counts)
        summary['subquery_stats']['graphrag']['max'] = max(graphrag_counts)
    
    # Confidence statistics
    confidences = [r.get('decomposer_confidence', 0.0) for r in decomposer_results if r.get('decomposer_confidence', 0) > 0]
    
    if confidences:
        import statistics
        summary['confidence_stats']['mean'] = statistics.mean(confidences)
        summary['confidence_stats']['median'] = statistics.median(confidences)
        summary['confidence_stats']['min'] = min(confidences)
        summary['confidence_stats']['max'] = max(confidences)
        if len(confidences) > 1:
            summary['confidence_stats']['std_dev'] = statistics.stdev(confidences)
    
    # Average stats by strategy
    for strategy, data in summary['by_strategy'].items():
        queries = data['queries']
        if queries:
            data['avg_confidence'] = sum(q['confidence'] for q in queries) / len(queries)
            data['avg_llm_calls'] = sum(q['llm_calls'] for q in queries) / len(queries)
            data['avg_naive_subqs'] = sum(q['naive_subqs'] for q in queries) / len(queries)
            data['avg_graphrag_subqs'] = sum(q['graphrag_subqs'] for q in queries) / len(queries)
    
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
        'total_llm_calls', 'decomposer_calls', 'agent1_calls', 'agent1b_calls',
        'agent2_naive_calls', 'agent2_graphrag_calls', 'aggregator_calls',
        'decomposer_strategy', 'decomposer_confidence',
        'num_naive_subqs', 'num_graphrag_subqs',
        'aggregator_chosen', 'aggregator_confidence',
        'embed_calls', 'chunks_retrieved', 'triples_retrieved'
    ]
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(results)
    
    print(f"Detailed results saved to: {csv_path}")
    
    # 2. Save full JSON
    json_path = output_folder / f"llm_calls_detailed_{timestamp}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"JSON results saved to: {json_path}")
    
    # 3. Create and save decomposer summary
    dec_summary = create_decomposer_summary(results)
    dec_summary_path = output_folder / f"decomposer_summary_{timestamp}.json"
    with open(dec_summary_path, 'w', encoding='utf-8') as f:
        # Convert Counter to dict for JSON serialization
        summary_copy = dec_summary.copy()
        summary_copy['strategies'] = dict(dec_summary['strategies'])
        summary_copy['aggregator_choices'] = dict(dec_summary['aggregator_choices'])
        json.dump(summary_copy, f, indent=2, ensure_ascii=False)
    
    print(f"Decomposer summary saved to: {dec_summary_path}")
    
    # 4. Save decomposer statistics CSV
    dec_stats_csv = output_folder / f"decomposer_statistics_{timestamp}.csv"
    with open(dec_stats_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Value'])
        writer.writerow(['Total Queries', dec_summary['total_queries']])
        writer.writerow([''])
        writer.writerow(['Strategy Distribution:', ''])
        for strategy, count in dec_summary['strategies'].items():
            writer.writerow([f'  {strategy}', count])
        writer.writerow([''])
        writer.writerow(['Naive Subqueries:', ''])
        writer.writerow(['  Total', dec_summary['subquery_stats']['naive']['total']])
        writer.writerow(['  Average', f"{dec_summary['subquery_stats']['naive']['avg']:.2f}"])
        writer.writerow(['  Min/Max', f"{dec_summary['subquery_stats']['naive']['min']}/{dec_summary['subquery_stats']['naive']['max']}"])
        writer.writerow([''])
        writer.writerow(['GraphRAG Subqueries:', ''])
        writer.writerow(['  Total', dec_summary['subquery_stats']['graphrag']['total']])
        writer.writerow(['  Average', f"{dec_summary['subquery_stats']['graphrag']['avg']:.2f}"])
        writer.writerow(['  Min/Max', f"{dec_summary['subquery_stats']['graphrag']['min']}/{dec_summary['subquery_stats']['graphrag']['max']}"])
        writer.writerow([''])
        writer.writerow(['Confidence Stats:', ''])
        writer.writerow(['  Mean', f"{dec_summary['confidence_stats']['mean']:.4f}"])
        writer.writerow(['  Median', f"{dec_summary['confidence_stats']['median']:.4f}"])
        writer.writerow(['  Min', f"{dec_summary['confidence_stats']['min']:.4f}"])
        writer.writerow(['  Max', f"{dec_summary['confidence_stats']['max']:.4f}"])
        writer.writerow(['  Std Dev', f"{dec_summary['confidence_stats']['std_dev']:.4f}"])
        writer.writerow([''])
        writer.writerow(['Aggregator Choices:', ''])
        for choice, count in dec_summary['aggregator_choices'].items():
            writer.writerow([f'  {choice}', count])
    
    print(f"Decomposer statistics CSV saved to: {dec_stats_csv}")
    
    # 5. Save human-readable report
    readable_path = output_folder / f"decomposer_readable_{timestamp}.txt"
    with open(readable_path, 'w', encoding='utf-8') as f:
        separator = "=" * 80
        title = "QUERY DECOMPOSER + AGGREGATOR DECISIONS - HUMAN READABLE"
        f.write(separator + "\n")
        f.write(title + "\n")
        f.write(separator + "\n")
        
        f.write("SUMMARY STATISTICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total queries analyzed: {dec_summary['total_queries']}\n")
        
        f.write("Decomposer Strategy Distribution:\n")
        for strategy, count in dec_summary['strategies'].items():
            pct = (count / dec_summary['total_queries'] * 100) if dec_summary['total_queries'] > 0 else 0
            f.write(f"  {strategy}: {count} ({pct:.1f}%)\n")
        
        f.write("\nSubquery Statistics:\n")
        f.write(f"  Naive RAG:\n")
        f.write(f"    Total: {dec_summary['subquery_stats']['naive']['total']}\n")
        f.write(f"    Average: {dec_summary['subquery_stats']['naive']['avg']:.2f}\n")
        f.write(f"    Range: {dec_summary['subquery_stats']['naive']['min']}-{dec_summary['subquery_stats']['naive']['max']}\n")
        f.write(f"  GraphRAG:\n")
        f.write(f"    Total: {dec_summary['subquery_stats']['graphrag']['total']}\n")
        f.write(f"    Average: {dec_summary['subquery_stats']['graphrag']['avg']:.2f}\n")
        f.write(f"    Range: {dec_summary['subquery_stats']['graphrag']['min']}-{dec_summary['subquery_stats']['graphrag']['max']}\n")
        
        f.write(f"\nDecomposer Confidence Statistics:\n")
        f.write(f"  Mean: {dec_summary['confidence_stats']['mean']:.4f}\n")
        f.write(f"  Median: {dec_summary['confidence_stats']['median']:.4f}\n")
        f.write(f"  Range: {dec_summary['confidence_stats']['min']:.4f} - {dec_summary['confidence_stats']['max']:.4f}\n")
        f.write(f"  Std Dev: {dec_summary['confidence_stats']['std_dev']:.4f}\n")
        
        f.write("\nAggregator Choice Distribution:\n")
        for choice, count in dec_summary['aggregator_choices'].items():
            pct = (count / dec_summary['total_queries'] * 100) if dec_summary['total_queries'] > 0 else 0
            f.write(f"  {choice}: {count} ({pct:.1f}%)\n")
        
        f.write("\n" + separator + "\n")
        f.write("DETAILED BY STRATEGY\n")
        f.write("-" * 80 + "\n")
        
        for strategy, data in dec_summary['by_strategy'].items():
            if data['count'] > 0:
                f.write(f"{strategy.upper()} ({data['count']} queries)\n")
                f.write(f"  Avg Confidence: {data['avg_confidence']:.4f}\n")
                f.write(f"  Avg LLM Calls: {data['avg_llm_calls']:.2f}\n")
                f.write(f"  Avg Naive Subqs: {data['avg_naive_subqs']:.2f}\n")
                f.write(f"  Avg GraphRAG Subqs: {data['avg_graphrag_subqs']:.2f}\n")
                f.write("-" * 40 + "\n")
                
                for i, q in enumerate(data['queries'][:10], 1):
                    f.write(f"{i}. {q['query']}\n")
                    f.write(f"   Conf: {q['confidence']:.2f} | LLM: {q['llm_calls']} | N:{q['naive_subqs']} G:{q['graphrag_subqs']} | Agg:{q['agg_chosen']}\n")
                
                f.write("\n")
        
        f.write(separator + "\n")
        f.write("INDIVIDUAL DECISIONS\n")
        f.write("-" * 80 + "\n")
        
        for i, r in enumerate(results, 1):
            if r.get('decomposer_strategy'):
                f.write(f"[{i}] {r['log_file']}\n")
                f.write(f"Query: {r['query']}\n")
                f.write(f"Strategy: {r['decomposer_strategy']} | Conf: {r['decomposer_confidence']:.2f}\n")
                f.write(f"Subqueries: Naive={r['num_naive_subqs']}, GraphRAG={r['num_graphrag_subqs']}\n")
                f.write(f"Aggregator: {r['aggregator_chosen']} | Conf: {r['aggregator_confidence']:.2f}\n")
                f.write(f"LLM Calls: {r['total_llm_calls']}\n")
                f.write("-" * 80 + "\n")
    
    print(f"Human-readable report saved to: {readable_path}")
    
    # 6. Create summary
    summary = create_summary(results, dec_summary)
    summary_path = output_folder / f"llm_calls_summary_{timestamp}.txt"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(summary)
    
    print(f"Summary saved to: {summary_path}")
    print(summary)


def create_summary(results: List[Dict], dec_summary: Dict) -> str:
    """Create a text summary of the analysis."""
    
    if not results:
        return "No results to summarize."
    
    total_files = len(results)
    
    # Filter by mode
    dec_results = [r for r in results if r['mode'] == 'decomposer']
    other_results = [r for r in results if r['mode'] != 'decomposer']
    
    # Overall stats
    total_llm_calls = sum(r['total_llm_calls'] for r in results)
    total_decomposer = sum(r['decomposer_calls'] for r in results)
    total_agent1 = sum(r['agent1_calls'] for r in results)
    total_agent1b = sum(r['agent1b_calls'] for r in results)
    total_agent2_naive = sum(r['agent2_naive_calls'] for r in results)
    total_agent2_graphrag = sum(r['agent2_graphrag_calls'] for r in results)
    total_aggregator = sum(r['aggregator_calls'] for r in results)
    total_embed_calls = sum(r['embed_calls'] for r in results)
    
    avg_llm_per_query = total_llm_calls / total_files if total_files > 0 else 0
    avg_embed_per_query = total_embed_calls / total_files if total_files > 0 else 0
    
    min_llm = min(r['total_llm_calls'] for r in results) if results else 0
    max_llm = max(r['total_llm_calls'] for r in results) if results else 0
    
    # Build summary sections using docstrings
    header = """
╔══════════════════════════════════════════════════════════════╗
║   QUERY DECOMPOSER + AGGREGATOR RAG LLM CALL ANALYSIS       ║
╚══════════════════════════════════════════════════════════════╝
"""
    
    overall_section = f"""
📊 Overall Statistics:
├─ Total log files analyzed: {total_files}
│  ├─ Decomposer mode: {len(dec_results)}
│  └─ Other modes: {len(other_results)}
├─ Total LLM calls: {total_llm_calls}
│  ├─ Decomposer: {total_decomposer}
│  ├─ Agent 1 (entity extraction): {total_agent1}
│  ├─ Agent 1b (triple extraction): {total_agent1b}
│  ├─ Agent 2 Naive: {total_agent2_naive}
│  ├─ Agent 2 GraphRAG: {total_agent2_graphrag}
│  └─ Aggregator: {total_aggregator}
└─ Total embedding calls: {total_embed_calls}
"""
    
    averages_section = f"""
📈 Per-Query Averages:
├─ Average LLM calls per query: {avg_llm_per_query:.2f}
└─ Average embedding calls per query: {avg_embed_per_query:.2f}
"""
    
    ranges_section = f"""
📉 Range:
├─ Minimum LLM calls in a query: {min_llm}
└─ Maximum LLM calls in a query: {max_llm}
"""
    
    if dec_results and dec_summary['total_queries'] > 0:
        dec_section = f"""
🎯 QUERY DECOMPOSER ANALYSIS ({dec_summary['total_queries']} queries):

📊 Strategy Distribution:
"""
        for strategy, count in dec_summary['strategies'].most_common():
            percentage = (count / dec_summary['total_queries']) * 100
            bar = "█" * int(percentage / 2)
            dec_section += f"   ├─ {strategy}: {count} ({percentage:.1f}%) {bar}\n"
        
        dec_section += f"""
📋 Subquery Statistics:
├─ Naive RAG:
│  ├─ Total subqueries: {dec_summary['subquery_stats']['naive']['total']}
│  ├─ Average per query: {dec_summary['subquery_stats']['naive']['avg']:.2f}
│  └─ Range: {dec_summary['subquery_stats']['naive']['min']}-{dec_summary['subquery_stats']['naive']['max']}
└─ GraphRAG:
   ├─ Total subqueries: {dec_summary['subquery_stats']['graphrag']['total']}
   ├─ Average per query: {dec_summary['subquery_stats']['graphrag']['avg']:.2f}
   └─ Range: {dec_summary['subquery_stats']['graphrag']['min']}-{dec_summary['subquery_stats']['graphrag']['max']}

📈 Decomposer Confidence:
├─ Mean: {dec_summary['confidence_stats']['mean']:.4f}
├─ Median: {dec_summary['confidence_stats']['median']:.4f}
├─ Range: {dec_summary['confidence_stats']['min']:.4f} - {dec_summary['confidence_stats']['max']:.4f}
└─ Std Dev: {dec_summary['confidence_stats']['std_dev']:.4f}

🎯 Aggregator Choices:
"""
        for choice, count in dec_summary['aggregator_choices'].most_common():
            percentage = (count / dec_summary['total_queries']) * 100
            dec_section += f"   ├─ {choice}: {count} ({percentage:.1f}%)\n"
    else:
        dec_section = ""
    
    pipeline_doc = """
💡 Pipeline Pattern (Query Decomposer + Aggregator):
   └─ For each query:
      1. Decomposer analyzes query (1 LLM call - JSON)
         - Chooses strategy (split, primary_*, both_full, single_*)
         - Generates subqueries for each pipeline
      
      2. Execute pipelines with subqueries:
         a) Naive RAG (per subquery):
            - Vector search
            - Agent 1/1b: called but ignored
            - Agent 2: answer (1 LLM call per subquery)
         
         b) GraphRAG (per subquery):
            - Agent 1: entity extraction (1 LLM call)
            - Agent 1b: triple extraction (1 LLM call)
            - Graph retrieval
            - Agent 2: answer (1 LLM call)
            - Total: 3 LLM calls per subquery
      
      3. Aggregator synthesizes final answer (1 LLM call - JSON)
         - Combines both pipeline outputs
         - Chooses: naive/graphrag/mixed
   
   └─ Total LLM calls = 1 (decomposer) + N (naive) + 3M (graphrag) + 1 (aggregator)
   └─ Where N = naive subqueries, M = graphrag subqueries
   └─ Agent 1/1b in Naive RAG not counted per user request
"""
    
    agent_breakdown = f"""
🔍 Agent Call Breakdown (across all queries):
├─ Decomposer (query decomposition): {total_decomposer}
├─ Naive RAG pipeline:
│  ├─ Agent 1/1b: present but ignored in count
│  └─ Agent 2 (per subquery): {total_agent2_naive}
├─ GraphRAG pipeline:
│  ├─ Agent 1 (entity extraction): {total_agent1}
│  ├─ Agent 1b (triple extraction): {total_agent1b}
│  └─ Agent 2 (per subquery): {total_agent2_graphrag}
├─ Aggregator (final synthesis): {total_aggregator}
└─ Total LLM calls: {total_llm_calls}
"""
    
    strategy_section = """
📌 TOP STRATEGIES BY USAGE:
"""
    if dec_summary['strategies']:
        for strategy, count in dec_summary['strategies'].most_common(5):
            strategy_data = dec_summary['by_strategy'].get(strategy, {})
            avg_llm = strategy_data.get('avg_llm_calls', 0.0)
            avg_naive = strategy_data.get('avg_naive_subqs', 0.0)
            avg_graphrag = strategy_data.get('avg_graphrag_subqs', 0.0)
            strategy_section += f"{strategy}: {count} queries\n"
            strategy_section += f"  Avg LLM calls: {avg_llm:.2f} | Naive subqs: {avg_naive:.2f} | GraphRAG subqs: {avg_graphrag:.2f}\n"
    
    comparison_doc = """
📊 COMPARISON TO OTHER METHODS:
└─ Query Decomposer + Aggregator characteristics:
   ├─ Decomposes complex queries into subqueries
   ├─ Flexible strategies: split, primary+support, both, single
   ├─ Multiple subqueries per pipeline possible
   ├─ Both pipelines run (unlike router)
   ├─ Aggregator synthesizes from all results
   ├─ Variable cost: 2 + N + 3M LLM calls
   ├─ N and M can vary widely by query complexity
   └─ Best for: complex multi-aspect queries benefiting from decomposition
"""
    
    notes_doc = f"""
📝 Notes:
   - Pattern: 1 (decomposer) + N (naive) + 3M (graphrag) + 1 (aggregator)
   - Decomposer strategies: split, primary_naive_with_support_from_graphrag,
     primary_graphrag_with_support_from_naive, both_full, single_naive, single_graphrag
   - Agent 1/1b called in Naive RAG "for logs only" but not counted
   - Subquery counts vary by query complexity and strategy
   - Aggregator chooses: naive, graphrag, or mixed
   - Total queries analyzed: {dec_summary['total_queries']}
   - Avg naive subqueries: {dec_summary['subquery_stats']['naive']['avg']:.2f}
   - Avg graphrag subqueries: {dec_summary['subquery_stats']['graphrag']['avg']:.2f}
"""
    
    summary = (
        header + overall_section + averages_section + ranges_section + 
        dec_section + pipeline_doc + agent_breakdown + strategy_section + 
        comparison_doc + notes_doc
    )
    
    return summary


def main():
    """Main execution function."""
    
    separator = "=" * 70
    title = "QUERY DECOMPOSER + AGGREGATOR LLM CALL ANALYZER"
    subtitle = "(Tracking decomposition, multiple subqueries, and aggregation)"
    
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