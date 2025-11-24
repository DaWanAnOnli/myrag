# analyze_llm_calls_router.py
"""
Analyzes log files from the Router-based Multi-Agent RAG.
Tracks LLM calls per query including router decision and pipeline execution.
Router chooses ONE pipeline (GraphRAG OR Naive RAG), not both.
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
LOG_FOLDER = SCRIPT_DIR / "../../4b_retrieval/4b_iv_multi_agents/question_terminal_logs_multi_agent/no_15_approach_2_router_new"
OUTPUT_FOLDER = SCRIPT_DIR / "llm_calls_analysis_results_approach_2_router"


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


def extract_router_decision(log_path: Path) -> Dict[str, any]:
    """Extract router decision from the log."""
    decision = {
        'decision': None,  # Changed from 'chosen' to 'decision'
        'confidence': 0.0,
        'rationale': '',
        'signals': [],
        'fallback': False
    }
    
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
            # Extract decision - now looks for "Decision=graphrag" or "Decision=naiverag"
            decision_match = re.search(r'\[Router\] Decision=(\w+)', content)
            if decision_match:
                decision['decision'] = decision_match.group(1)
            
            # Also check for "Router choice:" format in summary
            if not decision['decision']:
                choice_match = re.search(r'Router choice:\s*(\w+)', content)
                if choice_match:
                    decision['decision'] = choice_match.group(1)
            
            # Extract rationale (may span multiple lines)
            rat_match = re.search(r'rationale[=:]\s*(.+?)(?:\n\[|\n-|\n===|$)', content, re.DOTALL | re.IGNORECASE)
            if rat_match:
                decision['rationale'] = rat_match.group(1).strip()
            
            # Check for fallback indicator
            if 'fallback heuristic' in content.lower() or 'Fallback' in content:
                decision['fallback'] = True
    
    except Exception as e:
        print(f"Error extracting router decision from {log_path.name}: {e}")
    
    return decision


def detect_pipeline_executed(log_path: Path) -> str:
    """Detect which pipeline was actually executed."""
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
            # Check for pipeline execution markers
            if "=== GraphRAG (single-pass) started ===" in content:
                return "graphrag"
            elif "=== Naive RAG (Answerer-only) started ===" in content:
                return "naiverag"  # Changed from "naive" to "naiverag"
            
            # Check for iteration markers from Script 1 style
            if "--- GraphRAG Iteration" in content or "[G" in content:
                return "graphrag"
            elif "--- NaiveRAG Iteration" in content or "[N" in content:
                return "naiverag"
            
            # Fallback: check for completion markers
            if "[G.Step" in content or "GraphRAG finished" in content:
                return "graphrag"
            elif "[N.Step" in content or "Naive RAG finished" in content:
                return "naiverag"
    
    except Exception as e:
        print(f"Error detecting pipeline from {log_path.name}: {e}")
    
    return "unknown"


def analyze_log_file(log_path: Path) -> Dict:
    """
    Analyze a single log file and count LLM calls.
    
    Returns dict with metrics, counting Router + executed pipeline's LLM calls.
    Ignores Agent 1/1b in Naive RAG pipeline.
    """
    
    result = {
        'log_file': log_path.name,
        'query': '',
        'router_calls': 0,  # Router decision (1 JSON call)
        'agent1_calls': 0,  # Agent 1 (only counted in GraphRAG)
        'agent1b_calls': 0,  # Agent 1b (only counted in GraphRAG)
        'agent2_calls': 0,  # Agent 2 answerer
        'total_llm_calls': 0,  # router + pipeline calls
        'embed_calls': 0,
        'chunks_retrieved': 0,
        'triples_retrieved': 0,
        'pipeline_executed': None,  # 'graphrag' or 'naiverag'
        'router_decision': None,  # Changed from 'router_chosen'
        'router_confidence': 0.0,
        'router_rationale': '',
        'router_signals': [],
        'router_fallback': False,
        'mode': 'unknown'
    }
    
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
            # Extract query
            result['query'] = extract_query_from_log(log_path)
            
            # Detect mode
            if "Multi-Agent RAG (Router) run started" in content:
                result['mode'] = 'router'
            
            # Detect which pipeline executed
            result['pipeline_executed'] = detect_pipeline_executed(log_path)
            
            # Extract router decision
            router_dec = extract_router_decision(log_path)
            result['router_decision'] = router_dec['decision']
            result['router_confidence'] = router_dec['confidence']
            result['router_rationale'] = router_dec['rationale']
            result['router_signals'] = router_dec['signals']
            result['router_fallback'] = router_dec['fallback']
            
            # Count Router prompts (should be 1)
            router_matches = re.findall(r'\[Router\] Prompt:', content)
            result['router_calls'] = len(router_matches)
            
            # Count Agent 1 and 1b prompts
            agent1_matches = re.findall(r'\[Agent 1\] Prompt:', content)
            agent1b_matches = re.findall(r'\[Agent 1b\] Prompt:', content)
            
            # Only count Agent 1/1b if GraphRAG was executed
            if result['pipeline_executed'] == 'graphrag':
                result['agent1_calls'] = len(agent1_matches)
                result['agent1b_calls'] = len(agent1b_matches)
            # If Naive RAG, ignore Agent 1/1b calls per user request
            
            # Count Agent 2 prompts
            agent2_matches = re.findall(r'\[Agent 2\] Prompt:', content)
            result['agent2_calls'] = len(agent2_matches)
            
            # Total LLM calls
            result['total_llm_calls'] = (
                result['router_calls'] +
                result['agent1_calls'] +
                result['agent1b_calls'] +
                result['agent2_calls']
            )
            
            # Count embedding calls
            embed_matches = re.findall(r'\[Embed\]|Embedded|embed_text', content)
            result['embed_calls'] = len(embed_matches)
            
            # Extract retrieval stats based on pipeline
            if result['pipeline_executed'] == 'graphrag':
                chunks = re.search(r'Selected (\d+) chunks|chunks_selected=(\d+)', content)
                if chunks:
                    result['chunks_retrieved'] = int(chunks.group(1) or chunks.group(2))
                
                triples = re.search(r'Selected (\d+) triples|triples_selected=(\d+)', content)
                if triples:
                    result['triples_retrieved'] = int(triples.group(1) or triples.group(2))
            
            elif result['pipeline_executed'] == 'naiverag':
                chunks = re.search(r'Vector search returned (\d+) candidates|num_candidates[=:](\d+)', content)
                if chunks:
                    result['chunks_retrieved'] = int(chunks.group(1) or chunks.group(2) if chunks.lastindex >= 2 else chunks.group(1))
            
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


def create_router_summary(results: List[Dict]) -> Dict[str, any]:
    """Create comprehensive router decision summary statistics."""
    
    router_results = [r for r in results if r['mode'] == 'router']
    
    summary = {
        'total_queries': len(router_results),
        'decisions': {
            'graphrag': 0,
            'naiverag': 0,
            'graphrag_fallback': 0,
            'naiverag_fallback': 0,
            'unknown': 0
        },
        'pipeline_executed': {
            'graphrag': 0,
            'naiverag': 0,
            'unknown': 0
        },
        'fallback_count': 0,
        'confidence_stats': {
            'mean': 0.0,
            'median': 0.0,
            'min': 0.0,
            'max': 0.0,
            'std_dev': 0.0
        },
        'by_decision': {
            'graphrag': {'count': 0, 'avg_confidence': 0.0, 'avg_llm_calls': 0.0, 'queries': []},
            'naiverag': {'count': 0, 'avg_confidence': 0.0, 'avg_llm_calls': 0.0, 'queries': []}
        },
        'common_signals': Counter()
    }
    
    if not router_results:
        return summary
    
    # Count decisions
    for r in router_results:
        choice = r.get('router_decision', 'unknown')
        if choice in summary['decisions']:
            summary['decisions'][choice] += 1
        else:
            summary['decisions']['unknown'] += 1
        
        # Count fallbacks
        if r.get('router_fallback', False):
            summary['fallback_count'] += 1
        
        # Count actual pipeline executed
        pipeline = r.get('pipeline_executed', 'unknown')
        if pipeline in summary['pipeline_executed']:
            summary['pipeline_executed'][pipeline] += 1
        
        # Collect signals
        for sig in r.get('router_signals', []):
            if sig:
                summary['common_signals'][sig] += 1
        
        # Collect by decision type (normalize to base choice)
        base_choice = choice.replace('_fallback', '')
        if base_choice in summary['by_decision']:
            summary['by_decision'][base_choice]['count'] += 1
            summary['by_decision'][base_choice]['queries'].append({
                'query': r['query'][:100],
                'confidence': r.get('router_confidence', 0.0),
                'rationale': r.get('router_rationale', '')[:200],
                'llm_calls': r.get('total_llm_calls', 0),
                'signals': r.get('router_signals', []),
                'fallback': r.get('router_fallback', False)
            })
    
    # Confidence statistics
    confidences = [r.get('router_confidence', 0.0) for r in router_results if r.get('router_confidence', 0) > 0]
    
    if confidences:
        import statistics
        summary['confidence_stats']['mean'] = statistics.mean(confidences)
        summary['confidence_stats']['median'] = statistics.median(confidences)
        summary['confidence_stats']['min'] = min(confidences)
        summary['confidence_stats']['max'] = max(confidences)
        if len(confidences) > 1:
            summary['confidence_stats']['std_dev'] = statistics.stdev(confidences)
    
    # Average stats by decision type
    for choice in ['graphrag', 'naiverag']:
        queries = summary['by_decision'][choice]['queries']
        if queries:
            summary['by_decision'][choice]['avg_confidence'] = sum(q['confidence'] for q in queries) / len(queries)
            summary['by_decision'][choice]['avg_llm_calls'] = sum(q['llm_calls'] for q in queries) / len(queries)
    
    return summary


def save_results(results: List[Dict]):
    """Save analysis results to multiple output files."""
    
    output_folder = OUTPUT_FOLDER.resolve()
    output_folder.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Save main detailed results as CSV
    csv_path = output_folder / f"llm_calls_detailed_{timestamp}.csv"
    csv_fields = [
        'log_file', 'query', 'mode', 'pipeline_executed',
        'total_llm_calls', 'router_calls', 'agent1_calls', 'agent1b_calls', 'agent2_calls',
        'router_decision', 'router_confidence', 'router_fallback',
        'embed_calls', 'chunks_retrieved', 'triples_retrieved'
    ]
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(results)
    
    print(f"Detailed results saved to: {csv_path}")
    
    # 2. Save full JSON (includes rationale and signals)
    json_path = output_folder / f"llm_calls_detailed_{timestamp}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"JSON results saved to: {json_path}")
    
    # 3. Create and save router summary statistics
    router_summary = create_router_summary(results)
    router_summary_path = output_folder / f"router_summary_{timestamp}.json"
    with open(router_summary_path, 'w', encoding='utf-8') as f:
        # Convert Counter to dict for JSON serialization
        summary_copy = router_summary.copy()
        summary_copy['common_signals'] = dict(router_summary['common_signals'])
        json.dump(summary_copy, f, indent=2, ensure_ascii=False)
    
    print(f"Router summary statistics saved to: {router_summary_path}")
    
    # 4. Save router decisions breakdown
    router_path = output_folder / f"router_decisions_{timestamp}.json"
    router_data = []
    for r in results:
        if r.get('router_decision'):
            router_data.append({
                'log_file': r['log_file'],
                'query': r['query'],
                'decision': r['router_decision'],
                'executed': r['pipeline_executed'],
                'confidence': r['router_confidence'],
                'rationale': r['router_rationale'],
                'signals': r['router_signals'],
                'fallback': r['router_fallback']
            })
    
    with open(router_path, 'w', encoding='utf-8') as f:
        json.dump(router_data, f, indent=2, ensure_ascii=False)
    
    print(f"Router decisions saved to: {router_path}")
    
    # 5. Save router statistics as CSV
    router_stats_csv_path = output_folder / f"router_statistics_{timestamp}.csv"
    with open(router_stats_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Value'])
        writer.writerow(['Total Queries', router_summary['total_queries']])
        writer.writerow(['Fallback Decisions', router_summary['fallback_count']])
        writer.writerow([''])
        writer.writerow(['Router Decisions:', ''])
        for decision, count in router_summary['decisions'].items():
            if count > 0:
                writer.writerow([f'  {decision}', count])
        writer.writerow([''])
        writer.writerow(['Pipeline Executed:', ''])
        for pipeline, count in router_summary['pipeline_executed'].items():
            if count > 0:
                writer.writerow([f'  {pipeline}', count])
        writer.writerow([''])
        writer.writerow(['Confidence Mean', f"{router_summary['confidence_stats']['mean']:.4f}"])
        writer.writerow(['Confidence Median', f"{router_summary['confidence_stats']['median']:.4f}"])
        writer.writerow(['Confidence Min', f"{router_summary['confidence_stats']['min']:.4f}"])
        writer.writerow(['Confidence Max', f"{router_summary['confidence_stats']['max']:.4f}"])
        writer.writerow(['Confidence Std Dev', f"{router_summary['confidence_stats']['std_dev']:.4f}"])
        writer.writerow([''])
        writer.writerow(['GraphRAG Avg Confidence', f"{router_summary['by_decision']['graphrag']['avg_confidence']:.4f}"])
        writer.writerow(['GraphRAG Avg LLM Calls', f"{router_summary['by_decision']['graphrag']['avg_llm_calls']:.2f}"])
        writer.writerow(['NaiveRAG Avg Confidence', f"{router_summary['by_decision']['naiverag']['avg_confidence']:.4f}"])
        writer.writerow(['NaiveRAG Avg LLM Calls', f"{router_summary['by_decision']['naiverag']['avg_llm_calls']:.2f}"])
    
    print(f"Router statistics CSV saved to: {router_stats_csv_path}")
    
    # 6. Save human-readable router report
    router_txt_path = output_folder / f"router_readable_{timestamp}.txt"
    with open(router_txt_path, 'w', encoding='utf-8') as f:
        separator = "=" * 80
        title = "ROUTER DECISIONS - HUMAN READABLE"
        f.write(separator + "\n")
        f.write(title + "\n")
        f.write(separator + "\n")
        
        f.write("SUMMARY STATISTICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total queries analyzed: {router_summary['total_queries']}\n")
        f.write(f"Fallback (heuristic) decisions: {router_summary['fallback_count']}\n")
        
        f.write("Router Decision Distribution:\n")
        for decision, count in router_summary['decisions'].items():
            if count > 0:
                pct = (count / router_summary['total_queries'] * 100) if router_summary['total_queries'] > 0 else 0
                f.write(f"  {decision.upper()}: {count} ({pct:.1f}%)\n")
        
        f.write("\nPipeline Actually Executed:\n")
        for pipeline, count in router_summary['pipeline_executed'].items():
            if count > 0:
                pct = (count / router_summary['total_queries'] * 100) if router_summary['total_queries'] > 0 else 0
                f.write(f"  {pipeline.upper()}: {count} ({pct:.1f}%)\n")
        
        f.write(f"\nConfidence Statistics:\n")
        f.write(f"  Mean: {router_summary['confidence_stats']['mean']:.4f}\n")
        f.write(f"  Median: {router_summary['confidence_stats']['median']:.4f}\n")
        f.write(f"  Range: {router_summary['confidence_stats']['min']:.4f} - {router_summary['confidence_stats']['max']:.4f}\n")
        f.write(f"  Std Dev: {router_summary['confidence_stats']['std_dev']:.4f}\n")
        
        f.write("\nMost Common Router Signals:\n")
        for signal, count in router_summary['common_signals'].most_common(10):
            f.write(f"  {signal}: {count}\n")
        
        f.write("\n" + separator + "\n")
        f.write("DETAILED DECISIONS BY TYPE\n")
        f.write("-" * 80 + "\n")
        
        for decision_type in ['graphrag', 'naiverag']:
            decision_data = router_summary['by_decision'][decision_type]
            if decision_data['count'] > 0:
                f.write(f"{decision_type.upper()} ({decision_data['count']} queries)\n")
                f.write(f"Average Confidence: {decision_data['avg_confidence']:.4f}\n")
                f.write(f"Average LLM Calls: {decision_data['avg_llm_calls']:.2f}\n")
                f.write("-" * 40 + "\n")
                
                for i, q in enumerate(decision_data['queries'][:10], 1):
                    fallback_marker = " [FALLBACK]" if q.get('fallback', False) else ""
                    f.write(f"{i}. Confidence: {q['confidence']:.2f} | LLM Calls: {q['llm_calls']}{fallback_marker}\n")
                    f.write(f"   Query: {q['query']}\n")
                    f.write(f"   Signals: {', '.join(q['signals']) if q['signals'] else 'N/A'}\n")
                    f.write(f"   Rationale: {q['rationale']}\n")
        
        f.write(separator + "\n")
        f.write("INDIVIDUAL DECISIONS\n")
        f.write("-" * 80 + "\n")
        
        for i, r in enumerate(results, 1):
            if r.get('router_decision'):
                fallback_marker = " [FALLBACK HEURISTIC]" if r.get('router_fallback', False) else ""
                f.write(f"[{i}] {r['log_file']}\n")
                f.write(f"Query: {r['query']}\n")
                f.write(f"Router Decided: {r['router_decision']} | Executed: {r['pipeline_executed']}{fallback_marker}\n")
                f.write(f"Confidence: {r['router_confidence']:.2f} | LLM Calls: {r['total_llm_calls']}\n")
                f.write(f"Signals: {', '.join(r['router_signals']) if r['router_signals'] else 'N/A'}\n")
                f.write(f"Rationale: {r['router_rationale']}\n")
                f.write("-" * 80 + "\n")
    
    print(f"Human-readable router report saved to: {router_txt_path}")
    
    # 7. Create summary statistics with router info
    summary = create_summary(results, router_summary)
    summary_path = output_folder / f"llm_calls_summary_{timestamp}.txt"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(summary)
    
    print(f"Summary saved to: {summary_path}")
    print(summary)


def create_summary(results: List[Dict], router_summary: Dict) -> str:
    """Create a text summary of the analysis."""
    
    if not results:
        return "No results to summarize."
    
    total_files = len(results)
    
    # Filter by mode
    router_results = [r for r in results if r['mode'] == 'router']
    other_results = [r for r in results if r['mode'] != 'router']
    
    # Overall stats
    total_llm_calls = sum(r['total_llm_calls'] for r in results)
    total_router = sum(r['router_calls'] for r in results)
    total_agent1 = sum(r['agent1_calls'] for r in results)
    total_agent1b = sum(r['agent1b_calls'] for r in results)
    total_agent2 = sum(r['agent2_calls'] for r in results)
    total_embed_calls = sum(r['embed_calls'] for r in results)
    total_chunks = sum(r['chunks_retrieved'] for r in results)
    total_triples = sum(r['triples_retrieved'] for r in results)
    
    avg_llm_per_query = total_llm_calls / total_files if total_files > 0 else 0
    avg_embed_per_query = total_embed_calls / total_files if total_files > 0 else 0
    
    # Pipeline execution counts
    graphrag_executed = sum(1 for r in router_results if r.get('pipeline_executed') == 'graphrag')
    naiverag_executed = sum(1 for r in router_results if r.get('pipeline_executed') == 'naiverag')
    
    min_llm = min(r['total_llm_calls'] for r in results) if results else 0
    max_llm = max(r['total_llm_calls'] for r in results) if results else 0
    
    # Build summary sections using docstrings
    header = """
╔══════════════════════════════════════════════════════════════╗
║        ROUTER-BASED MULTI-AGENT RAG LLM CALL ANALYSIS       ║
╚══════════════════════════════════════════════════════════════╝
"""
    
    overall_section = f"""
📊 Overall Statistics:
├─ Total log files analyzed: {total_files}
│  ├─ Router mode: {len(router_results)}
│  └─ Other modes: {len(other_results)}
├─ Total LLM calls: {total_llm_calls}
│  ├─ Router: {total_router}
│  ├─ Agent 1 (entity extraction): {total_agent1}
│  ├─ Agent 1b (triple extraction): {total_agent1b}
│  └─ Agent 2 (answerer): {total_agent2}
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
    
    if router_results and router_summary['total_queries'] > 0:
        router_section = f"""
🎯 ROUTER DECISION ANALYSIS ({router_summary['total_queries']} queries):

📊 Router Decision Distribution:
"""
        for choice in ['graphrag', 'naiverag', 'graphrag_fallback', 'naiverag_fallback']:
            count = router_summary['decisions'][choice]
            if count > 0:
                percentage = (count / router_summary['total_queries']) * 100
                bar = "█" * int(percentage / 2)
                router_section += f"   ├─ {choice.upper()}: {count} ({percentage:.1f}%) {bar}\n"
        
        router_section += f"""
🔧 Pipeline Actually Executed:
├─ GraphRAG: {graphrag_executed} queries ({graphrag_executed/len(router_results)*100:.1f}%)
├─ NaiveRAG: {naiverag_executed} queries ({naiverag_executed/len(router_results)*100:.1f}%)
└─ Avg LLM calls: GraphRAG={router_summary['by_decision']['graphrag']['avg_llm_calls']:.2f}, NaiveRAG={router_summary['by_decision']['naiverag']['avg_llm_calls']:.2f}

⚠️  Fallback Decisions: {router_summary['fallback_count']} queries used heuristic fallback

📈 Confidence Statistics:
├─ Mean: {router_summary['confidence_stats']['mean']:.4f}
├─ Median: {router_summary['confidence_stats']['median']:.4f}
├─ Range: {router_summary['confidence_stats']['min']:.4f} - {router_summary['confidence_stats']['max']:.4f}
└─ Standard Deviation: {router_summary['confidence_stats']['std_dev']:.4f}
"""
        
        if router_summary['common_signals']:
            router_section += "\n🔍 Most Common Router Signals:\n"
            for signal, count in router_summary['common_signals'].most_common(5):
                router_section += f"   ├─ {signal}: {count}\n"
    else:
        router_section = ""
    
    pipeline_doc = """
💡 Pipeline Pattern (Router-based - Aligned with Script 1):
   └─ For each query:
      1. Router agent analyzes query (1 LLM call - JSON)
         - Chooses EITHER GraphRAG OR NaiveRAG (not both)
         - Uses same prompt/schema as Script 1 for consistency
         - Decision values: "graphrag" or "naiverag"
         - Considers: explicit citations, quote/define intent, multi-hop needs, 
                     entity relationships, predicates, scope/exceptions
         - Fallback: if LLM fails, heuristic based on regex patterns
      
      2a. IF Router chooses GraphRAG:
         - Agent 1: entity extraction (1 LLM call - JSON)
         - Agent 1b: triple extraction (1 LLM call - JSON)
         - Graph-based retrieval (triples + chunks)
         - Agent 2: answer from graph context (1 LLM call - TEXT)
         - Total: 1 (router) + 3 (graphrag) = 4 LLM calls
      
      2b. IF Router chooses NaiveRAG:
         - Direct vector search over chunks
         - Agent 1/1b: called but ignored (for logs only)
         - Agent 2: answer from chunk context (1 LLM call - TEXT)
         - Total: 1 (router) + 1 (naiverag) = 2 LLM calls
   
   └─ Key advantage: Only one pipeline runs (cost-efficient)
   └─ Agent 1/1b in Naive RAG not counted per user request
   └─ Router logic matches Script 1 exactly for consistent distribution
"""
    
    agent_breakdown = f"""
🔍 Agent Call Breakdown (across all queries):
├─ Router (chooses pipeline): {total_router}
├─ GraphRAG pipeline (when chosen):
│  ├─ Agent 1 (entity extraction): {total_agent1}
│  ├─ Agent 1b (triple extraction): {total_agent1b}
│  └─ Agent 2 (graph-based answerer): counted in total below
├─ NaiveRAG pipeline (when chosen):
│  ├─ Agent 1/1b: present but ignored in count
│  └─ Agent 2 (chunk-based answerer): counted in total below
├─ Agent 2 (total, both pipelines): {total_agent2}
└─ Total LLM calls: {total_llm_calls}
"""
    
    retrieval_stats = f"""
📦 Retrieval Statistics:
├─ Total chunks retrieved: {total_chunks}
└─ Total triples retrieved (GraphRAG only): {total_triples}
"""
    
    top_graphrag_section = """
📌 TOP QUERIES WHERE ROUTER CHOSE GRAPHRAG:
"""
    graphrag_queries = [r for r in router_results if r.get('router_decision', '').startswith('graphrag')]
    graphrag_sorted = sorted(graphrag_queries, key=lambda x: x.get('router_confidence', 0), reverse=True)
    for i, r in enumerate(graphrag_sorted[:5], 1):
        query_preview = r['query'][:60] + "..." if len(r['query']) > 60 else r['query']
        fallback = " [FALLBACK]" if r.get('router_fallback', False) else ""
        top_graphrag_section += f"{i}. Confidence: {r['router_confidence']:.2f} | LLM Calls: {r['total_llm_calls']}{fallback}\n"
        top_graphrag_section += f"   {query_preview}\n"
    
    if not graphrag_queries:
        top_graphrag_section += "   (No queries where GraphRAG was chosen)\n"
    
    top_naiverag_section = """
📌 TOP QUERIES WHERE ROUTER CHOSE NAIVERAG:
"""
    naiverag_queries = [r for r in router_results if r.get('router_decision', '').startswith('naiverag')]
    naiverag_sorted = sorted(naiverag_queries, key=lambda x: x.get('router_confidence', 0), reverse=True)
    for i, r in enumerate(naiverag_sorted[:5], 1):
        query_preview = r['query'][:60] + "..." if len(r['query']) > 60 else r['query']
        fallback = " [FALLBACK]" if r.get('router_fallback', False) else ""
        top_naiverag_section += f"{i}. Confidence: {r['router_confidence']:.2f} | LLM Calls: {r['total_llm_calls']}{fallback}\n"
        top_naiverag_section += f"   {query_preview}\n"
    
    if not naiverag_queries:
        top_naiverag_section += "   (No queries where NaiveRAG was chosen)\n"
    
    comparison_doc = """
📊 COMPARISON TO OTHER METHODS:
└─ Router-based Multi-Agent characteristics:
   ├─ Router intelligently selects ONE pipeline (not both)
   ├─ Router logic aligned with Script 1 for consistent decisions
   ├─ Lower cost than running both pipelines + aggregator
   ├─ GraphRAG: for multi-hop, relational, structural queries
   ├─ NaiveRAG: for direct, localized, single-article queries
   ├─ LLM cost: 4 calls (GraphRAG) or 2 calls (NaiveRAG)
   ├─ No aggregation overhead (unlike multi-agent with aggregator)
   ├─ Fallback heuristic ensures robustness
   └─ Best for: mixed workloads with varying complexity
"""
    
    notes_doc = f"""
📝 Notes:
   - Router pattern: 1 router call + 1 chosen pipeline
   - GraphRAG: 1 + 3 = 4 LLM calls total
   - NaiveRAG: 1 + 1 = 2 LLM calls total (Agent 1/1b ignored)
   - Agent 1/1b called in Naive RAG "for logs only" but not counted
   - Only ONE pipeline executes per query (cost-efficient)
   - Router uses exact same prompt/schema as Script 1
   - Decision values: "graphrag" or "naiverag" (aligned with Script 1)
   - Router signals: explicit_citations, request_quote_or_definition, 
                     entity_count_estimate, multi_hop_intent, comparison_intent,
                     exception_or_scope_intent, timeframe_or_effectivity
   - Fallback handling: if LLM router fails, heuristic routing based on:
     • explicit_citation regex patterns
     • quote_define keywords
     • multi_hop indicators
     • entity count (capitalized words)
   - Fallback decisions: {router_summary['fallback_count']} of {router_summary['total_queries']} queries
   - Common signals help understand routing logic
"""
    
    summary = (
        header + overall_section + averages_section + ranges_section + 
        router_section + pipeline_doc + agent_breakdown + retrieval_stats + 
        top_graphrag_section + top_naiverag_section + 
        comparison_doc + notes_doc
    )
    
    return summary


def main():
    """Main execution function."""
    
    separator = "=" * 70
    title = "ROUTER-BASED MULTI-AGENT RAG LLM CALL ANALYZER"
    subtitle = "(Tracking router decisions and single-pipeline execution)"
    
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