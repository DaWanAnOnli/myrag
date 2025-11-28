# analyze_llm_calls_multi_agent.py
"""
Analyzes log files from the Multi-Agent RAG (GraphRAG + Naive + Aggregator).
Tracks LLM calls per query including both pipelines and final aggregation.
Counts Agent 1/1b calls from GraphRAG pipeline.
Includes comprehensive aggregator decision analysis.

Adapted for Script 2 after aggregator changes:
- Aggregator decision line: "[Aggregator] Decision=choose_graphrag | Rationale: ..."
- Decisions: choose_graphrag | choose_naiverag | merge
- Confidence is no longer logged explicitly; we treat it as 0.0 (unknown).
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
LOG_FOLDER = SCRIPT_DIR / "../../4b_retrieval/4b_iv_multi_agents/question_terminal_logs_multi_agent/no_14_approach_2_both_newest"
OUTPUT_FOLDER = SCRIPT_DIR / "llm_calls_analysis_results_approach_2_both_5_hops_1250"


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


def extract_aggregator_decision(log_path: Path) -> Dict[str, any]:
    """
    Extract aggregator decision from the log.

    New Script 2 aggregator logs (after alignment with Script 1) look like:
      [Aggregator] Decision=choose_graphrag | Rationale: ...
    We map:
      choose_graphrag -> graphrag
      choose_naiverag -> naive
      merge           -> mixed

    Confidence is not explicitly logged anymore, so we set it to 0.0.
    """

    decision = {
        'chosen': None,
        'confidence': 0.0,
        'rationale': ''
    }

    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            content = f.read()

            # Extract decision (new pattern)
            # Example: [Aggregator] Decision=choose_graphrag | Rationale: ...
            decision_match = re.search(r'\[Aggregator\]\s+Decision=([a-zA-Z_]+)', content)
            if decision_match:
                raw_decision = decision_match.group(1).strip()
                # Map Script-1-style labels to old analyzer labels
                if raw_decision == "choose_graphrag":
                    decision['chosen'] = "graphrag"
                elif raw_decision == "choose_naiverag":
                    decision['chosen'] = "naive"
                elif raw_decision == "merge":
                    decision['chosen'] = "mixed"
                else:
                    decision['chosen'] = raw_decision  # unknown / future-proof

            # Confidence: not logged anymore → keep default 0.0

            # Extract rationale (unchanged pattern)
            rat_match = re.search(r'\[Aggregator\]\s+Rationale:\s*(.+)', content)
            if rat_match:
                decision['rationale'] = rat_match.group(1).strip()

    except Exception as e:
        print(f"Error extracting aggregator decision from {log_path.name}: {e}")

    return decision


def analyze_log_file(log_path: Path) -> Dict:
    """
    Analyze a single log file and count LLM calls.

    Returns dict with metrics, counting all LLM calls including Agent 1/1b.
    """

    result = {
        'log_file': log_path.name,
        'query': '',
        'agent1_calls': 0,  # Agent 1 in GraphRAG (counted)
        'agent1b_calls': 0,  # Agent 1b in GraphRAG (counted)
        'agent2_graphrag_calls': 0,  # Agent 2 in GraphRAG
        'agent2_naive_calls': 0,  # Agent 2 in Naive RAG
        'aggregator_calls': 0,  # Aggregator
        'total_llm_calls': 0,  # agent1 + agent1b + agent2_graphrag + agent2_naive + aggregator
        'embed_calls': 0,
        'chunks_graphrag': 0,
        'chunks_naive': 0,
        'triples_graphrag': 0,
        'aggregator_chosen': None,
        'aggregator_confidence': 0.0,
        'aggregator_rationale': '',
        'total_runtime_ms': 0.0,
        'mode': 'unknown'
    }

    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            content = f.read()

            # Extract query
            result['query'] = extract_query_from_log(log_path)

            # Detect mode
            if "Multi-Agent RAG run started" in content:
                result['mode'] = 'multi-agent'

            # Count Agent 1 prompts (GraphRAG only)
            agent1_matches = re.findall(r'\[Agent 1\] Prompt:', content)
            result['agent1_calls'] = len(agent1_matches)

            # Count Agent 1b prompts (GraphRAG only)
            agent1b_matches = re.findall(r'\[Agent 1b\] Prompt:', content)
            result['agent1b_calls'] = len(agent1b_matches)

            # Count Agent 2 GraphRAG prompts
            agent2_graphrag = re.findall(r'\[Agent 2 - GraphRAG\] Prompt:', content)
            result['agent2_graphrag_calls'] = len(agent2_graphrag)

            # Count Agent 2 Naive prompts
            agent2_naive = re.findall(r'\[Agent 2 - Naive\] Prompt:', content)
            result['agent2_naive_calls'] = len(agent2_naive)

            # Count Aggregator prompts
            aggregator_matches = re.findall(r'\[Aggregator\] Prompt:', content)
            result['aggregator_calls'] = len(aggregator_matches)

            # Total LLM calls = agent1 + agent1b + agent2_graphrag + agent2_naive + aggregator
            result['total_llm_calls'] = (
                result['agent1_calls'] +
                result['agent1b_calls'] +
                result['agent2_graphrag_calls'] +
                result['agent2_naive_calls'] +
                result['aggregator_calls']
            )

            # Count embedding calls
            embed_matches = re.findall(r'\[Embed\] text_len=', content)
            result['embed_calls'] = len(embed_matches)

            # Extract GraphRAG stats
            graphrag_chunks = re.search(r'\[GraphRAG\] Reranked chunks: selected (\d+)', content)
            if graphrag_chunks:
                result['chunks_graphrag'] = int(graphrag_chunks.group(1))

            graphrag_triples = re.search(r'\[GraphRAG\] Reranked triples: selected (\d+)', content)
            if graphrag_triples:
                result['triples_graphrag'] = int(graphrag_triples.group(1))

            # Extract Naive RAG stats
            naive_chunks = re.search(r'\[NaiveRAG\] Vector search returned (\d+) candidates', content)
            if naive_chunks:
                result['chunks_naive'] = int(naive_chunks.group(1))

            # Extract aggregator decision
            agg_decision = extract_aggregator_decision(log_path)
            result['aggregator_chosen'] = agg_decision['chosen']
            result['aggregator_confidence'] = agg_decision['confidence']
            result['aggregator_rationale'] = agg_decision['rationale']

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


def create_aggregator_summary(results: List[Dict]) -> Dict[str, any]:
    """Create comprehensive aggregator decision summary statistics."""

    multi_results = [r for r in results if r['mode'] == 'multi-agent']

    summary = {
        'total_queries': len(multi_results),
        'decisions': {
            'graphrag': 0,
            'naive': 0,
            'mixed': 0,
            'unknown': 0
        },
        'confidence_stats': {
            'mean': 0.0,
            'median': 0.0,
            'min': 0.0,
            'max': 0.0,
            'std_dev': 0.0
        },
        'by_decision': {
            'graphrag': {'count': 0, 'avg_confidence': 0.0, 'queries': []},
            'naive': {'count': 0, 'avg_confidence': 0.0, 'queries': []},
            'mixed': {'count': 0, 'avg_confidence': 0.0, 'queries': []}
        }
    }

    if not multi_results:
        return summary

    # Count decisions
    for r in multi_results:
        choice = r.get('aggregator_chosen', 'unknown')
        if choice in summary['decisions']:
            summary['decisions'][choice] += 1
        else:
            summary['decisions']['unknown'] += 1

        # Collect by decision type
        if choice in summary['by_decision']:
            summary['by_decision'][choice]['count'] += 1
            summary['by_decision'][choice]['queries'].append({
                'query': r['query'][:100],
                'confidence': r.get('aggregator_confidence', 0.0),
                'rationale': r.get('aggregator_rationale', '')[:200]
            })

    # Confidence statistics
    confidences = [r.get('aggregator_confidence', 0.0) for r in multi_results if r.get('aggregator_confidence', 0) > 0]

    if confidences:
        import statistics
        summary['confidence_stats']['mean'] = statistics.mean(confidences)
        summary['confidence_stats']['median'] = statistics.median(confidences)
        summary['confidence_stats']['min'] = min(confidences)
        summary['confidence_stats']['max'] = max(confidences)
        if len(confidences) > 1:
            summary['confidence_stats']['std_dev'] = statistics.stdev(confidences)

    # Average confidence by decision type
    for choice in ['graphrag', 'naive', 'mixed']:
        choice_confs = [q['confidence'] for q in summary['by_decision'][choice]['queries']]
        if choice_confs:
            summary['by_decision'][choice]['avg_confidence'] = sum(choice_confs) / len(choice_confs)

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
        'total_llm_calls', 'agent1_calls', 'agent1b_calls',
        'agent2_graphrag_calls', 'agent2_naive_calls', 'aggregator_calls',
        'aggregator_chosen', 'aggregator_confidence',
        'embed_calls', 'chunks_graphrag', 'chunks_naive', 'triples_graphrag'
    ]

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(results)

    print(f"Detailed results saved to: {csv_path}")

    # 2. Save full JSON (includes rationale)
    json_path = output_folder / f"llm_calls_detailed_{timestamp}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"JSON results saved to: {json_path}")

    # 3. Create and save aggregator summary statistics
    agg_summary = create_aggregator_summary(results)
    agg_summary_path = output_folder / f"aggregator_summary_{timestamp}.json"
    with open(agg_summary_path, 'w', encoding='utf-8') as f:
        json.dump(agg_summary, f, indent=2, ensure_ascii=False)

    print(f"Aggregator summary statistics saved to: {agg_summary_path}")

    # 4. Save aggregator decisions breakdown
    agg_path = output_folder / f"aggregator_decisions_{timestamp}.json"
    agg_data = []
    for r in results:
        if r.get('aggregator_chosen'):
            agg_data.append({
                'log_file': r['log_file'],
                'query': r['query'],
                'chosen': r['aggregator_chosen'],
                'confidence': r['aggregator_confidence'],
                'rationale': r['aggregator_rationale']
            })

    with open(agg_path, 'w', encoding='utf-8') as f:
        json.dump(agg_data, f, indent=2, ensure_ascii=False)

    print(f"Aggregator decisions saved to: {agg_path}")

    # 5. Save aggregator statistics as CSV
    agg_stats_csv_path = output_folder / f"aggregator_statistics_{timestamp}.csv"
    with open(agg_stats_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Value'])
        writer.writerow(['Total Queries', agg_summary['total_queries']])
        writer.writerow(['GraphRAG Chosen', agg_summary['decisions']['graphrag']])
        writer.writerow(['Naive Chosen', agg_summary['decisions']['naive']])
        writer.writerow(['Mixed Chosen', agg_summary['decisions']['mixed']])
        writer.writerow(['Unknown', agg_summary['decisions']['unknown']])
        writer.writerow([''])
        writer.writerow(['Confidence Mean', f"{agg_summary['confidence_stats']['mean']:.4f}"])
        writer.writerow(['Confidence Median', f"{agg_summary['confidence_stats']['median']:.4f}"])
        writer.writerow(['Confidence Min', f"{agg_summary['confidence_stats']['min']:.4f}"])
        writer.writerow(['Confidence Max', f"{agg_summary['confidence_stats']['max']:.4f}"])
        writer.writerow(['Confidence Std Dev', f"{agg_summary['confidence_stats']['std_dev']:.4f}"])
        writer.writerow([''])
        writer.writerow(['GraphRAG Avg Confidence', f"{agg_summary['by_decision']['graphrag']['avg_confidence']:.4f}"])
        writer.writerow(['Naive Avg Confidence', f"{agg_summary['by_decision']['naive']['avg_confidence']:.4f}"])
        writer.writerow(['Mixed Avg Confidence', f"{agg_summary['by_decision']['mixed']['avg_confidence']:.4f}"])

    print(f"Aggregator statistics CSV saved to: {agg_stats_csv_path}")

    # 6. Save human-readable aggregator report
    agg_txt_path = output_folder / f"aggregator_readable_{timestamp}.txt"
    with open(agg_txt_path, 'w', encoding='utf-8') as f:
        separator = "=" * 80
        title = "AGGREGATOR DECISIONS - HUMAN READABLE"
        f.write(separator + "\n")
        f.write(title + "\n")
        f.write(separator + "\n")

        f.write("SUMMARY STATISTICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total queries analyzed: {agg_summary['total_queries']}\n")

        f.write("Decision Distribution:\n")
        for decision, count in agg_summary['decisions'].items():
            if count > 0:
                pct = (count / agg_summary['total_queries'] * 100) if agg_summary['total_queries'] > 0 else 0
                f.write(f"  {decision.upper()}: {count} ({pct:.1f}%)\n")

        f.write(f"\nConfidence Statistics:\n")
        f.write(f"  Mean: {agg_summary['confidence_stats']['mean']:.4f}\n")
        f.write(f"  Median: {agg_summary['confidence_stats']['median']:.4f}\n")
        f.write(f"  Range: {agg_summary['confidence_stats']['min']:.4f} - {agg_summary['confidence_stats']['max']:.4f}\n")
        f.write(f"  Std Dev: {agg_summary['confidence_stats']['std_dev']:.4f}\n")

        f.write("\n" + separator + "\n")
        f.write("DETAILED DECISIONS BY TYPE\n")
        f.write("-" * 80 + "\n")

        for decision_type in ['graphrag', 'naive', 'mixed']:
            decision_data = agg_summary['by_decision'][decision_type]
            if decision_data['count'] > 0:
                f.write(f"{decision_type.upper()} ({decision_data['count']} queries)\n")
                f.write(f"Average Confidence: {decision_data['avg_confidence']:.4f}\n")
                f.write("-" * 40 + "\n")

                for i, q in enumerate(decision_data['queries'][:10], 1):  # Top 10
                    f.write(f"{i}. Confidence: {q['confidence']:.2f}\n")
                    f.write(f"   Query: {q['query']}\n")
                    f.write(f"   Rationale: {q['rationale']}\n")

                f.write("\n")

        f.write(separator + "\n")
        f.write("INDIVIDUAL DECISIONS\n")
        f.write("-" * 80 + "\n")

        for i, r in enumerate(results, 1):
            if r.get('aggregator_chosen'):
                f.write(f"[{i}] {r['log_file']}\n")
                f.write(f"Query: {r['query']}\n")
                f.write(f"Chosen: {r['aggregator_chosen']} | Confidence: {r['aggregator_confidence']:.2f}\n")
                f.write(f"Rationale: {r['aggregator_rationale']}\n")
                f.write("-" * 80 + "\n")

    print(f"Human-readable aggregator report saved to: {agg_txt_path}")

    # 7. Create summary statistics with aggregator info
    summary = create_summary(results, agg_summary)
    summary_path = output_folder / f"llm_calls_summary_{timestamp}.txt"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(summary)

    print(f"Summary saved to: {summary_path}")
    print(summary)


def create_summary(results: List[Dict], agg_summary: Dict) -> str:
    """Create a text summary of the analysis."""

    if not results:
        return "No results to summarize."

    total_files = len(results)

    # Filter by mode
    multi_results = [r for r in results if r['mode'] == 'multi-agent']
    other_results = [r for r in results if r['mode'] != 'multi-agent']

    # Overall stats
    total_llm_calls = sum(r['total_llm_calls'] for r in results)
    total_agent1 = sum(r['agent1_calls'] for r in results)
    total_agent1b = sum(r['agent1b_calls'] for r in results)
    total_agent2_graphrag = sum(r['agent2_graphrag_calls'] for r in results)
    total_agent2_naive = sum(r['agent2_naive_calls'] for r in results)
    total_aggregator = sum(r['aggregator_calls'] for r in results)
    total_embed_calls = sum(r['embed_calls'] for r in results)
    total_chunks_graphrag = sum(r['chunks_graphrag'] for r in results)
    total_chunks_naive = sum(r['chunks_naive'] for r in results)
    total_triples = sum(r['triples_graphrag'] for r in results)

    avg_llm_per_query = total_llm_calls / total_files if total_files > 0 else 0
    avg_embed_per_query = total_embed_calls / total_files if total_files > 0 else 0
    avg_chunks_graphrag = total_chunks_graphrag / total_files if total_files > 0 else 0
    avg_chunks_naive = total_chunks_naive / total_files if total_files > 0 else 0

    # Aggregator decision distribution
    agg_choices = [r['aggregator_chosen'] for r in multi_results if r.get('aggregator_chosen')]
    agg_distribution = Counter(agg_choices)

    min_llm = min(r['total_llm_calls'] for r in results) if results else 0
    max_llm = max(r['total_llm_calls'] for r in results) if results else 0

    # Build summary sections
    header = """
╔══════════════════════════════════════════════════════════════╗
║   MULTI-AGENT RAG (GRAPHRAG + NAIVE + AGGREGATOR) ANALYSIS  ║
╚══════════════════════════════════════════════════════════════╝
"""

    overall_section = f"""
📊 Overall Statistics:
├─ Total log files analyzed: {total_files}
│  ├─ Multi-agent mode: {len(multi_results)}
│  └─ Other modes: {len(other_results)}
├─ Total LLM calls: {total_llm_calls}
│  ├─ Agent 1 (entity extraction): {total_agent1}
│  ├─ Agent 1b (triple extraction): {total_agent1b}
│  ├─ Agent 2 GraphRAG: {total_agent2_graphrag}
│  ├─ Agent 2 Naive: {total_agent2_naive}
│  └─ Aggregator: {total_aggregator}
└─ Total embedding calls: {total_embed_calls}
"""

    averages_section = f"""
📈 Per-Query Averages:
├─ Average LLM calls per query: {avg_llm_per_query:.2f}
├─ Average embedding calls per query: {avg_embed_per_query:.2f}
├─ Average chunks (GraphRAG): {avg_chunks_graphrag:.2f}
└─ Average chunks (Naive): {avg_chunks_naive:.2f}
"""

    ranges_section = f"""
📉 Range:
├─ Minimum LLM calls in a query: {min_llm}
└─ Maximum LLM calls in a query: {max_llm}
"""

    if multi_results and agg_summary['total_queries'] > 0:
        agg_section = f"""
🤖 AGGREGATOR DECISION ANALYSIS ({agg_summary['total_queries']} queries):

📊 Decision Distribution:
"""
        for choice in ['graphrag', 'naive', 'mixed']:
            count = agg_summary['decisions'][choice]
            if count > 0:
                percentage = (count / agg_summary['total_queries']) * 100
                avg_conf = agg_summary['by_decision'][choice]['avg_confidence']
                bar = "█" * int(percentage / 2)
                agg_section += f"   ├─ {choice.upper()}: {count} queries ({percentage:.1f}%) | Avg confidence: {avg_conf:.3f} {bar}\n"

        agg_section += f"""
📈 Confidence Statistics:
├─ Mean: {agg_summary['confidence_stats']['mean']:.4f}
├─ Median: {agg_summary['confidence_stats']['median']:.4f}
├─ Range: {agg_summary['confidence_stats']['min']:.4f} - {agg_summary['confidence_stats']['max']:.4f}
└─ Standard Deviation: {agg_summary['confidence_stats']['std_dev']:.4f}
"""
    else:
        agg_section = ""

    pipeline_doc = """
💡 Pipeline Pattern (Multi-Agent):
   └─ For each query:
      1. Run GraphRAG pipeline:
         - Agent 1: entity extraction (1 LLM call - JSON)
         - Agent 1b: triple extraction (1 LLM call - JSON)
         - Graph-based retrieval (triples + chunks)
         - Agent 2: answer from graph context (1 LLM call - TEXT)
      2. Run Naive RAG pipeline (parallel):
         - Direct vector search over chunks (no Agent 1/1b)
         - Agent 2: answer from chunk context (1 LLM call - TEXT)
      3. Aggregator:
         - Compares both answers
         - Chooses best or combines them (1 LLM call - JSON)
   └─ Total LLM calls = 5 (3 GraphRAG + 1 Naive + 1 Aggregator)
   └─ Agent 1/1b only used in GraphRAG pipeline
"""

    agent_breakdown = f"""
🔍 Agent Call Breakdown (across all queries):
├─ GraphRAG pipeline:
│  ├─ Agent 1 (entity extraction): {total_agent1}
│  ├─ Agent 1b (triple extraction): {total_agent1b}
│  └─ Agent 2 (graph-based answerer): {total_agent2_graphrag}
├─ Naive RAG pipeline:
│  └─ Agent 2 (chunk-based answerer): {total_agent2_naive}
├─ Aggregation:
│  └─ Aggregator (synthesis): {total_aggregator}
└─ Total LLM calls: {total_llm_calls}
"""

    retrieval_stats = f"""
📦 Retrieval Statistics:
├─ GraphRAG:
│  ├─ Total triples retrieved: {total_triples}
│  └─ Total chunks retrieved: {total_chunks_graphrag}
└─ Naive RAG:
   └─ Total chunks retrieved: {total_chunks_naive}
"""

    top_graphrag_section = """
📌 TOP QUERIES WHERE GRAPHRAG WAS CHOSEN:
"""
    graphrag_chosen = [r for r in multi_results if r.get('aggregator_chosen') == 'graphrag']
    graphrag_chosen_sorted = sorted(graphrag_chosen, key=lambda x: x.get('aggregator_confidence', 0), reverse=True)
    for i, r in enumerate(graphrag_chosen_sorted[:5], 1):
        query_preview = r['query'][:60] + "..." if len(r['query']) > 60 else r['query']
        top_graphrag_section += f"{i}. Confidence: {r['aggregator_confidence']:.2f}\n"
        top_graphrag_section += f"   {query_preview}\n"

    if not graphrag_chosen:
        top_graphrag_section += "   (No queries where GraphRAG was chosen)\n"

    top_naive_section = """
📌 TOP QUERIES WHERE NAIVE WAS CHOSEN:
"""
    naive_chosen = [r for r in multi_results if r.get('aggregator_chosen') == 'naive']
    naive_chosen_sorted = sorted(naive_chosen, key=lambda x: x.get('aggregator_confidence', 0), reverse=True)
    for i, r in enumerate(naive_chosen_sorted[:5], 1):
        query_preview = r['query'][:60] + "..." if len(r['query']) > 60 else r['query']
        top_naive_section += f"{i}. Confidence: {r['aggregator_confidence']:.2f}\n"
        top_naive_section += f"   {query_preview}\n"

    if not naive_chosen:
        top_naive_section += "   (No queries where Naive was chosen)\n"

    mixed_chosen = [r for r in multi_results if r.get('aggregator_chosen') == 'mixed']
    mixed_section = ""
    if mixed_chosen:
        mixed_section = f"""
📌 QUERIES WHERE MIXED APPROACH WAS USED ({len(mixed_chosen)} queries):
"""
        for i, r in enumerate(mixed_chosen[:5], 1):
            query_preview = r['query'][:55] + "..." if len(r['query']) > 55 else r['query']
            mixed_section += f"{i}. Confidence: {r['aggregator_confidence']:.2f}\n"
            mixed_section += f"   {query_preview}\n"

    comparison_doc = """
📊 COMPARISON TO OTHER METHODS:
└─ Multi-Agent characteristics:
   ├─ Combines strengths of both approaches
   ├─ GraphRAG: knowledge graph navigation, triple matching, entity extraction
   ├─ Naive RAG: direct vector similarity over chunks
   ├─ Aggregator intelligently selects or merges answers
   ├─ Higher cost: 5 LLM calls (1+1+1 GraphRAG + 1 Naive + 1 Aggregator)
   ├─ Embeddings: high (both pipelines embed independently)
   └─ Best for: complex queries benefiting from multiple perspectives
"""

    notes_doc = """
📝 Notes:
   - Fixed pattern: always 5 LLM calls per query
   - GraphRAG: 3 calls (Agent 1 + Agent 1b + Agent 2)
   - Naive RAG: 1 call (Agent 2 only, no entity/triple extraction)
   - Aggregator: 1 call (evaluates and synthesizes both answers)
   - No iterative loops (single pass)
   - Agent 1/1b used ONLY in GraphRAG pipeline
   - Aggregator chooses: 'graphrag', 'naive', or 'mixed'
   - Aggregator considers citation density and completeness
   - Both pipelines run independently and conceptually in parallel
   - Embedding calls: high due to dual retrieval paths
   - Aggregator statistics saved to separate files for detailed analysis
"""

    summary = (
        header + overall_section + averages_section + ranges_section +
        agg_section + pipeline_doc + agent_breakdown + retrieval_stats +
        top_graphrag_section + top_naive_section + mixed_section +
        comparison_doc + notes_doc
    )

    return summary


def main():
    """Main execution function."""

    separator = "=" * 70
    title = "MULTI-AGENT RAG (GRAPHRAG + NAIVE + AGGREGATOR) ANALYZER"
    subtitle = "(Tracking dual pipelines with intelligent aggregation)"

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