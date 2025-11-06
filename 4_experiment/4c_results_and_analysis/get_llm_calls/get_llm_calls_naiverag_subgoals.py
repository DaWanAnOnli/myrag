# analyze_llm_calls_subgoal.py
"""
Analyzes log files from the Subgoal + Aggregator RAG.
Tracks LLM calls per query including subgoal generation, parallel answering, and aggregation.
Ignores Agent 1/1b calls per user request.
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
LOG_FOLDER = SCRIPT_DIR / "../../4b_retrieval/4b_ii_naiverag/question_terminal_logs_naive_over_graph/naiverag_5_subgoals_1250"
OUTPUT_FOLDER = SCRIPT_DIR / "llm_calls_analysis_results_naiverag_5_subgoals_1250"


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


def extract_subgoals_info(log_path: Path) -> tuple[int, List[str]]:
    """
    Extract number of subgoals and their questions from the log.
    Returns: (num_subgoals, list_of_subgoal_questions)
    """
    num_subgoals = 0
    subgoal_questions = []
    
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
            # Look for the subgoal production log
            subgoal_match = re.search(r'\[SubgoalGen\] Produced (\d+) subgoal\(s\):', content)
            if subgoal_match:
                num_subgoals = int(subgoal_match.group(1))
            
            # Extract subgoal questions from the summary section
            summary_match = re.search(r'=== Agentic RAG summary ===(.*?)=== Final Answer ===', content, re.DOTALL)
            if summary_match:
                summary_section = summary_match.group(1)
                # Look for lines like "• SG1: What is..."
                sg_lines = re.findall(r'•\s*\w+:\s*(.+)', summary_section)
                subgoal_questions = [q.strip() for q in sg_lines]
            
            # Fallback: count [Subgoal X] markers
            if num_subgoals == 0:
                subgoal_markers = re.findall(r'\[Subgoal \w+\]', content)
                # Count unique subgoal IDs
                unique_ids = set(subgoal_markers)
                num_subgoals = len(unique_ids)
    
    except Exception as e:
        print(f"Error extracting subgoals from {log_path.name}: {e}")
    
    return num_subgoals, subgoal_questions


def analyze_log_file(log_path: Path) -> Dict:
    """
    Analyze a single log file and count LLM calls.
    
    Returns dict with metrics, ignoring Agent 1/1b LLM calls per user request.
    """
    
    result = {
        'log_file': log_path.name,
        'query': '',
        'num_subgoals': 0,
        'subgoal_questions': [],
        'subgoal_gen_calls': 0,  # Subgoal generator (1 JSON call)
        'agent2_calls': 0,  # Agent 2 per subgoal (N TEXT calls)
        'aggregator_calls': 0,  # Aggregator (1 TEXT call)
        'agent1_1b_calls_ignored': 0,  # Agent 1/1b (not counted)
        'total_llm_calls': 0,  # subgoal_gen + agent2 + aggregator
        'embed_calls': 0,
        'chunks_retrieved_total': 0,
        'total_runtime_ms': 0.0,
        'mode': 'unknown'
    }
    
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
            # Extract query
            result['query'] = extract_query_from_log(log_path)
            
            # Detect mode
            if "Agentic RAG run started" in content and "Subgoal" in content:
                result['mode'] = 'subgoal-aggregator'
            
            # Extract subgoals info
            num_subgoals, subgoal_questions = extract_subgoals_info(log_path)
            result['num_subgoals'] = num_subgoals
            result['subgoal_questions'] = subgoal_questions
            
            # Count Subgoal Generator prompts (should be 1)
            subgoal_gen_matches = re.findall(r'\[SubgoalGen\] Prompt:', content)
            result['subgoal_gen_calls'] = len(subgoal_gen_matches)
            
            # Count Agent 2 prompts (one per subgoal)
            agent2_matches = re.findall(r'\[Agent 2\] Prompt:', content)
            result['agent2_calls'] = len(agent2_matches)
            
            # Count Aggregator prompts (should be 1)
            aggregator_matches = re.findall(r'\[Aggregator\] Prompt:', content)
            result['aggregator_calls'] = len(aggregator_matches)
            
            # Count Agent 1 and 1b prompts (for info only - not counted in total)
            agent1_matches = re.findall(r'\[Agent 1\] Prompt:', content)
            agent1b_matches = re.findall(r'\[Agent 1b\] Prompt:', content)
            result['agent1_1b_calls_ignored'] = len(agent1_matches) + len(agent1b_matches)
            
            # Total LLM calls = subgoal_gen + agent2 + aggregator
            result['total_llm_calls'] = (
                result['subgoal_gen_calls'] + 
                result['agent2_calls'] + 
                result['aggregator_calls']
            )
            
            # Count embedding calls (1 for original query + 1 per subgoal)
            # Look for embedded query logs
            embed_original = re.findall(r'\[Step 0\] Embedded query', content)
            embed_subgoals = re.findall(r'\[Subgoal \w+\] Embedded in', content)
            result['embed_calls'] = len(embed_original) + len(embed_subgoals)
            
            # Extract chunks retrieved (sum across all subgoals)
            chunks_matches = re.findall(r'Vector search returned (\d+) candidates', content)
            result['chunks_retrieved_total'] = sum(int(m) for m in chunks_matches)
            
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


def save_results(results: List[Dict]):
    """Save analysis results to multiple output files."""
    
    output_folder = OUTPUT_FOLDER.resolve()
    output_folder.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Save main detailed results as CSV
    csv_path = output_folder / f"llm_calls_detailed_{timestamp}.csv"
    csv_fields = [
        'log_file', 'query', 'mode', 'num_subgoals',
        'total_llm_calls', 'subgoal_gen_calls', 'agent2_calls', 'aggregator_calls',
        'agent1_1b_calls_ignored', 'embed_calls', 'chunks_retrieved_total'
    ]
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(results)
    
    print(f"Detailed results saved to: {csv_path}")
    
    # 2. Save full JSON (includes subgoal questions)
    json_path = output_folder / f"llm_calls_detailed_{timestamp}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"JSON results saved to: {json_path}")
    
    # 3. Save subgoals breakdown
    subgoals_path = output_folder / f"subgoals_breakdown_{timestamp}.json"
    subgoals_data = []
    for r in results:
        if r.get('num_subgoals', 0) > 0:
            subgoals_data.append({
                'log_file': r['log_file'],
                'query': r['query'],
                'num_subgoals': r['num_subgoals'],
                'subgoal_questions': r['subgoal_questions']
            })
    
    with open(subgoals_path, 'w', encoding='utf-8') as f:
        json.dump(subgoals_data, f, indent=2, ensure_ascii=False)
    
    print(f"Subgoals breakdown saved to: {subgoals_path}")
    
    # 4. Save human-readable subgoals report
    subgoals_txt_path = output_folder / f"subgoals_readable_{timestamp}.txt"
    with open(subgoals_txt_path, 'w', encoding='utf-8') as f:
        separator = "=" * 80
        title = "SUBGOALS BREAKDOWN - HUMAN READABLE"
        f.write(separator + "\n")
        f.write(title + "\n")
        f.write(separator + "\n")
        
        for i, r in enumerate(results, 1):
            if r.get('num_subgoals', 0) > 0:
                f.write(f"[{i}] {r['log_file']}\n")
                f.write(f"Original Query: {r['query']}\n")
                f.write(f"Number of subgoals: {r['num_subgoals']}\n")
                f.write("-" * 80 + "\n")
                
                for j, sq in enumerate(r['subgoal_questions'], 1):
                    f.write(f"  Subgoal {j}: {sq}\n")
                
                f.write("\n")
    
    print(f"Human-readable subgoals saved to: {subgoals_txt_path}")
    
    # 5. Create summary statistics
    summary = create_summary(results)
    summary_path = output_folder / f"llm_calls_summary_{timestamp}.txt"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(summary)
    
    print(f"Summary saved to: {summary_path}")
    print(summary)


def create_summary(results: List[Dict]) -> str:
    """Create a text summary of the analysis."""
    
    if not results:
        return "No results to summarize."
    
    total_files = len(results)
    
    # Filter by mode
    subgoal_results = [r for r in results if r['mode'] == 'subgoal-aggregator']
    other_results = [r for r in results if r['mode'] != 'subgoal-aggregator']
    
    # Overall stats
    total_llm_calls = sum(r['total_llm_calls'] for r in results)
    total_subgoal_gen = sum(r['subgoal_gen_calls'] for r in results)
    total_agent2 = sum(r['agent2_calls'] for r in results)
    total_aggregator = sum(r['aggregator_calls'] for r in results)
    total_ignored = sum(r['agent1_1b_calls_ignored'] for r in results)
    total_embed_calls = sum(r['embed_calls'] for r in results)
    total_chunks = sum(r['chunks_retrieved_total'] for r in results)
    
    avg_llm_per_query = total_llm_calls / total_files if total_files > 0 else 0
    avg_embed_per_query = total_embed_calls / total_files if total_files > 0 else 0
    avg_chunks = total_chunks / total_files if total_files > 0 else 0
    
    # Subgoal statistics
    subgoal_counts = [r['num_subgoals'] for r in subgoal_results if r.get('num_subgoals', 0) > 0]
    subgoal_distribution = Counter(subgoal_counts)
    
    if subgoal_results:
        total_subgoals = sum(r['num_subgoals'] for r in subgoal_results)
        avg_subgoals = total_subgoals / len(subgoal_results)
        min_subgoals = min(subgoal_counts) if subgoal_counts else 0
        max_subgoals = max(subgoal_counts) if subgoal_counts else 0
    else:
        total_subgoals = avg_subgoals = min_subgoals = max_subgoals = 0
    
    min_llm = min(r['total_llm_calls'] for r in results) if results else 0
    max_llm = max(r['total_llm_calls'] for r in results) if results else 0
    
    # Build summary sections using docstrings to avoid issues
    header = """
╔══════════════════════════════════════════════════════════════╗
║    SUBGOAL + AGGREGATOR RAG LLM CALL ANALYSIS SUMMARY       ║
╚══════════════════════════════════════════════════════════════╝
"""
    
    overall_section = f"""
📊 Overall Statistics:
├─ Total log files analyzed: {total_files}
│  ├─ Subgoal-aggregator mode: {len(subgoal_results)}
│  └─ Other modes: {len(other_results)}
├─ Total LLM calls: {total_llm_calls}
│  ├─ Subgoal generator calls: {total_subgoal_gen}
│  ├─ Agent 2 (answerer) calls: {total_agent2}
│  └─ Aggregator calls: {total_aggregator}
├─ LLM calls ignored (Agent 1/1b): {total_ignored}
└─ Total embedding calls: {total_embed_calls}
"""
    
    averages_section = f"""
📈 Per-Query Averages:
├─ Average LLM calls per query: {avg_llm_per_query:.2f}
├─ Average embedding calls per query: {avg_embed_per_query:.2f}
├─ Average subgoals per query: {avg_subgoals:.2f}
└─ Average chunks retrieved per query: {avg_chunks:.2f}
"""
    
    ranges_section = f"""
📉 Range:
├─ Minimum LLM calls in a query: {min_llm}
└─ Maximum LLM calls in a query: {max_llm}
"""
    
    if subgoal_results:
        subgoals_section = f"""
🎯 SUBGOAL ANALYSIS ({len(subgoal_results)} queries):
├─ Total subgoals generated: {total_subgoals}
├─ Average subgoals per query: {avg_subgoals:.2f}
├─ Min/Max subgoals: {min_subgoals}/{max_subgoals}
└─ Subgoal Distribution:
"""
        for count in sorted(subgoal_distribution.keys()):
            freq = subgoal_distribution[count]
            percentage = (freq / len(subgoal_results)) * 100
            bar = "█" * int(percentage / 2)
            subgoals_section += f"   ├─ {count} subgoal(s): {freq} queries ({percentage:.1f}%) {bar}\n"
    else:
        subgoals_section = ""
    
    pipeline_doc = """
💡 Pipeline Pattern (Subgoal + Aggregator):
   └─ For each query:
      1. Subgoal Generator decomposes query into N subgoals (1 LLM call)
      2. Answer each subgoal in parallel:
         - Embed subgoal question (1 embedding per subgoal)
         - Vector search TextChunk nodes
         - Agent 2 answers (1 LLM call per subgoal)
      3. Aggregator synthesizes final answer from all Q/A pairs (1 LLM call)
   └─ Total LLM calls = 1 (subgoal gen) + N (agent2) + 1 (aggregator) = N + 2
   └─ Parallelism: subgoals answered concurrently (SUBGOAL_MAX_WORKERS threads)
   └─ Agent 1/1b present but calls excluded from LLM count per user request
"""
    
    agent_breakdown = f"""
🔍 Agent Call Breakdown (across all queries):
├─ Subgoal Generator: {total_subgoal_gen}
├─ Agent 2 (answerer, per subgoal): {total_agent2}
├─ Aggregator: {total_aggregator}
├─ Agent 1/1b (ignored): {total_ignored}
└─ Total LLM calls counted: {total_llm_calls}
"""
    
    top_queries_section = """
📌 TOP QUERIES BY NUMBER OF SUBGOALS:
"""
    sorted_by_subgoals = sorted(subgoal_results, key=lambda x: x['num_subgoals'], reverse=True)
    for i, r in enumerate(sorted_by_subgoals[:5], 1):
        query_preview = r['query'][:60] + "..." if len(r['query']) > 60 else r['query']
        top_queries_section += f"{i}. {r['num_subgoals']} subgoals | {r['total_llm_calls']} LLM calls\n"
        top_queries_section += f"   {query_preview}\n"
    
    top_llm_section = """
📌 TOP QUERIES BY TOTAL LLM CALLS:
"""
    sorted_by_calls = sorted(results, key=lambda x: x['total_llm_calls'], reverse=True)
    for i, r in enumerate(sorted_by_calls[:5], 1):
        query_preview = r['query'][:60] + "..." if len(r['query']) > 60 else r['query']
        sg_info = f"{r['num_subgoals']} subgoals" if r.get('num_subgoals') else "N/A"
        top_llm_section += f"{i}. {r['total_llm_calls']} calls | {sg_info}\n"
        top_llm_section += f"   {query_preview}\n"
    
    comparison_doc = """
📊 COMPARISON TO OTHER METHODS:
└─ Subgoal + Aggregator characteristics:
   ├─ Decomposes complex queries into independent sub-questions
   ├─ Parallel execution: faster than sequential approaches
   ├─ Uses naive RAG (vector search) for each subgoal
   ├─ No graph traversal or triple matching per subgoal
   ├─ Final aggregation synthesizes coherent answer
   ├─ Cost scales linearly with number of subgoals (N + 2 LLM calls)
   └─ Best for multi-part questions requiring independent retrieval
"""
    
    notes_doc = f"""
📝 Notes:
   - Subgoals are independent and answered in parallel
   - Hard cap on subgoals: MAX_SUBGOALS = 2 (configurable)
   - Expected LLM call pattern: 1 + N + 1 where N = num_subgoals
   - Embedding calls: typically N + 1 (original query may or may not be embedded)
   - Agent 1/1b used for entity/triple extraction but calls not counted
   - Chunks retrieved reported as sum across all subgoals
"""
    
    summary = (
        header + overall_section + averages_section + ranges_section + 
        subgoals_section + pipeline_doc + agent_breakdown + 
        top_queries_section + top_llm_section + comparison_doc + notes_doc
    )
    
    return summary


def main():
    """Main execution function."""
    
    separator = "=" * 70
    title = "SUBGOAL + AGGREGATOR RAG LLM CALL ANALYZER"
    subtitle = "(Tracking subgoal generation and parallel answering)"
    
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