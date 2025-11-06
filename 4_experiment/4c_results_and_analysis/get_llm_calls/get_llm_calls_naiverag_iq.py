# analyze_llm_calls_iq.py
"""
Analyzes log files from the Intermediate Questions (IQ) + Query Modifier RAG.
Tracks LLM calls per query including IQ planning, sequential enrichment, and answering.
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
LOG_FOLDER = SCRIPT_DIR / "../../4b_retrieval/4b_ii_naiverag/question_terminal_logs_naive_over_graph/naiverag_5_iq_1250"
OUTPUT_FOLDER = SCRIPT_DIR / "llm_calls_analysis_results_naiverag_5_iq_1250"


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


def extract_iq_info(log_path: Path) -> tuple[int, int, List[str]]:
    """
    Extract IQ information from the log.
    Returns: (num_iqs_planned, num_iqs_executed, list_of_iq_questions)
    """
    num_iqs_planned = 0
    num_iqs_executed = 0
    iq_questions = []
    
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
            # Look for the IQ planning log
            iq_match = re.search(r'\[IQ Planner\] Planned (\d+) IQ\(s\):', content)
            if iq_match:
                num_iqs_planned = int(iq_match.group(1))
            
            # Count executed IQs from --- IQ X/Y --- markers
            iq_markers = re.findall(r'--- IQ (\d+)/(\d+) ---', content)
            if iq_markers:
                num_iqs_executed = max(int(m[0]) for m in iq_markers)
            
            # Extract IQ questions from the summary section
            summary_match = re.search(
                r'=== Agentic RAG \(Sequential IQs\) summary ===(.*?)=== Final Answer ===',
                content,
                re.DOTALL
            )
            if summary_match:
                summary_section = summary_match.group(1)
                # Look for lines like "• [IQ1] What is..."
                iq_lines = re.findall(r'•\s*\[?\w+\]?\s*(.+)', summary_section)
                iq_questions = [q.strip() for q in iq_lines]
            
            # Fallback: if num_iqs_executed is 0, use the planned count
            if num_iqs_executed == 0:
                num_iqs_executed = num_iqs_planned
    
    except Exception as e:
        print(f"Error extracting IQs from {log_path.name}: {e}")
    
    return num_iqs_planned, num_iqs_executed, iq_questions


def analyze_log_file(log_path: Path) -> Dict:
    """
    Analyze a single log file and count LLM calls.
    
    Returns dict with metrics, ignoring Agent 1/1b LLM calls per user request.
    """
    
    result = {
        'log_file': log_path.name,
        'query': '',
        'num_iqs_planned': 0,
        'num_iqs_executed': 0,
        'iq_questions': [],
        'iq_planner_calls': 0,  # IQ planner (1 JSON call)
        'query_modifier_calls': 0,  # Query modifier per IQ after first (N-1 JSON calls)
        'agent2_calls': 0,  # Agent 2 per IQ (N TEXT calls)
        'agent1_1b_calls_ignored': 0,  # Agent 1/1b (not counted)
        'total_llm_calls': 0,  # iq_planner + query_modifier + agent2
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
            if "Agentic RAG (Sequential IQs)" in content:
                result['mode'] = 'sequential-iq'
            
            # Extract IQ info
            num_planned, num_executed, iq_questions = extract_iq_info(log_path)
            result['num_iqs_planned'] = num_planned
            result['num_iqs_executed'] = num_executed
            result['iq_questions'] = iq_questions
            
            # Count IQ Planner prompts (should be 1)
            iq_planner_matches = re.findall(r'\[IQ Planner\] Prompt:', content)
            result['iq_planner_calls'] = len(iq_planner_matches)
            
            # Count Query Modifier prompts (should be N-1 for N IQs)
            query_modifier_matches = re.findall(r'\[Query Modifier\] Prompt:', content)
            result['query_modifier_calls'] = len(query_modifier_matches)
            
            # Count Agent 2 prompts (one per IQ)
            agent2_matches = re.findall(r'\[Agent 2\] Prompt:', content)
            result['agent2_calls'] = len(agent2_matches)
            
            # Count Agent 1 and 1b prompts (for info only - not counted in total)
            agent1_matches = re.findall(r'\[Agent 1\] Prompt:', content)
            agent1b_matches = re.findall(r'\[Agent 1b\] Prompt:', content)
            result['agent1_1b_calls_ignored'] = len(agent1_matches) + len(agent1b_matches)
            
            # Total LLM calls = iq_planner + query_modifier + agent2
            result['total_llm_calls'] = (
                result['iq_planner_calls'] + 
                result['query_modifier_calls'] + 
                result['agent2_calls']
            )
            
            # Count embedding calls (1 per IQ executed)
            embed_matches = re.findall(r'Embedded used IQ in', content)
            result['embed_calls'] = len(embed_matches)
            
            # Extract chunks retrieved (sum across all IQs)
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
        'log_file', 'query', 'mode', 
        'num_iqs_planned', 'num_iqs_executed',
        'total_llm_calls', 'iq_planner_calls', 'query_modifier_calls', 'agent2_calls',
        'agent1_1b_calls_ignored', 'embed_calls', 'chunks_retrieved_total'
    ]
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(results)
    
    print(f"Detailed results saved to: {csv_path}")
    
    # 2. Save full JSON (includes IQ questions)
    json_path = output_folder / f"llm_calls_detailed_{timestamp}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"JSON results saved to: {json_path}")
    
    # 3. Save IQ breakdown
    iq_path = output_folder / f"iq_breakdown_{timestamp}.json"
    iq_data = []
    for r in results:
        if r.get('num_iqs_planned', 0) > 0:
            iq_data.append({
                'log_file': r['log_file'],
                'query': r['query'],
                'num_iqs_planned': r['num_iqs_planned'],
                'num_iqs_executed': r['num_iqs_executed'],
                'iq_questions': r['iq_questions']
            })
    
    with open(iq_path, 'w', encoding='utf-8') as f:
        json.dump(iq_data, f, indent=2, ensure_ascii=False)
    
    print(f"IQ breakdown saved to: {iq_path}")
    
    # 4. Save human-readable IQ report
    iq_txt_path = output_folder / f"iq_readable_{timestamp}.txt"
    with open(iq_txt_path, 'w', encoding='utf-8') as f:
        separator = "=" * 80
        title = "INTERMEDIATE QUESTIONS (IQ) BREAKDOWN - HUMAN READABLE"
        f.write(separator + "\n")
        f.write(title + "\n")
        f.write(separator + "\n")
        
        for i, r in enumerate(results, 1):
            if r.get('num_iqs_planned', 0) > 0:
                f.write(f"[{i}] {r['log_file']}\n")
                f.write(f"Original Query: {r['query']}\n")
                f.write(f"IQs Planned: {r['num_iqs_planned']} | IQs Executed: {r['num_iqs_executed']}\n")
                f.write("-" * 80 + "\n")
                
                for j, iq in enumerate(r['iq_questions'], 1):
                    f.write(f"  IQ {j}: {iq}\n")
                
                f.write("\n")
    
    print(f"Human-readable IQs saved to: {iq_txt_path}")
    
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
    iq_results = [r for r in results if r['mode'] == 'sequential-iq']
    other_results = [r for r in results if r['mode'] != 'sequential-iq']
    
    # Overall stats
    total_llm_calls = sum(r['total_llm_calls'] for r in results)
    total_iq_planner = sum(r['iq_planner_calls'] for r in results)
    total_query_modifier = sum(r['query_modifier_calls'] for r in results)
    total_agent2 = sum(r['agent2_calls'] for r in results)
    total_ignored = sum(r['agent1_1b_calls_ignored'] for r in results)
    total_embed_calls = sum(r['embed_calls'] for r in results)
    total_chunks = sum(r['chunks_retrieved_total'] for r in results)
    
    avg_llm_per_query = total_llm_calls / total_files if total_files > 0 else 0
    avg_embed_per_query = total_embed_calls / total_files if total_files > 0 else 0
    avg_chunks = total_chunks / total_files if total_files > 0 else 0
    
    # IQ statistics
    iq_planned_counts = [r['num_iqs_planned'] for r in iq_results if r.get('num_iqs_planned', 0) > 0]
    iq_executed_counts = [r['num_iqs_executed'] for r in iq_results if r.get('num_iqs_executed', 0) > 0]
    iq_planned_distribution = Counter(iq_planned_counts)
    iq_executed_distribution = Counter(iq_executed_counts)
    
    if iq_results:
        total_iqs_planned = sum(r['num_iqs_planned'] for r in iq_results)
        total_iqs_executed = sum(r['num_iqs_executed'] for r in iq_results)
        avg_iqs_planned = total_iqs_planned / len(iq_results)
        avg_iqs_executed = total_iqs_executed / len(iq_results)
        min_iqs_planned = min(iq_planned_counts) if iq_planned_counts else 0
        max_iqs_planned = max(iq_planned_counts) if iq_planned_counts else 0
        min_iqs_executed = min(iq_executed_counts) if iq_executed_counts else 0
        max_iqs_executed = max(iq_executed_counts) if iq_executed_counts else 0
        completion_rate = (total_iqs_executed / total_iqs_planned * 100) if total_iqs_planned > 0 else 0
    else:
        total_iqs_planned = total_iqs_executed = avg_iqs_planned = avg_iqs_executed = 0
        min_iqs_planned = max_iqs_planned = min_iqs_executed = max_iqs_executed = 0
        completion_rate = 0
    
    min_llm = min(r['total_llm_calls'] for r in results) if results else 0
    max_llm = max(r['total_llm_calls'] for r in results) if results else 0
    
    # Build summary sections using docstrings
    header = """
╔══════════════════════════════════════════════════════════════╗
║  INTERMEDIATE QUESTIONS (IQ) + QUERY MODIFIER ANALYSIS       ║
╚══════════════════════════════════════════════════════════════╝
"""
    
    overall_section = f"""
📊 Overall Statistics:
├─ Total log files analyzed: {total_files}
│  ├─ Sequential IQ mode: {len(iq_results)}
│  └─ Other modes: {len(other_results)}
├─ Total LLM calls: {total_llm_calls}
│  ├─ IQ planner calls: {total_iq_planner}
│  ├─ Query modifier calls: {total_query_modifier}
│  └─ Agent 2 (answerer) calls: {total_agent2}
├─ LLM calls ignored (Agent 1/1b): {total_ignored}
└─ Total embedding calls: {total_embed_calls}
"""
    
    averages_section = f"""
📈 Per-Query Averages:
├─ Average LLM calls per query: {avg_llm_per_query:.2f}
├─ Average embedding calls per query: {avg_embed_per_query:.2f}
├─ Average IQs planned per query: {avg_iqs_planned:.2f}
├─ Average IQs executed per query: {avg_iqs_executed:.2f}
└─ Average chunks retrieved per query: {avg_chunks:.2f}
"""
    
    ranges_section = f"""
📉 Range:
├─ Minimum LLM calls in a query: {min_llm}
└─ Maximum LLM calls in a query: {max_llm}
"""
    
    if iq_results:
        iq_section = f"""
🔄 INTERMEDIATE QUESTIONS (IQ) ANALYSIS ({len(iq_results)} queries):
├─ Total IQs planned: {total_iqs_planned}
├─ Total IQs executed: {total_iqs_executed}
├─ Completion rate: {completion_rate:.1f}%
├─ Average IQs planned per query: {avg_iqs_planned:.2f}
├─ Average IQs executed per query: {avg_iqs_executed:.2f}
├─ Min/Max IQs planned: {min_iqs_planned}/{max_iqs_planned}
└─ Min/Max IQs executed: {min_iqs_executed}/{max_iqs_executed}

📈 Planned IQ Distribution:
"""
        for count in sorted(iq_planned_distribution.keys()):
            freq = iq_planned_distribution[count]
            percentage = (freq / len(iq_results)) * 100
            bar = "█" * int(percentage / 2)
            iq_section += f"   ├─ {count} IQ(s) planned: {freq} queries ({percentage:.1f}%) {bar}\n"
        
        iq_section += """
📊 Executed IQ Distribution:
"""
        for count in sorted(iq_executed_distribution.keys()):
            freq = iq_executed_distribution[count]
            percentage = (freq / len(iq_results)) * 100
            bar = "█" * int(percentage / 2)
            iq_section += f"   ├─ {count} IQ(s) executed: {freq} queries ({percentage:.1f}%) {bar}\n"
    else:
        iq_section = ""
    
    pipeline_doc = """
💡 Pipeline Pattern (Sequential IQ + Query Modifier):
   └─ For each query:
      1. IQ Planner creates sequential plan of N dependent IQs (1 LLM call)
      2. For each IQ in sequence:
         - First IQ: use as-is
         - Subsequent IQs: Query Modifier enriches using prior Q/A pairs (1 LLM call)
         - Embed enriched IQ (1 embedding)
         - Vector search TextChunk nodes
         - Agent 2 answers (1 LLM call)
      3. Final answer = answer from last IQ (no aggregator)
   └─ Total LLM calls = 1 (planner) + (N-1) (modifiers) + N (agent2) = 2N
   └─ Sequential execution: each IQ depends on previous answers
   └─ Agent 1/1b present but calls excluded from LLM count per user request
"""
    
    agent_breakdown = f"""
🔍 Agent Call Breakdown (across all queries):
├─ IQ Planner: {total_iq_planner}
├─ Query Modifier (enriches IQs 2..N): {total_query_modifier}
├─ Agent 2 (answerer, per IQ): {total_agent2}
├─ Agent 1/1b (ignored): {total_ignored}
└─ Total LLM calls counted: {total_llm_calls}
"""
    
    top_queries_section = """
📌 TOP QUERIES BY NUMBER OF IQS PLANNED:
"""
    sorted_by_planned = sorted(iq_results, key=lambda x: x['num_iqs_planned'], reverse=True)
    for i, r in enumerate(sorted_by_planned[:5], 1):
        query_preview = r['query'][:60] + "..." if len(r['query']) > 60 else r['query']
        top_queries_section += f"{i}. {r['num_iqs_planned']} planned / {r['num_iqs_executed']} executed | {r['total_llm_calls']} LLM calls\n"
        top_queries_section += f"   {query_preview}\n"
    
    top_llm_section = """
📌 TOP QUERIES BY TOTAL LLM CALLS:
"""
    sorted_by_calls = sorted(results, key=lambda x: x['total_llm_calls'], reverse=True)
    for i, r in enumerate(sorted_by_calls[:5], 1):
        query_preview = r['query'][:60] + "..." if len(r['query']) > 60 else r['query']
        iq_info = f"{r['num_iqs_planned']} planned / {r['num_iqs_executed']} executed" if r.get('num_iqs_planned') else "N/A"
        top_llm_section += f"{i}. {r['total_llm_calls']} calls | IQs: {iq_info}\n"
        top_llm_section += f"   {query_preview}\n"
    
    comparison_doc = """
📊 COMPARISON TO OTHER METHODS:
└─ Sequential IQ + Query Modifier characteristics:
   ├─ Decomposes complex queries into dependent sequence
   ├─ Sequential execution: later IQs use earlier answers
   ├─ Query Modifier enriches each IQ with concrete facts from prior steps
   ├─ Uses naive RAG (vector search) for each IQ
   ├─ No graph traversal or triple matching per IQ
   ├─ No final aggregator (last IQ's answer is the final answer)
   ├─ Cost scales as 2N where N = number of IQs
   └─ Best for questions requiring step-by-step reasoning
"""
    
    # Queries with mismatches
    mismatches = [r for r in iq_results if r['num_iqs_planned'] != r['num_iqs_executed']]
    mismatch_section = ""
    if mismatches:
        mismatch_section = f"""
⚠️  QUERIES WITH PLANNED vs EXECUTED MISMATCH ({len(mismatches)} queries):
"""
        for r in mismatches[:5]:
            query_preview = r['query'][:60] + "..." if len(r['query']) > 60 else r['query']
            mismatch_section += f"- {r['log_file']}: {r['num_iqs_planned']} planned → {r['num_iqs_executed']} executed\n"
            mismatch_section += f"  {query_preview}\n"
    
    notes_doc = f"""
📝 Notes:
   - IQs are sequential and dependent (not parallel like subgoals)
   - Hard cap on IQs: MAX_INTERMEDIATE_QUESTIONS = 5 (configurable)
   - Expected LLM call pattern: 1 + (N-1) + N = 2N where N = num_iqs
   - Query Modifier enriches IQs 2 through N (N-1 calls)
   - Embedding calls: typically N (one per IQ executed)
   - Agent 1/1b used for entity/triple extraction but calls not counted
   - Chunks retrieved reported as sum across all IQs
   - Final answer comes from the last IQ only (no aggregation)
"""
    
    summary = (
        header + overall_section + averages_section + ranges_section + 
        iq_section + pipeline_doc + agent_breakdown + 
        top_queries_section + top_llm_section + mismatch_section + 
        comparison_doc + notes_doc
    )
    
    return summary


def main():
    """Main execution function."""
    
    separator = "=" * 70
    title = "INTERMEDIATE QUESTIONS (IQ) + QUERY MODIFIER ANALYZER"
    subtitle = "(Tracking sequential IQ planning, enrichment, and answering)"
    
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