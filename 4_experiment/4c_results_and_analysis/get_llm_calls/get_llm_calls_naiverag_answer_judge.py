# analyze_llm_calls_answer_judge.py
"""
Analyzes log files from the Answer Judge + Query Modifier RAG.
Tracks LLM calls per query including iterative answer evaluation and query refinement.
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
LOG_FOLDER = SCRIPT_DIR / "../../4b_retrieval/4b_ii_naiverag/question_terminal_logs_naive_over_graph/naiverag_5_answer_judge_1250_3rd"
OUTPUT_FOLDER = SCRIPT_DIR / "llm_calls_analysis_results_naiverag_5_answer_judge_1250_3rd"


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


def extract_iterations_info(log_path: Path) -> tuple[int, List[Dict[str, bool]]]:
    """
    Extract iteration count and judge decisions from the log.
    Returns: (num_iterations, list_of_iteration_details)
    """
    num_iterations = 0
    iteration_details = []
    
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
            # Look for iterations executed from summary
            iter_match = re.search(r'Iterations executed:\s*(\d+)', content)
            if iter_match:
                num_iterations = int(iter_match.group(1))
            
            # Count iteration markers
            iter_markers = re.findall(r'--- Iteration (\d+)/(\d+) ---', content)
            if iter_markers and num_iterations == 0:
                num_iterations = max(int(m[0]) for m in iter_markers)
            
            # Extract judge decisions
            judge_acceptable = re.findall(r'\[Judge\] Decision: ACCEPTABLE', content)
            judge_insufficient = re.findall(r'\[Judge\] Decision: INSUFFICIENT', content)
            
            # Build iteration details
            for i in range(1, num_iterations + 1):
                detail = {
                    'iteration': i,
                    'judge_decision': None,
                    'query_modified': False,
                    'final_iteration': False
                }
                
                # Determine judge decision for this iteration
                if i <= len(judge_acceptable) + len(judge_insufficient):
                    # Simple heuristic: if this is the last logged decision, check if acceptable
                    if i == len(judge_acceptable) + len(judge_insufficient):
                        detail['judge_decision'] = 'acceptable' if judge_acceptable else 'insufficient'
                
                # Check if this iteration ended the loop
                if i == num_iterations:
                    detail['final_iteration'] = True
                else:
                    detail['query_modified'] = True
                
                iteration_details.append(detail)
    
    except Exception as e:
        print(f"Error extracting iterations from {log_path.name}: {e}")
    
    return num_iterations, iteration_details


def analyze_log_file(log_path: Path) -> Dict:
    """
    Analyze a single log file and count LLM calls.
    
    Returns dict with metrics, ignoring Agent 1/1b LLM calls per user request.
    """
    
    result = {
        'log_file': log_path.name,
        'query': '',
        'num_iterations': 0,
        'iteration_details': [],
        'judge_acceptable_count': 0,
        'judge_insufficient_count': 0,
        'answer_judge_calls': 0,  # Answer Judge per iteration
        'query_modifier_calls': 0,  # Query Modifier when insufficient
        'agent2_calls': 0,  # Answerer (N times, one per iteration)
        'agent1_1b_calls_ignored': 0,  # Agent 1/1b (not counted)
        'total_llm_calls': 0,  # judge + modifier + agent2
        'embed_calls': 0,
        'chunks_retrieved_total': 0,
        'total_runtime_ms': 0.0,
        'mode': 'unknown',
        'final_verdict': None  # 'acceptable', 'cap_reached', or None
    }
    
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
            # Extract query
            result['query'] = extract_query_from_log(log_path)
            
            # Detect mode
            if "Agentic RAG (Answer Judge + Modifier)" in content:
                result['mode'] = 'answer-judge-modifier'
            
            # Extract iterations info
            num_iters, iter_details = extract_iterations_info(log_path)
            result['num_iterations'] = num_iters
            result['iteration_details'] = iter_details
            
            # Count Answer Judge prompts
            judge_matches = re.findall(r'\[Judge\] Prompt:', content)
            result['answer_judge_calls'] = len(judge_matches)
            
            # Count Query Modifier prompts
            modifier_matches = re.findall(r'\[Modifier\] Prompt:', content)
            result['query_modifier_calls'] = len(modifier_matches)
            
            # Count Agent 2 prompts (one per iteration)
            agent2_matches = re.findall(r'\[Agent 2\] Prompt:', content)
            result['agent2_calls'] = len(agent2_matches)
            
            # Count Agent 1 and 1b prompts (for info only - not counted in total)
            agent1_matches = re.findall(r'\[Agent 1\] Prompt:', content)
            agent1b_matches = re.findall(r'\[Agent 1b\] Prompt:', content)
            result['agent1_1b_calls_ignored'] = len(agent1_matches) + len(agent1b_matches)
            
            # Total LLM calls = judge + modifier + agent2
            result['total_llm_calls'] = (
                result['answer_judge_calls'] + 
                result['query_modifier_calls'] + 
                result['agent2_calls']
            )
            
            # Count judge verdicts
            judge_acceptable = re.findall(r'\[Judge\] Decision: ACCEPTABLE', content)
            judge_insufficient = re.findall(r'\[Judge\] Decision: INSUFFICIENT', content)
            result['judge_acceptable_count'] = len(judge_acceptable)
            result['judge_insufficient_count'] = len(judge_insufficient)
            
            # Determine final verdict
            if "Judge deemed answer ACCEPTABLE" in content:
                result['final_verdict'] = 'acceptable'
            elif "Iteration limit reached" in content:
                result['final_verdict'] = 'cap_reached'
            
            # Count embedding calls (1 per iteration)
            embed_matches = re.findall(r'Embedded query in', content)
            result['embed_calls'] = len(embed_matches)
            
            # Extract chunks retrieved (sum across all iterations)
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
        'log_file', 'query', 'mode', 'num_iterations', 'final_verdict',
        'total_llm_calls', 'answer_judge_calls', 'query_modifier_calls', 'agent2_calls',
        'judge_acceptable_count', 'judge_insufficient_count',
        'agent1_1b_calls_ignored', 'embed_calls', 'chunks_retrieved_total'
    ]
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(results)
    
    print(f"Detailed results saved to: {csv_path}")
    
    # 2. Save full JSON (includes iteration details)
    json_path = output_folder / f"llm_calls_detailed_{timestamp}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"JSON results saved to: {json_path}")
    
    # 3. Save iterations breakdown
    iter_path = output_folder / f"iterations_breakdown_{timestamp}.json"
    iter_data = []
    for r in results:
        if r.get('num_iterations', 0) > 0:
            iter_data.append({
                'log_file': r['log_file'],
                'query': r['query'],
                'num_iterations': r['num_iterations'],
                'final_verdict': r.get('final_verdict'),
                'iteration_details': r['iteration_details']
            })
    
    with open(iter_path, 'w', encoding='utf-8') as f:
        json.dump(iter_data, f, indent=2, ensure_ascii=False)
    
    print(f"Iterations breakdown saved to: {iter_path}")
    
    # 4. Save human-readable iterations report
    iter_txt_path = output_folder / f"iterations_readable_{timestamp}.txt"
    with open(iter_txt_path, 'w', encoding='utf-8') as f:
        separator = "=" * 80
        title = "ANSWER JUDGE ITERATIONS BREAKDOWN - HUMAN READABLE"
        f.write(separator + "\n")
        f.write(title + "\n")
        f.write(separator + "\n")
        
        for i, r in enumerate(results, 1):
            if r.get('num_iterations', 0) > 0:
                f.write(f"[{i}] {r['log_file']}\n")
                f.write(f"Original Query: {r['query']}\n")
                f.write(f"Iterations: {r['num_iterations']} | Final: {r.get('final_verdict')}\n")
                f.write("-" * 80 + "\n")
                
                for detail in r['iteration_details']:
                    iter_num = detail['iteration']
                    decision = detail.get('judge_decision', 'N/A')
                    
                    f.write(f"  Iteration {iter_num}:\n")
                    f.write(f"    Answer judge decision: {decision}\n")
                    
                    if detail.get('query_modified'):
                        f.write(f"    → Query modified for next iteration\n")
                    elif detail.get('final_iteration'):
                        f.write(f"    → Final iteration\n")
                
                f.write("\n")
    
    print(f"Human-readable iterations saved to: {iter_txt_path}")
    
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
    judge_results = [r for r in results if r['mode'] == 'answer-judge-modifier']
    other_results = [r for r in results if r['mode'] != 'answer-judge-modifier']
    
    # Overall stats
    total_llm_calls = sum(r['total_llm_calls'] for r in results)
    total_judge = sum(r['answer_judge_calls'] for r in results)
    total_modifier = sum(r['query_modifier_calls'] for r in results)
    total_agent2 = sum(r['agent2_calls'] for r in results)
    total_ignored = sum(r['agent1_1b_calls_ignored'] for r in results)
    total_embed_calls = sum(r['embed_calls'] for r in results)
    total_chunks = sum(r['chunks_retrieved_total'] for r in results)
    
    avg_llm_per_query = total_llm_calls / total_files if total_files > 0 else 0
    avg_embed_per_query = total_embed_calls / total_files if total_files > 0 else 0
    avg_chunks = total_chunks / total_files if total_files > 0 else 0
    
    # Iteration statistics
    iteration_counts = [r['num_iterations'] for r in judge_results if r.get('num_iterations', 0) > 0]
    iteration_distribution = Counter(iteration_counts)
    
    if judge_results:
        total_iterations = sum(r['num_iterations'] for r in judge_results)
        avg_iterations = total_iterations / len(judge_results)
        min_iterations = min(iteration_counts) if iteration_counts else 0
        max_iterations = max(iteration_counts) if iteration_counts else 0
        
        # Verdict analysis
        verdict_acceptable = sum(1 for r in judge_results if r.get('final_verdict') == 'acceptable')
        verdict_cap = sum(1 for r in judge_results if r.get('final_verdict') == 'cap_reached')
        verdict_other = len(judge_results) - verdict_acceptable - verdict_cap
        
        total_acceptable = sum(r['judge_acceptable_count'] for r in judge_results)
        total_insufficient = sum(r['judge_insufficient_count'] for r in judge_results)
    else:
        total_iterations = avg_iterations = min_iterations = max_iterations = 0
        verdict_acceptable = verdict_cap = verdict_other = 0
        total_acceptable = total_insufficient = 0
    
    min_llm = min(r['total_llm_calls'] for r in results) if results else 0
    max_llm = max(r['total_llm_calls'] for r in results) if results else 0
    
    # Build summary sections using docstrings
    header = """
╔══════════════════════════════════════════════════════════════╗
║     ANSWER JUDGE + QUERY MODIFIER LLM CALL ANALYSIS         ║
╚══════════════════════════════════════════════════════════════╝
"""
    
    overall_section = f"""
📊 Overall Statistics:
├─ Total log files analyzed: {total_files}
│  ├─ Answer judge-modifier mode: {len(judge_results)}
│  └─ Other modes: {len(other_results)}
├─ Total LLM calls: {total_llm_calls}
│  ├─ Answer judge calls: {total_judge}
│  ├─ Query modifier calls: {total_modifier}
│  └─ Agent 2 (answerer) calls: {total_agent2}
├─ LLM calls ignored (Agent 1/1b): {total_ignored}
└─ Total embedding calls: {total_embed_calls}
"""
    
    averages_section = f"""
📈 Per-Query Averages:
├─ Average LLM calls per query: {avg_llm_per_query:.2f}
├─ Average embedding calls per query: {avg_embed_per_query:.2f}
├─ Average iterations per query: {avg_iterations:.2f}
└─ Average chunks retrieved per query: {avg_chunks:.2f}
"""
    
    ranges_section = f"""
📉 Range:
├─ Minimum LLM calls in a query: {min_llm}
└─ Maximum LLM calls in a query: {max_llm}
"""
    
    if judge_results:
        iter_section = f"""
🔄 ITERATIVE ANSWER EVALUATION ANALYSIS ({len(judge_results)} queries):
├─ Total iterations across all queries: {total_iterations}
├─ Average iterations per query: {avg_iterations:.2f}
├─ Min/Max iterations: {min_iterations}/{max_iterations}

📈 Iteration Distribution:
"""
        for count in sorted(iteration_distribution.keys()):
            freq = iteration_distribution[count]
            percentage = (freq / len(judge_results)) * 100
            bar = "█" * int(percentage / 2)
            iter_section += f"   ├─ {count} iteration(s): {freq} queries ({percentage:.1f}%) {bar}\n"
        
        iter_section += f"""
⚖️ Answer Judge Verdict Analysis:
├─ Total "acceptable" verdicts: {total_acceptable}
├─ Total "insufficient" verdicts: {total_insufficient}
├─ Total query modifications: {total_modifier}
└─ Final outcomes:
   ├─ Accepted answer: {verdict_acceptable} queries ({verdict_acceptable/len(judge_results)*100:.1f}%)
   ├─ Stopped at iteration cap: {verdict_cap} queries ({verdict_cap/len(judge_results)*100:.1f}%)
   └─ Other: {verdict_other} queries ({verdict_other/len(judge_results)*100:.1f}%)
"""
    else:
        iter_section = ""
    
    pipeline_doc = """
💡 Pipeline Pattern (Answer Judge + Query Modifier):
   └─ For each query:
      For each iteration (up to MAX_ANSWER_JUDGE_ITERATIONS):
         1. Embed current query (1 embedding)
         2. Vector search TextChunk nodes
         3. Agent 2 generates answer from context (1 LLM call)
         4. Answer Judge evaluates the answer (1 LLM call)
            ├─ If acceptable → DONE (return this answer)
            └─ If insufficient → Query Modifier rewrites query (1 LLM call) → next iteration
      At MAX_ANSWER_JUDGE_ITERATIONS cap: return last answer
   └─ Expected LLM calls: N (agents2) + N (judges) + (N-1 or fewer) (modifiers) = 2N to 3N-1
   └─ Key difference: generates answer each iteration (unlike context judge)
   └─ Agent 1/1b present but calls excluded from LLM count per user request
"""
    
    agent_breakdown = f"""
🔍 Agent Call Breakdown (across all queries):
├─ Answer Judge (evaluates answers): {total_judge}
├─ Query Modifier (refines query): {total_modifier}
├─ Agent 2 (answerer, per iteration): {total_agent2}
├─ Agent 1/1b (ignored): {total_ignored}
└─ Total LLM calls counted: {total_llm_calls}
"""
    
    top_queries_section = """
📌 TOP QUERIES BY ITERATION COUNT:
"""
    sorted_by_iters = sorted(judge_results, key=lambda x: x['num_iterations'], reverse=True)
    for i, r in enumerate(sorted_by_iters[:5], 1):
        query_preview = r['query'][:60] + "..." if len(r['query']) > 60 else r['query']
        top_queries_section += f"{i}. {r['num_iterations']} iterations | {r.get('final_verdict')} | {r['total_llm_calls']} LLM calls\n"
        top_queries_section += f"   {query_preview}\n"
    
    top_llm_section = """
📌 TOP QUERIES BY TOTAL LLM CALLS:
"""
    sorted_by_calls = sorted(results, key=lambda x: x['total_llm_calls'], reverse=True)
    for i, r in enumerate(sorted_by_calls[:5], 1):
        query_preview = r['query'][:60] + "..." if len(r['query']) > 60 else r['query']
        iter_info = f"{r['num_iterations']} iters" if r.get('num_iterations') else "N/A"
        top_llm_section += f"{i}. {r['total_llm_calls']} calls | {iter_info}\n"
        top_llm_section += f"   {query_preview}\n"
    
    comparison_doc = """
📊 COMPARISON TO OTHER METHODS:
└─ Answer Judge + Query Modifier characteristics:
   ├─ Evaluates answer quality (not just context)
   ├─ Generates new answer each iteration with refined query
   ├─ Uses naive RAG (vector search) for each iteration
   ├─ No graph traversal or triple matching
   ├─ Judge focuses on answer sufficiency vs user query
   ├─ Higher LLM cost due to answering each iteration
   ├─ Cost: 2N to 3N-1 LLM calls where N = iterations
   └─ Best for queries requiring precise answer targeting
"""
    
    # Queries that hit cap
    cap_queries = [r for r in judge_results if r.get('final_verdict') == 'cap_reached']
    cap_section = ""
    if cap_queries:
        cap_section = f"""
⚠️  QUERIES THAT HIT ITERATION CAP ({len(cap_queries)} queries):
"""
        for r in cap_queries[:5]:
            query_preview = r['query'][:55] + "..." if len(r['query']) > 55 else r['query']
            cap_section += f"- {r['log_file']}: {r['num_iterations']} iterations\n"
            cap_section += f"  {query_preview}\n"
            cap_section += f"  Judge: {r['judge_insufficient_count']} insufficient verdict(s)\n"
    
    # First-try acceptable
    first_try = [r for r in judge_results if r.get('num_iterations') == 1 and r.get('judge_acceptable_count') > 0]
    first_try_section = ""
    if first_try:
        first_try_section = f"""
✅ QUERIES WITH ACCEPTABLE ANSWER IN FIRST ITERATION ({len(first_try)} queries):
Total: {len(first_try)} queries ({len(first_try)/len(judge_results)*100:.1f}% of judge queries)
"""
        for r in first_try[:5]:
            query_preview = r['query'][:55] + "..." if len(r['query']) > 55 else r['query']
            first_try_section += f"- {query_preview}\n"
    
    notes_doc = """
📝 Notes:
   - Answer Judge evaluates if the generated answer addresses the query
   - Query Modifier rewrites query when answer is insufficient
   - Hard cap on iterations: MAX_ANSWER_JUDGE_ITERATIONS = 4 (configurable)
   - At cap: last answer returned regardless of quality
   - Pattern: retrieve → answer → judge → (accept OR modify → next iteration)
   - Key difference from Context Judge: answer generated EVERY iteration
   - Agent 1/1b used for entity/triple extraction but calls not counted
   - Embedding calls: typically N (one per iteration)
   - LLM calls: 2N (answer+judge) to 3N-1 (answer+judge+modifier except last)
   - Chunks retrieved reported as sum across all iterations
"""
    
    summary = (
        header + overall_section + averages_section + ranges_section + 
        iter_section + pipeline_doc + agent_breakdown + 
        top_queries_section + top_llm_section + cap_section + 
        first_try_section + comparison_doc + notes_doc
    )
    
    return summary


def main():
    """Main execution function."""
    
    separator = "=" * 70
    title = "ANSWER JUDGE + QUERY MODIFIER LLM CALL ANALYZER"
    subtitle = "(Tracking iterative answer evaluation and query refinement)"
    
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