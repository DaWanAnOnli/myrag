# analyze_llm_calls_judge.py
"""
Analyzes log files from the Context Judge + Query Modifier GraphRAG.
Tracks LLM calls per query including iterative judge/modify loop and context assessment.
"""

import re
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import csv
from collections import Counter

# Relative paths from this script's location
SCRIPT_DIR = Path(__file__).resolve().parent
LOG_FOLDER = SCRIPT_DIR / "../../4b_retrieval/4b_i_lexidkg_graphrag/question_terminal_logs/lexidkg_5_context_judge_5_hop_1250"
OUTPUT_FOLDER = SCRIPT_DIR / "llm_calls_analysis_results_lexidkg_5_context_judge_5_hop_1250"


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


def extract_iterations_info(log_path: Path) -> Tuple[int, List[Dict[str, any]]]:
    """
    Extract iteration count and details from the log.
    Returns: (iterations_used, iteration_details_list)
    """
    iterations_used = 0
    iteration_details = []
    
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
            # Extract iterations used from summary
            iter_match = re.search(r'Iterations used:\s*(\d+)', content)
            if iter_match:
                iterations_used = int(iter_match.group(1))
            
            # Extract iteration markers
            iter_markers = re.findall(r'--- Iteration (\d+)/(\d+) (START|DONE)', content)
            if iter_markers:
                max_iter = max(int(m[0]) for m in iter_markers if m[2] == 'START')
                if iterations_used == 0:
                    iterations_used = max_iter
            
            # Extract judge verdicts
            judge_sufficient = re.findall(r'\[Agent CJ\] Verdict: sufficient=(True|False)', content)
            
            # Extract modified queries
            modified_queries = re.findall(r'\[Agent QM\] Modified query:\s*\'(.+?)\'', content)
            
            # Build iteration details
            for i in range(1, iterations_used + 1):
                detail = {
                    'iteration': i,
                    'judge_verdict': None,
                    'modified_query': None,
                    'answered': False
                }
                
                if i <= len(judge_sufficient):
                    detail['judge_verdict'] = judge_sufficient[i-1] == 'True'
                
                if i <= len(modified_queries):
                    detail['modified_query'] = modified_queries[i-1]
                
                # Check if iteration ended with answer
                if i == iterations_used or (detail['judge_verdict'] == True):
                    detail['answered'] = True
                
                iteration_details.append(detail)
    
    except Exception as e:
        print(f"Error extracting iterations from {log_path.name}: {e}")
    
    return iterations_used, iteration_details


def analyze_log_file(log_path: Path) -> Dict:
    """
    Analyze a single log file and count LLM calls.
    
    Returns dict with comprehensive metrics including iteration details.
    """
    
    result = {
        'log_file': log_path.name,
        'query': '',
        'iterations_used': 0,
        'iteration_details': [],
        'judge_sufficient_count': 0,
        'judge_insufficient_count': 0,
        'query_modifications': 0,
        'agent_cj_calls': 0,
        'agent_qm_calls': 0,
        'agent1_calls': 0,
        'agent1b_calls': 0,
        'agent2_calls': 0,
        'llm_json_calls': 0,
        'llm_text_calls': 0,
        'total_llm_calls': 0,
        'embed_calls': 0,
        'total_runtime_ms': 0.0,
        'mode': 'unknown',
        'final_verdict': None  # True=sufficient, False=cap reached, None=unknown
    }
    
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
            # Extract query
            result['query'] = extract_query_from_log(log_path)
            
            # Detect mode
            if "Mode: Judge loop (retrieve → judge → modify or answer)" in content:
                result['mode'] = 'judge-loop'
            elif "Mode: Single-pass (Agents 1 & 2 only)" in content:
                result['mode'] = 'single-pass'
            
            # Extract iterations info
            iters, details = extract_iterations_info(log_path)
            result['iterations_used'] = iters
            result['iteration_details'] = details
            
            # Count LLM JSON calls
            llm_json_matches = re.findall(r'\[LLM JSON\] call completed', content)
            result['llm_json_calls'] = len(llm_json_matches)
            
            # Count LLM TEXT calls
            llm_text_matches = re.findall(r'\[LLM TEXT\] call completed', content)
            result['llm_text_calls'] = len(llm_text_matches)
            
            # Total LLM calls
            result['total_llm_calls'] = result['llm_json_calls'] + result['llm_text_calls']
            
            # Count embedding calls
            embed_matches = re.findall(r'\[Embed\] text_len=', content)
            result['embed_calls'] = len(embed_matches)
            
            # Count Agent-specific prompts
            agent_cj_matches = re.findall(r'\[Agent CJ\] Prompt:', content)
            result['agent_cj_calls'] = len(agent_cj_matches)
            
            agent_qm_matches = re.findall(r'\[Agent QM\] Prompt:', content)
            result['agent_qm_calls'] = len(agent_qm_matches)
            
            agent1_matches = re.findall(r'\[Agent 1\] Prompt:', content)
            result['agent1_calls'] = len(agent1_matches)
            
            agent1b_matches = re.findall(r'\[Agent 1b\] Prompt:', content)
            result['agent1b_calls'] = len(agent1b_matches)
            
            agent2_matches = re.findall(r'\[Agent 2\] Prompt:', content)
            result['agent2_calls'] = len(agent2_matches)
            
            # Count judge verdicts
            judge_sufficient = re.findall(r'\[Agent CJ\] Verdict: sufficient=True', content)
            judge_insufficient = re.findall(r'\[Agent CJ\] Verdict: sufficient=False', content)
            result['judge_sufficient_count'] = len(judge_sufficient)
            result['judge_insufficient_count'] = len(judge_insufficient)
            
            # Count query modifications
            result['query_modifications'] = result['agent_qm_calls']
            
            # Determine final verdict
            if "At iteration cap" in content:
                result['final_verdict'] = 'cap_reached'
            elif result['judge_sufficient_count'] > 0:
                result['final_verdict'] = 'sufficient'
            elif result['mode'] == 'single-pass':
                result['final_verdict'] = 'single_pass'
            
            # Extract timing information
            runtime_match = re.search(r'Total runtime:\s*([\d.]+)\s*ms', content)
            if runtime_match:
                result['total_runtime_ms'] = float(runtime_match.group(1))
                
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
        'iterations_used', 'final_verdict',
        'judge_sufficient_count', 'judge_insufficient_count', 'query_modifications',
        'total_llm_calls', 'llm_json_calls', 'llm_text_calls',
        'agent_cj_calls', 'agent_qm_calls', 'agent1_calls', 'agent1b_calls', 'agent2_calls',
        'embed_calls', 'total_runtime_ms'
    ]
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\n✓ Detailed results saved to: {csv_path}")
    
    # 2. Save full JSON (includes iteration details)
    json_path = output_folder / f"llm_calls_detailed_{timestamp}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"✓ JSON results saved to: {json_path}")
    
    # 3. Save iterations breakdown (NEW)
    iter_path = output_folder / f"iterations_breakdown_{timestamp}.json"
    iter_data = []
    for r in results:
        if r.get('iteration_details'):
            iter_data.append({
                'log_file': r['log_file'],
                'query': r['query'],
                'mode': r['mode'],
                'iterations_used': r['iterations_used'],
                'final_verdict': r.get('final_verdict'),
                'iteration_details': r['iteration_details']
            })
    
    with open(iter_path, 'w', encoding='utf-8') as f:
        json.dump(iter_data, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Iterations breakdown saved to: {iter_path}")
    
    # 4. Save human-readable iterations report (NEW)
    iter_txt_path = output_folder / f"iterations_readable_{timestamp}.txt"
    with open(iter_txt_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("JUDGE LOOP ITERATIONS BREAKDOWN - HUMAN READABLE\n")
        f.write("=" * 80 + "\n")
        
        for i, r in enumerate(results, 1):
            if r.get('iteration_details'):
                f.write(f"[{i}] {r['log_file']}\n")
                f.write(f"Original Query: {r['query']}\n")
                f.write(f"Mode: {r['mode']} | Iterations: {r['iterations_used']} | Final: {r.get('final_verdict')}\n")
                f.write("-" * 80 + "\n")
                
                for detail in r['iteration_details']:
                    iter_num = detail['iteration']
                    verdict = detail.get('judge_verdict')
                    verdict_str = "SUFFICIENT" if verdict == True else "INSUFFICIENT" if verdict == False else "N/A"
                    
                    f.write(f"  Iteration {iter_num}:\n")
                    f.write(f"    Judge verdict: {verdict_str}\n")
                    
                    if detail.get('modified_query'):
                        f.write(f"    Modified query: {detail['modified_query'][:70]}...\n" 
                               if len(detail['modified_query']) > 70 
                               else f"    Modified query: {detail['modified_query']}\n")
                    
                    if detail.get('answered'):
                        f.write(f"    → Answered in this iteration\n")
                    else:
                        f.write(f"    → Continued to next iteration\n")
                
                f.write("\n")
    
    print(f"✓ Human-readable iterations saved to: {iter_txt_path}")
    
    # 5. Create summary statistics
    summary = create_summary(results)
    summary_path = output_folder / f"llm_calls_summary_{timestamp}.txt"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(summary)
    
    print(f"✓ Summary saved to: {summary_path}")
    print("\n" + summary)


def create_summary(results: List[Dict]) -> str:
    """Create a text summary of the analysis with enhanced iteration statistics."""
    
    if not results:
        return "No results to summarize."
    
    total_files = len(results)
    
    # Separate by mode
    judge_results = [r for r in results if r['mode'] == 'judge-loop']
    single_pass_results = [r for r in results if r['mode'] == 'single-pass']
    
    # Overall stats
    total_llm_calls = sum(r['total_llm_calls'] for r in results)
    total_json_calls = sum(r['llm_json_calls'] for r in results)
    total_text_calls = sum(r['llm_text_calls'] for r in results)
    total_embed_calls = sum(r['embed_calls'] for r in results)
    
    avg_llm_per_query = total_llm_calls / total_files if total_files > 0 else 0
    avg_embed_per_query = total_embed_calls / total_files if total_files > 0 else 0
    
    # Iteration statistics (ENHANCED)
    iteration_counts = [r['iterations_used'] for r in judge_results if r.get('iterations_used', 0) > 0]
    iteration_distribution = Counter(iteration_counts)
    
    if judge_results:
        total_iterations = sum(r['iterations_used'] for r in judge_results)
        avg_iterations = total_iterations / len(judge_results) if len(judge_results) > 0 else 0
        min_iterations = min(r['iterations_used'] for r in judge_results) if judge_results else 0
        max_iterations = max(r['iterations_used'] for r in judge_results) if judge_results else 0
        
        total_agent_cj = sum(r['agent_cj_calls'] for r in judge_results)
        total_agent_qm = sum(r['agent_qm_calls'] for r in judge_results)
        
        # Verdict analysis
        verdict_sufficient = sum(1 for r in judge_results if r.get('final_verdict') == 'sufficient')
        verdict_cap = sum(1 for r in judge_results if r.get('final_verdict') == 'cap_reached')
        verdict_other = len(judge_results) - verdict_sufficient - verdict_cap
        
        # Total judge verdicts
        total_sufficient_verdicts = sum(r['judge_sufficient_count'] for r in judge_results)
        total_insufficient_verdicts = sum(r['judge_insufficient_count'] for r in judge_results)
        total_modifications = sum(r['query_modifications'] for r in judge_results)
        
        # Average modifications per query
        avg_modifications = total_modifications / len(judge_results) if len(judge_results) > 0 else 0
    else:
        total_iterations = avg_iterations = min_iterations = max_iterations = 0
        total_agent_cj = total_agent_qm = 0
        verdict_sufficient = verdict_cap = verdict_other = 0
        total_sufficient_verdicts = total_insufficient_verdicts = total_modifications = 0
        avg_modifications = 0
    
    total_runtime = sum(r['total_runtime_ms'] for r in results)
    avg_runtime = total_runtime / total_files if total_files > 0 else 0
    
    min_llm = min(r['total_llm_calls'] for r in results)
    max_llm = max(r['total_llm_calls'] for r in results)
    
    summary = f"""
╔══════════════════════════════════════════════════════════════╗
║   CONTEXT JUDGE + QUERY MODIFIER LLM CALL ANALYSIS SUMMARY   ║
╚══════════════════════════════════════════════════════════════╝

📊 Overall Statistics:
├─ Total log files analyzed: {total_files}
│  ├─ Judge loop mode: {len(judge_results)}
│  └─ Single-pass mode: {len(single_pass_results)}
├─ Total LLM calls (all queries): {total_llm_calls}
│  ├─ JSON generation calls: {total_json_calls}
│  └─ Text generation calls: {total_text_calls}
└─ Total embedding calls: {total_embed_calls}

📈 Per-Query Averages:
├─ Average LLM calls per query: {avg_llm_per_query:.2f}
├─ Average embedding calls per query: {avg_embed_per_query:.2f}
└─ Average runtime per query: {avg_runtime:.2f} ms ({avg_runtime/1000:.2f} sec)

📉 Range:
├─ Minimum LLM calls in a query: {min_llm}
└─ Maximum LLM calls in a query: {max_llm}

⏱️  Total Runtime:
└─ All queries combined: {total_runtime:.2f} ms ({total_runtime/1000:.2f} sec)
"""
    
    if judge_results:
        summary += f"""
🔄 ITERATIVE JUDGE LOOP ANALYSIS ({len(judge_results)} queries):
╔══════════════════════════════════════════════════════════════╗
║                  ITERATION STATISTICS                         ║
╚══════════════════════════════════════════════════════════════╝

📊 Iteration Counts:
├─ Total iterations across all queries: {total_iterations}
├─ Average iterations per query: {avg_iterations:.2f}
├─ Min/Max iterations: {min_iterations}/{max_iterations}
└─ Average query modifications: {avg_modifications:.2f}

📈 Iteration Distribution:
"""
        # Add distribution histogram
        for count in sorted(iteration_distribution.keys()):
            freq = iteration_distribution[count]
            percentage = (freq / len(judge_results)) * 100
            bar = "█" * int(percentage / 2)
            summary += f"├─ {count} iteration(s): {freq} queries ({percentage:.1f}%) {bar}\n"
        
        summary += f"""
⚖️ Judge Verdict Analysis:
├─ Total "sufficient" verdicts: {total_sufficient_verdicts}
├─ Total "insufficient" verdicts: {total_insufficient_verdicts}
├─ Total query modifications: {total_modifications}
└─ Final outcomes:
   ├─ Answered with sufficient context: {verdict_sufficient} queries ({verdict_sufficient/len(judge_results)*100:.1f}%)
   ├─ Answered at iteration cap: {verdict_cap} queries ({verdict_cap/len(judge_results)*100:.1f}%)
   └─ Other: {verdict_other} queries ({verdict_other/len(judge_results)*100:.1f}%)

🤖 Agent Statistics (Judge Loop Mode):
├─ Agent CJ (context judge) calls: {total_agent_cj}
├─ Agent QM (query modifier) calls: {total_agent_qm}
└─ Pattern: CJ = QM (for insufficient verdicts), CJ > QM (when sufficient)

💡 Judge Loop Pattern:
   └─ For each iteration: Retrieve → CJ assess context → 
      ├─ If sufficient → Answer (done)
      └─ If insufficient → QM modify query → next iteration
   └─ At MAX_JUDGE_ITERS cap → Answer regardless of verdict
"""
    
    # Agent breakdown
    total_agent1 = sum(r['agent1_calls'] for r in results)
    total_agent1b = sum(r['agent1b_calls'] for r in results)
    total_agent2 = sum(r['agent2_calls'] for r in results)
    total_agent_cj_all = sum(r['agent_cj_calls'] for r in results)
    total_agent_qm_all = sum(r['agent_qm_calls'] for r in results)
    
    summary += f"""
🔍 Agent Call Breakdown (across all queries):
├─ Agent CJ (context judge): {total_agent_cj_all}
├─ Agent QM (query modifier): {total_agent_qm_all}
├─ Agent 1 (entity extraction): {total_agent1}
├─ Agent 1b (triple extraction): {total_agent1b}
└─ Agent 2 (answer generation): {total_agent2}

💡 Notes:
   - In judge loop: N iterations → N retrievals + N CJ calls + (N-1 or fewer) QM calls + 1 answer
   - In single-pass mode: 1 retrieval + 1 answer (no CJ/QM)
   - Iteration limit (hardcoded): MAX_JUDGE_ITERS = 4
   - Embedding calls vary based on entities, triples, chunks, and iterations
"""
    
    # Top queries by different metrics
    if judge_results:
        summary += "\n" + "=" * 70 + "\n"
        summary += "📌 TOP QUERIES BY ITERATION COUNT:\n"
        summary += "=" * 70 + "\n"
        sorted_by_iters = sorted(judge_results, key=lambda x: x['iterations_used'], reverse=True)
        for i, r in enumerate(sorted_by_iters[:10], 1):
            query_preview = r['query'][:55] + "..." if len(r['query']) > 55 else r['query']
            summary += f"\n[{i}] {r['iterations_used']} iterations | {r.get('final_verdict')} | {r['log_file']}\n"
            summary += f"    Query: {query_preview}\n"
            summary += f"    Judge: {r['judge_sufficient_count']} sufficient / {r['judge_insufficient_count']} insufficient\n"
            summary += f"    Modifications: {r['query_modifications']}\n"
    
    summary += "\n" + "=" * 70 + "\n"
    summary += "📌 TOP QUERIES BY TOTAL LLM CALLS:\n"
    summary += "=" * 70 + "\n"
    sorted_by_calls = sorted(results, key=lambda x: x['total_llm_calls'], reverse=True)
    for i, r in enumerate(sorted_by_calls[:5], 1):
        query_preview = r['query'][:55] + "..." if len(r['query']) > 55 else r['query']
        iter_info = f"{r['iterations_used']} iters" if r.get('iterations_used') else "N/A"
        summary += f"{i}. {r['total_llm_calls']} calls | {iter_info} | {r['mode']}\n"
        summary += f"   {query_preview}\n"
    
    summary += "\n" + "=" * 70 + "\n"
    summary += "📌 TOP QUERIES BY EMBEDDING CALLS:\n"
    summary += "=" * 70 + "\n"
    sorted_by_embeds = sorted(results, key=lambda x: x['embed_calls'], reverse=True)
    for i, r in enumerate(sorted_by_embeds[:5], 1):
        query_preview = r['query'][:55] + "..." if len(r['query']) > 55 else r['query']
        iter_info = f"{r['iterations_used']} iters" if r.get('iterations_used') else "N/A"
        summary += f"{i}. {r['embed_calls']} embeds | {iter_info} | {r['mode']}\n"
        summary += f"   {query_preview}\n"
    
    # Queries that hit iteration cap
    cap_queries = [r for r in judge_results if r.get('final_verdict') == 'cap_reached']
    if cap_queries:
        summary += "\n" + "=" * 70 + "\n"
        summary += "⚠️  QUERIES THAT HIT ITERATION CAP:\n"
        summary += "=" * 70 + "\n"
        for r in cap_queries[:5]:
            query_preview = r['query'][:55] + "..." if len(r['query']) > 55 else r['query']
            summary += f"- {r['log_file']}: {r['iterations_used']} iterations (cap)\n"
            summary += f"  Query: {query_preview}\n"
            summary += f"  Judge: {r['judge_insufficient_count']} insufficient verdict(s)\n"
    
    # Queries answered with 1 iteration (context was sufficient immediately)
    first_try_queries = [r for r in judge_results if r.get('iterations_used') == 1 and r.get('judge_sufficient_count') > 0]
    if first_try_queries:
        summary += "\n" + "=" * 70 + "\n"
        summary += "✅ QUERIES ANSWERED IN FIRST ITERATION (sufficient context):\n"
        summary += "=" * 70 + "\n"
        summary += f"Total: {len(first_try_queries)} queries ({len(first_try_queries)/len(judge_results)*100:.1f}% of judge loop queries)\n"
        for r in first_try_queries[:5]:
            query_preview = r['query'][:55] + "..." if len(r['query']) > 55 else r['query']
            summary += f"- {query_preview}\n"
    
    return summary


def main():
    """Main execution function."""
    
    print("=" * 70)
    print("CONTEXT JUDGE + QUERY MODIFIER LLM CALL ANALYZER")
    print("(With Iterative Judge/Modify Loop Tracking)")
    print("=" * 70)
    print(f"\nInput folder: {LOG_FOLDER.resolve()}")
    print(f"Output folder: {OUTPUT_FOLDER.resolve()}")
    print()
    
    # Analyze all log files
    results = analyze_all_logs()
    
    if not results:
        print("\nNo results to save. Exiting.")
        return
    
    # Save results
    save_results(results)
    
    print("\n" + "=" * 70)
    print("✓ Analysis complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()