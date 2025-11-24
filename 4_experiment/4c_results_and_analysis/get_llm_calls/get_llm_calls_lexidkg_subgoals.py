# analyze_llm_calls_agentic.py
"""
Analyzes log files from the agentic GraphRAG with subgoals (Agent 0-3).
Tracks LLM calls per query including detailed subgoal decomposition and aggregation.
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
LOG_FOLDER = SCRIPT_DIR / "../../4b_retrieval/4b_i_lexidkg_graphrag/question_terminal_logs/no_4_lexidkg_5_subgoals_new_fix"
OUTPUT_FOLDER = SCRIPT_DIR / "llm_calls_analysis_results_lexidkg_5_subgoals"


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


def extract_subgoals_from_log(log_path: Path) -> Tuple[List[str], bool]:
    """
    Extract the list of subgoals from Agent 0 output.
    Returns: (subgoals_list, decomposition_needed)
    """
    subgoals = []
    decomposition_needed = None
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
            # Look for decomposition_needed flag
            decomp_match = re.search(r'Decomposition needed\?\s*(True|False)', content)
            if decomp_match:
                decomposition_needed = decomp_match.group(1) == 'True'
            
            # Look for Agent 0 final subgoal list
            match = re.search(r'\[Agent 0\] Final subgoal list \(used\):\s*(\[.*?\])', content, re.DOTALL)
            if match:
                try:
                    subgoals = eval(match.group(1))  # Safe since we control the log format
                except Exception:
                    pass
    except Exception as e:
        print(f"Error extracting subgoals from {log_path.name}: {e}")
    return subgoals, decomposition_needed


def analyze_log_file(log_path: Path) -> Dict:
    """
    Analyze a single log file and count LLM calls.
    
    Returns dict with comprehensive metrics including subgoal details.
    """
    
    result = {
        'log_file': log_path.name,
        'query': '',
        'num_subgoals': 0,
        'subgoals': [],
        'decomposition_needed': None,
        'agent0_calls': 0,
        'agent1_calls': 0,
        'agent1b_calls': 0,
        'agent2_calls': 0,
        'agent3_calls': 0,
        'llm_json_calls': 0,
        'llm_text_calls': 0,
        'total_llm_calls': 0,
        'embed_calls': 0,
        'total_runtime_ms': 0.0,
        'parallel_time_ms': 0.0,
        'agent0_time_ms': 0.0,
        'agent3_time_ms': 0.0,
        'mode': 'unknown'
    }
    
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
            # Extract query
            result['query'] = extract_query_from_log(log_path)
            
            # Detect mode (single-pass vs agentic)
            if "Mode: Agentic (Agent 0 → parallel Agents 1 & 2 → Agent 3)" in content:
                result['mode'] = 'agentic'
            elif "Mode: Single-pass (Agents 1 & 2 only)" in content:
                result['mode'] = 'single-pass'
            
            # Extract subgoals
            subgoals, decomp_needed = extract_subgoals_from_log(log_path)
            result['subgoals'] = subgoals
            result['num_subgoals'] = len(subgoals)
            result['decomposition_needed'] = decomp_needed
            
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
            agent0_matches = re.findall(r'\[Agent 0\] Prompt:', content)
            result['agent0_calls'] = len(agent0_matches)
            
            agent1_matches = re.findall(r'\[Agent 1\] Prompt:', content)
            result['agent1_calls'] = len(agent1_matches)
            
            agent1b_matches = re.findall(r'\[Agent 1b\] Prompt:', content)
            result['agent1b_calls'] = len(agent1b_matches)
            
            agent2_matches = re.findall(r'\[Agent 2\] Prompt:', content)
            result['agent2_calls'] = len(agent2_matches)
            
            agent3_matches = re.findall(r'\[Agent 3\] Prompt:', content)
            result['agent3_calls'] = len(agent3_matches)
            
            # Extract timing information
            runtime_match = re.search(r'Total runtime:\s*([\d.]+)\s*ms', content)
            if runtime_match:
                result['total_runtime_ms'] = float(runtime_match.group(1))
            
            parallel_match = re.search(r'Parallel subgoals time:\s*([\d.]+)\s*ms', content)
            if parallel_match:
                result['parallel_time_ms'] = float(parallel_match.group(1))
            
            agent0_match = re.search(r'Agent 0 time:\s*([\d.]+)\s*ms', content)
            if agent0_match:
                result['agent0_time_ms'] = float(agent0_match.group(1))
            
            agent3_match = re.search(r'Aggregator time:\s*([\d.]+)\s*ms', content)
            if agent3_match:
                result['agent3_time_ms'] = float(agent3_match.group(1))
                
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
        'log_file', 'query', 'mode', 'num_subgoals', 'decomposition_needed',
        'total_llm_calls', 'llm_json_calls', 'llm_text_calls',
        'agent0_calls', 'agent1_calls', 'agent1b_calls', 'agent2_calls', 'agent3_calls',
        'embed_calls', 'total_runtime_ms', 'agent0_time_ms', 'parallel_time_ms', 'agent3_time_ms'
    ]
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\n✓ Detailed results saved to: {csv_path}")
    
    # 2. Save full JSON (includes subgoals array)
    json_path = output_folder / f"llm_calls_detailed_{timestamp}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"✓ JSON results saved to: {json_path}")
    
    # 3. Save subgoals breakdown (NEW)
    subgoals_path = output_folder / f"subgoals_breakdown_{timestamp}.json"
    subgoals_data = []
    for r in results:
        if r.get('subgoals'):
            subgoals_data.append({
                'log_file': r['log_file'],
                'query': r['query'],
                'mode': r['mode'],
                'num_subgoals': r['num_subgoals'],
                'decomposition_needed': r.get('decomposition_needed'),
                'subgoals': r['subgoals']
            })
    
    with open(subgoals_path, 'w', encoding='utf-8') as f:
        json.dump(subgoals_data, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Subgoals breakdown saved to: {subgoals_path}")
    
    # 4. Save human-readable subgoals list (NEW)
    subgoals_txt_path = output_folder / f"subgoals_readable_{timestamp}.txt"
    with open(subgoals_txt_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("SUBGOALS BREAKDOWN - HUMAN READABLE\n")
        f.write("=" * 80 + "\n")
        
        for i, r in enumerate(results, 1):
            if r.get('subgoals'):
                f.write(f"[{i}] {r['log_file']}\n")
                f.write(f"Original Query: {r['query']}\n")
                f.write(f"Mode: {r['mode']} | Subgoals: {r['num_subgoals']} | Decomposition: {r.get('decomposition_needed')}\n")
                f.write("-" * 80 + "\n")
                for j, sg in enumerate(r['subgoals'], 1):
                    f.write(f"  Subgoal {j}: {sg}\n")
                f.write("\n")
    
    print(f"✓ Human-readable subgoals saved to: {subgoals_txt_path}")
    
    # 5. Create summary statistics
    summary = create_summary(results)
    summary_path = output_folder / f"llm_calls_summary_{timestamp}.txt"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(summary)
    
    print(f"✓ Summary saved to: {summary_path}")
    print("\n" + summary)


def create_summary(results: List[Dict]) -> str:
    """Create a text summary of the analysis with enhanced subgoal statistics."""
    
    if not results:
        return "No results to summarize."
    
    total_files = len(results)
    
    # Separate by mode
    agentic_results = [r for r in results if r['mode'] == 'agentic']
    single_pass_results = [r for r in results if r['mode'] == 'single-pass']
    
    # Overall stats
    total_llm_calls = sum(r['total_llm_calls'] for r in results)
    total_json_calls = sum(r['llm_json_calls'] for r in results)
    total_text_calls = sum(r['llm_text_calls'] for r in results)
    total_embed_calls = sum(r['embed_calls'] for r in results)
    
    avg_llm_per_query = total_llm_calls / total_files if total_files > 0 else 0
    avg_embed_per_query = total_embed_calls / total_files if total_files > 0 else 0
    
    # Subgoal statistics (ENHANCED)
    subgoal_counts = [r['num_subgoals'] for r in agentic_results if r.get('num_subgoals', 0) > 0]
    subgoal_distribution = Counter(subgoal_counts)
    
    if agentic_results:
        total_subgoals = sum(r['num_subgoals'] for r in agentic_results)
        avg_subgoals = total_subgoals / len(agentic_results) if len(agentic_results) > 0 else 0
        min_subgoals = min(r['num_subgoals'] for r in agentic_results)
        max_subgoals = max(r['num_subgoals'] for r in agentic_results)
        
        total_agent0 = sum(r['agent0_calls'] for r in agentic_results)
        total_agent3 = sum(r['agent3_calls'] for r in agentic_results)
        avg_agent0_time = sum(r['agent0_time_ms'] for r in agentic_results) / len(agentic_results)
        avg_parallel_time = sum(r['parallel_time_ms'] for r in agentic_results) / len(agentic_results)
        avg_agent3_time = sum(r['agent3_time_ms'] for r in agentic_results) / len(agentic_results)
        
        # Decomposition analysis
        decomp_true = sum(1 for r in agentic_results if r.get('decomposition_needed') == True)
        decomp_false = sum(1 for r in agentic_results if r.get('decomposition_needed') == False)
        decomp_unknown = len(agentic_results) - decomp_true - decomp_false
    else:
        total_subgoals = avg_subgoals = min_subgoals = max_subgoals = 0
        total_agent0 = total_agent3 = 0
        avg_agent0_time = avg_parallel_time = avg_agent3_time = 0
        decomp_true = decomp_false = decomp_unknown = 0
    
    total_runtime = sum(r['total_runtime_ms'] for r in results)
    avg_runtime = total_runtime / total_files if total_files > 0 else 0
    
    min_llm = min(r['total_llm_calls'] for r in results)
    max_llm = max(r['total_llm_calls'] for r in results)
    
    summary = f"""
╔══════════════════════════════════════════════════════════════╗
║      AGENTIC GRAPHRAG LLM CALL ANALYSIS SUMMARY              ║
╚══════════════════════════════════════════════════════════════╝

📊 Overall Statistics:
├─ Total log files analyzed: {total_files}
│  ├─ Agentic mode (with subgoals): {len(agentic_results)}
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
    
    if agentic_results:
        summary += f"""
🔀 SUBGOAL ANALYSIS ({len(agentic_results)} agentic queries):
╔══════════════════════════════════════════════════════════════╗
║                    SUBGOAL STATISTICS                        ║
╚══════════════════════════════════════════════════════════════╝

📊 Subgoal Counts:
├─ Total subgoals generated: {total_subgoals}
├─ Average subgoals per query: {avg_subgoals:.2f}
├─ Minimum subgoals: {min_subgoals}
└─ Maximum subgoals: {max_subgoals}

📈 Subgoal Distribution:
"""
        # Add distribution histogram
        for count in sorted(subgoal_distribution.keys()):
            freq = subgoal_distribution[count]
            percentage = (freq / len(agentic_results)) * 100
            bar = "█" * int(percentage / 2)  # Scale bar to reasonable length
            summary += f"├─ {count} subgoal(s): {freq} queries ({percentage:.1f}%) {bar}\n"
        
        summary += f"""
🎯 Decomposition Decisions:
├─ Decomposition needed (True): {decomp_true} queries
├─ No decomposition needed (False): {decomp_false} queries
└─ Unknown/Not recorded: {decomp_unknown} queries

🤖 Agent Statistics (Agentic Mode):
├─ Agent 0 (subgoal generator) calls: {total_agent0}
├─ Agent 3 (aggregator) calls: {total_agent3}
└─ Average timing breakdown:
   ├─ Agent 0 (decomposition): {avg_agent0_time:.2f} ms
   ├─ Parallel subgoals: {avg_parallel_time:.2f} ms
   └─ Agent 3 (aggregation): {avg_agent3_time:.2f} ms
"""
    
    # Agent breakdown
    total_agent1 = sum(r['agent1_calls'] for r in results)
    total_agent1b = sum(r['agent1b_calls'] for r in results)
    total_agent2 = sum(r['agent2_calls'] for r in results)
    
    summary += f"""
🔍 Agent Call Breakdown (across all queries):
├─ Agent 0 (subgoal generator): {total_agent0}
├─ Agent 1 (entity extraction): {total_agent1}
├─ Agent 1b (triple extraction): {total_agent1b}
├─ Agent 2 (answer generation): {total_agent2}
└─ Agent 3 (aggregator): {total_agent3}

💡 Notes:
   - In agentic mode: 1 Agent 0 + (Agents 1,1b,2 × num_subgoals) + 1 Agent 3
   - In single-pass mode: 1 Agent 1 + 1 Agent 1b + 1 Agent 2
   - Subgoal limit (hardcoded): SUBGOAL_MAX_COUNT = 2
   - Embedding calls vary based on entities, triples, and chunks
"""
    
    # Top queries by different metrics
    if agentic_results:
        summary += "\n" + "=" * 70 + "\n"
        summary += "📌 TOP QUERIES BY SUBGOAL COUNT:\n"
        summary += "=" * 70 + "\n"
        sorted_by_subgoals = sorted(agentic_results, key=lambda x: x['num_subgoals'], reverse=True)
        for i, r in enumerate(sorted_by_subgoals[:10], 1):  # Show top 10
            query_preview = r['query'][:65] + "..." if len(r['query']) > 65 else r['query']
            summary += f"\n[{i}] {r['num_subgoals']} subgoals | {r['log_file']}\n"
            summary += f"    Query: {query_preview}\n"
            if r.get('subgoals'):
                summary += f"    Subgoals:\n"
                for j, sg in enumerate(r['subgoals'], 1):
                    sg_preview = sg[:60] + "..." if len(sg) > 60 else sg
                    summary += f"      {j}. {sg_preview}\n"
    
    summary += "\n" + "=" * 70 + "\n"
    summary += "📌 TOP QUERIES BY TOTAL LLM CALLS:\n"
    summary += "=" * 70 + "\n"
    sorted_by_calls = sorted(results, key=lambda x: x['total_llm_calls'], reverse=True)
    for i, r in enumerate(sorted_by_calls[:5], 1):
        query_preview = r['query'][:65] + "..." if len(r['query']) > 65 else r['query']
        summary += f"{i}. {r['total_llm_calls']} calls | {r['num_subgoals']} subgoals | {r['mode']}\n"
        summary += f"   {query_preview}\n"
    
    summary += "\n" + "=" * 70 + "\n"
    summary += "📌 TOP QUERIES BY EMBEDDING CALLS:\n"
    summary += "=" * 70 + "\n"
    sorted_by_embeds = sorted(results, key=lambda x: x['embed_calls'], reverse=True)
    for i, r in enumerate(sorted_by_embeds[:5], 1):
        query_preview = r['query'][:65] + "..." if len(r['query']) > 65 else r['query']
        summary += f"{i}. {r['embed_calls']} embeds | {r['num_subgoals']} subgoals | {r['mode']}\n"
        summary += f"   {query_preview}\n"
    
    return summary


def main():
    """Main execution function."""
    
    print("=" * 70)
    print("AGENTIC GRAPHRAG LLM CALL ANALYZER")
    print("(With Detailed Subgoal Tracking)")
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