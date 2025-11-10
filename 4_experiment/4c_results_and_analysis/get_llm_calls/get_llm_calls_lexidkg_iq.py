# analyze_llm_calls_iq.py
"""
Analyzes log files from the Intermediate Questions (IQ) based GraphRAG.
Tracks LLM calls per query including IQ planning, query modification, and sequential execution.
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
LOG_FOLDER = SCRIPT_DIR / "../../4b_retrieval/4b_i_lexidkg_graphrag/question_terminal_logs/lexidkg_3_iq_fix"
OUTPUT_FOLDER = SCRIPT_DIR / "llm_calls_analysis_results_lexidkg_3_iq_fix"


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


def extract_iq_plan_from_log(log_path: Path) -> Tuple[List[str], List[str]]:
    """
    Extract the IQ plan and completed IQs from Agent I0 and execution logs.
    Returns: (planned_iqs, completed_iqs)
    """
    planned_iqs = []
    completed_iqs = []
    
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
            # Look for Agent I0 final IQ plan
            plan_match = re.search(r'\[Agent I0\] Final IQ plan \(used\):\s*(\[.*?\])', content, re.DOTALL)
            if plan_match:
                try:
                    planned_iqs = eval(plan_match.group(1))
                except Exception:
                    pass
            
            # Look for completed IQs from Agent Q output
            completed_matches = re.findall(r'\[Agent Q\] Completed IQ:\s*(.+?)(?:\s+\||\n)', content)
            completed_iqs.extend([m.strip() for m in completed_matches if m.strip()])
            
            # Also look for IQ tags in context
            iq_tags = re.findall(r'\[iq=(\d+)/(\d+)\]', content)
            if iq_tags and not completed_iqs:
                # Fallback: use planned IQs if we can't find completed ones
                max_iq = max(int(t[0]) for t in iq_tags) if iq_tags else 0
                if max_iq > 0 and len(planned_iqs) >= max_iq:
                    completed_iqs = planned_iqs[:max_iq]
    
    except Exception as e:
        print(f"Error extracting IQ plan from {log_path.name}: {e}")
    
    return planned_iqs, completed_iqs


def analyze_log_file(log_path: Path) -> Dict:
    """
    Analyze a single log file and count LLM calls.
    
    Returns dict with comprehensive metrics including IQ details.
    """
    
    result = {
        'log_file': log_path.name,
        'query': '',
        'num_iqs_planned': 0,
        'num_iqs_executed': 0,
        'planned_iqs': [],
        'completed_iqs': [],
        'agent_i0_calls': 0,
        'agent_q_calls': 0,
        'agent1_calls': 0,
        'agent1b_calls': 0,
        'agent2_calls': 0,
        'llm_json_calls': 0,
        'llm_text_calls': 0,
        'total_llm_calls': 0,
        'embed_calls': 0,
        'total_runtime_ms': 0.0,
        'agent_i0_time_ms': 0.0,
        'mode': 'unknown'
    }
    
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
            # Extract query
            result['query'] = extract_query_from_log(log_path)
            
            # Detect mode
            if "Mode: Sequential IQs (I0 → (Q → 1&1b → 2)*N; no final aggregator)" in content:
                result['mode'] = 'sequential-iq'
            elif "Mode: Single-pass (Agents 1 & 2 only)" in content:
                result['mode'] = 'single-pass'
            
            # Extract IQ plan and completed IQs
            planned, completed = extract_iq_plan_from_log(log_path)
            result['planned_iqs'] = planned
            result['completed_iqs'] = completed
            result['num_iqs_planned'] = len(planned)
            result['num_iqs_executed'] = len(completed) if completed else len(planned)  # fallback
            
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
            agent_i0_matches = re.findall(r'\[Agent I0\] Prompt:', content)
            result['agent_i0_calls'] = len(agent_i0_matches)
            
            agent_q_matches = re.findall(r'\[Agent Q\] Prompt:', content)
            result['agent_q_calls'] = len(agent_q_matches)
            
            agent1_matches = re.findall(r'\[Agent 1\] Prompt:', content)
            result['agent1_calls'] = len(agent1_matches)
            
            agent1b_matches = re.findall(r'\[Agent 1b\] Prompt:', content)
            result['agent1b_calls'] = len(agent1b_matches)
            
            agent2_matches = re.findall(r'\[Agent 2\] Prompt:', content)
            result['agent2_calls'] = len(agent2_matches)
            
            # Extract timing information
            runtime_match = re.search(r'Total runtime:\s*([\d.]+)\s*ms', content)
            if runtime_match:
                result['total_runtime_ms'] = float(runtime_match.group(1))
            
            agent_i0_match = re.search(r'Agent I0 time:\s*([\d.]+)\s*ms', content)
            if agent_i0_match:
                result['agent_i0_time_ms'] = float(agent_i0_match.group(1))
            
            # Extract number of IQs from summary if available
            iqs_match = re.search(r'Number of IQs executed:\s*(\d+)', content)
            if iqs_match:
                result['num_iqs_executed'] = int(iqs_match.group(1))
                
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
        'total_llm_calls', 'llm_json_calls', 'llm_text_calls',
        'agent_i0_calls', 'agent_q_calls', 'agent1_calls', 'agent1b_calls', 'agent2_calls',
        'embed_calls', 'total_runtime_ms', 'agent_i0_time_ms'
    ]
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\n✓ Detailed results saved to: {csv_path}")
    
    # 2. Save full JSON (includes IQ arrays)
    json_path = output_folder / f"llm_calls_detailed_{timestamp}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"✓ JSON results saved to: {json_path}")
    
    # 3. Save IQ breakdown (NEW)
    iq_path = output_folder / f"iq_breakdown_{timestamp}.json"
    iq_data = []
    for r in results:
        if r.get('planned_iqs') or r.get('completed_iqs'):
            iq_data.append({
                'log_file': r['log_file'],
                'query': r['query'],
                'mode': r['mode'],
                'num_iqs_planned': r['num_iqs_planned'],
                'num_iqs_executed': r['num_iqs_executed'],
                'planned_iqs': r['planned_iqs'],
                'completed_iqs': r['completed_iqs']
            })
    
    with open(iq_path, 'w', encoding='utf-8') as f:
        json.dump(iq_data, f, indent=2, ensure_ascii=False)
    
    print(f"✓ IQ breakdown saved to: {iq_path}")
    
    # 4. Save human-readable IQ list (NEW)
    iq_txt_path = output_folder / f"iq_readable_{timestamp}.txt"
    with open(iq_txt_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("INTERMEDIATE QUESTIONS (IQ) BREAKDOWN - HUMAN READABLE\n")
        f.write("=" * 80 + "\n")
        
        for i, r in enumerate(results, 1):
            if r.get('planned_iqs') or r.get('completed_iqs'):
                f.write(f"[{i}] {r['log_file']}\n")
                f.write(f"Original Query: {r['query']}\n")
                f.write(f"Mode: {r['mode']} | Planned: {r['num_iqs_planned']} | Executed: {r['num_iqs_executed']}\n")
                f.write("-" * 80 + "\n")
                
                if r.get('planned_iqs'):
                    f.write("PLANNED IQs:\n")
                    for j, iq in enumerate(r['planned_iqs'], 1):
                        f.write(f"  {j}. {iq}\n")
                
                if r.get('completed_iqs') and r['completed_iqs'] != r['planned_iqs']:
                    f.write("\nCOMPLETED/MODIFIED IQs:\n")
                    for j, iq in enumerate(r['completed_iqs'], 1):
                        f.write(f"  {j}. {iq}\n")
                
                f.write("\n")
    
    print(f"✓ Human-readable IQs saved to: {iq_txt_path}")
    
    # 5. Create summary statistics
    summary = create_summary(results)
    summary_path = output_folder / f"llm_calls_summary_{timestamp}.txt"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(summary)
    
    print(f"✓ Summary saved to: {summary_path}")
    print("\n" + summary)


def create_summary(results: List[Dict]) -> str:
    """Create a text summary of the analysis with enhanced IQ statistics."""
    
    if not results:
        return "No results to summarize."
    
    total_files = len(results)
    
    # Separate by mode
    iq_results = [r for r in results if r['mode'] == 'sequential-iq']
    single_pass_results = [r for r in results if r['mode'] == 'single-pass']
    
    # Overall stats
    total_llm_calls = sum(r['total_llm_calls'] for r in results)
    total_json_calls = sum(r['llm_json_calls'] for r in results)
    total_text_calls = sum(r['llm_text_calls'] for r in results)
    total_embed_calls = sum(r['embed_calls'] for r in results)
    
    avg_llm_per_query = total_llm_calls / total_files if total_files > 0 else 0
    avg_embed_per_query = total_embed_calls / total_files if total_files > 0 else 0
    
    # IQ statistics (ENHANCED)
    iq_planned_counts = [r['num_iqs_planned'] for r in iq_results if r.get('num_iqs_planned', 0) > 0]
    iq_executed_counts = [r['num_iqs_executed'] for r in iq_results if r.get('num_iqs_executed', 0) > 0]
    iq_planned_distribution = Counter(iq_planned_counts)
    iq_executed_distribution = Counter(iq_executed_counts)
    
    if iq_results:
        total_iqs_planned = sum(r['num_iqs_planned'] for r in iq_results)
        total_iqs_executed = sum(r['num_iqs_executed'] for r in iq_results)
        avg_iqs_planned = total_iqs_planned / len(iq_results) if len(iq_results) > 0 else 0
        avg_iqs_executed = total_iqs_executed / len(iq_results) if len(iq_results) > 0 else 0
        min_iqs_planned = min(r['num_iqs_planned'] for r in iq_results) if iq_results else 0
        max_iqs_planned = max(r['num_iqs_planned'] for r in iq_results) if iq_results else 0
        min_iqs_executed = min(r['num_iqs_executed'] for r in iq_results) if iq_results else 0
        max_iqs_executed = max(r['num_iqs_executed'] for r in iq_results) if iq_results else 0
        
        total_agent_i0 = sum(r['agent_i0_calls'] for r in iq_results)
        total_agent_q = sum(r['agent_q_calls'] for r in iq_results)
        avg_agent_i0_time = sum(r['agent_i0_time_ms'] for r in iq_results) / len(iq_results)
        
        # Calculate completion rate
        completion_rate = (total_iqs_executed / total_iqs_planned * 100) if total_iqs_planned > 0 else 0
    else:
        total_iqs_planned = total_iqs_executed = avg_iqs_planned = avg_iqs_executed = 0
        min_iqs_planned = max_iqs_planned = min_iqs_executed = max_iqs_executed = 0
        total_agent_i0 = total_agent_q = 0
        avg_agent_i0_time = 0
        completion_rate = 0
    
    total_runtime = sum(r['total_runtime_ms'] for r in results)
    avg_runtime = total_runtime / total_files if total_files > 0 else 0
    
    min_llm = min(r['total_llm_calls'] for r in results)
    max_llm = max(r['total_llm_calls'] for r in results)
    
    summary = f"""
╔══════════════════════════════════════════════════════════════╗
║    INTERMEDIATE QUESTIONS (IQ) LLM CALL ANALYSIS SUMMARY     ║
╚══════════════════════════════════════════════════════════════╝

📊 Overall Statistics:
├─ Total log files analyzed: {total_files}
│  ├─ Sequential IQ mode: {len(iq_results)}
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
    
    if iq_results:
        summary += f"""
🔄 INTERMEDIATE QUESTIONS (IQ) ANALYSIS ({len(iq_results)} queries):
╔══════════════════════════════════════════════════════════════╗
║                    IQ STATISTICS                             ║
╚══════════════════════════════════════════════════════════════╝

📊 IQ Planning & Execution:
├─ Total IQs planned: {total_iqs_planned}
├─ Total IQs executed: {total_iqs_executed}
├─ Completion rate: {completion_rate:.1f}%
├─ Average IQs planned per query: {avg_iqs_planned:.2f}
├─ Average IQs executed per query: {avg_iqs_executed:.2f}
├─ Min/Max IQs planned: {min_iqs_planned}/{max_iqs_planned}
└─ Min/Max IQs executed: {min_iqs_executed}/{max_iqs_executed}

📈 Planned IQ Distribution:
"""
        # Add distribution histogram for planned IQs
        for count in sorted(iq_planned_distribution.keys()):
            freq = iq_planned_distribution[count]
            percentage = (freq / len(iq_results)) * 100
            bar = "█" * int(percentage / 2)
            summary += f"├─ {count} IQ(s) planned: {freq} queries ({percentage:.1f}%) {bar}\n"
        
        summary += f"""
📊 Executed IQ Distribution:
"""
        # Add distribution histogram for executed IQs
        for count in sorted(iq_executed_distribution.keys()):
            freq = iq_executed_distribution[count]
            percentage = (freq / len(iq_results)) * 100
            bar = "█" * int(percentage / 2)
            summary += f"├─ {count} IQ(s) executed: {freq} queries ({percentage:.1f}%) {bar}\n"
        
        summary += f"""
🤖 Agent Statistics (IQ Mode):
├─ Agent I0 (IQ planner) calls: {total_agent_i0}
├─ Agent Q (query modifier) calls: {total_agent_q}
└─ Average Agent I0 time: {avg_agent_i0_time:.2f} ms

💡 Sequential Execution Pattern:
   └─ For each query: I0 → (Q → 1 → 1b → 2) × num_IQs
      ├─ Agent Q enriches IQs 2..N using prior answers
      └─ Final answer = answer from last IQ (no aggregator)
"""
    
    # Agent breakdown
    total_agent1 = sum(r['agent1_calls'] for r in results)
    total_agent1b = sum(r['agent1b_calls'] for r in results)
    total_agent2 = sum(r['agent2_calls'] for r in results)
    total_agent_i0 = sum(r['agent_i0_calls'] for r in results)
    total_agent_q = sum(r['agent_q_calls'] for r in results)
    
    summary += f"""
🔍 Agent Call Breakdown (across all queries):
├─ Agent I0 (IQ planner): {total_agent_i0}
├─ Agent Q (query modifier): {total_agent_q}
├─ Agent 1 (entity extraction): {total_agent1}
├─ Agent 1b (triple extraction): {total_agent1b}
└─ Agent 2 (answer generation): {total_agent2}

💡 Notes:
   - In IQ mode: 1 Agent I0 + (N-1) Agent Q + N × (Agent 1 + 1b + 2)
     where N = number of IQs executed
   - In single-pass mode: 1 Agent 1 + 1 Agent 1b + 1 Agent 2
   - IQ limit (hardcoded): IQ_MAX_STEPS = 5
   - Embedding calls vary based on entities, triples, and chunks
"""
    
    # Top queries by different metrics
    if iq_results:
        summary += "\n" + "=" * 70 + "\n"
        summary += "📌 TOP QUERIES BY PLANNED IQ COUNT:\n"
        summary += "=" * 70 + "\n"
        sorted_by_planned = sorted(iq_results, key=lambda x: x['num_iqs_planned'], reverse=True)
        for i, r in enumerate(sorted_by_planned[:10], 1):
            query_preview = r['query'][:60] + "..." if len(r['query']) > 60 else r['query']
            summary += f"\n[{i}] {r['num_iqs_planned']} planned / {r['num_iqs_executed']} executed | {r['log_file']}\n"
            summary += f"    Query: {query_preview}\n"
            if r.get('planned_iqs'):
                summary += f"    Planned IQs:\n"
                for j, iq in enumerate(r['planned_iqs'][:5], 1):  # Show first 5
                    iq_preview = iq[:55] + "..." if len(iq) > 55 else iq
                    summary += f"      {j}. {iq_preview}\n"
                if len(r['planned_iqs']) > 5:
                    summary += f"      ... and {len(r['planned_iqs']) - 5} more\n"
    
    summary += "\n" + "=" * 70 + "\n"
    summary += "📌 TOP QUERIES BY TOTAL LLM CALLS:\n"
    summary += "=" * 70 + "\n"
    sorted_by_calls = sorted(results, key=lambda x: x['total_llm_calls'], reverse=True)
    for i, r in enumerate(sorted_by_calls[:5], 1):
        query_preview = r['query'][:60] + "..." if len(r['query']) > 60 else r['query']
        iq_info = f"{r['num_iqs_planned']} planned / {r['num_iqs_executed']} exec" if r.get('num_iqs_planned') else "N/A"
        summary += f"{i}. {r['total_llm_calls']} calls | IQs: {iq_info} | {r['mode']}\n"
        summary += f"   {query_preview}\n"
    
    summary += "\n" + "=" * 70 + "\n"
    summary += "📌 TOP QUERIES BY EMBEDDING CALLS:\n"
    summary += "=" * 70 + "\n"
    sorted_by_embeds = sorted(results, key=lambda x: x['embed_calls'], reverse=True)
    for i, r in enumerate(sorted_by_embeds[:5], 1):
        query_preview = r['query'][:60] + "..." if len(r['query']) > 60 else r['query']
        iq_info = f"{r['num_iqs_planned']} planned / {r['num_iqs_executed']} exec" if r.get('num_iqs_planned') else "N/A"
        summary += f"{i}. {r['embed_calls']} embeds | IQs: {iq_info} | {r['mode']}\n"
        summary += f"   {query_preview}\n"
    
    # Queries where planned != executed
    mismatches = [r for r in iq_results if r['num_iqs_planned'] != r['num_iqs_executed']]
    if mismatches:
        summary += "\n" + "=" * 70 + "\n"
        summary += "⚠️  QUERIES WITH PLANNED vs EXECUTED MISMATCH:\n"
        summary += "=" * 70 + "\n"
        for r in mismatches[:5]:
            query_preview = r['query'][:60] + "..." if len(r['query']) > 60 else r['query']
            summary += f"- {r['log_file']}: {r['num_iqs_planned']} planned → {r['num_iqs_executed']} executed\n"
            summary += f"  Query: {query_preview}\n"
    
    return summary


def main():
    """Main execution function."""
    
    print("=" * 70)
    print("INTERMEDIATE QUESTIONS (IQ) LLM CALL ANALYZER")
    print("(With Sequential IQ Planning & Execution Tracking)")
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