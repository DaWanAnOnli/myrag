# analyze_llm_calls.py
"""
Analyzes log files from lexidkg_graphrag_agentic.py to count LLM calls per question.
"""

import re
import json
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
import csv

# Relative paths from this script's location
SCRIPT_DIR = Path(__file__).resolve().parent
LOG_FOLDER = SCRIPT_DIR / "../../4b_retrieval/4b_i_lexidkg_graphrag/question_terminal_logs/no_2_lexidkg_5_hop_1250"
OUTPUT_FOLDER = SCRIPT_DIR / "llm_calls_analysis_results_lexidkg_5_hop_1250"


def extract_query_from_log(log_path: Path) -> str:
    """Extract the original query from the log file."""
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            for line in f:
                if "Original Query:" in line:
                    # Extract everything after "Original Query: "
                    match = re.search(r'Original Query:\s*(.+)$', line)
                    if match:
                        return match.group(1).strip()
    except Exception as e:
        print(f"Error extracting query from {log_path.name}: {e}")
    return "Unknown Query"


def analyze_log_file(log_path: Path) -> Dict:
    """
    Analyze a single log file and count LLM calls.
    
    Returns dict with:
    - log_file: filename
    - query: the user's question
    - llm_json_calls: count of JSON generation calls
    - llm_text_calls: count of text generation calls
    - total_llm_calls: sum of above
    - embed_calls: count of embedding calls
    - agent1_calls: calls from Agent 1
    - agent1b_calls: calls from Agent 1b
    - agent2_calls: calls from Agent 2
    - total_runtime_ms: total execution time
    """
    
    result = {
        'log_file': log_path.name,
        'query': '',
        'llm_json_calls': 0,
        'llm_text_calls': 0,
        'total_llm_calls': 0,
        'embed_calls': 0,
        'agent1_calls': 0,
        'agent1b_calls': 0,
        'agent2_calls': 0,
        'total_runtime_ms': 0.0,
        'iterations': 0,
    }
    
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
            # Extract query
            result['query'] = extract_query_from_log(log_path)
            
            # Count LLM JSON calls (from safe_generate_json)
            llm_json_matches = re.findall(r'\[LLM JSON\] call completed', content)
            result['llm_json_calls'] = len(llm_json_matches)
            
            # Count LLM TEXT calls (from safe_generate_text)
            llm_text_matches = re.findall(r'\[LLM TEXT\] call completed', content)
            result['llm_text_calls'] = len(llm_text_matches)
            
            # Total LLM calls
            result['total_llm_calls'] = result['llm_json_calls'] + result['llm_text_calls']
            
            # Count embedding calls
            embed_matches = re.findall(r'\[Embed\] text_len=', content)
            result['embed_calls'] = len(embed_matches)
            
            # Count Agent-specific prompts
            agent1_matches = re.findall(r'\[Agent 1\] Prompt:', content)
            result['agent1_calls'] = len(agent1_matches)
            
            agent1b_matches = re.findall(r'\[Agent 1b\] Prompt:', content)
            result['agent1b_calls'] = len(agent1b_matches)
            
            agent2_matches = re.findall(r'\[Agent 2\] Prompt:', content)
            result['agent2_calls'] = len(agent2_matches)
            
            # Extract total runtime
            runtime_match = re.search(r'Total runtime:\s*([\d.]+)\s*ms', content)
            if runtime_match:
                result['total_runtime_ms'] = float(runtime_match.group(1))
            
            # Extract iterations
            iter_match = re.search(r'Iterations used:\s*(\d+)', content)
            if iter_match:
                result['iterations'] = int(iter_match.group(1))
                
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
    """Save analysis results to CSV and JSON files."""
    
    output_folder = OUTPUT_FOLDER.resolve()
    output_folder.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save detailed results as CSV
    csv_path = output_folder / f"llm_calls_detailed_{timestamp}.csv"
    csv_fields = [
        'log_file', 'query', 'total_llm_calls', 'llm_json_calls', 'llm_text_calls',
        'embed_calls', 'agent1_calls', 'agent1b_calls', 'agent2_calls',
        'iterations', 'total_runtime_ms'
    ]
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\nDetailed results saved to: {csv_path}")
    
    # Save as JSON for easier programmatic access
    json_path = output_folder / f"llm_calls_detailed_{timestamp}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"JSON results saved to: {json_path}")
    
    # Create summary statistics
    summary = create_summary(results)
    summary_path = output_folder / f"llm_calls_summary_{timestamp}.txt"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(summary)
    
    print(f"Summary saved to: {summary_path}")
    print("\n" + summary)


def create_summary(results: List[Dict]) -> str:
    """Create a text summary of the analysis."""
    
    if not results:
        return "No results to summarize."
    
    total_files = len(results)
    total_llm_calls = sum(r['total_llm_calls'] for r in results)
    total_json_calls = sum(r['llm_json_calls'] for r in results)
    total_text_calls = sum(r['llm_text_calls'] for r in results)
    total_embed_calls = sum(r['embed_calls'] for r in results)
    
    avg_llm_per_query = total_llm_calls / total_files if total_files > 0 else 0
    avg_embed_per_query = total_embed_calls / total_files if total_files > 0 else 0
    
    min_llm = min(r['total_llm_calls'] for r in results)
    max_llm = max(r['total_llm_calls'] for r in results)
    
    total_runtime = sum(r['total_runtime_ms'] for r in results)
    avg_runtime = total_runtime / total_files if total_files > 0 else 0
    
    summary = f"""
╔══════════════════════════════════════════════════════════════╗
║           LLM CALL ANALYSIS SUMMARY                          ║
╚══════════════════════════════════════════════════════════════╝

📊 Overall Statistics:
├─ Total log files analyzed: {total_files}
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

🔍 Agent Breakdown (across all queries):
├─ Agent 1 (entity extraction) calls: {sum(r['agent1_calls'] for r in results)}
├─ Agent 1b (triple extraction) calls: {sum(r['agent1b_calls'] for r in results)}
└─ Agent 2 (answer generation) calls: {sum(r['agent2_calls'] for r in results)}

💡 Note: In single-pass mode (no Judge):
   - Each query typically makes 3 LLM calls (Agent 1, 1b, and 2)
   - Embedding calls vary based on entities, triples, and chunks
"""
    
    # Top 5 queries by LLM calls
    sorted_by_calls = sorted(results, key=lambda x: x['total_llm_calls'], reverse=True)
    summary += "\n📌 Top 5 Queries by LLM Calls:\n"
    for i, r in enumerate(sorted_by_calls[:5], 1):
        query_preview = r['query'][:80] + "..." if len(r['query']) > 80 else r['query']
        summary += f"{i}. {r['total_llm_calls']} calls - {query_preview}\n"
    
    return summary


def main():
    """Main execution function."""
    
    print("=" * 70)
    print("LLM CALL ANALYZER FOR AGENTIC GRAPHRAG")
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
    print("Analysis complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()