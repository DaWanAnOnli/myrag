# analyze_llm_calls_naive.py
"""
Analyzes log files from the Naive Agentic RAG (Answerer-only).
Tracks LLM calls per query (ignoring Agent 1/1b, focusing on Agent 2 answering).
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
LOG_FOLDER = SCRIPT_DIR / "../../4b_retrieval/4b_ii_naiverag/question_terminal_logs_naive_over_graph/naiverag_1250"
OUTPUT_FOLDER = SCRIPT_DIR / "llm_calls_analysis_results_naiverag_1250"


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


def analyze_log_file(log_path: Path) -> Dict:
    """
    Analyze a single log file and count LLM calls.
    
    Returns dict with metrics, ignoring Agent 1/1b LLM calls per user request.
    """
    
    result = {
        'log_file': log_path.name,
        'query': '',
        'agent2_calls': 0,  # Agent 2 (Answerer) prompts - this IS our LLM call count
        'agent1_1b_calls_ignored': 0,  # Agent 1/1b prompts (not counted)
        'total_llm_calls': 0,  # = agent2_calls
        'embed_calls': 0,
        'chunks_retrieved': 0,
        'chunks_used': 0,
        'total_runtime_ms': 0.0,
        'mode': 'unknown'
    }
    
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
            # Extract query
            result['query'] = extract_query_from_log(log_path)
            
            # Detect mode
            if "Agentic RAG (Answerer-only)" in content:
                result['mode'] = 'naive-answerer-only'
            elif "Single-pass" in content:
                result['mode'] = 'single-pass'
            
            # Count Agent 2 prompts (the answerer - what we care about)
            # Each Agent 2 prompt = 1 LLM call
            agent2_matches = re.findall(r'\[Agent 2\] Prompt:', content)
            result['agent2_calls'] = len(agent2_matches)
            
            # Count Agent 1 and 1b prompts (for info only - not counted in total)
            agent1_matches = re.findall(r'\[Agent 1\] Prompt:', content)
            agent1b_matches = re.findall(r'\[Agent 1b\] Prompt:', content)
            result['agent1_1b_calls_ignored'] = len(agent1_matches) + len(agent1b_matches)
            
            # Total LLM calls = Agent 2 calls only
            result['total_llm_calls'] = result['agent2_calls']
            
            # Count embedding calls
            embed_matches = re.findall(r'\[Step 0\] Embedded query', content)
            result['embed_calls'] = len(embed_matches)
            
            # Extract chunks retrieved
            chunks_match = re.search(r'Vector search returned (\d+) candidates', content)
            if chunks_match:
                result['chunks_retrieved'] = int(chunks_match.group(1))
            
            # Chunks used is typically same as retrieved (up to MAX_CHUNKS_FINAL)
            # Try to find actual chunks used in context
            chunks_context = re.findall(r'\[Chunk \d+\]', content)
            if chunks_context:
                result['chunks_used'] = len(chunks_context)
            else:
                result['chunks_used'] = min(result['chunks_retrieved'], 40)  # fallback to config default
            
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
        'total_llm_calls', 'agent2_calls',
        'agent1_1b_calls_ignored', 'embed_calls',
        'chunks_retrieved', 'chunks_used', 'total_runtime_ms'
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
    
    # 3. Create summary statistics
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
    
    # Separate by mode
    naive_results = [r for r in results if r['mode'] == 'naive-answerer-only']
    other_results = [r for r in results if r['mode'] != 'naive-answerer-only']
    
    # Overall stats
    total_llm_calls = sum(r['total_llm_calls'] for r in results)
    total_agent2_calls = sum(r['agent2_calls'] for r in results)
    total_ignored_calls = sum(r['agent1_1b_calls_ignored'] for r in results)
    total_embed_calls = sum(r['embed_calls'] for r in results)
    total_chunks_retrieved = sum(r['chunks_retrieved'] for r in results)
    total_chunks_used = sum(r['chunks_used'] for r in results)
    
    avg_llm_per_query = total_llm_calls / total_files if total_files > 0 else 0
    avg_embed_per_query = total_embed_calls / total_files if total_files > 0 else 0
    avg_chunks_retrieved = total_chunks_retrieved / total_files if total_files > 0 else 0
    avg_chunks_used = total_chunks_used / total_files if total_files > 0 else 0
    
    # Chunks distribution
    chunk_counts = [r['chunks_retrieved'] for r in naive_results if r.get('chunks_retrieved', 0) > 0]
    chunk_distribution = Counter(chunk_counts)
    
    min_llm = min(r['total_llm_calls'] for r in results) if results else 0
    max_llm = max(r['total_llm_calls'] for r in results) if results else 0
    
    header = """
╔══════════════════════════════════════════════════════════════╗
║    NAIVE AGENTIC RAG (ANSWERER-ONLY) LLM CALL ANALYSIS      ║
╚══════════════════════════════════════════════════════════════╝
"""
    
    overall_section = f"""
📊 Overall Statistics:
├─ Total log files analyzed: {total_files}
│  ├─ Naive answerer-only mode: {len(naive_results)}
│  └─ Other modes: {len(other_results)}
├─ Total LLM calls (Agent 2 only): {total_llm_calls}
│  └─ Agent 2 (answerer) calls: {total_agent2_calls}
├─ LLM calls ignored (Agent 1/1b): {total_ignored_calls}
└─ Total embedding calls: {total_embed_calls}
"""
    
    averages_section = f"""
📈 Per-Query Averages:
├─ Average LLM calls per query (Agent 2 only): {avg_llm_per_query:.2f}
├─ Average embedding calls per query: {avg_embed_per_query:.2f}
├─ Average chunks retrieved per query: {avg_chunks_retrieved:.2f}
└─ Average chunks used in context: {avg_chunks_used:.2f}
"""
    
    ranges_section = f"""
📉 Range:
├─ Minimum LLM calls in a query: {min_llm}
└─ Maximum LLM calls in a query: {max_llm}
"""
    
    pipeline_section = """
💡 Pipeline Pattern (Naive Answerer-Only):
   └─ For each query: 
      1. Embed query (1 embedding call)
      2. Vector search TextChunk nodes (retrieve ~40 chunks)
      3. Agent 2 answers from top chunks (1 LLM generation call)
   └─ Always exactly 1 iteration
   └─ No judge, no iterations, no query modification
   └─ Agent 1/1b present but calls excluded from LLM count per user request
"""
    
    if naive_results:
        chunks_section = f"""
📦 Chunk Retrieval Analysis ({len(naive_results)} naive queries):
├─ Total chunks retrieved: {total_chunks_retrieved}
├─ Total chunks used in context: {total_chunks_used}
├─ Average chunks retrieved per query: {avg_chunks_retrieved:.2f}
├─ Average chunks used per query: {avg_chunks_used:.2f}
├─ Min/Max chunks retrieved: {min(chunk_counts) if chunk_counts else 0}/{max(chunk_counts) if chunk_counts else 0}
"""
        
        if chunk_distribution:
            chunks_section += """
└─ Chunk Retrieval Distribution (top 10):
"""
            for count in sorted(chunk_distribution.keys(), reverse=True)[:10]:
                freq = chunk_distribution[count]
                percentage = (freq / len(naive_results)) * 100
                bar = "█" * int(percentage / 2)
                chunks_section += f"   ├─ {count} chunks: {freq} queries ({percentage:.1f}%) {bar}"
        
        pipeline_section += chunks_section
    
    agent_breakdown = f"""
🔍 Agent Call Breakdown (across all queries):
├─ Agent 2 (answerer) calls: {total_agent2_calls}
├─ Agent 1/1b calls (ignored): {total_ignored_calls}
└─ Total LLM calls counted: {total_llm_calls}
"""
    
    top_queries_section = """
📌 TOP QUERIES BY CHUNKS RETRIEVED:
"""
    sorted_by_chunks = sorted(results, key=lambda x: x['chunks_retrieved'], reverse=True)
    for i, r in enumerate(sorted_by_chunks[:5], 1):
        query_preview = r['query'][:60] + "..." if len(r['query']) > 60 else r['query']
        top_queries_section += f"{i}. {r['chunks_retrieved']} chunks retrieved | {r['chunks_used']} used | {r['total_llm_calls']} LLM call"
        top_queries_section += f"   {query_preview}"
    
    comparison_section = """
📊 COMPARISON TO OTHER METHODS:
└─ Naive method characteristics:
   ├─ Simplest approach: direct vector search over chunks
   ├─ No graph traversal or triple matching
   ├─ No iterative refinement or judging
   ├─ Fastest execution (single pass, minimal LLM calls)
   ├─ Fixed cost: 1 embedding + 1 generation per query
   └─ May miss context from related entities/triples
"""
    
    note_section = """
📝 Notes:
   - Each query results in exactly 1 Agent 2 LLM call
   - Agent 1/1b are used for entity/triple extraction but their LLM calls are not counted
   - Embedding calls are for the initial query vector search
   - Chunks retrieved may differ from chunks used (MAX_CHUNKS_FINAL limit)
"""
    
    summary = header + overall_section + averages_section + ranges_section + pipeline_section + agent_breakdown + top_queries_section + comparison_section + note_section
    
    return summary


def main():
    """Main execution function."""
    
    separator = "=" * 70
    title = "NAIVE AGENTIC RAG (ANSWERER-ONLY) LLM CALL ANALYZER"
    subtitle = "(Ignoring Agent 1/1b calls per user request)"
    
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