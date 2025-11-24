# analyze_llm_calls_multi_agent.py
"""
Analyzes log files from the Multi-Agent RAG (Decomposer + GraphRAG + NaiveRAG + Aggregator).
Tracks LLM calls per query, decomposer decisions, aggregator decisions, and iteration counts.
"""

import re
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import csv
from collections import Counter, defaultdict

# Relative paths from this script's location
SCRIPT_DIR = Path(__file__).resolve().parent
LOG_FOLDER = SCRIPT_DIR / "../../4b_retrieval/4b_iv_multi_agents/question_terminal_logs_multi_agent/approach_1_query_decomposer_5_hops_1250"
OUTPUT_FOLDER = SCRIPT_DIR / "llm_calls_analysis_results_approach_1_query_decomposer_5_hops_1250"


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


def extract_decomposer_info(content: str) -> Dict:
    """Extract decomposer decisions and task information."""
    decomposer_info = {
        'strategy': '',
        'total_tasks': 0,
        'graphrag_tasks': 0,
        'naiverag_tasks': 0,
        'graphrag_primary_tasks': 0,
        'graphrag_support_tasks': 0,
        'naiverag_primary_tasks': 0,
        'naiverag_support_tasks': 0,
        'task_details': []
    }
    
    # Extract strategy
    strategy_match = re.search(r'\[Decomposer\] Planned (\d+) task\(s\) \| strategy: (.+)$', content, re.MULTILINE)
    if strategy_match:
        decomposer_info['total_tasks'] = int(strategy_match.group(1))
        decomposer_info['strategy'] = strategy_match.group(2).strip()
    
    # Extract individual task details
    task_pattern = r'\[Decomposer\] Task (\d+): pipeline=(\w+) role=(\w+) aspect=(.*?) \| query='
    for match in re.finditer(task_pattern, content):
        task_num = int(match.group(1))
        pipeline = match.group(2)
        role = match.group(3)
        aspect = match.group(4).strip()
        
        decomposer_info['task_details'].append({
            'task_number': task_num,
            'pipeline': pipeline,
            'role': role,
            'aspect': aspect
        })
        
        if pipeline.lower() == 'graphrag':
            decomposer_info['graphrag_tasks'] += 1
            if role.lower() == 'primary':
                decomposer_info['graphrag_primary_tasks'] += 1
            elif role.lower() == 'support':
                decomposer_info['graphrag_support_tasks'] += 1
        elif pipeline.lower() == 'naiverag':
            decomposer_info['naiverag_tasks'] += 1
            if role.lower() == 'primary':
                decomposer_info['naiverag_primary_tasks'] += 1
            elif role.lower() == 'support':
                decomposer_info['naiverag_support_tasks'] += 1
    
    return decomposer_info


def extract_aggregator_info(content: str) -> Dict:
    """Extract aggregator decision information."""
    aggregator_info = {
        'decision': '',
        'rationale': ''
    }
    
    # Extract decision and rationale
    decision_match = re.search(r'\[Aggregator\] Decision=(\w+) \| rationale=(.+)$', content, re.MULTILINE)
    if decision_match:
        aggregator_info['decision'] = decision_match.group(1)
        aggregator_info['rationale'] = decision_match.group(2).strip()
    
    # Fallback: check summary section
    if not aggregator_info['decision']:
        summary_match = re.search(r'Aggregator decision: (\w+)', content)
        if summary_match:
            aggregator_info['decision'] = summary_match.group(1)
    
    return aggregator_info


def extract_iteration_counts(content: str) -> Tuple[int, int, List[int], List[int]]:
    """Extract iteration counts for GraphRAG and NaiveRAG tasks."""
    graphrag_iterations = []
    naiverag_iterations = []
    
    # GraphRAG iterations per task
    graphrag_pattern = r'\[Run\] GraphRAG Task (\d+)/\d+'
    graphrag_tasks = re.findall(graphrag_pattern, content)
    
    for task_num in set(graphrag_tasks):
        # Count iterations for this task
        iter_pattern = rf'\[iter=G(\d+)/\d+\].*GraphRAG Iteration \1/\d+ START'
        iterations = re.findall(iter_pattern, content)
        if iterations:
            max_iter = max(int(i) for i in iterations)
            graphrag_iterations.append(max_iter)
    
    # NaiveRAG iterations per task
    naiverag_pattern = r'\[Run\] NaiveRAG Task (\d+)/\d+'
    naiverag_tasks = re.findall(naiverag_pattern, content)
    
    for task_num in set(naiverag_tasks):
        # Count iterations for this task
        iter_pattern = rf'\[iter=N(\d+)/\d+\].*NaiveRAG Iteration \1/\d+ START'
        iterations = re.findall(iter_pattern, content)
        if iterations:
            max_iter = max(int(i) for i in iterations)
            naiverag_iterations.append(max_iter)
    
    total_graphrag_iters = sum(graphrag_iterations)
    total_naiverag_iters = sum(naiverag_iterations)
    
    return total_graphrag_iters, total_naiverag_iters, graphrag_iterations, naiverag_iterations


def count_llm_calls(content: str) -> Dict[str, int]:
    """
    Count LLM calls by type.
    
    LLM generation calls include:
    - Agent 1 (entity/predicate extraction)
    - Agent 1b (query triple extraction)
    - Agent 2 (answerer)
    - Answer Judge (AJ)
    - Query Modifier (QM)
    - Decomposer
    - Aggregator
    """
    llm_calls = {
        'decomposer': 0,
        'aggregator': 0,
        'agent1': 0,        # Entity/predicate extraction (GraphRAG)
        'agent1b': 0,       # Query triple extraction (GraphRAG)
        'agent2': 0,        # Answerer (both pipelines)
        'answer_judge': 0,  # Answer Judge (both pipelines)
        'query_modifier': 0, # Query Modifier (both pipelines)
        'embed_calls': 0,
        'total_generation': 0,
        'total_all': 0
    }
    
    # Decomposer
    llm_calls['decomposer'] = len(re.findall(r'\[Decomposer\] Prompt:', content))
    
    # Aggregator
    llm_calls['aggregator'] = len(re.findall(r'\[Aggregator\] Prompt:', content))
    
    # Agent 1 (GraphRAG entity extraction)
    llm_calls['agent1'] = len(re.findall(r'\[Agent 1\] Prompt:', content))
    
    # Agent 1b (GraphRAG triple extraction)
    llm_calls['agent1b'] = len(re.findall(r'\[Agent 1b\] Prompt:', content))
    
    # Agent 2 (Answerer - both pipelines)
    llm_calls['agent2'] = len(re.findall(r'\[Agent 2\] Prompt:', content))
    
    # Answer Judge (both pipelines use different log formats)
    # GraphRAG: "You are Agent AJ (Answer Judge)"
    # NaiveRAG: "You are the Answer Judge"
    aj_graphrag = len(re.findall(r'You are Agent AJ \(Answer Judge\)', content))
    aj_naive = len(re.findall(r'You are the Answer Judge\.', content))
    llm_calls['answer_judge'] = aj_graphrag + aj_naive
    
    # Query Modifier (both pipelines)
    # GraphRAG: "You are Agent QM (Query Modifier)"
    # NaiveRAG: "You are the Query Modifier"
    qm_graphrag = len(re.findall(r'You are Agent QM \(Query Modifier\)', content))
    qm_naive = len(re.findall(r'You are the Query Modifier\.', content))
    llm_calls['query_modifier'] = qm_graphrag + qm_naive
    
    # Embedding calls
    llm_calls['embed_calls'] = len(re.findall(r'\[Embed\] text_len=', content))
    
    # Total generation calls
    llm_calls['total_generation'] = (
        llm_calls['decomposer'] +
        llm_calls['aggregator'] +
        llm_calls['agent1'] +
        llm_calls['agent1b'] +
        llm_calls['agent2'] +
        llm_calls['answer_judge'] +
        llm_calls['query_modifier']
    )
    
    # Total all calls (generation + embedding)
    llm_calls['total_all'] = llm_calls['total_generation'] + llm_calls['embed_calls']
    
    return llm_calls


def extract_retrieval_stats(content: str) -> Dict:
    """Extract retrieval statistics from GraphRAG and NaiveRAG."""
    stats = {
        'graphrag_triples_retrieved': 0,
        'graphrag_chunks_retrieved': 0,
        'naiverag_chunks_retrieved': 0,
        'graphrag_triples_selected': 0,
        'graphrag_chunks_selected': 0,
        'naiverag_chunks_selected': 0
    }
    
    # GraphRAG stats
    # Example: "Step 4 merge triples in ... ms; merged=150"
    merged_triples = re.findall(r'Step 4 merge triples.*?merged=(\d+)', content)
    if merged_triples:
        stats['graphrag_triples_retrieved'] = sum(int(x) for x in merged_triples)
    
    # Example: "Step 5a collect chunks candidates=200"
    chunk_candidates = re.findall(r'Step 5a collect chunks candidates=(\d+)', content)
    if chunk_candidates:
        stats['graphrag_chunks_retrieved'] = sum(int(x) for x in chunk_candidates)
    
    # Example: "Step 6 rerank triples ... selected=60"
    triples_selected = re.findall(r'Step 6 rerank triples.*?selected=(\d+)', content)
    if triples_selected:
        stats['graphrag_triples_selected'] = sum(int(x) for x in triples_selected)
    
    # Example: "Step 5b rerank chunks ... selected=40"
    chunks_selected = re.findall(r'Step 5b rerank chunks.*?selected=(\d+)', content)
    if chunks_selected:
        stats['graphrag_chunks_selected'] = sum(int(x) for x in chunks_selected)
    
    # NaiveRAG stats
    # Example: "Vector search returned 40 candidates"
    naive_candidates = re.findall(r'Vector search returned (\d+) candidates', content)
    if naive_candidates:
        stats['naiverag_chunks_retrieved'] = sum(int(x) for x in naive_candidates)
    
    # NaiveRAG uses all retrieved chunks up to MAX_CHUNKS_FINAL
    stats['naiverag_chunks_selected'] = stats['naiverag_chunks_retrieved']
    
    return stats


def analyze_log_file(log_path: Path) -> Dict:
    """
    Analyze a single log file and extract all metrics.
    """
    
    result = {
        'log_file': log_path.name,
        'query': '',
        
        # Decomposer info
        'decomposer_strategy': '',
        'total_tasks': 0,
        'graphrag_tasks': 0,
        'naiverag_tasks': 0,
        'graphrag_primary_tasks': 0,
        'graphrag_support_tasks': 0,
        'naiverag_primary_tasks': 0,
        'naiverag_support_tasks': 0,
        'task_roles': [],
        'task_aspects': [],
        
        # Aggregator info
        'aggregator_decision': '',
        'aggregator_rationale': '',
        
        # Iteration counts
        'graphrag_total_iterations': 0,
        'naiverag_total_iterations': 0,
        'graphrag_iterations_per_task': [],
        'naiverag_iterations_per_task': [],
        
        # LLM calls
        'decomposer_calls': 0,
        'aggregator_calls': 0,
        'agent1_calls': 0,
        'agent1b_calls': 0,
        'agent2_calls': 0,
        'answer_judge_calls': 0,
        'query_modifier_calls': 0,
        'total_generation_calls': 0,
        'embed_calls': 0,
        'total_all_calls': 0,
        
        # Retrieval stats
        'graphrag_triples_retrieved': 0,
        'graphrag_chunks_retrieved': 0,
        'naiverag_chunks_retrieved': 0,
        'graphrag_triples_selected': 0,
        'graphrag_chunks_selected': 0,
        'naiverag_chunks_selected': 0,
        
        # Runtime
        'total_runtime_ms': 0.0
    }
    
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
            # Extract query
            result['query'] = extract_query_from_log(log_path)
            
            # Extract decomposer info
            decomposer = extract_decomposer_info(content)
            result['decomposer_strategy'] = decomposer['strategy']
            result['total_tasks'] = decomposer['total_tasks']
            result['graphrag_tasks'] = decomposer['graphrag_tasks']
            result['naiverag_tasks'] = decomposer['naiverag_tasks']
            result['graphrag_primary_tasks'] = decomposer['graphrag_primary_tasks']
            result['graphrag_support_tasks'] = decomposer['graphrag_support_tasks']
            result['naiverag_primary_tasks'] = decomposer['naiverag_primary_tasks']
            result['naiverag_support_tasks'] = decomposer['naiverag_support_tasks']
            result['task_roles'] = [t['role'] for t in decomposer['task_details']]
            result['task_aspects'] = [t['aspect'] for t in decomposer['task_details']]
            
            # Extract aggregator info
            aggregator = extract_aggregator_info(content)
            result['aggregator_decision'] = aggregator['decision']
            result['aggregator_rationale'] = aggregator['rationale']
            
            # Extract iteration counts
            g_iters, n_iters, g_per_task, n_per_task = extract_iteration_counts(content)
            result['graphrag_total_iterations'] = g_iters
            result['naiverag_total_iterations'] = n_iters
            result['graphrag_iterations_per_task'] = g_per_task
            result['naiverag_iterations_per_task'] = n_per_task
            
            # Count LLM calls
            llm_calls = count_llm_calls(content)
            result['decomposer_calls'] = llm_calls['decomposer']
            result['aggregator_calls'] = llm_calls['aggregator']
            result['agent1_calls'] = llm_calls['agent1']
            result['agent1b_calls'] = llm_calls['agent1b']
            result['agent2_calls'] = llm_calls['agent2']
            result['answer_judge_calls'] = llm_calls['answer_judge']
            result['query_modifier_calls'] = llm_calls['query_modifier']
            result['total_generation_calls'] = llm_calls['total_generation']
            result['embed_calls'] = llm_calls['embed_calls']
            result['total_all_calls'] = llm_calls['total_all']
            
            # Extract retrieval stats
            retrieval = extract_retrieval_stats(content)
            result.update(retrieval)
            
            # Extract runtime
            runtime_match = re.search(r'Total runtime: (\d+) ms', content)
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
        'log_file', 'query',
        'decomposer_strategy', 'total_tasks', 'graphrag_tasks', 'naiverag_tasks',
        'graphrag_primary_tasks', 'graphrag_support_tasks',
        'naiverag_primary_tasks', 'naiverag_support_tasks',
        'aggregator_decision',
        'graphrag_total_iterations', 'naiverag_total_iterations',
        'total_generation_calls', 'total_all_calls',
        'decomposer_calls', 'aggregator_calls',
        'agent1_calls', 'agent1b_calls', 'agent2_calls',
        'answer_judge_calls', 'query_modifier_calls',
        'embed_calls',
        'graphrag_triples_retrieved', 'graphrag_chunks_retrieved',
        'naiverag_chunks_retrieved',
        'total_runtime_ms'
    ]
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(results)
    
    print(f"Detailed results saved to: {csv_path}")
    
    # 2. Save full JSON (including lists of iterations per task)
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
    
    # Overall LLM call stats
    total_generation_calls = sum(r['total_generation_calls'] for r in results)
    total_embed_calls = sum(r['embed_calls'] for r in results)
    total_all_calls = sum(r['total_all_calls'] for r in results)
    
    # By agent type
    total_decomposer = sum(r['decomposer_calls'] for r in results)
    total_aggregator = sum(r['aggregator_calls'] for r in results)
    total_agent1 = sum(r['agent1_calls'] for r in results)
    total_agent1b = sum(r['agent1b_calls'] for r in results)
    total_agent2 = sum(r['agent2_calls'] for r in results)
    total_aj = sum(r['answer_judge_calls'] for r in results)
    total_qm = sum(r['query_modifier_calls'] for r in results)
    
    # Iteration stats
    total_graphrag_iters = sum(r['graphrag_total_iterations'] for r in results)
    total_naiverag_iters = sum(r['naiverag_total_iterations'] for r in results)
    
    # Task stats
    total_tasks = sum(r['total_tasks'] for r in results)
    total_graphrag_tasks = sum(r['graphrag_tasks'] for r in results)
    total_naiverag_tasks = sum(r['naiverag_tasks'] for r in results)
    
    # Role stats
    total_graphrag_primary = sum(r['graphrag_primary_tasks'] for r in results)
    total_graphrag_support = sum(r['graphrag_support_tasks'] for r in results)
    total_naiverag_primary = sum(r['naiverag_primary_tasks'] for r in results)
    total_naiverag_support = sum(r['naiverag_support_tasks'] for r in results)
    
    # Decomposer strategy distribution
    strategies = Counter(r['decomposer_strategy'] for r in results if r['decomposer_strategy'])
    
    # Aggregator decision distribution
    agg_decisions = Counter(r['aggregator_decision'] for r in results if r['aggregator_decision'])
    
    # Role distribution
    all_roles = []
    for r in results:
        all_roles.extend(r['task_roles'])
    role_dist = Counter(all_roles)
    
    # Averages
    avg_generation_per_query = total_generation_calls / total_files if total_files > 0 else 0
    avg_all_per_query = total_all_calls / total_files if total_files > 0 else 0
    avg_embed_per_query = total_embed_calls / total_files if total_files > 0 else 0
    avg_tasks_per_query = total_tasks / total_files if total_files > 0 else 0
    avg_graphrag_iters = total_graphrag_iters / total_graphrag_tasks if total_graphrag_tasks > 0 else 0
    avg_naiverag_iters = total_naiverag_iters / total_naiverag_tasks if total_naiverag_tasks > 0 else 0
    
    # Min/Max
    min_generation = min(r['total_generation_calls'] for r in results) if results else 0
    max_generation = max(r['total_generation_calls'] for r in results) if results else 0
    min_all = min(r['total_all_calls'] for r in results) if results else 0
    max_all = max(r['total_all_calls'] for r in results) if results else 0
    
    header = """
╔══════════════════════════════════════════════════════════════════════╗
║   MULTI-AGENT RAG (DECOMPOSER + GRAPHRAG + NAIVERAG + AGGREGATOR)   ║
║                      LLM CALL ANALYSIS                                ║
╚══════════════════════════════════════════════════════════════════════╝
"""
    
    overall_section = f"""
📊 Overall Statistics:
├─ Total log files analyzed: {total_files}
├─ Total LLM generation calls: {total_generation_calls}
├─ Total embedding calls: {total_embed_calls}
├─ Total all calls (generation + embedding): {total_all_calls}
├─ Total tasks created by decomposer: {total_tasks}
│  ├─ GraphRAG tasks: {total_graphrag_tasks}
│  │  ├─ Primary: {total_graphrag_primary} ({total_graphrag_primary/total_graphrag_tasks*100 if total_graphrag_tasks > 0 else 0:.1f}%)
│  │  └─ Support: {total_graphrag_support} ({total_graphrag_support/total_graphrag_tasks*100 if total_graphrag_tasks > 0 else 0:.1f}%)
│  └─ NaiveRAG tasks: {total_naiverag_tasks}
│     ├─ Primary: {total_naiverag_primary} ({total_naiverag_primary/total_naiverag_tasks*100 if total_naiverag_tasks > 0 else 0:.1f}%)
│     └─ Support: {total_naiverag_support} ({total_naiverag_support/total_naiverag_tasks*100 if total_naiverag_tasks > 0 else 0:.1f}%)
├─ Total GraphRAG iterations: {total_graphrag_iters}
└─ Total NaiveRAG iterations: {total_naiverag_iters}
"""
    
    averages_section = f"""
📈 Per-Query Averages:
├─ Average LLM generation calls per query: {avg_generation_per_query:.2f}
├─ Average total calls per query (gen + embed): {avg_all_per_query:.2f}
├─ Average embedding calls per query: {avg_embed_per_query:.2f}
├─ Average tasks per query: {avg_tasks_per_query:.2f}
├─ Average GraphRAG iterations per task: {avg_graphrag_iters:.2f}
└─ Average NaiveRAG iterations per task: {avg_naiverag_iters:.2f}
"""
    
    ranges_section = f"""
📉 Range:
├─ Min/Max generation calls in a query: {min_generation}/{max_generation}
└─ Min/Max total calls in a query: {min_all}/{max_all}
"""
    
    agent_breakdown = f"""
🤖 Agent Call Breakdown (across all queries):
├─ Decomposer calls: {total_decomposer}
├─ Aggregator calls: {total_aggregator}
├─ Agent 1 (entity extraction - GraphRAG): {total_agent1}
├─ Agent 1b (triple extraction - GraphRAG): {total_agent1b}
├─ Agent 2 (answerer - both pipelines): {total_agent2}
├─ Answer Judge (both pipelines): {total_aj}
└─ Query Modifier (both pipelines): {total_qm}
"""
    
    decomposer_section = """
🔀 Decomposer Strategy Distribution:
"""
    for strategy, count in strategies.most_common():
        percentage = (count / total_files) * 100
        bar = "█" * int(percentage / 2)
        strategy_preview = strategy[:80] + "..." if len(strategy) > 80 else strategy
        decomposer_section += f"   ├─ {count} queries ({percentage:.1f}%) {bar}\n      {strategy_preview}\n"
    
    aggregator_section = """
🔗 Aggregator Decision Distribution:
"""
    for decision, count in agg_decisions.most_common():
        percentage = (count / total_files) * 100
        bar = "█" * int(percentage / 2)
        aggregator_section += f"   ├─ {decision}: {count} queries ({percentage:.1f}%) {bar}\n"
    
    role_section = f"""
👥 Task Role Distribution:
├─ Primary tasks: {role_dist.get('primary', 0)} ({role_dist.get('primary', 0)/len(all_roles)*100 if all_roles else 0:.1f}%)
│  ├─ GraphRAG Primary: {total_graphrag_primary}
│  └─ NaiveRAG Primary: {total_naiverag_primary}
└─ Support tasks: {role_dist.get('support', 0)} ({role_dist.get('support', 0)/len(all_roles)*100 if all_roles else 0:.1f}%)
   ├─ GraphRAG Support: {total_graphrag_support}
   └─ NaiveRAG Support: {total_naiverag_support}
"""
    
    iteration_section = f"""
🔄 Iteration Analysis:
├─ Total GraphRAG iterations across all tasks: {total_graphrag_iters}
│  ├─ Average per GraphRAG task: {avg_graphrag_iters:.2f}
│  └─ Across {total_graphrag_tasks} tasks
├─ Total NaiveRAG iterations across all tasks: {total_naiverag_iters}
│  ├─ Average per NaiveRAG task: {avg_naiverag_iters:.2f}
│  └─ Across {total_naiverag_tasks} tasks
"""
    
    # Top queries by LLM calls
    top_queries_section = """
📌 TOP 5 QUERIES BY LLM CALLS:
"""
    sorted_by_calls = sorted(results, key=lambda x: x['total_generation_calls'], reverse=True)
    for i, r in enumerate(sorted_by_calls[:5], 1):
        query_preview = r['query'][:60] + "..." if len(r['query']) > 60 else r['query']
        top_queries_section += f"{i}. {r['total_generation_calls']} gen calls | {r['total_all_calls']} total | "
        top_queries_section += f"G:{r['graphrag_tasks']} tasks(P:{r['graphrag_primary_tasks']}/S:{r['graphrag_support_tasks']}, {r['graphrag_total_iterations']} iters) | "
        top_queries_section += f"N:{r['naiverag_tasks']} tasks(P:{r['naiverag_primary_tasks']}/S:{r['naiverag_support_tasks']}, {r['naiverag_total_iterations']} iters)\n"
        top_queries_section += f"   Strategy: {r['decomposer_strategy'][:60]}\n"
        top_queries_section += f"   {query_preview}\n\n"
    
    pipeline_section = """
💡 Multi-Agent Pipeline Pattern:
   1. Decomposer: analyzes query and creates tasks for GraphRAG/NaiveRAG
      - Assigns primary/support roles to each task
   2. For each GraphRAG task:
      - Agent 1: extract entities/predicates
      - Agent 1b: extract query triples
      - Triple-centric and entity-centric retrieval
      - Embed query and triples
      - Answer-Judge loop (up to max iterations):
        * Agent 2: generate answer
        * Answer Judge: evaluate
        * Query Modifier: refine query if needed
   3. For each NaiveRAG task:
      - Embed query
      - Vector search chunks
      - Answer-Judge loop (up to max iterations):
        * Agent 2: generate answer
        * Answer Judge: evaluate
        * Query Modifier: refine query if needed
   4. Aggregator: combines all task answers into final answer
"""
    
    note_section = """
📝 Notes:
   - Each query triggers exactly 1 Decomposer call and 1 Aggregator call
   - GraphRAG tasks include Agent 1, Agent 1b, and Answer-Judge iterations
   - NaiveRAG tasks are simpler but can still iterate with Answer-Judge
   - Embedding calls include query embeddings and entity/triple embeddings
   - Task roles: primary tasks directly answer query, support tasks provide context
   - Aggregator decisions: choose_graphrag, choose_naiverag, or merge
"""
    
    summary = (header + overall_section + averages_section + ranges_section + 
               agent_breakdown + decomposer_section + aggregator_section + 
               role_section + iteration_section + top_queries_section + 
               pipeline_section + note_section)
    
    return summary


def main():
    """Main execution function."""
    
    separator = "=" * 80
    title = "MULTI-AGENT RAG LLM CALL ANALYZER"
    subtitle = "(Decomposer + GraphRAG + NaiveRAG + Aggregator)"
    
    print(separator)
    print(title.center(80))
    print(subtitle.center(80))
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
    print("✓ Analysis complete!".center(80))
    print(separator)


if __name__ == "__main__":
    main()