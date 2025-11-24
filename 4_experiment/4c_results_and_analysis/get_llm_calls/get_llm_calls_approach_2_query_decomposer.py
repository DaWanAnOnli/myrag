# analyze_llm_calls_multi_agent_approach_2_improved.py
"""
Improved analyzer that handles both old and new decomposer output formats.
"""

import re
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import csv
from collections import Counter, defaultdict

# [Keep all the imports and config the same as before]
SCRIPT_DIR = Path(__file__).resolve().parent
LOG_FOLDER = SCRIPT_DIR / "../../4b_retrieval/4b_iv_multi_agents/question_terminal_logs_multi_agent/no_16_approach_2_query_decomposer_new"
OUTPUT_FOLDER = SCRIPT_DIR / "llm_calls_analysis_results_approach_2_query_decomposer"


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
    """Extract decomposer decisions and task information - adapted for new log format."""
    decomposer_info = {
        'strategy': '',
        'confidence': 0.0,
        'rationale': '',
        'combine_instructions': '',
        'signals': [],
        'total_tasks': 0,
        'graphrag_tasks': 0,
        'naiverag_tasks': 0,
        'graphrag_primary_tasks': 0,
        'graphrag_support_tasks': 0,
        'naiverag_primary_tasks': 0,
        'naiverag_support_tasks': 0,
        'task_details': [],
        'schema_type': 'unknown'
    }
    
    # Extract strategy from plan summary
    strategy_match = re.search(r'\[Plan\] Strategy=(\S+)', content)
    if strategy_match:
        decomposer_info['strategy'] = strategy_match.group(1).strip()
    
    # Extract confidence
    conf_match = re.search(r'confidence[=:]?\s*([0-9.]+)', content, re.IGNORECASE)
    if conf_match:
        try:
            decomposer_info['confidence'] = float(conf_match.group(1))
        except ValueError:
            pass
    
    # NEW: Extract task information from the inline log statements
    # Pattern: [Decomposer] NaiveRAG task: role=primary, aspect=, query=...
    # Pattern: [Decomposer] GraphRAG task: role=support, aspect=, query=...
    
    task_log_pattern = r'\[Decomposer\] (NaiveRAG|GraphRAG) task: role=(\w+), aspect=([^,]*), query=(.+?)(?:\n|$)'
    task_matches = re.findall(task_log_pattern, content, re.MULTILINE)
    
    if task_matches:
        decomposer_info['schema_type'] = 'tasks'
        
        for pipeline, role, aspect, query in task_matches:
            pipeline_lower = pipeline.lower()
            role_lower = role.lower().strip()
            aspect_clean = aspect.strip()
            query_clean = query.strip()
            
            # Add to task details
            task_info = {
                'pipeline': pipeline_lower,
                'role': role_lower,
                'aspect': aspect_clean,
                'query': query_clean
            }
            decomposer_info['task_details'].append(task_info)
            
            # Count by pipeline and role
            if 'graphrag' in pipeline_lower:
                decomposer_info['graphrag_tasks'] += 1
                if role_lower == 'primary':
                    decomposer_info['graphrag_primary_tasks'] += 1
                elif role_lower == 'support':
                    decomposer_info['graphrag_support_tasks'] += 1
                else:
                    # Default unknown roles to primary
                    decomposer_info['graphrag_primary_tasks'] += 1
                    
            elif 'naiverag' in pipeline_lower:
                decomposer_info['naiverag_tasks'] += 1
                if role_lower == 'primary':
                    decomposer_info['naiverag_primary_tasks'] += 1
                elif role_lower == 'support':
                    decomposer_info['naiverag_support_tasks'] += 1
                else:
                    # Default unknown roles to primary
                    decomposer_info['naiverag_primary_tasks'] += 1
        
        decomposer_info['total_tasks'] = len(task_matches)
    
    # FALLBACK: If no inline task logs found, try extracting from JSON
    if decomposer_info['total_tasks'] == 0:
        # Find decomposer section
        decomposer_start = content.find('[Decomposer] Prompt:')
        if decomposer_start != -1:
            next_agent = content.find('[', decomposer_start + 100)
            decomposer_section = content[decomposer_start:next_agent] if next_agent != -1 else content[decomposer_start:decomposer_start+5000]
            
            # Try multiple JSON extraction strategies
            json_data = None
            
            # Strategy 1: Look for well-formed JSON with "tasks" key
            tasks_json_pattern = r'\{[^{}]*"tasks"\s*:\s*\[[^\]]*\][^{}]*\}'
            tasks_match = re.search(tasks_json_pattern, decomposer_section, re.DOTALL)
            if tasks_match:
                try:
                    json_data = json.loads(tasks_match.group(0))
                    decomposer_info['schema_type'] = 'tasks'
                except json.JSONDecodeError:
                    pass
            
            # Strategy 2: Look for old schema with "subqueries_naive" and "subqueries_graphrag"
            if json_data is None:
                subqueries_pattern = r'\{[^{}]*"subqueries_(?:naive|graphrag)"[^{}]*\}'
                subq_match = re.search(subqueries_pattern, decomposer_section, re.DOTALL)
                if subq_match:
                    try:
                        json_data = json.loads(subq_match.group(0))
                        decomposer_info['schema_type'] = 'subqueries'
                    except json.JSONDecodeError:
                        pass
            
            # Strategy 3: Try to find any JSON-like structure
            if json_data is None:
                all_json_matches = re.finditer(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', decomposer_section, re.DOTALL)
                for match in all_json_matches:
                    try:
                        candidate = json.loads(match.group(0))
                        if 'tasks' in candidate or 'subqueries_naive' in candidate or 'subqueries_graphrag' in candidate:
                            json_data = candidate
                            break
                    except json.JSONDecodeError:
                        continue
            
            # Process the JSON data
            if json_data:
                # Handle NEW SCHEMA (tasks with roles)
                if 'tasks' in json_data:
                    decomposer_info['schema_type'] = 'tasks'
                    tasks = json_data.get('tasks', [])
                    
                    for task in tasks:
                        pipeline = (task.get('pipeline', '') or '').lower().strip()
                        role = (task.get('role', '') or '').lower().strip()
                        aspect = (task.get('aspect', '') or '').strip()
                        query = (task.get('query', '') or '').strip()
                        
                        # Default to 'primary' if no role specified
                        if not role:
                            role = 'primary'
                        
                        task_info = {
                            'pipeline': pipeline,
                            'role': role,
                            'aspect': aspect,
                            'query': query
                        }
                        decomposer_info['task_details'].append(task_info)
                        
                        if 'graphrag' in pipeline:
                            decomposer_info['graphrag_tasks'] += 1
                            if role == 'primary':
                                decomposer_info['graphrag_primary_tasks'] += 1
                            elif role == 'support':
                                decomposer_info['graphrag_support_tasks'] += 1
                            else:
                                decomposer_info['graphrag_primary_tasks'] += 1
                                
                        elif 'naive' in pipeline:
                            decomposer_info['naiverag_tasks'] += 1
                            if role == 'primary':
                                decomposer_info['naiverag_primary_tasks'] += 1
                            elif role == 'support':
                                decomposer_info['naiverag_support_tasks'] += 1
                            else:
                                decomposer_info['naiverag_primary_tasks'] += 1
                    
                    decomposer_info['total_tasks'] = len(tasks)
                    decomposer_info['strategy'] = json_data.get('strategy', decomposer_info['strategy'])
                    decomposer_info['rationale'] = json_data.get('notes', decomposer_info['rationale'])
                    
                # Handle OLD SCHEMA (subqueries lists)
                elif 'subqueries_naive' in json_data or 'subqueries_graphrag' in json_data:
                    decomposer_info['schema_type'] = 'subqueries'
                    
                    naive_subqs = json_data.get('subqueries_naive', [])
                    graphrag_subqs = json_data.get('subqueries_graphrag', [])
                    
                    # For old schema, all tasks are implicitly "primary"
                    decomposer_info['naiverag_tasks'] = len(naive_subqs)
                    decomposer_info['graphrag_tasks'] = len(graphrag_subqs)
                    decomposer_info['naiverag_primary_tasks'] = len(naive_subqs)
                    decomposer_info['graphrag_primary_tasks'] = len(graphrag_subqs)
                    decomposer_info['naiverag_support_tasks'] = 0
                    decomposer_info['graphrag_support_tasks'] = 0
                    decomposer_info['total_tasks'] = len(naive_subqs) + len(graphrag_subqs)
                    
                    # Build task details
                    for sq in graphrag_subqs:
                        decomposer_info['task_details'].append({
                            'pipeline': 'graphrag',
                            'role': 'primary',
                            'aspect': 'unknown',
                            'query': sq
                        })
                    
                    for sq in naive_subqs:
                        decomposer_info['task_details'].append({
                            'pipeline': 'naiverag',
                            'role': 'primary',
                            'aspect': 'unknown',
                            'query': sq
                        })
                    
                    decomposer_info['strategy'] = json_data.get('strategy', decomposer_info['strategy'])
                    decomposer_info['rationale'] = json_data.get('rationale', decomposer_info['rationale'])
                    decomposer_info['combine_instructions'] = json_data.get('combine_instructions', '')
    
    # LAST RESORT FALLBACK: Extract from plan summary counts
    if decomposer_info['total_tasks'] == 0:
        naive_count_match = re.search(r'naive_(?:subqs|tasks)=(\d+)', content)
        graphrag_count_match = re.search(r'graphrag_(?:subqs|tasks)=(\d+)', content)
        
        if naive_count_match:
            count = int(naive_count_match.group(1))
            decomposer_info['naiverag_tasks'] = count
            decomposer_info['naiverag_primary_tasks'] = count
        if graphrag_count_match:
            count = int(graphrag_count_match.group(1))
            decomposer_info['graphrag_tasks'] = count
            decomposer_info['graphrag_primary_tasks'] = count
        
        decomposer_info['total_tasks'] = decomposer_info['naiverag_tasks'] + decomposer_info['graphrag_tasks']
        decomposer_info['schema_type'] = 'inferred_from_summary'
    
    return decomposer_info


# [Keep all other functions the same: extract_aggregator_info, count_llm_calls, etc.]
def extract_aggregator_info(content: str) -> Dict:
    """Extract aggregator decision information."""
    aggregator_info = {
        'decision': '',
        'rationale': ''
    }
    
    # Extract from summary
    decision_match = re.search(r'Aggregator decision:\s*(\w+)', content)
    if decision_match:
        aggregator_info['decision'] = decision_match.group(1).strip()
    
    # Extract rationale
    rationale_match = re.search(r'rationale:\s*(.+?)(?:\n|$)', content)
    if rationale_match:
        aggregator_info['rationale'] = rationale_match.group(1).strip()
    
    # Try to extract from JSON output
    agg_json_pattern = r'\[Aggregator\].*?\{.*?"decision".*?\}'
    agg_match = re.search(agg_json_pattern, content, re.DOTALL)
    if agg_match:
        try:
            agg_text = agg_match.group(0)
            json_match = re.search(r'\{.*?"decision".*?\}', agg_text, re.DOTALL)
            if json_match:
                agg_json = json.loads(json_match.group(0))
                aggregator_info['decision'] = agg_json.get('decision', aggregator_info['decision'])
                aggregator_info['rationale'] = agg_json.get('rationale', aggregator_info['rationale'])
        except json.JSONDecodeError:
            pass
    
    return aggregator_info


def count_llm_calls(content: str) -> Dict[str, int]:
    """Count LLM calls by type for Approach 2."""
    llm_calls = {
        'decomposer': 0,
        'aggregator': 0,
        'agent1': 0,
        'agent1b': 0,
        'agent2': 0,
        'embed_calls': 0,
        'total_generation': 0,
        'total_all': 0
    }
    
    llm_calls['decomposer'] = len(re.findall(r'\[Decomposer\] Prompt:', content))
    llm_calls['aggregator'] = len(re.findall(r'\[Aggregator\] Prompt:', content))
    llm_calls['agent1'] = len(re.findall(r'\[Agent 1\] Prompt:', content))
    llm_calls['agent1b'] = len(re.findall(r'\[Agent 1b\] Prompt:', content))
    llm_calls['agent2'] = len(re.findall(r'\[Agent 2\] Prompt:', content))
    
    llm_calls['embed_calls'] = len(re.findall(r'\[Embed\]|embedded|Embedded', content))
    
    if llm_calls['embed_calls'] == 0:
        embed_patterns = [
            r'Step0\] (?:Whole-query )?[Ee]mbedded',
            r'Step0\] Embedded query',
            r'G\.Step0\] Whole-query embedded',
            r'N\.Step0\] Embedded query'
        ]
        for pattern in embed_patterns:
            llm_calls['embed_calls'] += len(re.findall(pattern, content))
    
    llm_calls['total_generation'] = (
        llm_calls['decomposer'] +
        llm_calls['aggregator'] +
        llm_calls['agent1'] +
        llm_calls['agent1b'] +
        llm_calls['agent2']
    )
    
    llm_calls['total_all'] = llm_calls['total_generation'] + llm_calls['embed_calls']
    
    return llm_calls


def extract_retrieval_stats(content: str) -> Dict:
    """Extract retrieval statistics."""
    stats = {
        'graphrag_triples_retrieved': 0,
        'graphrag_chunks_retrieved': 0,
        'naiverag_chunks_retrieved': 0,
        'graphrag_triples_selected': 0,
        'graphrag_chunks_selected': 0,
        'naiverag_chunks_selected': 0
    }
    
    merged_triples = re.findall(r'Merged triples:\s*(\d+)', content)
    if merged_triples:
        stats['graphrag_triples_retrieved'] = sum(int(x) for x in merged_triples)
    
    chunk_candidates = re.findall(r'Collected\s+(\d+)\s+chunk candidates', content)
    if chunk_candidates:
        stats['graphrag_chunks_retrieved'] = sum(int(x) for x in chunk_candidates)
    
    triples_selected = re.findall(r'Selected\s+(\d+)\s+triples', content)
    if triples_selected:
        stats['graphrag_triples_selected'] = sum(int(x) for x in triples_selected)
    
    chunks_selected = re.findall(r'Selected\s+(\d+)\s+chunks', content)
    if chunks_selected:
        stats['graphrag_chunks_selected'] = sum(int(x) for x in chunks_selected)
    
    naive_candidates = re.findall(r'Vector search returned\s+(\d+)\s+candidates', content)
    if naive_candidates:
        stats['naiverag_chunks_retrieved'] = sum(int(x) for x in naive_candidates)
        stats['naiverag_chunks_selected'] = stats['naiverag_chunks_retrieved']
    
    return stats


def analyze_log_file(log_path: Path) -> Dict:
    """Analyze a single log file - improved version."""
    
    result = {
        'log_file': log_path.name,
        'query': '',
        'decomposer_strategy': '',
        'decomposer_confidence': 0.0,
        'decomposer_rationale': '',
        'decomposer_schema_type': '',
        'total_tasks': 0,
        'graphrag_tasks': 0,
        'naiverag_tasks': 0,
        'graphrag_primary_tasks': 0,
        'graphrag_support_tasks': 0,
        'naiverag_primary_tasks': 0,
        'naiverag_support_tasks': 0,
        'task_aspects': [],
        'task_roles': [],
        'aggregator_decision': '',
        'aggregator_rationale': '',
        'decomposer_calls': 0,
        'aggregator_calls': 0,
        'agent1_calls': 0,
        'agent1b_calls': 0,
        'agent2_calls': 0,
        'total_generation_calls': 0,
        'embed_calls': 0,
        'total_all_calls': 0,
        'graphrag_triples_retrieved': 0,
        'graphrag_chunks_retrieved': 0,
        'naiverag_chunks_retrieved': 0,
        'graphrag_triples_selected': 0,
        'graphrag_chunks_selected': 0,
        'naiverag_chunks_selected': 0,
        'total_runtime_ms': 0.0
    }
    
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
            result['query'] = extract_query_from_log(log_path)
            
            decomposer = extract_decomposer_info(content)
            result['decomposer_strategy'] = decomposer['strategy']
            result['decomposer_confidence'] = decomposer['confidence']
            result['decomposer_rationale'] = decomposer['rationale']
            result['decomposer_schema_type'] = decomposer['schema_type']
            result['total_tasks'] = decomposer['total_tasks']
            result['graphrag_tasks'] = decomposer['graphrag_tasks']
            result['naiverag_tasks'] = decomposer['naiverag_tasks']
            result['graphrag_primary_tasks'] = decomposer['graphrag_primary_tasks']
            result['graphrag_support_tasks'] = decomposer['graphrag_support_tasks']
            result['naiverag_primary_tasks'] = decomposer['naiverag_primary_tasks']
            result['naiverag_support_tasks'] = decomposer['naiverag_support_tasks']
            result['task_aspects'] = [t['aspect'] for t in decomposer['task_details']]
            result['task_roles'] = [t['role'] for t in decomposer['task_details']]
            
            aggregator = extract_aggregator_info(content)
            result['aggregator_decision'] = aggregator['decision']
            result['aggregator_rationale'] = aggregator['rationale']
            
            llm_calls = count_llm_calls(content)
            result['decomposer_calls'] = llm_calls['decomposer']
            result['aggregator_calls'] = llm_calls['aggregator']
            result['agent1_calls'] = llm_calls['agent1']
            result['agent1b_calls'] = llm_calls['agent1b']
            result['agent2_calls'] = llm_calls['agent2']
            result['total_generation_calls'] = llm_calls['total_generation']
            result['embed_calls'] = llm_calls['embed_calls']
            result['total_all_calls'] = llm_calls['total_all']
            
            retrieval = extract_retrieval_stats(content)
            result.update(retrieval)
            
            runtime_match = re.search(r'Total runtime:\s*(\d+(?:\.\d+)?)\s*ms', content)
            if runtime_match:
                result['total_runtime_ms'] = float(runtime_match.group(1))
            
    except Exception as e:
        print(f"Error analyzing {log_path.name}: {e}")
    
    return result


# [Keep analyze_all_logs, save_results, create_summary, and main the same but update summary to include schema_type]

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
    
    csv_path = output_folder / f"llm_calls_detailed_{timestamp}.csv"
    csv_fields = [
        'log_file', 'query',
        'decomposer_strategy', 'decomposer_schema_type', 'decomposer_confidence', 'total_tasks', 
        'graphrag_tasks', 'naiverag_tasks',
        'graphrag_primary_tasks', 'graphrag_support_tasks',
        'naiverag_primary_tasks', 'naiverag_support_tasks',
        'aggregator_decision',
        'total_generation_calls', 'total_all_calls',
        'decomposer_calls', 'aggregator_calls',
        'agent1_calls', 'agent1b_calls', 'agent2_calls',
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
    
    json_path = output_folder / f"llm_calls_detailed_{timestamp}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"JSON results saved to: {json_path}")
    
    summary = create_summary(results)
    summary_path = output_folder / f"llm_calls_summary_{timestamp}.txt"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(summary)
    
    print(f"Summary saved to: {summary_path}")
    print(summary)


def create_summary(results: List[Dict]) -> str:
    """Create a text summary - updated version."""
    
    if not results:
        return "No results to summarize."
    
    total_files = len(results)
    
    # Schema type distribution
    schema_types = Counter(r['decomposer_schema_type'] for r in results if r['decomposer_schema_type'])
    
    # [Keep all other stats the same...]
    total_generation_calls = sum(r['total_generation_calls'] for r in results)
    total_embed_calls = sum(r['embed_calls'] for r in results)
    total_all_calls = sum(r['total_all_calls'] for r in results)
    
    total_decomposer = sum(r['decomposer_calls'] for r in results)
    total_aggregator = sum(r['aggregator_calls'] for r in results)
    total_agent1 = sum(r['agent1_calls'] for r in results)
    total_agent1b = sum(r['agent1b_calls'] for r in results)
    total_agent2 = sum(r['agent2_calls'] for r in results)
    
    total_tasks = sum(r['total_tasks'] for r in results)
    total_graphrag_tasks = sum(r['graphrag_tasks'] for r in results)
    total_naiverag_tasks = sum(r['naiverag_tasks'] for r in results)
    
    total_graphrag_primary = sum(r['graphrag_primary_tasks'] for r in results)
    total_graphrag_support = sum(r['graphrag_support_tasks'] for r in results)
    total_naiverag_primary = sum(r['naiverag_primary_tasks'] for r in results)
    total_naiverag_support = sum(r['naiverag_support_tasks'] for r in results)
    
    strategies = Counter(r['decomposer_strategy'] for r in results if r['decomposer_strategy'])
    agg_decisions = Counter(r['aggregator_decision'] for r in results if r['aggregator_decision'])
    
    confidences = [r['decomposer_confidence'] for r in results if r['decomposer_confidence'] > 0]
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
    
    all_roles = []
    for r in results:
        all_roles.extend(r['task_roles'])
    role_dist = Counter(all_roles)
    
    avg_generation_per_query = total_generation_calls / total_files if total_files > 0 else 0
    avg_all_per_query = total_all_calls / total_files if total_files > 0 else 0
    avg_embed_per_query = total_embed_calls / total_files if total_files > 0 else 0
    avg_tasks_per_query = total_tasks / total_files if total_files > 0 else 0
    
    min_generation = min(r['total_generation_calls'] for r in results) if results else 0
    max_generation = max(r['total_generation_calls'] for r in results) if results else 0
    min_all = min(r['total_all_calls'] for r in results) if results else 0
    max_all = max(r['total_all_calls'] for r in results) if results else 0
    
    header = """
╔══════════════════════════════════════════════════════════════════════╗
║   MULTI-AGENT RAG - APPROACH 2 (DECOMPOSER + AGGREGATOR)            ║
║                      LLM CALL ANALYSIS - IMPROVED                     ║
╚══════════════════════════════════════════════════════════════════════╝
"""
    
    schema_section = """
🔧 Decomposer Schema Detection:
"""
    for schema_type, count in schema_types.most_common():
        percentage = (count / total_files) * 100
        schema_section += f"   ├─ {schema_type}: {count} queries ({percentage:.1f}%)\n"
    
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
└─ Average decomposer confidence: {avg_confidence:.3f}
"""
    
    # [Rest of the summary sections remain the same...]
    averages_section = f"""
📈 Per-Query Averages:
├─ Average LLM generation calls per query: {avg_generation_per_query:.2f}
├─ Average total calls per query (gen + embed): {avg_all_per_query:.2f}
├─ Average embedding calls per query: {avg_embed_per_query:.2f}
└─ Average tasks per query: {avg_tasks_per_query:.2f}
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
├─ Agent 1 (entity extraction): {total_agent1}
├─ Agent 1b (triple extraction): {total_agent1b}
└─ Agent 2 (answerer): {total_agent2}
"""
    
    decomposer_section = """
🔀 Decomposer Strategy Distribution:
"""
    for strategy, count in strategies.most_common():
        percentage = (count / total_files) * 100
        bar = "█" * int(percentage / 2)
        decomposer_section += f"   ├─ {strategy}: {count} queries ({percentage:.1f}%) {bar}\n"
    
    aggregator_section = """
🔗 Aggregator Decision Distribution:
"""
    for decision, count in agg_decisions.most_common():
        percentage = (count / total_files) * 100
        bar = "█" * int(percentage / 2)
        aggregator_section += f"   ├─ {decision}: {count} queries ({percentage:.1f}%) {bar}\n"
    
    role_section = f"""
👥 Task Role Distribution (across all tasks):
├─ Primary tasks: {role_dist.get('primary', 0)} ({role_dist.get('primary', 0)/len(all_roles)*100 if all_roles else 0:.1f}%)
│  ├─ GraphRAG Primary: {total_graphrag_primary}
│  └─ NaiveRAG Primary: {total_naiverag_primary}
└─ Support tasks: {role_dist.get('support', 0)} ({role_dist.get('support', 0)/len(all_roles)*100 if all_roles else 0:.1f}%)
   ├─ GraphRAG Support: {total_graphrag_support}
   └─ NaiveRAG Support: {total_naiverag_support}
"""
    
    allocation_patterns = Counter()
    for r in results:
        g_p = r['graphrag_primary_tasks']
        g_s = r['graphrag_support_tasks']
        n_p = r['naiverag_primary_tasks']
        n_s = r['naiverag_support_tasks']
        pattern = f"G:{g_p}p+{g_s}s | N:{n_p}p+{n_s}s"
        allocation_patterns[pattern] += 1
    
    allocation_section = """
📋 Task Allocation Patterns (Primary + Support):
"""
    for pattern, count in allocation_patterns.most_common(10):
        percentage = (count / total_files) * 100
        bar = "█" * int(percentage / 2)
        allocation_section += f"   ├─ {pattern}: {count} queries ({percentage:.1f}%) {bar}\n"
    
    top_queries_section = """
📌 TOP 5 QUERIES BY LLM CALLS:
"""
    sorted_by_calls = sorted(results, key=lambda x: x['total_generation_calls'], reverse=True)
    for i, r in enumerate(sorted_by_calls[:5], 1):
        query_preview = r['query'][:60] + "..." if len(r['query']) > 60 else r['query']
        top_queries_section += f"{i}. {r['total_generation_calls']} gen calls | {r['total_all_calls']} total | schema={r['decomposer_schema_type']}\n"
        top_queries_section += f"   Strategy: {r['decomposer_strategy']} (conf={r['decomposer_confidence']:.2f})\n"
        top_queries_section += f"   Tasks: G:{r['graphrag_tasks']}(P:{r['graphrag_primary_tasks']}/S:{r['graphrag_support_tasks']}) | "
        top_queries_section += f"N:{r['naiverag_tasks']}(P:{r['naiverag_primary_tasks']}/S:{r['naiverag_support_tasks']})\n"
        top_queries_section += f"   Aggregator: {r['aggregator_decision']}\n"
        top_queries_section += f"   {query_preview}\n\n"
    
    strategy_decision_pairs = Counter()
    for r in results:
        if r['decomposer_strategy'] and r['aggregator_decision']:
            pair = f"{r['decomposer_strategy']} → {r['aggregator_decision']}"
            strategy_decision_pairs[pair] += 1
    
    correlation_section = """
🔄 Strategy → Aggregator Decision Correlation:
"""
    for pair, count in strategy_decision_pairs.most_common(10):
        percentage = (count / total_files) * 100
        bar = "█" * int(percentage / 2)
        correlation_section += f"   ├─ {pair}: {count} ({percentage:.1f}%) {bar}\n"
    
    note_section = """
📝 Notes:
   - Schema type 'tasks' = new format with roles; 'subqueries' = old format
   - Old schema treats all subqueries as 'primary' by default (no explicit support role)
   - If support counts are 0, the decomposer is using the old schema OR intentionally not creating support tasks
   - Check decomposer_schema_type column in CSV for per-query details
"""
    
    summary = (header + schema_section + overall_section + averages_section + ranges_section + 
               agent_breakdown + decomposer_section + aggregator_section + 
               role_section + allocation_section + top_queries_section + 
               correlation_section + note_section)
    
    return summary


def main():
    """Main execution function."""
    
    separator = "=" * 80
    title = "MULTI-AGENT RAG (APPROACH 2) LLM CALL ANALYZER - IMPROVED"
    subtitle = "(Handles both old and new decomposer schemas)"
    
    print(separator)
    print(title.center(80))
    print(subtitle.center(80))
    print(separator)
    print(f"Input folder: {LOG_FOLDER.resolve()}")
    print(f"Output folder: {OUTPUT_FOLDER.resolve()}")
    print()
    
    results = analyze_all_logs()
    
    if not results:
        print("No results to save. Exiting.")
        return
    
    save_results(results)
    
    print(separator)
    print("✓ Analysis complete!".center(80))
    print(separator)


if __name__ == "__main__":
    main()