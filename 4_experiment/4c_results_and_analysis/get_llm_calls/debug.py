# check_decomposer_response.py
"""
Extract and display actual decomposer LLM responses.
"""

import re
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
LOG_FOLDER = SCRIPT_DIR / "../../4b_retrieval/4b_iv_multi_agents/question_terminal_logs_multi_agent/approach_2_query_decomposer_new"

def extract_decomposer_response(log_path: Path) -> str:
    """Extract the actual LLM response from decomposer."""
    with open(log_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find decomposer prompt
    decomposer_start = content.find('[Decomposer] Prompt:')
    if decomposer_start == -1:
        return "No decomposer prompt found"
    
    # Find where the response likely starts (after the prompt)
    # Look for the next timestamp or agent marker
    response_start = decomposer_start + 100
    
    # Find where response ends (next agent or plan summary)
    plan_start = content.find('[Plan]', response_start)
    next_agent = content.find('[Agent', response_start)
    
    # Take whichever comes first
    end_markers = [x for x in [plan_start, next_agent] if x != -1]
    response_end = min(end_markers) if end_markers else response_start + 2000
    
    response_section = content[response_start:response_end]
    
    # Try to find the actual LLM output
    # It might be after a line like "Response:" or just after the prompt
    lines = response_section.split('\n')
    
    # Skip timestamp/log lines and get to the actual content
    actual_response = []
    in_response = False
    
    for line in lines:
        # Skip log metadata lines
        if '[2025-' in line or '[INFO]' in line or '[pid=' in line:
            continue
        
        # Skip empty lines at the start
        if not in_response and not line.strip():
            continue
        
        # Start collecting response
        if line.strip():
            in_response = True
            actual_response.append(line)
    
    return '\n'.join(actual_response[:50])  # First 50 lines

def main():
    log_folder = LOG_FOLDER.resolve()
    
    if not log_folder.exists():
        print(f"Error: Log folder not found: {log_folder}")
        return
    
    log_files = sorted(log_folder.glob("*.txt"))
    
    if not log_files:
        print(f"Warning: No .txt files found in {log_folder}")
        return
    
    print(f"Found {len(log_files)} log files")
    print("\n" + "="*80)
    print("CHECKING DECOMPOSER RESPONSES")
    print("="*80)
    
    # Check first 3 files
    for i, log_file in enumerate(log_files[:3], 1):
        print(f"\n{'='*80}")
        print(f"File {i}: {log_file.name}")
        print('='*80)
        
        response = extract_decomposer_response(log_file)
        print("\n--- DECOMPOSER LLM RESPONSE ---")
        print(response)
        print("\n" + "-"*80)

if __name__ == "__main__":
    main()