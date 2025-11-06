import os
import re
from pathlib import Path
from statistics import mean, median
import json
from datetime import datetime

def parse_duration(duration_str):
    """
    Parse duration string like '1m 44.83s', '1 m 44.83 s', or '30.5s' into total seconds.
    """
    duration_str = duration_str.strip()
    
    # Pattern for minutes and seconds: '1m 44.83s' or '1 m 44.83 s'
    pattern_min_sec = r'(\d+)\s*m\s*(\d+\.?\d*)\s*s'
    # Pattern for seconds only: '30.5s' or '30.5 s'
    pattern_sec = r'^(\d+\.?\d*)\s*s$'
    
    match_min_sec = re.search(pattern_min_sec, duration_str)
    if match_min_sec:
        minutes = int(match_min_sec.group(1))
        seconds = float(match_min_sec.group(2))
        return minutes * 60 + seconds
    
    match_sec = re.search(pattern_sec, duration_str)
    if match_sec:
        return float(match_sec.group(1))
    
    return None

def extract_duration_from_file(file_path):
    """
    Extract duration from a txt file.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        # Search from the bottom of the file
        for line in reversed(lines):
            if 'Duration (this question):' in line:
                # Extract the duration part
                match = re.search(r'Duration \(this question\):\s*(.+)', line)
                if match:
                    duration_str = match.group(1).strip()
                    duration_seconds = parse_duration(duration_str)
                    return duration_seconds, duration_str
        return None, None
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None, None

def format_seconds(seconds):
    """
    Format seconds back to readable format.
    """
    if seconds is None:
        return "N/A"
    minutes = int(seconds // 60)
    secs = seconds % 60
    if minutes > 0:
        return f"{minutes}m {secs:.2f}s"
    else:
        return f"{secs:.2f}s"

def scan_folders(base_path):
    """
    Scan all folders and extract duration information.
    """
    base_path = Path(base_path)
    results = {}
    
    if not base_path.exists():
        print(f"Error: Path {base_path} does not exist!")
        return results
    
    # Get all subdirectories
    folders = [f for f in base_path.iterdir() if f.is_dir()]
    
    for folder in folders:
        folder_name = folder.name
        print(f"\nScanning folder: {folder_name}")
        
        # Get all txt files in this folder
        txt_files = list(folder.glob("*.txt"))
        
        folder_data = {
            'files': [],
            'durations_seconds': [],
            'durations_original': []
        }
        
        for txt_file in txt_files:
            duration_seconds, duration_str = extract_duration_from_file(txt_file)
            
            if duration_seconds is not None:
                folder_data['files'].append(txt_file.name)
                folder_data['durations_seconds'].append(duration_seconds)
                folder_data['durations_original'].append(duration_str)
                print(f"  {txt_file.name}: {duration_str}")
            else:
                print(f"  {txt_file.name}: Duration not found")
        
        if folder_data['durations_seconds']:
            folder_data['statistics'] = {
                'count': len(folder_data['durations_seconds']),
                'mean_seconds': mean(folder_data['durations_seconds']),
                'median_seconds': median(folder_data['durations_seconds']),
                'min_seconds': min(folder_data['durations_seconds']),
                'max_seconds': max(folder_data['durations_seconds']),
                'total_seconds': sum(folder_data['durations_seconds'])
            }
        else:
            folder_data['statistics'] = None
        
        results[folder_name] = folder_data
    
    return results

def save_results(results, output_dir):
    """
    Save results to files in the output directory.
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save detailed results for each folder
    for folder_name, data in results.items():
        output_file = output_path / f"{folder_name}_results.txt"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"Results for folder: {folder_name}\n")
            f.write("=" * 80 + "\n")
            
            f.write(f"Number of files: {len(data['files'])}\n")
            
            f.write("Individual file durations:\n")
            f.write("-" * 80 + "\n")
            for file, dur_orig, dur_sec in zip(data['files'], 
                                                data['durations_original'], 
                                                data['durations_seconds']):
                f.write(f"{file}: {dur_orig} ({dur_sec:.2f}s)\n")
            
            if data['statistics']:
                f.write("\n" + "=" * 80 + "\n")
                f.write("STATISTICS\n")
                f.write("=" * 80 + "\n")
                stats = data['statistics']
                f.write(f"Total files:     {stats['count']}\n")
                f.write(f"Mean duration:   {format_seconds(stats['mean_seconds'])} ({stats['mean_seconds']:.2f}s)\n")
                f.write(f"Median duration: {format_seconds(stats['median_seconds'])} ({stats['median_seconds']:.2f}s)\n")
                f.write(f"Min duration:    {format_seconds(stats['min_seconds'])} ({stats['min_seconds']:.2f}s)\n")
                f.write(f"Max duration:    {format_seconds(stats['max_seconds'])} ({stats['max_seconds']:.2f}s)\n")
                f.write(f"Total time:      {format_seconds(stats['total_seconds'])} ({stats['total_seconds']:.2f}s)\n")
            else:
                f.write("\nNo durations found in this folder.\n")
    
    # Save summary report
    summary_file = output_path / f"summary_report_{timestamp}.txt"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("SUMMARY REPORT - ALL FOLDERS\n")
        f.write("=" * 80 + "\n")
        
        total_files = 0
        all_durations = []
        
        for folder_name, data in results.items():
            f.write(f"\nFolder: {folder_name}\n")
            f.write("-" * 80 + "\n")
            
            if data['statistics']:
                stats = data['statistics']
                f.write(f"  Files:          {stats['count']}\n")
                f.write(f"  Mean:           {format_seconds(stats['mean_seconds'])}\n")
                f.write(f"  Median:         {format_seconds(stats['median_seconds'])}\n")
                f.write(f"  Min:            {format_seconds(stats['min_seconds'])}\n")
                f.write(f"  Max:            {format_seconds(stats['max_seconds'])}\n")
                f.write(f"  Total:          {format_seconds(stats['total_seconds'])}\n")
                
                total_files += stats['count']
                all_durations.extend(data['durations_seconds'])
            else:
                f.write("  No durations found\n")
        
        if all_durations:
            f.write("\n" + "=" * 80 + "\n")
            f.write("OVERALL STATISTICS (ALL FOLDERS COMBINED)\n")
            f.write("=" * 80 + "\n")
            f.write(f"Total folders scanned:  {len(results)}\n")
            f.write(f"Total files processed:  {total_files}\n")
            f.write(f"Overall mean:           {format_seconds(mean(all_durations))}\n")
            f.write(f"Overall median:         {format_seconds(median(all_durations))}\n")
            f.write(f"Overall min:            {format_seconds(min(all_durations))}\n")
            f.write(f"Overall max:            {format_seconds(max(all_durations))}\n")
            f.write(f"Overall total time:     {format_seconds(sum(all_durations))}\n")
    
    # Save JSON version for easy processing
    json_file = output_path / f"results_{timestamp}.json"
    json_data = {}
    for folder_name, data in results.items():
        json_data[folder_name] = {
            'files': data['files'],
            'durations_seconds': data['durations_seconds'],
            'durations_original': data['durations_original'],
            'statistics': data['statistics']
        }
    
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"Results saved to: {output_path}")
    print(f"  - Individual folder results: {folder_name}_results.txt")
    print(f"  - Summary report: {summary_file.name}")
    print(f"  - JSON data: {json_file.name}")
    print(f"{'='*80}")

def main():
    # Define paths
    base_scan_path = "../../4b_retrieval/4b_iv_multi_agents/question_terminal_logs_multi_agent"
    output_folder = "duration_analysis_results_multi_agent"
    
    print("Starting duration extraction...")
    print(f"Scanning path: {base_scan_path}")
    
    # Scan folders and extract durations
    results = scan_folders(base_scan_path)
    
    if not results:
        print("No results found. Please check the path.")
        return
    
    # Save results
    save_results(results, output_folder)
    
    print("\nProcessing complete!")

if __name__ == "__main__":
    main()