import asyncio
import os
import re
from pathlib import Path
from typing import List, Dict, Tuple
from dotenv import load_dotenv
import subprocess
from datetime import datetime

# --- Configuration ---
# Set the desired RPM for EACH individual process.
# The total RPM will be this value * number of API keys.
RPM_PER_PROCESS = 14

# Set how many minutes the test should run.
DURATION_MINUTES = 1
# ---------------------

class GeminiRunner:
    def __init__(self):
        self.script_dir = Path(__file__).parent
        self.parent_dir = self.script_dir.parent
        self.env_file = self.parent_dir / ".env"
        self.gemini_script = self.script_dir / "gemini_concurrent_calls.py"
        self.output_dir = self.script_dir / "outputs"
        
        # Create outputs directory if it doesn't exist
        self.output_dir.mkdir(exist_ok=True)
    
    def load_api_keys(self) -> Dict[str, str]:
        """Load all GOOGLE_API_KEY_* variables from .env file"""
        load_dotenv(self.env_file)
        api_keys = {}
        pattern = re.compile(r'^GOOGLE_API_KEY_(\d+)$')
        for key, value in os.environ.items():
            match = pattern.match(key)
            if match:
                key_number = match.group(1)
                api_keys[key_number] = value
        return api_keys
    
    async def run_single_process(self, key_number: str, api_key: str, rate: int, duration: int) -> Tuple[str, str, int]:
        """Run a single instance of gemini_concurrent_calls.py"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"gemini_output_key_{key_number}_{timestamp}.txt"
        
        print(f"[START] Starting process for API key #{key_number} at {rate} RPM for {duration} min(s)...")
        print(f"   Output will be saved to: {output_file}")
        
        env = os.environ.copy()
        env["GOOGLE_API_KEY"] = api_key
        
        try:
            # Modified to pass rate and duration arguments
            process = await asyncio.create_subprocess_exec(
                "python", str(self.gemini_script),
                "--rate", str(rate),
                "--duration", str(duration),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                env=env,
                cwd=self.script_dir
            )
            
            stdout, _ = await process.communicate()
            output = stdout.decode('utf-8', errors='replace')
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(f"=== Gemini API Sustained Rate Test Results ===\n")
                f.write(f"API Key Number: {key_number}\n")
                f.write(f"Target Rate: {rate} RPM\n")
                f.write(f"Duration: {duration} minutes\n")
                f.write(f"Timestamp: {timestamp}\n")
                f.write(f"Return Code: {process.returncode}\n")
                f.write("=" * 50 + "\n\n")
                f.write(output)
            
            if process.returncode == 0:
                print(f"[SUCCESS] Process for API key #{key_number} completed successfully.")
            else:
                print(f"[ERROR] Process for API key #{key_number} failed with return code {process.returncode}.")
            
            return key_number, str(output_file), process.returncode
            
        except Exception as e:
            error_msg = f"Failed to run process for API key #{key_number}: {str(e)}"
            print(f"[ERROR] {error_msg}")
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(f"=== Gemini API Test Results ===\n")
                f.write(f"API Key Number: {key_number}\n")
                f.write(f"Timestamp: {timestamp}\n")
                f.write(f"Error: {error_msg}\n")
            return key_number, str(output_file), -1
    
    async def run_all_processes(self) -> List[Tuple[str, str, int]]:
        api_keys = self.load_api_keys()
        if not api_keys:
            print(f"[ERROR] No GOOGLE_API_KEY_* variables found in .env file at {self.env_file}")
            return []
        
        total_keys = len(api_keys)
        total_rpm = total_keys * RPM_PER_PROCESS
        
        print(f"[SUCCESS] Found {total_keys} API keys in {self.env_file}.")
        print(f"[INFO] Each process will run at {RPM_PER_PROCESS} RPM for {DURATION_MINUTES} minute(s).")
        print(f"[INFO] Total combined target rate: {total_keys} keys * {RPM_PER_PROCESS} RPM = {total_rpm} RPM.")
        print("-" * 50)
        
        if not self.gemini_script.exists():
            print(f"[ERROR] Script not found: {self.gemini_script}")
            return []
        
        tasks = [
            self.run_single_process(key_number, api_key, RPM_PER_PROCESS, DURATION_MINUTES)
            for key_number, api_key in api_keys.items()
        ]
        
        print(f"[INFO] Starting {len(tasks)} concurrent processes...")
        start_time = asyncio.get_event_loop().time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        end_time = asyncio.get_event_loop().time()
        print(f"\n[INFO] All processes completed in {end_time - start_time:.2f} seconds")
        return results
    
    def print_summary(self, results: List[Tuple[str, str, int]]):
        if not results: return
        successful = sum(1 for r in results if isinstance(r, tuple) and r[2] == 0)
        failed = len(results) - successful
        print("\n" + "=" * 60)
        print("EXECUTION SUMMARY")
        print("=" * 60)
        print(f"Total processes: {len(results)}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print(f"Success rate: {(successful/len(results)*100):.1f}%")
        print()
        print("OUTPUT FILES:")
        for result in results:
            if isinstance(result, tuple):
                key_number, output_file, return_code = result
                status = "SUCCESS" if return_code == 0 else f"FAILED (code: {return_code})"
                print(f"   - API Key #{key_number}: [{status}] -> {output_file}")
            else:
                print(f"   - Unexpected result: {result}")
        print("\n[INFO] Review individual output files for detailed rate-limiting results.")

async def main():
    print("Gemini Sustained Rate API Key Runner")
    print("=" * 50)
    runner = GeminiRunner()
    results = await runner.run_all_processes()
    runner.print_summary(results)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[WARNING] Process interrupted by user")
