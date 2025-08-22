import asyncio
import os
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
from google.api_core import exceptions as google_exceptions

class GeminiAPIClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        # Note: 'gemini-2.5-flash-lite' is a hypothetical name.
        # Use a real model name like 'gemini-1.5-flash-latest' or 'gemini-pro'.
        # I'm leaving your original name here as requested.
        self.model_name = 'gemini-2.5-flash-lite'
        
        try:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(self.model_name)
            self.generation_config = GenerationConfig(temperature=0.7)
        except Exception as e:
            print(f"FATAL: Failed to initialize GenerativeModel with '{self.model_name}'. Error: {e}")
            print("Please ensure the model name is correct and the API key has permissions.")
            raise
    
    async def call_gemini(self, call_id: int, prompt: str) -> Dict[str, Any]:
        """Makes a single API call and handles all possible exceptions."""
        start_time = time.monotonic()
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, lambda: self.model.generate_content(prompt, generation_config=self.generation_config)
            )
            if response.text:
                return {"success": True, "status_code": 200, "error": None, "response_time": time.monotonic() - start_time}
            else:
                reason = "Unknown"
                if response.candidates and hasattr(response.candidates[0], 'finish_reason'):
                    reason = response.candidates[0].finish_reason.name
                return {"success": False, "status_code": 200, "error": f"Response Blocked: {reason}", "response_time": time.monotonic() - start_time}
        except google_exceptions.ResourceExhausted as e:
            return {"success": False, "status_code": 429, "error": "Rate limit (429)", "response_time": time.monotonic() - start_time}
        except google_exceptions.PermissionDenied as e:
            return {"success": False, "status_code": 403, "error": f"Permission Denied (403): {e}", "response_time": time.monotonic() - start_time}
        except Exception as e:
            status_code = getattr(e, 'code', 'N/A')
            return {"success": False, "status_code": status_code, "error": f"Other Error: {type(e).__name__}", "response_time": time.monotonic() - start_time}

async def run_sustained_rate_test(rate: int, duration: int, prompt: str):
    """Runs API calls at a specific sustained rate (RPM) for a given duration."""
    
    API_KEY = os.getenv("GOOGLE_API_KEY")
    if not API_KEY:
        print("[FATAL] GOOGLE_API_KEY not found in environment. Exiting.")
        return

    print(f"--- Starting Sustained Rate Test ---")
    print(f"Target Rate: {rate} RPM")
    print(f"Test Duration: {duration} minute(s)")
    
    try:
        client = GeminiAPIClient(API_KEY)
    except Exception:
        # Error is printed inside the constructor
        return

    delay_between_calls = 60.0 / rate
    total_calls = rate * duration
    
    print(f"Total Calls to Make: {total_calls}")
    print(f"Delay Between Calls: {delay_between_calls:.2f} seconds")
    print("-" * 34)

    results = []
    start_time = time.monotonic()
    
    for i in range(total_calls):
        call_start_time = time.monotonic()
        
        print(f"[{i+1}/{total_calls}] Sending request... ", end="", flush=True)
        
        result = await client.call_gemini(i + 1, prompt)
        results.append(result)
        
        status = "SUCCESS" if result["success"] else f"FAILED ({result['error']})"
        print(f"Status: {status} ({result['response_time']:.2f}s)")
        
        # Calculate time taken for the call and sleep for the remaining time
        time_taken = time.monotonic() - call_start_time
        sleep_duration = max(0, delay_between_calls - time_taken)
        
        if i < total_calls - 1:
            await asyncio.sleep(sleep_duration)

    total_duration = time.monotonic() - start_time
    print("\n--- Test Complete ---")
    
    successful_calls = sum(1 for r in results if r["success"])
    failed_calls = len(results) - successful_calls
    
    print(f"Total time elapsed: {total_duration:.2f}s")
    print(f"Total calls made: {len(results)}")
    print(f"Successful: {successful_calls}")
    print(f"Failed (Rate Limited, etc.): {failed_calls}")
    
    if len(results) > 0:
        success_rate = (successful_calls / len(results)) * 100
        print(f"Success Rate: {success_rate:.1f}%")
        # Exit with a non-zero code if any calls failed
        if failed_calls > 0:
            exit(1)

def main():
    parser = argparse.ArgumentParser(description="Gemini Sustained Rate Test Client")
    parser.add_argument("--rate", type=int, required=True, help="Number of requests per minute (RPM) to send.")
    parser.add_argument("--duration", type=int, required=True, help="How many minutes to run the test for.")
    args = parser.parse_args()
    
    prompt = "Write a very short, two-sentence story about a robot discovering music."
    asyncio.run(run_sustained_rate_test(args.rate, args.duration, prompt))

if __name__ == "__main__":
    main()