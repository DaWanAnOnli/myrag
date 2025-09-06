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
from collections import deque

# Load .env from the project root (parent of this script's directory),
# so this file also works when run directly.
load_dotenv(Path(__file__).resolve().parent.parent / ".env")


class RateLimiter:
    """
    Sliding-window RPM limiter (bursty):
    - Allows up to `rpm` starts within any 60s window.
    - When the window is full, it sleeps until the oldest call ages out.
    - No inter-call smoothing: calls fire as soon as the limiter allows.
    """
    def __init__(self, rpm: int):
        self.rpm = max(int(rpm), 1)
        self.window = 60.0
        self.calls = deque()  # timestamps of call starts

    def acquire(self) -> float:
        now = time.time()
        waited = 0.0

        # Evict old timestamps
        while self.calls and (now - self.calls[0]) >= self.window:
            self.calls.popleft()

        if len(self.calls) >= self.rpm:
            sleep_time = self.window - (now - self.calls[0])
            if sleep_time > 0:
                time.sleep(sleep_time)
                waited += sleep_time
            # Clean again after sleep
            now = time.time()
            while self.calls and (now - self.calls[0]) >= self.window:
                self.calls.popleft()

        # Record the start time for this call
        self.calls.append(time.time())
        return waited


class GeminiAPIClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        # Note: 'gemini-2.5-flash-lite' is a hypothetical name.
        # Use a real model name like 'gemini-1.5-flash-latest' or 'gemini-pro'.
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
        """Makes a single API call and handles exceptions."""
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
    """
    Runs API calls at a specific sustained rate (RPM) for a given duration,
    but in BURSTS (sliding-window limiter), similar to your QA scripts.
    """
    API_KEY = os.getenv("GOOGLE_API_KEY")
    if not API_KEY:
        print("[FATAL] GOOGLE_API_KEY not found in environment. Exiting.")
        return

    print(f"--- Starting Bursty Rate Test ---")
    print(f"Target Rate: {rate} RPM (per process)")
    print(f"Test Duration: {duration} minute(s)")
    print(f"Scheduling: Sliding-window limiter (bursty), no inter-call smoothing")
    
    try:
        client = GeminiAPIClient(API_KEY)
    except Exception:
        return

    total_calls = rate * duration
    limiter = RateLimiter(rate)
    
    print(f"Total Calls to Make: {total_calls}")
    print("-" * 34)

    results = []
    start_time = time.monotonic()
    
    for i in range(total_calls):
        # Acquire a slot; may block until the 60s window allows another call
        waited = limiter.acquire()
        call_start_time = time.monotonic()
        
        print(f"[{i+1}/{total_calls}] Sending request... (rate_wait={waited:.2f}s) ", end="", flush=True)
        
        result = await client.call_gemini(i + 1, prompt)
        results.append(result)
        
        status = "SUCCESS" if result["success"] else f"FAILED ({result['error']})"
        print(f"Status: {status} ({result['response_time']:.2f}s)")
        # Note: No smoothing sleep here; next call fires as soon as limiter allows.

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
        if failed_calls > 0:
            exit(1)


def main():
    parser = argparse.ArgumentParser(description="Gemini Bursty Rate Test Client")
    parser.add_argument("--rate", type=int, required=True, help="Requests per minute (RPM) to target.")
    parser.add_argument("--duration", type=int, required=True, help="How many minutes to run.")
    args = parser.parse_args()
    
    prompt = "Write a very short, two-sentence story about a robot discovering music."
    asyncio.run(run_sustained_rate_test(args.rate, args.duration, prompt))


if __name__ == "__main__":
    main()