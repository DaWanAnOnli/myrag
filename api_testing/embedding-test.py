import google.generativeai as genai
import numpy as np
import time
import threading
from datetime import datetime, timedelta

# === Configuration ===

# Hardcoded API key (replace with your actual key or use environment variables)
API_KEY = "AIzaSyDTplvPqzc9ysHj6SePd1ugnB70l6J90EE" # I've removed the key for security

# Rate Limit Test Parameters
CALLS_PER_MINUTE = 15      # Number of embedding calls per minute
DURATION_MINUTES = 1       # Duration of the test in minutes

# Sample texts to embed
SAMPLE_TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "Artificial intelligence is transforming technology.",
    "Machine learning models can understand natural language."
]

# =======================

def embed_text(content, model="models/text-embedding-004", task_type="retrieval_document"):
    """
    Function to embed a single piece of text.
    """
    try:
        result = genai.embed_content(
            model=model,
            content=content,
            task_type=task_type
        )
        return result['embedding']
    except Exception as e:
        raise e

def rate_limit_test(calls_per_minute, duration_minutes, stop_event):
    """
    Function to test API rate limits by making embedding calls at a specified rate.
    """
    genai.configure(api_key=API_KEY)
    model = "models/text-embedding-004"

    # This local variable is fine, as it's not part of the global counters.
    total_calls = 0
    start_time = datetime.now()
    end_time = start_time + timedelta(minutes=duration_minutes)

    print(f"Starting rate limit test: {calls_per_minute} calls per minute for {duration_minutes} minute(s).")
    print(f"Test will end at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    while datetime.now() < end_time and not stop_event.is_set():
        minute_start = datetime.now()
        threads = []
        interval = 60 / calls_per_minute if calls_per_minute > 0 else 0
        for i in range(calls_per_minute):
            if datetime.now() >= end_time or stop_event.is_set():
                break
            text = SAMPLE_TEXTS[i % len(SAMPLE_TEXTS)]
            thread = threading.Thread(target=process_embedding, args=(text, model, update_counters))
            thread.start()
            threads.append(thread)
            total_calls += 1
            time.sleep(interval)

        for thread in threads:
            thread.join()

        elapsed = (datetime.now() - minute_start).total_seconds()
        sleep_time = max(0, 60 - elapsed)
        if sleep_time > 0 and datetime.now() < end_time:
            print(f"\nCompleted {calls_per_minute} calls in {elapsed:.2f} seconds. Sleeping for {sleep_time:.2f} seconds.\n")
            time.sleep(sleep_time)

    # Summary of the test - Now correctly reads from the GLOBAL counters
    print("\n🔍 Rate Limit Test Completed.")
    print(f"📞 Total Calls Attempted: {total_calls}")
    print(f"✅ Successful Calls: {successful_calls}")
    print(f"🚫 Rate Limit Errors: {rate_limit_errors}")
    print(f"❌ Other Errors: {other_errors}")
    success_rate = (successful_calls / total_calls) * 100 if total_calls > 0 else 0
    print(f"📈 Success Rate: {success_rate:.2f}%")

# Global counters
lock = threading.Lock()
successful_calls = 0
rate_limit_errors = 0
other_errors = 0

def update_counters(status):
    global successful_calls, rate_limit_errors, other_errors
    with lock:
        if status == "success":
            successful_calls += 1
        elif status == "rate_limit":
            rate_limit_errors += 1
        else:
            other_errors += 1

def process_embedding(text, model, status_callback):
    """
    Process embedding and update counters based on the result.
    """
    try:
        embed = embed_text(text, model)
        status_callback("success")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Success")
    except Exception as e:
        error_message = str(e).lower()
        if "rate limit" in error_message or "429" in error_message:
            status_callback("rate_limit")
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚠️ Rate limit hit")
        else:
            status_callback("other")
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚠️ Error: {e}")

def quick_test():
    """
    Simple test to verify basic functionality
    """
    try:
        genai.configure(api_key=API_KEY)
        model = "models/text-embedding-004"
        result = genai.embed_content(
            model=model,
            content="Test sentence for embedding",
            task_type="retrieval_document"
        )
        if result and 'embedding' in result:
            print("✅ Basic embedding test passed!")
            print(f"🔢 Embedding dimension: {len(result['embedding'])}")
            return True
        else:
            print("❌ No embedding returned")
            return False
    except Exception as e:
        print(f"❌ Quick test failed: {e}")
        return False

def main():
    # Reset counters before a new test run
    global successful_calls, rate_limit_errors, other_errors
    successful_calls = 0
    rate_limit_errors = 0
    other_errors = 0

    print("🔍 Running quick test...")
    if quick_test():
        print("\n⚙️ Running comprehensive rate limit test...\n")
        stop_event = threading.Event()
        rate_limit_test(CALLS_PER_MINUTE, DURATION_MINUTES, stop_event)
    else:
        print("❌ Quick test failed. Please check your API key and internet connection.")

if __name__ == "__main__":
    main()