import os
import time
import json
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Barrier

MODEL = "gemini-2.5-flash-lite"
ENDPOINT = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent"

API_KEY_1 = "AIzaSyBaa9146OHSEQ05jZWSX61ipsxgAy3s56k"
API_KEY_2 = "AIzaSyAYe82NWwBA18QrvicPVIQgKE9HBjp4_gA"

if not API_KEY_1 or not API_KEY_2:
    raise SystemExit("Please set GEMINI_API_KEY_1 and GEMINI_API_KEY_2 in your environment.")

def obfuscate(key: str) -> str:
    return f"...{key[-6:]}" if len(key) >= 6 else "..."

def call_gemini_worker(i: int, api_key: str, label: str, prompt: str, barrier: Barrier, timeout: float = 30.0):
    url = f"{ENDPOINT}?key={api_key}"
    headers = {"Content-Type": "application/json"}
    body = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}]
            }
        ]
    }

    # Wait until all threads are ready, then everyone fires together
    barrier.wait()
    t0 = time.time()
    try:
        resp = requests.post(url, headers=headers, json=body, timeout=timeout)
        elapsed = time.time() - t0
        status = resp.status_code
        data = None
        text = None
        try:
            data = resp.json()
        except Exception:
            data = {"note": "Non-JSON response", "raw": resp.text[:200]}

        if status == 200 and isinstance(data, dict):
            cands = data.get("candidates", [])
            if cands:
                parts = cands[0].get("content", {}).get("parts", [])
                if parts and isinstance(parts, list):
                    # Take first text part if present
                    text = parts[0].get("text")

        return {
            "i": i,
            "label": label,
            "status": status,
            "elapsed": elapsed,
            "text": (text or "").strip(),
            "details": None if status == 200 else data,
            "sent_at": t0,
        }
    except requests.RequestException as e:
        return {
            "i": i,
            "label": label,
            "status": -1,
            "elapsed": time.time() - t0,
            "text": "",
            "details": {"error": str(e)},
            "sent_at": t0,
        }

def main():
    total_calls = 32
    calls_per_project = 16

    print(f"Preparing {total_calls} simultaneous calls to {MODEL}")
    print(f"- First {calls_per_project}: key {obfuscate(API_KEY_1)} (Project A)")
    print(f"- Last  {calls_per_project}: key {obfuscate(API_KEY_2)} (Project B)")

    # Barrier so all 10 requests start at once
    barrier = Barrier(total_calls)

    work = []
    with ThreadPoolExecutor(max_workers=total_calls) as ex:
        for i in range(total_calls):
            key = API_KEY_1 if i < calls_per_project else API_KEY_2
            label = "Project A" if i < calls_per_project else "Project B"
            prompt = f"Quick ping #{i+1}. Reply with a brief acknowledgement."
            work.append(ex.submit(call_gemini_worker, i+1, key, label, prompt, barrier))

        results = [f.result() for f in as_completed(work)]
        results.sort(key=lambda r: r["i"])  # Present in numerical order

    # Reporting
    sent_times = [r["sent_at"] for r in results]
    jitter_ms = (max(sent_times) - min(sent_times)) * 1000 if sent_times else 0.0

    print("\nPer-call results:")
    for r in results:
        if r["status"] == 200:
            snippet = repr(r["text"])[:80]
            print(f"[{r['label']}] Call {r['i']:02d}: 200 OK in {r['elapsed']:.2f}s | Reply: {snippet}")
        elif r["status"] == 429:
            print(f"[{r['label']}] Call {r['i']:02d}: 429 RATE LIMITED in {r['elapsed']:.2f}s")
            if r["details"]:
                print(f"  Details: {json.dumps(r['details'], ensure_ascii=False)[:200]}")
        elif r["status"] == -1:
            print(f"[{r['label']}] Call {r['i']:02d}: NETWORK/CLIENT ERROR in {r['elapsed']:.2f}s")
            if r["details"]:
                print(f"  Details: {r['details']}")
        else:
            print(f"[{r['label']}] Call {r['i']:02d}: HTTP {r['status']} in {r['elapsed']:.2f}s")
            if r["details"]:
                print(f"  Details: {json.dumps(r['details'], ensure_ascii=False)[:200]}")

    ok = sum(1 for r in results if r["status"] == 200)
    limited = sum(1 for r in results if r["status"] == 429)
    other = len(results) - ok - limited

    print("\nSummary:")
    print(f"- Success: {ok}/{len(results)}")
    print(f"- Rate limited (429): {limited}")
    print(f"- Other errors: {other}")
    print(f"- Launch jitter across 10 calls: {jitter_ms:.1f} ms")

if __name__ == "__main__":
    main()