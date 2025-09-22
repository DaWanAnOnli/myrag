#!/usr/bin/env python3
"""
run_multi_agent.py
Batched parallel runner for multi_agent.py (self-contained multi-agent RAG).

- Reads QA JSONL from underscore paths (sample/full controlled by IS_SAMPLE in .env).
- Distributes tasks evenly across GOOGLE_API_KEY_N entries from .env.
- Launches workers in batches (BATCH_SIZE). Each worker:
  * Sets its GOOGLE_API_KEY (env + refresh Gemini).
  * Imports multi_agent and calls multi_agent_answer(question).
  * Tees stdout/stderr to per-question logs.
  * Streams per-worker .part JSONL.
- Merges part files into a combined JSONL.

Output JSONL per line:
  {
    "id": <int>,
    "question": <str>,
    "ground_truth": <str>,
    "graphrag_answer": <str>,
    "naiverag_answer": <str>,
    "final_answer": <str>,
    "decision": <"graphrag"|"naiverag"|"hybrid">
  }
"""

import json
import shutil
import sys
import time
import os
from pathlib import Path
from datetime import datetime
import contextlib
import re
import random
import multiprocessing as mp

import dotenv

# ----------------- Config -----------------

env_file_path = Path("../../../.env")
if not env_file_path.exists():
    raise ImportError(f"Error: .env file not found at {env_file_path}")

dotenv.load_dotenv(env_file_path)

is_sample = os.getenv('IS_SAMPLE', '').lower()
if is_sample == "true":
    IS_SAMPLE = True
elif is_sample == "false":
    IS_SAMPLE = False
else:
    raise ValueError(f"Wrong configuration of IS_SAMPLE in .env file: {is_sample}")

# Processing limits (0 or None => all)
MAX_QUESTIONS = int(os.getenv("MAX_QUESTIONS", "0"))
SELECTION_MODE = os.getenv("SELECTION_MODE", "random")  # "first" or "random"

# Underscore-only paths
if IS_SAMPLE:
    QA_PAIRS_REL   = "../../../dataset/samples/4_experiment/4a_qa_generation/4a_iii_qa_pairs_with_id/qa_pairs.jsonl"
    OUTPUT_DIR_REL = "../../../dataset/samples/4_experiment/4b_experiment_answers/4b_iii_multi_agent"
else:
    QA_PAIRS_REL   = "../../../dataset/4_experiment/4a_qa_generation/4a_iii_qa_pairs_with_id/qa_pairs.jsonl"
    OUTPUT_DIR_REL = "../../../dataset/4_experiment/4b_experiment_answers/4b_iii_multi_agent"

PER_QUESTION_LOGS_DIRNAME = "question_terminal_logs_multi_agent"
ENV_PATH_REL = "../../../.env"
CLEAN_PART_FILES = True

# Batched execution
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "5"))  # processes per batch; <=0 => all at once

# ----------------- Utilities & logging -----------------

def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

def mask_key(key: str, show_prefix: int = 4, show_suffix: int = 2) -> str:
    if not key:
        return "<empty>"
    if len(key) <= show_prefix + show_suffix:
        return "<hidden>"
    return f"{key[:show_prefix]}...{key[-show_suffix:]}"

def format_duration(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds*1000:.0f} ms"
    m, s = divmod(seconds, 60)
    if m < 1:
        return f"{s:.2f} s"
    h, m = divmod(int(m), 60)
    if h < 1:
        return f"{int(m)}m {s:.2f}s"
    return f"{h}h {int(m)}m {s:.2f}s"

def sanitize_snippet(text: str, max_len: int = 50) -> str:
    t = (text or "").strip()
    t = re.sub(r"\s+", " ", t)
    t = t[:max_len]
    t = re.sub(r"[^a-zA-Z0-9\\-_. ]", "_", t)
    t = t.strip().replace(" ", "_")
    return t or "question"

def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            s = line.strip()
            if not s:
                continue
            try:
                yield json.loads(s)
            except Exception as e:
                print(f"[WARN] Skipping malformed JSON on line {i}: {e}", file=sys.stderr)

def ensure_output_dir_ready(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    not_empty = any(output_dir.iterdir())
    if not_empty:
        print(f"WARNING: Output directory is not empty:\n {output_dir}")
        print("Continuing will OVERWRITE (delete) all files in this folder.")
        resp = input("Type 'YES' to confirm and continue, or anything else to abort: ").strip()
        if resp != "YES":
            print("Aborted by user. No changes made.")
            sys.exit(0)
        shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory cleared: {output_dir}")

def load_google_api_keys(env_path: Path):
    keys = []
    if not env_path.exists():
        return keys
    pat = re.compile(r"^\s*(GOOGLE_API_KEY_(\d+))\s*=\s*(.+?)\s*$")
    with env_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            m = pat.match(line)
            if not m:
                continue
            label = m.group(1)
            num = int(m.group(2))
            val = m.group(3).strip()
            if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
                val = val[1:-1]
            keys.append((label, val, num))
    keys.sort(key=lambda x: x[2])
    return keys

def chunk_evenly(items, n_chunks):
    if n_chunks <= 0:
        return []
    L = len(items)
    base = L // n_chunks
    rem = L % n_chunks
    chunks = []
    start = 0
    for i in range(n_chunks):
        sz = base + (1 if i < rem else 0)
        end = start + sz
        chunks.append(items[start:end])
        start = end
    return chunks

# ----------------- Worker -----------------

def worker_main(worker_id: int,
                key_label: str,
                key_value: str,
                tasks: list,
                total_count: int,
                logs_dir: str,
                part_path: str):
    import time as _time
    start = _time.monotonic()
    logs_dir = Path(logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)
    part_path = Path(part_path)

    # Configure key for this process
    if key_value:
        os.environ["GOOGLE_API_KEY"] = key_value

    try:
        import multi_agent as rag
    except Exception as e:
        log(f"[Worker {worker_id}] ERROR: Could not import multi_agent.py: {e}")
        part_path.write_text("", encoding="utf-8")
        return

    # Try to refresh Gemini SDK with this key (best-effort)
    try:
        if key_value:
            setattr(rag, "GOOGLE_API_KEY", key_value)
            genai = __import__("google.generativeai", fromlist=["configure"])
            genai.configure(api_key=key_value)
    except Exception:
        pass

    masked = mask_key(key_value) if key_value else "(module default / env)"
    log(f"[Worker {worker_id}] Using {key_label} = {masked}. Assigned {len(tasks)} task(s). Part file: {part_path.name}")

    processed = 0
    with part_path.open("w", encoding="utf-8") as out_f:
        for i, task in enumerate(tasks, 1):
            global_idx   = task["global_idx"]
            qid          = task["id"]
            question     = task["question"]
            ground_truth = task["ground_truth"]

            preview = (question or "")[:120]
            suffix = "..." if question and len(question) > 120 else ""
            q_snippet = sanitize_snippet(question or "", max_len=60)
            q_ts = datetime.now().strftime("%Y%m%d-%H%M%S")
            log_file = logs_dir / f"q{global_idx:04d}_wid{worker_id}_id{qid}_{q_ts}_{q_snippet}.txt"

            start_single = _time.monotonic()
            with log_file.open("w", encoding="utf-8") as lf:
                class TeeIO:
                    def __init__(self, primary, secondary_file):
                        self.primary = primary
                        self.secondary = secondary_file
                    def write(self, data):
                        try: self.primary.write(data)
                        except Exception: pass
                        try: self.secondary.write(data)
                        except Exception: pass
                    def flush(self):
                        try: self.primary.flush()
                        except Exception: pass
                        try: self.secondary.flush()
                        except Exception: pass

                tee_out = TeeIO(sys.stdout, lf)
                tee_err = TeeIO(sys.stderr, lf)
                with contextlib.redirect_stdout(tee_out), contextlib.redirect_stderr(tee_err):
                    print("=" * 100)
                    print(f"[Worker {worker_id}] Question global #{global_idx} (ID: {qid}) | Total: {total_count} | Assigned #{i}/{len(tasks)}")
                    print(f"Started at: {datetime.now().isoformat(timespec='seconds')}")
                    print(f"Question: {preview}{suffix}")
                    print("-" * 100)

                    final_answer = ""
                    graphrag_answer = ""
                    naiverag_answer = ""
                    decision = ""
                    try:
                        result = rag.multi_agent_answer(question)
                        final_answer = (result or {}).get("final_answer", "") or ""
                        answers = (result or {}).get("answers", {}) or {}
                        graphrag_answer = answers.get("graphrag", "") or ""
                        naiverag_answer = answers.get("naiverag", "") or ""
                        decision = (result or {}).get("decision", "") or ""
                    except Exception as e:
                        final_answer = f"(error during multi_agent_answer: {e})"
                        decision = "error"
                        print(f"[ERROR] {e}", file=sys.stderr)

                    dur_single = _time.monotonic() - start_single
                    print("-" * 100)
                    print(f"Finished at: {datetime.now().isoformat(timespec='seconds')}")
                    print(f"Duration (this question): {format_duration(dur_single)}")
                    print(f"[Worker {worker_id}] Progress (within worker): processed {i} / {len(tasks)} | Left: {len(tasks) - i}")
                    print("=" * 100)

            out_record = {
                "id": qid,
                "question": question,
                "ground_truth": ground_truth,
                "graphrag_answer": graphrag_answer,
                "naiverag_answer": naiverag_answer,
                "final_answer": final_answer,
                "decision": decision
            }
            out_f.write(json.dumps(out_record, ensure_ascii=False) + "\n")
            out_f.flush()
            processed += 1

    dur = _time.monotonic() - start
    log(f"[Worker {worker_id}] DONE. Wrote {processed} record(s) to {part_path.name}. Duration: {format_duration(dur)}")

# ----------------- Main orchestration -----------------

def main():
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass

    here = Path(__file__).resolve().parent
    input_path = (here / QA_PAIRS_REL).resolve()
    output_dir = (here / OUTPUT_DIR_REL).resolve()
    logs_dir = here / PER_QUESTION_LOGS_DIRNAME

    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    ensure_output_dir_ready(output_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = output_dir / f"multi_agent_answers_{ts}.jsonl"

    all_records = list(iter_jsonl(input_path))
    total_records = len(all_records)

    log(f"Input file: {input_path}")
    log(f"Total records: {total_records}")

    if total_records == 0:
        log("No records to process. Exiting.")
        return

    # Apply limit with selection mode
    if MAX_QUESTIONS and MAX_QUESTIONS > 0:
        k = min(MAX_QUESTIONS, total_records)
        if SELECTION_MODE == "first":
            selected = all_records[:k]
            limit_info = f"limited to MAX_QUESTIONS={MAX_QUESTIONS} (selection=first)"
        elif SELECTION_MODE == "random":
            selected = random.sample(all_records, k=k)
            limit_info = f"limited to MAX_QUESTIONS={MAX_QUESTIONS} (selection=random)"
        else:
            selected = all_records[:k]
            limit_info = f"limited to MAX_QUESTIONS={MAX_QUESTIONS} (selection=first; invalid SELECTION_MODE ignored)"
    else:
        selected = all_records
        limit_info = "no limit (processing all records)"

    log(f"Processing {len(selected)} out of {total_records} ({limit_info}).")

    # Build tasks
    tasks = []
    for global_idx, rec in enumerate(selected, 1):
        question = (rec.get("question") or "").strip()
        if not question:
            continue
        tasks.append({
            "global_idx": global_idx,
            "id": rec.get("id", -1),
            "question": question,
            "ground_truth": rec.get("answer") or ""
        })

    total_tasks = len(tasks)
    if total_tasks == 0:
        log("No non-empty questions after filtering. Exiting.")
        return

    # Load GOOGLE_API_KEY_* from .env
    env_path = (here / ENV_PATH_REL).resolve()
    api_keys = load_google_api_keys(env_path)
    if api_keys:
        log(f"Found {len(api_keys)} GOOGLE_API_KEY_* entries in {env_path}: " +
            ", ".join([f"{label}={mask_key(val)}" for (label, val, _) in api_keys]))
    else:
        log(f"No GOOGLE_API_KEY_* entries found in {env_path}. Running with a single worker using module default / existing environment.")
        api_keys = [("DEFAULT", None, 0)]

    # Distribute tasks evenly
    n_workers = min(len(api_keys), total_tasks)
    api_keys = api_keys[:n_workers]
    task_chunks = chunk_evenly(tasks, n_workers)

    log("Worker assignment:")
    for i, ((label, key, _), chunk) in enumerate(zip(api_keys, task_chunks)):
        ids_preview = [t["id"] for t in chunk[:5]]
        more = "" if len(chunk) <= 5 else f" ... (+{len(chunk)-5} more)"
        log(f"  - Worker {i}: {label}={mask_key(key)} | tasks={len(chunk)} | first IDs={ids_preview}{more}")

    # Prepare workers
    worker_specs = []
    part_paths = []
    for i, ((label, key, _), chunk) in enumerate(zip(api_keys, task_chunks)):
        if not chunk:
            log(f"[Worker {i}] No tasks assigned; skipping spawn.")
            continue
        part_path = out_path.parent / f"{out_path.stem}.part{i}.jsonl"
        part_paths.append(part_path)
        worker_specs.append({
            "index": i,
            "label": label,
            "key": key,
            "chunk": chunk,
            "part_path": part_path,
        })

    total_workers = len(worker_specs)
    if total_workers == 0:
        log("No workers to launch. Exiting.")
        return

    effective_batch_size = total_workers if not BATCH_SIZE or BATCH_SIZE <= 0 else max(1, int(BATCH_SIZE))
    num_batches = (total_workers + effective_batch_size - 1) // effective_batch_size
    log(f"Launching {total_workers} worker(s) in {num_batches} batch(es) of up to {effective_batch_size} concurrent process(es).")

    overall_start = time.monotonic()
    for b in range(num_batches):
        start_idx = b * effective_batch_size
        end_idx = min(start_idx + effective_batch_size, total_workers)
        batch = worker_specs[start_idx:end_idx]
        log(f"Starting batch {b+1}/{num_batches}: worker indices {[spec['index'] for spec in batch]}")

        procs = []
        for spec in batch:
            i = spec["index"]
            label = spec["label"]
            key = spec["key"]
            chunk = spec["chunk"]
            part_path = spec["part_path"]
            p = mp.Process(
                target=worker_main,
                args=(i, label, key, chunk, total_tasks, str(logs_dir), str(part_path)),
                name=f"worker-{i}"
            )
            p.start()
            procs.append(p)
            log(f"Spawned worker {i} (pid={p.pid}) with {len(chunk)} task(s). Part: {part_path.name}")

        for p in procs:
            p.join()
            log(f"Worker process {p.name} (pid={p.pid}) joined with exitcode={p.exitcode}")

        log(f"Completed batch {b+1}/{num_batches}.")

    # Merge part files
    merged = 0
    with out_path.open("w", encoding="utf-8") as out_f:
        for part in part_paths:
            if not part.exists():
                log(f"[WARN] Part file missing (skipped): {part.name}")
                continue
            with part.open("r", encoding="utf-8") as pf:
                for line in pf:
                    s = line.strip()
                    if not s:
                        continue
                    out_f.write(s + "\n")
                    merged += 1

    dur = time.monotonic() - overall_start
    log(f"Done. Merged {merged} record(s) into: {out_path}")
    log(f"Total time: {format_duration(dur)}")

    if CLEAN_PART_FILES:
        for part in part_paths:
            try:
                part.unlink(missing_ok=True)
            except Exception:
                pass
        log(f"Cleaned up {len(part_paths)} part file(s).")

if __name__ == "__main__":
    main()