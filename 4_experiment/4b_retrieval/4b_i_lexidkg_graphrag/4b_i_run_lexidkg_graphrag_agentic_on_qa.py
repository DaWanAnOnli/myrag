#!/usr/bin/env python3
"""
Parallel batch runner for agentic_graph_rag over QA pairs.

Key features:
- Reads QA pairs JSONL from underscore path only:
    - Sample: ../../../dataset/samples/4_experiment/4a_qa_generation/4a_iii_qa_pairs_with_id/qa_pairs.jsonl
    - Full:   ../../../dataset/4_experiment/4a_qa_generation/4a_iii_qa_pairs_with_id/qa_pairs.jsonl
- Filters to verification_choice == "A".
- Selection mode: "first" or "random", limited by MAX_QUESTIONS.
- Loads GOOGLE_API_KEY_1, GOOGLE_API_KEY_2, ... from ../../../.env.
- Spawns workers in shifts:
    * Number of concurrent workers per shift is controlled by PROCESSES_PER_SHIFT.
    * Each worker uses its own API key and gets its preassigned subset of questions.
    * Waits for all workers in a shift to finish before launching the next shift.
- Each worker:
    * Sets its own GOOGLE_API_KEY (both env var and rag.GOOGLE_API_KEY constant).
    * Captures terminal output per question into question_terminal_logs.
    * Writes results to a per-worker .part JSONL file in the output directory.
- Parent merges all .part files into one combined JSONL:
    ../dataset/samples/4_experiment/4b_experiment_answers/4b_i_lexidkg_graphrag/graph_rag_answers_<timestamp>.jsonl

Output JSONL per line:
  { "id": <int>, "question": <str>, "ground_truth": <str>, "generated_answer": <str> }
"""

import json
import sys
import time
import shutil
import os
from pathlib import Path
from datetime import datetime
import contextlib
import re
import random
import multiprocessing as mp

import dotenv

# ----------------- Hardcoded testing limit and shift size -----------------

env_file_path = Path("../../../.env")
    
# Load the .env file
if not env_file_path.exists():
    raise(ImportError(f"Error: .env file not found at {env_file_path}"))

dotenv.load_dotenv(env_file_path)

is_sample = os.getenv('IS_SAMPLE', '').lower()

if is_sample == "true":
    IS_SAMPLE = True
elif is_sample == "false":
    IS_SAMPLE = False
else:
    raise(ValueError(f"Wrong configuration of IS_SAMPLE in .env file: {is_sample}"))

# Set to a positive integer to limit how many 'A' questions get processed.
# Set to 0 or None to process all available 'A' questions.
MAX_QUESTIONS = None

# Selection mode (hardcoded):
#   "first"  -> pick the first MAX_QUESTIONS 'A' questions
#   "random" -> pick MAX_QUESTIONS random 'A' questions (uniform, without replacement)
SELECTION_MODE = "random"  # change to "first" to select deterministically

# New: number of processes to run concurrently per shift
PROCESSES_PER_SHIFT = 9  # Adjust this to control shift size

# ----------------- Paths (underscore-only) -----------------

if IS_SAMPLE:
    QA_PAIRS_REL = "../../../dataset/samples/4_experiment/4a_qa_generation/4a_iii_qa_pairs_with_id/qa_pairs.jsonl"
    ANSWERS_OUT_REL = "../../../dataset/samples/4_experiment/4b_experiment_answers/4b_i_lexidkg_graphrag"
else:
    QA_PAIRS_REL = "../../../dataset/4_experiment/4a_qa_generation/4a_iii_qa_pairs_with_id/qa_pairs.jsonl"
    ANSWERS_OUT_REL = "../../../dataset/4_experiment/4b_experiment_answers/4b_i_lexidkg_graphrag"

PER_QUESTION_LOGS_DIRNAME = "question_terminal_logs"
ENV_PATH_REL = "../../../.env"
CLEAN_PART_FILES = True  # remove worker part files after merging

# ------------- Utilities and logging helpers -------------

def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

def mask_key(key: str, show_prefix: int = 4, show_suffix: int = 2) -> str:
    if not key:
        return "<empty>"
    if len(key) <= show_prefix + show_suffix:
        return "<hidden>"
    return f"{key[:show_prefix]}...{key[-show_suffix:]}"

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
    t = text.strip()
    t = re.sub(r"\s+", " ", t)
    t = t[:max_len]
    t = re.sub(r"[^a-zA-Z0-9\-_. ]", "_", t)
    t = t.strip().replace(" ", "_")
    return t or "question"

class TeeIO:
    """Simple tee that writes to both a primary stream and a file handle."""
    def __init__(self, primary, secondary_file):
        self.primary = primary
        self.secondary = secondary_file

    def write(self, data):
        try:
            self.primary.write(data)
        except Exception:
            pass
        try:
            self.secondary.write(data)
        except Exception:
            pass

    def flush(self):
        try:
            self.primary.flush()
        except Exception:
            pass
        try:
            self.secondary.flush()
        except Exception:
            pass

def ensure_output_dir_ready(output_dir: Path):
    """Warn if not empty, ask for confirmation, and wipe it if confirmed."""
    output_dir.mkdir(parents=True, exist_ok=True)
    not_empty = any(output_dir.iterdir())
    if not_empty:
        print(f"WARNING: Output directory is not empty:\n  {output_dir}")
        print("Continuing will OVERWRITE (delete) all files in this folder.")
        resp = input("Type 'YES' to confirm and continue, or anything else to abort: ").strip()
        if resp != "YES":
            print("Aborted by user. No changes made.")
            sys.exit(0)
        # Wipe directory
        shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory cleared: {output_dir}")

def load_google_api_keys(env_path: Path):
    """
    Returns a list of tuples: [(label, key, numeric_suffix), ...]
    where label is like 'GOOGLE_API_KEY_1'. Sorted by numeric_suffix asc.
    """
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
            # Strip optional surrounding quotes
            if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
                val = val[1:-1]
            keys.append((label, val, num))
    keys.sort(key=lambda x: x[2])
    return keys

def chunk_evenly(items, n_chunks):
    """
    Split 'items' into n_chunks lists with sizes as equal as possible.
    """
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

# ----------------- Worker process -----------------

def worker_main(worker_id: int,
                key_label: str,
                key_value: str,
                tasks: list,
                total_count: int,
                logs_dir: str,
                part_path: str):
    """
    Worker entry point.
    Each 'task' is a dict: {
        "global_idx": int,
        "id": int,
        "question": str,
        "ground_truth": str
    }
    """
    start = time.monotonic()
    logs_dir = Path(logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)
    part_path = Path(part_path)

    # Configure key for this process
    if key_value:
        os.environ["GOOGLE_API_KEY"] = key_value

    # Import after setting env so the module can also read env if it does.
    try:
        import lexidkg_graphrag as rag
    except Exception as e:
        log(f"[Worker {worker_id}] ERROR: Could not import lexidkg_graphrag_agentic.py: {e}")
        # Still write an empty part file so parent can proceed
        part_path.write_text("", encoding="utf-8")
        return

    # Force the constant override, in case the module doesn't read from env.
    try:
        if key_value:
            setattr(rag, "GOOGLE_API_KEY", key_value)
    except Exception as e:
        log(f"[Worker {worker_id}] WARN: Could not set rag.GOOGLE_API_KEY: {e}")

    masked = mask_key(key_value) if key_value else "(module default / env)"
    log(f"[Worker {worker_id}] Using {key_label} = {masked}. Assigned {len(tasks)} task(s). Part file: {part_path.name}")

    processed = 0
    with part_path.open("w", encoding="utf-8") as out_f:
        for i, task in enumerate(tasks, 1):
            global_idx = task["global_idx"]
            qid = task["id"]
            question = task["question"]
            ground_truth = task["ground_truth"]

            # Prepare per-question log file and tee the outputs
            preview = (question or "")[:120]
            suffix = "..." if question and len(question) > 120 else ""
            q_snippet = sanitize_snippet(question or "", max_len=60)
            q_ts = datetime.now().strftime("%Y%m%d-%H%M%S")
            log_file = logs_dir / f"q{global_idx:04d}_wid{worker_id}_id{qid}_{q_ts}_{q_snippet}.txt"

            start_single = time.monotonic()
            with log_file.open("w", encoding="utf-8") as lf:
                tee_out = TeeIO(sys.stdout, lf)
                tee_err = TeeIO(sys.stderr, lf)
                with contextlib.redirect_stdout(tee_out), contextlib.redirect_stderr(tee_err):
                    print("=" * 100)
                    print(f"[Worker {worker_id}] Question global #{global_idx} (ID: {qid}) | Total: {total_count} | Assigned #{i}/{len(tasks)}")
                    print(f"Started at: {datetime.now().isoformat(timespec='seconds')}")
                    print(f"Question: {preview}{suffix}")
                    print("-" * 100)

                    generated_answer = ""
                    error_msg = None
                    try:
                        result = rag.agentic_graph_rag(question)
                        generated_answer = (result or {}).get("final_answer", "")
                        if not isinstance(generated_answer, str):
                            generated_answer = json.dumps(generated_answer, ensure_ascii=False)
                    except Exception as e:
                        error_msg = str(e)
                        generated_answer = f"(error during agentic_graph_rag: {error_msg})"
                        print(f"[ERROR] {error_msg}", file=sys.stderr)

                    dur_single = time.monotonic() - start_single
                    print("-" * 100)
                    print(f"Finished at: {datetime.now().isoformat(timespec='seconds')}")
                    print(f"Duration (this question): {format_duration(dur_single)}")
                    print(f"[Worker {worker_id}] Progress (within worker): processed {i} / {len(tasks)} | Left: {len(tasks) - i}")
                    print("=" * 100)

            # Write JSONL output record (streaming)
            out_record = {
                "id": qid,
                "question": question,
                "ground_truth": ground_truth,
                "generated_answer": generated_answer
            }
            out_f.write(json.dumps(out_record, ensure_ascii=False) + "\n")
            out_f.flush()
            processed += 1

    # Best-effort cleanup of Neo4j driver created in graph_rag_agentic
    try:
        if hasattr(rag, "driver") and rag.driver is not None:
            rag.driver.close()
    except Exception:
        pass

    dur = time.monotonic() - start
    log(f"[Worker {worker_id}] DONE. Wrote {processed} record(s) to {part_path.name}. Duration: {format_duration(dur)}")

# ----------------- Main orchestration -----------------

def main():
    # Use 'spawn' to avoid surprises with module state across processes
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        # It's okay if already set
        pass

    here = Path(__file__).resolve().parent

    # Resolve underscore-only input file path
    input_path = (here / QA_PAIRS_REL).resolve()
    output_dir = (here / ANSWERS_OUT_REL).resolve()
    per_question_logs_dir = here / PER_QUESTION_LOGS_DIRNAME
    per_question_logs_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    # Ensure output directory is ready (warn and confirm if not empty)
    ensure_output_dir_ready(output_dir)

    # Load all QA records
    all_records = list(iter_jsonl(input_path))
    total_records = len(all_records)

    log(f"Input file: {input_path}")
    log(f"Total QA pairs in input: {total_records}")

    if total_records == 0:
        log("No records to process. Exiting.")
        return

    # Apply hardcoded limit with selection mode (no filtering; previous step already filtered upstream)
    if MAX_QUESTIONS is None or MAX_QUESTIONS <= 0:
        to_process_records = all_records
        limit_info = "no limit (processing all records)"
    else:
        k = min(MAX_QUESTIONS, len(all_records))
        if SELECTION_MODE == "first":
            to_process_records = all_records[:k]
            limit_info = f"limited to MAX_QUESTIONS={MAX_QUESTIONS} (selection=first)"
        elif SELECTION_MODE == "random":
            to_process_records = random.sample(all_records, k=k)
            limit_info = f"limited to MAX_QUESTIONS={MAX_QUESTIONS} (selection=random)"
        else:
            to_process_records = all_records[:k]
            limit_info = f"limited to MAX_QUESTIONS={MAX_QUESTIONS} (selection=first; invalid SELECTION_MODE='{SELECTION_MODE}' ignored)"

    log(f"Processing {len(to_process_records)} out of {total_records} questions ({limit_info}).")

    # Prepare output file name
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = output_dir / f"graph_rag_answers_{ts}.jsonl"

    # Load GOOGLE_API_KEY_* from .env
    env_path = (here / ENV_PATH_REL).resolve()
    api_keys = load_google_api_keys(env_path)
    if api_keys:
        log(f"Found {len(api_keys)} GOOGLE_API_KEY_* entries in {env_path}: " +
            ", ".join([f"{label}={mask_key(val)}" for (label, val, _) in api_keys]))
    else:
        log(f"No GOOGLE_API_KEY_* entries found in {env_path}. Running with a single worker using module default / existing environment.")
        api_keys = [("DEFAULT", None, 0)]

    # Build task list with stable global indices for logging and ordering
    tasks = []
    for global_idx, rec in enumerate(to_process_records, 1):
        question = (rec.get("question") or "").strip()
        if not question:
            # skip empty question; keep indices contiguous for logs
            continue
        task = {
            "global_idx": global_idx,
            "id": rec.get("id", -1),
            "question": question,
            "ground_truth": rec.get("answer") or ""
        }
        tasks.append(task)

    total_tasks = len(tasks)
    if total_tasks == 0:
        log("No non-empty questions after filtering. Exiting.")
        return

    # Determine how many workers (one per API key) we will use overall
    total_workers = min(len(api_keys), total_tasks)  # don't create workers that would get 0 tasks
    api_keys = api_keys[:total_workers]

    # Distribute tasks evenly across all workers (keys)
    task_chunks = chunk_evenly(tasks, total_workers)
    assignments = list(zip(api_keys, task_chunks))  # [( (label, key, idx), [tasks_for_worker] ), ...]

    # Print assignment overview by shifts
    log("Worker assignment (by shifts):")
    num_shifts = (total_workers + PROCESSES_PER_SHIFT - 1) // PROCESSES_PER_SHIFT
    for shift_idx in range(num_shifts):
        start = shift_idx * PROCESSES_PER_SHIFT
        end = min(start + PROCESSES_PER_SHIFT, total_workers)
        log(f"  - Shift {shift_idx+1}/{num_shifts}: workers {start}..{end-1}")
        for i in range(start, end):
            (label, key, num), chunk = assignments[i]
            ids_preview = [t["id"] for t in chunk[:5]]
            more = "" if len(chunk) <= 5 else f" ... (+{len(chunk)-5} more)"
            log(f"      Worker {i}: {label}={mask_key(key)} | tasks={len(chunk)} | first IDs={ids_preview}{more}")

    # Launch workers shift-by-shift
    part_paths = []
    overall_start = time.monotonic()
    for shift_idx in range(num_shifts):
        start = shift_idx * PROCESSES_PER_SHIFT
        end = min(start + PROCESSES_PER_SHIFT, total_workers)
        current_assignments = list(enumerate(assignments[start:end], start=start))

        log(f"Launching Shift {shift_idx+1}/{num_shifts} with up to {PROCESSES_PER_SHIFT} process(es).")
        procs = []

        # Spawn processes for this shift
        for i, ((label, key, _), chunk) in current_assignments:
            if not chunk:
                log(f"[Worker {i}] No tasks assigned; skipping spawn.")
                continue
            part_path = out_path.parent / f"{out_path.stem}.part{i}.jsonl"
            part_paths.append(part_path)
            p = mp.Process(
                target=worker_main,
                args=(i, label, key, chunk, total_tasks, str(per_question_logs_dir), str(part_path)),
                name=f"worker-{i}"
            )
            p.start()
            procs.append(p)
            log(f"Spawned worker {i} (pid={p.pid}) with {len(chunk)} task(s). Part: {part_path.name} [Shift {shift_idx+1}]")

        # Wait for this shift to finish
        for p in procs:
            p.join()
            log(f"Worker process {p.name} (pid={p.pid}) joined with exitcode={p.exitcode} [Shift {shift_idx+1}]")

        log(f"Shift {shift_idx+1}/{num_shifts} complete.")

    # Merge part files into single output
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
    log(f"Total time for all processed questions: {format_duration(dur)}")

    # Cleanup part files
    if CLEAN_PART_FILES:
        for part in part_paths:
            try:
                part.unlink(missing_ok=True)
            except Exception:
                pass
        log(f"Cleaned up {len(part_paths)} part file(s).")

if __name__ == "__main__":
    main()