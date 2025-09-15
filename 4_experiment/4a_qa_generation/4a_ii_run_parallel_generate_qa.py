#!/usr/bin/env python3
"""
Parallel runner for generate_qa.py.

What it does:
- Detects batch folders under: ../dataset/samples/json-batches-for-qa
  (i.e., batch-001, batch-002, ...), relative to this script.
- Reads API keys GOOGLE_API_KEY_1, GOOGLE_API_KEY_2, ... from the repository .env
  located at ../../.env relative to this script, and maps key N to batch-N.
- Spawns subprocesses using the venv at ../env, running generate_qa.py
  with:
    - an explicit --file-list pointing to that batch's *.json files
    - a per-process output JSONL path:
      ../dataset/samples/qa_pairs/qa_pairs.batch-XXX.jsonl
  The environment variable GOOGLE_API_KEY is set per process from the mapped key.
- Streams each process's output live to the terminal with color-coded prefixes
  like [PROCESS-1], and simultaneously writes to per-process log files.
- Runs processes in "shifts": only SHIFT_SIZE processes run concurrently.
  After a shift completes, the next shift starts, and so on.
- When all processes finish, combines per-batch JSONL files into:
    ../dataset/samples/qa_pairs/qa_pairs_combined.jsonl
  and also writes a JSON array version:
    ../dataset/samples/qa_pairs/qa_pairs_combined.json

Error cases:
- If number of keys does not equal number of batch folders, exit with error.
- If any required paths are missing (venv, generate_qa.py, batches), exit with error.
- If SHIFT_SIZE < 1, exit with error.

Requirements:
- No third-party deps. ANSI colors used for most terminals.

Usage:
  python run_parallel_generate_qa.py
"""

import os
import re
import sys
import json
import time
import queue
import signal
import threading
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

import dotenv

# ----------------------------
# Paths, constants, and config
# ----------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PARENT_DIR = SCRIPT_DIR.parent

# Hardcoded shift size: how many generate_qa instances to run at once
SHIFT_SIZE = 6  # Set this to the desired concurrent process count per "shift"

# .env file location relative to this script (repo-level .env)
env_file_path = (SCRIPT_DIR / ".." / ".." / ".env").resolve()

# Load the .env file for IS_SAMPLE and other environment variables
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

if IS_SAMPLE:
    BATCH_BASE = (SCRIPT_DIR / ".." / ".." / "dataset" / "samples" / "4_experiment" / "4a_qa_generation" / "4a_i_json_batches_for_qa").resolve()
    QA_OUTPUT_DIR = (SCRIPT_DIR / ".." / ".." / "dataset" / "samples" / "4_experiment" / "4a_qa_generation" / "4a_ii_qa_pairs").resolve()
else:
    BATCH_BASE = (SCRIPT_DIR / ".." / ".." / "dataset" / "4_experiment" / "4a_qa_generation" / "4a_i_json_batches_for_qa").resolve()
    QA_OUTPUT_DIR = (SCRIPT_DIR / ".." / ".." / "dataset" / "4_experiment" / "4a_qa_generation" / "4a_ii_qa_pairs").resolve()

# generate_qa.py must be next to this script
GENERATE_QA = (SCRIPT_DIR / "generate_qa.py").resolve()

# Virtual environment python (../env)
if os.name == "nt":
    VENV_PY = (PARENT_DIR / ".." / "env" / "Scripts" / "python.exe").resolve()
else:
    VENV_PY = (PARENT_DIR / ".." / "env" / "bin" / "python").resolve()

# Local helper dirs
FILE_LISTS_DIR = (SCRIPT_DIR / "file_lists").resolve()
LOGS_DIR = (SCRIPT_DIR / "runner_logs").resolve()

# ANSI colors (cycled)
ANSI_COLORS = [
    "\033[31m",  # red
    "\033[32m",  # green
    "\033[33m",  # yellow
    "\033[34m",  # blue
    "\033[35m",  # magenta
    "\033[36m",  # cyan
    "\033[91m",  # bright red
    "\033[92m",  # bright green
    "\033[93m",  # bright yellow
    "\033[94m",  # bright blue
    "\033[95m",  # bright magenta
    "\033[96m",  # bright cyan
]
ANSI_RESET = "\033[0m"

PRINT_LOCK = threading.Lock()


# ----------------------------
# Utilities
# ----------------------------
def abort(msg: str, code: int = 1) -> None:
    print(f"Error: {msg}", file=sys.stderr, flush=True)
    sys.exit(code)


def parse_env_file(env_path: Path) -> Dict[str, str]:
    """Minimal .env parser for KEY=VALUE lines (supports optional quotes and 'export')."""
    env: Dict[str, str] = {}
    if not env_path.exists():
        return env
    try:
        for raw in env_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.lower().startswith("export "):
                line = line[7:].strip()
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            env[k] = v
    except Exception as e:
        abort(f"Failed to read .env at {env_path}: {e}")
    return env


def find_batches(base: Path) -> List[Tuple[int, Path]]:
    """Return list of (index, path) for batch-XXX folders sorted by index."""
    if not base.exists() or not base.is_dir():
        abort(f"Batch base directory not found: {base}")
    batches: List[Tuple[int, Path]] = []
    for p in base.iterdir():
        if p.is_dir():
            m = re.fullmatch(r"batch-(\d{3})", p.name)
            if m:
                idx = int(m.group(1))
                batches.append((idx, p.resolve()))
    batches.sort(key=lambda t: t[0])
    if not batches:
        abort(f"No batch folders found under: {base}")
    return batches


def discover_json_files(batch_dir: Path) -> List[Path]:
    files = sorted([p.resolve() for p in batch_dir.rglob("*.json") if p.is_file()])
    return files


def write_file_list(idx: int, batch_name: str, files: List[Path]) -> Path:
    FILE_LISTS_DIR.mkdir(parents=True, exist_ok=True)
    path = FILE_LISTS_DIR / f"{batch_name}.txt"
    with path.open("w", encoding="utf-8") as f:
        for p in files:
            f.write(str(p) + "\n")
    return path


def ensure_paths() -> None:
    if not GENERATE_QA.exists():
        abort(f"generate_qa.py not found at: {GENERATE_QA}")
    if not VENV_PY.exists():
        abort(f"Virtualenv python not found at: {VENV_PY}\n"
              f"Expected venv at: {PARENT_DIR / 'env'}")
    QA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)


def build_cmd(out_jsonl: Path, file_list: Path) -> List[str]:
    # -u for unbuffered output to stream logs live
    return [str(VENV_PY), "-u", str(GENERATE_QA), str(out_jsonl), "--file-list", str(file_list)]


def color_for(i: int) -> str:
    return ANSI_COLORS[(i - 1) % len(ANSI_COLORS)]


def prefix_for(i: int) -> str:
    return f"[PROCESS-{i}] "


def tee_stream(proc: subprocess.Popen, proc_idx: int, log_path: Path, color: str) -> None:
    """Read process stdout live, write to console (prefixed/colorized) and to file."""
    prefix = prefix_for(proc_idx)
    with log_path.open("w", encoding="utf-8") as log_f:
        for line in iter(proc.stdout.readline, ""):
            if line == "":
                break
            line = line.rstrip("\r\n")
            with PRINT_LOCK:
                print(f"{color}{prefix}{line}{ANSI_RESET}", flush=True)
            log_f.write(line + "\n")
            log_f.flush()


def combine_jsonl(per_batch_paths: List[Path], combined_jsonl: Path, combined_json_array: Path) -> Tuple[int, int]:
    """Combine JSONL files into one JSONL and a JSON array; returns (lines, records)."""
    total_lines = 0
    all_records = []
    with combined_jsonl.open("w", encoding="utf-8") as out_f:
        for p in per_batch_paths:
            if not p.exists():
                continue
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.rstrip("\n")
                    if not line:
                        continue
                    out_f.write(line + "\n")
                    total_lines += 1
                    try:
                        all_records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    with combined_json_array.open("w", encoding="utf-8") as jf:
        json.dump(all_records, jf, ensure_ascii=False, indent=2)
    return total_lines, len(all_records)


def main():
    ensure_paths()

    # Validate shift size
    if not isinstance(SHIFT_SIZE, int) or SHIFT_SIZE < 1:
        abort(f"Invalid SHIFT_SIZE={SHIFT_SIZE}. Must be an integer >= 1.")

    # Locate batches
    batches = find_batches(BATCH_BASE)  # [(idx, path)]
    n_batches = len(batches)

    # Load API keys from .env (consistent, script-relative path)
    env_map = parse_env_file(env_file_path)
    keys: Dict[int, str] = {}
    for i in range(1, n_batches + 1):
        k = f"GOOGLE_API_KEY_{i}"
        if k in env_map and env_map[k].strip():
            keys[i] = env_map[k].strip()

    # Validate key count and mapping 1..N
    if len(keys) != n_batches:
        missing = [i for i in range(1, n_batches + 1) if i not in keys]
        present = sorted(keys.keys())
        abort(
            "Mismatch between number of API keys and number of batch folders.\n"
            f"- Batches found: {n_batches} ({', '.join([f'batch-{i:03d}' for i, _ in batches])})\n"
            f"- Keys present: {len(keys)} ({', '.join([f'GOOGLE_API_KEY_{i}' for i in present])})\n"
            f"- Missing keys: {', '.join([f'GOOGLE_API_KEY_{i}' for i in missing]) if missing else 'None'}\n"
            "Define keys in ../.env as GOOGLE_API_KEY_1, GOOGLE_API_KEY_2, ... matching batch indices."
        )

    # Build file lists and process specs
    per_proc_specs = []  # list of dicts with: idx, batch_name, file_list, out_jsonl, log_path, env
    for idx, batch_dir in batches:
        batch_name = f"batch-{idx:03d}"
        files = discover_json_files(batch_dir)
        if not files:
            abort(f"No JSON files found in {batch_dir}")

        file_list = write_file_list(idx, batch_name, files)
        out_jsonl = QA_OUTPUT_DIR / f"qa_pairs.{batch_name}.jsonl"
        log_path = LOGS_DIR / f"process-{idx:03d}.log"

        # Environment for this subprocess
        proc_env = os.environ.copy()
        proc_env["GOOGLE_API_KEY"] = keys[idx]
        proc_env["PYTHONUNBUFFERED"] = "1"

        per_proc_specs.append(
            {
                "idx": idx,
                "batch_name": batch_name,
                "file_list": file_list,
                "out_jsonl": out_jsonl,
                "log_path": log_path,
                "env": proc_env,
                "color": color_for(idx),
            }
        )

    # Shifts execution
    start_time = time.time()
    total = len(per_proc_specs)
    total_shifts = (total + SHIFT_SIZE - 1) // SHIFT_SIZE
    print(f"Launching {total} processes in shifts of {SHIFT_SIZE} using venv Python at: {VENV_PY}", flush=True)

    # Ensure child processes are killed on Ctrl+C (for current shift)
    stop_event = threading.Event()
    procs: List[subprocess.Popen] = []  # Will be re-assigned per shift

    def handle_sigint(sig, frame):
        with PRINT_LOCK:
            print("\nReceived interrupt; terminating child processes...", flush=True)
        stop_event.set()
        for p in procs:
            if p.poll() is None:
                try:
                    p.terminate()
                except Exception:
                    pass

    signal.signal(signal.SIGINT, handle_sigint)

    all_exit_codes: List[int] = []

    try:
        for s_idx in range(total_shifts):
            start = s_idx * SHIFT_SIZE
            end = min(start + SHIFT_SIZE, total)
            chunk = per_proc_specs[start:end]

            procs = []
            threads = []

            print(f"\n=== Shift {s_idx + 1}/{total_shifts} "
                  f"({len(chunk)} processes: {chunk[0]['batch_name']}..{chunk[-1]['batch_name']}) ===",
                  flush=True)

            # Start processes for this shift
            for spec in chunk:
                cmd = build_cmd(spec["out_jsonl"], spec["file_list"])
                p = subprocess.Popen(
                    cmd,
                    cwd=str(SCRIPT_DIR),
                    env=spec["env"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                procs.append(p)
                t = threading.Thread(
                    target=tee_stream,
                    args=(p, spec["idx"], spec["log_path"], spec["color"]),
                    daemon=True,
                )
                t.start()
                threads.append(t)

            # Wait for processes in this shift to finish
            shift_exit_codes = []
            for p in procs:
                shift_exit_codes.append(p.wait())

            # Ensure all logging threads drain for this shift
            for t in threads:
                t.join(timeout=2.0)

            all_exit_codes.extend(shift_exit_codes)

    finally:
        # If any still running (e.g., on exception), try to kill them
        for p in procs:
            if p.poll() is None:
                try:
                    p.kill()
                except Exception:
                    pass

    elapsed = time.time() - start_time
    any_fail = any(code != 0 for code in all_exit_codes)

    # Report status
    for spec, code in zip(per_proc_specs, all_exit_codes):
        status = "OK" if code == 0 else f"EXIT={code}"
        print(f"{prefix_for(spec['idx'])}{status} | output={spec['out_jsonl']} | log={spec['log_path']}", flush=True)

    if any_fail:
        abort("One or more processes exited with non-zero status. See per-process logs for details.", code=2)

    # Combine JSONL outputs
    combined_jsonl = QA_OUTPUT_DIR / "qa_pairs_combined.jsonl"
    combined_json = QA_OUTPUT_DIR / "qa_pairs_combined.json"
    per_batch_paths = [spec["out_jsonl"] for spec in per_proc_specs]

    lines, records = combine_jsonl(per_batch_paths, combined_jsonl, combined_json)
    print(f"\nCombined {len(per_batch_paths)} files into:", flush=True)
    print(f"- {combined_jsonl} (JSONL, {lines} lines)", flush=True)
    print(f"- {combined_json} (JSON array, {records} records)", flush=True)
    print(f"Total elapsed: {elapsed:.2f}s", flush=True)


if __name__ == "__main__":
    main()