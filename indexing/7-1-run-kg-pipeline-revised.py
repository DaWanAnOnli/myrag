#!/usr/bin/env python3
import os
import re
import sys
import signal
import threading
import subprocess
import time
from pathlib import Path
from dotenv import load_dotenv  # pip install python-dotenv

# -------------- Paths (relative to this script) --------------
BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent  # parent directory of this script
# BATCHES_ROOT = (BASE_DIR / "../dataset/samples/langchain-batches").resolve()
# JSON_OUTPUT_DIR = (BASE_DIR / "../dataset/samples/llm-json-outputs").resolve()
BATCHES_ROOT = (BASE_DIR / "../dataset/langchain-batches").resolve()
JSON_OUTPUT_DIR = (BASE_DIR / "../dataset/llm-json-outputs").resolve()
KG_PIPELINE_PATH = (BASE_DIR / "kg_pipeline.py").resolve()
TMP_SCRIPT_PREFIX = "_kg_pipeline_run_"
LOGS_DIR = (BASE_DIR / "run_logs").resolve()

# How many runs to execute at the same time
BATCH_SIZE = 5

# ANSI colors for per-process tagging
COLORS = [
    "\x1b[31m", "\x1b[32m", "\x1b[33m", "\x1b[34m", "\x1b[35m", "\x1b[36m",
    "\x1b[37m", "\x1b[91m", "\x1b[92m", "\x1b[93m", "\x1b[94m", "\x1b[95m", "\x1b[96m",
]
RESET = "\x1b[0m"

PRINT_LOCK = threading.Lock()

def colorize(i: int, stream_name: str, line: str) -> str:
    color = COLORS[(i - 1) % len(COLORS)]
    prefix = f"[RUN {i:02d} {stream_name}]"
    return f"{color}{prefix} {line.rstrip()}{RESET}"

def eprint(msg: str):
    with PRINT_LOCK:
        print(f"\x1b[31m[ERROR]\x1b[0m {msg}", file=sys.stderr)

def wprint(msg: str):
    with PRINT_LOCK:
        print(f"\x1b[33m[WARN]\x1b[0m {msg}")

def iprint(msg: str):
    with PRINT_LOCK:
        print(f"\x1b[36m[INFO]\x1b[0m {msg}")

def dprint(msg: str):
    with PRINT_LOCK:
        print(f"\x1b[90m[DEBUG]\x1b[0m {msg}")

def load_env_from_parent():
    env_path = REPO_ROOT / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
        iprint(f"Loaded .env from {env_path}")
    else:
        wprint(f"No .env found at {env_path}; relying on existing OS environment variables.")

def confirm_json_output_dir(skip_confirm: bool = False):
    if not JSON_OUTPUT_DIR.exists():
        iprint(f"Creating JSON output directory: {JSON_OUTPUT_DIR}")
        JSON_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        return
    if any(JSON_OUTPUT_DIR.iterdir()):
        wprint(f"JSON output directory exists and is not empty: {JSON_OUTPUT_DIR}")
        if skip_confirm:
            iprint("--yes provided; continuing despite non-empty directory.")
            return
        ans = input("Proceed and continue writing into this directory? [y/N]: ").strip().lower()
        if ans not in ("y", "yes"):
            eprint("Aborting by user choice.")
            sys.exit(1)

def list_numeric_subfolders(root: Path):
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Batches root not found: {root}")
    nums = []
    for p in root.iterdir():
        if p.is_dir() and p.name.isdigit():
            nums.append(int(p.name))
    if not nums:
        raise RuntimeError(f"No numeric subfolders found under {root}. Expected 1..n (e.g., 1, 2, 3).")
    nums = sorted(nums)
    expected = list(range(1, len(nums) + 1))
    if nums != expected:
        raise RuntimeError(f"Expected contiguous numeric folders 1..{len(nums)} under {root}, but found: {nums}")
    return nums

def ensure_api_keys_for(n: int):
    missing = []
    for i in range(1, n + 1):
        if not os.environ.get(f"GOOGLE_API_KEY_{i}"):
            missing.append(i)
    extra_ids = sorted(
        int(m.group(1))
        for name in os.environ
        for m in [re.match(r"^GOOGLE_API_KEY_(\d+)$", name)]
        if m and int(m.group(1)) > n
    )
    if missing or extra_ids:
        parts = []
        if missing:
            parts.append(f"missing GOOGLE_API_KEY_{{{', '.join(map(str, missing))}}}")
        if extra_ids:
            parts.append(f"extra keys present: {', '.join(map(str, extra_ids))} (env has more keys than folders)")
        raise RuntimeError("Mismatch between folder count and GOOGLE_API_KEY_* in env: " + "; ".join(parts))
    return True

def read_kg_pipeline_source() -> str:
    if not KG_PIPELINE_PATH.exists():
        raise FileNotFoundError(f"kg_pipeline.py not found at {KG_PIPELINE_PATH}")
    return KG_PIPELINE_PATH.read_text(encoding="utf-8")

def write_modified_script(i: int, lang_dir: Path, source: str) -> Path:
    """
    Creates a sibling copy of kg_pipeline.py named _kg_pipeline_run_{i}.py
    with:
      - GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY_{i}"]
      - DEFAULT_LANGCHAIN_DIR = Path(r"...").resolve() pointing to folder i
    """
    text = source

    # Patch GOOGLE_API_KEY line
    pat_api = re.compile(r'GOOGLE_API_KEY\s*=\s*os\.environ\[\s*([\'"])GOOGLE_API_KEY\1\s*\]')
    def repl_api(_match):
        return f'GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY_{i}"]'
    text, n_api = pat_api.subn(repl_api, text)
    if n_api == 0:
        raise RuntimeError("Failed to patch GOOGLE_API_KEY line in kg_pipeline.py")

    # Patch DEFAULT_LANGCHAIN_DIR line
    pat_dir = re.compile(r'^DEFAULT_LANGCHAIN_DIR\s*=\s*.*$', re.MULTILINE)
    def repl_dir(_match):
        return f'DEFAULT_LANGCHAIN_DIR = Path(r"{str(lang_dir)}").resolve()'
    text, n_dir = pat_dir.subn(repl_dir, text, count=1)
    if n_dir == 0:
        raise RuntimeError("Failed to patch DEFAULT_LANGCHAIN_DIR in kg_pipeline.py")

    out_path = KG_PIPELINE_PATH.parent / f"{TMP_SCRIPT_PREFIX}{i}.py"
    out_path.write_text(text, encoding="utf-8")
    return out_path

def pump_stream(stream, run_id: int, stream_name: str, log_fh, log_lock: threading.Lock):
    try:
        for line in iter(stream.readline, ""):
            with PRINT_LOCK:
                print(colorize(run_id, stream_name, line), flush=True)
            with log_lock:
                log_fh.write(f"[{stream_name}] {line}")
                log_fh.flush()
    finally:
        try:
            stream.close()
        except Exception:
            pass

def launch_process(run_id: int, script_path: Path, json_out_dir: Path, key_id: int):
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["JSON_OUTPUT_DIR"] = str(json_out_dir)

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    log_path = LOGS_DIR / f"run_{run_id:02d}_key{key_id}_{ts}.log"
    log_fh = open(log_path, "w", encoding="utf-8", buffering=1)
    log_lock = threading.Lock()

    cmd = [sys.executable, str(script_path)]
    iprint(f"Launching RUN {run_id:02d}: {cmd} [API=GOOGLE_API_KEY_{key_id}] (log: {log_path})")

    proc = subprocess.Popen(
        cmd,
        cwd=str(KG_PIPELINE_PATH.parent),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        env=env,
    )
    t_out = threading.Thread(target=pump_stream, args=(proc.stdout, run_id, "OUT", log_fh, log_lock), daemon=True)
    t_err = threading.Thread(target=pump_stream, args=(proc.stderr, run_id, "ERR", log_fh, log_lock), daemon=True)
    t_out.start()
    t_err.start()
    return proc, t_out, t_err, log_fh, log_path

def main():
    skip_confirm = "--yes" in sys.argv or "-y" in sys.argv

    # Load .env from parent directory of this script
    load_env_from_parent()

    # Ensure JSON output dir
    confirm_json_output_dir(skip_confirm=skip_confirm)

    # Discover 1..n batch folders
    nums = list_numeric_subfolders(BATCHES_ROOT)
    n = len(nums)
    iprint(f"Found {n} folder(s) under {BATCHES_ROOT}: {nums}")

    # Verify GOOGLE_API_KEY_1..n exist (after loading .env)
    ensure_api_keys_for(n)
    iprint(f"All required GOOGLE_API_KEY_1..{n} present.")

    # Prepare patched per-run scripts
    source = read_kg_pipeline_source()
    script_paths = []
    for i in nums:
        lang_dir = (BATCHES_ROOT / str(i)).resolve()
        sp = write_modified_script(i, lang_dir, source)
        script_paths.append((i, sp))
        dprint(f"Prepared {sp} for folder {lang_dir}")

    exit_code = 0
    current_procs = []  # processes in the currently running batch

    # Launch and wait in batches of BATCH_SIZE
    try:
        total = len(script_paths)
        batch_num = 0
        for start in range(0, total, BATCH_SIZE):
            batch_num += 1
            batch = script_paths[start:start + BATCH_SIZE]
            run_ids = [i for i, _ in batch]
            iprint(f"Starting batch {batch_num} with runs: {run_ids}")

            current_procs = []
            for i, sp in batch:
                p, t_out, t_err, log_fh, log_path = launch_process(i, sp, JSON_OUTPUT_DIR, key_id=i)
                current_procs.append((i, p, t_out, t_err, log_fh, log_path))

            # Wait for the whole batch to finish
            for i, p, t_out, t_err, log_fh, log_path in current_procs:
                rc = p.wait()
                t_out.join(timeout=2)
                t_err.join(timeout=2)
                try: log_fh.flush()
                except Exception: pass
                try: log_fh.close()
                except Exception: pass
                if rc != 0:
                    eprint(f"RUN {i:02d} exited with code {rc} (see log: {log_path})")
                    exit_code = 1
                else:
                    iprint(f"RUN {i:02d} completed successfully (log: {log_path}).")

            iprint(f"Batch {batch_num} finished.")
            current_procs = []

        # Cleanup temp scripts
        for i, sp in script_paths:
            try: sp.unlink()
            except Exception: pass

        sys.exit(exit_code)

    except KeyboardInterrupt:
        wprint("KeyboardInterrupt received. Terminating current batch...")
        for _, p, *_ in current_procs:
            try: p.send_signal(signal.SIGINT)
            except Exception: pass
        for _, p, *_ in current_procs:
            try: p.wait(timeout=10)
            except Exception:
                try: p.kill()
                except Exception: pass
        for _, _, _, _, log_fh, _ in current_procs:
            try: log_fh.close()
            except Exception: pass
        # Cleanup temp scripts even on interrupt
        for _, sp in script_paths:
            try: sp.unlink()
            except Exception: pass
        sys.exit(130)

if __name__ == "__main__":
    main()