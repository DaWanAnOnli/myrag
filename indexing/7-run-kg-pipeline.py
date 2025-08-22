#!/usr/bin/env python3
import os
import sys
import re
import signal
import subprocess
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict

try:
    from dotenv import find_dotenv, dotenv_values
except ImportError:
    print("This script requires python-dotenv. Install it with: pip install python-dotenv")
    sys.exit(1)


def mask_key(k: str, visible: int = 4) -> str:
    if not k:
        return ""
    if len(k) <= visible:
        return "*" * len(k)
    return "*" * (len(k) - visible) + k[-visible:]


def find_env_path() -> Path:
    # Try to locate .env by walking up from CWD
    p = find_dotenv(usecwd=True)
    if p:
        return Path(p)
    # Fallbacks: next to this script, or one/two directories up
    here = Path(__file__).resolve().parent
    for cand in [here / ".env", here.parent / ".env", here.parent.parent / ".env"]:
        if cand.exists():
            return cand
    return Path()


def load_numbered_google_keys(dotenv_path: Path) -> List[Tuple[int, str]]:
    # Parse .env without mutating os.environ
    values: Dict[str, str] = dotenv_values(str(dotenv_path)) if dotenv_path and dotenv_path.exists() else {}
    pat = re.compile(r"^GOOGLE_API_KEY_(\d+)$", re.IGNORECASE)
    keys: List[Tuple[int, str]] = []
    for name, val in values.items():
        if not name or val is None:
            continue
        m = pat.match(name.strip())
        if m:
            idx = int(m.group(1))
            v = val.strip()
            if v:
                keys.append((idx, v))
    keys.sort(key=lambda t: t[0])
    return keys


def determine_json_output_dir(kg_path: Path) -> Path:
    # Respect JSON_OUTPUT_DIR env if set; otherwise mirror kg_pipeline default
    env_dir = os.environ.get("JSON_OUTPUT_DIR")
    if env_dir:
        return Path(env_dir).expanduser().resolve()
    # kg_pipeline default: Path(__file__).parent / "../../dataset/llm-json-outputs"
    return (kg_path.parent / "../../dataset/llm-json-outputs").resolve()


def ensure_output_dir(json_dir: Path) -> None:
    if not json_dir.exists():
        json_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created JSON output directory: {json_dir}")
        return

    # If exists and not empty, warn and ask for confirmation
    non_empty = any(json_dir.iterdir())
    if non_empty:
        print(f"Warning: {json_dir} exists and is not empty.")
        print("Continuing will add new JSON files to this directory.")
        ans = input("Proceed? [y/N]: ").strip().lower()
        if ans not in ("y", "yes"):
            print("Aborted by user.")
            sys.exit(0)


def kill_process_tree(proc: subprocess.Popen):
    try:
        if proc.poll() is None:
            # Try terminate first
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                # Force kill
                proc.kill()
    except Exception:
        pass


def main():
    script_dir = Path(__file__).resolve().parent
    kg_path = script_dir / "kg_pipeline.py"
    if not kg_path.exists():
        print(f"Error: kg_pipeline.py not found at {kg_path}")
        print("Place this script next to kg_pipeline.py or pass the correct path.")
        sys.exit(1)

    # Determine and prepare JSON output directory
    json_dir = determine_json_output_dir(kg_path)
    ensure_output_dir(json_dir)

    # Load numbered GOOGLE_API_KEY_N from .env
    dotenv_path = find_env_path()
    if not dotenv_path or not dotenv_path.exists():
        print("Error: .env file not found. Please create one with GOOGLE_API_KEY_1, GOOGLE_API_KEY_2, ...")
        sys.exit(1)

    numbered_keys = load_numbered_google_keys(dotenv_path)
    if not numbered_keys:
        print("Error: No numbered GOOGLE_API_KEY_N entries found in .env (e.g., GOOGLE_API_KEY_1, GOOGLE_API_KEY_2).")
        sys.exit(1)

    print(f"Found {len(numbered_keys)} API key(s) in {dotenv_path}")

    # Logs directory
    logs_dir = script_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")

    procs: List[Tuple[subprocess.Popen, Path, int]] = []
    log_files = []  # keep handles to close later

    # Handle Ctrl+C gracefully
    interrupted = {"flag": False}

    def handle_sigint(signum, frame):
        if interrupted["flag"]:
            return
        interrupted["flag"] = True
        print("\nInterrupt received. Stopping all child processes...")
        for p, _, _ in procs:
            kill_process_tree(p)

    signal.signal(signal.SIGINT, handle_sigint)

    # Start one kg_pipeline.py process per key
    for idx, key in numbered_keys:
        env = os.environ.copy()
        env["GOOGLE_API_KEY"] = key
        # Ensure kg_pipeline uses this output dir (so our pre-checks match)
        env["JSON_OUTPUT_DIR"] = str(json_dir)
        # Improve real-time logging
        env["PYTHONUNBUFFERED"] = "1"

        log_path = logs_dir / f"kg_run_{ts}_key{idx}.txt"
        lf = open(log_path, "w", encoding="utf-8")
        log_files.append(lf)

        print(f"- Starting run for GOOGLE_API_KEY_{idx} ({mask_key(key)}) -> {log_path}")

        # Launch kg_pipeline.py as a child process
        p = subprocess.Popen(
            [sys.executable, str(kg_path)],
            stdout=lf,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=str(kg_path.parent),
        )
        procs.append((p, log_path, idx))

    # Wait for all to complete
    exit_codes: List[Tuple[int, int, Path]] = []
    for p, log_path, idx in procs:
        rc = p.wait()
        exit_codes.append((idx, rc, log_path))

    # Close logs
    for lf in log_files:
        try:
            lf.flush()
            lf.close()
        except Exception:
            pass

    # Summary
    print("\nAll runs finished. Summary:")
    for idx, rc, log_path in sorted(exit_codes, key=lambda x: x[0]):
        status = "OK" if rc == 0 else f"Failed (rc={rc})"
        print(f"  - Key {idx}: {status} | Log: {log_path}")


if __name__ == "__main__":
    main()