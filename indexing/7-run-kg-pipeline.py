#!/usr/bin/env python3
"""
Concurrent runner for kg_pipeline.py.

- Loads env vars from the .env file in this script's parent folder.
- Uses venv Python at ../env (Windows: ../env/Scripts/python.exe, Unix: ../env/bin/python).
- Counts numeric subfolders in ../dataset/langchain-batches/samples and requires 1..n with no gaps.
- Verifies GOOGLE_API_KEY_1..GOOGLE_API_KEY_n exist (after loading .env).
- For each i in 1..n, rewrites a temp copy of kg_pipeline.py to set:
    GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY_i"]
    DEFAULT_LANGCHAIN_DIR = Path(".../langchain-batches/samples/i").resolve()
  and runs them concurrently.
- Per-run logs at .kg_runs/logs/run_i.log (UTF-8) with live prefixed console tail.

Run:
  python run-kg-pipeline.py
"""

import os
import re
import sys
import time
import shutil
import threading
import subprocess
from pathlib import Path
from typing import List, Tuple

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE
KG_PIPELINE = (PROJECT_ROOT / "kg_pipeline.py").resolve()
BATCHES_ROOT = (PROJECT_ROOT / "../dataset/langchain-batches/samples").resolve()
RUNS_DIR = (PROJECT_ROOT / ".kg_runs").resolve()

# .env is in the parent folder of this script
DOTENV_PATH = (HERE.parent / ".env").resolve()

# Use the venv in the parent folder of this script
VENV_DIR = (HERE.parent / "env").resolve()
if os.name == "nt":
    VENV_PYTHON = (VENV_DIR / "Scripts" / "python.exe").resolve()
else:
    VENV_PYTHON = (VENV_DIR / "bin" / "python").resolve()

FOLLOW_LOGS = True

COLORS = [
    "\033[38;5;39m",   # blue
    "\033[38;5;208m",  # orange
    "\033[38;5;34m",   # green
    "\033[38;5;199m",  # magenta
    "\033[38;5;124m",  # red
    "\033[38;5;44m",   # cyan
    "\033[38;5;136m",  # brown-ish
    "\033[38;5;63m",   # purple-blue
    "\033[38;5;28m",   # dark green
    "\033[38;5;160m",  # dark red
]
RESET = "\033[0m"

def color_for(idx: int) -> str:
    return COLORS[(idx - 1) % len(COLORS)]

def load_dotenv_file(path: Path, overwrite: bool = False) -> int:
    if not path.exists():
        return 0

    def parse_line(line: str):
        s = line.strip()
        if not s or s.startswith("#"):
            return None
        if s.startswith("export "):
            s = s[len("export "):]
        if "=" not in s:
            return None
        k, v = s.split("=", 1)
        key = k.strip()
        val = v.strip()
        if (val and val[0] not in ["'", '"']) and "#" in val:
            val = val.split("#", 1)[0].rstrip()
        if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
            val = val[1:-1]
        return key, val

    count = 0
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            parsed = parse_line(raw)
            if not parsed:
                continue
            key, val = parsed
            if overwrite or key not in os.environ:
                os.environ[key] = val
                count += 1
    return count

def find_numeric_subfolders(root: Path) -> List[Tuple[int, Path]]:
    if not root.exists() or not root.is_dir():
        return []
    items = []
    for d in root.iterdir():
        if d.is_dir() and d.name.isdigit():
            items.append((int(d.name), d.resolve()))
    items.sort(key=lambda t: t[0])
    return items

def ensure_sequential(indices: List[int]) -> None:
    if not indices:
        return
    expected = list(range(1, len(indices) + 1))
    if indices != expected:
        raise RuntimeError(
            f"Expected batch folders named 1..{len(indices)}, found: {', '.join(map(str, indices))}"
        )

def check_api_keys(n: int) -> None:
    pattern = re.compile(r"^GOOGLE_API_KEY_(\d+)$")
    present = {}
    for k, v in os.environ.items():
        m = pattern.match(k)
        if m and v:
            present[int(m.group(1))] = v

    expected = list(range(1, n + 1))
    missing = [i for i in expected if i not in present]
    extra = sorted([i for i in present if i not in expected])

    if missing or extra or len(present) != n:
        msg = [
            f"Mismatch between folder count ({n}) and GOOGLE_API_KEY_i env vars.",
            "Expected: " + ", ".join(f"GOOGLE_API_KEY_{i}" for i in expected),
            "Found: " + (", ".join(f"GOOGLE_API_KEY_{i}" for i in sorted(present.keys())) or "(none)"),
        ]
        if missing:
            msg.append("Missing: " + ", ".join(f"GOOGLE_API_KEY_{i}" for i in missing))
        if extra:
            msg.append("Extra: " + ", ".join(f"GOOGLE_API_KEY_{i}" for i in extra))
        raise RuntimeError("\n".join(msg))

def ensure_runs_dirs() -> Tuple[Path, Path]:
    if RUNS_DIR.exists():
        shutil.rmtree(RUNS_DIR)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    logs = RUNS_DIR / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    return RUNS_DIR, logs

def escape_path_literal(p: Path) -> str:
    # Use forward slashes to avoid Windows backslash escapes
    return repr(p.as_posix())

def ensure_import(code: str, import_line: str) -> str:
    if import_line in code:
        return code
    lines = code.splitlines(keepends=True)
    if lines and lines[0].startswith("#!"):
        return lines[0] + import_line + "\n" + "".join(lines[1:])
    return import_line + "\n" + code

def rewrite_default_dir(src_code: str, new_dir: Path) -> str:
    src_code = ensure_import(src_code, "from pathlib import Path")
    pattern = re.compile(r"^[ \t]*DEFAULT_LANGCHAIN_DIR[ \t]*=.*$", re.MULTILINE)
    replacement = f'DEFAULT_LANGCHAIN_DIR = Path({escape_path_literal(new_dir)}).resolve()'
    new_code, count = pattern.subn(replacement, src_code, count=1)
    if count == 0:
        raise RuntimeError("Could not find DEFAULT_LANGCHAIN_DIR assignment in kg_pipeline.py")
    return new_code

def rewrite_google_key(src_code: str, idx: int) -> str:
    src_code = ensure_import(src_code, "import os")
    pattern = re.compile(r"^[ \t]*GOOGLE_API_KEY[ \t]*=.*$", re.MULTILINE)
    replacement = f'GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY_{idx}"]'
    new_code, count = pattern.subn(replacement, src_code, count=1)
    if count == 0:
        raise RuntimeError("Could not find GOOGLE_API_KEY assignment in kg_pipeline.py")
    return new_code

def prepare_modified_pipeline(original: Path, run_index: int, folder_path: Path) -> Path:
    code = original.read_text(encoding="utf-8")
    code = rewrite_google_key(code, run_index)
    code = rewrite_default_dir(code, folder_path)
    target = RUNS_DIR / f"kg_pipeline_{run_index}.py"
    target.write_text(code, encoding="utf-8")
    return target

def safe_print_console(s: str) -> None:
    """
    Print to console without crashing on unencodable characters.
    Falls back to 'replace' for the current stdout encoding.
    """
    try:
        print(s)
    except UnicodeEncodeError:
        enc = sys.stdout.encoding or "utf-8"
        sys.stdout.buffer.write((s + "\n").encode(enc, errors="replace"))
        sys.stdout.flush()

def tail_log_live(log_path: Path, idx: int, proc: subprocess.Popen) -> None:
    prefix = f"[{idx}] "
    color = color_for(idx)
    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            while True:
                line = f.readline()
                if line:
                    if line.endswith("\n"):
                        line = line[:-1]
                    safe_print_console(f"{color}{prefix}{line}{RESET}")
                else:
                    if proc.poll() is not None:
                        rest = f.read()
                        if rest:
                            for l in rest.splitlines():
                                safe_print_console(f"{color}{prefix}{l}{RESET}")
                        break
                    time.sleep(0.1)
    except Exception as e:
        safe_print_console(f"[{idx}] Tail error for {log_path}: {e}")

def main() -> int:
    loaded = load_dotenv_file(DOTENV_PATH, overwrite=False)
    if loaded:
        print(f"Loaded {loaded} env var(s) from {DOTENV_PATH}")
    else:
        print(f"No .env loaded from {DOTENV_PATH} (file missing or empty)")

    if not VENV_PYTHON.exists():
        print(f"Error: Could not find virtualenv Python at: {VENV_PYTHON}", file=sys.stderr)
        print("Ensure the parent folder contains 'env' and it is a valid virtual environment.")
        print("On Windows, activate with: ..\\env\\Scripts\\activate")
        return 2

    if not KG_PIPELINE.exists():
        print(f"Error: kg_pipeline.py not found at: {KG_PIPELINE}", file=sys.stderr)
        return 2

    subfolders = find_numeric_subfolders(BATCHES_ROOT)
    if not subfolders:
        print(f"No numeric subfolders found in {BATCHES_ROOT}. Nothing to run.")
        return 0

    indices = [i for i, _ in subfolders]
    try:
        ensure_sequential(indices)
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        return 2

    n = len(subfolders)
    try:
        check_api_keys(n)
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        return 2

    _, logs_dir = ensure_runs_dirs()

    print(f"Using venv Python: {VENV_PYTHON}")
    print(f"Discovered {n} batch folder(s): " + ", ".join(str(i) for i in indices))
    print(f"Logs directory: {logs_dir}")

    procs = []
    tails: List[threading.Thread] = []

    for idx, folder in subfolders:
        try:
            modified_script = prepare_modified_pipeline(KG_PIPELINE, idx, folder)
        except Exception as e:
            print(f"Failed to prepare modified pipeline for folder {idx}: {e}", file=sys.stderr)
            return 2

        env = os.environ.copy()
        key_name = f"GOOGLE_API_KEY_{idx}"
        if not env.get(key_name):
            print(f"Missing environment variable: {key_name}", file=sys.stderr)
            return 2

        # Force UTF-8 in child process to avoid UnicodeEncodeError on Windows consoles
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"

        log_path = logs_dir / f"run_{idx}.log"
        log_fh = open(log_path, "w", encoding="utf-8", buffering=1)
        print(f"Starting run {idx} -> folder={folder} | log={log_path}")
        try:
            p = subprocess.Popen(
                [str(VENV_PYTHON), "-u", str(modified_script)],
                cwd=KG_PIPELINE.parent,
                env=env,
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                text=True,
            )
        except Exception as e:
            log_fh.close()
            print(f"Failed to start run {idx}: {e}", file=sys.stderr)
            return 2

        procs.append((idx, p, log_fh, log_path))

    if FOLLOW_LOGS:
        for idx, p, _, log_path in procs:
            t = threading.Thread(target=tail_log_live, args=(log_path, idx, p), daemon=True)
            t.start()
            tails.append(t)

    exit_code = 0
    for idx, p, log_fh, _ in procs:
        ret = p.wait()
        log_fh.close()
        status = "OK" if ret == 0 else f"FAIL (exit {ret})"
        print(f"{color_for(idx)}[{idx}] finished: {status}{RESET}")
        if ret != 0 and exit_code == 0:
            exit_code = ret

    time.sleep(0.2)
    print("All runs complete.")
    for idx, _, _, log_path in procs:
        print(f"  [{idx}] log: {log_path}")

    return exit_code

if __name__ == "__main__":
    sys.exit(main())