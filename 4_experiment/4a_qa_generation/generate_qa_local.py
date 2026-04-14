#!/usr/bin/env python3
"""
Generate English Q&A pairs from Indonesian legal document JSONs using a local
LM Studio model (OpenAI-compatible API), then verify each Q&A with a second
LLM call (classification A/B/C/D).

Key differences from generate_qa.py (Gemini version):
- Uses LM Studio local server via OpenAI-compatible API instead of Gemini.
- Concurrent sliding window: up to LLM_CONCURRENCY groups are processed
  simultaneously. When any in-flight group finishes, the next queued group
  starts immediately (semaphore + ThreadPoolExecutor pattern).
- Thread-safe logging with millisecond timestamps and attempt IDs.
- Rate limiter is thread-safe (lock-protected).

Config via environment variables (or .env in parent directory):
  LMSTUDIO_BASE_URL   (default: http://localhost:1234/v1)
  LOCAL_GEN_MODEL     (default: qwen/qwen3.5-9b)
  LLM_CONCURRENCY     (default: 4)  — sliding-window width
  LLM_REQUEST_TIMEOUT (default: 1200)  — seconds per request
  RPM_LIMIT           (default: 0 = unlimited, set >0 to cap)
  IS_SAMPLE           (true/false)

Run:
  python generate_qa_local.py [output_jsonl_path] [--file-list <path>]
"""

import argparse
import json
import os
import random
import re
import sys
import time
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor, Future
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import dotenv
from openai import OpenAI

# -----------------------
# Logging helpers (thread-safe, millisecond timestamps)
# -----------------------
_PRINT_LOCK = threading.Lock()

def _ts() -> str:
    now = time.time()
    lt = time.localtime(now)
    ms = int((now - int(now)) * 1000)
    return time.strftime("%Y-%m-%d %H:%M:%S", lt) + f".{ms:03d}"

def _fmt_prefix(level: str, attempt_id: Optional[int]) -> str:
    ts = _ts()
    if attempt_id is None:
        return f"[{ts}] [{level}]"
    return f"[{ts}] [{level}] [attempt_id={attempt_id}]"

def log_info(msg: str, attempt_id: Optional[int] = None) -> None:
    with _PRINT_LOCK:
        print(f"{_fmt_prefix('INFO', attempt_id)} {msg}", flush=True)

def log_warn(msg: str, attempt_id: Optional[int] = None) -> None:
    with _PRINT_LOCK:
        print(f"{_fmt_prefix('WARN', attempt_id)} {msg}", flush=True)

def log_error(msg: str, attempt_id: Optional[int] = None) -> None:
    with _PRINT_LOCK:
        print(f"{_fmt_prefix('ERROR', attempt_id)} {msg}", flush=True)

# Keep a simple log() alias that matches generate_qa.py callsites
def log(msg: str):
    log_info(msg)

# -----------------------
# Load .env
# -----------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_ENV_PATH = (_SCRIPT_DIR / ".." / ".." / ".env").resolve()

if not _ENV_PATH.exists():
    raise ImportError(f"Error: .env file not found at {_ENV_PATH}")

dotenv.load_dotenv(_ENV_PATH)

# -----------------------
# Config from env
# -----------------------
LMSTUDIO_BASE_URL   = os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
LOCAL_GEN_MODEL     = os.getenv("LOCAL_GEN_MODEL_JUDGE",   "qwen/qwen3.5-27b")
LLM_REQUEST_TIMEOUT = float(os.getenv("LLM_REQUEST_TIMEOUT", "1200"))

# Number of groups processed concurrently (sliding window).
# 1 = fully sequential. N > 1 = as soon as any slot is free the next group starts.
LLM_CONCURRENCY = int(os.getenv("LLM_CONCURRENCY_QA_GENERATION", "10"))

# Optional RPM cap (0 or negative = disabled)
_rpm_env = os.getenv("RPM_LIMIT", "0")
RPM_LIMIT: Optional[int] = int(_rpm_env) if _rpm_env.strip().lstrip("-").isdigit() else None
if RPM_LIMIT is not None and RPM_LIMIT <= 0:
    RPM_LIMIT = None

is_sample = os.getenv("IS_SAMPLE", "").strip().lower()
if is_sample == "true":
    IS_SAMPLE = True
elif is_sample == "false":
    IS_SAMPLE = False
else:
    raise ValueError(f"Wrong configuration of IS_SAMPLE in .env file: {is_sample!r}")

if IS_SAMPLE:
    INPUT_DIR = (_SCRIPT_DIR / ".." / ".." / "dataset" / "samples" / "2_extract_text_results").resolve()
    OUTPUT_DIR = (_SCRIPT_DIR / ".." / ".." / "dataset" / "samples" / "4_experiment" / "4a_qa_generation" / "4a_ii_qa_pairs").resolve()
else:
    INPUT_DIR = (_SCRIPT_DIR / ".." / ".." / "dataset" / "2_extract_text_results").resolve()
    OUTPUT_DIR = (_SCRIPT_DIR / ".." / ".." / "dataset" / "4_experiment" / "4a_qa_generation" / "4a_ii_qa_pairs").resolve()



# Generation settings
MAX_JSON_FILES: Optional[int] = None
MAX_PAGES_PER_DOC: Optional[int] = None
PAGES_PER_GROUP = 10
NUM_GROUPS_TO_SAMPLE = 10_000_000_000_000_000_000_000_000

TEMPERATURE = 0.2
MAX_CONTEXT_CHARS = None
SEED = None

# Retry backoff
RETRY_DELAY_MIN_SECONDS = 5
RETRY_DELAY_MAX_SECONDS = 15

# Verification settings (deterministic)
VERIFICATION_TEMPERATURE = 0.0

# -----------------------
# Initialize LM Studio client
# -----------------------
lmstudio_client = OpenAI(
    base_url=LMSTUDIO_BASE_URL,
    api_key="lm-studio",
    timeout=LLM_REQUEST_TIMEOUT,
)
log_info(f"Using LMStudio model: {LOCAL_GEN_MODEL}")
log_info(f"LMStudio base URL: {LMSTUDIO_BASE_URL}")
log_info(f"LLM concurrency (sliding window): {LLM_CONCURRENCY}")
log_info(f"RPM limit: {RPM_LIMIT or 'unlimited'}")

# -----------------------
# Attempt ID generator (thread-safe, for logging)
# -----------------------
_attempt_lock = threading.Lock()
_attempt_counter = 0

def next_attempt_id() -> int:
    global _attempt_counter
    with _attempt_lock:
        aid = _attempt_counter
        _attempt_counter += 1
        return aid

# -----------------------
# Thread-safe Rate Limiter
# -----------------------
class RateLimiter:
    """
    Sliding-window rate limiter. Thread-safe.
    Allows at most `rpm` acquire() calls per 60-second window.
    """
    def __init__(self, rpm: Optional[int]):
        self.rpm = rpm if rpm and rpm > 0 else None
        self.window = 60.0
        self.calls: deque = deque()
        self.lock = threading.Lock()
        self.total_calls = 0

    def acquire(self, attempt_id: Optional[int] = None) -> float:
        """Block until a slot is free; return seconds waited (0 if none)."""
        with self.lock:
            self.total_calls += 1
            if self.rpm is None:
                return 0.0

            now = time.time()
            while self.calls and (now - self.calls[0]) >= self.window:
                self.calls.popleft()

            wait = 0.0
            if len(self.calls) >= self.rpm:
                wait = self.window - (now - self.calls[0])

        if wait > 0:
            log_info(f"[RateLimiter] Sleeping {wait:.2f}s to respect RPM limit", attempt_id=attempt_id)
            time.sleep(wait)

        with self.lock:
            self.calls.append(time.time())

        return wait

_rate_limiter = RateLimiter(RPM_LIMIT)

# -----------------------
# Metrics (thread-safe)
# -----------------------
class APIMetrics:
    def __init__(self):
        self._lock = threading.Lock()
        self.gen_attempts = 0
        self.gen_successes = 0
        self.gen_api_errors = 0
        self.gen_parse_failures = 0
        self.ver_attempts = 0
        self.ver_successes = 0
        self.ver_api_errors = 0
        self.ver_parse_failures = 0

    def inc(self, attr: str, n: int = 1):
        with self._lock:
            setattr(self, attr, getattr(self, attr) + n)

# -----------------------
# File and grouping helpers (unchanged from generate_qa.py)
# -----------------------
def find_json_files(input_dir: Path, limit: Optional[int]) -> List[Path]:
    files = sorted([p for p in input_dir.rglob("*.json") if p.is_file()])
    if limit is not None and limit >= 0:
        files = files[:limit]
    return files

def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def pages_from_doc(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    pages = doc.get("pages", [])
    return pages if isinstance(pages, list) else []

def page_text_plus_ocr(page: Dict[str, Any]) -> str:
    t = page.get("text") or ""
    o = page.get("ocr") or ""
    if t and o:
        return f"{t}\n{o}"
    return t or o

def build_groups_for_doc(doc: Dict[str, Any], pages_per_group: int) -> List[Dict[str, Any]]:
    groups: List[Dict[str, Any]] = []
    filename = doc.get("filename") or doc.get("metadata", {}).get("Title") or "UNKNOWN"
    pages = pages_from_doc(doc)
    if not pages:
        return groups

    if MAX_PAGES_PER_DOC is not None and MAX_PAGES_PER_DOC >= 0:
        pages = pages[:MAX_PAGES_PER_DOC]

    n = len(pages)
    group_idx = 0
    for start in range(0, n, pages_per_group):
        end = min(start + pages_per_group, n)
        page_slice = pages[start:end]

        page_blocks = []
        for local_i, p in enumerate(page_slice):
            pn = p.get("page_number")
            if pn is None:
                pn = start + 1 + local_i
            content = page_text_plus_ocr(p).strip()
            page_blocks.append(f"=== Page {pn} ===\n{content}".strip())

        context = ("\n\n").join(page_blocks).strip()
        if MAX_CONTEXT_CHARS is not None and len(context) > MAX_CONTEXT_CHARS:
            context = context[:MAX_CONTEXT_CHARS]

        start_page_num = page_slice[0].get("page_number") or (start + 1)
        end_page_num = page_slice[-1].get("page_number") or end

        groups.append(
            {
                "source_filename": filename,
                "group_index": group_idx,
                "page_start": int(start_page_num),
                "page_end": int(end_page_num),
                "context": context,
            }
        )
        group_idx += 1

    return groups

def build_all_groups(json_files: List[Path]) -> List[Dict[str, Any]]:
    all_groups: List[Dict[str, Any]] = []
    for p in json_files:
        try:
            doc = load_json(p)
            groups = build_groups_for_doc(doc, PAGES_PER_GROUP)
            all_groups.extend(groups)
        except Exception as e:
            log_warn(f"Failed to process {p}: {e}")
    return all_groups

# -----------------------
# Prompt builders (unchanged from generate_qa.py)
# -----------------------
def build_prompt(context: str, filename: str, page_range: Tuple[int, int]) -> str:
    start, end = page_range
    instructions = (
        "You are given an excerpt from Indonesian legal document pages.\n"
        "Task: Create exactly one question-answer pair in English using ONLY the provided context.\n"
        "- The question should be specific and answerable from the context (e.g., purpose, amounts, dates, duties, scope, definitions).\n"
        "- The answer must be concise (1-3 sentences), accurate, and strictly grounded in the context. Do not invent facts.\n"
        "- Use names, dates, amounts, and terms as they appear; translate Indonesian to English when forming the question and answer.\n"
        "- Output must be a single JSON object with keys 'question' and 'answer'. No extra text.\n"
        "- Only generate question with sufficient information found in the context. DO NOT generate questions where the answers would be along the lines of: Insufficient information in the provided context.\n"
        "- DO NOT GENERATE questions with phrases like 'According to document X', or 'According to page X of document Y', or 'According toe Law X of year Y', as these question answer pair will be use to test a model's capability to identify the correct answers from relevant documents.\n\n"
        "EXAMPLE QUESTIONS:\n"
        "- What is the content of Pasal 1320 KUHPerdata and what does it mean?\n"
        "- Please explain Pasal 27 ayat (3) UU ITE in simple terms.\n"
        "- What is the legal basis for filing a breach of contract (wanprestasi) lawsuit in Indonesia?\n"
        "- What is the latest criminal law basis?\n"
        "- What is the legal basis for divorce under Islamic law and civil law in Indonesia?\n"
        "- What is wanprestasi?\n"
        "- What is meant by perbuatan melawan hukum (PMH)?\n"
        "- What is the difference between an akta otentik and an akta di bawah tangan?\n"
        "- What is the principle of pacta sunt servanda?\n"
        "- What is regulated in UU No. 13 Tahun 2003 tentang Ketenagakerjaan?\n"
        "- What is the main content of the UU Perlindungan Konsumen?\n"
        "- What civil law principles apply in employment contract disputes, and how are they implemented in practice?\n"
        "- What is the principle of asas legalitas in criminal law?\n"
        "- What are the important civil law principles?\n"
        "- How to file a civil lawsuit in a district court (pengadilan negeri)?\n"
        "- What are the steps in the criminal process, from reporting to verdict?\n"
        "- What are the criminal penalties for defamation under UU ITE?\n"
        "- What are the legal consequences of a contract that does not meet the validity requirements?\n"
        "- Compare the provisions on PKWT in UU Ketenagakerjaan before and after the amendment by UU Cipta Kerja.\n"
        "- What are the key points of the latest law on the notary profession, and how does it differ from the previous law?\n"
        "- How do courts typically interpret Pasal 1266 and 1267 KUHPerdata in cases of unilateral contract termination?\n"
        "- Compare legal protections for whistleblowers in Indonesia and the United States.\n"
        "- What are the legal risks of using outsourcing work contracts in the manufacturing industry under UU Cipta Kerja and the latest MK ruling?\n"
        "- Summarize three relevant MA rulings in disputes over wanprestasi in house construction.\n"
        "- Does Komnas HAM have the authority to handle reports of violations by law enforcement officials? Explain the legal basis.\n"
        "- What is the relationship between Undang-Undang Pers and KUHP in defamation cases involving journalists?\n"
        "- What legal protections exist for victims of Kekerasan Dalam Rumah Tangga (KDRT) under UU No. 23 Tahun 2004, and what is the legal mechanism?\n\n"
        "Also, for each answers, make sure the answers are comprehensive, contains the document(s) which the answers are based on, and other evidencs, e.g. quotes from documents, etc."
    )
    header = f"Document: {filename}\nPages: {start}-{end}\n"
    return f"{instructions}\n{header}\nContext:\n{context}\n"

def build_verification_prompt(context: str, question: str, answer: str) -> str:
    return (
        "LLM Verification Prompt\n"
        "Based on the context below, classify the answer as one of the choices (A, B, C, or D):\n"
        f"Context: {context}\n"
        f"Question: {question}\n"
        f"Answer: {answer}\n"
        "Choices:\n"
        "A. Correct answer (answers the question) and matches the given context\n"
        "B. Correct answer (answers the question) but does NOT match the given context\n"
        "C. Incorrect answer (does not answer the question) but the question matches the context\n"
        "D. Incorrect answer (does not answer the question) and the question does NOT match the context\n"
        "Provide only the letter corresponding to the correct choice (A, B, C, or D)."
    )

# -----------------------
# Response parsing helpers
# -----------------------
def parse_verification_choice(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    s = text.strip().upper()
    if s.startswith("```") and s.endswith("```"):
        s = s.strip("`").strip()
    m = re.search(r"\b([ABCD])\b", s)
    if m:
        return m.group(1)
    for ch in "ABCD":
        if ch in s:
            return ch
    return None

def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    trimmed = text.strip()
    # Strip markdown code fences
    if trimmed.startswith("```"):
        # Remove opening fence line and closing fence
        lines = trimmed.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        trimmed = "\n".join(lines).strip()
    start = trimmed.find("{")
    end = trimmed.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    candidate = trimmed[start: end + 1]
    try:
        obj = json.loads(candidate)
        if isinstance(obj, dict) and "question" in obj and "answer" in obj:
            return obj
    except json.JSONDecodeError:
        return None
    return None

# -----------------------
# Low-level LLM caller (LM Studio)
# -----------------------
def _call_lmstudio(
    messages: List[Dict[str, str]],
    temperature: float,
    attempt_id: int,
) -> Tuple[str, Dict[str, int]]:
    """
    Make a single chat completion call to LM Studio.
    Returns (text, usage_dict) where usage_dict has keys:
      input_tokens, reasoning_tokens, output_tokens, total_tokens.
    Raises on API errors (caller handles retry).
    """
    completion = lmstudio_client.chat.completions.create(
        model=LOCAL_GEN_MODEL,
        messages=messages,
        temperature=temperature,
    )
    text = completion.choices[0].message.content or ""
    usage = completion.usage
    if usage is not None:
        input_tokens    = usage.prompt_tokens or 0
        output_tokens   = usage.completion_tokens or 0
        # LM Studio exposes reasoning tokens under completion_tokens_details
        details         = getattr(usage, "completion_tokens_details", None)
        reasoning_tokens = (
            getattr(details, "reasoning_tokens", 0) or 0
            if details is not None else 0
        )
        total_tokens    = usage.total_tokens or (input_tokens + output_tokens)
    else:
        input_tokens = reasoning_tokens = output_tokens = total_tokens = 0
    usage_dict = {
        "input_tokens":     input_tokens,
        "reasoning_tokens": reasoning_tokens,
        "output_tokens":    output_tokens,
        "total_tokens":     total_tokens,
    }
    return text, usage_dict

# -----------------------
# Generation caller (with retry)
# -----------------------
def generate_qa_for_group(
    prompt: str,
    meta: Dict[str, Any],
    metrics: APIMetrics,
) -> Dict[str, str]:
    """
    Generate a QA pair for *prompt* using the local LM Studio model.
    Retries indefinitely; uniformly random backoff between attempts.
    Thread-safe: uses the shared rate limiter.
    """
    attempt = 1
    attempt_id = next_attempt_id()

    file = meta.get("source_filename", "UNKNOWN")
    group_index = meta.get("group_index", -1)
    page_start = meta.get("page_start", "?")
    page_end = meta.get("page_end", "?")
    idx = meta.get("global_index", "?")
    total = meta.get("total_groups", "?")

    while attempt == 1:
        wait_s = _rate_limiter.acquire(attempt_id=attempt_id)
        attempt_start = time.time()

        log_info(
            f"[Group {idx}/{total}] GEN attempt {attempt} | "
            f"file='{file}' group={group_index} pages={page_start}-{page_end} | "
            f"rate_wait={wait_s:.2f}s | prompt_chars={len(prompt)}",
            attempt_id=attempt_id,
        )

        metrics.inc("gen_attempts")
        try:
            raw, usage = _call_lmstudio(
                messages=[{"role": "user", "content": prompt}],
                temperature=TEMPERATURE,
                attempt_id=attempt_id,
            )
            latency = time.time() - attempt_start

            qa = extract_json_object(raw)
            if qa and isinstance(qa.get("question"), str) and isinstance(qa.get("answer"), str):
                qa["question"] = qa["question"].strip()
                qa["answer"] = qa["answer"].strip()
                metrics.inc("gen_successes")
                log_info(
                    f"[Group {idx}/{total}] GEN attempt {attempt} SUCCESS | "
                    f"file='{file}' pages={page_start}-{page_end} | "
                    f"latency={latency:.2f}s | resp_chars={len(raw)} | "
                    f"tokens(in={usage['input_tokens']} reason={usage['reasoning_tokens']} "
                    f"out={usage['output_tokens']} total={usage['total_tokens']})",
                    attempt_id=attempt_id,
                )
                qa["gen_usage"] = usage
                return qa

            # Unparseable
            metrics.inc("gen_parse_failures")
            log_warn(
                f"[Group {idx}/{total}] GEN attempt {attempt} FAILED (unparseable) | "
                f"file='{file}' pages={page_start}-{page_end} | latency={latency:.2f}s | "
                f"raw_preview={raw[:200]!r}",
                attempt_id=attempt_id,
            )

        except Exception as e:
            latency = time.time() - attempt_start
            metrics.inc("gen_api_errors")
            log_error(
                f"[Group {idx}/{total}] GEN attempt {attempt} ERROR | "
                f"file='{file}' pages={page_start}-{page_end} | "
                f"latency={latency:.2f}s | {e.__class__.__name__}: {e}",
                attempt_id=attempt_id,
            )

        backoff = random.uniform(RETRY_DELAY_MIN_SECONDS, RETRY_DELAY_MAX_SECONDS)
        log_info(
            f"[Group {idx}/{total}] GEN retry delay {backoff:.2f}s | "
            f"file='{file}' pages={page_start}-{page_end}",
            attempt_id=attempt_id,
        )
        time.sleep(backoff)
        attempt += 1

# -----------------------
# Verification caller (with retry)
# -----------------------
def verify_qa_for_group(
    context: str,
    question: str,
    answer: str,
    meta: Dict[str, Any],
    metrics: APIMetrics,
) -> Dict[str, Any]:
    """
    Verify a QA pair. Retries until a valid choice (A/B/C/D) is parsed.
    Thread-safe: uses the shared rate limiter.
    """
    prompt = build_verification_prompt(context, question, answer)
    attempt = 1
    attempt_id = next_attempt_id()

    file = meta.get("source_filename", "UNKNOWN")
    group_index = meta.get("group_index", -1)
    page_start = meta.get("page_start", "?")
    page_end = meta.get("page_end", "?")
    idx = meta.get("global_index", "?")
    total = meta.get("total_groups", "?")

    while True:
        wait_s = _rate_limiter.acquire(attempt_id=attempt_id)
        attempt_start = time.time()

        log_info(
            f"[Group {idx}/{total}] VERIFY attempt {attempt} | "
            f"file='{file}' group={group_index} pages={page_start}-{page_end} | "
            f"rate_wait={wait_s:.2f}s | prompt_chars={len(prompt)}",
            attempt_id=attempt_id,
        )

        metrics.inc("ver_attempts")
        try:
            raw, usage = _call_lmstudio(
                messages=[{"role": "user", "content": prompt}],
                temperature=VERIFICATION_TEMPERATURE,
                attempt_id=attempt_id,
            )
            latency = time.time() - attempt_start

            choice = parse_verification_choice(raw)
            if choice in ("A", "B", "C", "D"):
                metrics.inc("ver_successes")
                log_info(
                    f"[Group {idx}/{total}] VERIFY attempt {attempt} SUCCESS | "
                    f"file='{file}' pages={page_start}-{page_end} | "
                    f"latency={latency:.2f}s | choice={choice} | "
                    f"tokens(in={usage['input_tokens']} reason={usage['reasoning_tokens']} "
                    f"out={usage['output_tokens']} total={usage['total_tokens']})",
                    attempt_id=attempt_id,
                )
                return {"choice": choice, "latency_s": latency, "ver_usage": usage}

            metrics.inc("ver_parse_failures")
            log_warn(
                f"[Group {idx}/{total}] VERIFY attempt {attempt} FAILED (unparseable) | "
                f"file='{file}' pages={page_start}-{page_end} | latency={latency:.2f}s | "
                f"raw_preview={raw[:200]!r}",
                attempt_id=attempt_id,
            )

        except Exception as e:
            latency = time.time() - attempt_start
            metrics.inc("ver_api_errors")
            log_error(
                f"[Group {idx}/{total}] VERIFY attempt {attempt} ERROR | "
                f"file='{file}' pages={page_start}-{page_end} | "
                f"latency={latency:.2f}s | {e.__class__.__name__}: {e}",
                attempt_id=attempt_id,
            )

        backoff = random.uniform(RETRY_DELAY_MIN_SECONDS, RETRY_DELAY_MAX_SECONDS)
        log_info(
            f"[Group {idx}/{total}] VERIFY retry delay {backoff:.2f}s | "
            f"file='{file}' pages={page_start}-{page_end}",
            attempt_id=attempt_id,
        )
        time.sleep(backoff)
        attempt += 1

# -----------------------
# Per-group file writer
# -----------------------
def _write_group_file(per_group_dir: Path, g: Dict[str, Any], record: Dict[str, Any]) -> None:
    """Write *record* to an individual file for this specific group.

    Filename format:
        {safe_source_filename}_group{group_index}.jsonl
    where *safe_source_filename* is the source filename with path separators
    and special characters replaced by underscores so it is a valid filename.
    """
    safe_name = re.sub(r"[\\/:*?\"<>|]", "_", str(g.get("source_filename", "unknown")))
    group_idx = g.get("group_index", 0)
    fname = f"{safe_name}_group{group_idx}.jsonl"
    group_file = per_group_dir / fname
    try:
        with group_file.open("w", encoding="utf-8") as gf:
            gf.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        log_warn(f"Failed to write per-group file {group_file}: {e}")

# -----------------------
# Per-group worker (gen + verify)
# -----------------------
def _process_group(
    g: Dict[str, Any],
    idx: int,
    total: int,
    metrics: APIMetrics,
    out_lock: threading.Lock,
    out_file,          # open file handle (shared, writes protected by out_lock)
    counters: Dict,    # shared mutable dict: {produced, errors}
    counters_lock: threading.Lock,
    semaphore: threading.Semaphore,
    per_group_dir: Path,  # directory where individual group files are written
) -> None:
    """
    Worker function submitted to the thread pool for each group.
    Releases semaphore when done so the next queued group can start.
    """
    meta = {
        "source_filename": g["source_filename"],
        "group_index": g["group_index"],
        "page_start": g["page_start"],
        "page_end": g["page_end"],
        "global_index": idx,
        "total_groups": total,
    }

    group_start = time.time()
    log_info(
        f"[Group {idx}/{total}] START | "
        f"file='{g['source_filename']}' pages={g['page_start']}-{g['page_end']}"
    )

    try:
        prompt = build_prompt(
            context=g["context"],
            filename=g["source_filename"],
            page_range=(g["page_start"], g["page_end"]),
        )

        # First LLM call: generate QA
        qa = generate_qa_for_group(prompt=prompt, meta=meta, metrics=metrics)

        # Second LLM call: verify QA
        ver = verify_qa_for_group(
            context=g["context"],
            question=qa["question"],
            answer=qa["answer"],
            meta=meta,
            metrics=metrics,
        )

        gen_usage = qa.get("gen_usage", {})
        ver_usage = ver.get("ver_usage", {})
        record = {
            "source_filename": g["source_filename"],
            "group_index": g["group_index"],
            "page_start": g["page_start"],
            "page_end": g["page_end"],
            "question": qa["question"],
            "answer": qa["answer"],
            "verification_choice": ver.get("choice"),
            "model": LOCAL_GEN_MODEL,
            "timestamp_utc": datetime.utcnow().isoformat() + "Z",
            "gen_input_tokens":     gen_usage.get("input_tokens", 0),
            "gen_reasoning_tokens": gen_usage.get("reasoning_tokens", 0),
            "gen_output_tokens":    gen_usage.get("output_tokens", 0),
            "gen_total_tokens":     gen_usage.get("total_tokens", 0),
            "ver_input_tokens":     ver_usage.get("input_tokens", 0),
            "ver_reasoning_tokens": ver_usage.get("reasoning_tokens", 0),
            "ver_output_tokens":    ver_usage.get("output_tokens", 0),
            "ver_total_tokens":     ver_usage.get("total_tokens", 0),
        }
        with out_lock:
            out_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_file.flush()
        _write_group_file(per_group_dir, g, record)
        with counters_lock:
            counters["produced"] += 1
        status = "SUCCESS"

    except Exception as e:
        record = {
            "source_filename": g["source_filename"],
            "group_index": g["group_index"],
            "page_start": g["page_start"],
            "page_end": g["page_end"],
            "error": str(e),
            "model": LOCAL_GEN_MODEL,
            "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        }
        with out_lock:
            out_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_file.flush()
        _write_group_file(per_group_dir, g, record)
        with counters_lock:
            counters["errors"] += 1
        status = "ERROR"
        log_error(
            f"[Group {idx}/{total}] Unhandled exception: {e.__class__.__name__}: {e}"
        )

    finally:
        # Always release the semaphore slot so the next group can start immediately
        semaphore.release()

    group_elapsed = time.time() - group_start
    with counters_lock:
        prod = counters["produced"]
        err = counters["errors"]
    log_info(
        f"[Group {idx}/{total}] END | status={status} | "
        f"file='{g['source_filename']}' pages={g['page_start']}-{g['page_end']} | "
        f"group_time={group_elapsed:.2f}s | Produced={prod} Errors={err}"
    )

# -----------------------
# Main pipeline
# -----------------------
def main(output_path: Optional[str] = None, file_list_path: Optional[Path] = None):
    total_start = time.time()

    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    if SEED is not None:
        random.seed(SEED)

    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"Input directory not found: {INPUT_DIR}")

    # Load JSON file list
    if file_list_path:
        json_files = [
            Path(p).resolve()
            for p in file_list_path.read_text(encoding="utf-8").splitlines()
            if p.strip()
        ]
        if MAX_JSON_FILES is not None and MAX_JSON_FILES >= 0:
            json_files = json_files[:MAX_JSON_FILES]
        log_info(f"Using explicit file list from {file_list_path} ({len(json_files)} files after limit).")
    else:
        json_files = find_json_files(INPUT_DIR, MAX_JSON_FILES)
        log_info(f"Scanned {len(json_files)} JSON files from {INPUT_DIR} (limit={MAX_JSON_FILES}).")

    if not json_files:
        raise RuntimeError("No JSON files to process.")

    all_groups = build_all_groups(json_files)
    if not all_groups:
        raise RuntimeError("No groups could be built from the input JSON files.")

    log_info(f"Total groups available: {len(all_groups)}")
    k = min(NUM_GROUPS_TO_SAMPLE, len(all_groups))
    sampled_groups = random.sample(all_groups, k=k)
    log_info(f"Randomly selected {len(sampled_groups)} groups (target={NUM_GROUPS_TO_SAMPLE}).")
    log_info(f"Concurrency (sliding window): LLM_CONCURRENCY={LLM_CONCURRENCY}")

    out_path = Path(output_path).resolve() if output_path else (OUTPUT_DIR / "qa_pairs_local.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    per_group_dir = out_path.parent / "per_group"
    per_group_dir.mkdir(parents=True, exist_ok=True)

    log_info(f"Main output file: {out_path}")
    log_info(f"Per-group files will be written to: {per_group_dir}")

    metrics = APIMetrics()
    counters = {"produced": 0, "errors": 0}
    counters_lock = threading.Lock()
    out_lock = threading.Lock()

    # Semaphore controls the sliding window width.
    # When any group finishes it releases a slot; the main thread
    # acquires a slot before submitting the next group.
    semaphore = threading.Semaphore(LLM_CONCURRENCY)
    total = len(sampled_groups)

    with out_path.open("w", encoding="utf-8") as out_f:
        with ThreadPoolExecutor(max_workers=LLM_CONCURRENCY) as executor:
            futures: List[Future] = []
            for idx, g in enumerate(sampled_groups, start=1):
                # Block until a concurrency slot is free
                semaphore.acquire()
                fut = executor.submit(
                    _process_group,
                    g, idx, total,
                    metrics, out_lock, out_f,
                    counters, counters_lock,
                    semaphore,
                    per_group_dir,
                )
                futures.append(fut)
            # executor.__exit__ waits for all futures to finish

    total_elapsed = time.time() - total_start
    produced = counters["produced"]
    errors = counters["errors"]

    planned_gen = total
    planned_ver = total
    planned_total = planned_gen + planned_ver
    actual_gen = metrics.gen_attempts
    actual_ver = metrics.ver_attempts
    actual_total = actual_gen + actual_ver

    log_info(f"Done. Wrote {produced + errors} lines to {out_path}")
    log_info(f"Successful Q&A pairs: {produced}, Errors (group-level): {errors}")
    log_info(
        f"Total runtime: {total_elapsed:.2f}s | "
        f"Total rate-limiter acquire() calls: {_rate_limiter.total_calls} | "
        f"RPM_LIMIT={RPM_LIMIT or 'unlimited'}"
    )
    log_info("API usage summary:")
    log_info(
        f"- Planned calls (no retries): generation={planned_gen}, verification={planned_ver}, total={planned_total}"
    )
    log_info(
        f"- Actual calls: generation={actual_gen} "
        f"(successes={metrics.gen_successes}, api_errors={metrics.gen_api_errors}, parse_failures={metrics.gen_parse_failures}); "
        f"verification={actual_ver} "
        f"(successes={metrics.ver_successes}, api_errors={metrics.ver_api_errors}, parse_failures={metrics.ver_parse_failures}); "
        f"total={actual_total}"
    )

# -----------------------
# CLI
# -----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Generate Q&A pairs from Indonesian legal JSON documents using LM Studio, "
            "then verify each pair. Concurrent sliding window: up to LLM_CONCURRENCY "
            "groups are processed simultaneously."
        )
    )
    parser.add_argument(
        "output_path",
        nargs="?",
        default=None,
        help=(
            f"Output JSONL file path "
            f"(default: {OUTPUT_DIR / 'qa_pairs_local.jsonl'})"
        ),
    )
    parser.add_argument(
        "--file-list",
        type=Path,
        default=None,
        help=(
            "Path to a text file containing one JSON file path per line. "
            "When supplied, the script ignores the default directory scan and "
            "processes only the listed files. Used by the parallel runner."
        ),
    )
    args = parser.parse_args()
    main(args.output_path, args.file_list)
