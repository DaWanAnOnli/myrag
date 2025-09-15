#!/usr/bin/env python3
"""
Generate English Q&A pairs from Indonesian legal document JSONs using Gemini,
then verify each Q&A with a second LLM call (classification A/B/C/D).

Enhancements in this version:
- Added an RPM rate limiter (hardcoded constant RPM_LIMIT) to cap API calls per minute.
- Added detailed logging for each API attempt (success/failure), including document, pages, attempt number, wait time due to rate limiting, and latency.
- Added per-group processing time and total runtime reporting.
- Added optional --file-list argument to let a parent process supply a pre‑split list of JSON files (used by the parallel runner).
- Logging prints flush immediately so output streams live when run under a subprocess.
- Changed retry behavior: for each group, continue attempting until successful.
  On every failed attempt, wait a uniformly random 5–15 seconds before retrying.
  Every attempt still passes through the RPM rate limiter.
- NEW: After generating a QA, perform a second LLM call to verify the QA against the context
  and classify it as A/B/C/D (see prompt). The verification call also uses the same
  rate limiter and retry strategy, and the classification is added to each output record.
- NEW: Track API metrics separately for generation and verification: attempts, successes,
  API errors (exceptions), and parse failures. Print a final summary comparing actual calls
  to planned calls (if there were no retries).

Install dependencies:
  pip install google-generativeai

Run:
  python generate_qa.py [output_jsonl_path] [--file-list <path>]

API key loading order:
  1) Environment variable GOOGLE_API_KEY
  2) Parent directory .env file containing: GOOGLE_API_KEY=your_key
  3) Parent directory file named "GOOGLE_API_KEY" with the key as its content
"""

import argparse
import json
import os
import random
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import deque

import dotenv

try:
    import google.generativeai as genai
except ImportError:
    print(
        "Missing dependency: google-generativeai. Install with:\n  pip install google-generativeai",
        file=sys.stderr,
    )
    raise

# -----------------------
# Constants (easy to modify)
# -----------------------

env_file_path = Path("../../.env")
    
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

SCRIPT_DIR = Path(__file__).resolve().parent

if IS_SAMPLE:
    INPUT_DIR = (SCRIPT_DIR / ".." / ".." / "dataset" / "samples" / "2_extract_text_results").resolve()
else:
    INPUT_DIR = (SCRIPT_DIR / ".." / ".." / "dataset" / "2_extract_text_results").resolve()



MAX_JSON_FILES: Optional[int] = None
MAX_PAGES_PER_DOC: Optional[int] = None
PAGES_PER_GROUP = 10
NUM_GROUPS_TO_SAMPLE = 10000000000000000000000000

MODEL_NAME = "gemini-2.5-flash-lite"
TEMPERATURE = 0.2
MAX_OUTPUT_TOKENS = 8192
TOP_P = 0.95
TOP_K = 40

# Verification call settings (deterministic and tiny output)
VERIFICATION_TEMPERATURE = 0.0
VERIFICATION_MAX_OUTPUT_TOKENS = 8  # letter only

SEED = None
MAX_CONTEXT_CHARS = None

# Unlimited retries per group; uniform backoff between 5-15 seconds on failure
RETRY_DELAY_MIN_SECONDS = 5
RETRY_DELAY_MAX_SECONDS = 15

RPM_LIMIT: Optional[int] = 8  # max API calls per minute (set <=0 / None to disable)

# -----------------------
# Logging helpers
# -----------------------
def utc_now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"

def log(msg: str):
    # Flush immediately so output streams live when used as a subprocess
    print(f"[{utc_now()}] {msg}", flush=True)

# -----------------------
# Rate Limiter
# -----------------------
class RateLimiter:
    """
    Simple sliding‑window limiter that allows at most `rpm` acquire() calls
    per 60‑second window.
    """
    def __init__(self, rpm: Optional[int]):
        self.rpm = rpm if rpm and rpm > 0 else None
        self.window = 60.0
        self.calls = deque()
        self.total_calls = 0

    def acquire(self) -> float:
        """Block until a slot is free; return seconds waited (0 if none)."""
        if self.rpm is None:
            self.total_calls += 1
            return 0.0

        waited = 0.0
        now = time.time()

        # Remove timestamps older than the window
        while self.calls and (now - self.calls[0]) >= self.window:
            self.calls.popleft()

        if len(self.calls) >= self.rpm:
            sleep_time = self.window - (now - self.calls[0])
            if sleep_time > 0:
                time.sleep(sleep_time)
                waited += sleep_time
            # Clean again after sleeping
            now = time.time()
            while self.calls and (now - self.calls[0]) >= self.window:
                self.calls.popleft()

        self.calls.append(time.time())
        self.total_calls += 1
        return waited

# -----------------------
# API Key Loading
# -----------------------
def load_api_key() -> str:
    """Load GOOGLE_API_KEY from env/.env/file."""
    key = os.environ.get("GOOGLE_API_KEY")
    if key:
        return key.strip()

    parent_dir = SCRIPT_DIR.parent
    env_path = parent_dir / ".env"
    if env_path.exists():
        try:
            for line in env_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    k, v = line.split("=", 1)
                    if k.strip() == "GOOGLE_API_KEY":
                        return v.strip().strip("'").strip('"')
        except Exception as e:
            print(f"Warning: Failed to read .env file at {env_path}: {e}", file=sys.stderr)

    key_file = parent_dir / "GOOGLE_API_KEY"
    if key_file.exists():
        try:
            content = key_file.read_text(encoding="utf-8").strip()
            if content:
                return content
        except Exception as e:
            print(f"Warning: Failed to read key file at {key_file}: {e}", file=sys.stderr)

    raise RuntimeError(
        "GOOGLE_API_KEY not found. Set it as an environment variable, or put it in a .env file "
        "in the parent directory of this script as:\nGOOGLE_API_KEY=your_key_here\n"
        "Or create a parent file named 'GOOGLE_API_KEY' containing only your key."
    )

# -----------------------
# File and grouping helpers
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
            print(f"Warning: failed to process {p}: {e}", file=sys.stderr)
    return all_groups

# -----------------------
# Prompt / Generation
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
    # Strict prompt as requested; ask for a single letter only.
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

def parse_verification_choice(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    s = text.strip().upper()
    # Strip simple code fences if present
    if s.startswith("```") and s.endswith("```"):
        s = s.strip("`").strip()
    # Common formats: "A", "A.", "(A)", "Choice: A"
    m = re.search(r"\b([ABCD])\b", s)
    if m:
        return m.group(1)
    # Fallback: any A/B/C/D in string
    for ch in "ABCD":
        if ch in s:
            return ch
    return None

def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    trimmed = text.strip()
    if trimmed.startswith("```") and trimmed.endswith("```"):
        trimmed = trimmed.strip("`")
    start = trimmed.find("{")
    end = trimmed.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    candidate = trimmed[start : end + 1]
    try:
        obj = json.loads(candidate)
        if isinstance(obj, dict) and "question" in obj and "answer" in obj:
            return obj
    except json.JSONDecodeError:
        return None
    return None

def configure_model(api_key: str):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(MODEL_NAME)
    generation_config = {
        "temperature": TEMPERATURE,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "top_p": TOP_P,
        "top_k": TOP_K,
    }
    verification_config = {
        "temperature": VERIFICATION_TEMPERATURE,
        "max_output_tokens": VERIFICATION_MAX_OUTPUT_TOKENS,
        "top_p": TOP_P,
        "top_k": TOP_K,
    }
    return model, generation_config, verification_config

# -----------------------
# Metrics
# -----------------------
class APIMetrics:
    """
    Track API usage per call type.
    - attempts: number of API calls attempted
    - successes: number of attempts that yielded a valid, parsed result
    - api_errors: number of attempts that raised exceptions
    - parse_failures: number of attempts that returned content but couldn't be parsed
    """
    def __init__(self):
        self.gen_attempts = 0
        self.gen_successes = 0
        self.gen_api_errors = 0
        self.gen_parse_failures = 0

        self.ver_attempts = 0
        self.ver_successes = 0
        self.ver_api_errors = 0
        self.ver_parse_failures = 0

# -----------------------
# Callers
# -----------------------
def generate_qa_for_group(
    model,
    generation_config: Dict[str, Any],
    prompt: str,
    rate_limiter: RateLimiter,
    meta: Dict[str, Any],
    metrics: APIMetrics,
) -> Dict[str, str]:
    """
    Generate a QA pair for *prompt*.
    Behavior: keep attempting until successful. Between failed attempts,
    wait a uniformly random delay in [RETRY_DELAY_MIN_SECONDS, RETRY_DELAY_MAX_SECONDS].
    Every attempt passes through the RPM RateLimiter.
    """
    attempt = 1
    while True:
        wait_s = rate_limiter.acquire()
        attempt_start = time.time()

        file = meta.get("source_filename", "UNKNOWN")
        group_index = meta.get("group_index", -1)
        page_start = meta.get("page_start", "?")
        page_end = meta.get("page_end", "?")
        idx = meta.get("global_index", "?")
        total = meta.get("total_groups", "?")

        log(
            f"[Group {idx}/{total}] API attempt {attempt} | "
            f"file='{file}' group_index={group_index} pages={page_start}-{page_end} | "
            f"rate_wait={wait_s:.2f}s | prompt_chars={len(prompt)}"
        )

        metrics.gen_attempts += 1
        try:
            response = model.generate_content(
                prompt, generation_config=generation_config
            )
            latency = time.time() - attempt_start

            text = getattr(response, "text", None)
            qa = extract_json_object(text or "")
            if qa and isinstance(qa.get("question"), str) and isinstance(qa.get("answer"), str):
                qa["question"] = qa["question"].strip()
                qa["answer"] = qa["answer"].strip()
                metrics.gen_successes += 1
                log(
                    f"[Group {idx}/{total}] Attempt {attempt} SUCCESS | "
                    f"file='{file}' pages={page_start}-{page_end} | "
                    f"latency={latency:.2f}s | resp_chars={len(text or '')}"
                )
                return qa

            # fallback: combine parts if model returned them separately
            combined = ""
            try:
                parts = (
                    response.candidates[0].content.parts
                    if response and response.candidates
                    else []
                )
                combined = "\n".join(
                    [getattr(p, "text", "") for p in parts if getattr(p, "text", "")]
                )
            except Exception:
                pass
            qa = extract_json_object(combined)
            if qa:
                qa["question"] = qa["question"].strip()
                qa["answer"] = qa["answer"].strip()
                metrics.gen_successes += 1
                log(
                    f"[Group {idx}/{total}] Attempt {attempt} SUCCESS (fallback) | "
                    f"file='{file}' pages={page_start}-{page_end} | "
                    f"latency={latency:.2f}s | resp_chars={len(combined)}"
                )
                return qa

            # Unparseable output; will retry
            metrics.gen_parse_failures += 1
            log(
                f"[Group {idx}/{total}] Attempt {attempt} FAILED (unparseable) | "
                f"file='{file}' pages={page_start}-{page_end} | latency={latency:.2f}s"
            )

        except Exception as e:
            latency = time.time() - attempt_start
            metrics.gen_api_errors += 1
            log(
                f"[Group {idx}/{total}] Attempt {attempt} ERROR | "
                f"file='{file}' pages={page_start}-{page_end} | "
                f"latency={latency:.2f}s | {e.__class__.__name__}: {e}"
            )

        # Uniform backoff before the next retry
        backoff = random.uniform(RETRY_DELAY_MIN_SECONDS, RETRY_DELAY_MAX_SECONDS)
        log(
            f"[Group {idx}/{total}] Retry delay {backoff:.2f}s before next attempt | "
            f"file='{file}' pages={page_start}-{page_end}"
        )
        time.sleep(backoff)
        attempt += 1

def verify_qa_for_group(
    model,
    verification_config: Dict[str, Any],
    context: str,
    question: str,
    answer: str,
    rate_limiter: RateLimiter,
    meta: Dict[str, Any],
    metrics: APIMetrics,
) -> Dict[str, Any]:
    """
    Verify a QA pair using the specified verification prompt.
    Retries until a valid choice letter (A/B/C/D) is parsed.
    Every attempt passes through the same RPM RateLimiter.
    """
    prompt = build_verification_prompt(context, question, answer)
    attempt = 1
    while True:
        wait_s = rate_limiter.acquire()
        attempt_start = time.time()

        file = meta.get("source_filename", "UNKNOWN")
        group_index = meta.get("group_index", -1)
        page_start = meta.get("page_start", "?")
        page_end = meta.get("page_end", "?")
        idx = meta.get("global_index", "?")
        total = meta.get("total_groups", "?")

        log(
            f"[Group {idx}/{total}] VERIFY attempt {attempt} | "
            f"file='{file}' group_index={group_index} pages={page_start}-{page_end} | "
            f"rate_wait={wait_s:.2f}s | prompt_chars={len(prompt)}"
        )

        metrics.ver_attempts += 1
        try:
            response = model.generate_content(
                prompt, generation_config=verification_config
            )
            latency = time.time() - attempt_start

            text = getattr(response, "text", None)
            choice = parse_verification_choice(text or "")

            if not choice:
                # Fallback: combine parts if needed
                combined = ""
                try:
                    parts = (
                        response.candidates[0].content.parts
                        if response and response.candidates
                        else []
                    )
                    combined = "\n".join(
                        [getattr(p, "text", "") for p in parts if getattr(p, "text", "")]
                    )
                except Exception:
                    pass
                choice = parse_verification_choice(combined)

            if choice in ("A", "B", "C", "D"):
                metrics.ver_successes += 1
                log(
                    f"[Group {idx}/{total}] VERIFY SUCCESS | "
                    f"file='{file}' pages={page_start}-{page_end} | "
                    f"latency={latency:.2f}s | choice={choice}"
                )
                return {
                    "choice": choice,
                    "latency_s": latency,
                }

            # Parse failure; will retry
            metrics.ver_parse_failures += 1
            log(
                f"[Group {idx}/{total}] VERIFY FAILED (unparseable) | "
                f"file='{file}' pages={page_start}-{page_end} | latency={latency:.2f}s"
            )
        except Exception as e:
            latency = time.time() - attempt_start
            metrics.ver_api_errors += 1
            log(
                f"[Group {idx}/{total}] VERIFY ERROR | "
                f"file='{file}' pages={page_start}-{page_end} | "
                f"latency={latency:.2f}s | {e.__class__.__name__}: {e}"
            )

        # Uniform backoff
        backoff = random.uniform(RETRY_DELAY_MIN_SECONDS, RETRY_DELAY_MAX_SECONDS)
        log(
            f"[Group {idx}/{total}] VERIFY retry delay {backoff:.2f}s before next attempt | "
            f"file='{file}' pages={page_start}-{page_end}"
        )
        time.sleep(backoff)
        attempt += 1

# -----------------------
# Main pipeline
# -----------------------
def main(output_path: str, file_list_path: Optional[Path] = None):
    total_start = time.time()

    # Make stdout line-buffered if possible (helps on some platforms)
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    if SEED is not None:
        random.seed(SEED)

    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"Input directory not found: {INPUT_DIR}")

    # -------------------------------------------------
    # Load the list of JSON files
    # -------------------------------------------------
    if file_list_path:
        json_files = [
            Path(p).resolve()
            for p in file_list_path.read_text(encoding="utf-8").splitlines()
            if p.strip()
        ]
        if MAX_JSON_FILES is not None and MAX_JSON_FILES >= 0:
            json_files = json_files[:MAX_JSON_FILES]
        log(f"Using explicit file list from {file_list_path} ({len(json_files)} files after limit).")
    else:
        json_files = find_json_files(INPUT_DIR, MAX_JSON_FILES)
        log(f"Scanned {len(json_files)} JSON files from {INPUT_DIR} (limit={MAX_JSON_FILES}).")

    if not json_files:
        raise RuntimeError("No JSON files to process.")

    all_groups = build_all_groups(json_files)
    if not all_groups:
        raise RuntimeError("No groups could be built from the input JSON files.")

    log(f"Total groups available: {len(all_groups)}")
    k = min(NUM_GROUPS_TO_SAMPLE, len(all_groups))
    sampled_groups = random.sample(all_groups, k=k)
    log(f"Randomly selected {len(sampled_groups)} groups for generation (target={NUM_GROUPS_TO_SAMPLE}).")

    api_key = load_api_key()
    # Print the key used by this process (masked for safety)
    log(f"Using GOOGLE_API_KEY (masked): {api_key[:6]}...{api_key[-4:] if len(api_key) > 10 else ''}")

    model, generation_config, verification_config = configure_model(api_key)

    out_path = Path(output_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rate_limiter = RateLimiter(RPM_LIMIT)
    log(f"Rate limiter configured: RPM_LIMIT={RPM_LIMIT}")

    metrics = APIMetrics()
    produced = 0
    errors = 0

    with out_path.open("w", encoding="utf-8") as out_f:
        for idx, g in enumerate(sampled_groups, start=1):
            group_start = time.time()
            log(
                f"[Group {idx}/{len(sampled_groups)}] START | "
                f"file='{g['source_filename']}' pages={g['page_start']}-{g['page_end']}"
            )

            prompt = build_prompt(
                context=g["context"],
                filename=g["source_filename"],
                page_range=(g["page_start"], g["page_end"]),
            )

            try:
                # First LLM call: generate QA
                qa = generate_qa_for_group(
                    model=model,
                    generation_config=generation_config,
                    prompt=prompt,
                    rate_limiter=rate_limiter,
                    meta={
                        "source_filename": g["source_filename"],
                        "group_index": g["group_index"],
                        "page_start": g["page_start"],
                        "page_end": g["page_end"],
                        "global_index": idx,
                        "total_groups": len(sampled_groups),
                    },
                    metrics=metrics,
                )

                # Second LLM call: verify QA
                ver = verify_qa_for_group(
                    model=model,
                    verification_config=verification_config,
                    context=g["context"],
                    question=qa["question"],
                    answer=qa["answer"],
                    rate_limiter=rate_limiter,
                    meta={
                        "source_filename": g["source_filename"],
                        "group_index": g["group_index"],
                        "page_start": g["page_start"],
                        "page_end": g["page_end"],
                        "global_index": idx,
                        "total_groups": len(sampled_groups),
                    },
                    metrics=metrics,
                )

                record = {
                    "source_filename": g["source_filename"],
                    "group_index": g["group_index"],
                    "page_start": g["page_start"],
                    "page_end": g["page_end"],
                    "question": qa["question"],
                    "answer": qa["answer"],
                    "verification_choice": ver.get("choice"),
                    "model": MODEL_NAME,
                    "timestamp_utc": datetime.utcnow().isoformat() + "Z",
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                produced += 1
                status = "SUCCESS"
            except Exception as e:
                record = {
                    "source_filename": g["source_filename"],
                    "group_index": g["group_index"],
                    "page_start": g["page_start"],
                    "page_end": g["page_end"],
                    "error": str(e),
                    "model": MODEL_NAME,
                    "timestamp_utc": datetime.utcnow().isoformat() + "Z",
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                errors += 1
                status = "ERROR"

            group_elapsed = time.time() - group_start
            total_elapsed = time.time() - total_start
            log(
                f"[Group {idx}/{len(sampled_groups)}] END | status={status} | "
                f"file='{g['source_filename']}' pages={g['page_start']}-{g['page_end']} | "
                f"group_time={group_elapsed:.2f}s | Produced={produced} Errors={errors} | "
                f"Total elapsed={total_elapsed:.1f}s"
            )

    # -----------------------
    # Final summary
    # -----------------------
    total_elapsed = time.time() - total_start
    planned_gen_calls = len(sampled_groups)
    planned_ver_calls = len(sampled_groups)
    planned_total_calls = planned_gen_calls + planned_ver_calls

    actual_gen_calls = metrics.gen_attempts
    actual_ver_calls = metrics.ver_attempts
    actual_total_calls = actual_gen_calls + actual_ver_calls

    log(f"Done. Wrote {produced + errors} lines to {out_path}")
    log(f"Successful Q&A pairs: {produced}, Errors (group-level): {errors}")
    log(
        f"Total runtime: {total_elapsed:.2f}s | Total API acquire() calls: {rate_limiter.total_calls} | RPM_LIMIT={RPM_LIMIT}"
    )

    # Detailed API usage summary
    log("API usage summary:")
    log(
        f"- Planned calls (no retries): generation={planned_gen_calls}, verification={planned_ver_calls}, total={planned_total_calls}"
    )
    log(
        f"- Actual calls: generation={actual_gen_calls} "
        f"(successes={metrics.gen_successes}, api_errors={metrics.gen_api_errors}, parse_failures={metrics.gen_parse_failures}); "
        f"verification={actual_ver_calls} "
        f"(successes={metrics.ver_successes}, api_errors={metrics.ver_api_errors}, parse_failures={metrics.ver_parse_failures}); "
        f"total={actual_total_calls}"
    )

# -----------------------
# CLI
# -----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Q&A pairs from Indonesian legal JSON documents using Gemini, then verify each pair."
    )
    parser.add_argument(
        "output_path",
        nargs="?",
        default="../dataset/samples/qa_pairs/qa_pairs.jsonl",
        help="Output JSONL file path (default: ../dataset/samples/qa_pairs/qa_pairs.jsonl)",
    )
    parser.add_argument(
        "--file-list",
        type=Path,
        default=None,
        help="Path to a text file containing one JSON file path per line. "
             "When supplied, the script ignores the default directory scan and "
             "processes only the listed files. Used by the parallel runner.",
    )
    args = parser.parse_args()
    main(args.output_path, args.file_list)