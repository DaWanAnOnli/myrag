#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LLM Judge Script: Scores multiple model answers (GraphRAG, NaiveRAG, NaiveKG-GraphRAG, LexiDKG-GraphRAG, etc.) against ground truth.

Enhancements:
- Optional CLI to run on a shard (--shard-index/--num-shards) with round-robin distribution.
- Optional --output-suffix to disambiguate per-process CSV and log filenames.
- Optional --labels-file for consistent CSV columns across shards.
- Optional --force-include-id to ensure ID column is present for merging/sorting.

Update (max attempts + failure handling):
- Adds a MAX_ATTEMPTS cap (default 5) for validation/parse/exception failures in model responses.
- After 5 failed attempts for a question, logs detailed diagnostics including the question id and the exact issue,
  then continues with the next question.
- In the CSV, sets score = -1 and reason = "N/A" for all expected labels of the failed question.
"""

import os
import random
import sys
import json
import csv
import time
import re
import glob
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List

import dotenv

try:
    from dotenv import load_dotenv
except ImportError:
    print("Missing dependency: python-dotenv. Install with: pip install python-dotenv", file=sys.stderr)
    sys.exit(1)

try:
    import google.generativeai as genai
except ImportError:
    print("Missing dependency: google-generativeai. Install with: pip install google-generativeai", file=sys.stderr)
    sys.exit(1)


# =========================
# Configuration
# =========================

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

MODEL_NAME = "gemini-2.5-flash-lite"
TEMPERATURE = 0.0
TOP_P = 0.0
TOP_K = 1

# Requests per minute (RPM). All attempts (including failed ones) respect this limit.
RPM = 10

# Maximum attempts per question when response is invalid (labels/scores/JSON/exception)
MAX_ATTEMPTS = 10

# I/O paths (relative to this script)
SCRIPT_DIR = Path(__file__).resolve().parent
PARENT_DIR = SCRIPT_DIR.parent

if IS_SAMPLE:
    INPUT_DIR = SCRIPT_DIR / "../../dataset/samples/4_experiment/4b_experiment_answers"
    OUTPUT_DIR = (SCRIPT_DIR / "../../dataset/samples/4_experiment/4c_experiment_results").resolve()
else:
    INPUT_DIR = SCRIPT_DIR / "../../dataset/4_experiment/4b_experiment_answers"
    OUTPUT_DIR = (SCRIPT_DIR / "../../dataset/4_experiment/4c_experiment_results").resolve()

INPUT_PATTERN = "combined_answers_*.jsonl"
OUTPUT_CSV_BASENAME = "llm_judge_results"

# Log directory
LOG_DIR = (SCRIPT_DIR / "llm_judge_logs").resolve()

# Preferred label ordering for known models (extras will be appended)
PREFERRED_LABEL_ORDER = [
    "naiverag",
    "lexidkg_graphrag_0_hop",
    "lexidkg_graphrag_1_hop",
    "lexidkg_graphrag_2_hop",
    "lexidkg_graphrag_3_hop",
    "lexidkg_graphrag_4_hop",
    "lexidkg_graphrag_5_hop",
]

# Map JSONL keys to friendly, stable labels (optional explicit mapping)
MODEL_KEY_TO_LABEL = {
    "naive_rag_answer": "naiverag",
    "lexidkg_graphrag_0_hop_answer": "lexidkg_graphrag_0_hop",
    "lexidkg_graphrag_1_hop_answer": "lexidkg_graphrag_1_hop",
    "lexidkg_graphrag_2_hop_answer": "lexidkg_graphrag_2_hop",
    "lexidkg_graphrag_3_hop_answer": "lexidkg_graphrag_3_hop",
    "lexidkg_graphrag_4_hop_answer": "lexidkg_graphrag_4_hop",
    "lexidkg_graphrag_5_hop_answer": "lexidkg_graphrag_5_hop",
}


# =========================
# Utilities
# =========================

def now_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def ensure_dirs(*paths: Path) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def truncate(text: str, max_len: int = 120) -> str:
    if text is None:
        return ""
    s = str(text).replace("\n", " ").strip()
    return s if len(s) <= max_len else s[:max_len - 3] + "..."


def normalize_label(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", (label or "").lower()).strip("_")


def label_from_key(key: str) -> Optional[str]:
    # Prefer explicit mapping
    if key in MODEL_KEY_TO_LABEL:
        return MODEL_KEY_TO_LABEL[key]
    # Fallback: parse agentic_(... )_answer
    m = re.match(r"^agentic_(.+?)_answer$", key)
    if not m:
        return None
    raw = m.group(1).strip().lower()
    # normalize common variants
    raw = raw.replace("graph_rag", "graphrag")
    raw = raw.replace("naive_rag", "naiverag")
    return normalize_label(raw)


def collect_model_answers(item: Dict[str, Any]) -> Dict[str, str]:
    """
    Returns a dict: {label -> answer_text} for any agentic_*_answer fields present.
    """
    out: Dict[str, str] = {}
    for key, value in item.items():
        if not isinstance(value, str):
            continue
        if key.endswith("_answer"):
            label = label_from_key(key)
            if label:
                out[label] = value.strip()
    return out


def detect_all_labels(items: List[Dict[str, Any]]) -> List[str]:
    found = set()
    for it in items:
        model_answers = collect_model_answers(it)
        found.update([lbl for lbl, ans in model_answers.items() if ans])
    # Stable, preferred order first, then any extras
    ordered = [lbl for lbl in PREFERRED_LABEL_ORDER if lbl in found]
    extras = sorted(list(found.difference(ordered)))
    return ordered + extras


class RateLimiter:
    def __init__(self, rpm: int):
        if rpm <= 0:
            raise ValueError("RPM must be > 0")
        self.interval = 60.0 / float(rpm)
        self._last_request_started_at = 0.0

    def wait_for_slot(self, logger: Optional[logging.Logger] = None) -> None:
        now = time.perf_counter()
        elapsed = now - self._last_request_started_at
        if elapsed < self.interval:
            sleep_for = self.interval - elapsed
            if logger:
                logger.debug(f"RateLimiter: sleeping {sleep_for:.3f}s to respect RPM.")
            time.sleep(sleep_for)
        self._last_request_started_at = time.perf_counter()


def configure_logging(run_ts: str, log_path_override: Optional[Path] = None) -> Tuple[logging.Logger, Path]:
    ensure_dirs(LOG_DIR)
    log_path = Path(log_path_override) if log_path_override else (LOG_DIR / f"llm_judge_run_{run_ts}.txt")

    logger = logging.getLogger("LLM_Judge")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)

    ensure_dirs(log_path.parent)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.INFO)

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    ch.setFormatter(fmt)
    fh.setFormatter(fmt)

    logger.addHandler(ch)
    logger.addHandler(fh)

    return logger, log_path


def load_api_key(logger: logging.Logger, env_var_name: str = "GOOGLE_API_KEY", dotenv_override: Optional[Path] = None) -> str:
    """
    Looks up API key:
    1) environment variable env_var_name if set
    2) falls back to .env located at PARENT_DIR.parent/.env or provided dotenv_override
    """
    key = os.getenv(env_var_name)
    if key:
        return key

    env_path = dotenv_override or (PARENT_DIR.parent / ".env")
    load_dotenv(dotenv_path=env_path)
    key = os.getenv(env_var_name)
    if not key:
        logger.error(f"Environment variable {env_var_name} not found. Ensure it's set (or present in .env).")
        sys.exit(1)
    return key


def init_model(api_key: str) -> genai.GenerativeModel:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(MODEL_NAME)
    return model


def resolve_input_file(logger: logging.Logger) -> Path:
    base_dir = INPUT_DIR
    pattern = str(base_dir / INPUT_PATTERN)
    matches = glob.glob(pattern)
    if not matches:
        logger.error(f"No input files found for pattern: {pattern}")
        sys.exit(1)
    matches.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    chosen = Path(matches[0]).resolve()
    logger.info(f"Found {len(matches)} file(s) matching pattern. Using latest: {chosen}")
    return chosen


def read_jsonl(path: Path, logger: logging.Logger) -> List[Dict[str, Any]]:
    if not path.exists():
        logger.error(f"Input JSONL not found at: {path}")
        sys.exit(1)

    items = []
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                items.append(obj)
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping malformed JSON at line {line_num}: {e}")
    return items


def build_prompt(question: str, ground_truth: str, candidates: Dict[str, str]) -> str:
    schema_entries = []
    for i, label in enumerate(candidates.keys()):
        comma = "," if i < len(candidates) - 1 else ""
        schema_entries.append(f'  "{label}": {{ "score": 0 | 1 | 2, "reason": "short explanation" }}{comma}')
    schema = "{\n" + "\n".join(schema_entries) + "\n}"

    cand_blocks = []
    for label, answer in candidates.items():
        cand_blocks.append(f"{label}:\n{answer}")

    return f"""
You are an impartial evaluator. Score each candidate answer strictly against the Ground Truth.

Scoring:
- 0 = Did not answer the question OR is incorrect/contradicts the Ground Truth.
- 1 = Partially correct per the Ground Truth (some correct elements but missing or includes inaccuracies).
- 2 = Correct per the Ground Truth (fully aligns; extra info is allowed only if it does not change the meaning or contradict).

Rules:
- Judge only on factual alignment to the Ground Truth.
- Ignore style, verbosity, or citations unless they introduce contradictions.
- Output valid, minimal JSON only. No code fences, no extra commentary.
- Use EXACTLY these keys (do not rename): {", ".join(candidates.keys())}

Output JSON schema:
{schema}

Question:
{question}

Ground Truth:
{ground_truth}

Candidate Answers:
{chr(10).join(cand_blocks)}
""".strip()


def extract_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None

    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None

    candidate = text[start:end + 1].strip()

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    cleaned = re.sub(r"\n", " ", candidate)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return None


def call_llm_with_retry(model: genai.GenerativeModel,
                        limiter: RateLimiter,
                        prompt: str,
                        logger: logging.Logger,
                        q_index: int,
                        total: int,
                        expected_labels: List[str],
                        qid: Any,
                        max_attempts: int = MAX_ATTEMPTS) -> Tuple[Optional[Dict[str, Any]], float, int, Optional[Dict[str, Any]]]:
    """
    Returns:
        parsed_json (or None if failed after max attempts),
        last_attempt_duration_seconds,
        attempts_made,
        error_info (dict with details if failed; None if success)
    """
    exp_norm = [normalize_label(x) for x in expected_labels]
    attempts = 0
    last_duration = 0.0
    last_error_info: Optional[Dict[str, Any]] = None
    last_text: Optional[str] = None

    while attempts < max_attempts:
        attempts += 1
        limiter.wait_for_slot(logger=logger)
        t0 = time.perf_counter()
        try:
            logger.info(f"[Q{q_index+1}/{total} | id={qid}] Attempt {attempts}: sending request...")
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    top_k=TOP_K,
                ),
            )
            last_duration = time.perf_counter() - t0

            text = getattr(response, "text", None)
            if not text and hasattr(response, "candidates") and response.candidates:
                parts = response.candidates[0].content.parts if response.candidates[0].content else []
                text = "".join(getattr(p, "text", "") for p in parts)
            last_text = text or ""

            parsed = extract_json(last_text)
            if not isinstance(parsed, dict):
                # JSON parse failure
                logger.warning(f"[Q{q_index+1} | id={qid}] Attempt {attempts}: Could not parse JSON. Retrying...")
                logger.info(f"[Q{q_index+1} | id={qid}] Attempt {attempts} duration: {last_duration:.3f}s - FAILED")
                last_error_info = {
                    "type": "parse_error",
                    "message": "Could not parse JSON from model output.",
                    "raw_text_excerpt": truncate(last_text, 500),
                }
                continue

            norm_parsed = {normalize_label(k): v for k, v in parsed.items()}

            def valid_score(x: Any) -> bool:
                return isinstance(x, int) and x in (0, 1, 2)

            missing_labels = [lbl for lbl in exp_norm if lbl not in norm_parsed]
            invalid_details: List[Dict[str, Any]] = []
            for lbl_norm in exp_norm:
                if lbl_norm not in norm_parsed:
                    continue
                sv = norm_parsed[lbl_norm]
                if not isinstance(sv, dict):
                    invalid_details.append({
                        "label": lbl_norm,
                        "issue": "value is not an object",
                        "observed_type": type(sv).__name__,
                    })
                else:
                    sc = sv.get("score", None)
                    if sc is None:
                        invalid_details.append({"label": lbl_norm, "issue": "missing score"})
                    elif not isinstance(sc, int):
                        invalid_details.append({"label": lbl_norm, "issue": "non-integer score", "observed_type": type(sc).__name__, "observed_value": sc})
                    elif sc not in (0, 1, 2):
                        invalid_details.append({"label": lbl_norm, "issue": "score out of allowed range", "observed_value": sc})

            if missing_labels or invalid_details:
                # Validation failure on expected labels/scores
                summary_invalid = [f"{d.get('label')}: {d.get('issue')}" for d in invalid_details]
                logger.warning(
                    f"[Q{q_index+1} | id={qid}] Attempt {attempts}: Missing/invalid expected labels/scores. "
                    f"Missing={missing_labels or '[]'} | Invalid={summary_invalid or '[]'}. Retrying..."
                )
                logger.info(f"[Q{q_index+1} | id={qid}] Attempt {attempts} duration: {last_duration:.3f}s - FAILED")
                last_error_info = {
                    "type": "validation_error",
                    "expected_labels": exp_norm,
                    "returned_labels": list(norm_parsed.keys()),
                    "missing_labels": missing_labels,
                    "invalid_details": invalid_details,
                    "raw_text_excerpt": truncate(last_text, 500),
                }
                time.sleep(random.uniform(50.0, 80.0))
                continue

            # Success
            logger.info(f"[Q{q_index+1} | id={qid}] Attempt {attempts} duration: {last_duration:.3f}s - SUCCESS")
            return parsed, last_duration, attempts, None

        except Exception as e:
            last_duration = time.perf_counter() - t0
            logger.warning(f"[Q{q_index+1} | id={qid}] Attempt {attempts} duration: {last_duration:.3f}s - EXCEPTION: {repr(e)}. Retrying...")
            last_error_info = {
                "type": "exception",
                "exception": repr(e),
            }
            time.sleep(random.uniform(50.0, 80.0))
            continue

    # Reached max attempts
    logger.error(
        f"[Q{q_index+1} | id={qid}] Max attempts reached ({max_attempts}). Aborting this question. "
        f"Last error: {json.dumps(last_error_info, ensure_ascii=False) if last_error_info else 'N/A'}"
    )
    return None, last_duration, attempts, last_error_info


def compute_csv_columns(all_labels: List[str], include_id: bool) -> List[str]:
    cols = []
    if include_id:
        cols.append("id")
    cols.extend(["question", "ground truth"])
    for lbl in all_labels:
        cols.append(f"answer by {lbl}")
    for lbl in all_labels:
        cols.append(f"{lbl} score")
        cols.append(f"reason for {lbl} score")
    return cols


def evaluate_dataset(items: List[Dict[str, Any]],
                     all_labels: List[str],
                     include_id: bool,
                     model: genai.GenerativeModel,
                     logger: logging.Logger) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    limiter = RateLimiter(RPM)

    rows: List[Dict[str, Any]] = []
    total_questions = len(items)
    total_attempts = 0
    successful_items = 0
    failed_attempts_count = 0
    total_duration_all_success = 0.0

    totals_by_label: Dict[str, int] = {lbl: 0 for lbl in all_labels}
    counts_by_label: Dict[str, int] = {lbl: 0 for lbl in all_labels}
    distributions_by_label: Dict[str, Dict[int, int]] = {
        lbl: {0: 0, 1: 0, 2: 0} for lbl in all_labels
    }

    t_start = time.perf_counter()

    logger.info(f"Starting evaluation of {total_questions} questions.")
    logger.info(f"Detected model labels in dataset: {', '.join(all_labels) if all_labels else '(none)'}")

    for idx, item in enumerate(items):
        qid = item.get("id", "")
        q = (item.get("question") or "").strip()
        gt = (item.get("ground_truth") or "").strip()

        if not (q and gt):
            logger.warning(f"[Q{idx+1}] id={qid} Missing question or ground truth. Skipping.")
            continue

        model_answers = collect_model_answers(item)
        expected_labels = [lbl for lbl in all_labels if lbl in model_answers and model_answers[lbl]]

        if not expected_labels:
            logger.warning(f"[Q{idx+1}] id={qid} No model answers found. Skipping.")
            continue

        candidates = {lbl: model_answers[lbl] for lbl in expected_labels}
        prompt = build_prompt(q, gt, candidates)

        parsed, duration, attempts, error_info = call_llm_with_retry(
            model, limiter, prompt, logger, idx, total_questions, expected_labels, qid=qid, max_attempts=MAX_ATTEMPTS
        )
        total_attempts += attempts

        # Failure path: mark -1 / N/A and continue
        if parsed is None:
            failed_attempts_count += attempts  # all attempts failed for this question

            # Build row with -1 score and N/A reason for expected labels
            row: Dict[str, Any] = {}
            if include_id:
                row["id"] = qid
            row["question"] = q
            row["ground truth"] = gt
            for lbl in all_labels:
                row[f"answer by {lbl}"] = model_answers.get(lbl, "")

            for lbl in expected_labels:
                row[f"{lbl} score"] = -1
                row[f"reason for {lbl} score"] = "N/A"

            for lbl in all_labels:
                row.setdefault(f"{lbl} score", "")
                row.setdefault(f"reason for {lbl} score", "")

            # Detailed diagnostics to the log for traceability
            if error_info:
                etype = error_info.get("type", "unknown")
                if etype == "validation_error":
                    logger.error(
                        f"[Q{idx+1}] id={qid} Validation failed after {attempts} attempt(s). "
                        f"Expected={error_info.get('expected_labels')} | Returned={error_info.get('returned_labels')} | "
                        f"Missing={error_info.get('missing_labels')} | Invalid={error_info.get('invalid_details')} | "
                        f"LastResponseExcerpt='{error_info.get('raw_text_excerpt')}'"
                    )
                elif etype == "parse_error":
                    logger.error(
                        f"[Q{idx+1}] id={qid} JSON parse failed after {attempts} attempt(s). "
                        f"LastResponseExcerpt='{error_info.get('raw_text_excerpt')}'"
                    )
                elif etype == "exception":
                    logger.error(
                        f"[Q{idx+1}] id={qid} Exception after {attempts} attempt(s): {error_info.get('exception')}"
                    )
                else:
                    logger.error(f"[Q{idx+1}] id={qid} Aborted after {attempts} attempt(s). Details: {error_info}")
            else:
                logger.error(f"[Q{idx+1}] id={qid} Aborted after {attempts} attempt(s). No error details available.")

            logger.info(
                f"[Q{idx+1}] id={qid} FAILED after {attempts} attempt(s). "
                + ", ".join([f"{lbl}: {row[f'{lbl} score'] if row[f'{lbl} score']!='' else '-'}" for lbl in all_labels])
            )

            rows.append(row)
            # Continue to next question
            continue

        # Success path
        failed_attempts_count += (attempts - 1)
        # Only successful items contribute to success stats
        successful_items += 1
        total_duration_all_success += duration

        parsed_norm = {normalize_label(k): v for k, v in parsed.items()}

        row: Dict[str, Any] = {}
        if include_id:
            row["id"] = qid
        row["question"] = q
        row["ground truth"] = gt
        for lbl in all_labels:
            row[f"answer by {lbl}"] = model_answers.get(lbl, "")

        for lbl in expected_labels:
            sv = parsed_norm.get(normalize_label(lbl), {})
            score = int(sv.get("score", 0))
            reason = str(sv.get("reason", "") or "").strip()

            row[f"{lbl} score"] = score
            row[f"reason for {lbl} score"] = reason

            totals_by_label[lbl] += score
            counts_by_label[lbl] += 1
            distributions_by_label[lbl][score] = distributions_by_label[lbl].get(score, 0) + 1

        for lbl in all_labels:
            row.setdefault(f"{lbl} score", "")
            row.setdefault(f"reason for {lbl} score", "")

        logger.info(
            f"[Q{idx+1}] id={qid} Completed after {attempts} attempt(s). "
            + ", ".join([f"{lbl}: {row[f'{lbl} score'] if row[f'{lbl} score']!='' else '-'}" for lbl in all_labels])
        )

        rows.append(row)

    t_end = time.perf_counter()
    total_duration = t_end - t_start
    avg_duration_success_attempt = (total_duration_all_success / successful_items) if successful_items else 0.0

    percentages_by_label: Dict[str, float] = {}
    for lbl in all_labels:
        possible = 2 * counts_by_label[lbl]
        pct = (100.0 * totals_by_label[lbl] / possible) if possible else 0.0
        percentages_by_label[lbl] = pct

    if all_labels:
        max_pct = max(percentages_by_label.values()) if percentages_by_label else 0.0
        winners = [lbl for lbl in all_labels if percentages_by_label.get(lbl, 0.0) == max_pct]
        winner = "Tie: " + ", ".join(winners) if len(winners) > 1 else (winners[0] if winners else "N/A")
    else:
        winner = "N/A"

    summary = {
        "total_questions": successful_items,  # kept for backward compatibility with existing logs
        "total_attempts": total_attempts,
        "successful_attempts": successful_items,
        "failed_attempts": failed_attempts_count,
        "total_duration_seconds": total_duration,
        "avg_duration_per_success_attempt_seconds": avg_duration_success_attempt,
        "totals_by_label": totals_by_label,
        "counts_by_label": counts_by_label,
        "distributions_by_label": distributions_by_label,
        "percentages_by_label": percentages_by_label,
        "winner": winner,
    }

    return rows, summary


def write_csv(rows: List[Dict[str, Any]], output_path: Path, columns: List[str], logger: logging.Logger) -> None:
    ensure_dirs(output_path.parent)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore", quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for r in rows:
            safe_row = {col: r.get(col, "") for col in columns}
            writer.writerow(safe_row)
    logger.info(f"CSV results written to: {output_path}")


def log_summary(summary: Dict[str, Any], all_labels: List[str], logger: logging.Logger) -> None:
    logger.info("----- Summary -----")
    logger.info(f"Total questions processed: {summary['total_questions']}")
    logger.info(f"Total attempts (incl. retries): {summary['total_attempts']}")
    logger.info(f"Successful attempts: {summary['successful_attempts']}")
    logger.info(f"Failed attempts: {summary['failed_attempts']}")
    logger.info(f"Total duration: {summary['total_duration_seconds']:.3f}s")
    logger.info(f"Average duration per successful attempt: {summary['avg_duration_per_success_attempt_seconds']:.3f}s")

    for lbl in all_labels:
        total = summary["totals_by_label"][lbl]
        count = summary["counts_by_label"][lbl]
        possible = 2 * count
        pct = summary["percentages_by_label"][lbl]
        dist = summary["distributions_by_label"][lbl]
        logger.info(
            f"{lbl} total score: {total} out of {possible} ({pct:.2f}%) | "
            f"distribution (0/1/2): {dist.get(0,0)}/{dist.get(1,0)}/{dist.get(2,0)} | "
            f"items scored: {count}"
        )

    logger.info(f"Winner: {summary['winner']}")
    logger.info("-------------------")


# -------------------------
# CLI
# -------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LLM Judge")
    # Input and sharding
    p.add_argument("--input-file", "--input-jsonl", dest="input_file", type=str, default=None,
                   help="Path to input JSONL to evaluate.")
    p.add_argument("--shard-index", type=int, default=None,
                   help="Shard index for round-robin distribution (1-based). Use with --num-shards.")
    p.add_argument("--num-shards", type=int, default=None, help="Total number of shards.")
    # Run controls and outputs
    p.add_argument("--run-ts", "--run-timestamp", dest="run_ts", type=str, default=None,
                   help="Override run timestamp used in filenames/logs.")
    p.add_argument("--output-suffix", type=str, default=None,
                   help="Suffix appended to output filenames/logs for this process (e.g., part_1).")
    p.add_argument("--output-csv", type=str, default=None, help="Explicit CSV output path.")
    p.add_argument("--log-file", type=str, default=None, help="Explicit log file path.")
    # Consistency helpers
    p.add_argument("--labels-file", type=str, default=None,
                   help="Path to JSON file with a list of labels to enforce consistent CSV columns across shards.")
    p.add_argument("--force-include-id", action="store_true",
                   help="Force include 'id' column even if missing in this shard.")
    # API and rate limits
    p.add_argument("--api-key-envvar", type=str, default="GOOGLE_API_KEY",
                   help="Environment variable name holding the API key.")
    p.add_argument("--rpm", type=int, default=None, help="Override requests-per-minute limit.")
    return p.parse_args()


def main():
    global RPM  # allow override via CLI
    args = parse_args()

    run_ts = args.run_ts or now_timestamp()

    # Compute default log path (can be overridden)
    log_path_override = None
    if args.log_file:
        log_path_override = Path(args.log_file)
    elif args.output_suffix:
        suffix = normalize_label(args.output_suffix)
        log_path_override = LOG_DIR / f"llm_judge_run_{run_ts}__{suffix}.txt"

    logger, log_path = configure_logging(run_ts, log_path_override)
    logger.info("LLM Judge run started.")
    logger.info(f"Model: {MODEL_NAME} | RPM limit: {RPM} | Max attempts per question: {MAX_ATTEMPTS}")

    if args.rpm:
        RPM = args.rpm
        logger.info(f"RPM overridden via CLI to: {RPM}")

    # Input path
    if args.input_file:
        input_path = Path(args.input_file).resolve()
        logger.info(f"Input file (override): {input_path}")
    else:
        input_path = resolve_input_file(logger)
        logger.info(f"Input file: {input_path}")

    # API key
    api_key = load_api_key(logger, env_var_name=args.api_key_envvar)
    model = init_model(api_key)

    # Load all items, then apply round-robin sharding if requested
    items_full = read_jsonl(input_path, logger)
    logger.info(f"Loaded {len(items_full)} records from dataset.")

    if args.num_shards and args.shard_index is not None and args.num_shards > 1:
        # 1-based shard index preferred; treat 0 as 0-based first shard for flexibility
        shard_idx_raw = int(args.shard_index)
        rr_idx = shard_idx_raw - 1 if shard_idx_raw >= 1 else shard_idx_raw
        if rr_idx < 0 or rr_idx >= args.num_shards:
            logger.error(f"Invalid shard settings: shard-index={args.shard_index}, num-shards={args.num_shards}")
            sys.exit(1)
        items = [it for j, it in enumerate(items_full) if (j % args.num_shards) == rr_idx]
        logger.info(f"Sharding enabled: shard {rr_idx+1}/{args.num_shards} -> {len(items)} item(s).")
    else:
        items = items_full

    # Labels
    if args.labels_file:
        try:
            with open(args.labels_file, "r", encoding="utf-8") as lf:
                all_labels = list(map(str, json.load(lf)))
            logger.info(f"Loaded {len(all_labels)} enforced labels from labels_file.")
        except Exception as e:
            logger.error(f"Failed to read labels_file '{args.labels_file}': {e}")
            sys.exit(1)
    else:
        all_labels = detect_all_labels(items)

    if not all_labels:
        logger.error("No model answers detected in dataset. Nothing to evaluate.")
        sys.exit(1)

    include_id = args.force_include_id or any("id" in it for it in items)

    rows, summary = evaluate_dataset(items, all_labels, include_id, model, logger)

    ensure_dirs(OUTPUT_DIR)

    # Decide output CSV path
    if args.output_csv:
        output_csv = Path(args.output_csv).resolve()
    else:
        base = f"{OUTPUT_CSV_BASENAME}_{run_ts}"
        if args.output_suffix:
            base += f"__{normalize_label(args.output_suffix)}"
        output_csv = OUTPUT_DIR / f"{base}.csv"

    csv_columns = compute_csv_columns(all_labels, include_id)
    write_csv(rows, output_csv, csv_columns, logger)

    log_summary(summary, all_labels, logger)
    logger.info(f"Logs saved to: {log_path}")
    logger.info("LLM Judge run finished.")


if __name__ == "__main__":
    main()