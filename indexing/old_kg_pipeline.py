import os, json, hashlib, time, threading, pickle, math, re, random
from collections import deque
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED

from dotenv import load_dotenv
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
from neo4j import GraphDatabase

# ----------------- Thread-safe logging with timestamps and attempt IDs -----------------
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
        print(f"{_fmt_prefix('INFO', attempt_id)} {msg}")

def log_warn(msg: str, attempt_id: Optional[int] = None) -> None:
    with _PRINT_LOCK:
        print(f"{_fmt_prefix('WARN', attempt_id)} {msg}")

def log_error(msg: str, attempt_id: Optional[int] = None) -> None:
    with _PRINT_LOCK:
        print(f"{_fmt_prefix('ERROR', attempt_id)} {msg}")

# ----------------- Load .env from the parent directory -----------------
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

# ----------------- Config from env with sensible defaults -----------------
GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]

def _normalize_model_name(m: str) -> str:
    if not m:
        return m
    m = m.strip()
    if m.startswith("models/") or m.startswith("tunedModels/"):
        return m
    return f"models/{m}"

GEN_MODEL = _normalize_model_name(os.getenv("GEN_MODEL", "gemini-2.5-flash-lite"))
EMBED_MODEL = _normalize_model_name(os.getenv("EMBED_MODEL", "text-embedding-004"))

# Directory of LangChain per-document pickle files
DEFAULT_LANGCHAIN_DIR = (Path(__file__).resolve().parent / "../dataset/langchain-results/samples").resolve()
LANGCHAIN_DIR = DEFAULT_LANGCHAIN_DIR

# Neo4j
NEO4J_URI  = os.getenv("NEO4J_URI",  "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASS", "@ik4nkus")

# LLM rate limiter (calls per minute)
LLM_MAX_CALLS_PER_MIN = int(os.getenv("LLM_MAX_CALLS_PER_MIN", "15"))

# Parallelism
INDEX_WORKERS = int(os.getenv("INDEX_WORKERS", "15"))

# Stagger worker starts (seconds) - randomized uniformly in [5.0, 15.0]
STAGGER_WORKER_SECONDS = random.uniform(7.0, 17.0)

# API budget controls
def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "y", "on")

ENFORCE_API_BUDGET = _env_bool("ENFORCE_API_BUDGET", True)
API_BUDGET_TOTAL = int(os.getenv("API_BUDGET_TOTAL", "700"))
COUNT_EMBEDDINGS_IN_BUDGET = _env_bool("COUNT_EMBEDDINGS_IN_BUDGET", False)

# Files to skip explicitly
SKIP_FILES = {"all_langchain_documents.pkl"}

# Embedding dimensions for text-embedding-004 (for reference)
EMBED_DIM = 768

# Prompt token control (heuristic)
PROMPT_TOKEN_LIMIT = int(os.getenv("PROMPT_TOKEN_LIMIT", "8000"))
PRACTICAL_MAX_ITEMS_PER_BATCH = int(os.getenv("PRACTICAL_MAX_ITEMS_PER_BATCH", "40"))

# Single-chunk parse error split threshold
SINGLE_PARSE_SPLIT_AFTER = int(os.getenv("SINGLE_PARSE_SPLIT_AFTER", "2"))

# JSON output directory (must already exist; do not create it)
JSON_OUTPUT_DIR = Path(os.getenv("JSON_OUTPUT_DIR", str((Path(__file__).resolve().parent / "../../dataset/llm-json-outputs").resolve())))
if not JSON_OUTPUT_DIR.exists() or not JSON_OUTPUT_DIR.is_dir():
    raise FileNotFoundError(f"JSON_OUTPUT_DIR does not exist or is not a directory: {JSON_OUTPUT_DIR}. Please create it or set JSON_OUTPUT_DIR.")

# ----------------- Initialize SDKs -----------------
genai.configure(api_key=GOOGLE_API_KEY)
try:
    gen_model = genai.GenerativeModel(GEN_MODEL)
    log_info(f"Using generation model: {GEN_MODEL}")
    log_info(f"Using embedding model:  {EMBED_MODEL}")
    log_info(f"LLM JSON outputs directory: {JSON_OUTPUT_DIR}")
except Exception as e:
    raise RuntimeError(f"Failed to initialize GenerativeModel {GEN_MODEL}: {e}")

# ----------------- Indonesian vocab -----------------
LEGAL_ENTITY_TYPES = [
    "UU", "PASAL", "AYAT", "INSTANSI", "ORANG", "ISTILAH", "SANKSI", "NOMINAL", "TANGGAL"
]
LEGAL_PREDICATES = [
    "mendefinisikan",
    "mengubah",
    "mencabut",
    "mulai_berlaku",
    "mewajibkan",
    "melarang",
    "memberikan_sanksi",
    "berlaku_untuk",
    "termuat_dalam",
    "mendelegasikan_kepada",
    "berjumlah",
    "berdurasi"
]

# Schemas
TRIPLE_SCHEMA = {
  "type": "object",
  "properties": {
    "triples": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "subject": {
            "type": "object",
            "properties": {
              "text": {"type": "string"},
              "type": {"type": "string"},
              "canonical_id": {"type": "string"}
            },
            "required": ["text"]
          },
          "predicate": {"type": "string"},
          "object": {
            "type": "object",
            "properties": {
              "text": {"type": "string"},
              "type": {"type": "string"},
              "canonical_id": {"type": "string"}
            },
            "required": ["text"]
          },
          "evidence": {
            "type": "object",
            "properties": {
              "quote": {"type": "string"},
              "char_start": {"type": "integer"},
              "char_end": {"type": "integer"},
              "article_ref": {"type": "string"}
            }
          },
          "confidence": {"type": "number"}
        },
        "required": ["subject", "predicate", "object"]
      }
    }
  },
  "required": ["triples"]
}

BATCH_SCHEMA = {
  "type": "object",
  "properties": {
    "results": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "chunk_id": {"type": "string"},
          "triples": TRIPLE_SCHEMA["properties"]["triples"]
        },
        "required": ["chunk_id", "triples"]
      }
    }
  },
  "required": ["results"]
}

SYSTEM_HINT = f"""
You extract knowledge graph triples from Indonesian legal text (Undang-Undang).

Output requirements:
- subject.type, object.type, and predicate MUST be Indonesian strings only.
- Allowed entity types: {", ".join(LEGAL_ENTITY_TYPES)}.
- Prefer predicates from: {", ".join(LEGAL_PREDICATES)} (snake_case where applicable).
- Use 'UU' for the Law, 'PASAL' for Article, 'AYAT' for Clause, 'INSTANSI' for institutions/agencies, 'ISTILAH' for defined terms.

Rules:
- Be precise and strictly grounded in the chunk; if unsupported, omit the triple.
- Include a short evidence.quote, and article_ref like 'Pasal X ayat (Y)' when present.
- Normalize predicate to a lowercase Indonesian single-verb phrase (use the list above when possible).
- Keep numbers/dates verbatim from the text.
"""

# ----------------- Attempt ID generator -----------------
_attempt_lock = threading.Lock()
_attempt_counter = 0

def next_attempt_id() -> int:
    global _attempt_counter
    with _attempt_lock:
        aid = _attempt_counter
        _attempt_counter += 1
        return aid

# ----------------- Rate Limiter (LLM) -----------------
class RateLimiter:
    def __init__(self, max_calls: int, period_sec: float):
        self.max_calls = max_calls
        self.period = period_sec
        self.calls = deque()
        self.lock = threading.Lock()

    def acquire(self, attempt_id: Optional[int] = None):
        with self.lock:
            now = time.time()
            while self.calls and (now - self.calls[0]) >= self.period:
                self.calls.popleft()
            wait = 0.0
            if len(self.calls) >= self.max_calls:
                wait = self.period - (now - self.calls[0])
        if wait > 0:
            log_info(f"[RateLimiter] Sleeping {wait:.2f}s to respect client-side RPM limit", attempt_id=attempt_id)
            time.sleep(wait)
        with self.lock:
            self.calls.append(time.time())

LLM_RATE_LIMITER = RateLimiter(max_calls=LLM_MAX_CALLS_PER_MIN, period_sec=60.0)

# ----------------- Global API Budget -----------------
class ApiBudget:
    def __init__(self, total: int, enforce: bool, count_embeddings: bool):
        self.total = total
        self.enforce = enforce
        self.count_embeddings = count_embeddings
        self.used = 0
        self.lock = threading.Lock()
        self.resource_usage = {"llm": 0, "embed": 0}

    def _should_count(self, kind: str) -> bool:
        if kind == "embed" and not self.count_embeddings:
            return False
        return True

    def will_allow(self, kind: str, n: int = 1) -> bool:
        if not self.enforce or not self._should_count(kind):
            return True
        with self.lock:
            return (self.used + n) <= self.total

    def register(self, kind: str, n: int = 1):
        if not self.enforce or not self._should_count(kind):
            return
        with self.lock:
            if (self.used + n) > self.total:
                raise RuntimeError(f"API budget exceeded: used={self.used}, trying={n}, total={self.total}")
            self.used += n
            self.resource_usage[kind] = self.resource_usage.get(kind, 0) + n

API_BUDGET = ApiBudget(total=API_BUDGET_TOTAL, enforce=ENFORCE_API_BUDGET, count_embeddings=COUNT_EMBEDDINGS_IN_BUDGET)

# ----------------- Helpers -----------------
def _slug(s: str) -> str:
    return " ".join((s or "").strip().split()).lower()

def sanitize_filename_component(s: str) -> str:
    s = (s or "").strip().lower().replace(" ", "-")
    s = re.sub(r"[^a-z0-9_\-+]", "", s)
    s = re.sub(r"-{2,}", "-", s)
    return s or "unknown"

def normalize_entity_key(text: str, etype: str, uu_number: Optional[str] = None) -> str:
    if etype in ("UU", "PASAL", "AYAT") and uu_number:
        return f"{etype}::{_slug(uu_number)}::{_slug(text)}"
    return f"{etype}::{_slug(text)}"

def deterministic_triple_uid(
    subject_key: str, predicate: str, object_key: str, doc_id: Optional[str], span: Optional[Tuple[int,int]]
) -> str:
    h = hashlib.sha256()
    payload = "|".join([
        subject_key, (predicate or "").strip().lower(), object_key, doc_id or "",
        str(span[0]) if span else "", str(span[1]) if span else ""
    ])
    h.update(payload.encode("utf-8"))
    return h.hexdigest()

def estimate_tokens_for_text(text: str) -> int:
    # Rough heuristic: ~3.5 chars/token to be conservative
    return max(1, int(len(text) / 3.5))

def aggregate_pages(items_metas: List[Dict[str, Any]]) -> List[int]:
    pages: set[int] = set()
    for m in items_metas:
        p = m.get("pages")
        if isinstance(p, (list, tuple, set)):
            for x in p:
                if isinstance(x, int):
                    pages.add(x)
                elif isinstance(x, str):
                    for tok in re.split(r"[,\s\-]+", x):
                        if tok.isdigit():
                            pages.add(int(tok))
        elif isinstance(p, int):
            pages.add(p)
        elif isinstance(p, str):
            for tok in re.split(r"[,\s\-]+", p):
                if tok.isdigit():
                    pages.add(int(tok))
    return sorted(pages)

def build_json_output_filename(context: Dict[str, Any]) -> Path:
    kind = context.get("kind", "batch")
    items_metas: List[Dict[str, Any]] = context.get("items_metas", [])
    items_count = context.get("items_count", len(items_metas) or "?")

    uu_set = {m.get("uu_number") for m in items_metas if m.get("uu_number")}
    if len(uu_set) == 1:
        uu_lab = f"uu-{sanitize_filename_component(list(uu_set)[0])}"
    elif len(uu_set) == 0:
        uu_lab = "uu-unknown"
    else:
        uu_lab = "uu-multi"

    pages_sorted = aggregate_pages(items_metas)
    if pages_sorted:
        pmin, pmax = pages_sorted[0], pages_sorted[-1]
        pages_lab = f"p-{pmin}" if pmin == pmax else f"p-{pmin}-{pmax}"
    else:
        pages_lab = "p-unknown"

    chunk_ids = [m.get("chunk_id") or "" for m in items_metas]
    short = hashlib.sha1("|".join(sorted(chunk_ids)).encode("utf-8")).hexdigest()[:8]
    ts = time.strftime("%Y%m%d-%H%M%S")

    fname = f"{uu_lab}__{pages_lab}__{kind}__{items_count}__{ts}_{short}.json"
    return (JSON_OUTPUT_DIR / fname)

def save_llm_json_output(data: Dict[str, Any], context: Dict[str, Any], attempt_id: int) -> None:
    try:
        out_path = build_json_output_filename(context)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        log_info(f"[LLM] Saved JSON output: {out_path}", attempt_id=attempt_id)
    except Exception as e:
        log_warn(f"[LLM] Warning: failed to save JSON output: {e}", attempt_id=attempt_id)

def split_text_in_two(text: str) -> Tuple[str, str]:
    """
    Split text into two halves near the middle, preferring paragraph/sentence/space boundaries.
    """
    n = len(text)
    if n <= 1:
        return text, ""
    mid = n // 2
    window = 300
    start = max(0, mid - window)
    end = min(n, mid + window)
    seps = ["\n\n", "\n", ". ", " "]
    best_idx = None
    for sep in seps:
        li = text.rfind(sep, 0, mid)
        ri = text.find(sep, mid, end)
        candidates = []
        if li != -1:
            candidates.append((abs(mid - (li + len(sep)//2)), li + len(sep)))
        if ri != -1:
            candidates.append((abs(ri - mid), ri + len(sep)))
        if candidates:
            candidates.sort(key=lambda x: x[0])
            best_idx = candidates[0][1]
            break
    if best_idx is None:
        best_idx = mid
    left = text[:best_idx].strip()
    right = text[best_idx:].strip()
    return left, right

# ----------------- Prompt builders -----------------
def build_single_prompt(meta: Dict[str, Any], chunk_text: str) -> str:
    return f"""
{SYSTEM_HINT}

Chunk metadata:
- document_id: {meta.get('document_id')}
- chunk_id: {meta.get('chunk_id')}
- uu_number: {meta.get('uu_number')}
- pages: {meta.get('pages')}

Text:
\"\"\"{chunk_text}\"\"\"
"""

def build_batch_prompt(items: List[Dict[str, Any]]) -> str:
    intro = f"{SYSTEM_HINT}\n\nYou will receive several chunks. For each chunk, return an object with 'chunk_id' and 'triples'."
    lines = []
    for it in items:
        m = it["meta"]
        lines.append(
            f"- chunk_id: {m.get('chunk_id')} | document_id: {m.get('document_id')} | uu_number: {m.get('uu_number')} | pages: {m.get('pages')}\n"
            f"TEXT:\n\"\"\"{it['text']}\"\"\""
        )
    return intro + "\n\nChunks:\n" + "\n\n".join(lines)

# ----------------- Error classes and classification -----------------
class RateLimitError(Exception):
    def __init__(self, message: str, attempt_id: int, category: str = "unknown", reason: str = ""):
        super().__init__(message)
        self.attempt_id = attempt_id
        self.category = category  # 'rpm', 'quota', or 'unknown'
        self.reason = reason

class JsonParseError(Exception):
    def __init__(self, message: str, attempt_id: int):
        super().__init__(message)
        self.attempt_id = attempt_id

class LlmCallError(RuntimeError):
    def __init__(self, message: str, attempt_id: int):
        super().__init__(message)
        self.attempt_id = attempt_id

def is_rate_limit_error(e: Exception) -> bool:
    msg = str(e).lower()
    if "429" in msg or "rate limit" in msg or "too many requests" in msg or "quota" in msg or "resourceexhausted" in msg or "resource exhausted" in msg:
        return True
    code = getattr(e, "code", None)
    if code == 429:
        return True
    status = getattr(e, "status", None) or getattr(e, "reason", None)
    if isinstance(status, str) and ("exhausted" in status.lower() or "429" in status):
        return True
    return False

def classify_rate_limit_error(e: Exception) -> Tuple[str, str]:
    """
    Returns (category, reason) where category in {'rpm','quota','unknown'}.
    Heuristics based on message text from the Gemini API.
    """
    msg = (str(e) or "").lower()
    # Any mention of quota/billing -> quota
    if "quota" in msg or "exceeded your current quota" in msg or "billing" in msg:
        return "quota", "quota exceeded"
    # Generic rate/too many requests/resource exhausted without quota -> rpm
    if "rate" in msg or "too many requests" in msg or "resource exhausted" in msg or "exhausted" in msg:
        return "rpm", "request rate exceeded"
    return "unknown", "unclassified 429"

# ----------------- LLM call wrapper (logs every attempt) -----------------
def run_llm_json(prompt: str, schema: dict, context: Dict[str, Any]) -> Tuple[Dict[str, Any], float, int]:
    """
    Makes a single LLM call with JSON schema and returns parsed JSON, duration, and attempt_id.
    Raises RateLimitError, JsonParseError, or LlmCallError on failures.
    Budget note: budget is charged ONLY after successful call and JSON parsing.
    Also saves the successful JSON output to a file (one file per batch).
    """
    # Budget gate (we only check; we will register after success)
    if not API_BUDGET.will_allow("llm", 1):
        raise RuntimeError("API budget would be exceeded by another LLM call; stopping extraction.")

    # Allocate global attempt ID first so we can tag all subsequent logs
    attempt_id = next_attempt_id()

    # Respect client-side rate limiter
    LLM_RATE_LIMITER.acquire(attempt_id=attempt_id)

    # Log the attempt
    try:
        kind = context.get("kind", "unknown")
        items_count = context.get("items_count", "?")
        token_est = estimate_tokens_for_text(prompt)
        # Use '~' for compatibility with Windows terminals
        log_info(f"[LLM] Attempt: model={GEN_MODEL} | kind={kind} | items={items_count} | est_tokens~{token_est}", attempt_id=attempt_id)
    except Exception:
        log_info(f"[LLM] Attempt: model={GEN_MODEL}", attempt_id=attempt_id)

    cfg = GenerationConfig(
        temperature=0,
        response_mime_type="application/json",
        response_schema=schema,
    )

    start = time.time()
    try:
        resp = gen_model.generate_content(prompt, generation_config=cfg)
    except Exception as e:
        dur = time.time() - start
        if is_rate_limit_error(e):
            cat, reason = classify_rate_limit_error(e)
            raise RateLimitError(f"LLM rate limit error ({cat}): {e}", attempt_id=attempt_id, category=cat, reason=reason) from e
        raise LlmCallError(f"LLM call failed: {e}", attempt_id=attempt_id) from e

    dur = time.time() - start

    # Parse JSON
    raw = None
    try:
        raw = resp.text
        data = json.loads(raw)
    except Exception:
        try:
            raw = resp.candidates[0].content.parts[0].text
            data = json.loads(raw)
        except Exception as e:
            preview = raw[:200] if isinstance(raw, str) else "None"
            raise JsonParseError(f"Failed to parse model JSON: {e}; raw={preview}", attempt_id=attempt_id) from e

    # Budget: count only successful, parsed responses
    API_BUDGET.register("llm", 1)

    # Save JSON output to file
    save_llm_json_output(data, context, attempt_id=attempt_id)

    return data, dur, attempt_id

# ----------------- LLM Extraction (single attempt) -----------------
def extract_triples_from_chunk(chunk_text: str, meta: Dict[str, Any], prompt_override: Optional[str] = None) -> Tuple[List[Dict[str, Any]], float, int]:
    prompt = prompt_override or build_single_prompt(meta, chunk_text)
    data, gemini_duration, attempt_id = run_llm_json(
        prompt, TRIPLE_SCHEMA,
        context={"kind": "single", "items_count": 1, "chunk_id": meta.get("chunk_id"), "items_metas": [meta]}
    )

    uu_number = meta.get("uu_number")
    triples: List[Dict[str, Any]] = []
    for t in data.get("triples", []):
        subj = t["subject"]; obj = t["object"]; pred = (t["predicate"] or "").strip().lower()
        s_type = subj.get("type") or "ISTILAH"
        o_type = obj.get("type") or "ISTILAH"
        s_key = normalize_entity_key(subj["text"], s_type, uu_number)
        o_key = normalize_entity_key(obj["text"],  o_type,  uu_number)

        ev = t.get("evidence") or {}
        span = (ev["char_start"], ev["char_end"]) if ev.get("char_start") is not None and ev.get("char_end") is not None else None
        triple_uid = deterministic_triple_uid(s_key, pred, o_key, meta.get("document_id"), span)

        triples.append({
            "triple_uid": triple_uid,
            "subject": {"text": subj["text"], "type": s_type, "key": s_key},
            "predicate": pred,
            "object": {"text": obj["text"], "type": o_type, "key": o_key},
            "evidence": {
                "quote": ev.get("quote"),
                "char_start": ev.get("char_start"),
                "char_end": ev.get("char_end"),
                "article_ref": ev.get("article_ref"),
            },
            "confidence": float(t.get("confidence", 0.0)),
            "provenance": {
                "document_id": meta.get("document_id"),
                "chunk_id": meta.get("chunk_id"),
                "uu_number": uu_number,
                "pages": meta.get("pages"),
            }
        })
    return triples, gemini_duration, attempt_id

def extract_triples_from_chunks_batch(items: List[Dict[str, Any]], prompt: str) -> Tuple[Dict[str, List[Dict[str, Any]]], float, int]:
    metas = [it["meta"] for it in items]
    data, gemini_duration, attempt_id = run_llm_json(
        prompt, BATCH_SCHEMA,
        context={"kind": "batch", "items_count": len(items), "items_metas": metas}
    )

    results_map: Dict[str, List[Dict[str, Any]]] = {}
    meta_map = {it["meta"]["chunk_id"]: it["meta"] for it in items}

    for res in data.get("results", []):
        cid = res.get("chunk_id")
        if not cid or cid not in meta_map:
            continue
        meta = meta_map[cid]
        uu_number = meta.get("uu_number")
        triples_for_chunk: List[Dict[str, Any]] = []
        for t in res.get("triples", []):
            subj = t["subject"]; obj = t["object"]; pred = (t["predicate"] or "").strip().lower()
            s_type = subj.get("type") or "ISTILAH"
            o_type = obj.get("type") or "ISTILAH"
            s_key = normalize_entity_key(subj["text"], s_type, uu_number)
            o_key = normalize_entity_key(obj["text"],  o_type,  uu_number)

            ev = t.get("evidence") or {}
            span = (ev["char_start"], ev["char_end"]) if ev.get("char_start") is not None and ev.get("char_end") is not None else None
            triple_uid = deterministic_triple_uid(s_key, pred, o_key, meta.get("document_id"), span)

            triples_for_chunk.append({
                "triple_uid": triple_uid,
                "subject": {"text": subj["text"], "type": s_type, "key": s_key},
                "predicate": pred,
                "object": {"text": obj["text"], "type": o_type, "key": o_key},
                "evidence": {
                    "quote": ev.get("quote"),
                    "char_start": ev.get("char_start"),
                    "char_end": ev.get("char_end"),
                    "article_ref": ev.get("article_ref"),
                },
                "confidence": float(t.get("confidence", 0.0)),
                "provenance": {
                    "document_id": meta.get("document_id"),
                    "chunk_id": meta.get("chunk_id"),
                    "uu_number": uu_number,
                    "pages": meta.get("pages"),
                }
            })
        results_map[cid] = triples_for_chunk

    return results_map, gemini_duration, attempt_id

# ----------------- Embeddings -----------------
def embed_text(text: str) -> Tuple[List[float], float]:
    if not API_BUDGET.will_allow("embed", 1):
        raise RuntimeError("API budget would be exceeded by another embedding call; stopping embedding.")
    start = time.time()
    try:
        res = genai.embed_content(model=EMBED_MODEL, content=text)
    except Exception as e:
        msg = str(e)
        if "EmbedContentRequest.model" in msg and "unexpected model name format" in msg.lower():
            raise RuntimeError(f"Embedding model name invalid: {EMBED_MODEL}. Use 'models/text-embedding-004' or set EMBED_MODEL accordingly.") from e
        raise
    dur = time.time() - start

    vec: Optional[List[float]] = None

    if isinstance(res, dict):
        emb = res.get("embedding")
        if isinstance(emb, dict) and "values" in emb:
            vec = emb["values"]
        elif isinstance(emb, list):
            vec = emb
    if vec is None:
        try:
            vec = res.embedding.values  # type: ignore[attr-defined]
        except Exception:
            pass
    if vec is None:
        raise RuntimeError("Unexpected embedding response shape")

    # Budget: count only successful embedding responses (if embeddings are counted)
    API_BUDGET.register("embed", 1)

    return vec, dur

def node_embedding_text(name: str, etype: str) -> str:
    return f"{(name or '').strip()} | {etype}"

def triple_embedding_text(t: Dict[str, Any]) -> str:
    s = t["subject"]["text"]; p = t["predicate"]; o = t["object"]["text"]
    uu = t["provenance"].get("uu_number") or ""
    art = (t.get("evidence") or {}).get("article_ref") or ""
    return f"{s} [{p}] {o} | {uu} | {art}"

# ----------------- Neo4j Storage -----------------
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

_entity_emb_cache: dict[str, List[float]] = {}
_triple_emb_cache: dict[str, List[float]] = {}
_entity_emb_cache_lock = threading.Lock()
_triple_emb_cache_lock = threading.Lock()

def _get_entity_emb(name: str, etype: str, key: str) -> Tuple[List[float], float]:
    with _entity_emb_cache_lock:
        cached = _entity_emb_cache.get(key)
    if cached is not None:
        return cached, 0.0
    vec, dur = embed_text(node_embedding_text(name, etype))
    with _entity_emb_cache_lock:
        _entity_emb_cache[key] = vec
    return vec, dur

def _get_triple_emb(triple_uid: str, t: Dict[str, Any]) -> Tuple[List[float], float]:
    with _triple_emb_cache_lock:
        cached = _triple_emb_cache.get(triple_uid)
    if cached is not None:
        return cached, 0.0
    vec, dur = embed_text(triple_embedding_text(t))
    with _triple_emb_cache_lock:
        _triple_emb_cache[triple_uid] = vec
    return vec, dur

def upsert_triple(tx, t: Dict[str, Any], s_emb: List[float], o_emb: List[float], tr_emb: List[float]) -> float:
    s, o = t["subject"], t["object"]
    s_name, s_type, s_key = s["text"], s["type"], s["key"]
    o_name, o_type, o_key = o["text"], o["type"], o["key"]
    pred = t["predicate"]
    triple_uid = t["triple_uid"]

    prov = t["provenance"]
    ev = t.get("evidence") or {}
    confidence = float(t.get("confidence", 0.0))

    start = time.time()
    cypher = """
    MERGE (s:Entity {key:$s_key})
      ON CREATE SET s.name=$s_name, s.type=$s_type, s.createdAt=timestamp()
    SET s.embedding=$s_emb

    MERGE (o:Entity {key:$o_key})
      ON CREATE SET o.name=$o_name, o.type=$o_type, o.createdAt=timestamp()
    SET o.embedding=$o_emb

    MERGE (tr:Triple {triple_uid:$triple_uid})
      ON CREATE SET tr.createdAt=timestamp()
    SET tr.predicate=$pred,
        tr.embedding=$tr_emb,
        tr.document_id=$doc_id,
        tr.chunk_id=$chunk_id,
        tr.uu_number=$uu_number,
        tr.pages=$pages,
        tr.evidence_quote=$evidence_quote,
        tr.evidence_char_start=$evidence_char_start,
        tr.evidence_char_end=$evidence_char_end,
        tr.evidence_article_ref=$evidence_article_ref,
        tr.confidence=$confidence

    MERGE (tr)-[:SUBJECT]->(s)
    MERGE (tr)-[:OBJECT]->(o)

    MERGE (s)-[r:REL {triple_uid:$triple_uid}]->(o)
    SET r.predicate=$pred, r.chunk_id=$chunk_id, r.document_id=$doc_id
    """
    tx.run(
        cypher,
        s_key=s_key, s_name=s_name, s_type=s_type, s_emb=s_emb,
        o_key=o_key, o_name=o_name, o_type=o_type, o_emb=o_emb,
        triple_uid=triple_uid, pred=pred, tr_emb=tr_emb,
        doc_id=prov.get("document_id"),
        chunk_id=prov.get("chunk_id"),
        uu_number=prov.get("uu_number"),
        pages=prov.get("pages", []),
        evidence_quote=ev.get("quote"),
        evidence_char_start=ev.get("char_start"),
        evidence_char_end=ev.get("char_end"),
        evidence_article_ref=ev.get("article_ref"),
        confidence=confidence,
    )
    return time.time() - start

def write_triples_for_chunk(triples: List[Dict[str, Any]]) -> Tuple[int, float, float]:
    if not triples:
        return 0, 0.0, 0.0

    total_emb_dur = 0.0
    total_neo4j_dur = 0.0
    written = 0

    with driver.session() as session:
        for t in triples:
            s = t["subject"]; o = t["object"]
            s_emb, d1 = _get_entity_emb(s["text"], s["type"], s["key"])
            o_emb, d2 = _get_entity_emb(o["text"], o["type"], o["key"])
            tr_emb, d3 = _get_triple_emb(t["triple_uid"], t)
            total_emb_dur += (d1 + d2 + d3)

            neo4j_dur = session.execute_write(upsert_triple, t, s_emb, o_emb, tr_emb)
            total_neo4j_dur += neo4j_dur
            written += 1

    return written, total_emb_dur, total_neo4j_dur

# ----------------- Folder utilities -----------------
def list_pickles(dir_path: Path) -> List[Path]:
    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {dir_path}")
    return sorted([p for p in dir_path.glob("*.pkl") if p.name not in SKIP_FILES])

def load_chunks_from_file(pkl_path: Path) -> List[Any]:
    with open(pkl_path, "rb") as f:
        chunks = pickle.load(f)
    if not isinstance(chunks, list):
        raise ValueError(f"Unexpected pickle content in {pkl_path}")
    log_info(f"  - {pkl_path.name}: {len(chunks)} chunks")
    return chunks

# ----------------- Greedy and budget-aware batch builders -----------------
def build_greedy_batch(use_items: List[Dict[str, Any]], start_idx: int) -> Tuple[List[Dict[str, Any]], int, int]:
    items_batch: List[Dict[str, Any]] = []
    idx = start_idx
    est_tokens = 0

    while idx < len(use_items) and len(items_batch) < PRACTICAL_MAX_ITEMS_PER_BATCH:
        candidate = use_items[idx]
        tentative = items_batch + [candidate]
        prompt = build_batch_prompt(tentative) if len(tentative) > 1 else build_single_prompt(candidate["meta"], candidate["text"])
        tokens = estimate_tokens_for_text(prompt)
        if tokens <= PROMPT_TOKEN_LIMIT or len(items_batch) == 0:
            items_batch = tentative
            est_tokens = tokens
            idx += 1
        else:
            break

    return items_batch, idx, est_tokens

def pack_batches(use_items: List[Dict[str, Any]]) -> List[Tuple[List[Dict[str, Any]], int]]:
    batches: List[Tuple[List[Dict[str, Any]], int]] = []
    i = 0
    while i < len(use_items):
        items_batch, next_i, est_tokens = build_greedy_batch(use_items, i)
        if not items_batch:
            i += 1
            continue
        batches.append((items_batch, est_tokens))
        i = next_i
    return batches

def try_merge_batches(b1: Tuple[List[Dict[str, Any]], int], b2: Tuple[List[Dict[str, Any]], int]) -> Optional[Tuple[List[Dict[str, Any]], int]]:
    items1, _ = b1
    items2, _ = b2
    merged = items1 + items2
    if len(merged) > PRACTICAL_MAX_ITEMS_PER_BATCH:
        return None
    merged_prompt = build_batch_prompt(merged) if len(merged) > 1 else build_single_prompt(merged[0]["meta"], merged[0]["text"])
    tokens = estimate_tokens_for_text(merged_prompt)
    if tokens <= PROMPT_TOKEN_LIMIT:
        return (merged, tokens)
    return None

def pack_batches_with_cap(use_items: List[Dict[str, Any]], max_batches_allowed: Optional[int]) -> Tuple[List[Tuple[List[Dict[str, Any]], int]], int]:
    """
    Build batches greedily and then enforce a hard cap on number of batches.
    If we exceed the cap, attempt to merge adjacent batches within token limits.
    If still over the cap, truncate extra batches (defer remaining chunks).

    Returns:
      (batches_to_run, deferred_chunks_count)
    """
    batches = pack_batches(use_items)
    if max_batches_allowed is None:
        return batches, 0

    if len(batches) <= max_batches_allowed:
        return batches, 0

    # Try to merge adjacent batches to reduce count
    changed = True
    while len(batches) > max_batches_allowed and changed:
        changed = False
        merged_list: List[Tuple[List[Dict[str, Any]], int]] = []
        i = 0
        while i < len(batches):
            if i < len(batches) - 1:
                merged = try_merge_batches(batches[i], batches[i+1])
                if merged:
                    merged_list.append(merged)
                    i += 2
                    changed = True
                    continue
            merged_list.append(batches[i])
            i += 1
        batches = merged_list

    # Enforce hard cap by truncation if needed
    if len(batches) > max_batches_allowed:
        to_run = batches[:max_batches_allowed]
        deferred = batches[max_batches_allowed:]
        deferred_chunks = sum(len(b[0]) for b in deferred)
        return to_run, deferred_chunks

    return batches, 0

# ----------------- Processing helper (rate-limit retry + parse-error split) -----------------
def process_items(items: List[Dict[str, Any]]) -> Tuple[int, int, float, Dict[str, Dict[str, float]], Dict[str, int]]:
    """
    Processes a list of items (chunks). Implements:
    - Rate-limit errors (429): retry the same batch indefinitely with backoff and classification (rpm vs quota).
    - JSON parse errors: split strategy, tagged with attempt_id.
    - All logs timestamped; LLM attempts and related logs tagged with attempt_id.

    Returns:
    - num_chunks_processed
    - num_triples_stored
    - gemini_duration_total
    - per_chunk_stats: {chunk_id: {"gemini": x, "embed": y, "neo4j": z, "triples": n}}
    - extra_counters: {"final_batches": b, "rate_limit_retries": r, "json_parse_errors": j, "llm_calls": c}
    """
    if not items:
        return 0, 0, 0.0, {}, {"final_batches": 0, "rate_limit_retries": 0, "json_parse_errors": 0, "llm_calls": 0}

    # If no budget remains for even a single attempt, skip
    if ENFORCE_API_BUDGET and not API_BUDGET.will_allow("llm", 1):
        log_warn("Stopping: API budget for LLM calls would be exceeded.", attempt_id=None)
        return 0, 0, 0.0, {}, {"final_batches": 0, "rate_limit_retries": 0, "json_parse_errors": 0, "llm_calls": 0}

    rate_limit_retries = 0
    json_parse_errors = 0
    llm_calls = 0

    # Single item case
    if len(items) == 1:
        item = items[0]
        meta = item["meta"]
        text = item["text"]
        chunk_id = meta.get("chunk_id")
        per_chunk: Dict[str, Dict[str, float]] = {}
        backoff = random.uniform(2.0, 7.0)  # seconds, exponential for rate-limit
        single_parse_err_count = 0
        last_attempt_id: Optional[int] = None

        while True:
            if ENFORCE_API_BUDGET and not API_BUDGET.will_allow("llm", 1):
                log_warn(f"Stopping (budget exhausted) before LLM call for single chunk {chunk_id}", attempt_id=last_attempt_id)
                return 0, 0, 0.0, {}, {"final_batches": 0, "rate_limit_retries": rate_limit_retries, "json_parse_errors": json_parse_errors, "llm_calls": llm_calls}

            prompt = build_single_prompt(meta, text)
            llm_calls += 1
            try:
                triples, gemini_dur, attempt_id = extract_triples_from_chunk(text, meta, prompt_override=prompt)
                last_attempt_id = attempt_id
                written, emb_dur, neo4j_dur = write_triples_for_chunk(triples)
                per_chunk[chunk_id] = {"gemini": gemini_dur, "embed": emb_dur, "neo4j": neo4j_dur, "triples": written}
                return 1, written, gemini_dur, per_chunk, {"final_batches": 1, "rate_limit_retries": rate_limit_retries, "json_parse_errors": json_parse_errors, "llm_calls": llm_calls}
            except RateLimitError as rle:
                last_attempt_id = rle.attempt_id
                rate_limit_retries += 1
                sleep_for = min(backoff, 60.0)
                log_warn(f"Rate-limit ({rle.category}) on single chunk {chunk_id}. Reason: {rle.reason}. Retrying after {sleep_for:.1f}s (retry #{rate_limit_retries}).", attempt_id=last_attempt_id)
                time.sleep(sleep_for)
                backoff = min(backoff * 2.0, 60.0)
            except JsonParseError as e:
                last_attempt_id = e.attempt_id
                json_parse_errors += 1
                single_parse_err_count += 1
                if single_parse_err_count >= SINGLE_PARSE_SPLIT_AFTER:
                    left_text, right_text = split_text_in_two(text)
                    if not right_text:
                        log_warn(f"JSON parse error on single chunk {chunk_id}, could not split further. Continuing retries. Detail: {e}", attempt_id=last_attempt_id)
                        time.sleep(1.0)
                        continue
                    left_meta = dict(meta); left_meta["chunk_id"] = f"{chunk_id}::part1"
                    right_meta = dict(meta); right_meta["chunk_id"] = f"{chunk_id}::part2"
                    log_warn(f"JSON parse error repeated {single_parse_err_count} on {chunk_id}. Splitting into 2 parts: {len(left_text)} + {len(right_text)} chars.", attempt_id=last_attempt_id)
                    l_chunks, l_triples, l_gemini, l_stats, l_extra = process_items([{"text": left_text, "meta": left_meta}])
                    r_chunks, r_triples, r_gemini, r_stats, r_extra = process_items([{"text": right_text, "meta": right_meta}])

                    per_chunk_agg: Dict[str, Dict[str, float]] = {}
                    per_chunk_agg.update(l_stats)
                    per_chunk_agg.update(r_stats)

                    return (
                        l_chunks + r_chunks,
                        l_triples + r_triples,
                        l_gemini + r_gemini,
                        per_chunk_agg,
                        {
                            "final_batches": l_extra.get("final_batches", 0) + r_extra.get("final_batches", 0),
                            "rate_limit_retries": rate_limit_retries + l_extra.get("rate_limit_retries", 0) + r_extra.get("rate_limit_retries", 0),
                            "json_parse_errors": json_parse_errors + l_extra.get("json_parse_errors", 0) + r_extra.get("json_parse_errors", 0),
                            "llm_calls": llm_calls + l_extra.get("llm_calls", 0) + r_extra.get("llm_calls", 0),
                        }
                    )
                else:
                    log_warn(f"JSON parse error on single chunk {chunk_id}. Retrying. Detail: {e}", attempt_id=last_attempt_id)
                    time.sleep(1.0)
            except LlmCallError as e:
                last_attempt_id = e.attempt_id
                json_parse_errors += 1
                log_warn(f"Runtime error on single chunk {chunk_id}. Treating as parse error and retrying. Detail: {e}", attempt_id=last_attempt_id)
                time.sleep(1.0)
            except RuntimeError as e:
                # Might be budget exhaustion or other runtime errors without attempt_id
                if "API budget would be exceeded" in str(e):
                    log_warn(f"Budget exhausted while processing single chunk {chunk_id}", attempt_id=last_attempt_id)
                    return 0, 0, 0.0, {}, {"final_batches": 0, "rate_limit_retries": rate_limit_retries, "json_parse_errors": json_parse_errors, "llm_calls": llm_calls}
                json_parse_errors += 1
                log_warn(f"Runtime error on single chunk {chunk_id}. Treating as parse error and retrying. Detail: {e}", attempt_id=last_attempt_id)
                time.sleep(1.0)
            except Exception as e:
                json_parse_errors += 1
                log_warn(f"Unexpected error on single chunk {chunk_id}. Treating as parse error and retrying. Detail: {e}", attempt_id=last_attempt_id)
                time.sleep(1.0)

    # Batch case
    else:
        per_chunk: Dict[str, Dict[str, float]] = {}
        backoff = random.uniform(2.0, 7.0)  # seconds, exponential for rate-limit
        last_attempt_id: Optional[int] = None

        while True:
            if ENFORCE_API_BUDGET and not API_BUDGET.will_allow("llm", 1):
                log_warn("Stopping: API budget for LLM calls would be exceeded (batch).", attempt_id=last_attempt_id)
                return 0, 0, 0.0, {}, {"final_batches": 0, "rate_limit_retries": 0, "json_parse_errors": 0, "llm_calls": 0}

            prompt = build_batch_prompt(items)
            llm_calls += 1
            try:
                results_map, gemini_dur, attempt_id = extract_triples_from_chunks_batch(items, prompt)
                last_attempt_id = attempt_id
                # Success: write each chunk's triples
                num_chunks_processed = 0
                num_triples_stored = 0
                for it in items:
                    cid = it["meta"]["chunk_id"]
                    triples = results_map.get(cid, [])
                    written, emb_dur, neo4j_dur = write_triples_for_chunk(triples)
                    per_chunk[cid] = {"gemini": gemini_dur / max(1, len(items)), "embed": emb_dur, "neo4j": neo4j_dur, "triples": written}
                    num_chunks_processed += 1
                    num_triples_stored += written

                return num_chunks_processed, num_triples_stored, gemini_dur, per_chunk, {
                    "final_batches": 1,
                    "rate_limit_retries": rate_limit_retries,
                    "json_parse_errors": json_parse_errors,
                    "llm_calls": llm_calls
                }

            except RateLimitError as rle:
                last_attempt_id = rle.attempt_id
                rate_limit_retries += 1
                sleep_for = min(backoff, 60.0)
                log_warn(f"Rate-limit ({rle.category}) on batch (size={len(items)}). Reason: {rle.reason}. Retrying after {sleep_for:.1f}s (retry #{rate_limit_retries}).", attempt_id=last_attempt_id)
                time.sleep(sleep_for)
                backoff = min(backoff * 2.0, 60.0)
                continue
            except JsonParseError as e:
                last_attempt_id = e.attempt_id
                json_parse_errors += 1
                # Split the batch into two halves and process recursively
                mid = max(1, len(items) // 2)
                left = items[:mid]
                right = items[mid:]
                log_warn(f"JSON parse error on batch (size={len(items)}). Splitting into {len(left)} + {len(right)}. Detail: {e}", attempt_id=last_attempt_id)
                l_chunks, l_triples, l_gemini, l_stats, l_extra = process_items(left)
                r_chunks, r_triples, r_gemini, r_stats, r_extra = process_items(right)

                # Aggregate
                per_chunk_agg = {}
                per_chunk_agg.update(l_stats)
                per_chunk_agg.update(r_stats)

                return (l_chunks + r_chunks,
                        l_triples + r_triples,
                        l_gemini + r_gemini,
                        per_chunk_agg,
                        {
                            "final_batches": l_extra.get("final_batches", 0) + r_extra.get("final_batches", 0),
                            "rate_limit_retries": rate_limit_retries + l_extra.get("rate_limit_retries", 0) + r_extra.get("rate_limit_retries", 0),
                            "json_parse_errors": json_parse_errors + l_extra.get("json_parse_errors", 0) + r_extra.get("json_parse_errors", 0),
                            "llm_calls": llm_calls + l_extra.get("llm_calls", 0) + r_extra.get("llm_calls", 0),
                        })
            except LlmCallError as e:
                last_attempt_id = e.attempt_id
                json_parse_errors += 1
                # Treat unknown runtime errors as parse errors triggering split
                mid = max(1, len(items) // 2)
                left = items[:mid]
                right = items[mid:]
                log_warn(f"Runtime error on batch (size={len(items)}). Splitting into {len(left)} + {len(right)}. Detail: {e}", attempt_id=last_attempt_id)
                l_chunks, l_triples, l_gemini, l_stats, l_extra = process_items(left)
                r_chunks, r_triples, r_gemini, r_stats, r_extra = process_items(right)

                per_chunk_agg = {}
                per_chunk_agg.update(l_stats)
                per_chunk_agg.update(r_stats)

                return (l_chunks + r_chunks,
                        l_triples + r_triples,
                        l_gemini + r_gemini,
                        per_chunk_agg,
                        {
                            "final_batches": l_extra.get("final_batches", 0) + r_extra.get("final_batches", 0),
                            "rate_limit_retries": rate_limit_retries + l_extra.get("rate_limit_retries", 0) + r_extra.get("rate_limit_retries", 0),
                            "json_parse_errors": json_parse_errors + l_extra.get("json_parse_errors", 0) + r_extra.get("json_parse_errors", 0),
                            "llm_calls": llm_calls + l_extra.get("llm_calls", 0) + r_extra.get("llm_calls", 0),
                        })
            except RuntimeError as e:
                # Budget exhausted or other runtime error (no attempt id)
                if "API budget would be exceeded" in str(e):
                    log_warn("Budget exhausted during batch processing.", attempt_id=last_attempt_id)
                    return 0, 0, 0.0, {}, {"final_batches": 0, "rate_limit_retries": rate_limit_retries, "json_parse_errors": json_parse_errors, "llm_calls": llm_calls}
                json_parse_errors += 1
                mid = max(1, len(items) // 2)
                left = items[:mid]
                right = items[mid:]
                log_warn(f"Runtime error on batch (size={len(items)}). Splitting into {len(left)} + {len(right)}. Detail: {e}", attempt_id=last_attempt_id)
                l_chunks, l_triples, l_gemini, l_stats, l_extra = process_items(left)
                r_chunks, r_triples, r_gemini, r_stats, r_extra = process_items(right)

                per_chunk_agg = {}
                per_chunk_agg.update(l_stats)
                per_chunk_agg.update(r_stats)

                return (l_chunks + r_chunks,
                        l_triples + r_triples,
                        l_gemini + r_gemini,
                        per_chunk_agg,
                        {
                            "final_batches": l_extra.get("final_batches", 0) + r_extra.get("final_batches", 0),
                            "rate_limit_retries": rate_limit_retries + l_extra.get("rate_limit_retries", 0) + r_extra.get("rate_limit_retries", 0),
                            "json_parse_errors": json_parse_errors + l_extra.get("json_parse_errors", 0) + r_extra.get("json_parse_errors", 0),
                            "llm_calls": llm_calls + l_extra.get("llm_calls", 0) + r_extra.get("llm_calls", 0),
                        })
            except Exception as e:
                json_parse_errors += 1
                mid = max(1, len(items) // 2)
                left = items[:mid]
                right = items[mid:]
                log_warn(f"Unexpected error on batch (size={len(items)}). Splitting into {len(left)} + {len(right)}. Detail: {e}", attempt_id=last_attempt_id)
                l_chunks, l_triples, l_gemini, l_stats, l_extra = process_items(left)
                r_chunks, r_triples, r_gemini, r_stats, r_extra = process_items(right)

                per_chunk_agg = {}
                per_chunk_agg.update(l_stats)
                per_chunk_agg.update(r_stats)

                return (l_chunks + r_chunks,
                        l_triples + r_triples,
                        l_gemini + r_gemini,
                        per_chunk_agg,
                        {
                            "final_batches": l_extra.get("final_batches", 0) + r_extra.get("final_batches", 0),
                            "rate_limit_retries": rate_limit_retries + l_extra.get("rate_limit_retries", 0) + r_extra.get("rate_limit_retries", 0),
                            "json_parse_errors": json_parse_errors + l_extra.get("json_parse_errors", 0) + r_extra.get("json_parse_errors", 0),
                            "llm_calls": llm_calls + l_extra.get("llm_calls", 0) + r_extra.get("llm_calls", 0),
                        })

# ----------------- Main pipeline (parallel, budget-capped batching) -----------------
def run_kg_pipeline_over_folder(
    dir_path: Path,
    max_files: Optional[int] = None,
    max_chunks_per_file: Optional[int] = None
):
    pkls = list_pickles(dir_path)
    if max_files is not None:
        pkls = pkls[:max_files]

    total_chunks_planned = 0
    for p in pkls:
        try:
            with open(p, "rb") as f:
                chunks = pickle.load(f)
            n = len(chunks)
            total_chunks_planned += n if max_chunks_per_file is None else min(n, max_chunks_per_file)
        except Exception:
            continue

    log_info(f"Found {len(pkls)} pickle files in {dir_path} (skipping: {', '.join(SKIP_FILES) or 'none'})")
    log_info(f"Total chunks planned: {total_chunks_planned}")
    if ENFORCE_API_BUDGET:
        allowed_calls_remaining = max(0, API_BUDGET.total - API_BUDGET.used)
        log_info(f"API budget: enforce={API_BUDGET.enforce}, total={API_BUDGET.total}, used={API_BUDGET.used}, allowed_calls_remaining={allowed_calls_remaining}")
    else:
        log_info(f"API budget: enforce={API_BUDGET.enforce} (unlimited LLM calls)")
    log_info(f"LLM rate limit: {LLM_MAX_CALLS_PER_MIN} calls/minute")
    log_info(f"Greedy batching target: <= {PRACTICAL_MAX_ITEMS_PER_BATCH} items, est tokens <= {PROMPT_TOKEN_LIMIT}")
    log_info(f"Parallel workers: {INDEX_WORKERS}")
    if STAGGER_WORKER_SECONDS > 0:
        log_info(f"Worker ramp-up enabled: start 1 worker, add 1 every {STAGGER_WORKER_SECONDS:.3f}s (up to {INDEX_WORKERS}).")
    else:
        log_info("Worker ramp-up disabled (all workers may start immediately).")

    # Build 'raw_items_all' list across all files
    raw_items_all: List[Dict[str, Any]] = []
    for file_idx, pkl in enumerate(pkls, 1):
        log_info(f"[{file_idx}/{len(pkls)}] Scanning {pkl.name}")
        try:
            chunks = load_chunks_from_file(pkl)
        except Exception as e:
            log_warn(f"Failed to load {pkl.name}: {e}")
            continue

        for idx, ch in enumerate(chunks):
            if max_chunks_per_file is not None and idx >= max_chunks_per_file:
                break
            meta_source = getattr(ch, "metadata", {}) if hasattr(ch, "metadata") else {}
            meta = {
                "document_id": meta_source.get("document_id"),
                "chunk_id": meta_source.get("chunk_id"),
                "uu_number": meta_source.get("uu_number"),
                "pages": meta_source.get("pages"),
            }
            text = getattr(ch, "page_content", str(ch))
            if not meta.get("chunk_id"):
                meta["chunk_id"] = f"{pkl.stem}_chunk_{idx}"
            raw_items_all.append({"chunk_id": meta["chunk_id"], "text": text, "meta": meta})

    if not raw_items_all:
        log_info("No chunks found. Exiting.")
        return

    # Determine max number of batches allowed by budget (at most remaining LLM calls)
    max_batches_allowed = None
    if ENFORCE_API_BUDGET:
        max_batches_allowed = max(0, API_BUDGET.total - API_BUDGET.used)

    # Pack into batches and enforce a hard cap by budget
    all_batches_capped, deferred_chunks = pack_batches_with_cap(raw_items_all, max_batches_allowed)
    total_batches_planned = len(all_batches_capped)
    log_info(f"Packed {len(raw_items_all)} chunks into {total_batches_planned} batch(es) (budget-capped).")
    for i, (batch_items, est_tokens) in enumerate(all_batches_capped, 1):
        log_info(f"  • Batch {i}: chunks={len(batch_items)}, est_tokens~{est_tokens}")
    if deferred_chunks > 0:
        log_warn(f"Deferring {deferred_chunks} chunk(s) to future runs due to API budget cap of {max_batches_allowed} batch(es).")

    # Stats
    total_triples_stored = 0
    total_chunks_done = 0

    total_gemini_duration = 0.0
    total_embedding_duration = 0.0
    total_neo4j_duration = 0.0

    per_batch_component_sums: Dict[int, float] = {}
    per_batch_wall_times: Dict[int, float] = {}
    per_batch_sizes: Dict[int, int] = {}

    # Aggregated counters
    global_final_batches = 0
    global_rate_limit_retries = 0
    global_json_parse_errors = 0
    global_llm_call_attempts = 0

    overall_start = time.time()

    def batch_desc(idx: int, size: int, tokens: int) -> str:
        return f"[Batch {idx+1}/{total_batches_planned}] size={size}, est_tokens~{tokens}"

    def remaining_budget() -> int:
        if not ENFORCE_API_BUDGET:
            return 1_000_000_000
        return max(0, API_BUDGET.total - API_BUDGET.used)

    # Staggered worker start scheduling
    with ThreadPoolExecutor(max_workers=INDEX_WORKERS) as executor:
        ramp_start = time.time()

        def allowed_workers_now() -> int:
            if STAGGER_WORKER_SECONDS <= 0:
                return INDEX_WORKERS
            elapsed = time.time() - ramp_start
            allowed = 1 + int(elapsed // STAGGER_WORKER_SECONDS)
            return min(INDEX_WORKERS, max(1, allowed))

        futures_set = set()
        futures_meta: Dict[Any, Tuple[int, int, List[Dict[str, Any]], float]] = {}
        next_idx = 0

        # Utility to compute time until next ramp step
        def time_until_next_ramp() -> float:
            if STAGGER_WORKER_SECONDS <= 0:
                return 1.0
            elapsed = time.time() - ramp_start
            steps_completed = int(elapsed // STAGGER_WORKER_SECONDS)
            next_step_time = (steps_completed + 1) * STAGGER_WORKER_SECONDS
            return max(0.0, next_step_time - elapsed)

        # Main scheduling loop: ramp up workers and process results
        while True:
            # Determine target concurrency considering ramp-up and (optionally) budget
            target = allowed_workers_now()
            if ENFORCE_API_BUDGET:
                target = min(target, remaining_budget())

            # Fill up to target
            while next_idx < total_batches_planned and len(futures_set) < target:
                items_batch, est_tokens = all_batches_capped[next_idx]
                submit_time = time.time()
                fut = executor.submit(process_items, items_batch)
                futures_set.add(fut)
                futures_meta[fut] = (next_idx, est_tokens, items_batch, submit_time)
                next_idx += 1

            # Exit if nothing running and nothing left to submit
            if not futures_set and next_idx >= total_batches_planned:
                break

            # Wait for either a future to complete or the next ramp increment
            timeout = min(1.0, time_until_next_ramp())
            done, _ = wait(futures_set, timeout=timeout, return_when=FIRST_COMPLETED)

            # Process completed futures (if any)
            for fut in list(done):
                b_idx, est_tokens, items_batch, submit_time = futures_meta.pop(fut)
                futures_set.remove(fut)

                start_label = batch_desc(b_idx, len(items_batch), est_tokens)
                try:
                    chunks_processed, triples_stored, gemini_dur, per_chunk_stats, extra = fut.result()
                except Exception as e:
                    log_error(f"{start_label} failed with error: {e}")
                    continue

                end_time = time.time()
                wall_time = end_time - submit_time

                embed_total = sum(stats['embed'] for stats in per_chunk_stats.values())
                neo4j_total = sum(stats['neo4j'] for stats in per_chunk_stats.values())
                comp_sum = gemini_dur + embed_total + neo4j_total

                total_chunks_done += chunks_processed
                total_triples_stored += triples_stored
                total_gemini_duration += gemini_dur
                total_embedding_duration += embed_total
                total_neo4j_duration += neo4j_total

                # Aggregate new counters
                final_batches = extra.get("final_batches", 0)
                rate_limit_retries = extra.get("rate_limit_retries", 0)
                json_parse_errors = extra.get("json_parse_errors", 0)
                llm_calls = extra.get("llm_calls", 0)

                global_final_batches += final_batches
                global_rate_limit_retries += rate_limit_retries
                global_json_parse_errors += json_parse_errors
                global_llm_call_attempts += llm_calls

                per_batch_component_sums[b_idx] = comp_sum
                per_batch_wall_times[b_idx] = wall_time
                per_batch_sizes[b_idx] = len(items_batch)

                if per_chunk_stats:
                    for it in items_batch:
                        cid = it["meta"]["chunk_id"]
                        stats = per_chunk_stats.get(cid)
                        if stats:
                            log_info(f"      · Chunk {cid}: Gemini={stats['gemini']:.2f}s | Embedding={stats['embed']:.2f}s | Neo4j={stats['neo4j']:.2f}s | Triples={int(stats['triples'])}")

                overhead = wall_time - comp_sum
                if chunks_processed == 0 and comp_sum == 0.0:
                    log_info(f"{start_label} skipped (budget exhausted). Batch wall time: {wall_time:.2f}s")
                else:
                    log_info(f"    ✓ {start_label}")
                    log_info(f"        - KG extraction (LLM): {gemini_dur:.2f}s")
                    log_info(f"        - Embedding total:     {embed_total:.2f}s")
                    log_info(f"        - Neo4j insert total:  {neo4j_total:.2f}s")
                    log_info(f"        - Component sum:       {comp_sum:.2f}s")
                    log_info(f"        - Batch wall time:     {wall_time:.2f}s")
                    log_info(f"        - Overhead (wall - components): {overhead:+.2f}s")
                    log_info(f"        - Chunks processed: {chunks_processed} | Triples stored: {triples_stored}")
                    log_info(f"        - Attempts: llm_calls={llm_calls}, rate_limit_retries={rate_limit_retries}, json_parse_errors={json_parse_errors}, final_batches={final_batches}")

            # Loop re-checks target and submits more if ramp allows

    total_time_real = time.time() - overall_start
    total_llm_calls_used = API_BUDGET.resource_usage.get("llm", 0)

    sequential_estimate = sum(per_batch_component_sums.values())
    speedup = (sequential_estimate / total_time_real) if total_time_real > 0 else float('inf')

    log_info("Summary")
    log_info(f"- Batches planned (initial, capped): {total_batches_planned}")
    log_info(f"- Batches processed after JSON-split: {global_final_batches} (extra from splits: {max(0, global_final_batches - total_batches_planned)})")
    log_info(f"- Chunks processed: {total_chunks_done}/{total_chunks_planned}")
    log_info(f"- Triples stored: {total_triples_stored}")
    log_info(f"- LLM calls attempted (total): {global_llm_call_attempts}")
    log_info(f"- Rate-limit retries (429): {global_rate_limit_retries}")
    log_info(f"- JSON parse errors encountered: {global_json_parse_errors}")
    log_info(f"- LLM calls used (budget counter): {total_llm_calls_used}{' (budget enforced)' if ENFORCE_API_BUDGET else ''}")
    log_info(f"- Total Gemini time (sum of batch LLM times): {total_gemini_duration:.2f}s")
    log_info(f"- Total Embedding time (sum): {total_embedding_duration:.2f}s")
    log_info(f"- Total Neo4j time (sum): {total_neo4j_duration:.2f}s")
    log_info(f"- Total real wall time: {total_time_real:.2f}s")
    log_info(f"- Sequential time estimate (components sum): {sequential_estimate:.2f}s")
    log_info(f"- Speedup vs sequential: {speedup:.2f}× faster")
    if ENFORCE_API_BUDGET:
        log_info(f"- API used: {API_BUDGET.used}/{API_BUDGET.total} (LLM calls and {'embeddings' if COUNT_EMBEDDINGS_IN_BUDGET else 'no embeddings'} counted)")

if __name__ == "__main__":
    run_kg_pipeline_over_folder(
        LANGCHAIN_DIR,
        max_files=40,
        max_chunks_per_file=None
    )