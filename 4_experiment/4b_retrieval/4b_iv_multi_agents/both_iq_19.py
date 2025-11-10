#!/usr/bin/env python3
# multi_agent.py
# Combined RAG orchestrator with agentic supervisor (sequential IQ flow):
# - Runs GraphRAG (single-pass, Agents 1 & 2) and Naive RAG (Answerer-only)
# - Aggregator agent synthesizes the per-step answer from both approaches
# - Replaces Subgoal+Final Aggregator with:
#     1) Intermediate Question (IQ) Generator (sequential, dependent steps)
#     2) Query Modifier (completes/enriches each next IQ based on previous answers)
# - The final IQ's answer becomes the overall final answer
# - Comprehensive logging with timestamps, PID, TID, IQ context; rate limiting is thread-safe

import os, sys, time, json, math, pickle, re, random, threading
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, Set
from threading import Lock
from collections import deque

from dotenv import load_dotenv
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
from neo4j import GraphDatabase

# ----------------- Load .env (parent directory of this file) -----------------
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent.parent.parent / ".env")

# ----------------- Config -----------------
# Credentials and endpoints
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
NEO4J_URI  = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASS", "password")

# Neo4j retry/timeout
NEO4J_TX_TIMEOUT_S = float(os.getenv("NEO4J_TX_TIMEOUT_S", "60"))
NEO4J_MAX_ATTEMPTS = int(os.getenv("NEO4J_MAX_ATTEMPTS", "1000000000"))

# Models
GEN_MODEL   = os.getenv("GEN_MODEL", "models/gemini-2.5-flash")
EMBED_MODEL = os.getenv("EMBED_MODEL", "models/text-embedding-004")

# Dataset folder for original chunk pickles (GraphRAG)
DEFAULT_LANGCHAIN_DIR = (Path(__file__).resolve().parent / "../../../dataset/3_indexing/3a_langchain_results/").resolve()
LANGCHAIN_DIR = Path(os.getenv("LANGCHAIN_DIR") or str(DEFAULT_LANGCHAIN_DIR))
SKIP_FILES = {"all_langchain_documents.pkl"}

# ---------- GraphRAG parameters ----------
GR_ENTITY_MATCH_TOP_K = 15
GR_ENTITY_SUBGRAPH_HOPS = 5
GR_ENTITY_SUBGRAPH_PER_HOP_LIMIT = 2000
GR_SUBGRAPH_TRIPLES_TOP_K = 30
GR_QUERY_TRIPLE_MATCH_TOP_K_PER = 20
GR_MAX_TRIPLES_FINAL = 60
GR_MAX_CHUNKS_FINAL = 40
GR_CHUNK_RERANK_CAND_LIMIT = 200
GR_ANSWER_MAX_TOKENS = int(os.getenv("ANSWER_MAX_TOKENS", "4096"))

# ---------- Naive RAG parameters ----------
NV_TOP_K_CHUNKS = int(os.getenv("TOP_K_CHUNKS", "40"))
NV_MAX_CHUNKS_FINAL = int(os.getenv("MAX_CHUNKS_FINAL", "40"))
NV_CHUNK_TEXT_CLAMP = int(os.getenv("CHUNK_TEXT_CLAMP", "1000000"))

# ---------- Shared Agent loop params ----------
MAX_ITERS = 1  # single pass
LLM_CALLS_PER_MINUTE = int(os.getenv("LLM_CALLS_PER_MINUTE", "13"))
EMBEDDING_CALLS_PER_MINUTE = int(os.getenv("EMBEDDING_CALLS_PER_MINUTE", "0"))  # 0 disables embedding throttling

# ---------- IQ Orchestrator parameters ----------
IQ_MAX_N = int(os.getenv("IQ_MAX_N", "5"))  # maximum sequential IQs
IQ_ANSWER_SNIPPET_CLAMP = int(os.getenv("IQ_ANSWER_SNIPPET_CLAMP", "20000000"))

# ---------- JSON-output (text-mode) parameters ----------
JSON_MAX_TOKENS = int(os.getenv("JSON_MAX_TOKENS", "4096"))
JSON_PARSER_STRICTNESS = os.getenv("JSON_PARSER_STRICTNESS", "relaxed").lower()  # "relaxed" or "strict"
LOG_JSON_RAW_PREVIEW_CHARS = int(os.getenv("LOG_JSON_RAW_PREVIEW_CHARS", "200"))

# ----------------- Initialize SDKs -----------------
genai.configure(api_key=GOOGLE_API_KEY)
gen_model = genai.GenerativeModel(GEN_MODEL)
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

# ----------------- Logger -----------------
_THREAD_LOCAL = threading.local()

def _now_ts() -> str:
    t = time.time()
    base = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t))
    ms = int((t % 1) * 1000)
    return f"{base}.{ms:03d}"

def _pid() -> int:
    try:
        return os.getpid()
    except Exception:
        return -1

def _tid() -> int:
    try:
        return threading.get_ident()
    except Exception:
        return -1

def set_log_context(prefix: Optional[str]):
    setattr(_THREAD_LOCAL, "prefix", prefix or "")

def _get_log_context() -> str:
    s = getattr(_THREAD_LOCAL, "prefix", "")
    return f" [{s}]" if s else ""

def _prefix(level: str = "INFO") -> str:
    return f"[{_now_ts()}] [{level}] [pid={_pid()} tid={_tid()}]{_get_log_context()}"

class FileLogger:
    def __init__(self, file_path: Path, also_console: bool = True):
        self.file_path = file_path
        self.also_console = also_console
        self._fh = open(file_path, "w", encoding="utf-8")
        self._lock = Lock()

    def log(self, msg: Any = ""):
        if not isinstance(msg, str):
            try:
                msg = json.dumps(msg, ensure_ascii=False, default=str)
            except Exception:
                msg = str(msg)
        lines = (msg or "").splitlines() or [""]
        prefixed = [f"{_prefix()} {line}" for line in lines]
        out = "\n".join(prefixed) + "\n"
        with self._lock:
            self._fh.write(out)
            self._fh.flush()
            if self.also_console:
                print(out, end="", flush=True)

    def close(self):
        try:
            with self._lock:
                self._fh.flush()
                self._fh.close()
        except Exception:
            pass

_LOGGER: Optional[FileLogger] = None

def log(msg: Any = "", level: str = "INFO"):
    global _LOGGER
    if _LOGGER is None:
        lines = (str(msg) if isinstance(msg, str) else json.dumps(msg, ensure_ascii=False, default=str)).splitlines() or [""]
        print("\n".join(f"{_prefix(level)} {line}" for line in lines), flush=True)
        return
    if isinstance(msg, str):
        lines = msg.splitlines() or [""]
        for line in lines:
            _LOGGER.log(f"[{level}] {line}")
    else:
        _LOGGER.log(f"[{level}] {msg}")

def make_timestamp_name() -> str:
    t = time.time()
    base = time.strftime("%Y%m%d-%H%M%S", time.localtime(t))
    ms = int((t % 1) * 1000)
    return f"{base}-{ms:03d}"

# ----------------- Utilities -----------------
def now_ms() -> float:
    return time.time()

def dur_ms(start: float) -> float:
    return (time.time() - start) * 1000.0

def _norm_id(x) -> str:
    return str(x).strip() if x is not None else ""

def estimate_tokens_for_text(text: str) -> int:
    return max(1, int(len(text) / 4))

def _as_float_list(vec) -> List[float]:
    if vec is None:
        return []
    try:
        if hasattr(vec, "tolist"):
            vec = vec.tolist()
    except Exception:
        pass
    try:
        return [float(x) for x in list(vec)]
    except Exception:
        try:
            return [float(vec)]
        except Exception:
            return []

def cos_sim(a: List[float], b: List[float]) -> float:
    a = _as_float_list(a); b = _as_float_list(b)
    dot = sum(x*y for x,y in zip(a,b))
    na = math.sqrt(sum(x*x for x in a)); nb = math.sqrt(sum(y*y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)

# --- Rate limiting and retries (thread-safe) ---
class RateLimiter:
    def __init__(self, calls_per_minute: int, name: str = "LLM"):
        self.calls_per_minute = max(0, int(calls_per_minute))
        self.name = name
        self.window = deque()
        self.window_seconds = 60.0
        self._lock = Lock()

    def wait_for_slot(self):
        if self.calls_per_minute <= 0:
            return
        while True:
            with self._lock:
                now = time.monotonic()
                while self.window and (now - self.window[0]) >= self.window_seconds:
                    self.window.popleft()
                if len(self.window) < self.calls_per_minute:
                    self.window.append(now)
                    return
                sleep_time = self.window_seconds - (now - self.window[0])
            if sleep_time > 0:
                log(f"[RateLimit:{self.name}] Sleeping {sleep_time:.2f}s to respect {self.name}_CALLS_PER_MINUTE={self.calls_per_minute}", level="INFO")
                time.sleep(sleep_time)
            else:
                time.sleep(0.01)

_LLM_RATE_LIMITER = RateLimiter(LLM_CALLS_PER_MINUTE, "LLM")
_EMBED_RATE_LIMITER = RateLimiter(EMBEDDING_CALLS_PER_MINUTE, "EMBED")

def _rand_wait_seconds() -> float:
    return random.uniform(80.0, 120.0)

def _api_call_with_retry(func, *args, **kwargs):
    while True:
        try:
            if func.__name__ == "embed_content":
                _EMBED_RATE_LIMITER.wait_for_slot()
            else:
                _LLM_RATE_LIMITER.wait_for_slot()
        except Exception:
            pass
        try:
            return func(*args, **kwargs)
        except Exception as e:
            wait_s = _rand_wait_seconds()
            log(f"[Retry] API call failed: {e}. Retrying in {wait_s:.1f}s.", level="WARN")
            time.sleep(wait_s)

def embed_text(text: str) -> List[float]:
    t0 = now_ms()
    res = _api_call_with_retry(genai.embed_content, model=EMBED_MODEL, content=text)
    vec = None
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
        raise RuntimeError("Unexpected embedding response shape for embeddings")
    out = _as_float_list(vec)
    log(f"[Embed] text_len={len(text)} -> vec_len={len(out)} | {dur_ms(t0):.0f} ms", level="DEBUG")
    return out

# ----------------- Language detection -----------------
def detect_user_language(text: str) -> str:
    t = (text or "").lower()
    if re.search(r"\b(pasal|undang[- ]?undang|uu\s*\d|peraturan|menteri|ayat|bab|bagian|paragraf|ketentuan|sebagaimana|dimaksud)\b", t):
        return "id"
    if re.search(r"\b(article|act|law|regulation|minister|section|paragraph|chapter|pursuant|provided that)\b", t):
        return "en"
    id_tokens = {"yang","dan","atau","tidak","adalah","berdasarkan","sebagaimana","pada","dalam","dapat","harus","wajib","pasal","undang","peraturan","menteri","ayat","bab","bagian","paragraf","ketentuan","pengundangan","apabila","jika"}
    en_tokens = {"the","and","or","not","is","based","as","provided","pursuant","in","may","must","shall","article","act","law","regulation","minister","section","paragraph","chapter","whereas"}
    words = re.findall(r"[a-z]+", t)
    score_id = sum(1 for w in words if w in id_tokens)
    score_en = sum(1 for w in words if w in en_tokens)
    if score_id > score_en: return "id"
    if score_en > score_id: return "en"
    return "en"

# ----------------- Safe LLM helpers (text-only + strict JSON prompts) -----------------

STRICT_JSON_DIRECTIVE = (
    "Return ONLY a single valid JSON object as the entire response.\n"
    "- No markdown, no code fences, no explanations, no surrounding text.\n"
    "- Use double quotes for all keys and string values.\n"
    "- Do not include trailing commas or comments.\n"
    "- If a value is unknown, use an empty string, empty array, 0/0.0, false, or null as appropriate.\n"
    "- Ensure the JSON is parseable by json.loads in Python."
)

def get_finish_info(resp) -> Dict[str, Any]:
    info = {}
    try:
        cand = resp.candidates[0] if resp.candidates else None
        if cand:
            info["finish_reason"] = getattr(cand, "finish_reason", None)
            safety = []
            try:
                for sr in getattr(cand, "safety_ratings", []) or []:
                    safety.append({"category": getattr(sr, "category", None), "prob": getattr(sr, "probability", None)})
            except Exception:
                pass
            info["safety_ratings"] = safety
    except Exception:
        pass
    return info

def extract_text_from_response(resp) -> Optional[str]:
    try:
        text = resp.text
        if isinstance(text, str) and text.strip():
            return text.strip()
    except Exception:
        pass
    try:
        for cand in (resp.candidates or []):
            parts = getattr(cand, "content", None)
            if parts and getattr(parts, "parts", None):
                buf = []
                for p in parts.parts:
                    t = getattr(p, "text", None)
                    if isinstance(t, str): buf.append(t)
                if buf:
                    return "\n".join(buf).strip()
    except Exception:
        pass
    return None

def safe_generate_text(prompt: str, max_tokens: int, temperature: float = 0.2) -> str:
    cfg = GenerationConfig(temperature=temperature, max_output_tokens=max_tokens)
    t0 = now_ms()
    resp = _api_call_with_retry(gen_model.generate_content, prompt, generation_config=cfg)
    took = dur_ms(t0)
    text = extract_text_from_response(resp)
    if text is not None and text.strip():
        log(f"[LLM TEXT] call completed in {took:.0f} ms, len={len(text)}", level="DEBUG")
        return text.strip()
    info = get_finish_info(resp)
    log(f"[LLM text warning] No text returned. Diagnostics: {info}. Took={took:.0f} ms", level="WARN")
    return f"(Model returned no text. finish_info={info})"

# ---- JSON parsing helpers (from normal text) ----
def _strip_code_fences(s: str) -> str:
    fence = re.search(r"```(?:json)?\s*(.*?)```", s, flags=re.IGNORECASE | re.DOTALL)
    if fence:
        return fence.group(1).strip()
    return s.strip()

def _extract_balanced_json_object(s: str) -> Optional[str]:
    start = s.find("{")
    if start == -1:
        return None
    i = start
    depth = 0
    in_str = False
    esc = False
    while i < len(s):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return s[start:i+1]
        i += 1
    return None

def _cleanup_relaxed_json_text(s: str) -> str:
    s = re.sub(r",\s*(\})", r"\1", s)
    s = re.sub(r",\s*(\])", r"\1", s)
    return s

def try_parse_json_from_text(raw_text: str) -> Dict[str, Any]:
    if not isinstance(raw_text, str):
        return {}
    candidate = _strip_code_fences(raw_text)
    try:
        return json.loads(candidate)
    except Exception:
        pass
    balanced = _extract_balanced_json_object(candidate)
    if balanced:
        try:
            return json.loads(balanced)
        except Exception:
            if JSON_PARSER_STRICTNESS != "strict":
                try:
                    return json.loads(_cleanup_relaxed_json_text(balanced))
                except Exception:
                    pass
    f = candidate.find("{"); l = candidate.rfind("}")
    if 0 <= f < l:
        snippet = candidate[f:l+1]
        try:
            return json.loads(snippet)
        except Exception:
            if JSON_PARSER_STRICTNESS != "strict":
                try:
                    return json.loads(_cleanup_relaxed_json_text(snippet))
                except Exception:
                    pass
    if candidate != raw_text:
        try:
            return json.loads(raw_text)
        except Exception:
            if JSON_PARSER_STRICTNESS != "strict":
                try:
                    return json.loads(_cleanup_relaxed_json_text(raw_text))
                except Exception:
                    pass
    return {}

def _default_for_type(t: Any):
    if t == "string":
        return ""
    if t == "number":
        return 0.0
    if t == "integer":
        return 0
    if t == "boolean":
        return False
    if t == "array":
        return []
    if t == "object":
        return {}
    return None

def _ensure_required_fields(obj: Dict[str, Any], schema: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(obj, dict) or not isinstance(schema, dict):
        return obj if isinstance(obj, dict) else {}
    sch_type = schema.get("type")
    props = schema.get("properties", {})
    required = schema.get("required", [])
    if sch_type == "object" and isinstance(props, dict):
        for k in required:
            if k not in obj:
                prop = props.get(k, {})
                d = _default_for_type(prop.get("type"))
                obj[k] = d
        for k, prop in props.items():
            if k in obj:
                if prop.get("type") == "array" and not isinstance(obj[k], list):
                    obj[k] = [] if obj[k] is None else [obj[k]]
                if prop.get("type") == "object" and not isinstance(obj[k], dict):
                    try:
                        if isinstance(obj[k], str):
                            maybe = json.loads(obj[k])
                            if isinstance(maybe, dict):
                                obj[k] = maybe
                            else:
                                obj[k] = {}
                        else:
                            obj[k] = {}
                    except Exception:
                        obj[k] = {}
    return obj

def safe_generate_json(prompt: str, schema: Dict[str, Any], temp: float = 0.0) -> Dict[str, Any]:
    est = estimate_tokens_for_text(prompt)
    log("[LLM JSON-TEXT] Prompt (strict JSON mode) begins:", level="INFO")
    log(prompt, level="INFO")
    log(f"[LLM JSON-TEXT] Prompt size: {len(prompt)} chars, est_tokens≈{est}", level="INFO")

    raw = safe_generate_text(prompt, max_tokens=JSON_MAX_TOKENS, temperature=temp)
    preview = (raw[:LOG_JSON_RAW_PREVIEW_CHARS] + ("..." if len(raw) > LOG_JSON_RAW_PREVIEW_CHARS else ""))
    log(f"[LLM JSON-TEXT] Raw output preview ({min(len(raw), LOG_JSON_RAW_PREVIEW_CHARS)} chars): {preview!r}", level="DEBUG")

    parsed = try_parse_json_from_text(raw)
    if not parsed:
        log("[LLM JSON-TEXT] Parse failed; returning empty dict.", level="WARN")
        return {}

    normalized = _ensure_required_fields(parsed, schema)
    return normalized

# ----------------- Agent 1 / 1b Schemas (GraphRAG) -----------------
LEGAL_ENTITY_TYPES = [
    "UU", "PASAL", "AYAT", "INSTANSI", "ORANG", "ISTILAH", "SANKSI", "NOMINAL", "TANGGAL"
]
LEGAL_PREDICATES = [
    "mendefinisikan", "mengubah", "mencabut", "mulai_berlaku", "mewajibkan",
    "melarang", "memberikan_sanksi", "berlaku_untuk", "termuat_dalam",
    "mendelegasikan_kepada", "berjumlah", "berdurasi"
]

QUERY_SCHEMA = {
  "type": "object",
  "properties": {
    "entities": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "text": {"type": "string"},
          "type": {"type": "string"}
        },
        "required": ["text"]
      }
    },
    "predicates": {
      "type": "array",
      "items": {"type": "string"}
    }
  },
  "required": ["entities", "predicates"]
}

QUERY_TRIPLES_SCHEMA = {
    "type": "object",
    "properties": {
        "triples": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "subject": {
                        "type": "object",
                        "properties": {"text": {"type": "string"}, "type": {"type": "string"}},
                        "required": ["text"]
                    },
                    "predicate": {"type": "string"},
                    "object": {
                        "type": "object",
                        "properties": {"text": {"type": "string"}, "type": {"type": "string"}},
                        "required": ["text"]
                    }
                },
                "required": ["subject", "predicate", "object"]
            }
        }
    },
    "required": ["triples"]
}

def agent1_extract_entities_predicates(query: str) -> Dict[str, Any]:
    prompt = f"""
You are Agent 1. Task: extract the legal entities and predicates referenced or implied by the user's question.

Output format:
- JSON with:
  - "entities": array of objects with fields {{text, type(optional)}}
  - "predicates": array of strings (Indonesian, snake_case when applicable)
- If entity type is provided, it MUST be one of: {", ".join(LEGAL_ENTITY_TYPES)}.
- Predicates should ideally be one of: {", ".join(LEGAL_PREDICATES)}.

{STRICT_JSON_DIRECTIVE}

User question:
\"\"\"{query}\"\"\"
"""
    est = estimate_tokens_for_text(prompt)
    log("\n[Agent 1] Prompt:", level="INFO")
    log(prompt, level="INFO")
    log(f"[Agent 1] Prompt size: {len(prompt)} chars, est_tokens≈{est}", level="INFO")
    out = safe_generate_json(prompt, QUERY_SCHEMA, temp=0.0)
    if "entities" not in out: out["entities"] = []
    if "predicates" not in out: out["predicates"] = []
    log(f"[Agent 1] entities={[e.get('text') for e in out['entities']]}, predicates={out['predicates']}", level="INFO")
    return out

def agent1b_extract_query_triples(query: str) -> List[Dict[str, Any]]:
    prompt = f"""
You are Agent 1b. Task: extract explicit or implied triples from the user's question in the form:
subject — predicate — object.

Rules:
- Use short, literal subject/object texts as they appear in the question.
- Predicates should be concise (lowercase, snake_case if multiword).
- If type is unknown, leave it blank.
- Do not invent or speculate; extract only what is clearly suggested by the question.

Return JSON with a key "triples" as specified.

{STRICT_JSON_DIRECTIVE}

User question:
\"\"\"{query}\"\"\"
"""
    est = estimate_tokens_for_text(prompt)
    log("\n[Agent 1b] Prompt:", level="INFO")
    log(prompt, level="INFO")
    log(f"[Agent 1b] Prompt size: {len(prompt)} chars, est_tokens≈{est}", level="INFO")
    out = safe_generate_json(prompt, QUERY_TRIPLES_SCHEMA, temp=0.0) or {}
    triples = out.get("triples", [])
    clean: List[Dict[str, Any]] = []
    for t in triples or []:
        try:
            s = (t.get("subject") or {}).get("text","").strip()
            p = (t.get("predicate") or "").strip()
            o = (t.get("object")  or {}).get("text","").strip()
            if s and p and o:
                clean.append({
                    "subject":{"text":s,"type":(t.get("subject") or {}).get("type","").strip()},
                    "predicate":p,
                    "object":{"text":o,"type":(t.get("object") or {}).get("type","").strip()}
                })
        except Exception:
            pass
    formatted = [f"{x['subject']['text']} [{x['predicate']}] {x['object']['text']}" for x in clean]
    log(f"[Agent 1b] Extracted query triples: {formatted}", level="INFO")
    return clean

def query_triple_to_text(t: Dict[str, Any]) -> str:
    s = ((t.get("subject") or {}).get("text") or "").strip()
    p = (t.get("predicate") or "").strip()
    o = ((t.get("object") or {}).get("text") or "").strip()
    return f"{s} [{p}] {o}"

# ----------------- Neo4j vector and cypher helpers -----------------
_NEO4J_QUERY_SEQ = 0
_NEO4J_QUERY_LOCK = Lock()

def _next_query_id() -> int:
    global _NEO4J_QUERY_SEQ
    with _NEO4J_QUERY_LOCK:
        _NEO4J_QUERY_SEQ += 1
        return _NEO4J_QUERY_SEQ

def _summarize_params(params: Dict[str, Any]) -> str:
    parts = []
    for k, v in (params or {}).items():
        try:
            if k.lower() in ("q_emb", "embedding", "emb", "q"):
                if isinstance(v, (list, tuple)):
                    parts.append(f"{k}=list(len={len(v)})")
                else:
                    parts.append(f"{k}=vector")
            elif isinstance(v, list):
                parts.append(f"{k}=list(len={len(v)})")
            elif isinstance(v, str) and len(v) > 60:
                parts.append(f"{k}=str(len={len(v)})")
            else:
                parts.append(f"{k}={v}")
        except Exception:
            parts.append(f"{k}=<?>")
    return ", ".join(parts)

def run_cypher_with_retry(cypher: str, params: Dict[str, Any]) -> List[Any]:
    attempts = 0
    last_e: Optional[Exception] = None
    qid = _next_query_id()
    preview = " ".join((cypher or "").split())
    if len(preview) > 220:
        preview = preview[:220] + "..."
    param_summary = _summarize_params(params)
    while attempts < max(1, NEO4J_MAX_ATTEMPTS):
        attempts += 1
        t0 = now_ms()
        log(f"[Neo4j] Attempt {attempts}/{NEO4J_MAX_ATTEMPTS} | qid={qid} | timeout={NEO4J_TX_TIMEOUT_S:.1f}s | Cypher=\"{preview}\" | Params: {param_summary}", level="INFO")
        try:
            with driver.session() as session:
                res = session.run(cypher, **params, timeout=NEO4J_TX_TIMEOUT_S)
                records = list(res)
            took = dur_ms(t0)
            log(f"[Neo4j] Success | qid={qid} | rows={len(records)} | {took:.0f} ms", level="INFO")
            return records
        except Exception as e:
            took = dur_ms(t0)
            last_e = e
            if attempts >= NEO4J_MAX_ATTEMPTS:
                break
            wait_s = random.uniform(5.0, 15.0)
            log(f"[Neo4j] Failure | qid={qid} | attempt={attempts}/{NEO4J_MAX_ATTEMPTS} | {took:.0f} ms | error={e}. Retrying in {wait_s:.1f}s.", level="WARN")
            time.sleep(wait_s)
    raise RuntimeError(f"Neo4j query failed after {NEO4J_MAX_ATTEMPTS} attempts (qid={qid}): {last_e}")

def _vector_query_nodes(index_name: str, q_emb: List[float], k: int) -> List[Dict[str, Any]]:
    q_emb = _as_float_list(q_emb)
    cypher = """
    CALL db.index.vector.queryNodes($index_name, $k, $q_emb) YIELD node AS n, score
    RETURN n, score, elementId(n) AS elem_id
    ORDER BY score DESC
    LIMIT $k
    """
    res = run_cypher_with_retry(cypher, {"index_name": index_name, "k": k, "q_emb": q_emb})
    rows = []
    for r in res:
        n = r["n"]
        rows.append({
            "key": n.get("key"),
            "elem_id": r["elem_id"],
            "name": n.get("name"),
            "type": n.get("type"),
            "score": r["score"],
        })
    return rows

def search_similar_entities_by_embedding(q_emb: List[float], k: int) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    for idx in ("document_vec", "content_vec", "expression_vec"):
        try:
            candidates.extend(_vector_query_nodes(idx, q_emb, k))
        except Exception as e:
            log(f"[Warn] {idx} query failed: {e}", level="WARN")
    best: Dict[str, Dict[str, Any]] = {}
    for row in candidates:
        dedup_key = row.get("elem_id") or f"{row.get('key')}|{row.get('type')}"
        if dedup_key not in best or (row.get("score", -1) > best[dedup_key].get("score", -1)):
            best[dedup_key] = row
    merged = list(best.values())
    merged.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    return merged[:k]

def search_similar_triples_by_embedding(q_emb: List[float], k: int) -> List[Dict[str, Any]]:
    q_emb = _as_float_list(q_emb)
    cypher = """
    CALL db.index.vector.queryNodes('triple_vec', $k, $q_emb) YIELD node AS tr, score
    OPTIONAL MATCH (tr)-[:SUBJECT]->(s)
    OPTIONAL MATCH (tr)-[:OBJECT]->(o)
    RETURN tr, s, o, score
    ORDER BY score DESC
    LIMIT $k
    """
    res = run_cypher_with_retry(cypher, {"k": k, "q_emb": q_emb})
    rows = []
    for r in res:
        tr = r["tr"]; s = r["s"]; o = r["o"]
        rows.append({
            "triple_uid": tr.get("triple_uid"),
            "predicate": tr.get("predicate"),
            "uu_number": tr.get("uu_number"),
            "evidence_quote": tr.get("evidence_quote"),
            "subject": s.get("name") if s else None,
            "subject_key": s.get("key") if s else None,
            "subject_type": s.get("type") if s else None,
            "object": o.get("name") if o else None,
            "object_key": o.get("key") if o else None,
            "object_type": o.get("type") if o else None,
            "score": r["score"],
            "embedding": tr.get("embedding"),
            "document_id": tr.get("document_id"),
            "chunk_id": tr.get("chunk_id"),
            "evidence_article_ref": tr.get("evidence_article_ref"),
        })
    return rows

# ----------------- Graph expansion and ChunkStore -----------------
def expand_from_entities(
    entity_keys: List[str],
    hops: int,
    per_hop_limit: int,
    entity_elem_ids: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    triples: Dict[str, Dict[str, Any]] = {}
    current_ids: Set[str] = set(x for x in (entity_elem_ids or []) if x)
    current_keys: Set[str] = set(x for x in (entity_keys or []) if x)

    for _ in range(hops):
        if not current_ids and not current_keys:
            break

        if current_ids:
            cypher = """
            UNWIND $ids AS eid
            MATCH (e) WHERE elementId(e) = eid
            MATCH (e)-[r:PREDICATE]->()
            WITH DISTINCT r.triple_uid AS uid LIMIT $limit
            MATCH (tr:Triple {triple_uid: uid})
            OPTIONAL MATCH (tr)-[:SUBJECT]->(s)
            OPTIONAL MATCH (tr)-[:OBJECT]->(o)
            RETURN tr, s, o, elementId(s) AS s_id, elementId(o) AS o_id
            """
            params = {"ids": list(current_ids), "limit": per_hop_limit}
        else:
            cypher = """
            UNWIND $keys AS k
            MATCH (e:Entity {key:k})-[r:PREDICATE]->()
            WITH DISTINCT r.triple_uid AS uid LIMIT $limit
            MATCH (tr:Triple {triple_uid: uid})
            OPTIONAL MATCH (tr)-[:SUBJECT]->(s)
            OPTIONAL MATCH (tr)-[:OBJECT]->(o)
            RETURN tr, s, o, elementId(s) AS s_id, elementId(o) AS o_id
            """
            params = {"keys": list(current_keys), "limit": per_hop_limit}

        res = run_cypher_with_retry(cypher, params)

        next_ids: Set[str] = set()
        next_keys: Set[str] = set()
        for r in res:
            tr = r["tr"]; s = r["s"]; o = r["o"]
            uid = tr.get("triple_uid")
            if uid not in triples:
                triples[uid] = {
                    "triple_uid": uid,
                    "predicate": tr.get("predicate"),
                    "uu_number": tr.get("uu_number"),
                    "evidence_quote": tr.get("evidence_quote"),
                    "embedding": tr.get("embedding"),
                    "document_id": tr.get("document_id"),
                    "chunk_id": tr.get("chunk_id"),
                    "evidence_article_ref": tr.get("evidence_article_ref"),
                    "subject": s.get("name") if s else None,
                    "subject_key": s.get("key") if s else None,
                    "subject_type": s.get("type") if s else None,
                    "object": o.get("name") if o else None,
                    "object_key": o.get("key") if o else None,
                    "object_type": o.get("type") if o else None,
                }
            if s:
                if s.get("key"): next_keys.add(s.get("key"))
            if o:
                if o.get("key"): next_keys.add(o.get("key"))
            s_id = r.get("s_id");  o_id = r.get("o_id")
            if s_id: next_ids.add(s_id)
            if o_id: next_ids.add(o_id)

        current_ids = next_ids if next_ids else set()
        current_keys = set() if next_ids else next_keys

    return list(triples.values())

class ChunkStore:
    def __init__(self, root: Path, skip: Set[str]):
        self.root = root
        self.skip = skip
        self._index: Dict[Tuple[str, str], str] = {}
               # (doc_id, chunk_id) -> text
        self._by_chunk: Dict[str, List[Tuple[str, str]]] = {}
        self._loaded_files: Set[Path] = set()
        self._built = False

    def _build_index(self):
        if self._built:
            return
        start = time.monotonic()
        log(f"[ChunkStore] Building index from {self.root}...", level="INFO")
        pkls = [p for p in self.root.glob("*.pkl") if p.name not in self.skip]
        total_chunks_indexed = 0
        for pkl in pkls:
            try:
                with open(pkl, "rb") as f:
                    chunks = pickle.load(f)
                loaded_count = 0
                for ch in chunks:
                    meta = getattr(ch, "metadata", {}) if hasattr(ch, "metadata") else {}
                    doc_id = _norm_id(meta.get("document_id"))
                    chunk_id = _norm_id(meta.get("chunk_id"))
                    text = getattr(ch, "page_content", None)
                    if doc_id and chunk_id and isinstance(text, str):
                        self._index[(doc_id, chunk_id)] = text
                        self._by_chunk.setdefault(chunk_id, []).append((doc_id, chunk_id))
                        loaded_count += 1
                self._loaded_files.add(pkl)
                total_chunks_indexed += loaded_count
                log(f"[ChunkStore] Loaded {loaded_count} chunks from {pkl.name}", level="INFO")
            except Exception as e:
                log(f"[ChunkStore] Failed to load or process {pkl.name}: {e}", level="WARN")
                continue
        elapsed = time.monotonic() - start
        log(f"[ChunkStore] Index built. Total chunks indexed: {total_chunks_indexed} from {len(self._loaded_files)} files in {elapsed:.3f}s.", level="INFO")
        self._built = True

    def get_chunk(self, document_id: Any, chunk_id: Any) -> Optional[str]:
        if not self._built:
            self._build_index()

        doc_id_s = _norm_id(document_id)
        chunk_id_s = _norm_id(chunk_id)

        val = self._index.get((doc_id_s, chunk_id_s))
        if val is not None:
            log(f"[ChunkStore] HIT exact: doc={doc_id_s} chunk={chunk_id_s} len={len(val)}", level="DEBUG")
            return val

        if "::" in chunk_id_s:
            base_id = chunk_id_s.split("::", 1)[0]
            val = self._index.get((doc_id_s, base_id))
            if val is not None:
                log(f"[ChunkStore] HIT base-id: doc={doc_id_s} chunk={chunk_id_s} -> base={base_id} len={len(val)}", level="DEBUG")
                return val

        matches = self._by_chunk.get(chunk_id_s)
        if matches:
            chosen_doc, chosen_chunk = matches[0]
            val = self._index.get((chosen_doc, chosen_chunk))
            if val is not None:
                note = "" if len(matches) == 1 else f" (warn: chunk_id occurs in {len(matches)} docs; chose doc={chosen_doc})"
                log(f"[ChunkStore] HIT by chunk_id only: requested doc={doc_id_s} chunk={chunk_id_s}; using doc={chosen_doc}{note}. len={len(val)}", level="WARN")
                return val

        log(f"[ChunkStore] MISS: doc={doc_id_s} chunk={chunk_id_s} (no exact/base-id/chunk-id-only match)", level="WARN")
        return None

# ----------------- Scoring helpers -----------------
def mean_similarity_to_query_triples(triple_emb: Optional[List[float]], q_trip_embs: List[List[float]]) -> float:
    if not isinstance(triple_emb, list) and triple_emb is not None:
        triple_emb = _as_float_list(triple_emb)
    if not isinstance(triple_emb, list) or not q_trip_embs:
        return 0.0
    sims = [cos_sim(triple_emb, q) for q in q_trip_embs]
    if not sims:
        return 0.0
    return sum(sims) / len(sims)

# ----------------- GraphRAG retrieval pipeline -----------------
def entity_centric_retrieval(
    query_entities: List[Dict[str, Any]],
    q_trip_embs: List[List[float]],
    q_emb_fallback: Optional[List[float]] = None
) -> List[Dict[str, Any]]:
    all_matched_keys: Set[str] = set()
    all_matched_ids: Set[str] = set()

    for e in query_entities:
        text = (e.get("text") or "").strip()
        if not text:
            continue
        try:
            e_emb = embed_text(text)
        except Exception as ex:
            log(f"[EntityRetrieval] Embedding failed for entity '{text}': {ex}", level="WARN")
            continue
        matches = search_similar_entities_by_embedding(e_emb, k=GR_ENTITY_MATCH_TOP_K)
        keys = [m.get("key") for m in matches if m.get("key")]
        ids  = [m.get("elem_id") for m in matches if m.get("elem_id")]
        all_matched_keys.update(keys)
        all_matched_ids.update(ids)

    if not (all_matched_keys or all_matched_ids):
        log("[EntityRetrieval] No KG entity matches found from query entities.", level="WARN")
        return []

    t0 = now_ms()
    expanded_triples = expand_from_entities(
        list(all_matched_keys),
        hops=GR_ENTITY_SUBGRAPH_HOPS,
        per_hop_limit=GR_ENTITY_SUBGRAPH_PER_HOP_LIMIT,
        entity_elem_ids=list(all_matched_ids) if all_matched_ids else None
    )
    log(f"[EntityRetrieval] Expanded subgraph triples: {len(expanded_triples)} | {dur_ms(t0):.0f} ms", level="INFO")

    if not expanded_triples:
        return []

    def score(t: Dict[str, Any]) -> float:
        emb = t.get("embedding")
        if q_trip_embs:
            return mean_similarity_to_query_triples(emb, q_trip_embs)
        if q_emb_fallback and isinstance(emb, list):
            return cos_sim(q_emb_fallback, emb)
        return 0.0

    ranked = sorted(expanded_triples, key=score, reverse=True)
    top = ranked[:GR_SUBGRAPH_TRIPLES_TOP_K]
    log(f"[EntityRetrieval] Selected top-{len(top)} triples from subgraph", level="INFO")
    return top

def triple_centric_retrieval(query_triples: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[List[float]]]:
    triples_map: Dict[str, Dict[str, Any]] = {}
    q_trip_embs: List[List[float]] = []
    for qt in query_triples:
        try:
            txt = query_triple_to_text(qt)
            emb = embed_text(txt)
            q_trip_embs.append(emb)
        except Exception as ex:
            log(f"[TripleRetrieval] Embedding failed for query triple '{qt}': {ex}", level="WARN")
            continue

        matches = search_similar_triples_by_embedding(emb, k=GR_QUERY_TRIPLE_MATCH_TOP_K_PER)
        for m in matches:
            uid = m.get("triple_uid")
            if uid:
                if uid not in triples_map:
                    triples_map[uid] = m
                else:
                    if m.get("score", 0.0) > triples_map[uid].get("score", 0.0):
                        triples_map[uid] = m

    merged = list(triples_map.values())
    log(f"[TripleRetrieval] Collected {len(merged)} unique KG triples across {len(q_trip_embs)} query triple(s)", level="INFO")
    return merged, q_trip_embs

def collect_chunks_for_triples(
    triples: List[Dict[str, Any]],
    chunk_store: ChunkStore
) -> List[Tuple[Tuple[Any, Any], str, Dict[str, Any]]]:
    seen_pairs: Set[Tuple[Any, Any]] = set()
    out: List[Tuple[Tuple[Any, Any], str, Dict[str, Any]]] = []

    for t in triples:
        doc_id = t.get("document_id")
        chunk_id = t.get("chunk_id")

        if doc_id is None or chunk_id is None:
            quote = t.get("evidence_quote")
            if quote:
                key = (t.get("triple_uid"), "quote")
                if key not in seen_pairs:
                    t["_is_quote_fallback"] = True
                    out.append((key, quote, t))
                    seen_pairs.add(key)
            continue

        norm_key = (_norm_id(doc_id), _norm_id(chunk_id))
        if norm_key in seen_pairs:
            continue

        text = chunk_store.get_chunk(doc_id, chunk_id)
        if text:
            t["_is_quote_fallback"] = False
            out.append((norm_key, text, t))
            seen_pairs.add(norm_key)
        else:
            quote = t.get("evidence_quote")
            if quote:
                key2 = (t.get("triple_uid"), "quote")
                if key2 not in seen_pairs:
                    t["_is_quote_fallback"] = True
                    log(f"[ChunkStore] FALLBACK to quote for doc={_norm_id(doc_id)} chunk={_norm_id(chunk_id)}", level="WARN")
                    out.append((key2, quote, t))
                    seen_pairs.add(key2)

    return out

def rerank_chunks_by_query(
    chunk_records: List[Tuple[Tuple[Any, Any], str, Dict[str, Any]]],
    q_emb_query: List[float],
    top_k: int
) -> List[Tuple[Tuple[Any, Any], str, Dict[str, Any], float]]:
    cand = chunk_records[:GR_CHUNK_RERANK_CAND_LIMIT]
    t0 = now_ms()
    scored: List[Tuple[Tuple[Any, Any], str, Dict[str, Any], float]] = []
    for key, text, t in cand:
        try:
            emb = embed_text(text)
            s = cos_sim(q_emb_query, emb)
            scored.append((key, text, t, s))
        except Exception as ex:
            log(f"[ChunkRerank] Embedding failed for chunk {key}: {ex}", level="WARN")
            continue
    scored.sort(key=lambda x: x[3], reverse=True)
    log(f"[ChunkRerank] Scored {len(scored)} candidates | picked top {min(top_k, len(scored))} | {dur_ms(t0):.0f} ms", level="INFO")
    return scored[:top_k]

def rerank_triples_by_query_triples(
    triples: List[Dict[str, Any]],
    q_trip_embs: List[List[float]],
    q_emb_fallback: Optional[List[float]],
    top_k: int
) -> List[Dict[str, Any]]:
    t0 = now_ms()
    def score(t: Dict[str, Any]) -> float:
        emb = t.get("embedding")
        if q_trip_embs:
            return mean_similarity_to_query_triples(emb, q_trip_embs)
        if q_emb_fallback and isinstance(emb, list):
            return cos_sim(q_emb_fallback, emb)
        return 0.0

    ranked = sorted(triples, key=score, reverse=True)
    log(f"[TripleRerank] Input={len(triples)} | Output={min(top_k, len(ranked))} | {dur_ms(t0):.0f} ms", level="INFO")
    return ranked[:top_k]

def build_combined_context_text(
    triples_ranked: List[Dict[str, Any]],
    chunks_ranked: List[Tuple[Tuple[Any, Any], str, Dict[str, Any], float]]
) -> Tuple[str, str, List[Dict[str, Any]]]:
    summary_lines = []
    summary_lines.append("Ringkasan triple yang relevan:")
    for t in triples_ranked[:min(50, len(triples_ranked))]:
        s = t.get("subject"); p = t.get("predicate"); o = t.get("object")
        uu = t.get("uu_number") or ""
        art = t.get("evidence_article_ref") or ""
        quote = (t.get("evidence_quote") or "")[:300]
        summary_lines.append(f"- {s} [{p}] {o} | {uu} | {art} | bukti: {quote}")
    summary_text = "\n".join(summary_lines)

    lines = [summary_text, "\nPotongan teks terkait (chunk):"]
    for idx, (key, text, t, score) in enumerate(chunks_ranked, start=1):
        doc_id = t.get("document_id")
        chunk_id = t.get("chunk_id")
        uu = t.get("uu_number") or ""
        fb = " | quote-fallback" if t.get("_is_quote_fallback") else ""
        lines.append(f"[Chunk {idx}] doc={doc_id} chunk={chunk_id} | {uu} | score={score:.3f}{fb}\n{text}")
    context = "\n".join(lines)

    chunk_records = [{"key": key, "text": text, "triple": t, "score": score} for key, text, t, score in chunks_ranked]
    return context, summary_text, chunk_records

def agent2_answer_graphrag(query_original: str, context: str, guidance: Optional[str], output_lang: str = "id") -> str:
    instructions = (
        "Answer concisely and accurately based strictly on the provided context. "
        "Cite UU/Article references when they are clear. "
        "Respond in the same language as the user's question."
    )
    guidance_text = guidance.strip() if isinstance(guidance, str) and guidance.strip() else "(No additional guidance.)"
    prompt = f"""
You are Agent 2 (Answerer). Provide an answer based on the context only.

Core instructions:
{instructions}

Additional guidance (if any):
\"\"\"{guidance_text}\"\"\"

Original user question:
\"\"\"{query_original}\"\"\"

Context:
\"\"\"{context}\"\"\"
"""
    est = estimate_tokens_for_text(prompt)
    log("\n[Agent 2 - GraphRAG] Prompt:", level="INFO")
    log(prompt, level="INFO")
    log(f"[Agent 2 - GraphRAG] Prompt size: {len(prompt)} chars, est_tokens≈{est}", level="INFO")
    answer = safe_generate_text(prompt, max_tokens=GR_ANSWER_MAX_TOKENS, temperature=0.2)
    log(f"[Agent 2 - GraphRAG] Answer length={len(answer)}", level="INFO")
    return answer

# ----------------- Naive RAG helpers -----------------
def vector_query_chunks(q_emb: List[float], k: int) -> List[Dict[str, Any]]:
    cypher = """
    WITH $q AS q
    CALL db.index.vector.queryNodes('chunk_embedding_index', $k, q)
    YIELD node, score
    RETURN node AS c, score
    ORDER BY score DESC
    LIMIT $k
    """
    rows = run_cypher_with_retry(cypher, {"q": q_emb, "k": k})
    out = []
    for r in rows:
        c = r["c"]
        out.append({
            "key": c.get("key"),
            "chunk_id": c.get("chunk_id"),
            "document_id": c.get("document_id"),
            "uu_number": c.get("uu_number"),
            "pages": c.get("pages"),
            "content": c.get("content"),
            "score": r["score"],
        })
    return out

def clamp(s: Optional[str], n: int) -> str:
    t = (s or "").strip()
    return t[:n]

def build_context_from_chunks(chunks: List[Dict[str, Any]], max_chunks: int) -> str:
    chosen = chunks[:max_chunks]
    lines = ["Potongan teks terkait (chunk):"]
    for i, c in enumerate(chosen, 1):
        doc = c.get("document_id")
        cid = c.get("chunk_id")
        uu = c.get("uu_number") or ""
        pg = c.get("pages")
        txt = clamp(c.get("content") or "", NV_CHUNK_TEXT_CLAMP)
        lines.append(f"[Chunk {i}] doc={doc} chunk={cid} | {uu} | pages={pg} | score={c.get('score'):.3f}\n{txt}")
    return "\n".join(lines)

def agent2_answer_naive(query_original: str, context: str, guidance: Optional[str], output_lang: str = "id") -> str:
    instructions = (
        "Answer concisely and accurately based strictly on the provided context. "
        "Cite UU/Article references when they are clear. "
        "Respond in the same language as the user's question."
    )
    guidance_text = guidance.strip() if isinstance(guidance, str) and guidance.strip() else "(No additional guidance.)"
    prompt = f"""
You are Agent 2 (Answerer). Provide an answer based on the context only.

Core instructions:
{instructions}

Additional guidance (if any):
\"\"\"{guidance_text}\"\"\"

Original user question:
\"\"\"{query_original}\"\"\"

Context:
\"\"\"{context}\"\"\"
"""
    est = estimate_tokens_for_text(prompt)
    log("\n[Agent 2 - Naive] Prompt:", level="INFO")
    log(prompt, level="INFO")
    log(f"[Agent 2 - Naive] Prompt size: {len(prompt)} chars, est_tokens≈{est}", level="INFO")
    answer = safe_generate_text(prompt, max_tokens=GR_ANSWER_MAX_TOKENS, temperature=0.2)
    log(f"[Agent 2 - Naive] Answer length={len(answer)}", level="INFO")
    return answer

# ----------------- Sub-orchestrators -----------------
def run_naive_rag(query_original: str) -> Dict[str, Any]:
    user_lang = detect_user_language(query_original)
    log(f"[NaiveRAG] Detected user language: {user_lang}", level="INFO")

    t0 = now_ms()
    q_emb = embed_text(query_original)
    log(f"[NaiveRAG] Embedded query in {dur_ms(t0):.0f} ms", level="INFO")

    candidates = vector_query_chunks(q_emb, k=NV_TOP_K_CHUNKS)
    log(f"[NaiveRAG] Vector search returned {len(candidates)} candidates", level="INFO")

    if not candidates:
        context_text = "Tidak ada potongan teks yang ditemukan." if user_lang == "id" else "No relevant chunks found."
        answer = agent2_answer_naive(query_original, context_text, guidance=None, output_lang=user_lang)
    else:
        top_context = build_context_from_chunks(candidates, max_chunks=NV_MAX_CHUNKS_FINAL)
        log("[NaiveRAG Context preview]:", level="INFO")
        log("\n".join(top_context.splitlines()[:30]), level="INFO")
        answer = agent2_answer_naive(query_original, top_context, guidance=None, output_lang=user_lang)

    return {"answer": answer, "candidates": len(candidates)}

def run_graph_rag(query_original: str) -> Dict[str, Any]:
    user_lang = detect_user_language(query_original)
    log(f"[GraphRAG] Detected user language: {user_lang}", level="INFO")

    chunk_store = ChunkStore(LANGCHAIN_DIR, set(SKIP_FILES))

    t0 = now_ms()
    q_emb_query = embed_text(query_original)
    log(f"[GraphRAG] Whole-query embedding in {dur_ms(t0):.0f} ms", level="INFO")

    t1 = now_ms()
    extraction = agent1_extract_entities_predicates(query_original)
    ents = extraction.get("entities", [])
    preds = extraction.get("predicates", [])
    log(f"[GraphRAG] Agent1 extraction in {dur_ms(t1):.0f} ms | ents={len(ents)} preds={len(preds)}", level="INFO")

    t1b = now_ms()
    query_triples = agent1b_extract_query_triples(query_original)
    log(f"[GraphRAG] Agent1b triple extraction in {dur_ms(t1b):.0f} ms | triples={len(query_triples)}", level="INFO")

    t2 = now_ms()
    ctx2_triples, q_trip_embs = triple_centric_retrieval(query_triples)
    log(f"[GraphRAG] Triple-centric retrieval in {dur_ms(t2):.0f} ms | triples={len(ctx2_triples)} q_embs={len(q_trip_embs)}", level="INFO")

    t3 = now_ms()
    ctx1_triples = entity_centric_retrieval(ents, q_trip_embs=q_trip_embs, q_emb_fallback=q_emb_query)
    log(f"[GraphRAG] Entity-centric retrieval in {dur_ms(t3):.0f} ms | triples={len(ctx1_triples)}", level="INFO")

    t4 = now_ms()
    triple_map: Dict[str, Dict[str, Any]] = {}
    for t in ctx1_triples + ctx2_triples:
        uid = t.get("triple_uid")
        if uid:
            prev = triple_map.get(uid)
            if prev is None or (t.get("score", 0.0) > prev.get("score", 0.0)):
                triple_map[uid] = t
    merged_triples = list(triple_map.values())
    log(f"[GraphRAG] Merged triples: {len(merged_triples)} in {dur_ms(t4):.0f} ms", level="INFO")

    t5 = now_ms()
    chunk_records = collect_chunks_for_triples(merged_triples, chunk_store)
    log(f"[GraphRAG] Collected {len(chunk_records)} chunk candidates", level="INFO")
    chunks_ranked = rerank_chunks_by_query(chunk_records, q_emb_query, top_k=GR_MAX_CHUNKS_FINAL)
    log(f"[GraphRAG] Reranked chunks: selected {len(chunks_ranked)} in {dur_ms(t5):.0f} ms", level="INFO")

    t6 = now_ms()
    triples_ranked = rerank_triples_by_query_triples(merged_triples, q_trip_embs=q_trip_embs, q_emb_fallback=q_emb_query, top_k=GR_MAX_TRIPLES_FINAL)
    log(f"[GraphRAG] Reranked triples: selected {len(triples_ranked)} in {dur_ms(t6):.0f} ms", level="INFO")

    t_ctx = now_ms()
    context_text, context_summary, _ = build_combined_context_text(triples_ranked, chunks_ranked)
    log(f"[GraphRAG Context] Built in {dur_ms(t_ctx):.0f} ms", level="INFO")
    log("\n[GraphRAG Context Summary]:", level="INFO")
    log(context_summary, level="INFO")

    t7 = now_ms()
    answer = agent2_answer_graphrag(query_original, context_text, guidance=None, output_lang=user_lang)
    log(f"[GraphRAG] Answer generated in {dur_ms(t7):.0f} ms", level="INFO")

    return {"answer": answer, "triples": len(merged_triples), "chunks": len(chunks_ranked)}

# ----------------- Aggregator Agent (per IQ step; existing) -----------------
AGG_SCHEMA = {
  "type": "object",
  "properties": {
    "chosen": {"type": "string", "enum": ["naive", "graphrag", "mixed"]},
    "final_answer": {"type": "string"},
    "rationale": {"type": "string"},
    "confidence": {"type": "number"},
    "key_points": {"type": "array", "items": {"type": "string"}}
  },
  "required": ["chosen", "final_answer"]
}

def aggregator_agent(query: str, naive_answer: str, graphrag_answer: str, lang: str) -> Dict[str, Any]:
    prompt = f"""
You are the Aggregator. You receive:
- The original user question.
- Two candidate answers:
  * Naive RAG answer (vector-search over chunks)
  * GraphRAG answer (KG-assisted retrieval)

Goal:
- Choose the best answer or combine them into a stronger final answer.
- Prefer answers with explicit citations (UU, Pasal/Article numbers) and higher internal consistency.
- If both are solid and compatible, you may merge key points.
- If they conflict, select the more grounded/specific answer; briefly resolve the conflict if possible.
- Be concise and accurate.
- Respond in the same language as the user's question.

Return JSON with:
  - chosen: "naive" | "graphrag" | "mixed"
  - final_answer: the final response text
  - rationale: brief reasoning for your choice
  - confidence: 0.0–1.0 (float)
  - key_points: optional list of key bullets

{STRICT_JSON_DIRECTIVE}

Question:
\"\"\"{query}\"\"\"

Naive RAG answer:
\"\"\"{naive_answer}\"\"\"

GraphRAG answer:
\"\"\"{graphrag_answer}\"\"\"
"""
    log("\n[Aggregator] Prompt:", level="INFO")
    log(prompt, level="INFO")
    out = safe_generate_json(prompt, AGG_SCHEMA, temp=0.0) or {}
    chosen = (out.get("chosen") or "").strip().lower()
    final_answer = (out.get("final_answer") or "").strip()
    rationale = (out.get("rationale") or "").strip()
    confidence = float(out.get("confidence") or 0.0)

    if not final_answer or chosen not in ("naive", "graphrag", "mixed"):
        log("[Aggregator] Fallback selection heuristic activated.", level="WARN")
        def ref_score(txt: str) -> int:
            if not isinstance(txt, str):
                return 0
            patterns = [
                r"\b(Pasal|Ayat|UU|Undang[- ]?Undang|Peraturan|Bab|Bagian)\b",
                r"\b(Article|Section|Chapter|Act|Law|Regulation)\b",
                r"\bPasal\s+\d+",
                r"\bArticle\s+\d+",
            ]
            s = 0
            for pat in patterns:
                m = re.findall(pat, txt, flags=re.IGNORECASE)
                s += len(m)
            s += len(re.findall(r"\[\d+\]", txt))
            s += len(re.findall(r"\b\d{1,3}(\.\d+)?\b", txt)) // 4
            return s

        n_ok = isinstance(naive_answer, str) and len(naive_answer.strip()) > 0
        g_ok = isinstance(graphrag_answer, str) and len(graphrag_answer.strip()) > 0

        if g_ok and (ref_score(graphrag_answer) >= ref_score(naive_answer or "")):
            chosen = "graphrag"
            final_answer = graphrag_answer.strip()
        elif n_ok:
            chosen = "naive"
            final_answer = naive_answer.strip()
        else:
            chosen = "mixed"
            final_answer = "Maaf, saya tidak menemukan jawaban berdasarkan konteks yang tersedia." if lang == "id" else "Sorry, I could not find an answer based on the available context."
        rationale = rationale or "Selected based on citation density and completeness."
        confidence = confidence or 0.55

    log(f"[Aggregator] Decision: chosen={chosen}, confidence={confidence:.2f}", level="INFO")
    log(f"[Aggregator] Rationale: {rationale}", level="INFO")

    return {
        "chosen": chosen,
        "final_answer": final_answer,
        "rationale": rationale,
        "confidence": confidence,
        "key_points": out.get("key_points") or []
    }

# ----------------- Agentic Multi (single query; unchanged) -----------------
def agentic_multi(query_original: str, logger: Optional[FileLogger] = None, log_context_prefix: Optional[str] = None) -> Dict[str, Any]:
    """
    Runs NaiveRAG + GraphRAG + aggregator for a single query.
    If logger is provided, uses shared logger and does not close it. Otherwise creates/owns its own logger.
    Optionally sets a per-thread log prefix (e.g., iq=<id>) for log correlation.
    """
    global _LOGGER
    prev_logger = _LOGGER
    prev_ctx = getattr(_THREAD_LOCAL, "prefix", "")
    created_local_logger = False

    ts_name = make_timestamp_name()
    if logger is not None:
        _LOGGER = logger
        if log_context_prefix:
            set_log_context(log_context_prefix)
    else:
        log_file = Path.cwd() / f"{ts_name}.txt"
        _LOGGER = FileLogger(log_file, also_console=True)
        created_local_logger = True

    t_all = now_ms()
    try:
        log("=== Multi-Agent RAG run started ===", level="INFO")
        log(f"Process info: pid={_pid()} tid={_tid()}{_get_log_context()}", level="INFO")
        if logger is not None:
            log(f"Shared Log file: {logger.file_path}", level="INFO")
        else:
            log(f"Log file: {Path.cwd() / f'{ts_name}.txt'}", level="INFO")
        log(f"Original Query: {query_original}", level="INFO")
        log(f"Parameters:", level="INFO")
        log(f"  GraphRAG: ENTITY_MATCH_TOP_K={GR_ENTITY_MATCH_TOP_K}, ENTITY_SUBGRAPH_HOPS={GR_ENTITY_SUBGRAPH_HOPS}, "
            f"ENTITY_SUBGRAPH_PER_HOP_LIMIT={GR_ENTITY_SUBGRAPH_PER_HOP_LIMIT}, SUBGRAPH_TRIPLES_TOP_K={GR_SUBGRAPH_TRIPLES_TOP_K}, "
            f"QUERY_TRIPLE_MATCH_TOP_K_PER={GR_QUERY_TRIPLE_MATCH_TOP_K_PER}, MAX_TRIPLES_FINAL={GR_MAX_TRIPLES_FINAL}, "
            f"MAX_CHUNKS_FINAL={GR_MAX_CHUNKS_FINAL}, CHUNK_RERANK_CAND_LIMIT={GR_CHUNK_RERANK_CAND_LIMIT}, "
            f"ANSWER_MAX_TOKENS={GR_ANSWER_MAX_TOKENS}, MAX_ITERS={MAX_ITERS}", level="INFO")
        log(f"  Naive: TOP_K_CHUNKS={NV_TOP_K_CHUNKS}, MAX_CHUNKS_FINAL={NV_MAX_CHUNKS_FINAL}, CHUNK_TEXT_CLAMP={NV_CHUNK_TEXT_CLAMP}", level="INFO")
        log(f"  LLM limits: LLM_CALLS_PER_MINUTE={LLM_CALLS_PER_MINUTE}, EMBEDDING_CALLS_PER_MINUTE={EMBEDDING_CALLS_PER_MINUTE}", level="INFO")
        log(f"  Neo4j: NEO4J_MAX_ATTEMPTS={NEO4J_MAX_ATTEMPTS}, NEO4J_TX_TIMEOUT_S={NEO4J_TX_TIMEOUT_S:.1f}", level="INFO")

        user_lang = detect_user_language(query_original)
        log(f"[Language] Detected user language: {user_lang}", level="INFO")

        # Run Naive RAG
        naive_result: Dict[str, Any] = {}
        try:
            t_nv = now_ms()
            naive_result = run_naive_rag(query_original)
            log(f"[NaiveRAG] Completed in {dur_ms(t_nv):.0f} ms", level="INFO")
        except Exception as e:
            log(f"[NaiveRAG] Error: {e}", level="ERROR")
            naive_result = {"answer": ""}

        # Run GraphRAG
        graph_result: Dict[str, Any] = {}
        try:
            t_gr = now_ms()
            graph_result = run_graph_rag(query_original)
            log(f"[GraphRAG] Completed in {dur_ms(t_gr):.0f} ms", level="INFO")
        except Exception as e:
            log(f"[GraphRAG] Error: {e}", level="ERROR")
            graph_result = {"answer": ""}

        naive_answer = naive_result.get("answer", "") or ""
        graphrag_answer = graph_result.get("answer", "") or ""

        # Aggregate
        t_ag = now_ms()
        agg = aggregator_agent(query_original, naive_answer, graphrag_answer, user_lang)
        t_agg = dur_ms(t_ag)

        final_answer = agg.get("final_answer", "") or ""
        decision = {
            "chosen": agg.get("chosen"),
            "confidence": float(agg.get("confidence") or 0.0),
            "rationale": agg.get("rationale") or ""
        }

        total_ms = dur_ms(t_all)
        log("\n=== Multi-Agent RAG summary ===", level="INFO")
        log(f"- Iterations used: 1 (Naive + GraphRAG + Aggregator)", level="INFO")
        log(f"- Aggregator: chosen={decision['chosen']}, confidence={decision['confidence']:.2f}", level="INFO")
        log(f"- Total runtime: {total_ms:.0f} ms", level="INFO")
        log("\n=== Final Answer ===", level="INFO")
        log(final_answer, level="INFO")

        return {
            "final_answer": final_answer,
            "naive_answer": naive_answer,
            "graphrag_answer": graphrag_answer,
            "aggregator_decision": decision,
            "iterations": 1
        }
    finally:
        # Restore logger and context; close only if we created it locally
        if created_local_logger and _LOGGER is not None:
            _LOGGER.close()
        _LOGGER = prev_logger
        set_log_context(prev_ctx)

# ----------------- NEW: Intermediate Question (IQ) Generator Agent -----------------
# In Script 2, replace the IQ_GEN_SCHEMA with this (from Script 1):
IQ_PLAN_SCHEMA = {
  "type": "object",
  "properties": {
    "iqs": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "id": {"type": "string"},
          "question": {"type": "string"},
          "rationale": {"type": "string"},
          "depends_on": {"type": "array", "items": {"type": "string"}}
        },
        "required": ["question"]  # Changed from "text" to "question"
      }
    }
  },
  "required": ["iqs"]
}

# Replace the iq_generator_agent function with this:
def iq_generator_agent(query_original: str, lang: str, max_n: int = IQ_MAX_N) -> List[Dict[str, str]]:
    # Use exact same instruction text as Script 1
    instruction = (
        "Plan a short sequence of dependent intermediate questions (IQs) to answer the user's query. "
        "Make later IQs depend on earlier answers only if that helps. "
        f"Return at most {max_n} IQs. If decomposition isn't needed, return exactly one IQ identical to the original query. "
        "Write IQs in the same language as the user's query."
    )
    
    # Use exact same prompt structure as Script 1
    prompt = f"""
You are the Intermediate Question Planner.

Instruction:
{instruction}

User query:
\"\"\"{query_original}\"\"\"

Output JSON schema:
- iqs: array of items with fields:
  - id (string; optional; you may leave blank)
  - question (string; REQUIRED; the intermediate question)
  - rationale (string; optional; why this IQ is needed)
  - depends_on (array of prior IQ ids, optional)
"""
    
    est = estimate_tokens_for_text(prompt)
    log("\n[IQGenerator] Prompt:", level="INFO")
    log(prompt, level="INFO")
    log(f"[IQGenerator] Prompt size: {len(prompt)} chars, est_tokens≈{est}", level="INFO")
    
    # Use the schema (note: no STRICT_JSON_DIRECTIVE like Script 1, keeping Script 2's approach)
    out = safe_generate_json(prompt, IQ_PLAN_SCHEMA, temp=0.0) or {}
    raw_iqs = out.get("iqs") or []
    
    # Use Script 1's normalization function
    def _normalize_question(q: str) -> str:
        q = (q or "").strip().lower()
        q = re.sub(r"\s+", " ", q)
        return q
    
    # Use Script 1's cleanup logic
    cleaned: List[Dict[str, str]] = []
    seen = set()
    for i, iq in enumerate(raw_iqs, 1):
        q = (iq.get("question") or "").strip()  # Changed from "text" to "question"
        if not q:
            continue
        key = _normalize_question(q)
        if key in seen:
            continue
        seen.add(key)
        rid = (iq.get("id") or "").strip() or f"IQ{i}"
        rationale = (iq.get("rationale") or "").strip()
        deps = iq.get("depends_on") or []
        cleaned.append({"id": rid, "question": q, "rationale": rationale, "depends_on": deps})
        if len(cleaned) >= max_n:
            break
    
    # Use Script 1's fallback
    if not cleaned:
        cleaned = [{"id": "IQ1", "question": query_original.strip(), "rationale": "Decomposition not needed; answer the original query directly.", "depends_on": []}]
    
    log(f"[IQGenerator] Produced {len(cleaned)} IQ(s):", level="INFO")
    for iq in cleaned:
        log(f"  - {iq['id']}: {iq['question']} (rationale: {iq.get('rationale','')})", level="INFO")
    
    return cleaned



# ----------------- NEW: Query Modifier Agent -----------------
QUERY_MOD_SCHEMA = {
    "type": "object",
    "properties": {
        "updated_text": {"type": "string"},
        "proceed": {"type": "boolean"},
        "notes": {"type": "string"}
    },
    "required": ["updated_text"]
}

def query_modifier_agent(
    draft_next_iq_text: str,
    history: List[Dict[str, Any]],
    lang: str
) -> Dict[str, Any]:
    # Build concise history block
    lines = []
    for h in history:
        ans = (h.get("final_answer") or "").strip()
        snippet = clamp(ans, IQ_ANSWER_SNIPPET_CLAMP)
        chosen = (h.get("aggregator_decision") or {}).get("chosen", "")
        conf = float((h.get("aggregator_decision") or {}).get("confidence", 0.0))
        lines.append(
            f"- id={h.get('id')} | question: {h.get('question')}\n"
            f"  chosen_pipeline={chosen} confidence={conf:.2f}\n"
            f"  answer_snippet:\n{snippet}\n"
        )
    history_block = "\n".join(lines) if lines else "(no prior IQs)"

    prompt = f"""
You are the Query Modifier.

Task:
- Given the next draft IQ and the history of previous IQ answers, produce an updated IQ that is self-contained and ready for retrieval.
- Resolve placeholders like "(answer to iq1)" by substituting concrete values/details inferred from the history answers.
- Preserve legal entities/citations; keep the same language as the user.
- If the next IQ is redundant (already answered implicitly by history), set "proceed": false; otherwise true.

Return JSON with:
- updated_text (string): the completed/clarified IQ (in {lang})
- proceed (boolean): default true
- notes (optional)

{STRICT_JSON_DIRECTIVE}

Draft next IQ ({lang}):
\"\"\"{draft_next_iq_text}\"\"\"

History (earlier IQs and their answers):
{history_block}
"""
    log("\n[QueryModifier] Prompt:", level="INFO")
    log(prompt, level="INFO")
    out = safe_generate_json(prompt, QUERY_MOD_SCHEMA, temp=0.0) or {}
    updated_text = (out.get("updated_text") or "").strip()
    proceed = bool(out.get("proceed")) if "proceed" in out else True
    notes = (out.get("notes") or "").strip()

    if not updated_text:
        log("[QueryModifier] Empty updated_text; using draft IQ unchanged.", level="WARN")
        updated_text = draft_next_iq_text
        proceed = True

    log(f"[QueryModifier] proceed={proceed} | updated_text_len={len(updated_text)}", level="INFO")
    if notes:
        log(f"[QueryModifier] notes: {notes}", level="INFO")

    return {"updated_text": updated_text, "proceed": proceed, "notes": notes}

# ----------------- NEW: IQ Orchestrator (sequential) -----------------
def iq_orchestrator(query_original: str) -> Dict[str, Any]:
    """
    Top-level orchestrator (sequential IQ pipeline):
    - Generates up to N intermediate questions (IQs)
    - For each IQ in order:
        * Use Query Modifier to complete/enrich based on prior IQ answers
        * Run agentic_multi on the updated IQ
    - The final IQ's answer is the overall final answer
    """
    global _LOGGER
    ts_name = make_timestamp_name()
    root_log = Path.cwd() / f"{ts_name}-iq.txt"
    _LOGGER = FileLogger(root_log, also_console=True)
    set_log_context("iq-runner")

    t_all = now_ms()
    try:
        log("=== IQ Orchestrator run started ===", level="INFO")
        log(f"Process info: pid={_pid()} tid={_tid()}{_get_log_context()}", level="INFO")
        log(f"Log file: {root_log}", level="INFO")
        log(f"Original Query: {query_original}", level="INFO")
        log(f"Parameters:", level="INFO")
        log(f"  IQ: IQ_MAX_N={IQ_MAX_N}, IQ_ANSWER_SNIPPET_CLAMP={IQ_ANSWER_SNIPPET_CLAMP}", level="INFO")
        log(f"  LLM limits: LLM_CALLS_PER_MINUTE={LLM_CALLS_PER_MINUTE}, EMBEDDING_CALLS_PER_MINUTE={EMBEDDING_CALLS_PER_MINUTE}", level="INFO")
        log(f"  Neo4j: NEO4J_MAX_ATTEMPTS={NEO4J_MAX_ATTEMPTS}, NEO4J_TX_TIMEOUT_S={NEO4J_TX_TIMEOUT_S:.1f}", level="INFO")

        user_lang = detect_user_language(query_original)
        log(f"[Language] Detected user language: {user_lang}", level="INFO")

        # Step 1: Generate IQs
        iqs = iq_generator_agent(query_original, user_lang, max_n=IQ_MAX_N)
        if not iqs:
            iqs = [{"id": "iq1", "text": query_original.strip(), "rationale": "No decomposition.", "depends_on": []}]
        iqs = iqs[:IQ_MAX_N]
        log(f"[IQ Orchestrator] Executing {len(iqs)} IQ step(s).", level="INFO")

        history: List[Dict[str, Any]] = []
        iq_results: List[Dict[str, Any]] = []
        final_answer: str = ""
        last_executed_id: Optional[str] = None

        # Step 2: Sequentially process IQs
        for i, iq in enumerate(iqs, start=1):
            sid = iq.get("id") or f"iq{i}"
            draft_text = iq.get("question", "").strip()  # Changed from "text" to "question"
            if not draft_text:
                log(f"[IQ Orchestrator] Skipping empty IQ {sid}", level="WARN")
                continue

            set_log_context(f"iq={sid}")

            # Modify/complete the next IQ using history
            try:
                mod = query_modifier_agent(draft_text, history, user_lang)
                updated_text = mod.get("updated_text", draft_text)
                proceed = True
            except Exception as e:
                log(f"[IQ Orchestrator] Query modifier failed for {sid}: {e}. Using draft IQ as-is.", level="WARN")
                updated_text = draft_text
                proceed = True

            if not proceed:
                log(f"[IQ Orchestrator] Early stop signaled by Query Modifier at {sid}.", level="INFO")
                break

            # Run per-IQ RAG + aggregator
            try:
                res = agentic_multi(updated_text, logger=_LOGGER, log_context_prefix=f"iq={sid}")
            except Exception as e:
                log(f"[IQ Orchestrator] agentic_multi failed for {sid}: {e}", level="ERROR")
                res = {"final_answer": "", "aggregator_decision": {"chosen":"", "confidence":0.0, "rationale": str(e)}, "iterations": 1}

            # Update history
            final_answer = (res.get("final_answer") or "").strip()
            last_executed_id = sid
            hist_entry = {
                "id": sid,
                "question": updated_text,
                "final_answer": final_answer,
                "aggregator_decision": res.get("aggregator_decision", {}),
                "iterations": res.get("iterations", 1)
            }
            history.append(hist_entry)

            # Record result for output
            iq_results.append({
                "id": sid,
                "draft_question": draft_text,  # Renamed for consistency
                "updated_question": updated_text,  # Renamed for consistency
                "final_answer": final_answer,
                "aggregator_decision": res.get("aggregator_decision", {}),
                "iterations": res.get("iterations", 1)
            })


        # If no IQ produced an answer, fallback: run once on original query
        # If no IQ produced an answer, fallback: run once on original query
        if not final_answer.strip():
            log("[IQ Orchestrator] No IQ produced an answer; running fallback on original query.", level="WARN")
            try:
                res0 = agentic_multi(query_original, logger=_LOGGER, log_context_prefix="iq=fallback")
                final_answer = (res0.get("final_answer") or "").strip()
                iq_results.append({
                    "id": "iq_fallback",
                    "draft_question": query_original,  # Renamed for consistency
                    "updated_question": query_original,  # Renamed for consistency
                    "final_answer": final_answer,
                    "aggregator_decision": res0.get("aggregator_decision", {}),
                    "iterations": res0.get("iterations", 1)
                })
            except Exception as e:
                log(f"[IQ Orchestrator] Fallback run failed: {e}", level="ERROR")
                final_answer = "Maaf, saya tidak dapat menemukan jawaban." if user_lang == "id" else "Sorry, I could not find an answer."

        total_ms = dur_ms(t_all)
        log("\n=== IQ Orchestrator summary ===", level="INFO")
        log(f"- IQ steps generated: {len(iqs)}", level="INFO")
        log(f"- IQ steps executed: {len(history)}", level="INFO")
        if last_executed_id:
            log(f"- Last executed IQ: {last_executed_id}", level="INFO")
        log(f"- Total runtime: {total_ms:.0f} ms", level="INFO")
        log("\n=== Final Answer (IQ Orchestrator) ===", level="INFO")
        log(final_answer, level="INFO")
        log(f"\nLogs saved to: {root_log}", level="INFO")

        return {
            "final_answer": final_answer,
            "iqs": iqs,
            "iq_results": iq_results,
            "log_file": str(root_log),
            "iterations": len(history)
        }
    finally:
        if _LOGGER is not None:
            _LOGGER.close()
        _LOGGER = None
        set_log_context("")

# ----------------- Main -----------------
if __name__ == "__main__":
    try:
        user_query = input("Enter your query: ").strip()
        if not user_query:
            print("Empty query. Exiting.")
        else:
            # Run the new IQ orchestrator (sequential)
            result = iq_orchestrator(user_query)
            # Print only the final answer to stdout (logs contain details)
            print("\n----- Final Answer -----")
            print(result.get("final_answer", ""))
    finally:
        try:
            driver.close()
        except Exception:
            pass