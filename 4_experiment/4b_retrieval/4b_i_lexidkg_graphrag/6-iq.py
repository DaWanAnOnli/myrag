# lexidkg_graphrag_agentic.py
# Agentic GraphRAG with sequential Intermediate Questions (IQs):
# - Agent I0: Intermediate Question Generator (sequential, dependent plan; capped by hardcoded max)
# - Agent Q: Query Modifier (enriches each next IQ using previous IQ–answer pairs)
# - Agents 1 & 1b: Entity/predicate and triple extraction
# - Agent 2: Answerer per IQ (GraphRAG retrieval/answering)
# Orchestration:
# - Strictly sequential loop over IQs; no parallelism
# - Store all planned/completed IQs and intermediate answers
# - Final answer = the answer from the last IQ (no final aggregator)
# Plus:
# - Global LLM rate limiting (concurrency + QPS) for embeddings and generations
# - Thread-safe logging with per-IQ tags
# - Embedding cache
# - Shared ChunkStore across iterations
# - Optional Neo4j concurrency cap

import os, time, json, math, pickle, re, random, hashlib, threading
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, Set
from threading import Lock

from dotenv import load_dotenv
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
from neo4j import GraphDatabase

# ----------------- Load .env (parent directory of this file) -----------------
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent.parent.parent / ".env")

# ----------------- Config -----------------
# Credentials and endpoints should stay in env for safety.
GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]
NEO4J_URI  = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASS", "password")

# Neo4j retry/timeout controls (to prevent infinite loops on persistent errors)
NEO4J_TX_TIMEOUT_S = float(os.getenv("NEO4J_TX_TIMEOUT_S", "60"))  # per-query transaction timeout (seconds)
NEO4J_MAX_ATTEMPTS = int(os.getenv("NEO4J_MAX_ATTEMPTS", "10"))   # max attempts for run_cypher_with_retry
NEO4J_MAX_CONCURRENCY = int(os.getenv("NEO4J_MAX_CONCURRENCY", "0"))  # 0=unlimited

# Gemini models (can be overridden via env if desired)
GEN_MODEL = os.getenv("GEN_MODEL", "models/gemini-2.5-flash")
EMBED_MODEL = os.getenv("EMBED_MODEL", "models/text-embedding-004")

# Dataset folder for original chunk pickles (same as ingestion)
DEFAULT_LANGCHAIN_DIR = (Path(__file__).resolve().parent / "../../../dataset/3_indexing/3a_langchain_results/").resolve()
LANGCHAIN_DIR = Path(os.getenv("LANGCHAIN_DIR") or str(DEFAULT_LANGCHAIN_DIR))
SKIP_FILES = {"all_langchain_documents.pkl"}

# ----------------- Retrieval/agent parameters (hardcoded constants) -----------------
# Entity-centric path
ENTITY_MATCH_TOP_K = 15                 # top similar KG entities per extracted query entity
ENTITY_SUBGRAPH_HOPS = 1               # hop-depth for subgraph expansion from matched entities
ENTITY_SUBGRAPH_PER_HOP_LIMIT = 2000   # per-hop expansion limit
SUBGRAPH_TRIPLES_TOP_K = 30            # top triples selected from subgraph after triple-vs-triple similarity

# Triple-centric path
QUERY_TRIPLE_MATCH_TOP_K_PER = 20      # per query-triple, top similar KG triples

# Final context combination and reranking
MAX_TRIPLES_FINAL = 60                 # final number of triples after reranking
MAX_CHUNKS_FINAL = 40                  # final number of chunks after reranking
CHUNK_RERANK_CAND_LIMIT = 200          # cap chunk candidates before embedding/reranking to control cost

# Agent loop and output
ANSWER_MAX_TOKENS = 4096
MAX_ITERS = 3  # Note: ignored in current pipelines

# Language setting
OUTPUT_LANG = "id"  # retained for compatibility; we auto-detect based on query

# ----------------- Intermediate Questions (new) -----------------
# Hardcoded maximum number of intermediate questions Agent I0 may produce (per requirement)
IQ_MAX_STEPS = 5

# Aggregation removed by design (final answer is the last IQ's answer)

# ----------------- Global LLM throttling (concurrency + QPS) -----------------
LLM_EMBED_MAX_CONCURRENCY = max(1, int(os.getenv("LLM_EMBED_MAX_CONCURRENCY", "2")))
LLM_EMBED_QPS = float(os.getenv("LLM_EMBED_QPS", "2.0"))   # average embed calls per second (global)
LLM_GEN_MAX_CONCURRENCY   = max(1, int(os.getenv("LLM_GEN_MAX_CONCURRENCY", "1")))
LLM_GEN_QPS   = float(os.getenv("LLM_GEN_QPS", "1.0"))     # average generation calls per second (global)

# Embedding cache cap
CACHE_EMBED_MAX_ITEMS = int(os.getenv("CACHE_EMBED_MAX_ITEMS", "200000"))

# ----------------- Initialize SDKs -----------------
genai.configure(api_key=GOOGLE_API_KEY)
gen_model = genai.GenerativeModel(GEN_MODEL)

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

# ----------------- Logger -----------------
_LOGGER: Optional["FileLogger"] = None
_LOG_TL = threading.local()  # thread-local for iq tags

def set_log_context(tag: Optional[str]):
    setattr(_LOG_TL, "iq_tag", tag or None)

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

def _prefix(level: str = "INFO") -> str:
    tag = getattr(_LOG_TL, "iq_tag", None)
    tag_part = f" [iq={tag}]" if tag else ""
    return f"[{_now_ts()}] [{level}]{tag_part} [pid={_pid()}]"

class FileLogger:
    def __init__(self, file_path: Path, also_console: bool = True):
        self.file_path = file_path
        self.also_console = also_console
        self._fh = open(file_path, "w", encoding="utf-8")
        self._lock = Lock()

    def log(self, msg: str = ""):
        with self._lock:
            self._fh.write(msg + "\n")
            self._fh.flush()
            if self.also_console:
                print(msg, flush=True)

    def close(self):
        try:
            with self._lock:
                self._fh.flush()
                self._fh.close()
        except Exception:
            pass

def _fmt_msg(msg: Any, level: str) -> str:
    if not isinstance(msg, str):
        try:
            msg = json.dumps(msg, ensure_ascii=False, default=str)
        except Exception:
            msg = str(msg)
    lines = str(msg).splitlines() or [str(msg)]
    prefixed = [f"{_prefix(level)} {line}" for line in lines] if lines else [f"{_prefix(level)}"]
    return "\n".join(prefixed)

def log(msg: Any = "", level: str = "INFO"):
    global _LOGGER
    out = _fmt_msg(msg, level)
    if _LOGGER is not None:
        _LOGGER.log(out)
    else:
        print(out, flush=True)

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
    """Normalize IDs so (uuid object, stringified uuid, whitespace variants) compare the same."""
    return str(x).strip() if x is not None else ""

def estimate_tokens_for_text(text: str) -> int:
    # Quick heuristic: ~4 characters per token
    return max(1, int(len(text) / 4))

def _as_float_list(vec) -> List[float]:
    """
    Normalize any vector-like to a fresh Python list[float].
    Prevents Neo4j driver 'Existing exports of data: object cannot be re-sized.' errors.
    """
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

# Retry helpers for any external API call (e.g., rate limit, network)
def _rand_wait_seconds() -> float:
    # Uniform 5–20 seconds with jitter
    return random.uniform(5.0, 20.0)

def _api_call_with_retry(func, *args, **kwargs):
    """
    Call the given function with the provided args/kwargs.
    If any exception occurs (e.g., rate limit, network), wait 5–20s and retry.
    Retries indefinitely until success.
    """
    while True:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            wait_s = _rand_wait_seconds()
            log(f"[Retry] API call failed: {e}. Retrying in {wait_s:.1f}s.", level="WARN")
            time.sleep(wait_s)

# ----------------- Global rate limiters -----------------
class QpsLimiter:
    def __init__(self, qps: float):
        self.qps = max(0.0, float(qps))
        self._min_interval = 1.0 / self.qps if self.qps > 0 else 0.0
        self._lock = Lock()
        self._next_time = 0.0

    def acquire(self):
        if self.qps <= 0:
            return
        with self._lock:
            now = time.time()
            if now < self._next_time:
                time.sleep(self._next_time - now)
                now = time.time()
            self._next_time = now + self._min_interval

_EMBED_SEM = threading.Semaphore(LLM_EMBED_MAX_CONCURRENCY)
_GEN_SEM   = threading.Semaphore(LLM_GEN_MAX_CONCURRENCY)
_EMBED_QPS = QpsLimiter(LLM_EMBED_QPS)
_GEN_QPS   = QpsLimiter(LLM_GEN_QPS)
_NEO4J_SEM = threading.Semaphore(NEO4J_MAX_CONCURRENCY) if NEO4J_MAX_CONCURRENCY > 0 else None

# Embedding cache
_EMB_CACHE: Dict[str, List[float]] = {}
_EMB_CACHE_LOCK = Lock()

def _cache_key_for_text(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()

# ----------------- Embedding & similarity -----------------
def embed_text(text: str) -> List[float]:
    # Cache check
    key = _cache_key_for_text(text)
    with _EMB_CACHE_LOCK:
        if key in _EMB_CACHE:
            return list(_EMB_CACHE[key])

    # Throttle and call
    with _EMBED_SEM:
        _EMBED_QPS.acquire()
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

    # Cache store (bounded)
    with _EMB_CACHE_LOCK:
        if len(_EMB_CACHE) >= CACHE_EMBED_MAX_ITEMS:
            try:
                _EMB_CACHE.pop(next(iter(_EMB_CACHE)))
            except Exception:
                _EMB_CACHE.clear()
        _EMB_CACHE[key] = list(out)
    return out

def cos_sim(a: List[float], b: List[float]) -> float:
    a = _as_float_list(a)
    b = _as_float_list(b)
    dot = sum(x*y for x,y in zip(a,b))
    na = math.sqrt(sum(x*x for x in a))
    nb = math.sqrt(sum(y*y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)

# ----------------- Simple language detection (ID vs EN) -----------------
def detect_user_language(text: str) -> str:
    """
    Lightweight heuristic to detect Indonesian ('id') vs English ('en')
    for the purpose of matching the user's query language in the final answer.
    """
    t = (text or "").lower()

    # Strong hints
    if re.search(r"\b(pasal|undang[- ]?undang|uu\s*\d|peraturan|menteri|ayat|bab|bagian|paragraf|ketentuan|sebagaimana|dimaksud)\b", t):
        return "id"
    if re.search(r"\b(article|act|law|regulation|minister|section|paragraph|chapter|pursuant|provided that)\b", t):
        return "en"

    # Token-based scoring
    id_tokens = {
        "yang","dan","atau","tidak","adalah","berdasarkan","sebagaimana","pada","dalam","dapat","harus","wajib",
        "pasal","undang","peraturan","menteri","ayat","bab","bagian","paragraf","ketentuan","pengundangan","apabila","jika"
    }
    en_tokens = {
        "the","and","or","not","is","based","as","provided","pursuant","in","may","must","shall",
        "article","act","law","regulation","minister","section","paragraph","chapter","whereas"
    }
    words = re.findall(r"[a-z]+", t)
    score_id = sum(1 for w in words if w in id_tokens)
    score_en = sum(1 for w in words if w in en_tokens)
    if score_id > score_en:
        return "id"
    if score_en > score_id:
        return "en"

    # Fallback: prefer English for safety
    return "en"

# ----------------- Safe LLM helpers -----------------
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
                    if isinstance(t, str):
                        buf.append(t)
                if buf:
                    return "/".join(buf).strip()
    except Exception:
        pass
    return None

def safe_generate_json(prompt: str, schema: Dict[str, Any], temp: float = 0.0) -> Dict[str, Any]:
    cfg = GenerationConfig(
        temperature=temp,
        response_mime_type="application/json",
        response_schema=schema,
    )
    # Throttle
    with _GEN_SEM:
        _GEN_QPS.acquire()
        t0 = now_ms()
        resp = _api_call_with_retry(gen_model.generate_content, prompt, generation_config=cfg)
    log(f"[LLM JSON] call completed in {dur_ms(t0):.0f} ms", level="DEBUG")
    try:
        if isinstance(resp.text, str) and resp.text.strip():
            return json.loads(resp.text)
    except Exception:
        pass
    try:
        raw = resp.candidates[0].content.parts[0].text
        return json.loads(raw)
    except Exception as e:
        info = get_finish_info(resp)
        log(f"[LLM JSON parse warning] No JSON content returned. Diagnostics: {info}. Error: {e}", level="WARN")
        try:
            return json.loads("{}")
        except Exception:
            return {}

def safe_generate_text(prompt: str, max_tokens: int, temperature: float = 0.2) -> str:
    cfg = GenerationConfig(temperature=temperature, max_output_tokens=max_tokens)
    # Throttle
    with _GEN_SEM:
        _GEN_QPS.acquire()
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

# ----------------- Agent I0: Intermediate Question Generator -----------------
IQ_PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "intermediate_questions": {
            "type": "array",
            "items": {"type": "string"}
        },
        "rationale": {"type": "string"}
    },
    "required": ["intermediate_questions"]
}

def agent_i0_generate_iqs(query: str, user_lang: str) -> Dict[str, Any]:
    prompt = f"""
You are Agent I0 (Intermediate Question Planner).
Goal: produce a minimal, sequential plan of dependent intermediate questions (IQs) that, when answered in order, solve the user's query.
If decomposition is unnecessary, return exactly a single-item list containing the original query.

Rules:
- The questions are dependent and must be answered in sequence.
- Keep each IQ concise and self-contained for its step.
- Respect the maximum number of steps: {IQ_MAX_STEPS} (truncate if more are proposed).
- Keep the questions in the same language as the user query ("{user_lang}").

Return JSON:
- "intermediate_questions": array of strings (1..{IQ_MAX_STEPS})
- "rationale": optional brief explanation (for diagnostics)

User query:
\"\"\"{query}\"\"\"
"""
    est_tokens = estimate_tokens_for_text(prompt)
    log("\n[Agent I0] Prompt:")
    log(prompt)
    log(f"[Agent I0] Prompt size: {len(prompt)} chars, est_tokens≈{est_tokens}")
    t0 = now_ms()
    data = safe_generate_json(prompt, IQ_PLAN_SCHEMA, temp=0.0)
    took = dur_ms(t0)

    iqs_raw = data.get("intermediate_questions", []) if isinstance(data, dict) else []
    if not isinstance(iqs_raw, list):
        iqs_raw = []
    cleaned: List[str] = []
    seen = set()
    for q in iqs_raw:
        if not isinstance(q, str):
            continue
        q2 = q.strip()
        if not q2 or q2 in seen:
            continue
        seen.add(q2)
        cleaned.append(q2)
        if len(cleaned) >= IQ_MAX_STEPS:
            break
    if not cleaned:
        cleaned = [query.strip()]
    data["intermediate_questions"] = cleaned
    data["rationale"] = data.get("rationale") or ""
    log(f"[Agent I0] Planned IQ count: {len(cleaned)} | {took:.0f} ms")
    log(f"[Agent I0] IQ plan: {cleaned}")
    return data

# ----------------- Agent Q: Query Modifier (enrich next IQ from previous QA) -----------------
MODIFIED_IQ_SCHEMA = {
    "type": "object",
    "properties": {
        "completed_iq": {"type": "string"},
        "note": {"type": "string"}
    },
    "required": ["completed_iq"]
}

def agent_q_modify_next_iq(
    prior_qa_pairs: List[Dict[str, str]],
    next_planned_iq: str,
    user_lang: str
) -> Dict[str, Any]:
    """
    Produce an enriched/completed next IQ leveraging facts from prior answers.
    Input: prior QA and the next planned IQ.
    Output: a single 'completed_iq' ready for retrieval/answering.
    """
    prior_lines = []
    for i, qa in enumerate(prior_qa_pairs, start=1):
        q = (qa.get("question") or "").strip()
        a = (qa.get("answer") or "").strip()
        prior_lines.append(f"[Step {i}] Q: {q}\nA: {a}")
    prior_text = "\n".join(prior_lines) if prior_lines else "(no prior QA; this is the first step)"

    prompt = f"""
You are Agent Q (Query Modifier).
Task: Enrich or complete the next intermediate question using ONLY the information in the previously answered steps.

Guidelines:
- Use facts from prior answers to clarify terms, fill placeholders, and resolve ambiguities in the next question.
- Do NOT change the intent of the planned next IQ; only complete/enrich it as needed.
- Keep the completed IQ concise and in the same language as the user's query ("{user_lang}").

Previously answered steps:
\"\"\"{prior_text}\"\"\"

Next planned IQ:
\"\"\"{next_planned_iq}\"\"\"

Return JSON:
- "completed_iq": string
- "note": (optional) brief explanation for diagnostics
"""
    est_tokens = estimate_tokens_for_text(prompt)
    log("\n[Agent Q] Prompt:")
    log(prompt)
    log(f"[Agent Q] Prompt size: {len(prompt)} chars, est_tokens≈{est_tokens}")
    t0 = now_ms()
    data = safe_generate_json(prompt, MODIFIED_IQ_SCHEMA, temp=0.0)
    took = dur_ms(t0)
    completed_iq = (data.get("completed_iq") or "").strip()
    if not completed_iq:
        completed_iq = next_planned_iq.strip()
        data["note"] = (data.get("note") or "") + " (fallback to planned IQ)"
    log(f"[Agent Q] Completed IQ: {completed_iq} | {took:.0f} ms")
    return {"completed_iq": completed_iq, "note": data.get("note") or ""}

# ----------------- Agent 1: entity/predicate extraction -----------------
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
          "type": {"type": "string"}   # Indonesian type if known, else ""
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

def agent1_extract_entities_predicates(query: str) -> Dict[str, Any]:
    prompt = f"""
You are Agent 1. Task: extract the legal entities and predicates referenced or implied by the user's question.

Output format:
- JSON with:
  - "entities": array of objects with fields {{text, type(optional)}}
  - "predicates": array of strings (Indonesian, snake_case when applicable)
- If entity type is provided, it MUST be one of: {", ".join(LEGAL_ENTITY_TYPES)}.
- Predicates should ideally be one of: {", ".join(LEGAL_PREDICATES)}.

User question:
\"\"\"{query}\"\"\"
"""
    est_tokens = estimate_tokens_for_text(prompt)
    log("\n[Agent 1] Prompt:")
    log(prompt)
    log(f"[Agent 1] Prompt size: {len(prompt)} chars, est_tokens≈{est_tokens}")
    t0 = now_ms()
    data = safe_generate_json(prompt, QUERY_SCHEMA, temp=0.0)
    took = dur_ms(t0)
    if "entities" not in data: data["entities"] = []
    if "predicates" not in data: data["predicates"] = []
    log(f"[Agent 1] Output: entities={ [e.get('text') for e in data['entities']] }, predicates={ data['predicates'] } | {took:.0f} ms")
    return data

# ----------------- Agent 1b: triple extraction from query -----------------
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
                        "properties": {
                            "text": {"type": "string"},
                            "type": {"type": "string"}
                        },
                        "required": ["text"]
                    },
                    "predicate": {"type": "string"},
                    "object": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string"},
                            "type": {"type": "string"}
                        },
                        "required": ["text"]
                    }
                },
                "required": ["subject", "predicate", "object"]
            }
        }
    },
    "required": ["triples"]
}

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

User question:
\"\"\"{query}\"\"\"
"""
    est_tokens = estimate_tokens_for_text(prompt)
    log("\n[Agent 1b] Prompt:")
    log(prompt)
    log(f"[Agent 1b] Prompt size: {len(prompt)} chars, est_tokens≈{est_tokens}")
    t0 = now_ms()
    out = safe_generate_json(prompt, QUERY_TRIPLES_SCHEMA, temp=0.0)
    triples = out.get("triples", []) if isinstance(out, dict) else []
    # sanitize minimal fields
    clean: List[Dict[str, Any]] = []
    for t in triples:
        try:
            s = t.get("subject", {}) or {}
            o = t.get("object", {}) or {}
            p = (t.get("predicate") or "").strip()
            if (s.get("text") and o.get("text") and p):
                clean.append({
                    "subject": {"text": s.get("text","").strip(), "type": (s.get("type") or "").strip()},
                    "predicate": p,
                    "object":  {"text": o.get("text","").strip(), "type": (o.get("type") or "").strip()},
                })
        except Exception:
            continue
    took = dur_ms(t0)
    log(f"[Agent 1b] Extracted query triples: {['{} [{}] {}'.format(x['subject']['text'], x['predicate'], x['object']['text']) for x in clean]} | {took:.0f} ms")
    return clean

def query_triple_to_text(t: Dict[str, Any]) -> str:
    s = ((t.get("subject") or {}).get("text") or "").strip()
    p = (t.get("predicate") or "").strip()
    o = ((t.get("object") or "").get("text") or "").strip() if isinstance(t.get("object"), dict) else ((t.get("object") or "").strip())
    return f"{s} [{p}] {o}"

# ----------------- Neo4j vector search helpers -----------------
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
            if k.lower() in ("q_emb", "embedding", "emb"):
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
    """
    Execute a Cypher query with bounded retries on error.
    Logs attempt start, success, and failure with timestamps, PID, and durations.
    """
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
        log(f"[Neo4j] Attempt {attempts}/{NEO4J_MAX_ATTEMPTS} | qid={qid} | timeout={NEO4J_TX_TIMEOUT_S:.1f}s | Cypher=\"{preview}\" | Params: {param_summary}")
        try:
            if _NEO4J_SEM is not None:
                _NEO4J_SEM.acquire()
            try:
                with driver.session() as session:
                    res = session.run(cypher, **params, timeout=NEO4J_TX_TIMEOUT_S)
                    records = list(res)
            finally:
                if _NEO4J_SEM is not None:
                    _NEO4J_SEM.release()
            took = dur_ms(t0)
            log(f"[Neo4j] Success | qid={qid} | rows={len(records)} | {took:.0f} ms")
            return records
        except Exception as e:
            took = dur_ms(t0)
            last_e = e
            if attempts >= NEO4J_MAX_ATTEMPTS:
                break
            wait_s = _rand_wait_seconds()
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
    try:
        candidates.extend(_vector_query_nodes("document_vec", q_emb, k))
    except Exception as e:
        log(f"[Warn] document_vec query failed: {e}", level="WARN")
    try:
        candidates.extend(_vector_query_nodes("content_vec", q_emb, k))
    except Exception as e:
        log(f"[Warn] content_vec query failed: {e}", level="WARN")
    try:
        candidates.extend(_vector_query_nodes("expression_vec", q_emb, k))
    except Exception as e:
        log(f"[Warn] expression_vec query failed: {e}", level="WARN")

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

# ----------------- Graph expansion (as in LexID) -----------------
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
            if s and s.get("key"): next_keys.add(s.get("key"))
            if o and o.get("key"): next_keys.add(o.get("key"))
            s_id = r.get("s_id");  o_id = r.get("o_id")
            if s_id: next_ids.add(s_id)
            if o_id: next_ids.add(o_id)

        current_ids = next_ids if next_ids else set()
        current_keys = set() if next_ids else next_keys

    return list(triples.values())

# ----------------- Chunk store -----------------
class ChunkStore:
    def __init__(self, root: Path, skip: Set[str]):
        self.root = root
        self.skip = skip
        self._index: Dict[Tuple[str, str], str] = {}
        self._by_chunk: Dict[str, List[Tuple[str, str]]] = {}
        self._loaded_files: Set[Path] = set()
        self._built = False
        self._build_lock = Lock()

    def _build_index(self):
        if self._built:
            return
        with self._build_lock:
            if self._built:
                return
            start = time.monotonic()
            log(f"[ChunkStore] Building index from {self.root}...")
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
                    log(f"[ChunkStore] Loaded {loaded_count} chunks from {pkl.name}")
                except Exception as e:
                    log(f"[ChunkStore] Failed to load or process {pkl.name}: {e}")
                    continue

            elapsed = time.monotonic() - start
            log(f"[ChunkStore] Index built. Total chunks indexed: {total_chunks_indexed} from {len(self._loaded_files)} files in {elapsed:.3f}s.")
            self._built = True

    def get_chunk(self, document_id: Any, chunk_id: Any) -> Optional[str]:
        if not self._built:
            self._build_index()

        doc_id_s = _norm_id(document_id)
        chunk_id_s = _norm_id(chunk_id)

        # 1) Exact match
        val = self._index.get((doc_id_s, chunk_id_s))
        if val is not None:
            log(f"[ChunkStore] HIT exact: doc={doc_id_s} chunk={chunk_id_s} len={len(val)}")
            return val

        # 2) Strip ::part suffix if present and try exact again
        if "::" in chunk_id_s:
            base_id = chunk_id_s.split("::", 1)[0]
            val = self._index.get((doc_id_s, base_id))
            if val is not None:
                log(f"[ChunkStore] HIT base-id: doc={doc_id_s} chunk={chunk_id_s} -> base={base_id} len={len(val)}")
                return val

        # 3) Fallback by chunk_id only (rescue doc_id mismatches)
        matches = self._by_chunk.get(chunk_id_s)
        if matches:
            chosen_doc, chosen_chunk = matches[0]
            val = self._index.get((chosen_doc, chosen_chunk))
            if val is not None:
                note = "" if len(matches) == 1 else f" (warn: chunk_id occurs in {len(matches)} docs; chose doc={chosen_doc})"
                log(f"[ChunkStore] HIT by chunk_id only: requested doc={doc_id_s} chunk={chunk_id_s}; using doc={chosen_doc}{note}. len={len(val)}")
                return val

        log(f"[ChunkStore] MISS: doc={doc_id_s} chunk={chunk_id_s} (no exact/base-id/chunk-id-only match)")
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

# ----------------- New retrieval pipeline pieces -----------------
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
        matches = search_similar_entities_by_embedding(e_emb, k=ENTITY_MATCH_TOP_K)
        keys = [m.get("key") for m in matches if m.get("key")]
        ids  = [m.get("elem_id") for m in matches if m.get("elem_id")]
        all_matched_keys.update(keys)
        all_matched_ids.update(ids)

    if not (all_matched_keys or all_matched_ids):
        log("[EntityRetrieval] No KG entity matches found from query entities.")
        return []

    t0 = now_ms()
    expanded_triples = expand_from_entities(
        list(all_matched_keys),
        hops=ENTITY_SUBGRAPH_HOPS,
        per_hop_limit=ENTITY_SUBGRAPH_PER_HOP_LIMIT,
        entity_elem_ids=list(all_matched_ids) if all_matched_ids else None
    )
    log(f"[EntityRetrieval] Expanded subgraph triples: {len(expanded_triples)} | {dur_ms(t0):.0f} ms")

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
    top = ranked[:SUBGRAPH_TRIPLES_TOP_K]
    log(f"[EntityRetrieval] Selected top-{len(top)} triples from subgraph")
    return top

def triple_centric_retrieval(query_triples: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[List[float]]]:
    """
    For each extracted query triple:
      - embed "s [p] o"
      - search top-K similar KG triples
    Return merged, deduped triples and the list of query triple embeddings used.
    """
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

        matches = search_similar_triples_by_embedding(emb, k=QUERY_TRIPLE_MATCH_TOP_K_PER)
        for m in matches:
            uid = m.get("triple_uid")
            if uid:
                if uid not in triples_map:
                    triples_map[uid] = m
                else:
                    if m.get("score", 0.0) > triples_map[uid].get("score", 0.0):
                        triples_map[uid] = m

    merged = list(triples_map.values())
    log(f"[TripleRetrieval] Collected {len(merged)} unique KG triples across {len(q_trip_embs)} query triple(s)")
    return merged, q_trip_embs

def collect_chunks_for_triples(
    triples: List[Dict[str, Any]],
    chunk_store: ChunkStore
) -> List[Tuple[Tuple[Any, Any], str, Dict[str, Any]]]:
    """
    From triples, gather unique (doc_id, chunk_id) and load texts.
    Returns list of (key_pair, text, triple) for each chunk instance.
    Adds t['_is_quote_fallback']=True when we have to fall back.
    """
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
    top_k: int,
    cand_limit: Optional[int] = None
) -> List[Tuple[Tuple[Any, Any], str, Dict[str, Any], float]]:
    """
    Embed chunk texts and score similarity to whole user query.
    Returns list of records augmented with score, sorted desc.
    """
    limit = cand_limit if isinstance(cand_limit, int) and cand_limit > 0 else CHUNK_RERANK_CAND_LIMIT
    cand = chunk_records[:limit]
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
    took = dur_ms(t0)
    log(f"[ChunkRerank] Scored {len(scored)} candidates | picked top {min(top_k, len(scored))} | {took:.0f} ms")
    return scored[:top_k]

def rerank_triples_by_query_triples(
    triples: List[Dict[str, Any]],
    q_trip_embs: List[List[float]],
    q_emb_fallback: Optional[List[float]],
    top_k: int
) -> List[Dict[str, Any]]:
    """
    Sort triples by mean similarity to all query triple embeddings.
    If no query triple embeddings, fallback to whole-query embedding similarity.
    """
    t0 = now_ms()
    def score(t: Dict[str, Any]) -> float:
        emb = t.get("embedding")
        if q_trip_embs:
            return mean_similarity_to_query_triples(emb, q_trip_embs)
        if q_emb_fallback and isinstance(emb, list):
            return cos_sim(q_emb_fallback, emb)
        return 0.0

    ranked = sorted(triples, key=score, reverse=True)
    took = dur_ms(t0)
    log(f"[TripleRerank] Input={len(triples)} | Output={min(top_k, len(ranked))} | {took:.0f} ms")
    return ranked[:top_k]

def build_combined_context_text(
    triples_ranked: List[Dict[str, Any]],
    chunks_ranked: List[Tuple[Tuple[Any, Any], str, Dict[str, Any], float]]
) -> Tuple[str, str, List[Dict[str, Any]]]:
    """
    Create a readable context text with:
      - Triple summary (top up to 50)
      - Selected chunk texts (annotate when quote fallback is used)
    Returns (context_text, summary_text, list_of_chunk_records_dict)
    """
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

# ----------------- Agent 2 (Intermediate answer) -----------------
def agent2_answer(query_original: str, context: str, guidance: Optional[str], output_lang: str = "id") -> str:
    instructions = (
        "Answer concisely and accurately based strictly on the provided context. "
        "Cite UU/Article references when they are clear. "
        "Respond in the same language as the user's question."
    )

    guidance_text = guidance.strip() if isinstance(guidance, str) and guidance.strip() else "(No additional guidance.)"

    prompt = f"""
You are Agent 2 (Answerer). Task: provide an answer based on the context only.

Core instructions:
{instructions}

Additional guidance (if any):
\"\"\"{guidance_text}\"\"\"

Original question:
\"\"\"{query_original}\"\"\"

Context:
\"\"\"{context}\"\"\"
"""
    est_tokens = estimate_tokens_for_text(prompt)
    log("\n[Agent 2] Prompt:")
    log(prompt)
    log(f"[Agent 2] Prompt size: {len(prompt)} chars, est_tokens≈{est_tokens}")

    t0 = now_ms()
    answer = safe_generate_text(prompt, max_tokens=ANSWER_MAX_TOKENS, temperature=0.2)
    took = dur_ms(t0)
    log(f"[Agent 2] Intermediate answer length={len(answer)} | {took:.0f} ms")
    return answer

# ----------------- Single-pass runner (for any single question) -----------------
def run_single_pass_for_question(
    query_original: str,
    chunk_store: ChunkStore,
    user_lang: Optional[str] = None,
    cand_limit_override: Optional[int] = None,
    guidance: Optional[str] = None
) -> Dict[str, Any]:
    """
    Runs the single-pass GraphRAG (Agents 1 & 2) for a single question.
    Returns dict with intermediate answer and context summary.
    """
    user_lang = user_lang or detect_user_language(query_original)

    # Step 0: Embed whole user question
    t0 = now_ms()
    q_emb_query = embed_text(query_original)
    t_embed = dur_ms(t0)
    log(f"[Step 0] Whole-query embedding in {t_embed:.0f} ms")

    # Step 1: Agent 1 – extract entities/predicates
    t1 = now_ms()
    extraction = agent1_extract_entities_predicates(query_original)
    ents = extraction.get("entities", [])
    preds = extraction.get("predicates", [])
    t_extract = dur_ms(t1)
    log(f"[Step 1] Entity/Predicate extraction done in {t_extract:.0f} ms")

    # Step 1b: Agent 1b – extract triples from question
    t1b = now_ms()
    query_triples = agent1b_extract_query_triples(query_original)
    t_extract_tr = dur_ms(t1b)
    log(f"[Step 1b] Query triple extraction done in {t_extract_tr:.0f} ms")

    # Step 2: Triple-centric retrieval (per query triple)
    t2 = now_ms()
    ctx2_triples, q_trip_embs = triple_centric_retrieval(query_triples)
    t_triple = dur_ms(t2)
    log(f"[Step 2] Triple-centric retrieval in {t_triple:.0f} ms; ctx2_triples={len(ctx2_triples)}, q_trip_embs={len(q_trip_embs)}")

    # Step 3: Entity-centric retrieval
    t3 = now_ms()
    ctx1_triples = entity_centric_retrieval(ents, q_trip_embs=q_trip_embs, q_emb_fallback=q_emb_query)
    t_entity = dur_ms(t3)
    log(f"[Step 3] Entity-centric retrieval in {t_entity:.0f} ms; ctx1_triples={len(ctx1_triples)}")

    # Step 4: Merge contexts, dedupe triples
    t4 = now_ms()
    triple_map: Dict[str, Dict[str, Any]] = {}
    for t in ctx1_triples + ctx2_triples:
        uid = t.get("triple_uid")
        if uid:
            prev = triple_map.get(uid)
            if prev is None or (t.get("score", 0.0) > prev.get("score", 0.0)):
                triple_map[uid] = t
    merged_triples = list(triple_map.values())
    t_merge = dur_ms(t4)
    log(f"[Step 4] Merged triples from contexts: {len(merged_triples)} | {t_merge:.0f} ms")

    # Step 5: Gather chunks and rerank
    t5 = now_ms()
    chunk_records = collect_chunks_for_triples(merged_triples, chunk_store)
    log(f"[Step 5] Collected {len(chunk_records)} chunk candidates (pre-rerank)")
    chunks_ranked = rerank_chunks_by_query(chunk_records, q_emb_query, top_k=MAX_CHUNKS_FINAL, cand_limit=cand_limit_override)
    t_chunks = dur_ms(t5)
    log(f"[Step 5] Chunk rerank done in {t_chunks:.0f} ms; selected {len(chunks_ranked)}")

    # Step 6: Rerank triples
    t6 = now_ms()
    triples_ranked = rerank_triples_by_query_triples(merged_triples, q_trip_embs=q_trip_embs, q_emb_fallback=q_emb_query, top_k=MAX_TRIPLES_FINAL)
    t_rerank = dur_ms(t6)
    log(f"[Step 6] Triple rerank done in {t_rerank:.0f} ms; selected {len(triples_ranked)}")

    # Build combined context
    t_ctx = now_ms()
    context_text, context_summary, _ = build_combined_context_text(triples_ranked, chunks_ranked)
    log(f"[Context] Built in {dur_ms(t_ctx):.0f} ms")
    log("\n[Context summary for this pass]:")
    log(context_summary)

    # Step 7: Agent 2 – Answer
    t7 = now_ms()
    intermediate_answer = agent2_answer(query_original, context_text, guidance=guidance, output_lang=user_lang)
    t_answer = dur_ms(t7)
    log(f"[Step 7] Answer generated in {t_answer:.0f} ms")

    return {
        "question": query_original,
        "answer": intermediate_answer,
        "context_summary": context_summary,
        "counts": {
            "chunks_selected": len(chunks_ranked),
            "triples_selected": len(triples_ranked),
            "chunk_candidates": len(chunk_records)
        },
        "timings_ms": {
            "embed": int(t_embed),
            "extract_entities": int(t_extract),
            "extract_triples": int(t_extract_tr),
            "triple_retrieval": int(t_triple),
            "entity_retrieval": int(t_entity),
            "merge": int(t_merge),
            "chunks_rerank": int(t_chunks),
            "triples_rerank": int(t_rerank),
            "answer": int(t_answer),
        }
    }

# ----------------- Original single-pass GraphRAG (kept for compatibility) -----------------
def agentic_graph_rag(query_original: str) -> Dict[str, Any]:
    """
    Backward-compatible: runs a single-pass (Agents 1 & 2) GraphRAG for the given query and returns the answer.
    """
    global _LOGGER
    ts_name = make_timestamp_name()
    log_file = Path.cwd() / f"{ts_name}.txt"
    _LOGGER = FileLogger(log_file, also_console=True)
    set_log_context(None)

    t_all = now_ms()
    try:
        log("=== Agentic GraphRAG run started ===")
        log(f"Process info: pid={_pid()}")
        log(f"Log file: {log_file}")
        log(f"Original Query: {query_original}")
        log(f"Parameters: ENTITY_MATCH_TOP_K={ENTITY_MATCH_TOP_K}, ENTITY_SUBGRAPH_HOPS={ENTITY_SUBGRAPH_HOPS}, "
            f"ENTITY_SUBGRAPH_PER_HOP_LIMIT={ENTITY_SUBGRAPH_PER_HOP_LIMIT}, SUBGRAPH_TRIPLES_TOP_K={SUBGRAPH_TRIPLES_TOP_K}, "
            f"QUERY_TRIPLE_MATCH_TOP_K_PER={QUERY_TRIPLE_MATCH_TOP_K_PER}, MAX_TRIPLES_FINAL={MAX_TRIPLES_FINAL}, "
            f"MAX_CHUNKS_FINAL={MAX_CHUNKS_FINAL}, CHUNK_RERANK_CAND_LIMIT={CHUNK_RERANK_CAND_LIMIT}, "
            f"ANSWER_MAX_TOKENS={ANSWER_MAX_TOKENS}, MAX_ITERS={MAX_ITERS}, OUTPUT_LANG={OUTPUT_LANG}, IQ_MAX_STEPS={IQ_MAX_STEPS}")
        log(f"LLM limits: EMBED(max_conc={LLM_EMBED_MAX_CONCURRENCY}, qps={LLM_EMBED_QPS}), GEN(max_conc={LLM_GEN_MAX_CONCURRENCY}, qps={LLM_GEN_QPS})")
        log(f"Neo4j retry/timeout: NEO4J_MAX_ATTEMPTS={NEO4J_MAX_ATTEMPTS}, NEO4J_TX_TIMEOUT_S={NEO4J_TX_TIMEOUT_S:.1f}, NEO4J_MAX_CONCURRENCY={NEO4J_MAX_CONCURRENCY or 'unlimited'}")
        log("Mode: Single-pass (Agents 1 & 2 only)")

        chunk_store = ChunkStore(LANGCHAIN_DIR, set(SKIP_FILES))

        user_lang = detect_user_language(query_original)
        log(f"[Language] Detected user language: {user_lang}")

        result = run_single_pass_for_question(
            query_original=query_original,
            chunk_store=chunk_store,
            user_lang=user_lang,
            cand_limit_override=None,
            guidance=None
        )

        final_answer = result["answer"]

        total_ms = dur_ms(t_all)
        log("\n=== Agentic GraphRAG summary ===")
        log(f"- Iterations used: 1")
        log(f"- Total runtime: {total_ms:.0f} ms")
        log("\n=== Final Answer ===")
        log(final_answer)
        log(f"\nLogs saved to: {log_file}")

        return {
            "final_answer": final_answer,
            "log_file": str(log_file),
            "iterations": 1,
            "iq_plan": [query_original],
            "iq_completed": [query_original],
            "qa_pairs": [{"question": query_original, "answer": final_answer, "context_summary": result["context_summary"]}],
        }
    finally:
        if _LOGGER is not None:
            _LOGGER.close()
            _LOGGER = None

# ----------------- New Orchestrator: Intermediate Questions (I0 → loop of Q → 1&1b → 2) -----------------
def agentic_graph_rag_iq(query_original: str) -> Dict[str, Any]:
    """
    Full agentic pipeline with sequential Intermediate Questions:
      - Agent I0: generate sequential IQ plan (capped by IQ_MAX_STEPS)
      - For each IQ in order:
          - If step > 1: Agent Q enriches the planned IQ using prior QA pairs
          - Run single-pass GraphRAG (Agents 1 & 2) to answer current IQ
      - Final answer = the answer of the last IQ (no final aggregator)
    """
    global _LOGGER
    ts_name = make_timestamp_name()
    log_file = Path.cwd() / f"{ts_name}.txt"
    _LOGGER = FileLogger(log_file, also_console=True)
    set_log_context(None)

    t_all = now_ms()
    try:
        log("=== Agentic GraphRAG (Sequential Intermediate Questions) run started ===")
        log(f"Process info: pid={_pid()}")
        log(f"Log file: {log_file}")
        log(f"Original Query: {query_original}")
        log(f"Parameters: IQ_MAX_STEPS={IQ_MAX_STEPS} (hardcoded), "
            f"ENTITY_MATCH_TOP_K={ENTITY_MATCH_TOP_K}, ENTITY_SUBGRAPH_HOPS={ENTITY_SUBGRAPH_HOPS}, "
            f"ENTITY_SUBGRAPH_PER_HOP_LIMIT={ENTITY_SUBGRAPH_PER_HOP_LIMIT}, SUBGRAPH_TRIPLES_TOP_K={SUBGRAPH_TRIPLES_TOP_K}, "
            f"QUERY_TRIPLE_MATCH_TOP_K_PER={QUERY_TRIPLE_MATCH_TOP_K_PER}, MAX_TRIPLES_FINAL={MAX_TRIPLES_FINAL}, "
            f"MAX_CHUNKS_FINAL={MAX_CHUNKS_FINAL}, CHUNK_RERANK_CAND_LIMIT={CHUNK_RERANK_CAND_LIMIT}, "
            f"ANSWER_MAX_TOKENS={ANSWER_MAX_TOKENS}, MAX_ITERS={MAX_ITERS}")
        log(f"LLM limits: EMBED(max_conc={LLM_EMBED_MAX_CONCURRENCY}, qps={LLM_EMBED_QPS}), GEN(max_conc={LLM_GEN_MAX_CONCURRENCY}, qps={LLM_GEN_QPS})")
        log(f"Neo4j: NEO4J_MAX_ATTEMPTS={NEO4J_MAX_ATTEMPTS}, NEO4J_TX_TIMEOUT_S={NEO4J_TX_TIMEOUT_S:.1f}, NEO4J_MAX_CONCURRENCY={NEO4J_MAX_CONCURRENCY or 'unlimited'}")
        log("Mode: Sequential IQs (I0 → (Q → 1&1b → 2)*N; no final aggregator)")

        # Detect language once
        user_lang = detect_user_language(query_original)
        log(f"[Language] Detected user language: {user_lang}")

        # Agent I0: plan IQs
        t0 = now_ms()
        plan = agent_i0_generate_iqs(query_original, user_lang)
        planned_iqs: List[str] = plan.get("intermediate_questions", []) or [query_original.strip()]
        planned_iqs = planned_iqs[:IQ_MAX_STEPS]
        log(f"[Agent I0] Final IQ plan (used): {planned_iqs}")
        t0_ms = dur_ms(t0)

        # Build ChunkStore once
        chunk_store = ChunkStore(LANGCHAIN_DIR, set(SKIP_FILES))

        # Storage for results
        completed_iqs: List[str] = []
        qa_pairs: List[Dict[str, str]] = []
        per_step_results: List[Dict[str, Any]] = []

        # Sequential loop over IQs
        total_steps = len(planned_iqs)
        for idx, planned_iq in enumerate(planned_iqs, start=1):
            set_log_context(f"{idx}/{total_steps}")
            log(f"--- IQ {idx}/{total_steps} START ---")
            try:
                # Agent Q for steps > 1
                if idx == 1:
                    completed_iq = planned_iq
                    q_note = "(first step; no enrichment)"
                else:
                    mod = agent_q_modify_next_iq(
                        prior_qa_pairs=[{"question": p.get("question") or completed_iqs[i], "answer": qa_pairs[i]["answer"]} if isinstance(p, dict) else {"question": completed_iqs[i], "answer": qa_pairs[i]["answer"]}
                                        for i, p in enumerate(per_step_results)],
                        next_planned_iq=planned_iq,
                        user_lang=user_lang
                    )
                    completed_iq = (mod.get("completed_iq") or planned_iq).strip()
                    q_note = mod.get("note") or ""

                # Answer the (completed) IQ
                result = run_single_pass_for_question(
                    query_original=completed_iq,
                    chunk_store=chunk_store,
                    user_lang=user_lang,
                    cand_limit_override=None,
                    guidance=None
                )

                # Persist step artifacts
                completed_iqs.append(completed_iq)
                qa_pairs.append({
                    "question": completed_iq,
                    "answer": result["answer"],
                    "context_summary": result["context_summary"]
                })
                per_step_results.append({
                    "planned_iq": planned_iq,
                    "completed_iq": completed_iq,
                    "answer": result["answer"],
                    "context_summary": result["context_summary"],
                    "note": q_note,
                    "diagnostics": {
                        "counts": result.get("counts", {}),
                        "timings_ms": result.get("timings_ms", {})
                    }
                })
                log(f"--- IQ {idx}/{total_steps} DONE ---")
            except Exception as e:
                log(f"[IQ {idx}] Error: {e}", level="WARN")
                completed_iq = planned_iq
                completed_iqs.append(completed_iq)
                qa_pairs.append({
                    "question": completed_iq,
                    "answer": f"(No evidence found or step failed: {e})",
                    "context_summary": ""
                })
                per_step_results.append({
                    "planned_iq": planned_iq,
                    "completed_iq": completed_iq,
                    "answer": f"(No evidence found or step failed: {e})",
                    "context_summary": "",
                    "note": "(failure)",
                    "diagnostics": {}
                })
            finally:
                set_log_context(None)

        # Final answer is the last IQ's answer
        final_answer = qa_pairs[-1]["answer"] if qa_pairs else "(No answer produced)"

        # Summary
        total_ms = dur_ms(t_all)
        log("\n=== Agentic GraphRAG (IQ) summary ===")
        log(f"- Agent I0 time: {t0_ms:.0f} ms")
        log(f"- Number of IQs executed: {len(planned_iqs)}")
        log(f"- Total runtime: {total_ms:.0f} ms")
        log("\n=== Final Answer (from last IQ) ===")
        log(final_answer)
        log(f"\nLogs saved to: {log_file}")

        return {
            "final_answer": final_answer,
            "log_file": str(log_file),
            "iterations": len(planned_iqs),
            "iq_plan": planned_iqs,
            "iq_completed": completed_iqs,
            "qa_pairs": qa_pairs,
            "per_step_results": per_step_results
        }
    finally:
        if _LOGGER is not None:
            _LOGGER.close()
            _LOGGER = None

# ----------------- Main -----------------
if __name__ == "__main__":
    try:
        user_query = input("Enter your query: ").strip()
        if not user_query:
            print("Empty query. Exiting.")
        else:
            # Run the new IQ-based agentic pipeline by default
            agentic_graph_rag_iq(user_query)
    finally:
        try:
            driver.close()
        except Exception:
            pass