# lexidkg_graphrag_agentic.py
# Agentic GraphRAG with Context Judge + Query Modifier loop:
# - Agent CJ: Context Judge (assesses sufficiency/relevance of retrieved context; provides problem + suggestion)
# - Agent QM: Query Modifier (rewrites the query based on CJ feedback and prior query-feedback history)
# - Agents 1 & 1b: Entity/predicate and triple extraction (as before)
# - Agent 2: Answerer (answers strictly from retrieved context)
# Orchestration:
# - Iterative loop (hardcoded max iterations). Each iteration:
#   1) Retrieval to build context (no answering yet)
#   2) If iteration cap reached → skip CJ/QM and answer
#   3) Else run CJ; if sufficient → answer; else run QM → next iteration with modified query
# - Maintains history of query-feedback pairs for CJ/QM inputs
# Plus:
# - Global LLM rate limiting (concurrency + QPS)
# - Thread-safe logging with per-iteration tags
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

# ----------------- Retrieval/agent parameters -----------------
# Entity-centric path
ENTITY_MATCH_TOP_K = 15                 # top similar KG entities per extracted query entity
ENTITY_SUBGRAPH_HOPS = 5               # hop-depth for subgraph expansion from matched entities
ENTITY_SUBGRAPH_PER_HOP_LIMIT = 2000   # per-hop expansion limit
SUBGRAPH_TRIPLES_TOP_K = 30            # top triples selected from subgraph after triple-vs-triple similarity

# Triple-centric path
QUERY_TRIPLE_MATCH_TOP_K_PER = 20      # per query-triple, top similar KG triples

# Final context combination and reranking
MAX_TRIPLES_FINAL = 60                 # final number of triples after reranking
MAX_CHUNKS_FINAL = 40                  # final number of chunks after reranking
CHUNK_RERANK_CAND_LIMIT = 10000000            # cap chunk candidates before embedding/reranking to control cost

# Agent loop and output
ANSWER_MAX_TOKENS = 4096

# Language setting
OUTPUT_LANG = "id"  # retained for compatibility; we auto-detect based on query

# ----------------- New: Judge + Modifier loop controls -----------------
# Hardcoded maximum number of judge/modifier iterations (per requirement)
MAX_JUDGE_ITERS = 4

# Max characters of built context text fed to the Context Judge (to control tokens)
JUDGE_CONTEXT_MAX_CHARS = int(os.getenv("JUDGE_CONTEXT_MAX_CHARS", "1000000000000000000000000000000"))

# ----------------- Global LLM throttling (concurrency + QPS) -----------------
LLM_EMBED_MAX_CONCURRENCY = max(1, int(os.getenv("LLM_EMBED_MAX_CONCURRENCY", "165")))
LLM_EMBED_QPS = float(os.getenv("LLM_EMBED_QPS", "165.0"))   # average embed calls per second (global)
LLM_GEN_MAX_CONCURRENCY   = max(1, int(os.getenv("LLM_GEN_MAX_CONCURRENCY", "100")))
LLM_GEN_QPS   = float(os.getenv("LLM_GEN_QPS", "1.0"))     # average generation calls per second (global)

# Embedding cache cap
CACHE_EMBED_MAX_ITEMS = int(os.getenv("CACHE_EMBED_MAX_ITEMS", "200000"))

# ----------------- Initialize SDKs -----------------
genai.configure(api_key=GOOGLE_API_KEY)
gen_model = genai.GenerativeModel(GEN_MODEL)

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

# ----------------- Logger -----------------
_LOGGER: Optional["FileLogger"] = None
_LOG_TL = threading.local()  # thread-local for iteration tags

def set_log_context(tag: Optional[str]):
    setattr(_LOG_TL, "iter_tag", tag or None)

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
    tag = getattr(_LOG_TL, "iter_tag", None)
    tag_part = f" [iter={tag}]" if tag else ""
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
                    return "\n".join(buf).strip()
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

# ----------------- Pipeline brief for CJ/QM prompts -----------------
PIPELINE_BRIEF = """
GraphRAG pipeline summary (retrieval-centric):
1) Agent 1 extracts legal entities/predicates from the query (e.g., UU numbers, Pasal/Article, verbs like mewajibkan/melarang).
2) Agent 1b extracts query triples: subject [predicate] object.
3) Triple-centric retrieval: embed "s [p] o" and query a triple_vec index to find similar knowledge-graph triples.
4) Entity-centric retrieval: embed key entities, match similar KG entities via vector indexes, expand a 1-hop subgraph to collect related triples.
5) Merge triples, rerank by similarity to the query/triples.
6) Collect candidate document chunks for those triples (doc_id/chunk_id), then embed and rerank chunks by similarity to the whole query.
7) Answerer (Agent 2) must answer strictly from the selected chunks (context); if context lacks the needed specifics, the answer will be weak.
Common retrieval issues:
- Missing precise statute identifiers (UU number/year, Pasal, Ayat).
- Ambiguous entities/terms; need aliases, jurisdiction, or timeframe.
- Predicate/relationship too vague; specify obligations, prohibitions, definitions, amendments, effective dates.
- Scope too broad; constrain to a sector/institution or timeframe.
Your feedback/modification should target the query text so that entity/triple extraction and vector matching return more precise triples and chunks.
""".strip()

# ----------------- Agent CJ: Context Judge -----------------
CJ_SCHEMA = {
    "type": "object",
    "properties": {
        "sufficient": {"type": "boolean"},
        "problem": {"type": "string"},
        "suggestion": {"type": "string"},
        "notes": {"type": "string"}
    },
    "required": ["sufficient", "problem", "suggestion"]
}

def build_judge_context_excerpt(context_text: str, max_chars: int) -> str:
    if not isinstance(context_text, str):
        return ""
    if len(context_text) <= max_chars:
        return context_text
    head = context_text[: max_chars - 200]
    tail = "\n...[truncated]..."
    return head + tail

def agent_cj_context_judge(
    current_query: str,
    context_excerpt: str,
    prior_query_feedback: List[Dict[str, Any]],
    user_lang: str
) -> Dict[str, Any]:
    history_str = json.dumps(prior_query_feedback, ensure_ascii=False, indent=2) if prior_query_feedback else "[]"
    prompt = f"""
You are Agent CJ (Context Judge).
Task: Assess whether the retrieved context is sufficient and relevant to answer the current query within this GraphRAG pipeline.

Pipeline brief:
{PIPELINE_BRIEF}

Rules:
- If the provided context likely supports a precise, well-grounded answer, mark "sufficient": true.
- If insufficient, diagnose the concrete root problem ("problem") and propose an actionable "suggestion" to improve the query for the next retrieval iteration.
- Keep output in the same language as the user's query ("{user_lang}").

Current query:
\"\"\"{current_query}\"\"\"

Retrieved context (summary + top chunk snippets, truncated for length):
\"\"\"{context_excerpt}\"\"\"

Prior query–feedback history (latest last):
{history_str}

Return JSON:
- "sufficient": boolean
- "problem": short diagnosis
- "suggestion": concrete improvement(s) to the query text
- "notes": optional
"""
    est_tokens = estimate_tokens_for_text(prompt)
    log("\n[Agent CJ] Prompt:")
    log(prompt)
    log(f"[Agent CJ] Prompt size: {len(prompt)} chars, est_tokens≈{est_tokens}")
    t0 = now_ms()
    out = safe_generate_json(prompt, CJ_SCHEMA, temp=0.0)
    took = dur_ms(t0)
    sufficient = bool(out.get("sufficient")) if isinstance(out, dict) else False
    problem = (out.get("problem") or "").strip() if isinstance(out, dict) else ""
    suggestion = (out.get("suggestion") or "").strip() if isinstance(out, dict) else ""
    notes = (out.get("notes") or "").strip() if isinstance(out, dict) else ""
    log(f"[Agent CJ] Verdict: sufficient={sufficient} | problem='{problem[:120]}' | suggestion='{suggestion[:120]}' | {took:.0f} ms")
    return {"sufficient": sufficient, "problem": problem, "suggestion": suggestion, "notes": notes}

# ----------------- Agent QM: Query Modifier -----------------
QM_SCHEMA = {
    "type": "object",
    "properties": {
        "modified_query": {"type": "string"},
        "rationale": {"type": "string"}
    },
    "required": ["modified_query"]
}

def agent_qm_modify_query(
    current_query: str,
    feedback_problem: str,
    feedback_suggestion: str,
    prior_query_feedback: List[Dict[str, Any]],
    user_lang: str
) -> Dict[str, Any]:
    history_str = json.dumps(prior_query_feedback, ensure_ascii=False, indent=2) if prior_query_feedback else "[]"
    prompt = f"""
You are Agent QM (Query Modifier).
Task: Rewrite the current query to improve retrieval quality for the GraphRAG pipeline, using the judge's feedback and prior history.

Pipeline brief:
{PIPELINE_BRIEF}

Guidelines:
- Preserve the user's original intent and language ("{user_lang}").
- Apply the judge's suggestion(s) concretely to the query text (e.g., add UU number/year, Pasal/Ayat, scope, relation verbs, timeframe, aliases).
- Produce ONE improved query string; do not include explanations in the query text.

Current query:
\"\"\"{current_query}\"\"\"

Context Judge feedback:
- Problem: {feedback_problem}
- Suggestion: {feedback_suggestion}

Prior query–feedback history (latest last):
{history_str}

Return JSON:
- "modified_query": string (only the improved query, same language and intent)
- "rationale": (optional) short explanation for logs
"""
    est_tokens = estimate_tokens_for_text(prompt)
    log("\n[Agent QM] Prompt:")
    log(prompt)
    log(f"[Agent QM] Prompt size: {len(prompt)} chars, est_tokens≈{est_tokens}")
    t0 = now_ms()
    out = safe_generate_json(prompt, QM_SCHEMA, temp=0.0)
    took = dur_ms(t0)
    modified_query = (out.get("modified_query") or "").strip() if isinstance(out, dict) else ""
    rationale = (out.get("rationale") or "").strip() if isinstance(out, dict) else ""
    if not modified_query:
        modified_query = current_query
        rationale = (rationale + " (fallback to current query)").strip()
    log(f"[Agent QM] Modified query: '{modified_query}' | {took:.0f} ms")
    return {"modified_query": modified_query, "rationale": rationale}

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

# ----------------- Retrieval pipeline (reusable) -----------------
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

# ----------------- Agent 2 (Answerer) -----------------
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

# ----------------- Retrieval-only runner -----------------
def run_retrieval_for_query(
    query_original: str,
    chunk_store: ChunkStore,
    user_lang: Optional[str] = None,
    cand_limit_override: Optional[int] = None
) -> Dict[str, Any]:
    """
    Runs retrieval and context building (no answering).
    Returns dict with context_text, context_summary, diagnostics, and some counts/timings.
    """
    user_lang = user_lang or detect_user_language(query_original)

    # Step 0: Embed whole query
    t0 = now_ms()
    q_emb_query = embed_text(query_original)
    t_embed = dur_ms(t0)
    log(f"[Retrieval] Step 0 (embed) in {t_embed:.0f} ms")

    # Step 1: Agent 1 – extract entities/predicates
    t1 = now_ms()
    extraction = agent1_extract_entities_predicates(query_original)
    ents = extraction.get("entities", [])
    preds = extraction.get("predicates", [])
    t_extract = dur_ms(t1)
    log(f"[Retrieval] Step 1 (entities/predicates) in {t_extract:.0f} ms")

    # Step 1b: Agent 1b – extract triples from query
    t1b = now_ms()
    query_triples = agent1b_extract_query_triples(query_original)
    t_extract_tr = dur_ms(t1b)
    log(f"[Retrieval] Step 1b (query triples) in {t_extract_tr:.0f} ms")

    # Step 2: Triple-centric retrieval
    t2 = now_ms()
    ctx2_triples, q_trip_embs = triple_centric_retrieval(query_triples)
    t_triple = dur_ms(t2)
    log(f"[Retrieval] Step 2 (triple-centric) in {t_triple:.0f} ms; triples={len(ctx2_triples)}")

    # Step 3: Entity-centric retrieval
    t3 = now_ms()
    ctx1_triples = entity_centric_retrieval(ents, q_trip_embs=q_trip_embs, q_emb_fallback=q_emb_query)
    t_entity = dur_ms(t3)
    log(f"[Retrieval] Step 3 (entity-centric) in {t_entity:.0f} ms; triples={len(ctx1_triples)}")

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
    log(f"[Retrieval] Step 4 (merge triples) in {t_merge:.0f} ms; merged={len(merged_triples)}")

    # Step 5: Gather chunks and rerank
    t5 = now_ms()
    chunk_records = collect_chunks_for_triples(merged_triples, chunk_store)
    log(f"[Retrieval] Step 5a (collect chunks) candidates={len(chunk_records)}")
    chunks_ranked = rerank_chunks_by_query(
        chunk_records, q_emb_query, top_k=MAX_CHUNKS_FINAL, cand_limit=cand_limit_override
    )
    t_chunks = dur_ms(t5)
    log(f"[Retrieval] Step 5b (rerank chunks) in {t_chunks:.0f} ms; selected={len(chunks_ranked)}")

    # Step 6: Rerank triples
    t6 = now_ms()
    triples_ranked = rerank_triples_by_query_triples(
        merged_triples, q_trip_embs=q_trip_embs, q_emb_fallback=q_emb_query, top_k=MAX_TRIPLES_FINAL
    )
    t_rerank = dur_ms(t6)
    log(f"[Retrieval] Step 6 (rerank triples) in {t_rerank:.0f} ms; selected={len(triples_ranked)}")

    # Build combined context
    t_ctx = now_ms()
    context_text, context_summary, _ = build_combined_context_text(triples_ranked, chunks_ranked)
    t_ctx_build = dur_ms(t_ctx)
    log(f"[Retrieval] Context built in {t_ctx_build:.0f} ms")

    diagnostics = {
        "timings_ms": {
            "embed": int(t_embed),
            "extract_entities": int(t_extract),
            "extract_triples": int(t_extract_tr),
            "triple_retrieval": int(t_triple),
            "entity_retrieval": int(t_entity),
            "merge_triples": int(t_merge),
            "chunks_rerank": int(t_chunks),
            "triples_rerank": int(t_rerank),
            "context_build": int(t_ctx_build),
        },
        "counts": {
            "ctx2_triples": len(ctx2_triples),
            "ctx1_triples": len(ctx1_triples),
            "merged_triples": len(merged_triples),
            "chunk_candidates": len(chunk_records),
            "chunks_selected": len(chunks_ranked),
            "triples_selected": len(triples_ranked),
        }
    }

    return {
        "context_text": context_text,
        "context_summary": context_summary,
        "diagnostics": diagnostics
    }

# ----------------- Single-pass runner (compatibility) -----------------
def run_single_pass_for_question(
    query_original: str,
    chunk_store: ChunkStore,
    user_lang: Optional[str] = None,
    cand_limit_override: Optional[int] = None,
    guidance: Optional[str] = None
) -> Dict[str, Any]:
    user_lang = user_lang or detect_user_language(query_original)
    r = run_retrieval_for_query(
        query_original=query_original,
        chunk_store=chunk_store,
        user_lang=user_lang,
        cand_limit_override=cand_limit_override
    )
    context_text = r["context_text"]
    context_summary = r["context_summary"]

    t_ans = now_ms()
    intermediate_answer = agent2_answer(query_original, context_text, guidance=guidance, output_lang=user_lang)
    t_ans_ms = dur_ms(t_ans)
    log(f"[SinglePass] Answered in {t_ans_ms:.0f} ms")

    return {
        "question": query_original,
        "answer": intermediate_answer,
        "context_summary": context_summary,
        "counts": r["diagnostics"]["counts"],
        "timings_ms": r["diagnostics"]["timings_ms"]
    }

# ----------------- Orchestrator: Judge + Modifier loop -----------------
def agentic_graph_rag_judge(query_original: str) -> Dict[str, Any]:
    """
    Full agentic pipeline with Context Judge and Query Modifier:
      - Iteratively retrieve context for current query.
      - If at cap (MAX_JUDGE_ITERS): answer with current context.
      - Else ask Context Judge:
          - If sufficient: answer now.
          - Else pass feedback to Query Modifier to rewrite the query; continue loop.
      - Maintains history of query–feedback pairs.
    """
    global _LOGGER
    ts_name = make_timestamp_name()
    log_file = Path.cwd() / f"{ts_name}.txt"
    _LOGGER = FileLogger(log_file, also_console=True)
    set_log_context(None)

    t_all = now_ms()
    try:
        log("=== Agentic GraphRAG (Context Judge + Query Modifier) run started ===")
        log(f"Process info: pid={_pid()}")
        log(f"Log file: {log_file}")
        log(f"Original Query: {query_original}")
        log(f"Parameters: MAX_JUDGE_ITERS={MAX_JUDGE_ITERS} (hardcoded), "
            f"ENTITY_MATCH_TOP_K={ENTITY_MATCH_TOP_K}, ENTITY_SUBGRAPH_HOPS={ENTITY_SUBGRAPH_HOPS}, "
            f"ENTITY_SUBGRAPH_PER_HOP_LIMIT={ENTITY_SUBGRAPH_PER_HOP_LIMIT}, SUBGRAPH_TRIPLES_TOP_K={SUBGRAPH_TRIPLES_TOP_K}, "
            f"QUERY_TRIPLE_MATCH_TOP_K_PER={QUERY_TRIPLE_MATCH_TOP_K_PER}, MAX_TRIPLES_FINAL={MAX_TRIPLES_FINAL}, "
            f"MAX_CHUNKS_FINAL={MAX_CHUNKS_FINAL}, CHUNK_RERANK_CAND_LIMIT={CHUNK_RERANK_CAND_LIMIT}, "
            f"ANSWER_MAX_TOKENS={ANSWER_MAX_TOKENS}")
        log(f"LLM limits: EMBED(max_conc={LLM_EMBED_MAX_CONCURRENCY}, qps={LLM_EMBED_QPS}), GEN(max_conc={LLM_GEN_MAX_CONCURRENCY}, qps={LLM_GEN_QPS})")
        log(f"Neo4j: NEO4J_MAX_ATTEMPTS={NEO4J_MAX_ATTEMPTS}, NEO4J_TX_TIMEOUT_S={NEO4J_TX_TIMEOUT_S:.1f}, NEO4J_MAX_CONCURRENCY={NEO4J_MAX_CONCURRENCY or 'unlimited'}")
        log("Mode: Judge loop (retrieve → judge → modify or answer) with hard iteration cap")

        # Detect language once
        user_lang = detect_user_language(query_original)
        log(f"[Language] Detected user language: {user_lang}")

        # Build ChunkStore once
        chunk_store = ChunkStore(LANGCHAIN_DIR, set(SKIP_FILES))

        # History of query-feedback pairs
        query_feedback_history: List[Dict[str, Any]] = []

        # Iterative loop
        current_query = query_original
        final_answer = ""
        per_iteration: List[Dict[str, Any]] = []

        for i in range(1, MAX_JUDGE_ITERS + 1):
            set_log_context(f"{i}/{MAX_JUDGE_ITERS}")
            log(f"--- Iteration {i}/{MAX_JUDGE_ITERS} START ---")
            iter_start = now_ms()

            # Retrieval-only
            retr = run_retrieval_for_query(
                query_original=current_query,
                chunk_store=chunk_store,
                user_lang=user_lang,
                cand_limit_override=None
            )
            context_text = retr["context_text"]
            context_summary = retr["context_summary"]
            diagnostics = retr["diagnostics"]

            # If at iteration cap, answer regardless of CJ verdict
            if i >= MAX_JUDGE_ITERS:
                log("[Loop] At iteration cap; proceeding to answer regardless of CJ verdict.")
                answer = agent2_answer(current_query, context_text, guidance=None, output_lang=user_lang)
                final_answer = answer
                per_iteration.append({
                    "query": current_query,
                    "judge": {"skipped_at_cap": True},
                    "modified_query": None,
                    "context_summary": context_summary,
                    "retrieval_diagnostics": diagnostics,
                    "answer": answer
                })
                log(f"--- Iteration {i}/{MAX_JUDGE_ITERS} DONE (answered at cap) ---")
                break

            # Build a truncated excerpt of context for the judge
            excerpt = build_judge_context_excerpt(context_text, JUDGE_CONTEXT_MAX_CHARS)

            # Context Judge
            cj = agent_cj_context_judge(
                current_query=current_query,
                context_excerpt=excerpt,
                prior_query_feedback=query_feedback_history,
                user_lang=user_lang
            )
            sufficient = bool(cj.get("sufficient"))
            problem = (cj.get("problem") or "").strip()
            suggestion = (cj.get("suggestion") or "").strip()
            notes = (cj.get("notes") or "").strip()

            if sufficient:
                log("[Loop] Context sufficient; proceeding to answer.")
                answer = agent2_answer(current_query, context_text, guidance=None, output_lang=user_lang)
                final_answer = answer
                per_iteration.append({
                    "query": current_query,
                    "judge": {"sufficient": True, "problem": problem, "suggestion": suggestion, "notes": notes},
                    "modified_query": None,
                    "context_summary": context_summary,
                    "retrieval_diagnostics": diagnostics,
                    "answer": answer
                })
                log(f"--- Iteration {i}/{MAX_JUDGE_ITERS} DONE (answered) ---")
                break
            else:
                log("[Loop] Context insufficient; invoking Query Modifier.")
                qm = agent_qm_modify_query(
                    current_query=current_query,
                    feedback_problem=problem,
                    feedback_suggestion=suggestion,
                    prior_query_feedback=query_feedback_history,
                    user_lang=user_lang
                )
                modified_query = (qm.get("modified_query") or current_query).strip()
                rationale = (qm.get("rationale") or "").strip()

                # Record history pair
                query_feedback_history.append({
                    "query": current_query,
                    "feedback": {"problem": problem, "suggestion": suggestion},
                    "modified_query": modified_query
                })

                per_iteration.append({
                    "query": current_query,
                    "judge": {"sufficient": False, "problem": problem, "suggestion": suggestion, "notes": notes},
                    "modified_query": modified_query,
                    "modifier_rationale": rationale,
                    "context_summary": context_summary,
                    "retrieval_diagnostics": diagnostics
                })

                # Prepare next iteration
                if modified_query == current_query:
                    log("[Loop] Modified query identical to current; continuing anyway due to cap protection.", level="WARN")
                current_query = modified_query
                dur_iter = dur_ms(iter_start)
                log(f"--- Iteration {i}/{MAX_JUDGE_ITERS} DONE | {dur_iter:.0f} ms ---")
                # continue loop

        # If loop ended without breaking via answer (edge case)
        if not final_answer:
            set_log_context(None)
            log("[Loop] No answer produced during iterations; answering with last known context as fallback.", level="WARN")
            # Re-run retrieval for last current_query to get context (if needed)
            retr = run_retrieval_for_query(
                query_original=current_query,
                chunk_store=chunk_store,
                user_lang=user_lang,
                cand_limit_override=None
            )
            final_answer = agent2_answer(current_query, retr["context_text"], guidance=None, output_lang=user_lang)
            per_iteration.append({
                "query": current_query,
                "judge": {"forced_fallback": True},
                "modified_query": None,
                "context_summary": retr["context_summary"],
                "retrieval_diagnostics": retr["diagnostics"],
                "answer": final_answer
            })

        # Summary
        total_ms = dur_ms(t_all)
        set_log_context(None)
        log("\n=== Agentic GraphRAG (Judge loop) summary ===")
        log(f"- Iterations used: {len(per_iteration)} (cap={MAX_JUDGE_ITERS})")
        log(f"- Total runtime: {total_ms:.0f} ms")
        log("\n=== Final Answer ===")
        log(final_answer)
        log(f"\nLogs saved to: {log_file}")

        return {
            "final_answer": final_answer,
            "log_file": str(log_file),
            "iterations_used": len(per_iteration),
            "query_feedback_history": query_feedback_history,
            "per_iteration": per_iteration
        }
    finally:
        if _LOGGER is not None:
            _LOGGER.close()
            _LOGGER = None

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
            f"ANSWER_MAX_TOKENS={ANSWER_MAX_TOKENS}, MAX_JUDGE_ITERS={MAX_JUDGE_ITERS}")
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
            "iterations_used": 1,
            "per_iteration": [result]
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
            # Run the new Judge+Modifier agentic pipeline by default
            agentic_graph_rag_judge(user_query)
    finally:
        try:
            driver.close()
        except Exception:
            pass