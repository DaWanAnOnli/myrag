#!/usr/bin/env python3
# multi_agent.py
# Self-contained multi-agent RAG with a Query Decomposer + Aggregator:
# - Query Decomposer Agent:
#     Decides how to split the user's question into sub-queries for GraphRAG and/or NaiveRAG:
#       1) Route different aspects to different pipelines.
#       2) Prefer one pipeline as primary but also generate a supporting query for the other.
#       3) If both pipelines are suitable, send the original query to both.
# - GraphRAG: entity/triple extraction, triple search, entity-centric graph expansion, chunk rerank, Answer-Judge loop
# - NaiveRAG: chunk-embedding vector search, Answer-Judge loop
# - Aggregator Agent: combines multiple answers from both pipelines into one final answer
#
# Features:
# - Comprehensive thread-safe logging with per-iteration tags
# - Global LLM throttling (concurrency + QPS) for embeddings and generations
# - Robust retries with randomized backoff (LLM, embeddings, Neo4j)
# - Shared ChunkStore (for GraphRAG)
#
# Environment:
# - Loads .env from ../../../.env by default (adjust if needed)
#
# Note:
# - No imports from existing project scripts; this file is self-contained.
# - Requires: python-dotenv, neo4j, google-generativeai

import os, time, json, math, pickle, re, random, hashlib, threading, sys
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, Set
from threading import Lock, Semaphore

from dotenv import load_dotenv
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
from neo4j import GraphDatabase

# ----------------- Load .env -----------------
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent.parent.parent / ".env")

# ----------------- Config -----------------
# Credentials and endpoints
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
NEO4J_URI  = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASS", "password")

# Models
GEN_MODEL   = os.getenv("GEN_MODEL", "models/gemini-2.5-flash")
EMBED_MODEL = os.getenv("EMBED_MODEL", "models/text-embedding-004")

# Dataset folder for original chunk pickles (for GraphRAG)
DEFAULT_LANGCHAIN_DIR = (Path(__file__).resolve().parent / "../../../dataset/3_indexing/3a_langchain_results/").resolve()
LANGCHAIN_DIR = Path(os.getenv("LANGCHAIN_DIR") or str(DEFAULT_LANGCHAIN_DIR))
SKIP_FILES = {"all_langchain_documents.pkl"}

# GraphRAG parameters
ENTITY_MATCH_TOP_K = int(os.getenv("ENTITY_MATCH_TOP_K", "15"))
ENTITY_SUBGRAPH_HOPS = int(os.getenv("ENTITY_SUBGRAPH_HOPS", "5"))
ENTITY_SUBGRAPH_PER_HOP_LIMIT = int(os.getenv("ENTITY_SUBGRAPH_PER_HOP_LIMIT", "2000"))
SUBGRAPH_TRIPLES_TOP_K = int(os.getenv("SUBGRAPH_TRIPLES_TOP_K", "30"))
QUERY_TRIPLE_MATCH_TOP_K_PER = int(os.getenv("QUERY_TRIPLE_MATCH_TOP_K_PER", "20"))
MAX_TRIPLES_FINAL = int(os.getenv("MAX_TRIPLES_FINAL", "60"))
MAX_CHUNKS_FINAL_GRAPH = int(os.getenv("MAX_CHUNKS_FINAL_GRAPH", "40"))
CHUNK_RERANK_CAND_LIMIT = int(os.getenv("CHUNK_RERANK_CAND_LIMIT", "10000000"))
ANSWER_MAX_TOKENS = int(os.getenv("ANSWER_MAX_TOKENS", "4096"))
MAX_ANSWER_JUDGE_ITERS_GRAPH = int(os.getenv("MAX_ANSWER_JUDGE_ITERS_GRAPH", "5"))
AJ_ANSWER_MAX_CHARS = int(os.getenv("AJ_ANSWER_MAX_CHARS", "2000000"))

# NaiveRAG parameters
TOP_K_CHUNKS = int(os.getenv("TOP_K_CHUNKS", "40"))
MAX_CHUNKS_FINAL_NAIVE = int(os.getenv("MAX_CHUNKS_FINAL_NAIVE", "40"))
CHUNK_TEXT_CLAMP = int(os.getenv("CHUNK_TEXT_CLAMP", "1000000"))
MAX_ANSWER_JUDGE_ITERS_NAIVE = int(os.getenv("MAX_ANSWER_JUDGE_ITERS_NAIVE", "5"))

# Decomposer and Aggregator parameters
DECOMPOSER_MAX_TOKENS = int(os.getenv("DECOMPOSER_MAX_TOKENS", "4096"))
AGGREGATOR_MAX_TOKENS = int(os.getenv("AGGREGATOR_MAX_TOKENS", "4096"))

# Language default
OUTPUT_LANG_DEFAULT = "en"

# Global LLM throttling
LLM_EMBED_MAX_CONCURRENCY = max(1, int(os.getenv("LLM_EMBED_MAX_CONCURRENCY", "16")))
LLM_EMBED_QPS = float(os.getenv("LLM_EMBED_QPS", "16.0"))
LLM_GEN_MAX_CONCURRENCY   = max(1, int(os.getenv("LLM_GEN_MAX_CONCURRENCY", "8")))
LLM_GEN_QPS   = float(os.getenv("LLM_GEN_QPS", "2.0"))

# Embedding cache cap
CACHE_EMBED_MAX_ITEMS = int(os.getenv("CACHE_EMBED_MAX_ITEMS", "200000"))

# Neo4j retry/timeout
NEO4J_TX_TIMEOUT_S = float(os.getenv("NEO4J_TX_TIMEOUT_S", "60"))
NEO4J_MAX_ATTEMPTS = int(os.getenv("NEO4J_MAX_ATTEMPTS", "10"))
NEO4J_MAX_CONCURRENCY = int(os.getenv("NEO4J_MAX_CONCURRENCY", "0"))  # 0=unlimited

# ----------------- Initialize SDKs -----------------
if not GOOGLE_API_KEY:
    print("[WARN] GOOGLE_API_KEY is empty; set it in your environment or .env", file=sys.stderr)
genai.configure(api_key=GOOGLE_API_KEY)
gen_model = genai.GenerativeModel(GEN_MODEL)
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

# ----------------- Logger -----------------
_LOGGER: Optional["FileLogger"] = None
_LOG_TL = threading.local()

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
            if not isinstance(msg, str):
                try:
                    msg = json.dumps(msg, ensure_ascii=False, default=str)
                except Exception:
                    msg = str(msg)
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
def estimate_tokens_for_text(text: str) -> int:
    return max(1, int(len(text) / 4))  # rough heuristic

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
    a = _as_float_list(a)
    b = _as_float_list(b)
    dot = sum(x*y for x,y in zip(a,b))
    na = math.sqrt(sum(x*x for x in a))
    nb = math.sqrt(sum(y*y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)

def detect_user_language(text: str) -> str:
    t = (text or "").lower()
    if re.search(r"\b(pasal|undang[- ]?undang|uu\s*\d|peraturan|menteri|ayat|bab|bagian|paragraf|ketentuan|sebagaimana|dimaksud)\b", t):
        return "id"
    if re.search(r"\b(article|act|law|regulation|minister|section|paragraph|chapter|pursuant|provided that)\b", t):
        return "en"
    id_tokens = {"yang","dan","atau","tidak","adalah","berdasarkan","sebagaimana","pada","dalam","dapat","harus","wajib",
                 "pasal","undang","peraturan","menteri","ayat","bab","bagian","paragraf","ketentuan","pengundangan","apabila","jika"}
    en_tokens = {"the","and","or","not","is","based","as","provided","pursuant","in","may","must","shall",
                 "article","act","law","regulation","minister","section","paragraph","chapter","whereas"}
    words = re.findall(r"[a-z]+", t)
    score_id = sum(1 for w in words if w in id_tokens)
    score_en = sum(1 for w in words if w in en_tokens)
    if score_id > score_en: return "id"
    if score_en > score_id: return "en"
    return OUTPUT_LANG_DEFAULT

def clamp_text(s: Optional[str], n: int) -> str:
    t = (s or "").strip()
    return t[:n]

def _rand_wait_seconds(min_s: float = 50.0, max_s: float = 80.0) -> float:
    return random.uniform(min_s, max_s)

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

_EMBED_SEM = Semaphore(LLM_EMBED_MAX_CONCURRENCY)
_GEN_SEM   = Semaphore(LLM_GEN_MAX_CONCURRENCY)
_EMBED_QPS = QpsLimiter(LLM_EMBED_QPS)
_GEN_QPS   = QpsLimiter(LLM_GEN_QPS)
_NEO4J_SEM = Semaphore(NEO4J_MAX_CONCURRENCY) if NEO4J_MAX_CONCURRENCY > 0 else None

# Embedding cache
_EMB_CACHE: Dict[str, List[float]] = {}
_EMB_CACHE_LOCK = Lock()

def _cache_key_for_text(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()

# ----------------- Safe API wrappers -----------------
def _api_call_with_retry(func, *args, **kwargs):
    while True:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            wait_s = _rand_wait_seconds()
            log(f"[Retry] API call failed: {e}. Retrying in {wait_s:.1f}s.", level="WARN")
            time.sleep(wait_s)

def embed_text(text: str) -> List[float]:
    key = _cache_key_for_text(text)
    with _EMB_CACHE_LOCK:
        if key in _EMB_CACHE:
            return list(_EMB_CACHE[key])
    with _EMBED_SEM:
        _EMBED_QPS.acquire()
        t0 = time.time()
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
    log(f"[Embed] text_len={len(text)} -> vec_len={len(out)} | {(time.time()-t0)*1000:.0f} ms", level="DEBUG")
    with _EMB_CACHE_LOCK:
        if len(_EMB_CACHE) >= CACHE_EMBED_MAX_ITEMS:
            try:
                _EMB_CACHE.pop(next(iter(_EMB_CACHE)))
            except Exception:
                _EMB_CACHE.clear()
        _EMB_CACHE[key] = list(out)
    return out

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
    with _GEN_SEM:
        _GEN_QPS.acquire()
        t0 = time.time()
        resp = _api_call_with_retry(gen_model.generate_content, prompt, generation_config=cfg)
    log(f"[LLM JSON] call completed in {(time.time()-t0)*1000:.0f} ms", level="DEBUG")
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
        return {}

def safe_generate_text(prompt: str, max_tokens: int, temperature: float = 0.2) -> str:
    cfg = GenerationConfig(temperature=temperature, max_output_tokens=max_tokens)
    with _GEN_SEM:
        _GEN_QPS.acquire()
        t0 = time.time()
        resp = _api_call_with_retry(gen_model.generate_content, prompt, generation_config=cfg)
    took = (time.time()-t0)*1000
    text = extract_text_from_response(resp)
    if text is not None and text.strip():
        log(f"[LLM TEXT] call completed in {took:.0f} ms, len={len(text)}", level="DEBUG")
        return text.strip()
    info = get_finish_info(resp)
    log(f"[LLM text warning] No text returned. Diagnostics: {info}. Took={took:.0f} ms", level="WARN")
    return f"(Model returned no text. finish_info={info})"

# ----------------- Neo4j helpers -----------------
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
            if k.lower() in ("q_emb", "embedding", "q", "emb"):
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
    qid = _next_query_id()
    preview = " ".join((cypher or "").split())
    if len(preview) > 240:
        preview = preview[:240] + "..."
    param_summary = _summarize_params(params)
    attempts = 0
    last_e: Optional[Exception] = None
    while attempts < max(1, NEO4J_MAX_ATTEMPTS):
        attempts += 1
        t0 = time.time()
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
            took = (time.time()-t0)*1000
            log(f"[Neo4j] Success | qid={qid} | rows={len(records)} | {took:.0f} ms")
            return records
        except Exception as e:
            took = (time.time()-t0)*1000
            last_e = e
            if attempts >= NEO4J_MAX_ATTEMPTS:
                break
            wait_s = _rand_wait_seconds()
            log(f"[Neo4j] Failure | qid={qid} | attempt={attempts}/{NEO4J_MAX_ATTEMPTS} | {took:.0f} ms | error={e}. Retrying in {wait_s:.1f}s.", level="WARN")
            time.sleep(wait_s)
    raise RuntimeError(f"Neo4j query failed after {NEO4J_MAX_ATTEMPTS} attempts (qid={qid}): {last_e}")

# ----------------- ChunkStore -----------------
def _norm_id(x) -> str:
    return str(x).strip() if x is not None else ""

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
                log(f"[ChunkStore] HIT by chunk_id only: requested doc={doc_id_s} chunk={chunk_id_s}; using doc={chosen_doc}{note}. len={len(val)}", level="DEBUG")
                return val

        log(f"[ChunkStore] MISS: doc={doc_id_s} chunk={chunk_id_s}", level="DEBUG")
        return None

# ----------------- GraphRAG components -----------------
LEGAL_ENTITY_TYPES = [
    "UU", "PASAL", "AYAT", "INSTANSI", "ORANG", "ISTILAH", "SANKSI", "NOMINAL", "TANGGAL"
]
LEGAL_PREDICATES = [
    "mendefinisikan", "mengubah", "mencabut", "mulai_berlaku", "mewajibkan",
    "melarang", "memberikan_sanksi", "berlaku_untuk", "termuat_dalam",
    "mendelegasikan_kepada", "berjumlah", "berdurasi"
]

AJ_SCHEMA = {
    "type": "object",
    "properties": {
        "acceptable": {"type": "boolean"},
        "problems": {"type": "string"},
        "suggestion": {"type": "string"},
        "notes": {"type": "string"}
    },
    "required": ["acceptable", "problems", "suggestion"]
}

QM_SCHEMA = {
    "type": "object",
    "properties": {
        "modified_query": {"type": "string"},
        "rationale": {"type": "string"}
    },
    "required": ["modified_query"]
}

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

PIPELINE_BRIEF = """
GraphRAG pipeline summary:
1) Agent 1 extracts legal entities/predicates from the query.
2) Agent 1b extracts query triples.
3) Triple-centric retrieval: embed 's [p] o' and query triple_vec index to find similar KG triples.
4) Entity-centric retrieval: embed key entities, match similar KG entities via vector indexes, expand a subgraph to collect triples.
5) Merge triples, rerank by similarity.
6) Collect document chunks for those triples from ChunkStore; embed and rerank chunks by similarity to the query.
7) Answerer must answer strictly from the selected chunks; if context lacks specifics, the answer may be incomplete.
Common issues:
- Missing statute identifiers (UU number/year, Pasal/Ayat).
- Ambiguous entities/terms; need aliases, jurisdiction, timeframe.
- Predicate too vague.
- Scope too broad; constrain to sector/institution or date range.
""".strip()

def _truncate(s: str, max_chars: int) -> str:
    if not isinstance(s, str):
        return ""
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 20] + " ...[truncated]"

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
    out = safe_generate_json(prompt, QUERY_SCHEMA, temp=0.0) or {}
    if "entities" not in out: out["entities"] = []
    if "predicates" not in out: out["predicates"] = []
    log(f"[Agent 1] Output: entities={ [e.get('text') for e in out['entities']] }, predicates={ out['predicates'] }")
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

User question:
\"\"\"{query}\"\"\"
"""
    est_tokens = estimate_tokens_for_text(prompt)
    log("\n[Agent 1b] Prompt:")
    log(prompt)
    log(f"[Agent 1b] Prompt size: {len(prompt)} chars, est_tokens≈{est_tokens}")
    out = safe_generate_json(prompt, QUERY_TRIPLES_SCHEMA, temp=0.0) or {}
    triples = out.get("triples", []) if isinstance(out, dict) else []
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
    log(f"[Agent 1b] Extracted query triples: {['{} [{}] {}'.format(x['subject']['text'], x['predicate'], x['object']['text']) for x in clean]}")
    return clean

def query_triple_to_text(t: Dict[str, Any]) -> str:
    s = ((t.get("subject") or {}).get("text") or "").strip()
    p = (t.get("predicate") or "").strip()
    o = ((t.get("object") or "").get("text") or "").strip() if isinstance(t.get("object"), dict) else ((t.get("object") or "").strip())
    return f"{s} [{p}] {o}"

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
            uid = r["tr"].get("triple_uid")
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
    top_k: int,
    cand_limit: Optional[int] = None
) -> List[Tuple[Tuple[Any, Any], str, Dict[str, Any], float]]:
    limit = cand_limit if isinstance(cand_limit, int) and cand_limit > 0 else CHUNK_RERANK_CAND_LIMIT
    cand = chunk_records[:limit]
    t0 = time.time()
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
    took = (time.time()-t0)*1000
    log(f"[ChunkRerank] Scored {len(scored)} candidates | picked top {min(top_k, len(scored))} | {took:.0f} ms")
    return scored[:top_k]

def mean_similarity_to_query_triples(triple_emb: Optional[List[float]], q_trip_embs: List[List[float]]) -> float:
    if not isinstance(triple_emb, list) and triple_emb is not None:
        triple_emb = _as_float_list(triple_emb)
    if not isinstance(triple_emb, list) or not q_trip_embs:
        return 0.0
    sims = [cos_sim(triple_emb, q) for q in q_trip_embs]
    if not sims:
        return 0.0
    return sum(sims) / len(sims)

def rerank_triples_by_query_triples(
    triples: List[Dict[str, Any]],
    q_trip_embs: List[List[float]],
    q_emb_fallback: Optional[List[float]],
    top_k: int
) -> List[Dict[str, Any]]:
    t0 = time.time()
    def score(t: Dict[str, Any]) -> float:
        emb = t.get("embedding")
        if q_trip_embs:
            return mean_similarity_to_query_triples(emb, q_trip_embs)
        if q_emb_fallback and isinstance(emb, list):
            return cos_sim(q_emb_fallback, emb)
        return 0.0
    ranked = sorted(triples, key=score, reverse=True)
    took = (time.time()-t0)*1000
    log(f"[TripleRerank] Input={len(triples)} | Output={min(top_k, len(ranked))} | {took:.0f} ms")
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
    answer = safe_generate_text(prompt, max_tokens=ANSWER_MAX_TOKENS, temperature=0.2)
    log(f"[Agent 2] Answer length={len(answer)}")
    return answer

def graph_rag_run(query_original: str, chunk_store: ChunkStore, user_lang: Optional[str] = None) -> Dict[str, Any]:
    user_lang = user_lang or detect_user_language(query_original)
    per_iteration: List[Dict[str, Any]] = []
    qaf_history: List[Dict[str, Any]] = []
    current_query = query_original
    final_answer = ""
    set_log_context("Graph:init")

    for i in range(1, MAX_ANSWER_JUDGE_ITERS_GRAPH + 1):
        set_log_context(f"G{i}/{MAX_ANSWER_JUDGE_ITERS_GRAPH}")
        iter_start = time.time()
        log(f"--- GraphRAG Iteration {i}/{MAX_ANSWER_JUDGE_ITERS_GRAPH} START ---")

        # Step 0: Embed whole query
        t0 = time.time()
        q_emb_query = embed_text(current_query)
        log(f"[GraphRAG] Step 0 embed in {(time.time()-t0)*1000:.0f} ms")

        # Agent 1 & 1b
        t1 = time.time()
        extraction = agent1_extract_entities_predicates(current_query)
        ents = extraction.get("entities", [])
        log(f"[GraphRAG] Step 1 entities in {(time.time()-t1)*1000:.0f} ms")

        t1b = time.time()
        query_triples = agent1b_extract_query_triples(current_query)
        log(f"[GraphRAG] Step 1b triples in {(time.time()-t1b)*1000:.0f} ms")

        # Triple-centric
        t2 = time.time()
        triples_map: Dict[str, Dict[str, Any]] = {}
        q_trip_embs: List[List[float]] = []
        for qt in query_triples:
            try:
                txt = query_triple_to_text(qt)
                emb = embed_text(txt)
                q_trip_embs.append(emb)
            except Exception as ex:
                log(f"[GraphRAG] Embedding failed for query triple: {ex}", level="WARN")
                continue
            matches = search_similar_triples_by_embedding(emb, k=QUERY_TRIPLE_MATCH_TOP_K_PER)
            for m in matches:
                uid = m.get("triple_uid")
                if uid:
                    if uid not in triples_map or (m.get("score", 0.0) > triples_map[uid].get("score", 0.0)):
                        triples_map[uid] = m
        ctx2_triples = list(triples_map.values())
        log(f"[GraphRAG] Step 2 triple-centric in {(time.time()-t2)*1000:.0f} ms; triples={len(ctx2_triples)}")

        # Entity-centric
        t3 = time.time()
        all_matched_keys: Set[str] = set()
        all_matched_ids: Set[str] = set()
        for e in ents:
            text = (e.get("text") or "").strip()
            if not text:
                continue
            try:
                e_emb = embed_text(text)
                matches = search_similar_entities_by_embedding(e_emb, k=ENTITY_MATCH_TOP_K)
                keys = [m.get("key") for m in matches if m.get("key")]
                ids  = [m.get("elem_id") for m in matches if m.get("elem_id")]
                all_matched_keys.update(keys)
                all_matched_ids.update(ids)
            except Exception as ex:
                log(f"[GraphRAG] Entity embedding failed '{text}': {ex}", level="WARN")
                continue

        ctx1_triples: List[Dict[str, Any]] = []
        if all_matched_keys or all_matched_ids:
            expanded_triples = expand_from_entities(
                list(all_matched_keys),
                hops=ENTITY_SUBGRAPH_HOPS,
                per_hop_limit=ENTITY_SUBGRAPH_PER_HOP_LIMIT,
                entity_elem_ids=list(all_matched_ids) if all_matched_ids else None
            )
            log(f"[GraphRAG] Expanded subgraph triples: {len(expanded_triples)}")
            def score_ent(t):
                emb = t.get("embedding")
                if q_trip_embs:
                    return mean_similarity_to_query_triples(emb, q_trip_embs)
                if q_emb_query and isinstance(emb, list):
                    return cos_sim(q_emb_query, emb)
                return 0.0
            ranked = sorted(expanded_triples, key=score_ent, reverse=True)
            ctx1_triples = ranked[:SUBGRAPH_TRIPLES_TOP_K]
        log(f"[GraphRAG] Step 3 entity-centric in {(time.time()-t3)*1000:.0f} ms; triples={len(ctx1_triples)}")

        # Merge triples
        t4 = time.time()
        triple_map: Dict[str, Dict[str, Any]] = {}
        for t in ctx1_triples + ctx2_triples:
            uid = t.get("triple_uid")
            if uid:
                prev = triple_map.get(uid)
                if prev is None or (t.get("score", 0.0) > prev.get("score", 0.0)):
                    triple_map[uid] = t
        merged_triples = list(triple_map.values())
        log(f"[GraphRAG] Step 4 merge triples in {(time.time()-t4)*1000:.0f} ms; merged={len(merged_triples)}")

        # Gather chunks and rerank
        t5 = time.time()
        chunk_records = collect_chunks_for_triples(merged_triples, chunk_store)
        log(f"[GraphRAG] Step 5a collect chunks candidates={len(chunk_records)}")
        chunks_ranked = rerank_chunks_by_query(
            chunk_records, q_emb_query, top_k=MAX_CHUNKS_FINAL_GRAPH, cand_limit=None
        )
        log(f"[GraphRAG] Step 5b rerank chunks in {(time.time()-t5)*1000:.0f} ms; selected={len(chunks_ranked)}")

        # Rerank triples
        t6 = time.time()
        triples_ranked = rerank_triples_by_query_triples(
            merged_triples, q_trip_embs=q_trip_embs, q_emb_fallback=q_emb_query, top_k=MAX_TRIPLES_FINAL
        )
        log(f"[GraphRAG] Step 6 rerank triples in {(time.time()-t6)*1000:.0f} ms; selected={len(triples_ranked)}")

        # Build context and answer
        t_ctx = time.time()
        context_text, context_summary, _ = build_combined_context_text(triples_ranked, chunks_ranked)
        log(f"[GraphRAG] Context built in {(time.time()-t_ctx)*1000:.0f} ms")
        answer = agent2_answer(current_query, context_text, guidance=None, output_lang=user_lang)

        # Judge
        history_str = json.dumps(qaf_history, ensure_ascii=False, indent=2) if qaf_history else "[]"
        prompt_aj = f"""
You are Agent AJ (Answer Judge).
Task: Evaluate whether the generated answer sufficiently and accurately addresses the current query, given how the GraphRAG pipeline works.

Pipeline brief:
{PIPELINE_BRIEF}

Guidelines:
- Mark "acceptable": true only if the answer is relevant, sufficiently specific, and plausibly grounded (e.g., cites UU/Article or key text).
- If unacceptable, diagnose concrete issues in "problems" (e.g., vague, missing statute identifiers, wrong scope/jurisdiction, incomplete, lacks references).
- Provide an actionable "suggestion" for modifying the next query so the pipeline can retrieve better context and produce a stronger answer.
- Keep outputs in the same language as the user's question ("{user_lang}").

Current query:
\"\"\"{current_query}\"\"\"

Generated answer (truncated if long):
\"\"\"{_truncate(answer, AJ_ANSWER_MAX_CHARS)}\"\"\"

Prior query–answer–feedback history (latest last):
{history_str}

Return JSON:
- "acceptable": boolean
- "problems": short diagnosis of answer issues (if any)
- "suggestion": concrete improvements to the next query
- "notes": optional
"""
        aj = safe_generate_json(prompt_aj, AJ_SCHEMA, temp=0.0) or {}
        acceptable = bool(aj.get("acceptable"))
        problems = (aj.get("problems") or "").strip()
        suggestion = (aj.get("suggestion") or "").strip()
        notes = (aj.get("notes") or "").strip()
        log(f"[GraphRAG] Judge verdict acceptable={acceptable} | problems='{problems[:120]}' | suggestion='{suggestion[:120]}'")

        # record iteration
        per_iteration.append({
            "iteration": i,
            "query": current_query,
            "answer": answer,
            "judge": {"acceptable": acceptable, "problems": problems, "suggestion": suggestion, "notes": notes},
            "modified_query": None if acceptable or i>=MAX_ANSWER_JUDGE_ITERS_GRAPH else "...",
            "context_summary": context_summary,
            "retrieval_counts": {
                "ctx2_triples": len(ctx2_triples),
                "ctx1_triples": len(ctx1_triples),
                "merged_triples": len(merged_triples),
                "chunk_candidates": len(chunk_records),
                "chunks_selected": len(chunks_ranked),
                "triples_selected": len(triples_ranked),
            },
        })

        if acceptable or i >= MAX_ANSWER_JUDGE_ITERS_GRAPH:
            final_answer = answer
            log(f"[GraphRAG] Stop condition reached (acceptable={acceptable}, i={i}).")
            break

        # Query modifier
        prompt_qm = f"""
You are Agent QM (Query Modifier).
Task: Rewrite the current query to improve retrieval quality and, consequently, the final answer. Use the Answer Judge's feedback and prior history.

Pipeline brief:
{PIPELINE_BRIEF}

Guidelines:
- Preserve the user's original intent and language ("{user_lang}").
- Apply the judge's suggestion(s) concretely (add UU number/year, Pasal/Ayat, scope, relation verbs, timeframe, aliases).
- Produce ONE improved query string; do not include explanations in the query text.

Current query:
\"\"\"{current_query}\"\"\"

Current answer (truncated if long):
\"\"\"{_truncate(answer, AJ_ANSWER_MAX_CHARS)}\"\"\"

Answer Judge feedback:
- Problems: {problems}
- Suggestion: {suggestion}

Prior query–answer–feedback history (latest last):
{history_str}

Return JSON:
- "modified_query": string (only the improved query, same language and intent)
- "rationale": (optional) short explanation for logs
"""
        qm = safe_generate_json(prompt_qm, QM_SCHEMA, temp=0.0) or {}
        modified_query = (qm.get("modified_query") or current_query).strip()
        rationale = (qm.get("rationale") or "").strip()
        log(f"[GraphRAG] Modified query: '{modified_query}' | rationale: {rationale[:120]}")
        qaf_history.append({
            "query": current_query,
            "answer": answer,
            "feedback": {"problems": problems, "suggestion": suggestion},
            "modified_query": modified_query
        })
        per_iteration[-1]["modified_query"] = modified_query
        current_query = modified_query

        log(f"--- GraphRAG Iteration {i}/{MAX_ANSWER_JUDGE_ITERS_GRAPH} DONE | {(time.time()-iter_start)*1000:.0f} ms ---")

    set_log_context(None)
    return {
        "final_answer": final_answer or (per_iteration[-1]["answer"] if per_iteration else ""),
        "iterations_used": len(per_iteration),
        "per_iteration": per_iteration
    }

# ----------------- NaiveRAG components -----------------
ANSWER_JUDGE_SCHEMA_N = {
  "type": "object",
  "properties": {
    "decision": {"type": "string", "enum": ["acceptable", "insufficient"]},
    "reasoning": {"type": "string"},
    "problem": {"type": "string"},
    "suggested_solution": {"type": "string"}
  },
  "required": ["decision", "reasoning"]
}

QUERY_MODIFIER_SCHEMA_N = {
  "type": "object",
  "properties": {
    "modified_query": {"type": "string"},
    "notes": {"type": "string"}
  },
  "required": ["modified_query"]
}

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

def build_context_from_chunks_naive(chunks: List[Dict[str, Any]], max_chunks: int) -> str:
    chosen = chunks[:max_chunks]
    lines = ["Potongan teks terkait (chunk):"]
    for i, c in enumerate(chosen, 1):
        doc = c.get("document_id")
        cid = c.get("chunk_id")
        uu = c.get("uu_number") or ""
        pg = c.get("pages")
        txt = clamp_text(c.get("content") or "", CHUNK_TEXT_CLAMP)
        lines.append(f"[Chunk {i}] doc={doc} chunk={cid} | {uu} | pages={pg} | score={c.get('score'):.3f}\n{txt}\n")
    return "".join(lines)

def naive_rag_run(query_original: str, user_lang: Optional[str] = None) -> Dict[str, Any]:
    user_lang = user_lang or detect_user_language(query_original)
    per_iteration: List[Dict[str, Any]] = []
    qaf_history: List[Dict[str, Any]] = []
    current_query = query_original
    final_answer = ""
    set_log_context("Naive:init")

    for i in range(1, MAX_ANSWER_JUDGE_ITERS_NAIVE + 1):
        set_log_context(f"N{i}/{MAX_ANSWER_JUDGE_ITERS_NAIVE}")
        iter_start = time.time()
        log(f"--- NaiveRAG Iteration {i}/{MAX_ANSWER_JUDGE_ITERS_NAIVE} START ---")

        t_e = time.time()
        q_emb = embed_text(current_query)
        log(f"[NaiveRAG] Embedded query in {(time.time()-t_e)*1000:.0f} ms")

        t_v = time.time()
        candidates = vector_query_chunks(q_emb, k=TOP_K_CHUNKS)
        log(f"[NaiveRAG] Vector search returned {len(candidates)} candidates in {(time.time()-t_v)*1000:.0f} ms")

        if not candidates:
            context_text = "Tidak ada potongan teks yang ditemukan." if user_lang == "id" else "No relevant chunks found."
        else:
            context_text = build_context_from_chunks_naive(candidates, max_chunks=MAX_CHUNKS_FINAL_NAIVE)
            preview = "\n".join(context_text.splitlines()[:20])
            log(f"[NaiveRAG] Context preview:\n{preview}")

        t_ans = time.time()
        answer = agent2_answer(current_query, context_text, guidance=None, output_lang=user_lang)
        log(f"[NaiveRAG] Answer generated in {(time.time()-t_ans)*1000:.0f} ms")

        if i >= MAX_ANSWER_JUDGE_ITERS_NAIVE:
            final_answer = answer
            top_meta = [{"document_id": c.get("document_id"), "chunk_id": c.get("chunk_id"), "uu_number": c.get("uu_number"), "pages": c.get("pages"), "score": c.get("score")} for c in candidates[:5]]
            per_iteration.append({
                "iteration": i,
                "query": current_query,
                "answer": answer,
                "judge": {"decision": "skipped_limit"},
                "modified_query": None,
                "retrieval_meta": {"num_candidates": len(candidates), "top_chunks": top_meta}
            })
            log(f"--- NaiveRAG Iteration {i}/{MAX_ANSWER_JUDGE_ITERS_NAIVE} DONE (cap) ---")
            break

        # Judge
        hist_lines = []
        for idx, item in enumerate(qaf_history, 1):
            fb = (item.get("feedback") or {}) or {}
            hist_lines.append(
                f"[Iter {idx}] Query: {item.get('query')}\n[Iter {idx}] Answer: {item.get('answer')}\n"
                f"[Iter {idx}] Feedback.problem: {fb.get('problem','')}\n[Iter {idx}] Feedback.suggested_solution: {fb.get('suggested_solution','')}"
            )
        history_block = "\n".join(hist_lines) if hist_lines else "(no prior query–answer–feedback history)"

        instructions = (
            "You are the Answer Judge for a GraphRAG-like pipeline with naive chunk vector search.\n"
            "Judge whether the current answer adequately addresses the current query (relevance, completeness, specificity, legal grounding).\n"
            "If acceptable: decision='acceptable'. If insufficient: decision='insufficient' and provide a concise problem and suggested_solution.\n"
            f"Respond in the same language as the user's question ({user_lang})."
        )

        prompt = f"""
You are the Answer Judge.

Current query:
\"\"\"{current_query}\"\"\"

Generated answer:
\"\"\"{answer}\"\"\"

Prior query–answer–feedback history:
\"\"\"{history_block}\"\"\"

Instructions:
{instructions}

Return JSON with keys: decision ('acceptable'|'insufficient'), reasoning, and if insufficient also problem and suggested_solution.
"""
        judge = safe_generate_json(prompt, ANSWER_JUDGE_SCHEMA_N, temp=0.0) or {}
        decision = (judge.get("decision") or "").strip().lower()
        reasoning = (judge.get("reasoning") or "").strip()
        problem = (judge.get("problem") or "").strip()
        solution = (judge.get("suggested_solution") or "").strip()

        if decision not in ("acceptable","insufficient"):
            decision = "insufficient"
        log(f"[NaiveRAG] Judge decision: {decision.upper()} | Reasoning: {reasoning[:160]}")

        if decision == "acceptable":
            final_answer = answer
            top_meta = [{"document_id": c.get("document_id"), "chunk_id": c.get("chunk_id"), "uu_number": c.get("uu_number"), "pages": c.get("pages"), "score": c.get("score")} for c in candidates[:5]]
            per_iteration.append({
                "iteration": i,
                "query": current_query,
                "answer": answer,
                "judge": judge,
                "modified_query": None,
                "retrieval_meta": {"num_candidates": len(candidates), "top_chunks": top_meta}
            })
            log(f"[NaiveRAG] ACCEPTED at iteration {i}.")
            break

        # Modify query
        fb = {"problem": problem, "suggested_solution": solution}
        qaf_history.append({"iteration": i, "query": current_query, "answer": answer, "feedback": fb})

        mod_instructions = (
            "You are the Query Modifier for a naive RAG over legal chunks.\n"
            "Use the judge's problem diagnosis and suggestion plus prior history to rewrite the query for better retrieval.\n"
            "Keep user's intent and language. Prefer adding exact identifiers (law number, article, year), synonyms, constraints (jurisdiction/timeframe), canonical names.\n"
            "Be precise and avoid hallucinations."
        )
        prompt_mod = f"""
You are the Query Modifier.

Current query:
\"\"\"{current_query}\"\"\"

Current answer:
\"\"\"{answer}\"\"\"

Answer judge feedback:
- problem: {problem}
- suggested_solution: {solution}

Prior query–answer–feedback history:
\"\"\"{history_block}\"\"\"

Instructions:
{mod_instructions}

Return JSON with:
- modified_query: the improved query (same language as the user)
- notes: brief rationale (optional)
"""
        mod = safe_generate_json(prompt_mod, QUERY_MODIFIER_SCHEMA_N, temp=0.0) or {}
        new_query = (mod.get("modified_query") or current_query).strip()
        notes = (mod.get("notes") or "").strip()
        log(f"[NaiveRAG] Modified query -> {new_query}")
        if notes:
            log(f"[NaiveRAG] Modifier notes: {notes[:160]}")

        per_iteration.append({
            "iteration": i,
            "query": current_query,
            "answer": answer,
            "judge": judge,
            "modified_query": new_query,
            "retrieval_meta": {"num_candidates": len(candidates)}
        })

        current_query = new_query
        log(f"--- NaiveRAG Iteration {i}/{MAX_ANSWER_JUDGE_ITERS_NAIVE} DONE | {(time.time()-iter_start)*1000:.0f} ms ---")

    set_log_context(None)
    return {
        "final_answer": final_answer or (per_iteration[-1]["answer"] if per_iteration else ""),
        "iterations_used": len(per_iteration),
        "iteration_runs": per_iteration
    }

# ----------------- Query Decomposer Agent -----------------
DECOMPOSER_SCHEMA = {
    "type": "object",
    "properties": {
        "tasks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "pipeline": {"type": "string", "enum": ["graphrag", "naiverag"]},
                    "query": {"type": "string"},
                    "role": {"type": "string", "enum": ["primary", "support"]},
                    "aspect": {"type": "string"}
                },
                "required": ["pipeline", "query"]
            }
        },
        "strategy": {"type": "string"},
        "notes": {"type": "string"}
    },
    "required": ["tasks"]
}

DECOMPOSER_GUIDE = """
You are a Query Decomposer Agent for a dual-pipeline RAG system:

Pipelines:
- NaiveRAG: chunk-embedding vector search; excels at direct lookup of a specific passage, definitions, quotes, or single-hop answers when an explicit citation (UU number/year, Pasal/Article/Ayat/Section, PP/Perda/Permen, "No.") is present.
- GraphRAG: entity/triple extraction and KG traversal; excels at multi-hop reasoning, relationships between entities (mewajibkan, melarang, mendelegasikan, mengubah, mencabut), scope/actor applicability (berlaku untuk siapa), comparisons (perbedaan), exceptions (pengecualian), temporal/effectivity (mulai berlaku), and cross-law interactions.

Decomposition actions:
1) If different aspects of the query are better served by different pipelines, create separate tasks:
   - Assign each aspect to the pipeline best suited for it (GraphRAG for relationships/multi-hop/scope/time; NaiveRAG for direct passage retrieval/definition).
2) If the query is primarily suited to one pipeline, still consider creating one support task for the other pipeline to retrieve helpful context or precise citations/quotes.
3) If the query can benefit similarly from both pipelines, send the original query to both as 'primary'.

Guidelines:
- Preserve the user's language and intent; do not hallucinate facts.
- Keep sub-queries concise but precise; add statute numbers, articles, or actor/scope hints if they are clearly implied.
- Use roles:
   - 'primary' for the main task(s) that should directly answer an aspect.
   - 'support' for a task that provides useful reinforcing context (e.g., NaiveRAG fetching the exact article text while GraphRAG handles relationships).
- Prefer at least one 'primary' task. You may include zero or more 'support' tasks.
- Cap the total number of tasks to a reasonable number (e.g., 2–4) unless the user explicitly asks for multiple distinct aspects.

Return JSON with:
- tasks: array of {pipeline, query, role, aspect(optional)}
- strategy: brief description of how you decomposed and why
- notes: optional
""".strip()

def decomposer_decompose(user_query: str, lang: str) -> Dict[str, Any]:
    prompt = f"""
You are a Query Decomposer Agent.

User question (language={lang}):
\"\"\"{user_query}\"\"\"

{DECOMPOSER_GUIDE}

Return JSON only.
"""
    est = estimate_tokens_for_text(prompt)
    log("[Decomposer] Prompt:")
    log(prompt)
    log(f"[Decomposer] Prompt size: {len(prompt)} chars, est_tokens≈{est}")

    out = safe_generate_json(prompt, DECOMPOSER_SCHEMA, temp=0.0) or {}
    tasks = out.get("tasks") or []
    strategy = (out.get("strategy") or "").strip()
    notes = (out.get("notes") or "").strip()

    # Fallback if decomposer fails or returns empty
    if not tasks:
        q = (user_query or "").lower()
        explicit_citation = bool(re.search(r"\b(uu|undang[- ]?undang|pasal|ayat|article|section|pp|perda|permen|no\.)\b", q))
        relation_intent = bool(re.search(r"\b(hubungan|perbedaan|pengecualian|mendelegasikan|mengubah|mencabut|berlaku|berlaku untuk|dibandingkan|apakah .* mengatur)\b", q))
        if explicit_citation and not relation_intent:
            tasks = [
                {"pipeline": "naiverag", "query": user_query, "role": "primary", "aspect": "direct_passage"},
                {"pipeline": "graphrag", "query": user_query, "role": "support", "aspect": "structure_relationships"}
            ]
            strategy = "(fallback) citation-focused: primary NaiveRAG + GraphRAG support"
        elif relation_intent:
            tasks = [
                {"pipeline": "graphrag", "query": user_query, "role": "primary", "aspect": "relationships/scope/time"},
                {"pipeline": "naiverag", "query": user_query, "role": "support", "aspect": "exact_citations/quotes"}
            ]
            strategy = "(fallback) relationship-focused: primary GraphRAG + NaiveRAG support"
        else:
            tasks = [
                {"pipeline": "naiverag", "query": user_query, "role": "primary", "aspect": "direct_passage"},
                {"pipeline": "graphrag", "query": user_query, "role": "primary", "aspect": "structured_linkages"}
            ]
            strategy = "(fallback) ambiguous: send to both as primary"

    # Clamp to reasonable number of tasks (max 4)
    tasks = tasks[:4]
    log(f"[Decomposer] Planned {len(tasks)} task(s) | strategy: {strategy}")
    for idx, t in enumerate(tasks, 1):
        log(f"[Decomposer] Task {idx}: pipeline={t.get('pipeline')} role={t.get('role')} aspect={t.get('aspect','')} | query='{(t.get('query') or '')[:160]}'")
    return {"tasks": tasks, "strategy": strategy, "notes": notes}

# ----------------- Aggregator Agent -----------------
AGGREGATOR_SCHEMA = {
    "type": "object",
    "properties": {
        "decision": {"type": "string", "enum": ["choose_graphrag", "choose_naiverag", "merge"]},
        "final_answer": {"type": "string"},
        "rationale": {"type": "string"}
    },
    "required": ["decision", "final_answer"]
}

def _citation_score(text: str) -> int:
    if not text: return 0
    patterns = [
        r"\bUU\b", r"\bUndang[- ]?Undang\b", r"\bPasal\b", r"\bAyat\b",
        r"\bArticle\b", r"\bSection\b", r"\bLaw\b", r"\bNo\.\b", r"\bPeraturan\b"
    ]
    score = 0
    for pat in patterns:
        score += len(re.findall(pat, text, flags=re.IGNORECASE))
    score += len(re.findall(r"\b\d{4}\b", text))
    score += len(re.findall(r"\b\d+\b", text)) // 5
    return score

def _format_task_block(title: str, tasks: List[Dict[str, Any]]) -> str:
    lines = [title]
    for i, t in enumerate(tasks, 1):
        role = t.get("role","")
        asp = t.get("aspect","")
        q = t.get("query","")
        ans = t.get("result",{}).get("final_answer","")
        lines.append(f"[{i}] role={role} aspect={asp}\n- subquery: {q}\n- answer:\n{ans}\n")
    return "\n".join(lines)

def aggregate_answers(user_query: str, lang: str, g_tasks: List[Dict[str, Any]], n_tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
    # Build readable blocks for the aggregator
    g_block = _format_task_block("GraphRAG answers:", g_tasks) if g_tasks else "GraphRAG answers: (none)"
    n_block = _format_task_block("NaiveRAG answers:", n_tasks) if n_tasks else "NaiveRAG answers: (none)"

    prompt = f"""
You are an Aggregator Agent.
Task: Read the user's question and the answers produced by two pipelines (GraphRAG and NaiveRAG), possibly multiple sub-answers per pipeline.
Choose the best answer or synthesize a concise merged answer that is clearer, more specific, and grounded in the provided content.

Rules:
- Respond in the same language as the user's question: "{lang}".
- Prefer answers with specific statute references (UU/Article/Pasal) and clear, correct claims.
- Do not introduce new facts beyond the provided sub-answers; you may rephrase and combine.
- If the sub-answers conflict, prefer the one with explicit citations and coherent legal logic; if uncertainty remains, say so briefly.

Return JSON with:
- decision: one of ["choose_graphrag", "choose_naiverag", "merge"]
- final_answer: the final answer text (if 'merge', provide the merged text)
- rationale: brief reason for the decision

User question:
\"\"\"{user_query}\"\"\"

{g_block}

{n_block}
"""
    est = estimate_tokens_for_text(prompt)
    log("[Aggregator] Prompt:")
    log(prompt)
    log(f"[Aggregator] Prompt size: {len(prompt)} chars, est_tokens≈{est}")

    out = safe_generate_json(prompt, AGGREGATOR_SCHEMA, temp=0.0) or {}
    decision = (out.get("decision") or "").strip()
    final_answer = (out.get("final_answer") or "").strip()
    rationale = (out.get("rationale") or "").strip()

    if decision in ("choose_graphrag","choose_naiverag","merge") and final_answer:
        log(f"[Aggregator] Decision={decision} | rationale={rationale[:160]}")
        return {"decision": decision, "final_answer": final_answer, "rationale": rationale}

    # Fallback heuristic
    g_text = "\n".join([(t.get("result",{}) or {}).get("final_answer","") for t in g_tasks]) if g_tasks else ""
    n_text = "\n".join([(t.get("result",{}) or {}).get("final_answer","") for t in n_tasks]) if n_tasks else ""
    g_score = _citation_score(g_text)
    n_score = _citation_score(n_text)
    if g_score > n_score:
        decision = "choose_graphrag"
        final_answer = g_text or n_text
        rationale = f"Fallback heuristic: GraphRAG had more concrete citations (g={g_score} > n={n_score})."
    elif n_score > g_score:
        decision = "choose_naiverag"
        final_answer = n_text or g_text
        rationale = f"Fallback heuristic: NaiveRAG had more concrete citations (n={n_score} > g={g_score})."
    else:
        decision = "merge"
        final_answer = (g_text + "\n" + n_text).strip()
        rationale = "Fallback heuristic: tie; merged both."
    log(f"[Aggregator] Fallback decision={decision}")
    return {"decision": decision, "final_answer": final_answer, "rationale": rationale}

# ----------------- Orchestrator (Decomposer → Pipelines → Aggregator) -----------------
def _combine_pipeline_level_answer(tasks: List[Dict[str, Any]], label: str) -> str:
    """Combine multiple sub-answers from the same pipeline for compatibility fields."""
    if not tasks:
        return ""
    if len(tasks) == 1:
        return (tasks[0].get("result") or {}).get("final_answer", "")
    parts = []
    for i, t in enumerate(tasks, 1):
        role = t.get("role","")
        asp = t.get("aspect","")
        ans = (t.get("result") or {}).get("final_answer","")
        parts.append(f"[{label} Task {i} | role={role} aspect={asp}]\n{ans}")
    return "\n".join(parts)

def run_multi_agent(user_query: str) -> Dict[str, Any]:
    """
    Decomposer-based orchestrator:
    - Decompose the question into sub-queries for GraphRAG and/or NaiveRAG.
    - Run each sub-query through its respective pipeline.
    - Aggregate all sub-answers into one final answer.
    Returns final answer and rich diagnostics. Compatible with the existing run_multi_agent.py runner.
    """
    global _LOGGER
    ts_name = make_timestamp_name()
    log_file = Path.cwd() / f"{ts_name}.txt"
    _LOGGER = FileLogger(log_file, also_console=True)
    set_log_context(None)

    t_all = time.time()
    try:
        log("=== Multi-Agent RAG (Decomposer + Aggregator) run started ===")
        log(f"Process info: pid={_pid()}")
        log(f"Log file: {log_file}")
        log(f"Original Query: {user_query}")
        log(f"Parameters: GraphRAG iters={MAX_ANSWER_JUDGE_ITERS_GRAPH}, NaiveRAG iters={MAX_ANSWER_JUDGE_ITERS_NAIVE}, "
            f"LLM: EMBED(max_conc={LLM_EMBED_MAX_CONCURRENCY}, qps={LLM_EMBED_QPS}), GEN(max_conc={LLM_GEN_MAX_CONCURRENCY}, qps={LLM_GEN_QPS}), "
            f"Neo4j attempts={NEO4J_MAX_ATTEMPTS}, timeout={NEO4J_TX_TIMEOUT_S:.1f}s, concurrency={NEO4J_MAX_CONCURRENCY or 'unlimited'}")

        user_lang = detect_user_language(user_query)
        log(f"[Language] Detected user language: {user_lang}")

        # Decompose
        dec = decomposer_decompose(user_query, user_lang)
        tasks = dec.get("tasks", [])

        # Separate by pipeline
        g_tasks: List[Dict[str, Any]] = []
        n_tasks: List[Dict[str, Any]] = []
        for t in tasks:
            pipe = (t.get("pipeline") or "").lower()
            if pipe == "graphrag":
                g_tasks.append(t)
            elif pipe == "naiverag":
                n_tasks.append(t)

        # Run tasks (sequential for clarity; they share global QPS throttles)
        # GraphRAG needs ChunkStore once
        chunk_store = None
        if g_tasks:
            chunk_store = ChunkStore(LANGCHAIN_DIR, set(SKIP_FILES))
        for idx, t in enumerate(g_tasks, 1):
            set_log_context(f"Gtask {idx}/{len(g_tasks)}")
            subq = t.get("query","")
            role = t.get("role","")
            log(f"[Run] GraphRAG Task {idx}/{len(g_tasks)} | role={role} | subquery: {subq[:160]}")
            try:
                res = graph_rag_run(subq, chunk_store=chunk_store, user_lang=user_lang)
            except Exception as e:
                log(f"[GraphRAG Task {idx}] ERROR: {e}", level="WARN")
                res = {"final_answer": f"(GraphRAG error: {e})", "iterations_used": 0, "per_iteration": []}
            t["result"] = res

        for idx, t in enumerate(n_tasks, 1):
            set_log_context(f"Ntask {idx}/{len(n_tasks)}")
            subq = t.get("query","")
            role = t.get("role","")
            log(f"[Run] NaiveRAG Task {idx}/{len(n_tasks)} | role={role} | subquery: {subq[:160]}")
            try:
                res = naive_rag_run(subq, user_lang=user_lang)
            except Exception as e:
                log(f"[NaiveRAG Task {idx}] ERROR: {e}", level="WARN")
                res = {"final_answer": f"(NaiveRAG error: {e})", "iterations_used": 0, "iteration_runs": []}
            t["result"] = res

        set_log_context(None)

        # Aggregate results
        agg = aggregate_answers(user_query, user_lang, g_tasks, n_tasks)
        final_answer = agg.get("final_answer","") or ""

        # For compatibility with the runner:
        # - Provide pipeline-level combined answers under graphrag.final_answer and naiverag.final_answer.
        graphrag_combined = _combine_pipeline_level_answer(g_tasks, "GraphRAG")
        naiverag_combined = _combine_pipeline_level_answer(n_tasks, "NaiveRAG")

        # Build pipeline diagnostics
        graphrag_out = {
            "final_answer": graphrag_combined,
            "tasks": g_tasks,
            "iterations_used_total": sum((t.get("result") or {}).get("iterations_used", 0) for t in g_tasks)
        }
        naiverag_out = {
            "final_answer": naiverag_combined,
            "tasks": n_tasks,
            "iterations_used_total": sum((t.get("result") or {}).get("iterations_used", 0) for t in n_tasks)
        }

        total_ms = (time.time()-t_all)*1000
        log("\n=== Multi-Agent RAG (Decomposer + Aggregator) summary ===")
        log(f"- Decomposer planned tasks: {len(tasks)} | strategy: {dec.get('strategy','')}")
        log(f"- Aggregator decision: {agg.get('decision','')} | rationale: {agg.get('rationale','')}")
        log(f"- GraphRAG tasks: {len(g_tasks)} | NaiveRAG tasks: {len(n_tasks)}")
        log(f"- Total runtime: {total_ms:.0f} ms")
        log("\n=== Final Answer (Aggregated) ===")
        log(final_answer)
        log(f"\nLogs saved to: {log_file}")

        # The existing runner expects an "aggregator.decision" field; we keep it.
        return {
            "final_answer": final_answer,
            "decomposer": {"tasks": tasks, "strategy": dec.get("strategy",""), "notes": dec.get("notes","")},
            "aggregator": {"decision": agg.get("decision",""), "rationale": agg.get("rationale","")},
            "graphrag": graphrag_out,
            "naiverag": naiverag_out,
            "log_file": str(log_file),
            "timings_ms": {"total": int(total_ms)}
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
            result = run_multi_agent(user_query)
            print("\n" + "="*100)
            print("Decomposer strategy:", (result.get("decomposer") or {}).get("strategy",""))
            print("Aggregator decision:", (result.get("aggregator") or {}).get("decision",""))
            print("Final Answer:")
            print(result.get("final_answer",""))
            print("="*100)
    finally:
        try:
            driver.close()
        except Exception:
            pass