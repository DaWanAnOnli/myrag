#!/usr/bin/env python3
# multi_agent.py
# Agentic Iterative Multi-Pipeline RAG Orchestrator
# - Pipelines: GraphRAG (single pass) and Naive RAG (single pass)
# - Aggregator agent synthesizes the final answer from both approaches
# - Answer Judge agent and Query Modifier agent enable multi-iteration refinement
# - Iterative loop persists query–answer–feedback history (in-memory + JSONL) and stops at a hardcoded max
# - Comprehensive logging across stages with timestamps, PID, iterations, pipeline telemetry, and decisions
# - Revised: rate limiting, scheduling, and throttling follow Code 2’s method (QPS limiters + semaphores)
# - Pipeline logic/order remains unchanged

import os, sys, time, json, math, pickle, re, random, difflib, statistics
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, Set
from threading import Lock, Semaphore
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
NEO4J_MAX_ATTEMPTS = int(os.getenv("NEO4J_MAX_ATTEMPTS", "100000"))

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
MAX_ITERS = 1  # single pass for inner pipelines (kept)

# ---------- Agentic iterative controller (hardcoded max iterations) ----------
AGENTIC_MAX_ITERS = 2  # Hardcoded per requirement

# ---------- LLM/DB throttling (match Code 2 style) ----------
# QPS and concurrency controls (default values mirror Code 2 behavior)
LLM_EMBED_MAX_CONCURRENCY = max(1, int(os.getenv("LLM_EMBED_MAX_CONCURRENCY", "165")))
LLM_EMBED_QPS = float(os.getenv("LLM_EMBED_QPS", "165.0"))
LLM_GEN_MAX_CONCURRENCY   = max(1, int(os.getenv("LLM_GEN_MAX_CONCURRENCY", "100")))
LLM_GEN_QPS   = float(os.getenv("LLM_GEN_QPS", "13.0"))
NEO4J_MAX_CONCURRENCY = int(os.getenv("NEO4J_MAX_CONCURRENCY", "0"))  # 0 = unlimited

# ----------------- Initialize SDKs -----------------
genai.configure(api_key=GOOGLE_API_KEY)
gen_model = genai.GenerativeModel(GEN_MODEL)
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

# ----------------- Logger -----------------
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
    return f"[{_now_ts()}] [{level}] [pid={_pid()}]"

class FileLogger:
    def __init__(self, file_path: Path, also_console: bool = True):
        self.file_path = file_path
        self.also_console = also_console
        self._fh = open(file_path, "w", encoding="utf-8")

    def log(self, msg: Any = ""):
        if not isinstance(msg, str):
            try:
                msg = json.dumps(msg, ensure_ascii=False, default=str)
            except Exception:
                msg = str(msg)
        lines = (msg or "").splitlines() or [""]
        prefixed = [f"{_prefix()} {line}" for line in lines]
        out = "\n".join(prefixed) + "\n"
        self._fh.write(out)
        self._fh.flush()
        if self.also_console:
            print(out, end="", flush=True)

    def close(self):
        try:
            self._fh.flush(); self._fh.close()
        except Exception:
            pass

class JSONLLogger:
    def __init__(self, file_path: Path):
        self.file_path = file_path
        self._fh = open(file_path, "w", encoding="utf-8")

    def append(self, obj: Dict[str, Any]):
        try:
            obj = dict(obj or {})
            obj["_ts"] = _now_ts()
        except Exception:
            pass
        self._fh.write(json.dumps(obj, ensure_ascii=False, default=str) + "\n")
        self._fh.flush()

    def close(self):
        try:
            self._fh.flush(); self._fh.close()
        except Exception:
            pass

_LOGGER: Optional[FileLogger] = None
_JSONL: Optional[JSONLLogger] = None

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

def seq_ratio(a: str, b: str) -> float:
    a = (a or "").strip()
    b = (b or "").strip()
    if not a and not b:
        return 1.0
    return difflib.SequenceMatcher(a=a.lower(), b=b.lower()).ratio()

def clamp(s: Optional[str], n: int) -> str:
    t = (s or "").strip()
    return t[:n]

def safe_mean(nums: List[float]) -> float:
    try:
        return float(statistics.mean(nums)) if nums else 0.0
    except Exception:
        return 0.0

# ----------------- QPS limiters and semaphores (Code 2 style) -----------------
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

def _rand_wait_seconds() -> float:
    # Shorter jitter window to match Code 2
    return random.uniform(50.0, 80.0)

def _api_call_with_retry(func, *args, **kwargs):
    while True:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            wait_s = _rand_wait_seconds()
            log(f"[Retry] API call failed: {e}. Retrying in {wait_s:.1f}s.", level="WARN")
            time.sleep(wait_s)

# ----------------- Embeddings -----------------
def embed_text(text: str) -> List[float]:
    # Apply concurrency gate and QPS pacing just like Code 2
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
                    if isinstance(t, str): buf.append(t)
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
        return {}

def safe_generate_text(prompt: str, max_tokens: int, temperature: float = 0.2) -> str:
    cfg = GenerationConfig(temperature=temperature, max_output_tokens=max_tokens)
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
            log(f"[Neo4j] Success | qid={qid} | rows={len(records)} | {took:.0f} ms", level="INFO")
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

# ----------------- Telemetry and citation helpers -----------------
def citation_stats(txt: str) -> Dict[str, Any]:
    t = txt or ""
    pats = [
        r"\b(Pasal|Ayat|UU|Undang[- ]?Undang|Peraturan|Bab|Bagian)\b",
        r"\b(Article|Section|Chapter|Act|Law|Regulation)\b",
        r"\bPasal\s+\d+",
        r"\bArticle\s+\d+",
    ]
    count = 0
    for p in pats:
        count += len(re.findall(p, t, flags=re.IGNORECASE))
    return {"citation_like_count": count, "length": len(t)}

def chunk_score_stats(scored_chunks: List[Tuple[Tuple[Any, Any], str, Dict[str, Any], float]]) -> Dict[str, Any]:
    scores = [x[3] for x in scored_chunks]
    return {
        "count": len(scores),
        "min": float(min(scores)) if scores else 0.0,
        "max": float(max(scores)) if scores else 0.0,
        "mean": float(safe_mean(scores)),
    }

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

    telem = {
        "user_lang": user_lang,
        "candidates": len(candidates),
        "top_scores": [float(c.get("score", 0.0)) for c in candidates[:5]],
        "answer_stats": citation_stats(answer),
        "context_preview": "\n".join((build_context_from_chunks(candidates, max_chunks=min(5, len(candidates))) if candidates else "").splitlines()[:15]) if candidates else "",
    }
    return {"answer": answer, "telemetry": telem}

def run_graph_rag(query_original: str) -> Dict[str, Any]:
    user_lang = detect_user_language(query_original)
    log(f"[GraphRAG] Detected user language: {user_lang}", level="INFO")

    # Construct ChunkStore per original pipeline (unchanged behavior)
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

    # Telemetry
    quote_fb_count = sum(1 for _, _, t, _ in chunks_ranked if t.get("_is_quote_fallback"))
    telem = {
        "user_lang": user_lang,
        "entities_count": len(ents),
        "predicates_count": len(preds),
        "query_triples_count": len(query_triples),
        "triple_centric_matches": len(ctx2_triples),
        "entity_centric_matches": len(ctx1_triples),
        "merged_triples": len(merged_triples),
        "selected_triples": len(triples_ranked),
        "collected_chunks": len(chunk_records),
        "selected_chunks": len(chunks_ranked),
        "quote_fallback_count": quote_fb_count,
        "chunk_scores": chunk_score_stats(chunks_ranked),
        "context_summary": clamp(context_summary, 4000),
        "answer_stats": citation_stats(answer),
        "examples": {
            "entities": [e.get("text") for e in ents[:5]],
            "predicates": preds[:5],
        }
    }

    return {
        "answer": answer,
        "telemetry": telem
    }

# ----------------- Aggregator Agent -----------------
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
                s += len(re.findall(pat, txt, flags=re.IGNORECASE))
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

# ----------------- Answer Judge Agent -----------------
JUDGE_SCHEMA = {
  "type": "object",
  "properties": {
    "accepted": {"type": "boolean"},
    "verdict": {"type": "string", "enum": ["sufficient", "insufficient", "irrelevant", "conflicting", "inconclusive"]},
    "problems": {"type": "array", "items": {"type": "string"}},
    "recommendations": {"type": "array", "items": {"type": "string"}},
    "brief_rationale": {"type": "string"},
    "confidence": {"type": "number"}
  },
  "required": ["accepted", "verdict"]
}

def build_telemetry_summary(naive_t: Dict[str, Any], graph_t: Dict[str, Any], agg: Dict[str, Any]) -> str:
    parts = []
    parts.append("NaiveRAG:")
    parts.append(f"- candidates={naive_t.get('candidates')} top_scores={naive_t.get('top_scores')}")
    parts.append(f"- answer_stats={naive_t.get('answer_stats')}")
    if naive_t.get("context_preview"):
        parts.append("- context_preview (truncated):")
        parts.append(clamp(naive_t.get("context_preview",""), 1200))

    parts.append("\nGraphRAG:")
    parts.append(f"- entities={graph_t.get('entities_count')} preds={graph_t.get('predicates_count')} q_triples={graph_t.get('query_triples_count')}")
    parts.append(f"- triple_centric={graph_t.get('triple_centric_matches')} entity_centric={graph_t.get('entity_centric_matches')}")
    parts.append(f"- merged_triples={graph_t.get('merged_triples')} selected_triples={graph_t.get('selected_triples')}")
    parts.append(f"- collected_chunks={graph_t.get('collected_chunks')} selected_chunks={graph_t.get('selected_chunks')} quote_fallbacks={graph_t.get('quote_fallback_count')}")
    parts.append(f"- chunk_scores={graph_t.get('chunk_scores')}")
    parts.append("- context_summary (truncated):")
    parts.append(clamp(graph_t.get("context_summary",""), 1600))
    parts.append(f"- answer_stats={graph_t.get('answer_stats')}")

    parts.append("\nAggregator Decision:")
    parts.append(f"- chosen={agg.get('chosen')} confidence={agg.get('confidence')} rationale={agg.get('rationale')}")
    return "\n".join(parts)

def answer_judge_agent(
    query: str,
    aggregated_answer: str,
    aggregator_decision: Dict[str, Any],
    naive_telem: Dict[str, Any],
    graph_telem: Dict[str, Any],
    history_triples: List[Dict[str, Any]],
    user_lang: str
) -> Dict[str, Any]:
    primer = """
You are the Answer Judge agent. You assess whether the current aggregated answer is sufficient, given how retrieval and aggregation work.

You know these pipelines:
- NaiveRAG:
  - Embeds the user query
  - Vector-search over chunk embeddings
  - Builds a chunk-only context
  - Answerer must only answer from retrieved chunks
- GraphRAG:
  - Agent1 extracts entities & predicates
  - Agent1b extracts query triples
  - Triple-centric retrieval via vector search on KG triples
  - Entity-centric subgraph expansion over predicates
  - Collects source chunks (with quote fallback if chunk missing)
  - Reranks chunks using the whole-query embedding
  - Builds combined context (triples summary + chunks)
  - Answerer must only answer from this context
- Aggregator:
  - Chooses or merges NaiveRAG/GraphRAG answers
  - Prefers specific, consistent, well-cited answers (UU, Pasal/Ayat)
  - Outputs chosen=naive|graphrag|mixed, with rationale and confidence

Your job:
- Decide if the aggregated answer sufficiently addresses the query.
- If not sufficient, explain the likely problem from a retrieval/prompt perspective and provide clear, actionable recommendations to improve the query so pipelines retrieve better evidence.

Be precise, avoid fabrications, and account for the telemetry.
"""
    telemetry_summary = build_telemetry_summary(naive_t=naive_telem, graph_t=graph_telem, agg=aggregator_decision)
    history_note = f"{len(history_triples)} prior iteration(s) recorded."
    prompt = f"""
{primer}

Question:
\"\"\"{query}\"\"\"

Aggregated answer (chosen={aggregator_decision.get('chosen')}, conf={aggregator_decision.get('confidence')}):
\"\"\"{aggregated_answer}\"\"\"

Telemetry summary:
\"\"\"{telemetry_summary}\"\"\"

History size: {history_note}

Return JSON with fields:
- accepted: boolean
- verdict: one of [sufficient, insufficient, irrelevant, conflicting, inconclusive]
- problems: list of short root causes (e.g., ambiguous_query_entity, missing_law_identifier, insufficient_citations, low_retrieval_coverage, entity_linking_error, triple_mismatch, aggregation_conflict, language_mismatch)
- recommendations: list of actionable steps to improve the query for better retrieval (e.g., add specific UU number, add Pasal/Ayat, add synonyms/aliases, narrow by date/version, name the ministry/regulation, clarify sector/entity, keep language consistent)
- brief_rationale: 1-3 sentences explaining your decision
- confidence: 0.0–1.0

Respond in JSON only.
"""
    log("\n[AnswerJudge] Prompt:", level="INFO")
    log(prompt, level="INFO")
    out = safe_generate_json(prompt, JUDGE_SCHEMA, temp=0.0) or {}
    accepted = bool(out.get("accepted", False))
    verdict = (out.get("verdict") or "inconclusive").strip().lower()
    problems = out.get("problems") or []
    recs = out.get("recommendations") or []
    rationale = (out.get("brief_rationale") or "").strip()
    conf = float(out.get("confidence") or 0.0)

    # Fallback heuristic if invalid JSON or empty fields
    if "accepted" not in out or verdict not in ("sufficient","insufficient","irrelevant","conflicting","inconclusive"):
        log("[AnswerJudge] Fallback heuristic activated.", level="WARN")
        # Simple acceptance heuristic: GraphRAG or Naive has citations and decent retrieval coverage
        g_cites = (graph_telem.get("answer_stats") or {}).get("citation_like_count", 0)
        n_cites = (naive_telem.get("answer_stats") or {}).get("citation_like_count", 0)
        coverage = (graph_telem.get("selected_chunks", 0) + graph_telem.get("selected_triples", 0))
        if (g_cites + n_cites) >= 2 and coverage >= 10:
            accepted = True; verdict = "sufficient"; conf = max(conf, 0.6)
            rationale = rationale or "Answer appears well-cited with adequate retrieval coverage."
        else:
            accepted = False; verdict = "insufficient"; conf = max(conf, 0.5)
            problems = problems or ["insufficient_citations" if (g_cites + n_cites) == 0 else "low_retrieval_coverage"]
            recs = recs or ["add UU number and Pasal/Ayat if known", "clarify the specific entity/ministry/regulation", "include aliases or alternative terms"]

    log(f"[AnswerJudge] Decision: accepted={accepted}, verdict={verdict}, confidence={conf:.2f}", level="INFO")
    if problems:
        log(f"[AnswerJudge] Problems: {problems}", level="INFO")
    if recs:
        log(f"[AnswerJudge] Recommendations: {recs}", level="INFO")
    if rationale:
        log(f"[AnswerJudge] Rationale: {rationale}", level="INFO")

    return {
        "accepted": accepted,
        "verdict": verdict,
        "problems": problems,
        "recommendations": recs,
        "brief_rationale": rationale,
        "confidence": conf
    }

# ----------------- Query Modifier Agent -----------------
QUERY_MOD_SCHEMA = {
  "type": "object",
  "properties": {
    "modified_query": {"type": "string"},
    "rationale": {"type": "string"},
    "changes_applied": {"type": "array", "items": {"type": "string"}},
    "preserve_intent": {"type": "boolean"}
  },
  "required": ["modified_query"]
}

def query_modifier_agent(
    current_query: str,
    judge_feedback: Dict[str, Any],
    history_pairs: List[Dict[str, Any]],
    user_lang: str
) -> Dict[str, Any]:
    primer = """
You are the Query Modifier agent. You know how the multi-pipeline RAG system works:

- NaiveRAG benefits from queries that contain concrete named entities, law identifiers (UU No./Year), Pasal/Ayat numbers, and key terms used in the source.
- GraphRAG benefits from explicit entities, predicates (e.g., mewajibkan, melarang), and short subject–predicate–object phrasing. Aliases/synonyms improve entity linking and triple matches.
- Both pipelines perform better when the query is specific, uses canonical names, and avoids vague or multi-intent phrasing.

Your goals:
- Apply the judge's recommendations to produce a modified query that preserves the user's original intent but improves retrievability.
- Do not fabricate law numbers or articles. If unknown, prefer adding clarifying constraints (ministry, topic, sector, date range/version) or synonyms/aliases.
- Keep the user’s language unless the judge flagged a language mismatch.
- Be concise, retrieval-friendly, and avoid unnecessary filler.
"""
    history_summ = []
    for h in history_pairs[-5:]:
        history_summ.append(f"- iter={h.get('iteration')} query=\"{h.get('query')}\" feedback={h.get('feedback',{})}")
    history_text = "\n".join(history_summ) if history_summ else "(no prior query-feedback pairs)"

    prompt = f"""
{primer}

Current query:
\"\"\"{current_query}\"\"\"

Judge feedback:
{json.dumps(judge_feedback, ensure_ascii=False)}

Recent query-feedback history:
{history_text}

Produce JSON with:
- modified_query: single best rewritten query (do not include commentary)
- rationale: why these changes help retrieval
- changes_applied: bullets of changes (e.g., "added Pasal", "added UU alias", "clarified ministry")
- preserve_intent: boolean

Respond in JSON only.
"""
    log("\n[QueryModifier] Prompt:", level="INFO")
    log(prompt, level="INFO")
    out = safe_generate_json(prompt, QUERY_MOD_SCHEMA, temp=0.2) or {}
    modified = (out.get("modified_query") or "").strip()
    rationale = (out.get("rationale") or "").strip()
    changes = out.get("changes_applied") or []
    preserve = bool(out.get("preserve_intent", True))

    if not modified:
        log("[QueryModifier] Fallback rewrite applied due to empty output.", level="WARN")
        # Simple fallback: append clarifier hints from feedback
        fb = judge_feedback or {}
        hints = []
        for rec in fb.get("recommendations", []):
            if isinstance(rec, str):
                hints.append(rec)
        hint_text = "; ".join(hints[:2]) if hints else "jelaskan UU/Pasal/Ayat yang dimaksud"
        modified = f"{current_query} (Harap tentukan UU/Pasal/Ayat atau instansi/regulasi terkait; {hint_text})"
        rationale = rationale or "Appended clarifiers to increase specificity for retrieval."
        changes = changes or ["appended_clarifiers"]

    log(f"[QueryModifier] Modified query: {modified}", level="INFO")
    if rationale:
        log(f"[QueryModifier] Rationale: {rationale}", level="INFO")
    if changes:
        log(f"[QueryModifier] Changes: {changes}", level="INFO")

    return {
        "modified_query": modified,
        "rationale": rationale,
        "changes_applied": changes,
        "preserve_intent": preserve
    }

# ----------------- Orchestrator (single-iteration pass retained) -----------------
def agentic_multi_singlepass(query_original: str) -> Dict[str, Any]:
    t_all = now_ms()
    user_lang = detect_user_language(query_original)
    log("=== Single-pass Multi-Agent RAG run started ===", level="INFO")
    log(f"Original Query: {query_original}", level="INFO")
    log(f"[Language] Detected user language: {user_lang}", level="INFO")

    # Run Naive RAG
    naive_result: Dict[str, Any] = {}
    try:
        t_nv = now_ms()
        naive_result = run_naive_rag(query_original)
        log(f"[NaiveRAG] Completed in {dur_ms(t_nv):.0f} ms", level="INFO")
    except Exception as e:
        log(f"[NaiveRAG] Error: {e}", level="ERROR")
        naive_result = {"answer": "", "telemetry": {"error": str(e)}}

    # Run GraphRAG
    graph_result: Dict[str, Any] = {}
    try:
        t_gr = now_ms()
        graph_result = run_graph_rag(query_original)
        log(f"[GraphRAG] Completed in {dur_ms(t_gr):.0f} ms", level="INFO")
    except Exception as e:
        log(f"[GraphRAG] Error: {e}", level="ERROR")
        graph_result = {"answer": "", "telemetry": {"error": str(e)}}

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
    log("\n=== Single-pass summary ===", level="INFO")
    log(f"- Aggregator: chosen={decision['chosen']}, confidence={decision['confidence']:.2f}", level="INFO")
    log(f"- Total runtime: {total_ms:.0f} ms", level="INFO")
    log("\n=== Final Answer ===", level="INFO")
    log(final_answer, level="INFO")

    return {
        "final_answer": final_answer,
        "naive_answer": naive_answer,
        "graphrag_answer": graphrag_answer,
        "aggregator_decision": decision,
        "naive_telem": naive_result.get("telemetry", {}),
        "graph_telem": graph_result.get("telemetry", {}),
        "iterations": 1
    }

# ----------------- Iterative Agentic Orchestrator -----------------
def agentic_multi_iterative(query_original: str) -> Dict[str, Any]:
    global _LOGGER, _JSONL
    ts_name = make_timestamp_name()
    log_file = Path.cwd() / f"{ts_name}.txt"
    jsonl_file = Path.cwd() / f"{ts_name}.jsonl"
    _LOGGER = FileLogger(log_file, also_console=True)
    _JSONL = JSONLLogger(jsonl_file)

    history_pairs: List[Dict[str, Any]] = []   # [{iteration, query, feedback}]
    history_triples: List[Dict[str, Any]] = [] # [{iteration, query, aggregated_answer, aggregator_decision, naive_telem, graph_telem, judge_feedback}]

    user_lang = detect_user_language(query_original)
    current_query = query_original
    seen_queries: List[str] = []
    final_payload: Dict[str, Any] = {}
    start_all = now_ms()

    try:
        log("=== Agentic Multi-Iteration RAG run started ===", level="INFO")
        log(f"Process info: pid={_pid()}", level="INFO")
        log(f"Log file: {log_file}", level="INFO")
        log(f"JSONL history: {jsonl_file}", level="INFO")
        log(f"Original Query: {query_original}", level="INFO")
        log(f"Detected language: {user_lang}", level="INFO")
        log(f"Parameters:", level="INFO")
        log(f"  Iterative: AGENTIC_MAX_ITERS={AGENTIC_MAX_ITERS}", level="INFO")
        log(f"  GraphRAG: ENTITY_MATCH_TOP_K={GR_ENTITY_MATCH_TOP_K}, ENTITY_SUBGRAPH_HOPS={GR_ENTITY_SUBGRAPH_HOPS}, "
            f"ENTITY_SUBGRAPH_PER_HOP_LIMIT={GR_ENTITY_SUBGRAPH_PER_HOP_LIMIT}, SUBGRAPH_TRIPLES_TOP_K={GR_SUBGRAPH_TRIPLES_TOP_K}, "
            f"QUERY_TRIPLE_MATCH_TOP_K_PER={GR_QUERY_TRIPLE_MATCH_TOP_K_PER}, MAX_TRIPLES_FINAL={GR_MAX_TRIPLES_FINAL}, "
            f"MAX_CHUNKS_FINAL={GR_MAX_CHUNKS_FINAL}, CHUNK_RERANK_CAND_LIMIT={GR_CHUNK_RERANK_CAND_LIMIT}, "
            f"ANSWER_MAX_TOKENS={GR_ANSWER_MAX_TOKENS}, MAX_ITERS(inner)={MAX_ITERS}", level="INFO")
        log(f"  Naive: TOP_K_CHUNKS={NV_TOP_K_CHUNKS}, MAX_CHUNKS_FINAL={NV_MAX_CHUNKS_FINAL}, CHUNK_TEXT_CLAMP={NV_CHUNK_TEXT_CLAMP}", level="INFO")
        log(f"  LLM throttling (Code2-style): EMBED(qps={LLM_EMBED_QPS}, max_conc={LLM_EMBED_MAX_CONCURRENCY}), GEN(qps={LLM_GEN_QPS}, max_conc={LLM_GEN_MAX_CONCURRENCY})", level="INFO")
        log(f"  Neo4j concurrency cap: {NEO4J_MAX_CONCURRENCY or 'unlimited'}", level="INFO")

        for it in range(1, AGENTIC_MAX_ITERS + 1):
            t_iter = now_ms()
            log(f"\n========== [Iter {it}/{AGENTIC_MAX_ITERS}] ==========", level="INFO")
            log(f"[Iter {it}] Current query: {current_query}", level="INFO")

            # Guard against loops: check near-duplicates
            if any(seq_ratio(current_query, q) >= 0.97 for q in seen_queries):
                log(f"[Iter {it}] Query is highly similar to a previous one; minor rewrites detected. Proceeding but will avoid infinite loops.", level="WARN")
            seen_queries.append(current_query)

            # Run both pipelines + aggregator (sequential order preserved)
            naive_res = run_naive_rag(current_query)
            graph_res = run_graph_rag(current_query)
            agg = aggregator_agent(current_query, naive_res.get("answer",""), graph_res.get("answer",""), user_lang)

            aggregated_answer = agg.get("final_answer","")
            agg_decision = {
                "chosen": agg.get("chosen"),
                "confidence": float(agg.get("confidence") or 0.0),
                "rationale": agg.get("rationale") or "",
                "key_points": agg.get("key_points") or []
            }

            # Answer Judge
            judge = answer_judge_agent(
                query=current_query,
                aggregated_answer=aggregated_answer,
                aggregator_decision=agg_decision,
                naive_telem=naive_res.get("telemetry", {}),
                graph_telem=graph_res.get("telemetry", {}),
                history_triples=history_triples,
                user_lang=user_lang
            )

            # Build iteration record and persist to JSONL
            iter_record = {
                "iteration": it,
                "query": current_query,
                "aggregated_answer": aggregated_answer,
                "aggregator_decision": agg_decision,
                "naive_telemetry": naive_res.get("telemetry", {}),
                "graphrag_telemetry": graph_res.get("telemetry", {}),
                "judge_feedback": judge,
            }
            if _JSONL is not None:
                _JSONL.append(iter_record)

            history_triples.append(iter_record)

            # Acceptance or next step
            if judge.get("accepted", False):
                log(f"[Iter {it}] Judge accepted the answer. Finalizing.", level="INFO")
                final_payload = {
                    "final_answer": aggregated_answer,
                    "aggregator_decision": agg_decision,
                    "iterations": it,
                    "judge_feedback": judge,
                    "history_size": len(history_triples),
                    "log_file": str(log_file),
                    "jsonl_file": str(jsonl_file)
                }
                break

            if it >= AGENTIC_MAX_ITERS:
                log(f"[Iter {it}] Reached max iterations. Finalizing with current aggregated answer despite judge not accepting.", level="WARN")
                final_payload = {
                    "final_answer": aggregated_answer,
                    "aggregator_decision": agg_decision,
                    "iterations": it,
                    "judge_feedback": judge,
                    "history_size": len(history_triples),
                    "log_file": str(log_file),
                    "jsonl_file": str(jsonl_file)
                }
                break

            # Query Modifier
            qmod = query_modifier_agent(
                current_query=current_query,
                judge_feedback=judge,
                history_pairs=history_pairs,
                user_lang=user_lang
            )
            modified_query = qmod.get("modified_query","").strip() or current_query

            # Loop guard: if modified query is effectively unchanged, allow one minor tweak; else finalize
            similarity_to_prev = seq_ratio(modified_query, current_query)
            log(f"[Iter {it}] Modified query similarity to current: {similarity_to_prev:.3f}", level="INFO")

            history_pairs.append({
                "iteration": it,
                "query": current_query,
                "feedback": {
                    "problems": judge.get("problems"),
                    "recommendations": judge.get("recommendations")
                },
                "modified_query": modified_query
            })

            # Persist query-feedback pair to JSONL
            if _JSONL is not None:
                _JSONL.append({
                    "iteration": it,
                    "type": "query-feedback",
                    "query": current_query,
                    "feedback": {
                        "problems": judge.get("problems"),
                        "recommendations": judge.get("recommendations")
                    },
                    "modified_query": modified_query
                })

            # If modified query is too similar to multiple prior queries, avoid infinite loop by finalizing
            if any(seq_ratio(modified_query, q) >= 0.97 for q in seen_queries[-3:]):  # compare to last few queries
                log(f"[Iter {it}] Modified query appears too similar to recent queries; avoiding loop. Finalizing with current aggregated answer.", level="WARN")
                final_payload = {
                    "final_answer": aggregated_answer,
                    "aggregator_decision": agg_decision,
                    "iterations": it,
                    "judge_feedback": judge,
                    "history_size": len(history_triples),
                    "log_file": str(log_file),
                    "jsonl_file": str(jsonl_file)
                }
                break

            # Continue with modified query
            current_query = modified_query
            log(f"[Iter {it}] Proceeding to next iteration with modified query.", level="INFO")
            log(f"[Iter {it}] Iteration duration: {dur_ms(t_iter):.0f} ms", level="INFO")

        # End iterations
        total_ms = dur_ms(start_all)
        log("\n=== Agentic Multi-Iteration Summary ===", level="INFO")
        log(f"- Total iterations used: {final_payload.get('iterations', len(history_triples))}", level="INFO")
        log(f"- Total runtime: {total_ms:.0f} ms", level="INFO")
        log("\n=== Final Answer ===", level="INFO")
        log(final_payload.get("final_answer",""), level="INFO")
        log(f"\nLogs saved to: {log_file}", level="INFO")
        log(f"JSONL history saved to: {jsonl_file}", level="INFO")

        return final_payload

    finally:
        if _LOGGER is not None:
            _LOGGER.close()
        if _JSONL is not None:
            _JSONL.close()

# ----------------- Main -----------------
if __name__ == "__main__":
    try:
        user_query = input("Enter your query: ").strip()
        if not user_query:
            print("Empty query. Exiting.")
        else:
            # Run iterative agentic process
            result = agentic_multi_iterative(user_query)
            print("\n----- Final Answer -----")
            print(result.get("final_answer", ""))

    finally:
        try:
            driver.close()
        except Exception:
            pass