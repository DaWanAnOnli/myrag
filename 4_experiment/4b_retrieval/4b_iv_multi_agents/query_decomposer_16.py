#!/usr/bin/env python3
# multi-agent.py
# Multi-Agent RAG with Query Decomposer + Aggregator:
# - Decomposer agent can:
#    (1) split a query into sub-queries for NaiveRAG vs GraphRAG,
#    (2) pick a primary pipeline and send a supporting sub-query to the other,
#    (3) send the original query to both pipelines if appropriate.
# - NaiveRAG: Answerer-only, vector search over chunk_embedding_index
# - GraphRAG: Single-pass; Agents 1 & 2, entity/triple reasoning, KG expansion
# - Aggregator agent synthesizes final answer from both pipelines' outputs.
#
# Notes:
# - Self-contained; does not import your other scripts.
# - Compatible with run-multi-agent.py (same function name and return shape).
# - Requires: google-generativeai, neo4j, python-dotenv
# - Expects .env at ../../../.env (same convention as your other scripts)

import os, sys, time, json, math, pickle, re, random
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, Set
from threading import Lock
from collections import deque

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

# GraphRAG parameters
ENTITY_MATCH_TOP_K = int(os.getenv("ENTITY_MATCH_TOP_K", "15"))
ENTITY_SUBGRAPH_HOPS = int(os.getenv("ENTITY_SUBGRAPH_HOPS", "5"))
ENTITY_SUBGRAPH_PER_HOP_LIMIT = int(os.getenv("ENTITY_SUBGRAPH_PER_HOP_LIMIT", "2000"))
SUBGRAPH_TRIPLES_TOP_K = int(os.getenv("SUBGRAPH_TRIPLES_TOP_K", "30"))
QUERY_TRIPLE_MATCH_TOP_K_PER = int(os.getenv("QUERY_TRIPLE_MATCH_TOP_K_PER", "20"))

MAX_TRIPLES_FINAL = int(os.getenv("MAX_TRIPLES_FINAL", "60"))
MAX_CHUNKS_FINAL  = int(os.getenv("MAX_CHUNKS_FINAL", "40"))
CHUNK_RERANK_CAND_LIMIT = int(os.getenv("CHUNK_RERANK_CAND_LIMIT", "200"))

# Naive RAG parameters
TOP_K_CHUNKS = int(os.getenv("TOP_K_CHUNKS", "40"))
NAIVE_MAX_CHUNKS_FINAL = int(os.getenv("NAIVE_MAX_CHUNKS_FINAL", str(MAX_CHUNKS_FINAL)))
CHUNK_TEXT_CLAMP = int(os.getenv("CHUNK_TEXT_CLAMP", "1000000"))

# LLM / retries
ANSWER_MAX_TOKENS = int(os.getenv("ANSWER_MAX_TOKENS", "4096"))
LLM_CALLS_PER_MINUTE = int(os.getenv("LLM_CALLS_PER_MINUTE", "13"))
EMBEDDING_CALLS_PER_MINUTE = int(os.getenv("EMBEDDING_CALLS_PER_MINUTE", "0"))  # 0 disables

NEO4J_TX_TIMEOUT_S = float(os.getenv("NEO4J_TX_TIMEOUT_S", "60"))
NEO4J_MAX_ATTEMPTS = int(os.getenv("NEO4J_MAX_ATTEMPTS", "10"))

# ChunkStore path
DEFAULT_LANGCHAIN_DIR = (Path(__file__).resolve().parent / "../../../dataset/3_indexing/3a_langchain_results/").resolve()
LANGCHAIN_DIR = Path(os.getenv("LANGCHAIN_DIR") or str(DEFAULT_LANGCHAIN_DIR))
SKIP_FILES = {"all_langchain_documents.pkl"}

# Index names
INDEX_CHUNK = os.getenv("INDEX_CHUNK", "chunk_embedding_index")
INDEX_DOCUMENT = os.getenv("INDEX_DOCUMENT", "document_vec")
INDEX_CONTENT  = os.getenv("INDEX_CONTENT", "content_vec")
INDEX_EXPR     = os.getenv("INDEX_EXPR", "expression_vec")
INDEX_TRIPLE   = os.getenv("INDEX_TRIPLE", "triple_vec")

# ----------------- Initialize SDKs -----------------
if not GOOGLE_API_KEY:
    # Runner may inject the key later. Calls will fail if not set by then.
    pass
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

    def log(self, msg: Any = "", level: str = "INFO"):
        if not isinstance(msg, str):
            try:
                msg = json.dumps(msg, ensure_ascii=False, default=str)
            except Exception:
                msg = str(msg)
        lines = msg.splitlines() or [msg]
        out = "\n".join([f"{_prefix(level)} {ln}" for ln in lines])
        try:
            self._fh.write(out + "\n")
            self._fh.flush()
        except Exception:
            pass
        if self.also_console:
            print(out, flush=True)

    def close(self):
        try:
            self._fh.flush()
            self._fh.close()
        except Exception:
            pass

_LOGGER: Optional[FileLogger] = None

def log(msg: Any = "", level: str = "INFO"):
    if _LOGGER is not None:
        _LOGGER.log(msg, level=level)
    else:
        if not isinstance(msg, str):
            try:
                msg = json.dumps(msg, ensure_ascii=False, default=str)
            except Exception:
                msg = str(msg)
        lines = msg.splitlines() or [msg]
        print("\n".join([f"{_prefix(level)} {ln}" for ln in lines]), flush=True)

def make_timestamp_name() -> str:
    t = time.time()
    base = time.strftime("%Y%m%d-%H%M%S", time.localtime(t))
    ms = int((t % 1) * 1000)
    return f"{base}-{ms:03d}"

# ----------------- Utilities -----------------
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
    na = math.sqrt(sum(x*x for x in a)) or 1.0
    nb = math.sqrt(sum(y*y for y in b)) or 1.0
    if na == 0 or nb == 0: return 0.0
    return dot / (na * nb)

# --- Rate limiter ---
class RateLimiter:
    def __init__(self, calls_per_minute: int, name: str = "LLM"):
        self.cpm = max(0, int(calls_per_minute))
        self.name = name
        self.window = deque()
        self.win_s = 60.0
    def wait_for_slot(self):
        if self.cpm <= 0:
            return
        while True:
            now = time.monotonic()
            while self.window and (now - self.window[0]) >= self.win_s:
                self.window.popleft()
            if len(self.window) < self.cpm:
                self.window.append(now)
                return
            sleep_time = self.win_s - (now - self.window[0])
            sleep_time = max(0.01, sleep_time)
            log(f"[RateLimit:{self.name}] Sleeping {sleep_time:.2f}s to respect {self.name}_CALLS_PER_MINUTE={self.cpm}", level="DEBUG")
            time.sleep(sleep_time)

_LLM_RATE_LIMITER = RateLimiter(LLM_CALLS_PER_MINUTE, name="LLM")
_EMBED_RATE_LIMITER = RateLimiter(EMBEDDING_CALLS_PER_MINUTE, name="EMBED")

def _rand_wait_seconds() -> float:
    return random.uniform(5.0, 20.0)

def _api_call_with_retry(func, *args, **kwargs):
    while True:
        try:
            if func.__name__ == "embed_content":
                _EMBED_RATE_LIMITER.wait_for_slot()
            else:
                _LLM_RATE_LIMITER.wait_for_slot()
            return func(*args, **kwargs)
        except Exception as e:
            wait_s = _rand_wait_seconds()
            log(f"[Retry] API call failed: {e}. Retrying in {wait_s:.1f}s.", level="WARN")
            time.sleep(wait_s)

def embed_text(text: str) -> List[float]:
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
        raise RuntimeError("Unexpected embedding response shape")
    return _as_float_list(vec)

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
        if isinstance(resp.text, str) and resp.text.strip():
            return resp.text.strip()
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
    cfg = GenerationConfig(temperature=temp, response_mime_type="application/json", response_schema=schema)
    t0 = time.time()
    resp = _api_call_with_retry(gen_model.generate_content, prompt, generation_config=cfg)
    took = (time.time()-t0)*1000
    log(f"[LLM JSON] call completed in {took:.0f} ms", level="DEBUG")
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
    t0 = time.time()
    resp = _api_call_with_retry(gen_model.generate_content, prompt, generation_config=cfg)
    took = (time.time()-t0)*1000
    text = extract_text_from_response(resp)
    if text:
        log(f"[LLM TEXT] call completed in {took:.0f} ms, len={len(text)}", level="DEBUG")
        return text
    info = get_finish_info(resp)
    log(f"[LLM text warning] No text returned. Diagnostics: {info}. Took={took:.0f} ms", level="WARN")
    return f"(Model returned no text. finish_info={info})"

# ----------------- Language detection (id/en) -----------------
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
        t0 = time.time()
        log(f"[Neo4j] Attempt {attempts}/{NEO4J_MAX_ATTEMPTS} | qid={qid} | timeout={NEO4J_TX_TIMEOUT_S:.1f}s | Cypher=\"{preview}\" | Params: {param_summary}")
        try:
            with driver.session() as session:
                res = session.run(cypher, **params, timeout=NEO4J_TX_TIMEOUT_S)
                records = list(res)
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

# ----------------- GraphRAG agents and helpers -----------------
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
    "predicates": {"type": "array", "items": {"type": "string"}}
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

Output JSON keys:
- "entities": array of {{text, type(optional in {LEGAL_ENTITY_TYPES})}}
- "predicates": array of strings (Indonesian, snake_case when applicable)

User question:
\"\"\"{query}\"\"\"
"""
    est = estimate_tokens_for_text(prompt)
    log("\n[Agent 1] Prompt:")
    log(prompt)
    log(f"[Agent 1] Prompt size: {len(prompt)} chars, est_tokens≈{est}")
    out = safe_generate_json(prompt, QUERY_SCHEMA, temp=0.0) or {}
    if "entities" not in out: out["entities"] = []
    if "predicates" not in out: out["predicates"] = []
    log(f"[Agent 1] entities={ [e.get('text') for e in out['entities']] }, predicates={ out['predicates'] }")
    return out

def agent1b_extract_query_triples(query: str) -> List[Dict[str, Any]]:
    prompt = f"""
You are Agent 1b. Extract explicit or implied triples from the user's question in the form:
subject — predicate — object.

Rules:
- Use short, literal subject/object texts as they appear in the question.
- Predicates: concise, lowercase, snake_case if multiword.
- If type unknown, leave it blank.

Return JSON with key "triples".
User question:
\"\"\"{query}\"\"\"
"""
    est = estimate_tokens_for_text(prompt)
    log("\n[Agent 1b] Prompt:")
    log(prompt)
    log(f"[Agent 1b] Prompt size: {len(prompt)} chars, est_tokens≈{est}")
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
    log(f"[Agent 1b] Extracted query triples: {formatted}")
    return clean

def query_triple_to_text(t: Dict[str, Any]) -> str:
    s = ((t.get("subject") or {}).get("text") or "").strip()
    p = (t.get("predicate") or "").strip()
    o = ((t.get("object") or {}).get("text") or "").strip()
    return f"{s} [{p}] {o}"

def _vector_query_nodes(index_name: str, q_emb: List[float], k: int) -> List[Dict[str, Any]]:
    cypher = """
    CALL db.index.vector.queryNodes($index_name, $k, $q_emb) YIELD node AS n, score
    RETURN n, score, elementId(n) AS elem_id
    ORDER BY score DESC
    LIMIT $k
    """
    res = run_cypher_with_retry(cypher, {"index_name": index_name, "k": k, "q_emb": _as_float_list(q_emb)})
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
    for idx in [INDEX_DOCUMENT, INDEX_CONTENT, INDEX_EXPR]:
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
    cypher = f"""
    CALL db.index.vector.queryNodes('{INDEX_TRIPLE}', $k, $q_emb) YIELD node AS tr, score
    OPTIONAL MATCH (tr)-[:SUBJECT]->(s)
    OPTIONAL MATCH (tr)-[:OBJECT]->(o)
    RETURN tr, s, o, score
    ORDER BY score DESC
    LIMIT $k
    """
    res = run_cypher_with_retry(cypher, {"k": k, "q_emb": _as_float_list(q_emb)})
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

def expand_from_entities(entity_keys: List[str], hops: int, per_hop_limit: int, entity_elem_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
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
            s_id = r.get("s_id"); o_id = r.get("o_id")
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

    def _norm_id(self, x) -> str:
        return str(x).strip() if x is not None else ""

    def _build_index(self):
        if self._built:
            return
        start = time.monotonic()
        log(f"[ChunkStore] Building index from {self.root}...")
        pkls = [p for p in self.root.glob("*.pkl") if p.name not in self.skip]
        total = 0
        for pkl in pkls:
            try:
                with open(pkl, "rb") as f:
                    chunks = pickle.load(f)
                count = 0
                for ch in chunks:
                    meta = getattr(ch, "metadata", {}) if hasattr(ch, "metadata") else {}
                    doc_id = self._norm_id(meta.get("document_id"))
                    chunk_id = self._norm_id(meta.get("chunk_id"))
                    text = getattr(ch, "page_content", None)
                    if doc_id and chunk_id and isinstance(text, str):
                        self._index[(doc_id, chunk_id)] = text
                        self._by_chunk.setdefault(chunk_id, []).append((doc_id, chunk_id))
                        count += 1
                self._loaded_files.add(pkl)
                total += count
                log(f"[ChunkStore] Loaded {count} chunks from {pkl.name}")
            except Exception as e:
                log(f"[ChunkStore] Failed to load or process {pkl.name}: {e}", level="WARN")
        elapsed = time.monotonic() - start
        log(f"[ChunkStore] Index built. Total chunks indexed: {total} from {len(self._loaded_files)} files in {elapsed:.3f}s.")
        self._built = True

    def get_chunk(self, document_id: Any, chunk_id: Any) -> Optional[str]:
        if not self._built:
            self._build_index()
        d = self._norm_id(document_id)
        c = self._norm_id(chunk_id)
        val = self._index.get((d, c))
        if val is not None:
            log(f"[ChunkStore] HIT exact: doc={d} chunk={c} len={len(val)}", level="DEBUG")
            return val
        if "::" in c:
            base_id = c.split("::", 1)[0]
            val = self._index.get((d, base_id))
            if val is not None:
                log(f"[ChunkStore] HIT base-id: doc={d} chunk={c} -> base={base_id} len={len(val)}", level="DEBUG")
                return val
        matches = self._by_chunk.get(c)
        if matches:
            chosen_doc, chosen_chunk = matches[0]
            val = self._index.get((chosen_doc, chosen_chunk))
            if val is not None:
                note = "" if len(matches) == 1 else f" (warn: chunk_id occurs in {len(matches)} docs; chose doc={chosen_doc})"
                log(f"[ChunkStore] HIT by chunk_id only: requested doc={d} chunk={c}; using doc={chosen_doc}{note}. len={len(val)}", level="WARN")
                return val
        log(f"[ChunkStore] MISS: doc={d} chunk={c} (no match)", level="DEBUG")
        return None

def mean_similarity_to_query_triples(triple_emb: Optional[List[float]], q_trip_embs: List[List[float]]) -> float:
    if not isinstance(triple_emb, list) and triple_emb is not None:
        triple_emb = _as_float_list(triple_emb)
    if not isinstance(triple_emb, list) or not q_trip_embs:
        return 0.0
    sims = [cos_sim(triple_emb, q) for q in q_trip_embs]
    return sum(sims) / len(sims) if sims else 0.0

def entity_centric_retrieval(query_entities: List[Dict[str, Any]], q_trip_embs: List[List[float]], q_emb_fallback: Optional[List[float]] = None) -> List[Dict[str, Any]]:
    all_keys: Set[str] = set()
    all_ids: Set[str] = set()
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
        all_keys.update([m.get("key") for m in matches if m.get("key")])
        all_ids.update([m.get("elem_id") for m in matches if m.get("elem_id")])
    if not (all_keys or all_ids):
        log("[EntityRetrieval] No KG entity matches found from query entities.")
        return []
    t0 = time.time()
    expanded_triples = expand_from_entities(list(all_keys), hops=ENTITY_SUBGRAPH_HOPS, per_hop_limit=ENTITY_SUBGRAPH_PER_HOP_LIMIT, entity_elem_ids=list(all_ids) if all_ids else None)
    log(f"[EntityRetrieval] Expanded subgraph triples: {len(expanded_triples)} | {(time.time()-t0)*1000:.0f} ms")
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
    return ranked[:SUBGRAPH_TRIPLES_TOP_K]

def collect_chunks_for_triples(triples: List[Dict[str, Any]], chunk_store: ChunkStore) -> List[Tuple[Tuple[Any, Any], str, Dict[str, Any]]]:
    seen: Set[Tuple[Any, Any]] = set()
    out: List[Tuple[Tuple[Any, Any], str, Dict[str, Any]]] = []
    for t in triples:
        doc_id = t.get("document_id")
        chunk_id = t.get("chunk_id")
        if doc_id is None or chunk_id is None:
            quote = t.get("evidence_quote")
            if quote:
                key = (t.get("triple_uid"), "quote")
                if key not in seen:
                    t["_is_quote_fallback"] = True
                    out.append((key, quote, t))
                    seen.add(key)
            continue
        norm_key = (str(doc_id).strip(), str(chunk_id).strip())
        if norm_key in seen:
            continue
        text = chunk_store.get_chunk(doc_id, chunk_id)
        if text:
            t["_is_quote_fallback"] = False
            out.append((norm_key, text, t))
            seen.add(norm_key)
        else:
            quote = t.get("evidence_quote")
            if quote:
                key2 = (t.get("triple_uid"), "quote")
                if key2 not in seen:
                    t["_is_quote_fallback"] = True
                    log(f"[ChunkStore] FALLBACK to quote for doc={doc_id} chunk={chunk_id}", level="WARN")
                    out.append((key2, quote, t))
                    seen.add(key2)
    return out

def rerank_chunks_by_query(chunk_records: List[Tuple[Tuple[Any, Any], str, Dict[str, Any]]], q_emb_query: List[float], top_k: int) -> List[Tuple[Tuple[Any, Any], str, Dict[str, Any], float]]:
    cand = chunk_records[:CHUNK_RERANK_CAND_LIMIT]
    t0 = time.time()
    scored: List[Tuple[Tuple[Any, Any], str, Dict[str, Any], float]] = []
    for key, text, t in cand:
        try:
            emb = embed_text(text)
            s = cos_sim(q_emb_query, emb)
            scored.append((key, text, t, s))
        except Exception as ex:
            log(f"[ChunkRerank] Embedding failed for chunk {key}: {ex}", level="WARN")
    scored.sort(key=lambda x: x[3], reverse=True)
    log(f"[ChunkRerank] Scored {len(scored)} candidates | picked top {min(top_k, len(scored))} | {(time.time()-t0)*1000:.0f} ms")
    return scored[:top_k]

def rerank_triples_by_query_triples(triples: List[Dict[str, Any]], q_trip_embs: List[List[float]], q_emb_fallback: Optional[List[float]], top_k: int) -> List[Dict[str, Any]]:
    t0 = time.time()
    def score(t: Dict[str, Any]) -> float:
        emb = t.get("embedding")
        if q_trip_embs:
            return mean_similarity_to_query_triples(emb, q_trip_embs)
        if q_emb_fallback and isinstance(emb, list):
            return cos_sim(q_emb_fallback, emb)
        return 0.0
    ranked = sorted(triples, key=score, reverse=True)
    log(f"[TripleRerank] Input={len(triples)} | Output={min(top_k, len(ranked))} | {(time.time()-t0)*1000:.0f} ms")
    return ranked[:top_k]

def build_combined_context_text(triples_ranked: List[Dict[str, Any]], chunks_ranked: List[Tuple[Tuple[Any, Any], str, Dict[str, Any], float]]) -> Tuple[str, str, List[Dict[str, Any]]]:
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
    log("\n[Agent 2] Prompt:")
    log(prompt)
    log(f"[Agent 2] Prompt size: {len(prompt)} chars, est_tokens≈{est}")
    answer = safe_generate_text(prompt, max_tokens=ANSWER_MAX_TOKENS, temperature=0.2)
    log(f"[Agent 2] Answer length={len(answer)}")
    return answer

class GraphRAGAgent:
    def run(self, query_original: str) -> Dict[str, Any]:
        log("=== GraphRAG (single-pass) started ===")
        t_all = time.time()

        user_lang = detect_user_language(query_original)
        log(f"[Language] Detected user language: {user_lang}")

        chunk_store = ChunkStore(LANGCHAIN_DIR, set(SKIP_FILES))

        # Step 0: embed whole query
        t0 = time.time()
        q_emb_query = embed_text(query_original)
        log(f"[G.Step0] Whole-query embedded in {(time.time()-t0)*1000:.0f} ms")

        # Agent 1
        t1 = time.time()
        extraction = agent1_extract_entities_predicates(query_original)
        ents = extraction.get("entities", [])
        preds = extraction.get("predicates", [])
        log(f"[G.Step1] Entity/Predicate extraction in {(time.time()-t1)*1000:.0f} ms; ents={len(ents)}, preds={len(preds)}")

        # Agent 1b
        t1b = time.time()
        query_triples = agent1b_extract_query_triples(query_original)
        log(f"[G.Step1b] Query triple extraction in {(time.time()-t1b)*1000:.0f} ms; triples={len(query_triples)}")

        # Triple-centric retrieval
        t2 = time.time()
        triples_map: Dict[str, Dict[str, Any]] = {}
        q_trip_embs: List[List[float]] = []
        for qt in query_triples:
            try:
                emb = embed_text(query_triple_to_text(qt))
                q_trip_embs.append(emb)
            except Exception as ex:
                log(f"[G.Step2] Embedding failed for query triple: {ex}", level="WARN")
                continue
            matches = search_similar_triples_by_embedding(emb, k=QUERY_TRIPLE_MATCH_TOP_K_PER)
            for m in matches:
                uid = m.get("triple_uid")
                if not uid: continue
                if uid not in triples_map or (m.get("score", 0.0) > triples_map[uid].get("score", 0.0)):
                    triples_map[uid] = m
        ctx2_triples = list(triples_map.values())
        log(f"[G.Step2] Triple-centric collected {len(ctx2_triples)} triples in {(time.time()-t2)*1000:.0f} ms")

        # Entity-centric retrieval
        t3 = time.time()
        ctx1_triples = entity_centric_retrieval(ents, q_trip_embs=q_trip_embs, q_emb_fallback=q_emb_query)
        log(f"[G.Step3] Entity-centric returned {len(ctx1_triples)} triples in {(time.time()-t3)*1000:.0f} ms")

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
        log(f"[G.Step4] Merged triples: {len(merged_triples)} in {(time.time()-t4)*1000:.0f} ms")

        # Collect and rerank chunks
        t5 = time.time()
        chunk_records = collect_chunks_for_triples(merged_triples, chunk_store)
        log(f"[G.Step5] Collected {len(chunk_records)} chunk candidates (pre-rerank)")
        chunks_ranked = rerank_chunks_by_query(chunk_records, q_emb_query, top_k=MAX_CHUNKS_FINAL)
        log(f"[G.Step5] Selected {len(chunks_ranked)} chunks in {(time.time()-t5)*1000:.0f} ms")

        # Rerank triples
        t6 = time.time()
        triples_ranked = rerank_triples_by_query_triples(merged_triples, q_trip_embs=q_trip_embs, q_emb_fallback=q_emb_query, top_k=MAX_TRIPLES_FINAL)
        log(f"[G.Step6] Selected {len(triples_ranked)} triples in {(time.time()-t6)*1000:.0f} ms")

        # Build context + answer
        t7 = time.time()
        context_text, context_summary, _ = build_combined_context_text(triples_ranked, chunks_ranked)
        log(f"[G.Context] Built in {(time.time()-t7)*1000:.0f} ms")
        log("\n[G.Context summary]:")
        log(context_summary)

        t8 = time.time()
        answer = agent2_answer(query_original, context_text, guidance=None, output_lang=user_lang)
        log(f"[G.Step7] Answer generated in {(time.time()-t8)*1000:.0f} ms")

        total_ms = (time.time()-t_all)*1000
        log(f"=== GraphRAG finished in {total_ms:.0f} ms ===")
        return {
            "answer": answer,
            "context_summary": context_summary,
            "chunks_used": len(chunks_ranked),
            "triples_used": len(triples_ranked),
            "duration_ms": total_ms
        }

# ----------------- Naive RAG -----------------
def vector_query_chunks(q_emb: List[float], k: int) -> List[Dict[str, Any]]:
    cypher = f"""
    WITH $q AS q
    CALL db.index.vector.queryNodes('{INDEX_CHUNK}', $k, q)
    YIELD node, score
    RETURN node AS c, score
    ORDER BY score DESC
    LIMIT $k
    """
    rows = run_cypher_with_retry(cypher, {"q": _as_float_list(q_emb), "k": k})
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
        txt = clamp(c.get("content") or "", CHUNK_TEXT_CLAMP)
        lines.append(f"[Chunk {i}] doc={doc} chunk={cid} | {uu} | pages={pg} | score={c.get('score'):.3f}\n{txt}")
    return "\n".join(lines)

class NaiveRAGAgent:
    def run(self, query_original: str) -> Dict[str, Any]:
        log("=== Naive RAG (Answerer-only) started ===")
        t_all = time.time()

        user_lang = detect_user_language(query_original)
        log(f"[Language] Detected user language: {user_lang}")

        # Step 0: whole-query embedding
        t0 = time.time()
        q_emb = embed_text(query_original)
        log(f"[N.Step0] Embedded query in {(time.time()-t0)*1000:.0f} ms")

        # Optional Agent1/1b (for logs only)
        t1 = time.time()
        _ = agent1_extract_entities_predicates(query_original)
        _ = agent1b_extract_query_triples(query_original)
        log(f"[N.Step1] Entity/Triple extraction in {(time.time()-t1)*1000:.0f} ms")

        # Retrieval
        t2 = time.time()
        candidates = vector_query_chunks(q_emb, k=TOP_K_CHUNKS)
        log(f"[N.Step2] Vector search returned {len(candidates)} candidates in {(time.time()-t2)*1000:.0f} ms")

        # Build context
        if not candidates:
            context_text = "Tidak ada potongan teks yang ditemukan." if user_lang == "id" else "No relevant chunks found."
        else:
            context_text = build_context_from_chunks(candidates, max_chunks=NAIVE_MAX_CHUNKS_FINAL)
            log("[N.Context preview]:")
            prev = "\n".join(context_text.splitlines()[:30])
            log(prev)

        # Agent 2
        t3 = time.time()
        answer = agent2_answer(query_original, context_text, guidance=None, output_lang=user_lang)
        log(f"[N.Step3] Answer generated in {(time.time()-t3)*1000:.0f} ms")

        total_ms = (time.time()-t_all)*1000
        log(f"=== Naive RAG finished in {total_ms:.0f} ms ===")
        return {
            "answer": answer,
            "duration_ms": total_ms,
            "chunks_considered": len(candidates)
        }

# ----------------- Query Decomposer Agent -----------------
DECOMPOSER_SCHEMA = {
  "type":"object",
  "properties":{
    "strategy":{"type":"string","enum":[
        "split",
        "primary_naive_with_support_from_graphrag",
        "primary_graphrag_with_support_from_naive",
        "both_full",
        "single_naive",
        "single_graphrag"
    ]},
    "subqueries_naive":{"type":"array","items":{"type":"string"}},
    "subqueries_graphrag":{"type":"array","items":{"type":"string"}},
    "combine_instructions":{"type":"string"},
    "rationale":{"type":"string"},
    "signals":{"type":"array","items":{"type":"string"}},
    "confidence":{"type":"number"}
  },
  "required":["strategy","subqueries_naive","subqueries_graphrag"]
}

DECOMPOSER_GUIDANCE = """
You are the Query Decomposer agent for legal Q&A over Indonesian laws (UU, Pasal, Ayat, etc.).
You will decide how to route the user's question to two pipelines and optionally split it into sub-questions.

Pipelines:
- Naive RAG (vector search over text chunks; Answerer-only)
  Best for: localized answers likely in one paragraph/section; quoting/extracting a clause; definitions; specific Pasal/Ayat/UU references; dates, amounts, single-article scope, small focused summaries.

- GraphRAG (entity/triple reasoning; single-pass Agents 1 & 2)
  Best for: relational/multi-hop questions; combining multiple articles/UU; semantic predicates (mengubah, mencabut, mendelegasikan, berlaku_untuk, termuat_dalam); enumerations, comparisons, exceptions, scopes across entities.

Choose one strategy:
- "split": different aspects go to different pipelines (e.g., Naive to quote definitions; GraphRAG to assess relational scope).
- "primary_naive_with_support_from_graphrag": send original query (or close) to Naive; craft a supporting relational subquery for GraphRAG.
- "primary_graphrag_with_support_from_naive": send original query (or close) to GraphRAG; craft a supporting localized subquery for Naive (e.g., to fetch exact clause text).
- "both_full": the original query is meaningfully answerable by both pipelines; send it (or near variants) to both.
- "single_naive": clearly localized; only Naive needed.
- "single_graphrag": clearly relational/multi-hop; only GraphRAG needed.

Output strictly as JSON with:
- strategy: one of the above
- subqueries_naive: list of strings (empty allowed if none)
- subqueries_graphrag: list of strings (empty allowed if none)
- combine_instructions: optional guidance for the Aggregator on how to merge
- rationale: short reason grounded in the hints
- signals: short bullets of indicators (e.g., "explicit Pasal reference", "multi-article comparison", "mentions delegation")
- confidence: 0..1

Guidance for decomposition:
- Keep subqueries minimal and literal; if sending the original question to a pipeline, include it verbatim in that pipeline's list.
- For support queries, target the info that complements the primary (e.g., exact wording vs relationship/aggregation).
- Avoid inventing facts; only split what is explicitly or strongly implied by the user question.
"""

def decompose_query(query: str) -> Dict[str, Any]:
    prompt = f"""
{DECOMPOSER_GUIDANCE}

Question:
\"\"\"{query}\"\"\"
"""
    est = estimate_tokens_for_text(prompt)
    log("\n[Decomposer] Prompt:")
    log(prompt)
    log(f"[Decomposer] Prompt size: {len(prompt)} chars, est_tokens≈{est}")

    out = safe_generate_json(prompt, DECOMPOSER_SCHEMA, temp=0.0) or {}
    # Basic normalization
    out.setdefault("subqueries_naive", [])
    out.setdefault("subqueries_graphrag", [])
    out.setdefault("strategy", "both_full")
    out.setdefault("combine_instructions", "")
    out.setdefault("rationale", "")
    out.setdefault("signals", [])
    out.setdefault("confidence", 0.5)

    # Guardrails: if both lists empty, send original to both
    if not out["subqueries_naive"] and not out["subqueries_graphrag"]:
        out["strategy"] = "both_full"
        out["subqueries_naive"] = [query]
        out["subqueries_graphrag"] = [query]
        out["combine_instructions"] = (out.get("combine_instructions","") + " Use Naive for exact quotes; use GraphRAG for relationships.").strip()

    log(f"[Decomposer] Strategy={out.get('strategy')} | naive_q={len(out['subqueries_naive'])} | graphrag_q={len(out['subqueries_graphrag'])} | confidence={out.get('confidence')}")
    return out

# ----------------- Aggregator Agent -----------------
AGGREGATOR_SCHEMA = {
  "type":"object",
  "properties":{
    "chosen":{"type":"string","enum":["naive","graphrag","mixed"]},
    "final_answer":{"type":"string"},
    "rationale":{"type":"string"},
    "confidence":{"type":"number"},
    "key_points":{"type":"array","items":{"type":"string"}},
    "notes":{"type":"string"}
  },
  "required":["chosen","final_answer"]
}

def aggregator_decide(query: str, naive_answer: str, graphrag_answer: str, lang: str, plan: Dict[str, Any]) -> Dict[str, Any]:
    strategy = plan.get("strategy","")
    combine_instructions = plan.get("combine_instructions","")
    prompt = f"""
You are the Aggregator agent. You will receive:
- the original question,
- the combined output from Naive RAG (Answerer-only, localized chunks),
- the combined output from GraphRAG (triple/graph reasoning),
- the decomposer's strategy and any combine_instructions.

Goal:
- Produce the best possible final answer in the SAME language as the question ("{lang}").
- If one pipeline clearly provides the most accurate, well-cited answer, you may choose it.
- If both provide complementary insights (likely when strategy in ["split","both_full","primary_*_with_support_*"]), synthesize a concise, coherent answer that integrates both:
  - Use Naive-derived content for precise clause wording/definitions.
  - Use GraphRAG-derived content for relationships, scope, cross-article links, and enumerations.
- Prefer explicit legal references (UU/Article/Pasal/Ayat) and internal consistency.

Decomposer plan (strategy): {strategy}
Combine instructions:
\"\"\"{combine_instructions}\"\"\"

Return strict JSON with keys:
- chosen: "naive" | "graphrag" | "mixed"
- final_answer: the final answer text (same language as the question)
- rationale: brief reason for your choice
- confidence: number in [0,1]
- key_points: 3-8 bullets with essential points
- notes: optional

Question:
\"\"\"{query}\"\"\"

Naive RAG (combined):
\"\"\"{naive_answer}\"\"\"

GraphRAG (combined):
\"\"\"{graphrag_answer}\"\"\"
"""
    est = estimate_tokens_for_text(prompt)
    log("\n[Aggregator] Prompt:")
    log(prompt)
    log(f"[Aggregator] Prompt size: {len(prompt)} chars, est_tokens≈{est}")

    out = safe_generate_json(prompt, AGGREGATOR_SCHEMA, temp=0.2) or {}

    if not out.get("final_answer"):
        # Fallback heuristic
        log("[Aggregator] JSON missing or empty final_answer; falling back to heuristic.", level="WARN")
        a = (naive_answer or "").strip()
        b = (graphrag_answer or "").strip()
        chosen = "naive"
        final = a
        # Prefer graphrag if it carries relational/structured clues or is substantially richer
        if len(b) > len(a) and ("Pasal" in b or "UU" in b or "Article" in b or "Section" in b or "ayat" in b.lower()):
            chosen = "graphrag"
            final = b
        elif not a and b:
            chosen = "graphrag"; final = b
        elif not b and a:
            chosen = "naive"; final = a
        elif a and b and a != b:
            chosen = "mixed"
            final = f"{b}\nTambahan/Additional context:\n{a}" if lang == "id" else f"{b}\nAdditional context:\n{a}"
        out = {
            "chosen": chosen,
            "final_answer": final,
            "rationale": "Heuristic selection due to JSON failure",
            "confidence": 0.5,
            "key_points": [],
            "notes": ""
        }
    log(f"[Aggregator] Decision: chosen={out.get('chosen')} | confidence={out.get('confidence')}")
    return out

# ----------------- Helpers: run many subqueries per pipeline -----------------
def run_pipeline_many(pipeline_name: str, agent_cls, subqueries: List[str]) -> Tuple[str, List[str]]:
    """
    Runs a pipeline agent on a list of subqueries.
    Returns:
      - combined_answer: a readable aggregation of per-subquery answers
      - answers_list: the raw list of answers (same order as subqueries)
    """
    answers: List[str] = []
    if not subqueries:
        return "", answers
    t0 = time.time()
    for idx, q in enumerate(subqueries, 1):
        try:
            res = agent_cls().run(q)  # each run logs internally
            answers.append(res.get("answer","") or "")
        except Exception as e:
            log(f"[{pipeline_name}] Subquery #{idx} failed: {e}", level="WARN")
            answers.append(f"(error: {e})")
    took_ms = (time.time()-t0)*1000
    log(f"[{pipeline_name}] Completed {len(subqueries)} subqueries in {took_ms:.0f} ms")

    # Combine
    if len(subqueries) == 1:
        combined = answers[0] if answers else ""
    else:
        parts = []
        for i, (sq, ans) in enumerate(zip(subqueries, answers), 1):
            parts.append(f"[{pipeline_name} Subanswer {i}] For subquery: {sq}\n{ans}")
        combined = "\n".join(parts)
    return combined, answers

# ----------------- Orchestrator -----------------
def agentic_multi(query_original: str) -> Dict[str, Any]:
    global _LOGGER
    ts_name = make_timestamp_name()
    log_file = Path.cwd() / f"{ts_name}.txt"
    _LOGGER = FileLogger(log_file, also_console=True)

    t_all = time.time()
    try:
        log("=== Multi-Agent RAG (Decomposer + Aggregator) run started ===")
        log(f"Process info: pid={_pid()}")
        log(f"Log file: {log_file}")
        log(f"Original Query: {query_original}")
        log(f"Parameters: "
            f"GRAPH: ENTITY_MATCH_TOP_K={ENTITY_MATCH_TOP_K}, HOPS={ENTITY_SUBGRAPH_HOPS}, PER_HOP_LIMIT={ENTITY_SUBGRAPH_PER_HOP_LIMIT}, "
            f"SUBGRAPH_TRIPLES_TOP_K={SUBGRAPH_TRIPLES_TOP_K}, QUERY_TRIPLE_MATCH_TOP_K_PER={QUERY_TRIPLE_MATCH_TOP_K_PER}, "
            f"MAX_TRIPLES_FINAL={MAX_TRIPLES_FINAL}, MAX_CHUNKS_FINAL={MAX_CHUNKS_FINAL}, CHUNK_RERANK_CAND_LIMIT={CHUNK_RERANK_CAND_LIMIT}; "
            f"NAIVE: TOP_K_CHUNKS={TOP_K_CHUNKS}, NAIVE_MAX_CHUNKS_FINAL={NAIVE_MAX_CHUNKS_FINAL}, CHUNK_TEXT_CLAMP={CHUNK_TEXT_CLAMP}; "
            f"LLM: ANSWER_MAX_TOKENS={ANSWER_MAX_TOKENS}, LLM_CPM={LLM_CALLS_PER_MINUTE}, EMBED_CPM={EMBEDDING_CALLS_PER_MINUTE}; "
            f"Neo4j: MAX_ATTEMPTS={NEO4J_MAX_ATTEMPTS}, TX_TIMEOUT={NEO4J_TX_TIMEOUT_S}s")

        lang = detect_user_language(query_original)
        log(f"[Language] Detected: {lang}")

        # Decompose
        plan = {}
        try:
            plan = decompose_query(query_original)
        except Exception as e:
            log(f"[Decomposer] Failed: {e}. Falling back to both_full.", level="WARN")
            plan = {
                "strategy": "both_full",
                "subqueries_naive": [query_original],
                "subqueries_graphrag": [query_original],
                "combine_instructions": "Integrate both; use Naive for exact wording and GraphRAG for relationships.",
                "rationale": "Fallback due to decomposer error.",
                "signals": ["fallback_both"],
                "confidence": 0.45
            }

        naive_subqs = [s for s in (plan.get("subqueries_naive") or []) if isinstance(s, str) and s.strip()]
        graphrag_subqs = [s for s in (plan.get("subqueries_graphrag") or []) if isinstance(s, str) and s.strip()]
        log(f"[Plan] Strategy={plan.get('strategy')} | naive_subqs={len(naive_subqs)} | graphrag_subqs={len(graphrag_subqs)}")

        # Run pipelines as per plan
        naive_combined = ""
        graphrag_combined = ""

        if naive_subqs:
            naive_combined, _ = run_pipeline_many("NaiveRAG", NaiveRAGAgent, naive_subqs)
        if graphrag_subqs:
            graphrag_combined, _ = run_pipeline_many("GraphRAG", GraphRAGAgent, graphrag_subqs)

        # Aggregation
        agg = aggregator_decide(query_original, naive_combined, graphrag_combined, lang, plan)
        final_answer = (agg or {}).get("final_answer", "") or (graphrag_combined or naive_combined)

        total_ms = (time.time()-t_all)*1000
        log("\n=== Multi-Agent (Decomposer + Aggregator) summary ===")
        log(f"- Total runtime: {total_ms:.0f} ms")
        log(f"- Decomposer strategy: {plan.get('strategy')} (confidence={plan.get('confidence')})")
        log(f"- Aggregator choice: {agg.get('chosen')} (confidence={agg.get('confidence')})")
        log("\n=== Final Answer ===")
        log(final_answer)
        log(f"\nLogs saved to: {log_file}")

        return {
            "final_answer": final_answer,
            "naive_answer": naive_combined,
            "graphrag_answer": graphrag_combined,
            "aggregator_decision": {
                "chosen": agg.get("chosen"),
                "confidence": agg.get("confidence"),
                "rationale": agg.get("rationale"),
                "key_points": agg.get("key_points", []),
                "notes": agg.get("notes", "")
            },
            "decomposer_plan": {
                "strategy": plan.get("strategy"),
                "confidence": plan.get("confidence"),
                "rationale": plan.get("rationale"),
                "signals": plan.get("signals", []),
                "combine_instructions": plan.get("combine_instructions",""),
                "subqueries_naive": naive_subqs,
                "subqueries_graphrag": graphrag_subqs
            },
            "log_file": str(log_file),
            "timings_ms": {
                "total": total_ms
            }
        }
    finally:
        try:
            if _LOGGER is not None:
                _LOGGER.close()
        except Exception:
            pass

# ----------------- Main -----------------
if __name__ == "__main__":
    try:
        user_query = input("Enter your query: ").strip()
        if not user_query:
            print("Empty query. Exiting.")
        else:
            result = agentic_multi(user_query)
            print("\n" + "="*80)
            print("Final Answer:")
            print(result.get("final_answer",""))
            print("="*80 + "\n")
    finally:
        try:
            driver.close()
        except Exception:
            pass