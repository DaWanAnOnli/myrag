#!/usr/bin/env python3
"""
multi_agent.py
Self-contained Multi-Agent RAG that runs two approaches internally (no local imports):

1) GraphRAG (single-pass, Agents 1 & 2):
   - Agent 1: extract entities/predicates and query triples (LLM).
   - Retrieval A: entity-centric (match entities -> expand subgraph -> score triples).
   - Retrieval B: triple-centric (embed query triples -> match similar triples).
   - Collect supporting text chunks from local pickled dataset.
   - Agent 2: answer strictly from context.

2) NaiveRAG (Answerer-only):
   - Embed query -> vector search over TextChunk index -> Agent 2 answer.

Then an Aggregator Agent selects the better answer or synthesizes a hybrid.
The script logs to a timestamped file.

Environment (.env expected at ../../../../.env):
- GOOGLE_API_KEY[,_N]    (for Gemini)
- NEO4J_URI / NEO4J_USER / NEO4J_PASS
- GEN_MODEL (default: models/gemini-2.5-flash)
- EMBED_MODEL (default: models/text-embedding-004)
- Optional overrides: paths and numeric params below

Neo4j indexes expected:
- For GraphRAG entity search: document_vec, content_vec, expression_vec (db.index.vector.queryNodes)
- For GraphRAG triple search: triple_vec (db.index.vector.queryNodes)
- For NaiveRAG chunk search: chunk_embedding_index (db.index.vector.queryNodes)

Dataset (pickle chunks):
- Default: ../../../dataset/3_indexing/3a_langchain_results/*.pkl
  Can override with LANGCHAIN_DIR env var.
"""

import os, time, json, math, pickle, re, random, traceback
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
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
NEO4J_URI  = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASS", "password")

GEN_MODEL   = os.getenv("GEN_MODEL", "models/gemini-2.5-flash")
EMBED_MODEL = os.getenv("EMBED_MODEL", "models/text-embedding-004")

# GraphRAG params
ENTITY_MATCH_TOP_K = int(os.getenv("ENTITY_MATCH_TOP_K", "15"))
ENTITY_SUBGRAPH_HOPS = int(os.getenv("ENTITY_SUBGRAPH_HOPS", "2"))
ENTITY_SUBGRAPH_PER_HOP_LIMIT = int(os.getenv("ENTITY_SUBGRAPH_PER_HOP_LIMIT", "2000"))
SUBGRAPH_TRIPLES_TOP_K = int(os.getenv("SUBGRAPH_TRIPLES_TOP_K", "30"))
QUERY_TRIPLE_MATCH_TOP_K_PER = int(os.getenv("QUERY_TRIPLE_MATCH_TOP_K_PER", "20"))
MAX_TRIPLES_FINAL = int(os.getenv("MAX_TRIPLES_FINAL", "60"))
MAX_CHUNKS_FINAL = int(os.getenv("MAX_CHUNKS_FINAL", "40"))
CHUNK_RERANK_CAND_LIMIT = int(os.getenv("CHUNK_RERANK_CAND_LIMIT", "200"))
ANSWER_MAX_TOKENS = int(os.getenv("ANSWER_MAX_TOKENS", "4096"))
OUTPUT_LANG = os.getenv("OUTPUT_LANG", "id")
NEO4J_TX_TIMEOUT_S = float(os.getenv("NEO4J_TX_TIMEOUT_S", "60"))
NEO4J_MAX_ATTEMPTS = int(os.getenv("NEO4J_MAX_ATTEMPTS", "10"))

DEFAULT_LANGCHAIN_DIR = (Path(__file__).resolve().parent / "../../../dataset/3_indexing/3a_langchain_results/").resolve()
LANGCHAIN_DIR = Path(os.getenv("LANGCHAIN_DIR") or str(DEFAULT_LANGCHAIN_DIR))
SKIP_FILES = {"all_langchain_documents.pkl"}

# NaiveRAG params
TOP_K_CHUNKS = int(os.getenv("TOP_K_CHUNKS", "30"))
CHUNK_TEXT_CLAMP = int(os.getenv("CHUNK_TEXT_CLAMP", "1000000"))

# Rate limit
LLM_CALLS_PER_MINUTE = int(os.getenv("LLM_CALLS_PER_MINUTE", "10"))

# ----------------- Initialize SDKs -----------------
if GOOGLE_API_KEY:
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

def _make_log_name() -> str:
    t = time.time()
    base = time.strftime("%Y%m%d-%H%M%S", time.localtime(t))
    ms = int((t % 1) * 1000)
    return f"multi_agent_{base}-{ms:03d}.txt"

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
        head = f"[{_now_ts()}] [{level}] [pid={_pid()}]"
        line = f"{head} {msg}"
        self._fh.write(line + "\n")
        self._fh.flush()
        if self.also_console:
            print(line, flush=True)

    def close(self):
        try:
            self._fh.flush(); self._fh.close()
        except Exception:
            pass

_LOGGER: Optional[FileLogger] = None
def log(msg: Any = "", level: str = "INFO"):
    if _LOGGER is not None:
        _LOGGER.log(msg, level)
    else:
        print(f"[{_now_ts()}] [{level}] {msg}", flush=True)

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
    na = math.sqrt(sum(x*x for x in a))
    nb = math.sqrt(sum(y*y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)

# --- Simple per-process rate limiter (calls/minute) ---
class RateLimiter:
    def __init__(self, calls_per_minute: int):
        self.cpm = max(0, int(calls_per_minute))
        self.window = deque()
        self.win_sec = 60.0

    def wait(self):
        if self.cpm <= 0:
            return
        while True:
            now = time.monotonic()
            while self.window and (now - self.window[0]) >= self.win_sec:
                self.window.popleft()
            if len(self.window) < self.cpm:
                self.window.append(now)
                return
            sleep_time = self.win_sec - (now - self.window[0])
            time.sleep(max(0.01, sleep_time))

_RATE_LIMITER = RateLimiter(LLM_CALLS_PER_MINUTE)
def _rand_wait_seconds() -> float:
    return random.uniform(5.0, 20.0)

def _api_call_with_retry(func, *args, **kwargs):
    while True:
        try:
            _RATE_LIMITER.wait()
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
                if buf: return "".join(buf).strip()
    except Exception:
        pass
    return None

def safe_generate_json(prompt: str, schema: Dict[str, Any], temp: float = 0.0) -> Dict[str, Any]:
    cfg = GenerationConfig(temperature=temp, response_mime_type="application/json", response_schema=schema)
    resp = _api_call_with_retry(gen_model.generate_content, prompt, generation_config=cfg)
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
    resp = _api_call_with_retry(gen_model.generate_content, prompt, generation_config=cfg)
    text = extract_text_from_response(resp)
    if text: return text
    info = get_finish_info(resp)
    log(f"[LLM text warning] No text returned. Diagnostics: {info}", level="WARN")
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
            took = (time.time()-t0)*1000.0
            log(f"[Neo4j] Success | qid={qid} | rows={len(records)} | {took:.0f} ms")
            return records
        except Exception as e:
            took = (time.time()-t0)*1000.0
            last_e = e
            if attempts >= NEO4J_MAX_ATTEMPTS:
                break
            wait_s = _rand_wait_seconds()
            log(f"[Neo4j] Failure | qid={qid} | attempt={attempts}/{NEO4J_MAX_ATTEMPTS} | {took:.0f} ms | error={e}. Retrying in {wait_s:.1f}s.", level="WARN")
            time.sleep(wait_s)
    raise RuntimeError(f"Neo4j query failed after {NEO4J_MAX_ATTEMPTS} attempts (qid={qid}): {last_e}")

# ----------------- Chunk store -----------------
def _norm_id(x) -> str:
    return str(x).strip() if x is not None else ""

class ChunkStore:
    def __init__(self, root: Path, skip: Set[str]):
        self.root = root
        self.skip = skip
        self._index: Dict[Tuple[str, str], str] = {}
        self._by_chunk: Dict[str, List[Tuple[str, str]]] = {}
        self._built = False

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
                loaded = 0
                for ch in chunks:
                    meta = getattr(ch, "metadata", {}) if hasattr(ch, "metadata") else {}
                    doc_id = _norm_id(meta.get("document_id"))
                    chunk_id = _norm_id(meta.get("chunk_id"))
                    text = getattr(ch, "page_content", None)
                    if doc_id and chunk_id and isinstance(text, str):
                        self._index[(doc_id, chunk_id)] = text
                        self._by_chunk.setdefault(chunk_id, []).append((doc_id, chunk_id))
                        loaded += 1
                total += loaded
                log(f"[ChunkStore] Loaded {loaded} chunks from {pkl.name}")
            except Exception as e:
                log(f"[ChunkStore] Failed to load {pkl.name}: {e}", level="WARN")
        elapsed = time.monotonic() - start
        log(f"[ChunkStore] Index built. Total chunks: {total} in {elapsed:.2f}s.")
        self._built = True

    def get_chunk(self, document_id: Any, chunk_id: Any) -> Optional[str]:
        if not self._built:
            self._build_index()
        doc_id_s = _norm_id(document_id)
        chunk_id_s = _norm_id(chunk_id)
        # exact
        val = self._index.get((doc_id_s, chunk_id_s))
        if val is not None:
            return val
        # base-id rescue
        if "::" in chunk_id_s:
            base = chunk_id_s.split("::", 1)[0]
            val = self._index.get((doc_id_s, base))
            if val is not None:
                return val
        # chunk-only fallback
        matches = self._by_chunk.get(chunk_id_s)
        if matches:
            chosen_doc, chosen_chunk = matches[0]
            return self._index.get((chosen_doc, chosen_chunk))
        return None

# ----------------- Agent 1 / 1b (GraphRAG) -----------------
LEGAL_ENTITY_TYPES = ["UU", "PASAL", "AYAT", "INSTANSI", "ORANG", "ISTILAH", "SANKSI", "NOMINAL", "TANGGAL"]
LEGAL_PREDICATES  = ["mendefinisikan","mengubah","mencabut","mulai_berlaku","mewajibkan","melarang","memberikan_sanksi","berlaku_untuk","termuat_dalam","mendelegasikan_kepada","berjumlah","berdurasi"]

QUERY_SCHEMA = {
  "type": "object",
  "properties": {
    "entities": {
      "type": "array",
      "items": {"type":"object","properties":{"text":{"type":"string"},"type":{"type":"string"}},"required":["text"]}
    },
    "predicates": {"type":"array","items":{"type":"string"}}
  },
  "required": ["entities","predicates"]
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
    log("[Agent 1] Running extraction...")
    data = safe_generate_json(prompt, QUERY_SCHEMA, temp=0.0)
    if "entities" not in data: data["entities"] = []
    if "predicates" not in data: data["predicates"] = []
    return data

QUERY_TRIPLES_SCHEMA = {
  "type":"object",
  "properties":{
    "triples":{"type":"array","items":{
      "type":"object",
      "properties":{
        "subject":{"type":"object","properties":{"text":{"type":"string"},"type":{"type":"string"}},"required":["text"]},
        "predicate":{"type":"string"},
        "object":{"type":"object","properties":{"text":{"type":"string"},"type":{"type":"string"}},"required":["text"]}
      },
      "required":["subject","predicate","object"]
    }}
  },
  "required":["triples"]
}

def agent1b_extract_query_triples(query: str) -> List[Dict[str, Any]]:
    prompt = f"""
You are Agent 1b. Extract explicit or implied triples from the user's question in the form:
subject — predicate — object.

Rules:
- Use short, literal subject/object texts as they appear in the question.
- Predicates should be concise (lowercase, snake_case if multiword).
- If type is unknown, leave it blank.
- Do not invent; extract only what is clearly suggested by the question.

Return JSON with a key "triples".

User question:
\"\"\"{query}\"\"\"
"""
    log("[Agent 1b] Running triple extraction...")
    out = safe_generate_json(prompt, QUERY_TRIPLES_SCHEMA, temp=0.0) or {}
    triples = out.get("triples", []) if isinstance(out, dict) else []
    clean: List[Dict[str, Any]] = []
    for t in triples:
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
    return clean

def query_triple_to_text(t: Dict[str, Any]) -> str:
    s = ((t.get("subject") or {}).get("text") or "").strip()
    p = (t.get("predicate") or "").strip()
    o = ((t.get("object") or {}).get("text") or "").strip()
    return f"{s} [{p}] {o}"

# ----------------- GraphRAG: vector search helpers -----------------
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

# ----------------- Scoring helpers -----------------
def mean_similarity_to_query_triples(triple_emb: Optional[List[float]], q_trip_embs: List[List[float]]) -> float:
    if not isinstance(triple_emb, list) and triple_emb is not None:
        triple_emb = _as_float_list(triple_emb)
    if not isinstance(triple_emb, list) or not q_trip_embs:
        return 0.0
    sims = [cos_sim(triple_emb, q) for q in q_trip_embs]
    return sum(sims) / len(sims) if sims else 0.0

# ----------------- GraphRAG pipeline pieces -----------------
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
            matches = search_similar_entities_by_embedding(e_emb, k=ENTITY_MATCH_TOP_K)
            keys = [m.get("key") for m in matches if m.get("key")]
            ids  = [m.get("elem_id") for m in matches if m.get("elem_id")]
            all_matched_keys.update(keys); all_matched_ids.update(ids)
        except Exception as ex:
            log(f"[EntityRetrieval] Embedding failed for '{text}': {ex}", level="WARN")

    if not (all_matched_keys or all_matched_ids):
        log("[EntityRetrieval] No KG entity matches found.")
        return []

    expanded_triples = expand_from_entities(
        list(all_matched_keys),
        hops=ENTITY_SUBGRAPH_HOPS,
        per_hop_limit=ENTITY_SUBGRAPH_PER_HOP_LIMIT,
        entity_elem_ids=list(all_matched_ids) if all_matched_ids else None
    )
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

def triple_centric_retrieval(query_triples: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[List[float]]]:
    triples_map: Dict[str, Dict[str, Any]] = {}
    q_trip_embs: List[List[float]] = []
    for qt in query_triples:
        try:
            txt = query_triple_to_text(qt)
            emb = embed_text(txt)
            q_trip_embs.append(emb)
        except Exception as ex:
            log(f"[TripleRetrieval] Embedding failed for '{qt}': {ex}", level="WARN")
            continue

        matches = search_similar_triples_by_embedding(emb, k=QUERY_TRIPLE_MATCH_TOP_K_PER)
        for m in matches:
            uid = m.get("triple_uid")
            if not uid:
                continue
            if uid not in triples_map or (m.get("score", 0.0) > triples_map[uid].get("score", 0.0)):
                triples_map[uid] = m

    merged = list(triples_map.values())
    log(f"[TripleRetrieval] Collected {len(merged)} unique KG triples across {len(q_trip_embs)} query triple(s)")
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
    cand = chunk_records[:CHUNK_RERANK_CAND_LIMIT]
    scored: List[Tuple[Tuple[Any, Any], str, Dict[str, Any], float]] = []
    for key, text, t in cand:
        try:
            emb = embed_text(text)
            s = cos_sim(q_emb_query, emb)
            scored.append((key, text, t, s))
        except Exception as ex:
            log(f"[ChunkRerank] Embedding failed for chunk {key}: {ex}", level="WARN")
    scored.sort(key=lambda x: x[3], reverse=True)
    return scored[:top_k]

def rerank_triples_by_query_triples(
    triples: List[Dict[str, Any]],
    q_trip_embs: List[List[float]],
    q_emb_fallback: Optional[List[float]],
    top_k: int
) -> List[Dict[str, Any]]:
    def score(t: Dict[str, Any]) -> float:
        emb = t.get("embedding")
        if q_trip_embs:
            return mean_similarity_to_query_triples(emb, q_trip_embs)
        if q_emb_fallback and isinstance(emb, list):
            return cos_sim(q_emb_fallback, emb)
        return 0.0
    ranked = sorted(triples, key=score, reverse=True)
    return ranked[:top_k]

def build_combined_context_text(
    triples_ranked: List[Dict[str, Any]],
    chunks_ranked: List[Tuple[Tuple[Any, Any], str, Dict[str, Any], float]]
) -> Tuple[str, str, List[Dict[str, Any]]]:
    summary_lines = ["Ringkasan triple yang relevan:"]
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
    return safe_generate_text(prompt, max_tokens=ANSWER_MAX_TOKENS, temperature=0.2)

def agentic_graph_rag(query_original: str) -> Dict[str, Any]:
    log("=== GraphRAG (single-pass) ===")
    t_all = time.time()
    user_lang = detect_user_language(query_original)

    # Step 0: query embedding
    q_emb_query = embed_text(query_original)

    # Step 1: Agent 1 extract entities/predicates
    extraction = agent1_extract_entities_predicates(query_original)
    ents = extraction.get("entities", [])
    # Step 1b: Agent 1b triples
    query_triples = agent1b_extract_query_triples(query_original)

    # Step 2: triple-centric retrieval
    ctx2_triples, q_trip_embs = triple_centric_retrieval(query_triples)

    # Step 3: entity-centric retrieval
    ctx1_triples = entity_centric_retrieval(ents, q_trip_embs=q_trip_embs, q_emb_fallback=q_emb_query)

    # Step 4: merge triples
    triple_map: Dict[str, Dict[str, Any]] = {}
    for t in ctx1_triples + ctx2_triples:
        uid = t.get("triple_uid")
        if not uid: 
            continue
        if uid not in triple_map or (t.get("score", 0.0) > triple_map[uid].get("score", 0.0)):
            triple_map[uid] = t
    merged_triples = list(triple_map.values())

    # Step 5: gather chunks and rerank
    chunk_store = ChunkStore(LANGCHAIN_DIR, set(SKIP_FILES))
    chunk_records = collect_chunks_for_triples(merged_triples, chunk_store)
    chunks_ranked = rerank_chunks_by_query(chunk_records, q_emb_query, top_k=MAX_CHUNKS_FINAL)

    # Step 6: rerank triples
    triples_ranked = rerank_triples_by_query_triples(merged_triples, q_trip_embs=q_trip_embs, q_emb_fallback=q_emb_query, top_k=MAX_TRIPLES_FINAL)

    # Build context
    context_text, context_summary, _ = build_combined_context_text(triples_ranked, chunks_ranked)
    log("[GraphRAG] Context summary:")
    log(context_summary)

    # Step 7: Answer
    answer = agent2_answer(query_original, context_text, guidance=None, output_lang=user_lang)
    log(f"[GraphRAG] Done in {(time.time()-t_all):.2f}s.")
    return {
        "final_answer": answer,
        "iterations": 1
    }

# ----------------- NaiveRAG pipeline -----------------
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
        txt = clamp(c.get("content") or "", CHUNK_TEXT_CLAMP)
        lines.append(f"[Chunk {i}] doc={doc} chunk={cid} | {uu} | pages={pg} | score={c.get('score'):.3f}\n{txt}")
    return "\n".join(lines)

def agentic_naive_rag(query_original: str) -> Dict[str, Any]:
    log("=== NaiveRAG (answerer-only) ===")
    user_lang = detect_user_language(query_original)
    q_emb = embed_text(query_original)
    candidates = vector_query_chunks(q_emb, k=TOP_K_CHUNKS)
    if not candidates:
        context_text = "Tidak ada potongan teks yang ditemukan." if user_lang == "id" else "No relevant chunks found."
    else:
        context_text = build_context_from_chunks(candidates, max_chunks=MAX_CHUNKS_FINAL)
    answer = agent2_answer(query_original, context_text, guidance=None, output_lang=user_lang)
    return {"final_answer": answer}

# ----------------- Aggregator Agent -----------------
AGGREGATION_SCHEMA = {
    "type": "object",
    "properties": {
        "final_answer": {"type": "string"},
        "decision": {"type": "string", "description": "one of: graphrag | naiverag | hybrid"},
        "notes": {"type": "string"}
    },
    "required": ["final_answer", "decision"]
}

def aggregator_agent(query: str, answer_graphrag: str, answer_naive: str, user_lang: str) -> Dict[str, str]:
    lang_hint = "Bahasa Indonesia" if user_lang == "id" else "English"
    prompt = f"""
You are the Aggregator Agent. You receive:
- The user's question.
- Answer A (GraphRAG).
- Answer B (NaiveRAG).

Goal:
- Choose the better answer or synthesize a better combined answer.
- Prefer answers with clearer citations or more specific legal references.
- Do not invent new facts; rely on what's supported by the provided answers.
- Respond in {lang_hint}.
- If both answers are weak, pick the safer one and be transparent.
- Return JSON with: final_answer, decision (graphrag | naiverag | hybrid), and optional notes.

Question:
\"\"\"{query}\"\"\"

Answer A (GraphRAG):
\"\"\"{answer_graphrag or ''}\"\"\"

Answer B (NaiveRAG):
\"\"\"{answer_naive or ''}\"\"\"
"""
    out = safe_generate_json(prompt, AGGREGATION_SCHEMA, temp=0.2) or {}
    final_answer = (out.get("final_answer") or "").strip()
    decision = (out.get("decision") or "").strip().lower()
    notes = (out.get("notes") or "").strip()

    # Fallback if JSON didn't come back
    if not final_answer:
        join_prompt = f"""
Combine or pick the better of the two answers below to answer the question. Respond in {lang_hint}.
If one answer is clearly better, use it. Otherwise, synthesize a concise, accurate response.

Question:
\"\"\"{query}\"\"\"

Answer A:
\"\"\"{answer_graphrag or ''}\"\"\"

Answer B:
\"\"\"{answer_naive or ''}\"\"\"
"""
        final_answer = safe_generate_text(join_prompt, max_tokens=2048, temperature=0.2)
        decision = decision or ("hybrid" if answer_graphrag and answer_naive else ("graphrag" if answer_graphrag else "naiverag"))
        notes = notes or "Fallback textual merge."

    if decision not in ("graphrag", "naiverag", "hybrid"):
        decision = "hybrid"

    return {"final_answer": final_answer, "decision": decision, "notes": notes}

# ----------------- Top-level multi-agent entry -----------------
def multi_agent_answer(query_original: str) -> Dict[str, Any]:
    global _LOGGER
    log_path = Path.cwd() / _make_log_name()
    _LOGGER = FileLogger(log_path, also_console=True)

    try:
        log("=== Multi-Agent RAG run started ===")
        log(f"Original Query: {query_original}")
        log(f"GEN_MODEL={GEN_MODEL} | EMBED_MODEL={EMBED_MODEL}")
        user_lang = detect_user_language(query_original)

        # GraphRAG
        answer_graphrag = ""
        try:
            gr = agentic_graph_rag(query_original)
            answer_graphrag = (gr or {}).get("final_answer", "") or ""
        except Exception as e:
            log(f"[GraphRAG] ERROR: {e}\n{traceback.format_exc()}", level="ERROR")
            answer_graphrag = f"(GraphRAG error: {e})"

        # NaiveRAG
        answer_naive = ""
        try:
            nv = agentic_naive_rag(query_original)
            answer_naive = (nv or {}).get("final_answer", "") or ""
        except Exception as e:
            log(f"[NaiveRAG] ERROR: {e}\n{traceback.format_exc()}", level="ERROR")
            answer_naive = f"(NaiveRAG error: {e})"

        # Aggregate
        log("[Aggregator] Synthesizing final answer...")
        agg = aggregator_agent(query_original, answer_graphrag, answer_naive, user_lang)
        final_answer = agg.get("final_answer", "")
        decision = agg.get("decision", "hybrid")
        notes = agg.get("notes", "")

        log("=== Multi-Agent RAG summary ===")
        log(f"- Decision: {decision}")
        if notes:
            log(f"- Notes: {notes}")
        log(f"- Log file: {log_path}")

        return {
            "final_answer": final_answer,
            "answers": {
                "graphrag": answer_graphrag,
                "naiverag": answer_naive
            },
            "decision": decision,
            "notes": notes,
            "log_file": str(log_path)
        }
    finally:
        if _LOGGER is not None:
            _LOGGER.close()

# ----------------- CLI -----------------
if __name__ == "__main__":
    try:
        q = input("Enter your query: ").strip()
        if not q:
            print("Empty query. Exiting.")
        else:
            res = multi_agent_answer(q)
            print("\n=== Final Answer ===\n")
            print(res.get("final_answer", ""))
    finally:
        try:
            driver.close()
        except Exception:
            pass