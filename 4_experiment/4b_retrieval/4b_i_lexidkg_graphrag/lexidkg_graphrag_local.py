# lexidkg_graphrag_local.py
# Single-pass GraphRAG (Agent 1b + Agent 2 only), with structured per-question logging.
# Local version: uses LM Studio (OpenAI-compatible API) for LLM calls and BAAI/bge-m3 for embeddings.

import os, time, json, math, re, random
import numpy as np
import httpx
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, Set
from threading import Lock
import threading

from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from neo4j import GraphDatabase

# ----------------- Load .env (parent directory of this file) -----------------
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent.parent.parent / ".env")

# ----------------- Config -----------------
NEO4J_URI  = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASS", "password")

# Neo4j retry/timeout controls
NEO4J_TX_TIMEOUT_S = float(os.getenv("NEO4J_TX_TIMEOUT_S", "60"))
NEO4J_MAX_ATTEMPTS = int(os.getenv("NEO4J_MAX_ATTEMPTS", "5"))

# LM Studio local server
LMSTUDIO_BASE_URL = os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
LOCAL_GEN_MODEL   = os.getenv("LOCAL_GEN_MODEL",   "qwen/qwen3.5-9b")
LOCAL_EMBED_MODEL = os.getenv("LOCAL_EMBED_MODEL",  "BAAI/bge-m3")

# Per-request timeout in seconds for LMStudio LLM calls.
# Default 300 s – long enough for a large generation, short enough to detect
# dead connections before the process hangs indefinitely.
LLM_REQUEST_TIMEOUT = float(os.getenv("LLM_REQUEST_TIMEOUT", "300"))

# Number of questions that may be processed concurrently.
# N=1 → fully sequential. N>1 → sliding window: as soon as one finishes, next starts.
LLM_CONCURRENCY = int(os.getenv("LLM_CONCURRENCY", "15"))

# ----------------- Retrieval parameters -----------------
ENTITY_MATCH_TOP_K           = 10    # top similar KG entities per extracted query entity
ENTITY_SUBGRAPH_HOPS         = 4    # hop-depth for subgraph expansion
ENTITY_SUBGRAPH_PER_HOP_LIMIT = 2000
SUBGRAPH_TRIPLES_TOP_K       = 10   # top triples from subgraph after triple-vs-triple similarity

QUERY_TRIPLE_MATCH_TOP_K_PER = 10   # per query-triple, top similar KG triples

MAX_TRIPLES_FINAL      = 20
MAX_CHUNKS_FINAL       = 20
CHUNK_RERANK_CAND_LIMIT = 20

ANSWER_MAX_TOKENS = 4096

# ----------------- Initialize SDKs -----------------
# LM Studio client (OpenAI-compatible)
# Use a custom httpx transport with keep-alive disabled so every request
# gets a fresh TCP connection.  LM Studio silently drops keep-alive connections
# after a few requests which causes the OpenAI client to hang indefinitely on
# a socket that will never respond.
_http_transport = httpx.HTTPTransport(retries=0)
_http_client = httpx.Client(
    transport=_http_transport,
    timeout=httpx.Timeout(LLM_REQUEST_TIMEOUT),
    headers={"Connection": "close"},  # disable HTTP keep-alive
)
lmstudio_client = OpenAI(
    base_url=LMSTUDIO_BASE_URL,
    api_key="lm-studio",
    timeout=LLM_REQUEST_TIMEOUT,
    http_client=_http_client,
)
print(f"[Init] Using LMStudio generation model: {LOCAL_GEN_MODEL}")
print(f"[Init] LMStudio base URL: {LMSTUDIO_BASE_URL}")

# BGE-M3 embedding model (loaded once, thread-safe for inference)
import torch as _torch
_embed_device = "cuda" if _torch.cuda.is_available() else "cpu"
print(f"[Init] Loading embedding model: {LOCAL_EMBED_MODEL} on device={_embed_device} ...")
_bge_model = SentenceTransformer(LOCAL_EMBED_MODEL, trust_remote_code=True, device=_embed_device)
print(f"[Init] Embedding model loaded.")

# Embedding batch size: larger batches saturate the GPU better.
_EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "64"))

# Limit concurrent GPU embedding calls so batches don't oversubscribe the GPU.
_EMBED_SEMAPHORE = threading.Semaphore(3)

# Global semaphore to bound total concurrent Neo4j query operations.
# Set to 60% of max_connection_pool_size (200) to leave headroom for
# connection overhead and other clients. Threads block here rather than
# exhausting the pool and retrying indefinitely.
_NEO4J_QUERY_SEMAPHORE = threading.Semaphore(120)

# Global semaphore to bound how many questions can be in-flight simultaneously.
# This replaces the ThreadPoolExecutor max_workers as the primary backpressure
# mechanism — it gates the entire question lifecycle (LLM + Neo4j), not just
# individual Neo4j queries. Without this, fast questions finish their Neo4j
# steps, free their executor threads, and pull in more questions before the
# slower ones have finished, overwhelming LM Studio.
_QUESTION_SEMAPHORE = threading.Semaphore(LLM_CONCURRENCY)

driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=(NEO4J_USER, NEO4J_PASS),
    max_connection_pool_size=200,
    connection_acquisition_timeout=60,
)

# ----------------- Logger -----------------
def _now_ts() -> str:
    t = time.time()
    base = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t))
    ms = int((t % 1) * 1000)
    return f"{base}.{ms:03d}"

def _pid() -> int:
    return os.getpid()

class FileLogger:
    def __init__(self, file_path: Path, also_console: bool = True):
        self.file_path = file_path
        self.also_console = also_console
        self._fh = open(file_path, "w", encoding="utf-8")

    def log(self, msg: str = ""):
        self._fh.write(msg + "\n")
        self._fh.flush()
        if self.also_console:
            print(msg, flush=True)

    def close(self):
        self._fh.flush()
        self._fh.close()

_thread_local = threading.local()

def log(msg: Any = ""):
    if not isinstance(msg, str):
        msg = json.dumps(msg, ensure_ascii=False, default=str)
    logger: Optional[FileLogger] = getattr(_thread_local, "logger", None)
    if logger is not None:
        logger.log(msg)
    else:
        print(msg, flush=True)

def _tl_qid() -> str:
    """Return a question-id prefix string for terminal logs, e.g. '[q42]'."""
    qid = getattr(_thread_local, "question_id", None)
    return f"[q{qid}]" if qid is not None else "[q?]"

def _tl_hop() -> str:
    """Return current hop label, e.g. 'hop=2', or '' if not inside expansion."""
    hop = getattr(_thread_local, "cypher_hop", None)
    return f" hop={hop}/{ENTITY_SUBGRAPH_HOPS}" if hop is not None else ""

def _step_start(step_num: int, label: str) -> float:
    """Print a step-start banner and return the start timestamp."""
    t0 = now_ms()
    print(f"{_tl_qid()} [STEP {step_num} START] {label}", flush=True)
    return t0

def _step_finish(step_num: int, label: str, t0: float) -> None:
    """Print a step-finish banner with duration."""
    elapsed = dur_s(t0)
    print(f"{_tl_qid()} [STEP {step_num} FINISH] {label} | duration={elapsed:.3f}s", flush=True)

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

def dur_s(start: float) -> float:
    return time.time() - start

def _norm_id(x) -> str:
    return str(x).strip() if x is not None else ""

def count_tokens(text: str) -> int:
    """Heuristic fallback: ~4 characters per token. Prefer completion.usage fields when available."""
    return max(1, int(len(text) / 4))

def _as_float_list(vec) -> List[float]:
    if vec is None:
        return []
    if hasattr(vec, "tolist"):
        vec = vec.tolist()
    return [float(x) for x in list(vec)]

# Retry helpers
def _rand_wait_seconds() -> float:
    return random.uniform(5.0, 15.0)

def embed_text(text: str) -> List[float]:
    """Embed a single text using local BAAI/bge-m3 model (GPU if available)."""
    _EMBED_SEMAPHORE.acquire()
    try:
        vec = _bge_model.encode(text, normalize_embeddings=True, batch_size=1, show_progress_bar=False)
        return _as_float_list(vec)
    finally:
        _EMBED_SEMAPHORE.release()

def embed_texts(texts: List[str]) -> List[List[float]]:
    """Batch-encode a list of texts on GPU. Much faster than calling embed_text() in a loop."""
    if not texts:
        return []
    _EMBED_SEMAPHORE.acquire()
    try:
        vecs = _bge_model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=_EMBED_BATCH_SIZE,
            show_progress_bar=False,
        )
        return [_as_float_list(v) for v in vecs]
    finally:
        _EMBED_SEMAPHORE.release()

def cos_sim(a, b) -> float:
    """Cosine similarity using numpy (much faster than pure Python)."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

# ----------------- Safe LLM helpers -----------------
def safe_generate_json(prompt: str, schema: Dict[str, Any], temp: float = 0.0) -> Tuple[Dict[str, Any], float, float, float, str]:
    """Returns (result_dict, prompt_tokens, response_tokens, duration_s, raw_text).
    Uses LM Studio (OpenAI-compatible). Schema is ignored at API level; we parse JSON from response.
    prompt_tokens / response_tokens come from completion.usage when available.
    """
    t0 = now_ms()
    completion = lmstudio_client.chat.completions.create(
        model=LOCAL_GEN_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=temp,
    )
    duration_s = dur_s(t0)
    # Use server-reported token counts (accurate, includes chat-template overhead)
    usage = getattr(completion, "usage", None)
    prompt_tokens = (usage.prompt_tokens if usage and usage.prompt_tokens is not None else count_tokens(prompt))
    raw_text = ""
    result = {}
    try:
        raw_text = completion.choices[0].message.content or ""
        # print(raw_text)
        clean = raw_text.strip()
        # Collect <think>...</think> block contents before stripping (fallback if JSON is inside them)
        think_contents = re.findall(r"<think>([\s\S]*?)</think>", clean, flags=re.DOTALL)
        # Strip <think>...</think> blocks produced by reasoning/thinking models (e.g. Qwen3)
        clean = re.sub(r"<think>.*?</think>", "", clean, flags=re.DOTALL).strip()
        # Strategy 1: response starts directly with a code fence
        if clean.startswith("```json"):
            clean = clean[7:]
            if clean.endswith("```"):
                clean = clean[:-3]
            clean = clean.strip()
        elif clean.startswith("```"):
            clean = clean[3:]
            if clean.endswith("```"):
                clean = clean[:-3]
            clean = clean.strip()
        else:
            # Strategy 2: code fence appears anywhere in the response (e.g. model adds preamble text)
            m = re.search(r"```json\s*([\s\S]*?)```", clean)
            if m:
                clean = m.group(1).strip()
            else:
                m = re.search(r"```\s*([\s\S]*?)```", clean)
                if m:
                    clean = m.group(1).strip()
                else:
                    # Strategy 3: bare JSON object anywhere in the response
                    m = re.search(r"\{[\s\S]*\}", clean)
                    if m:
                        clean = m.group(0).strip()
        result = json.loads(clean)
    except Exception:
        raise Exception(f"Failed to parse JSON from response: {raw_text}")
    response_tokens = (usage.completion_tokens if usage and usage.completion_tokens is not None else count_tokens(raw_text))
    return result, prompt_tokens, response_tokens, duration_s, raw_text

def safe_generate_text(prompt: str, max_tokens: int, temperature: float = 0.2) -> Tuple[str, float, float, float]:
    """Returns (text, prompt_tokens, response_tokens, duration_s).
    Uses LM Studio (OpenAI-compatible).
    prompt_tokens / response_tokens come from completion.usage when available.
    """
    t0 = now_ms()
    completion = lmstudio_client.chat.completions.create(
        model=LOCAL_GEN_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    duration_s = dur_s(t0)
    # Use server-reported token counts (accurate, includes chat-template overhead)
    usage = getattr(completion, "usage", None)
    prompt_tokens = (usage.prompt_tokens if usage and usage.prompt_tokens is not None else count_tokens(prompt))
    text = (completion.choices[0].message.content or "").strip()
    if not text:
        raise RuntimeError("LLM returned an empty response.")
    response_tokens = (usage.completion_tokens if usage and usage.completion_tokens is not None else count_tokens(text))
    return text, prompt_tokens, response_tokens, duration_s

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

def agent1b_extract_query_triples(query: str) -> Tuple[List[Dict[str, Any]], float, float, float, str]:
    """
    Returns (triples, prompt_tokens, response_tokens, duration_s, raw_response_text).
    """
    prompt = f"""
You are Agent 1b. Task: extract explicit or implied triples from the user's question in the form:
subject — predicate — object.

Rules:
- Use short, literal subject/object texts as they appear in the question.
- Predicates should be concise (lowercase, snake_case if multiword).
- If type is unknown, leave it blank.
- Do not invent or speculate; extract only what is clearly suggested by the question.
-  If partial triples are available (at least two out of subject, predicate, and object), e.g. subject-predicate, predicate-object or subject-object, extract what is available and leave the rest as empty strings.

Return JSON with a key "triples" as specified.

User question:
\"\"\"{query}\"\"\"
"""
    data, prompt_tokens, response_tokens, duration_s, raw_response_text = safe_generate_json(prompt, QUERY_TRIPLES_SCHEMA, temp=0.0)
    triples = data.get("triples", []) if isinstance(data, dict) else []
    clean: List[Dict[str, Any]] = []
    for t in triples:
        s = t.get("subject", {}) or {}
        o = t.get("object", {}) or {}
        p = (t.get("predicate") or "").strip()
        if isinstance(s, str):
            s = {"text": s, "type": ""}
        if isinstance(o, str):
            o = {"text": o, "type": ""}
        s_text = s.get("text", "").strip()
        o_text = o.get("text", "").strip()
        if s_text or p or o_text:  # accept any partial triple with at least one non-empty component
            clean.append({
                "subject": {"text": s_text, "type": (s.get("type") or "").strip()},
                "predicate": p,
                "object":  {"text": o_text, "type": (o.get("type") or "").strip()},
            })
    return clean, prompt_tokens, response_tokens, duration_s, raw_response_text

def query_triple_to_text(t: Dict[str, Any]) -> str:
    s = ((t.get("subject") or {}).get("text") or "").strip()
    p = (t.get("predicate") or "").strip()
    o = ((t.get("object") or "").get("text") or "").strip() if isinstance(t.get("object"), dict) else ((t.get("object") or "").strip())
    return f"{s} [{p}] {o}"

def entities_from_triples(triples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract unique entity dicts (text, type) from subjects and objects of triples."""
    seen: Set[str] = set()
    ents: List[Dict[str, Any]] = []
    for t in triples:
        for role in ("subject", "object"):
            node = t.get(role) or {}
            text = (node.get("text") or "").strip()
            if text and text not in seen:
                seen.add(text)
                ents.append({"text": text, "type": (node.get("type") or "").strip()})
    return ents

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
    return ", ".join(parts)

def run_cypher_with_retry(cypher: str, params: Dict[str, Any], query_label: str = "") -> Tuple[List[Any], float]:
    """
    Execute a Cypher query with bounded retries.
    Returns (records, duration_s).
    query_label is a short human-readable description of what this query does.
    """
    attempts = 0
    last_e: Optional[Exception] = None
    qid = _next_query_id()
    hop_info = _tl_hop()
    prefix = _tl_qid()
    print(f"{prefix} [CYPHER START] qid={qid}{hop_info} | {query_label}", flush=True)
    t_cypher_start = now_ms()
    while attempts < max(1, NEO4J_MAX_ATTEMPTS):
        attempts += 1
        t0 = now_ms()
        try:
            acquired = _NEO4J_QUERY_SEMAPHORE.acquire(timeout=60)
            if not acquired:
                raise RuntimeError(
                    f"Could not acquire Neo4j query semaphore within 60s (qid={qid}). "
                    "Consider increasing NEO4J_QUERY_CONCURRENCY or reducing LLM_CONCURRENCY."
                )
            try:
                with driver.session() as session:
                    res = session.run(cypher, **params, timeout=NEO4J_TX_TIMEOUT_S)
                    records = list(res)
            finally:
                _NEO4J_QUERY_SEMAPHORE.release()
            cypher_dur = dur_s(t_cypher_start)
            print(f"{prefix} [CYPHER FINISH] qid={qid}{hop_info} | {query_label} | duration={cypher_dur:.3f}s", flush=True)
            return records, dur_s(t0)
        except Exception as e:
            last_e = e
            if attempts >= NEO4J_MAX_ATTEMPTS:
                break
            wait_s = random.uniform(5, 10)
            time.sleep(wait_s)
    raise RuntimeError(f"Neo4j query failed after {NEO4J_MAX_ATTEMPTS} attempts (qid={qid}): {last_e}")

def _vector_query_nodes(index_name: str, q_emb: List[float], k: int) -> Tuple[List[Dict[str, Any]], float]:
    q_emb = _as_float_list(q_emb)
    cypher = """
    CALL db.index.vector.queryNodes($index_name, $k, $q_emb) YIELD node AS n, score
    RETURN n, score, elementId(n) AS elem_id
    ORDER BY score DESC
    LIMIT $k
    """
    res, dur = run_cypher_with_retry(
        cypher,
        {"index_name": index_name, "k": k, "q_emb": q_emb},
        query_label=f"vector search on {index_name}"
    )
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
    return rows, dur

def search_similar_entities_by_embedding(q_emb: List[float], k: int) -> Tuple[List[Dict[str, Any]], float]:
    """Returns (matched_entities, total_duration_s).
    The three index searches are submitted in parallel so they run concurrently.
    """
    # Capture the question_id from the calling thread so inner workers can log correctly.
    _qid = getattr(_thread_local, "question_id", None)
    index_names = ("document_vec", "content_vec", "expression_vec")
    def _search_index(name: str) -> Tuple[List[Dict[str, Any]], float]:
        _thread_local.question_id = _qid  # propagate into worker thread
        return _vector_query_nodes(name, q_emb, k)
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {name: pool.submit(_search_index, name) for name in index_names}
        results = {name: futures[name].result() for name in index_names}

    candidates: List[Dict[str, Any]] = []
    total_dur = 0.0
    for name in index_names:
        rows, dur = results[name]
        candidates.extend(rows)
        total_dur += dur

    best: Dict[str, Dict[str, Any]] = {}
    for row in candidates:
        dedup_key = row.get("elem_id") or f"{row.get('key')}|{row.get('type')}"
        if dedup_key not in best or (row.get("score", -1) > best[dedup_key].get("score", -1)):
            best[dedup_key] = row

    merged = list(best.values())
    merged.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    return merged[:k], total_dur

def search_similar_triples_by_embedding(q_emb: List[float], k: int) -> Tuple[List[Dict[str, Any]], float]:
    q_emb = _as_float_list(q_emb)
    # s_name/s_key/s_type and o_name/o_key/o_type are cached directly on Triple nodes
    # by the post-indexing optimization script, so we no longer need to traverse
    # [:SUBJECT] and [:OBJECT] edges — saving 2 hops per result node.
    cypher = """
    CALL db.index.vector.queryNodes('triple_vec', $k, $q_emb) YIELD node AS tr, score
    RETURN tr, score
    ORDER BY score DESC
    LIMIT $k
    """
    res, dur = run_cypher_with_retry(
        cypher, {"k": k, "q_emb": q_emb},
        query_label="vector search triple_vec"
    )
    rows = []
    for r in res:
        tr = r["tr"]
        rows.append({
            "triple_uid":           tr.get("triple_uid"),
            "predicate":            tr.get("predicate"),
            "uu_number":            tr.get("uu_number"),
            "evidence_quote":       tr.get("evidence_quote"),
            "subject":              tr.get("s_name"),
            "subject_key":          tr.get("s_key"),
            "subject_type":         tr.get("s_type"),
            "object":               tr.get("o_name"),
            "object_key":           tr.get("o_key"),
            "object_type":          tr.get("o_type"),
            "score":                r["score"],
            "embedding":            tr.get("embedding"),
            "document_id":          tr.get("document_id"),
            "chunk_id":             tr.get("chunk_id"),
            "evidence_article_ref": tr.get("evidence_article_ref"),
        })
    return rows, dur

# ----------------- Graph expansion -----------------
def expand_from_entities(
    entity_keys: List[str],
    hops: int,
    per_hop_limit: int,
    entity_elem_ids: Optional[List[str]] = None  # kept for API compatibility; no longer used
) -> Tuple[List[Dict[str, Any]], float]:
    """Returns (triples, total_duration_s).

    s_name/s_key/s_type and o_name/o_key/o_type are now cached directly on
    Triple nodes (populated by the post-indexing optimization script), so we
    skip the [:SUBJECT] and [:OBJECT] traversals entirely — saving 2 hops per
    Triple per expansion hop.  The frontier is driven entirely by the cached
    s_key/o_key values returned with the Triple, so element-id tracking is
    also no longer needed.
    """
    triples: Dict[str, Dict[str, Any]] = {}
    current_keys: Set[str] = set(x for x in (entity_keys or []) if x)
    total_dur = 0.0

    for hop_idx in range(hops):
        if not current_keys:
            break

        # Expose current hop number to run_cypher_with_retry via thread-local
        _thread_local.cypher_hop = hop_idx + 1

        # Read s_key/o_key from cached Triple properties — no OPTIONAL MATCH.
        cypher = """
        UNWIND $keys AS k
        MATCH (e:Entity {key: k})-[r:PREDICATE]->()
        WITH DISTINCT r.triple_uid AS uid LIMIT $limit
        MATCH (tr:Triple {triple_uid: uid})
        RETURN tr, tr.s_key AS s_key, tr.o_key AS o_key
        """
        params = {"keys": list(current_keys), "limit": per_hop_limit}

        res, dur = run_cypher_with_retry(cypher, params, query_label="subgraph expansion hop")
        _thread_local.cypher_hop = None  # reset after each hop query
        total_dur += dur

        next_keys: Set[str] = set()
        for r in res:
            tr = r["tr"]
            uid = tr.get("triple_uid")
            if uid not in triples:
                triples[uid] = {
                    "triple_uid":           uid,
                    "predicate":            tr.get("predicate"),
                    "uu_number":            tr.get("uu_number"),
                    "evidence_quote":       tr.get("evidence_quote"),
                    "embedding":            tr.get("embedding"),
                    "document_id":          tr.get("document_id"),
                    "chunk_id":             tr.get("chunk_id"),
                    "evidence_article_ref": tr.get("evidence_article_ref"),
                    "subject":              tr.get("s_name"),
                    "subject_key":          tr.get("s_key"),
                    "subject_type":         tr.get("s_type"),
                    "object":               tr.get("o_name"),
                    "object_key":           tr.get("o_key"),
                    "object_type":          tr.get("o_type"),
                }
            # Advance frontier using cached keys (no elementId lookup needed)
            s_key = r.get("s_key"); o_key = r.get("o_key")
            if s_key: next_keys.add(s_key)
            if o_key: next_keys.add(o_key)

        current_keys = next_keys

    return list(triples.values()), total_dur

# ----------------- Chunk fetching from Neo4j (avoids GPU calls) -----------------
def fetch_chunks_from_neo4j(
    pairs: List[Tuple[Any, Any]]
) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """
    Batch-fetch TextChunk nodes from Neo4j by (document_id, chunk_id).
    Returns a dict mapping (norm_doc_id, norm_chunk_id) -> {content, embedding, uu_number, pages}.
    Embeddings are pre-stored from naive_pipeline indexing — no GPU call needed.
    """
    if not pairs:
        return {}

    # Deduplicate and normalize
    seen: Set[Tuple[str, str]] = set()
    unique_pairs: List[Tuple[str, str]] = []
    for doc_id, chunk_id in pairs:
        k = (_norm_id(doc_id), _norm_id(chunk_id))
        if k not in seen:
            seen.add(k)
            unique_pairs.append(k)

    cypher = """
    UNWIND $pairs AS p
    OPTIONAL MATCH (c:TextChunk {chunk_id: p.chunk_id, document_id: p.doc_id})
    RETURN p.chunk_id AS chunk_id, p.doc_id AS doc_id,
           c.content AS content, c.embedding AS embedding,
           c.uu_number AS uu_number, c.pages AS pages
    """

    params = {
        "pairs": [{"doc_id": doc_id, "chunk_id": chunk_id} for doc_id, chunk_id in unique_pairs]
    }

    res, _ = run_cypher_with_retry(
        cypher, params, query_label=f"fetch {len(unique_pairs)} TextChunks from Neo4j"
    )

    out: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for r in res:
        k = (_norm_id(r["doc_id"]), _norm_id(r["chunk_id"]))
        out[k] = {
            "content": r["content"],
            "embedding": r["embedding"],
            "uu_number": r["uu_number"],
            "pages": r["pages"],
        }
    return out

# ----------------- Scoring helpers -----------------
def mean_similarity_to_query_triples(triple_emb: Optional[List[float]], q_trip_embs: List[List[float]]) -> float:
    """Vectorized cosine similarity of one triple embedding vs a list of query-triple embeddings."""
    if triple_emb is None:
        return 0.0
    if not isinstance(triple_emb, list):
        triple_emb = _as_float_list(triple_emb)
    if not triple_emb or not q_trip_embs:
        return 0.0
    a = np.asarray(triple_emb, dtype=np.float32)
    na = np.linalg.norm(a)
    if na == 0.0:
        return 0.0
    a = a / na
    B = np.asarray(q_trip_embs, dtype=np.float32)  # shape (N, D)
    norms = np.linalg.norm(B, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    B = B / norms
    return float(np.mean(B @ a))

# ----------------- Retrieval pipeline -----------------
def entity_centric_retrieval(
    query_entities: List[Dict[str, Any]],
    q_trip_embs: List[List[float]],
    q_emb_fallback: Optional[List[float]] = None,
    cypher_durations: Optional[Dict[str, float]] = None,
    matched_entities_log: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """
    Retrieves triples via entity->KG match->subgraph expansion.
    Entity texts are batch-embedded in one GPU call; KG searches run in parallel.
    cypher_durations: accumulator dict for Cypher query durations.
    matched_entities_log: list to append matched entity info for logging.
    """
    all_matched_keys: Set[str] = set()
    all_matched_ids: Set[str] = set()

    # Filter to non-empty entities first
    active_entities = [e for e in query_entities if (e.get("text") or "").strip()]
    if not active_entities:
        return []

    # Batch-embed all entity texts in a single GPU call
    entity_texts = [(e.get("text") or "").strip() for e in active_entities]
    entity_embs = embed_texts(entity_texts)

    # Fan out: one KG search per entity, submitted concurrently
    # Capture question_id so inner worker threads log [qN] correctly.
    _qid = getattr(_thread_local, "question_id", None)
    def _search_entity(e_emb: List[float]) -> Tuple[List[Dict[str, Any]], float]:
        _thread_local.question_id = _qid
        return search_similar_entities_by_embedding(e_emb, k=ENTITY_MATCH_TOP_K)

    with ThreadPoolExecutor(max_workers=max(1, len(entity_embs))) as pool:
        ent_futures = [pool.submit(_search_entity, emb) for emb in entity_embs]
        ent_results = [f.result() for f in ent_futures]

    for entity, (matches, sim_dur) in zip(active_entities, ent_results):
        text = (entity.get("text") or "").strip()
        if cypher_durations is not None:
            cypher_durations["entity_vector_search"] = cypher_durations.get("entity_vector_search", 0.0) + sim_dur
        keys = [m.get("key") for m in matches if m.get("key")]
        ids  = [m.get("elem_id") for m in matches if m.get("elem_id")]
        all_matched_keys.update(keys)
        all_matched_ids.update(ids)
        # Log matched entities
        if matched_entities_log is not None:
            for m in matches:
                matched_entities_log.append({
                    "query_entity": text,
                    "kg_entity": m.get("name") or m.get("key"),
                    "type": m.get("type"),
                    "score": m.get("score"),
                    "elem_id": m.get("elem_id"),
                    "chunk_id": m.get("chunk_id"),
                })

    if not (all_matched_keys or all_matched_ids):
        return []

    expanded_triples, expand_dur = expand_from_entities(
        list(all_matched_keys),
        hops=ENTITY_SUBGRAPH_HOPS,
        per_hop_limit=ENTITY_SUBGRAPH_PER_HOP_LIMIT,
        entity_elem_ids=list(all_matched_ids) if all_matched_ids else None
    )
    if cypher_durations is not None:
        cypher_durations["subgraph_expansion"] = cypher_durations.get("subgraph_expansion", 0.0) + expand_dur

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

def triple_centric_retrieval(
    query_triples: List[Dict[str, Any]],
    cypher_durations: Optional[Dict[str, float]] = None,
) -> Tuple[List[Dict[str, Any]], List[List[float]]]:
    if not query_triples:
        return [], []

    # Batch-embed all query-triple texts in a single GPU call
    texts = [query_triple_to_text(qt) for qt in query_triples]
    q_trip_embs: List[List[float]] = embed_texts(texts)

    # Submit all Cypher searches concurrently.
    # Capture question_id so inner worker threads log [qN] correctly.
    _qid = getattr(_thread_local, "question_id", None)
    def _search(emb: List[float]) -> Tuple[List[Dict[str, Any]], float]:
        _thread_local.question_id = _qid
        return search_similar_triples_by_embedding(emb, k=QUERY_TRIPLE_MATCH_TOP_K_PER)

    with ThreadPoolExecutor(max_workers=len(q_trip_embs)) as pool:
        futures = [pool.submit(_search, emb) for emb in q_trip_embs]
        search_results = [f.result() for f in futures]

    triples_map: Dict[str, Dict[str, Any]] = {}
    for matches, dur in search_results:
        if cypher_durations is not None:
            cypher_durations["triple_vector_search"] = cypher_durations.get("triple_vector_search", 0.0) + dur
        for m in matches:
            uid = m.get("triple_uid")
            if uid:
                if uid not in triples_map:
                    triples_map[uid] = m
                else:
                    if m.get("score", 0.0) > triples_map[uid].get("score", 0.0):
                        triples_map[uid] = m

    return list(triples_map.values()), q_trip_embs

def collect_chunks_for_triples(
    triples: List[Dict[str, Any]]
) -> List[Tuple[Tuple[Any, Any], str, Dict[str, Any]]]:
    """Fetch chunk text and pre-stored embeddings from Neo4j TextChunk nodes.

    Embeddings were stored by naive_pipeline_local.py during indexing — no GPU call needed.
    Returns list of (norm_key, content, triple_dict) with the triple dict enriched with
    the stored embedding under key "_chunk_embedding".
    """
    seen_pairs: Set[Tuple[Any, Any]] = set()
    pairs_to_fetch: List[Tuple[Any, Any]] = []
    ordered_triples: List[Dict[str, Any]] = []

    for t in triples:
        doc_id = t.get("document_id")
        chunk_id = t.get("chunk_id")

        if doc_id is None or chunk_id is None:
            raise RuntimeError(
                f"Triple {t.get('triple_uid')!r} has missing document_id or chunk_id."
            )

        norm_key = (_norm_id(doc_id), _norm_id(chunk_id))
        if norm_key not in seen_pairs:
            seen_pairs.add(norm_key)
            pairs_to_fetch.append((doc_id, chunk_id))
            ordered_triples.append(t)

    if not pairs_to_fetch:
        return []

    # Batch-fetch all chunks from Neo4j in one query
    chunk_data = fetch_chunks_from_neo4j(pairs_to_fetch)

    out: List[Tuple[Tuple[Any, Any], str, Dict[str, Any]]] = []
    for doc_id, chunk_id, t in zip(
        [p[0] for p in pairs_to_fetch],
        [p[1] for p in pairs_to_fetch],
        ordered_triples
    ):
        norm_key = (_norm_id(doc_id), _norm_id(chunk_id))
        data = chunk_data.get(norm_key)
        if data is None:
            raise RuntimeError(
                f"Chunk not found in Neo4j for triple {t.get('triple_uid')!r}: "
                f"document_id={doc_id!r}, chunk_id={chunk_id!r}."
            )
        content = data["content"]
        if content is None:
            raise RuntimeError(
                f"Chunk content is null in Neo4j for triple {t.get('triple_uid')!r}: "
                f"document_id={doc_id!r}, chunk_id={chunk_id!r}."
            )
        t["_is_quote_fallback"] = False
        t["_chunk_embedding"] = data["embedding"]  # pre-stored, no GPU needed
        out.append((norm_key, content, t))

    return out

def rerank_chunks_by_query(
    chunk_records: List[Tuple[Tuple[Any, Any], str, Dict[str, Any]]],
    q_emb_query: List[float],
    top_k: int
) -> List[Tuple[Tuple[Any, Any], str, Dict[str, Any], float]]:
    """Rerank chunks by cosine similarity to query embedding.

    Chunk embeddings are pre-stored on TextChunk nodes in Neo4j (from naive_pipeline indexing).
    We read them from the triple dict (key "_chunk_embedding") — no GPU call needed.
    """
    cand = chunk_records[:CHUNK_RERANK_CAND_LIMIT]
    if not cand:
        return []

    # Use pre-stored embeddings from Neo4j — no embed_texts() call needed
    q = np.asarray(q_emb_query, dtype=np.float32)
    qn = np.linalg.norm(q)
    if qn > 0.0:
        q = q / qn
    scored: List[Tuple[Tuple[Any, Any], str, Dict[str, Any], float]] = []
    for (key, text, t) in cand:
        emb = t.get("_chunk_embedding")
        if not emb:
            continue
        e = np.asarray(emb, dtype=np.float32)
        en = np.linalg.norm(e)
        if en > 0.0:
            e = e / en
        s = float(np.dot(q, e))
        scored.append((key, text, t, s))
    scored.sort(key=lambda x: x[3], reverse=True)
    return scored[:top_k]

def rerank_triples_by_query_triples(
    triples: List[Dict[str, Any]],
    q_trip_embs: List[List[float]],
    q_emb_fallback: Optional[List[float]],
    top_k: int
) -> Tuple[List[Dict[str, Any]], List[float]]:
    """Returns (ranked_triples, scores_in_order)."""
    def score(t: Dict[str, Any]) -> float:
        emb = t.get("embedding")
        if q_trip_embs:
            return mean_similarity_to_query_triples(emb, q_trip_embs)
        if q_emb_fallback and isinstance(emb, list):
            return cos_sim(q_emb_fallback, emb)
        return 0.0

    ranked = sorted(triples, key=score, reverse=True)
    top = ranked[:top_k]
    scores = [score(t) for t in top]
    return top, scores

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
    for idx, (key, text, t, sc) in enumerate(chunks_ranked, start=1):
        doc_id = t.get("document_id")
        chunk_id = t.get("chunk_id")
        uu = t.get("uu_number") or ""
        fb = " | quote-fallback" if t.get("_is_quote_fallback") else ""
        lines.append(f"[Chunk {idx}] doc={doc_id} chunk={chunk_id} | {uu} | score={sc:.3f}{fb}\n{text}")
    context = "\n".join(lines)

    chunk_records = [{"key": key, "text": text, "triple": t, "score": sc} for key, text, t, sc in chunks_ranked]
    return context, summary_text, chunk_records

# ----------------- Agent 2 (Answer) -----------------
def agent2_answer(query_original: str, context: str) -> Tuple[str, float, float, float]:
    """
    Returns (answer_text, prompt_tokens, response_tokens, duration_s).
    Note: answer_text itself is the full response content.
    """
    prompt = f"""
You are Agent 2 (Answerer). Task: provide an answer based on the context only.

Core instructions:
Answer concisely and accurately based strictly on the provided context.
Cite UU/Article references when they are clear.
Respond in the same language as the user's question.

Original user question:
\"\"\"{query_original}\"\"\"

Context:
\"\"\"{context}\"\"\"
"""
    answer, prompt_tokens, response_tokens, duration_s = safe_generate_text(
        prompt, max_tokens=ANSWER_MAX_TOKENS, temperature=0.2
    )
    return answer, prompt_tokens, response_tokens, duration_s

# ----------------- Main pipeline -----------------
def agentic_graph_rag(
    query_original: str,
    question_id: Any = None,
    log_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    ts_name = make_timestamp_name()
    qid_str = f"q{question_id}_" if question_id is not None else ""
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"{qid_str}{ts_name}.txt"
    else:
        log_file = Path.cwd() / f"{qid_str}{ts_name}.txt"
    _thread_local.logger = FileLogger(log_file, also_console=True)
    _thread_local.question_id = question_id
    _thread_local.cypher_hop = None

    t_all_start = now_ms()
    try:
        # Accumulators
        cypher_durations: Dict[str, float] = {}
        llm_durations: Dict[str, float] = {}

        # ------------------------------------------------------------------
        # Step 1: Agent 1b – extract triples from query
        # ------------------------------------------------------------------
        _t1 = _step_start(1, "Agent 1b – extract triples from query")
        query_triples, a1b_prompt_tok, a1b_resp_tok, a1b_dur, a1b_raw_response = agent1b_extract_query_triples(query_original)
        llm_durations["agent1b"] = a1b_dur
        _step_finish(1, "Agent 1b – extract triples from query", _t1)

        # Entities = subjects + objects from Agent 1b triples
        query_entities = entities_from_triples(query_triples)

        # ------------------------------------------------------------------
        # Step 2: Embed whole query (for chunk reranking / fallback scoring)
        # ------------------------------------------------------------------
        _t2 = _step_start(2, "Embed whole query")
        q_emb_query = embed_text(query_original)
        _step_finish(2, "Embed whole query", _t2)

        # ------------------------------------------------------------------
        # Step 3: Triple-centric retrieval
        # ------------------------------------------------------------------
        _t3 = _step_start(3, "Triple-centric retrieval")
        matched_entities_log: List[Dict[str, Any]] = []
        ctx2_triples, q_trip_embs = triple_centric_retrieval(query_triples, cypher_durations=cypher_durations)
        _step_finish(3, "Triple-centric retrieval", _t3)

        # ------------------------------------------------------------------
        # Step 4: Entity-centric retrieval
        # ------------------------------------------------------------------
        _t4 = _step_start(4, "Entity-centric retrieval")
        ctx1_triples = entity_centric_retrieval(
            query_entities,
            q_trip_embs=q_trip_embs,
            q_emb_fallback=q_emb_query,
            cypher_durations=cypher_durations,
            matched_entities_log=matched_entities_log,
        )
        _step_finish(4, "Entity-centric retrieval", _t4)

        # Collect subgraph triples for logging (top k from entity-centric before merge)
        subgraph_triples_log = ctx1_triples  # already top-SUBGRAPH_TRIPLES_TOP_K

        # ------------------------------------------------------------------
        # Step 5: Merge and dedupe triples
        # ------------------------------------------------------------------
        _t5 = _step_start(5, "Merge and dedupe triples")
        triple_map: Dict[str, Dict[str, Any]] = {}
        for t in ctx1_triples + ctx2_triples:
            uid = t.get("triple_uid")
            if uid:
                prev = triple_map.get(uid)
                if prev is None or (t.get("score", 0.0) > prev.get("score", 0.0)):
                    triple_map[uid] = t
        merged_triples = list(triple_map.values())
        _step_finish(5, "Merge and dedupe triples", _t5)

        # ------------------------------------------------------------------
        # Step 6: Rerank triples
        # ------------------------------------------------------------------
        _t6 = _step_start(6, "Rerank triples")
        triples_ranked, triple_scores = rerank_triples_by_query_triples(
            merged_triples, q_trip_embs=q_trip_embs, q_emb_fallback=q_emb_query, top_k=MAX_TRIPLES_FINAL
        )
        _step_finish(6, "Rerank triples", _t6)

        # ------------------------------------------------------------------
        # Step 7: Collect chunks, rerank
        # ------------------------------------------------------------------
        _t7 = _step_start(7, "Collect and rerank chunks")
        chunk_records = collect_chunks_for_triples(merged_triples)
        chunks_ranked = rerank_chunks_by_query(chunk_records, q_emb_query, top_k=MAX_CHUNKS_FINAL)
        _step_finish(7, "Collect and rerank chunks", _t7)

        # ------------------------------------------------------------------
        # Step 8: Build context and generate answer (Agent 2)
        # ------------------------------------------------------------------
        _t8 = _step_start(8, "Build context and generate answer (Agent 2)")
        context_text, _, _ = build_combined_context_text(triples_ranked, chunks_ranked)

        answer, a2_prompt_tok, a2_resp_tok, a2_dur = agent2_answer(query_original, context_text)
        llm_durations["agent2"] = a2_dur
        _step_finish(8, "Build context and generate answer (Agent 2)", _t8)

        total_dur_s = dur_s(t_all_start)
        total_cypher_s = sum(cypher_durations.values())
        total_llm_s = sum(llm_durations.values())

        # ------------------------------------------------------------------
        # Structured logging
        # ------------------------------------------------------------------
        sep = "=" * 70

        log(sep)
        log(f"QUESTION: {query_original}")
        log(sep)

        log("\n--- [1] Prompt & Response Token Counts ---")
        log(f"  Agent 1b | Prompt tokens : {a1b_prompt_tok}")
        log(f"  Agent 1b | Response tokens: {a1b_resp_tok}")
        log(f"  Agent 2  | Prompt tokens : {a2_prompt_tok}")
        log(f"  Agent 2  | Response tokens: {a2_resp_tok}")

        log("\n--- [1b] Response Token Contents ---")
        log("  [Agent 1b response]")
        log(a1b_raw_response)
        log("  [Agent 2 response]")
        log(answer)

        log("\n--- [2] LLM Call Durations ---")
        log(f"  Agent 1b | Duration: {a1b_dur:.3f}s")
        log(f"  Agent 2  | Duration: {a2_dur:.3f}s")

        log("\n--- [3] Cypher Query Durations ---")
        for label, dur in cypher_durations.items():
            log(f"  [{label}] : {dur:.3f}s")

        log("\n--- [4] Cumulative Durations ---")
        log(f"  Cumulative Cypher : {total_cypher_s:.3f}s")
        log(f"  Cumulative LLM    : {total_llm_s:.3f}s")
        log(f"  Total (Q→Answer)  : {total_dur_s:.3f}s")

        log("\n--- [5] Extracted Triples (Agent 1b) ---")
        for i, qt in enumerate(query_triples, 1):
            s = qt.get("subject", {}).get("text", "")
            p = qt.get("predicate", "")
            o = qt.get("object", {}).get("text", "")
            log(f"  {i}. {s} [{p}] {o}")

        log("\n--- [6] Top Similar KG Entities (per query entity, with chunk_id) ---")
        for me in matched_entities_log:
            log(f"  query_entity={me['query_entity']} | kg_entity={me['kg_entity']} | type={me['type']} | score={me['score']} | chunk_id={me.get('chunk_id')}")

        log("\n--- [7] Top Subgraph Triples (entity-centric, after triple-vs-triple similarity) ---")
        for i, t in enumerate(subgraph_triples_log, 1):
            log(f"  {i}. {t.get('subject')} [{t.get('predicate')}] {t.get('object')} | chunk_id={t.get('chunk_id')}")

        log("\n--- [8] Final Reranked Triples (most→least similar) ---")
        for i, (t, sc) in enumerate(zip(triples_ranked, triple_scores), 1):
            log(f"  {i}. {t.get('subject')} [{t.get('predicate')}] {t.get('object')} | score={sc:.4f} | chunk_id={t.get('chunk_id')}")

        log("\n--- [9] Final Chunk IDs Used for Context ---")
        used_chunk_ids = []
        for _, _, t, _ in chunks_ranked:
            cid = t.get("chunk_id")
            if cid and cid not in used_chunk_ids:
                used_chunk_ids.append(cid)
        for cid in used_chunk_ids:
            log(f"  {cid}")

        log("\n--- [10] Final Answer ---")
        log(answer)

        log(f"\nLog saved to: {log_file}")
        log(sep)

        return {
            "final_answer": answer,
            "log_file": str(log_file),
        }
    finally:
        _local_logger: Optional[FileLogger] = getattr(_thread_local, "logger", None)
        if _local_logger is not None:
            _local_logger.close()
        _thread_local.logger = None
        _thread_local.question_id = None
        _thread_local.cypher_hop = None

# ----------------- Main -----------------
if __name__ == "__main__":
    # Paths
    REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
    QA_FILE = (
        REPO_ROOT
        / "dataset"
        / "4_experiment"
        / "4a_qa_generation"
        / "4a_ii_qa_pairs"
        / "qa_pairs_local_filtered_sampled_2000.jsonl"
    )
    OUTPUT_DIR = REPO_ROOT / "dataset" / "4_experiment" / "4b_retrieval" / "lexidkg_graphrag"
    LOG_DIR = OUTPUT_DIR / "logs"
    OUTPUT_FILE = OUTPUT_DIR / "lexidkg_graphrag_output.jsonl"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Load already-processed question IDs for resume support
    processed_ids: Set[Any] = set()
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f_in:
            for line in f_in:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if "question_id" in rec:
                    processed_ids.add(rec["question_id"])
        print(f"[Resume] Already processed {len(processed_ids)} question(s).")

    # Load QA pairs
    qa_pairs: List[Dict[str, Any]] = []
    with open(QA_FILE, "r", encoding="utf-8") as f_qa:
        for line in f_qa:
            line = line.strip()
            if not line:
                continue
            qa_pairs.append(json.loads(line))
    print(f"[Info] Total QA pairs loaded: {len(qa_pairs)}")

    # Filter out already-processed questions
    pending_pairs = [
        qa for qa in qa_pairs
        if qa.get("question_id") not in processed_ids
    ]
    skipped_count = len(qa_pairs) - len(pending_pairs)
    if skipped_count:
        print(f"[Resume] Skipping {skipped_count} already-processed question(s).")
    print(f"[Info] Questions to process: {len(pending_pairs)} (concurrency={LLM_CONCURRENCY})")

    _write_lock = threading.Lock()

    def process_one(qa: Dict[str, Any], idx: int) -> None:
        qid = qa.get("question_id")
        question = qa.get("question", "").strip()
        # Gate each question through the global question semaphore so that at most
        # LLM_CONCURRENCY questions are in-flight at any time — covering the full
        # lifecycle (LLM extraction, Neo4j retrieval, LLM answer generation).
        _QUESTION_SEMAPHORE.acquire()
        try:
            print(f"[Processing] ({idx}/{len(qa_pairs)}) question_id={qid}: {question[:80]}...")
            result = agentic_graph_rag(
                query_original=question,
                question_id=qid,
                log_dir=LOG_DIR,
            )
            output_record = {
                "question_id": qid,
                "question": question,
                "final_answer": result.get("final_answer", ""),
            }
            with _write_lock:
                with open(OUTPUT_FILE, "a", encoding="utf-8") as f_out:
                    f_out.write(json.dumps(output_record, ensure_ascii=False) + "\n")
                processed_ids.add(qid)
        finally:
            _QUESTION_SEMAPHORE.release()

    with ThreadPoolExecutor(max_workers=LLM_CONCURRENCY) as executor:
        global_idx = skipped_count  # already-processed count
        # Submit ALL questions at once. ThreadPoolExecutor enforces max_workers so
        # at most LLM_CONCURRENCY tasks run simultaneously; as soon as any
        # question finishes the next pending one starts immediately.
        all_futures: List[Future] = []
        for qa in pending_pairs:
            global_idx += 1
            fut = executor.submit(process_one, qa, global_idx)
            all_futures.append(fut)
        for fut in as_completed(all_futures):
            exc = fut.exception()
            if exc:
                driver.close()
                raise exc

    driver.close()
    print(f"[Done] Results saved to: {OUTPUT_FILE}")