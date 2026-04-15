# lexidkg_graphrag_local.py
# Single-pass GraphRAG (Agent 1b + Agent 2 only), with structured per-question logging.
# Local version: uses LM Studio (OpenAI-compatible API) for LLM calls and BAAI/bge-m3 for embeddings.

import os, time, json, math, pickle, re, random
import httpx
from concurrent.futures import ThreadPoolExecutor, as_completed, Future, as_completed
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
NEO4J_MAX_ATTEMPTS = int(os.getenv("NEO4J_MAX_ATTEMPTS", "100000"))

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
LLM_CONCURRENCY = int(os.getenv("LLM_CONCURRENCY", "2"))

# Dataset folder for original chunk pickles
DEFAULT_LANGCHAIN_DIR = (Path(__file__).resolve().parent / "../../../dataset/3_indexing/3a_langchain_results/").resolve()
LANGCHAIN_DIR = Path(os.getenv("LANGCHAIN_DIR") or str(DEFAULT_LANGCHAIN_DIR))
SKIP_FILES = {"all_langchain_documents.pkl"}

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
print(f"[Init] Loading embedding model: {LOCAL_EMBED_MODEL} ...")
_bge_model = SentenceTransformer(LOCAL_EMBED_MODEL, trust_remote_code=True)
print(f"[Init] Embedding model loaded.")

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

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
        self._lock = threading.Lock()

    def log(self, msg: str = ""):
        with self._lock:
            self._fh.write(msg + "\n")
            self._fh.flush()
        if self.also_console:
            print(msg, flush=True)

    def close(self):
        with self._lock:
            self._fh.flush()
            self._fh.close()

_thread_local = threading.local()
_log_file_logger: Optional[FileLogger] = None  # module-level, thread-safe for workers

def log(msg: Any = ""):
    """Write to the per-question log file. Also prints to console when in the main thread."""
    if not isinstance(msg, str):
        msg = json.dumps(msg, ensure_ascii=False, default=str)
    # Always write to the log file (thread-safe via FileLogger._lock)
    if _log_file_logger is not None:
        _log_file_logger.log(msg)
    # Only print to console when in the main thread (has logger on thread-local)
    if getattr(_thread_local, "logger", None) is not None:
        print(msg, flush=True)

def _tl_qid() -> str:
    """Return a question-id prefix string for terminal logs, e.g. '[q42]'."""
    qid = getattr(_thread_local, "question_id", None)
    return f"[q{qid}]" if qid is not None else "[q]"

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
    """Embed using local BAAI/bge-m3 model via SentenceTransformer."""
    vec = _bge_model.encode(text, normalize_embeddings=True)
    return _as_float_list(vec)

def cos_sim(a: List[float], b: List[float]) -> float:
    a = _as_float_list(a)
    b = _as_float_list(b)
    dot = sum(x*y for x,y in zip(a,b))
    na = math.sqrt(sum(x*x for x in a))
    nb = math.sqrt(sum(y*y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)

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

# ----------------- Parallel embedding helpers -----------------
def _parallel_embed_texts(
    texts: List[str],
    max_workers: int
) -> List[List[float]]:
    """Embed multiple texts in parallel using a thread pool.

    Returns list of embeddings in the same order as inputs.
    Note: embed_text() returns List[float] directly (no duration).
    """
    if not texts:
        return []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(embed_text, txt): i for i, txt in enumerate(texts)}
        results: List[Optional[List[float]]] = [None] * len(texts)
        for fut in as_completed(futures):
            results[futures[fut]] = fut.result()
    return results  # type: ignore[return-value]

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
            with driver.session() as session:
                res = session.run(cypher, **params, timeout=NEO4J_TX_TIMEOUT_S)
                records = list(res)
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
    """Returns (matched_entities, total_duration_s)."""
    idx_names = ("document_vec", "content_vec", "expression_vec")

    # Run all 3 index searches in parallel
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(_vector_query_nodes, idx_name, q_emb, k): idx_name for idx_name in idx_names}
        all_rows: List[Dict[str, Any]] = []
        total_dur = 0.0
        for fut in as_completed(futures):
            rows, dur = fut.result()
            all_rows.extend(rows)
            total_dur += dur

    best: Dict[str, Dict[str, Any]] = {}
    for row in all_rows:
        dedup_key = row.get("elem_id") or f"{row.get('key')}|{row.get('type')}"
        if dedup_key not in best or (row.get("score", -1) > best[dedup_key].get("score", -1)):
            best[dedup_key] = row

    merged = list(best.values())
    merged.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    return merged[:k], total_dur

def search_similar_triples_by_embedding(q_emb: List[float], k: int) -> Tuple[List[Dict[str, Any]], float]:
    q_emb = _as_float_list(q_emb)
    cypher = """
    CALL db.index.vector.queryNodes('triple_vec', $k, $q_emb) YIELD node AS tr, score
    MATCH (tr)-[:SUBJECT]->(s)
    MATCH (tr)-[:OBJECT]->(o)
    RETURN tr, s, o, score
    ORDER BY score DESC
    LIMIT $k
    """
    res, dur = run_cypher_with_retry(
        cypher, {"k": k, "q_emb": q_emb},
        query_label="vector search triple_vec"
    )
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
    return rows, dur

# ----------------- Graph expansion -----------------
def expand_from_entities(
    entity_keys: List[str],
    hops: int,
    per_hop_limit: int,
    entity_elem_ids: Optional[List[str]] = None
) -> Tuple[List[Dict[str, Any]], float]:
    """Returns (triples, total_duration_s)."""
    triples: Dict[str, Dict[str, Any]] = {}
    current_ids: Set[str] = set(x for x in (entity_elem_ids or []) if x)
    current_keys: Set[str] = set(x for x in (entity_keys or []) if x)
    total_dur = 0.0

    for hop_idx in range(hops):
        if not current_ids and not current_keys:
            break

        # Expose current hop number to run_cypher_with_retry via thread-local
        _thread_local.cypher_hop = hop_idx + 1

        if current_ids:
            cypher = """
            UNWIND $ids AS eid
            MATCH (e) WHERE elementId(e) = eid
            MATCH (e)-[r:PREDICATE]->()
            WITH DISTINCT r.triple_uid AS uid LIMIT $limit
            MATCH (tr:Triple {triple_uid: uid})
            MATCH (tr)-[:SUBJECT]->(s)
            MATCH (tr)-[:OBJECT]->(o)
            RETURN tr, s, o, elementId(s) AS s_id, elementId(o) AS o_id
            """
            params = {"ids": list(current_ids), "limit": per_hop_limit}
        else:
            cypher = """
            UNWIND $keys AS k
            MATCH (e:Entity {key:k})-[r:PREDICATE]->()
            WITH DISTINCT r.triple_uid AS uid LIMIT $limit
            MATCH (tr:Triple {triple_uid: uid})
            MATCH (tr)-[:SUBJECT]->(s)
            MATCH (tr)-[:OBJECT]->(o)
            RETURN tr, s, o, elementId(s) AS s_id, elementId(o) AS o_id
            """
            params = {"keys": list(current_keys), "limit": per_hop_limit}

        res, dur = run_cypher_with_retry(cypher, params, query_label="subgraph expansion hop")
        _thread_local.cypher_hop = None  # reset after each hop query
        total_dur += dur

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
            s_id = r.get("s_id"); o_id = r.get("o_id")
            if s_id: next_ids.add(s_id)
            if o_id: next_ids.add(o_id)

        current_ids = next_ids if next_ids else set()
        current_keys = set() if next_ids else next_keys

    return list(triples.values()), total_dur

# ----------------- Chunk store -----------------
class ChunkStore:
    def __init__(self, root: Path, skip: Set[str]):
        self.root = root
        self.skip = skip
        self._index: Dict[Tuple[str, str], str] = {}
        self._by_chunk: Dict[str, List[Tuple[str, str]]] = {}
        self._loaded_files: Set[Path] = set()
        self._lock = threading.Lock()
        # Discover all pickle files once (not on every get_chunk call)
        self._all_pkls: Optional[List[Path]] = None

    def _discover_pkls(self) -> List[Path]:
        """Lazily discover all pickle files once."""
        if self._all_pkls is None:
            with self._lock:
                if self._all_pkls is None:
                    self._all_pkls = [p for p in self.root.glob("*.pkl") if p.name not in self.skip]
        return self._all_pkls

    def _ensure_file_loaded(self, pkl: Path):
        """Lazily load a single pickle file if not already loaded."""
        if pkl in self._loaded_files:
            return
        with self._lock:
            if pkl in self._loaded_files:
                return
            try:
                with open(pkl, "rb") as f:
                    chunks = pickle.load(f)
                for ch in chunks:
                    meta = getattr(ch, "metadata", {}) if hasattr(ch, "metadata") else {}
                    doc_id = _norm_id(meta.get("document_id"))
                    chunk_id = _norm_id(meta.get("chunk_id"))
                    text = getattr(ch, "page_content", None)
                    if doc_id and chunk_id and isinstance(text, str):
                        self._index[(doc_id, chunk_id)] = text
                        self._by_chunk.setdefault(chunk_id, []).append((doc_id, chunk_id))
                self._loaded_files.add(pkl)
            except Exception:
                self._loaded_files.add(pkl)

    def get_chunk(self, document_id: Any, chunk_id: Any) -> Optional[str]:
        doc_id_s = _norm_id(document_id)
        chunk_id_s = _norm_id(chunk_id)

        # Fast path: already in memory
        val = self._index.get((doc_id_s, chunk_id_s))
        if val is not None:
            return val

        # Try base_id fallback without loading a file
        if "::" in chunk_id_s:
            base_id = chunk_id_s.split("::", 1)[0]
            val = self._index.get((doc_id_s, base_id))
            if val is not None:
                return val

        # Need to search by chunk_id — check if we have matches already loaded
        matches = self._by_chunk.get(chunk_id_s)
        if matches:
            chosen_doc, chosen_chunk = matches[0]
            val = self._index.get((chosen_doc, chosen_chunk))
            if val is not None:
                return val

        # Not found — try lazy loading every unloaded pickle file
        for pkl in self._discover_pkls():
            if pkl in self._loaded_files:
                continue
            self._ensure_file_loaded(pkl)
            # Check again after loading
            val = self._index.get((doc_id_s, chunk_id_s))
            if val is not None:
                return val
            if "::" in chunk_id_s:
                base_id = chunk_id_s.split("::", 1)[0]
                val = self._index.get((doc_id_s, base_id))
                if val is not None:
                    return val

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
    All entity embedding + search operations run in parallel.
    cypher_durations: accumulator dict for Cypher query durations.
    matched_entities_log: list to append matched entity info for logging.
    """
    # Prepare entity texts, skipping empty ones
    entity_work: List[Tuple[str, str]] = []  # (original_text, stripped_text)
    for e in query_entities:
        text = (e.get("text") or "").strip()
        if text:
            entity_work.append((e.get("text", "").strip(), text))

    if not entity_work:
        return []

    # Each entity task: embed + 3-index search
    def process_entity(item: Tuple[str, str]) -> Tuple[List[Dict[str, Any]], float]:
        _, text = item
        e_emb = embed_text(text)
        matches, sim_dur = search_similar_entities_by_embedding(e_emb, k=ENTITY_MATCH_TOP_K)
        return matches, sim_dur

    all_matched_keys: Set[str] = set()
    all_matched_ids: Set[str] = set()
    total_entity_dur = 0.0

    with ThreadPoolExecutor(max_workers=LLM_CONCURRENCY) as executor:
        futures = {executor.submit(process_entity, item): item for item in entity_work}
        for fut in as_completed(futures):
            matches, sim_dur = fut.result()
            total_entity_dur += sim_dur
            keys = [m.get("key") for m in matches if m.get("key")]
            ids  = [m.get("elem_id") for m in matches if m.get("elem_id")]
            all_matched_keys.update(keys)
            all_matched_ids.update(ids)
            if matched_entities_log is not None:
                orig_text = futures[fut][0]
                for m in matches:
                    matched_entities_log.append({
                        "query_entity": orig_text,
                        "kg_entity": m.get("name") or m.get("key"),
                        "type": m.get("type"),
                        "score": m.get("score"),
                        "elem_id": m.get("elem_id"),
                        "chunk_id": m.get("chunk_id"),
                    })

    if cypher_durations is not None:
        cypher_durations["entity_vector_search"] = cypher_durations.get("entity_vector_search", 0.0) + total_entity_dur

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
    # Prepare texts for all query triples
    triple_work: List[Tuple[Dict[str, Any], str]] = []
    for qt in query_triples:
        txt = query_triple_to_text(qt)
        triple_work.append((qt, txt))

    if not triple_work:
        return [], []

    # Each triple task: embed + vector search
    def process_triple(item: Tuple[Dict[str, Any], str]) -> Tuple[List[float], List[Dict[str, Any]], float]:
        qt, txt = item
        emb = embed_text(txt)
        matches, dur = search_similar_triples_by_embedding(emb, k=QUERY_TRIPLE_MATCH_TOP_K_PER)
        return emb, matches, dur

    triples_map: Dict[str, Dict[str, Any]] = {}
    emb_results: List[Tuple[int, List[float]]] = []  # (index, embedding)
    total_dur = 0.0

    with ThreadPoolExecutor(max_workers=LLM_CONCURRENCY) as executor:
        futures = {executor.submit(process_triple, item): idx for idx, item in enumerate(triple_work)}
        for fut in as_completed(futures):
            idx = futures[fut]
            emb, matches, dur = fut.result()
            emb_results.append((idx, emb))
            total_dur += dur
            for m in matches:
                uid = m.get("triple_uid")
                if uid:
                    if uid not in triples_map:
                        triples_map[uid] = m
                    elif m.get("score", 0.0) > triples_map[uid].get("score", 0.0):
                        triples_map[uid] = m

    # Restore original order
    emb_results.sort(key=lambda x: x[0])
    q_trip_embs = [emb for _, emb in emb_results]

    if cypher_durations is not None:
        cypher_durations["triple_vector_search"] = cypher_durations.get("triple_vector_search", 0.0) + total_dur

    return list(triples_map.values()), q_trip_embs

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
            raise RuntimeError(
                f"Triple {t.get('triple_uid')!r} has missing document_id or chunk_id."
            )

        norm_key = (_norm_id(doc_id), _norm_id(chunk_id))
        if norm_key in seen_pairs:
            continue

        text = chunk_store.get_chunk(doc_id, chunk_id)
        if text is None:
            raise RuntimeError(
                f"Chunk not found for triple {t.get('triple_uid')!r}: "
                f"document_id={doc_id!r}, chunk_id={chunk_id!r}."
            )
        t["_is_quote_fallback"] = False
        out.append((norm_key, text, t))
        seen_pairs.add(norm_key)

    return out

def rerank_chunks_by_query(
    chunk_records: List[Tuple[Tuple[Any, Any], str, Dict[str, Any]]],
    q_emb_query: List[float],
    top_k: int
) -> List[Tuple[Tuple[Any, Any], str, Dict[str, Any], float]]:
    cand = chunk_records[:CHUNK_RERANK_CAND_LIMIT]
    if not cand:
        return []

    # Parallelize embedding of all candidate chunks
    cand_texts = [(i, text) for i, (key, text, t) in enumerate(cand)]
    emb_results: List[Tuple[int, List[float]]] = []

    def embed_one(item: Tuple[int, str]) -> Tuple[int, List[float]]:
        idx, text = item
        emb = embed_text(text)
        return idx, emb

    with ThreadPoolExecutor(max_workers=LLM_CONCURRENCY) as executor:
        futures = {executor.submit(embed_one, item): item for item in cand_texts}
        for fut in as_completed(futures):
            idx, emb = fut.result()
            s = cos_sim(q_emb_query, emb)
            emb_results.append((idx, s))

    # Map back to original records and sort
    scored: List[Tuple[Tuple[Any, Any], str, Dict[str, Any], float]] = []
    for orig_idx, score_val in emb_results:
        key, text, t = cand[orig_idx]
        scored.append((key, text, t, score_val))

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
    global _log_file_logger
    logger = FileLogger(log_file, also_console=True)
    _thread_local.logger = logger
    _log_file_logger = logger
    _thread_local.question_id = question_id
    _thread_local.cypher_hop = None

    t_all_start = now_ms()
    try:
        chunk_store = ChunkStore(LANGCHAIN_DIR, set(SKIP_FILES))

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
        chunk_records = collect_chunks_for_triples(merged_triples, chunk_store)
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
        global _log_file_logger
        _log_file_logger = None
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

    with ThreadPoolExecutor(max_workers=LLM_CONCURRENCY) as executor:
        global_idx = skipped_count  # already-processed count
        # Submit questions one sliding-window batch at a time so we don't
        # queue thousands of futures before detecting an error.
        # With LLM_CONCURRENCY=1 this is fully sequential.
        chunk_size = max(1, LLM_CONCURRENCY)
        for batch_start in range(0, len(pending_pairs), chunk_size):
            batch = pending_pairs[batch_start : batch_start + chunk_size]
            futures: List[Future] = []
            for qa in batch:
                global_idx += 1
                fut = executor.submit(process_one, qa, global_idx)
                futures.append(fut)
            for fut in as_completed(futures):
                exc = fut.exception()
                if exc:
                    driver.close()
                    raise exc

    driver.close()
    print(f"[Done] Results saved to: {OUTPUT_FILE}")