# lexidkg_graphrag_local.py
# Single-pass GraphRAG using local LLM (LM Studio) and local embeddings (BAAI/bge-m3).
# - Agent 1: extracts S-P-O triples (entities derived from subjects/objects)
# - Agent 2: generates answer from retrieved context
# - Processes all questions from qa_pairs_local_filtered_sampled_2000.jsonl with LLM_CONCURRENCY
# - Real-time answers JSONL + per-question log files

import os, time, json, math, pickle, re, random, shutil, threading, concurrent.futures
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, Set
from threading import Lock

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

# LM Studio LLM (local)
LMSTUDIO_BASE_URL = os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
LOCAL_GEN_MODEL   = os.getenv("LOCAL_GEN_MODEL", "qwen/qwen3.5-9b")
LLM_REQUEST_TIMEOUT = float(os.getenv("LLM_REQUEST_TIMEOUT", "1200"))

# Local embedding (BAAI/bge-m3)
LOCAL_EMBED_MODEL = os.getenv("LOCAL_EMBED_MODEL", "BAAI/bge-m3")

# LLM Concurrency (sliding window: N questions processed simultaneously)
LLM_CONCURRENCY = int(os.getenv("LLM_CONCURRENCY", "15"))

# Dataset folders
_REPO_ROOT = (Path(__file__).resolve().parent / ".." / ".." / ".." ).resolve()
DEFAULT_LANGCHAIN_DIR = (_REPO_ROOT / "dataset" / "3_indexing" / "3a_langchain_results").resolve()
LANGCHAIN_DIR = Path(os.getenv("LANGCHAIN_DIR") or str(DEFAULT_LANGCHAIN_DIR))
SKIP_FILES = {"all_langchain_documents.pkl"}

# Input QA pairs file (produced by filter_sample_qa_pairs_local.py)
DEFAULT_QA_FILE = (_REPO_ROOT / "dataset" / "4_experiment" / "4a_qa_generation" /
                   "4a_ii_qa_pairs" / "qa_pairs_local_filtered_sampled_2000.jsonl")
QA_FILE = Path(os.getenv("QA_FILE") or str(DEFAULT_QA_FILE))

# Output directory for answers JSONL
DEFAULT_ANSWERS_DIR = (_REPO_ROOT / "dataset" / "4_experiment" / "4b_experiment_answers" /
                       "4b_i_lexidkg_graphrag_local")
ANSWERS_DIR = Path(os.getenv("ANSWERS_DIR") or str(DEFAULT_ANSWERS_DIR))

# Log directory for per-question logs
DEFAULT_LOGS_DIR = (Path(__file__).resolve().parent / "logs")
LOGS_DIR = Path(os.getenv("LOGS_DIR") or str(DEFAULT_LOGS_DIR))

# ----------------- Retrieval/agent parameters (unchanged from original) -----------------
ENTITY_MATCH_TOP_K = 15
ENTITY_SUBGRAPH_HOPS = 4
ENTITY_SUBGRAPH_PER_HOP_LIMIT = 2000
SUBGRAPH_TRIPLES_TOP_K = 30

QUERY_TRIPLE_MATCH_TOP_K_PER = 20

MAX_TRIPLES_FINAL = 60
MAX_CHUNKS_FINAL = 40
CHUNK_RERANK_CAND_LIMIT = 200

ANSWER_MAX_TOKENS = 4096

# ----------------- Initialize LM Studio client -----------------
lmstudio_client = OpenAI(base_url=LMSTUDIO_BASE_URL, api_key="lm-studio", timeout=LLM_REQUEST_TIMEOUT)

# ----------------- Initialize BGE-M3 embedding model -----------------
_bge_model = SentenceTransformer(LOCAL_EMBED_MODEL, trust_remote_code=True)
_bge_lock = threading.Lock()  # BGE-M3 encode is NOT thread-safe

# ----------------- Neo4j driver -----------------
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

# ----------------- Shared state (per-process) -----------------
# ChunkStore is built once per process and shared across all questions
_shared_chunk_store: Optional["ChunkStore"] = None
_shared_chunk_store_lock = Lock()

def _get_shared_chunk_store() -> "ChunkStore":
    global _shared_chunk_store
    with _shared_chunk_store_lock:
        if _shared_chunk_store is None:
            _shared_chunk_store = ChunkStore(LANGCHAIN_DIR, set(SKIP_FILES))
        return _shared_chunk_store

# LLM concurrency: ThreadPoolExecutor + semaphore
_semaphore = threading.Semaphore(LLM_CONCURRENCY)

# Shared file lock for answers JSONL (append from multiple threads)
_answers_lock = threading.Lock()

# ----------------- Logging utilities -----------------
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

def _fmt_prefix(qid: int, level: str = "INFO") -> str:
    return f"[{_now_ts()}] [{level}] [pid={_pid()}] [qid={qid}]"

class PerQuestionLogger:
    """Writes to both a per-question log file and the terminal, with qid prefix."""
    def __init__(self, file_path: Path, qid: int, also_console: bool = True):
        self.file_path = file_path
        self.qid = qid
        self.also_console = also_console
        self._fh = open(file_path, "w", encoding="utf-8")
        self._lock = Lock()

    def log(self, msg: str = ""):
        out = f"{_fmt_prefix(self.qid)} {msg}"
        with self._lock:
            self._fh.write(out + "\n")
            self._fh.flush()
            if self.also_console:
                print(out, flush=True)

    def close(self):
        try:
            with self._lock:
                self._fh.flush()
                self._fh.close()
        except Exception:
            pass

# Per-question logger: set per thread/task
_thread_logger: threading.local = threading.local()

def set_question_logger(qid: int, log_dir: Path):
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = PerQuestionLogger(log_dir / f"q{qid:04d}.txt", qid)
    setattr(_thread_logger, "logger", logger)

def clear_question_logger():
    logger = getattr(_thread_logger, "logger", None)
    if logger:
        logger.close()
    setattr(_thread_logger, "logger", None)

def _log(msg: str = ""):
    logger = getattr(_thread_logger, "logger", None)
    if logger:
        logger.log(msg)
    else:
        print(f"{_fmt_prefix(-1)} {msg}", flush=True)

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

def _rand_wait_seconds() -> float:
    return random.uniform(5.0, 20.0)

def _api_call_with_retry(func, *args, **kwargs):
    while True:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            wait_s = _rand_wait_seconds()
            _log(f"[Retry] API call failed: {e}. Retrying in {wait_s:.1f}s.")
            time.sleep(wait_s)

# ----------------- Embedding: BGE-M3 (local, thread-safe) -----------------
def embed_text(text: str) -> List[float]:
    with _bge_lock:
        vec = _bge_model.encode(
            [text],
            batch_size=1,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False
        )[0].tolist()
    return vec

def cos_sim(a: List[float], b: List[float]) -> float:
    a = _as_float_list(a)
    b = _as_float_list(b)
    dot = sum(x*y for x,y in zip(a,b))
    na = math.sqrt(sum(x*x for x in a))
    nb = math.sqrt(sum(y*y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)

# ----------------- Language detection (ID vs EN) -----------------
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
    if score_id > score_en:
        return "id"
    if score_en > score_id:
        return "en"
    return "en"

# ----------------- LM Studio LLM helpers -----------------
def _strip_markdown_json(text: str) -> str:
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()

def safe_generate_json(prompt: str, temp: float = 0.0) -> Dict[str, Any]:
    t0 = now_ms()
    resp = _api_call_with_retry(
        lmstudio_client.chat.completions.create,
        model=LOCAL_GEN_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=temp,
    )
    took = dur_ms(t0)
    _log(f"[LLM] Agent 1 | duration_ms={took:.0f}")
    try:
        raw = resp.choices[0].message.content or ""
        raw = _strip_markdown_json(raw)
        return json.loads(raw)
    except Exception as e:
        _log(f"[LLM] Agent 1 | parse error: {e} | raw: {raw[:200]}")
        return {}

def safe_generate_text(prompt: str, max_tokens: int, temperature: float = 0.2) -> Tuple[str, float]:
    t0 = now_ms()
    resp = _api_call_with_retry(
        lmstudio_client.chat.completions.create,
        model=LOCAL_GEN_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    took = dur_ms(t0)
    text = (resp.choices[0].message.content or "").strip()
    _log(f"[LLM] Agent 2 | duration_ms={took:.0f}")
    return text, took

# ----------------- Agent 1: merged triple extraction + entity derivation -----------------
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

def agent1_extract_triples_and_entities(query: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    prompt = f"""
You are Agent 1. Task: extract explicit or implied triples from the user's question in the form:
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
    _log(f"[Agent 1] Prompt size: {len(prompt)} chars, est_tokens≈{est_tokens}")

    out = safe_generate_json(prompt, temp=0.0)
    _log(f"[Agent 1] Raw output: {json.dumps(out, ensure_ascii=False)[:500]}")

    triples_raw = out.get("triples", []) if isinstance(out, dict) else []

    # Sanitize and collect
    triples: List[Dict[str, Any]] = []
    seen_entities: Dict[str, Dict[str, Any]] = {}

    for t in triples_raw:
        try:
            s = t.get("subject", {}) or {}
            o = t.get("object", {}) or {}
            p = (t.get("predicate") or "").strip()
            if s.get("text") and o.get("text") and p:
                s_text = s.get("text", "").strip()
                o_text = o.get("text", "").strip()
                s_type = (s.get("type") or "").strip()
                o_type = (o.get("type") or "").strip()

                triples.append({
                    "subject": {"text": s_text, "type": s_type},
                    "predicate": p,
                    "object": {"text": o_text, "type": o_type},
                })

                # Derive entities from subjects and objects
                for ent_text, ent_type in [(s_text, s_type), (o_text, o_type)]:
                    if ent_text and ent_text not in seen_entities:
                        seen_entities[ent_text] = {"text": ent_text, "type": ent_type}
                    elif ent_text in seen_entities and not seen_entities[ent_text]["type"] and ent_type:
                        seen_entities[ent_text]["type"] = ent_type
        except Exception:
            continue

    _log(f"[Agent 1] Extracted triples: {['{} [{}] {}'.format(x['subject']['text'], x['predicate'], x['object']['text']) for x in triples]}")
    entities = list(seen_entities.values())
    _log(f"[Agent 1] Derived entities for entity-centric retrieval: {[e['text'] for e in entities]}")

    return triples, entities

def query_triple_to_text(t: Dict[str, Any]) -> str:
    s = ((t.get("subject") or {}).get("text") or "").strip()
    p = (t.get("predicate") or "").strip()
    o = ((t.get("object") or {}).get("text") or "").strip()
    return f"{s} [{p}] {o}"

# ----------------- Neo4j helpers -----------------
_NEO4J_QUERY_SEQ = 0
_NEO4J_QUERY_LOCK = Lock()
_NEO4J_CUMULATIVE_MS = 0.0  # cumulative cypher duration for current question

def _next_query_id() -> int:
    global _NEO4J_QUERY_SEQ
    with _NEO4J_QUERY_LOCK:
        _NEO4J_QUERY_SEQ += 1
        return _NEO4J_QUERY_SEQ

def _reset_cypher_cumulative():
    global _NEO4J_CUMULATIVE_MS
    _NEO4J_CUMULATIVE_MS = 0.0

def _get_cypher_cumulative() -> float:
    return _NEO4J_CUMULATIVE_MS

def run_cypher_with_retry(cypher: str, params: Dict[str, Any]) -> List[Any]:
    global _NEO4J_CUMULATIVE_MS
    attempts = 0
    last_e: Optional[Exception] = None
    qid = _next_query_id()
    preview = " ".join((cypher or "").split())
    if len(preview) > 220:
        preview = preview[:220] + "..."
    while attempts < max(1, NEO4J_MAX_ATTEMPTS):
        attempts += 1
        t0 = now_ms()
        try:
            with driver.session() as session:
                res = session.run(cypher, **params, timeout=NEO4J_TX_TIMEOUT_S)
                records = list(res)
            took = dur_ms(t0)
            _NEO4J_CUMULATIVE_MS += took
            return records
        except Exception as e:
            took = dur_ms(t0)
            _NEO4J_CUMULATIVE_MS += took
            last_e = e
            if attempts >= NEO4J_MAX_ATTEMPTS:
                break
            wait_s = random.uniform(5, 10)
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
    except Exception:
        pass
    try:
        candidates.extend(_vector_query_nodes("content_vec", q_emb, k))
    except Exception:
        pass
    try:
        candidates.extend(_vector_query_nodes("expression_vec", q_emb, k))
    except Exception:
        pass

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

# ----------------- Graph expansion -----------------
def expand_from_entities(
    entity_keys: List[str],
    hops: int,
    per_hop_limit: int,
    entity_elem_ids: Optional[List[str]] = None
) -> Tuple[List[Dict[str, Any]], Dict[int, int]]:
    """
    Traverse PREDICATE edges from matched entities.
    Returns (triples, hop_entity_counts) where hop_entity_counts maps hop_num -> no. of retrieved entities.
    """
    triples: Dict[str, Dict[str, Any]] = {}
    current_ids: Set[str] = set(x for x in (entity_elem_ids or []) if x)
    current_keys: Set[str] = set(x for x in (entity_keys or []) if x)
    hop_entity_counts: Dict[int, int] = {}

    for hop_num in range(1, hops + 1):
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
        hop_entity_counts[hop_num] = len(res)
        _log(f"[Neo4j] Graph expansion hop {hop_num}: {len(res)} entities retrieved")

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

    return list(triples.values()), hop_entity_counts

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
                except Exception:
                    continue
            elapsed = time.monotonic() - start
            _log(f"[ChunkStore] Indexed {total_chunks_indexed} chunks from {len(self._loaded_files)} files in {elapsed:.3f}s.")
            self._built = True

    def get_chunk(self, document_id: Any, chunk_id: Any) -> Optional[str]:
        if not self._built:
            self._build_index()
        doc_id_s = _norm_id(document_id)
        chunk_id_s = _norm_id(chunk_id)
        val = self._index.get((doc_id_s, chunk_id_s))
        if val is not None:
            return val
        if "::" in chunk_id_s:
            base_id = chunk_id_s.split("::", 1)[0]
            val = self._index.get((doc_id_s, base_id))
            if val is not None:
                return val
        matches = self._by_chunk.get(chunk_id_s)
        if matches:
            chosen_doc, chosen_chunk = matches[0]
            val = self._index.get((chosen_doc, chosen_chunk))
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

# ----------------- Retrieval pipeline pieces -----------------
def entity_centric_retrieval(
    query_entities: List[Dict[str, Any]],
    q_trip_embs: List[List[float]],
    q_emb_fallback: Optional[List[float]] = None
) -> Tuple[List[Dict[str, Any]], Dict[int, int]]:
    all_matched_keys: Set[str] = set()
    all_matched_ids: Set[str] = set()

    for e in query_entities:
        text = (e.get("text") or "").strip()
        if not text:
            continue
        try:
            e_emb = embed_text(text)
        except Exception as ex:
            _log(f"[EntityRetrieval] Embedding failed for entity '{text}': {ex}")
            continue
        matches = search_similar_entities_by_embedding(e_emb, k=ENTITY_MATCH_TOP_K)
        keys = [m.get("key") for m in matches if m.get("key")]
        ids  = [m.get("elem_id") for m in matches if m.get("elem_id")]
        all_matched_keys.update(keys)
        all_matched_ids.update(ids)
        _log(f"[EntityRetrieval] '{text}' -> matched {len(keys)} KG keys")

    if not (all_matched_keys or all_matched_ids):
        _log("[EntityRetrieval] No KG entity matches found.")
        return [], {}

    expanded_triples, hop_counts = expand_from_entities(
        list(all_matched_keys),
        hops=ENTITY_SUBGRAPH_HOPS,
        per_hop_limit=ENTITY_SUBGRAPH_PER_HOP_LIMIT,
        entity_elem_ids=list(all_matched_ids) if all_matched_ids else None
    )
    _log(f"[EntityRetrieval] Expanded subgraph triples: {len(expanded_triples)}")

    if not expanded_triples:
        return [], {}

    def score(t: Dict[str, Any]) -> float:
        emb = t.get("embedding")
        if q_trip_embs:
            return mean_similarity_to_query_triples(emb, q_trip_embs)
        if q_emb_fallback and isinstance(emb, list):
            return cos_sim(q_emb_fallback, emb)
        return 0.0

    ranked = sorted(expanded_triples, key=score, reverse=True)
    top = ranked[:SUBGRAPH_TRIPLES_TOP_K]
    _log(f"[EntityRetrieval] Selected top-{len(top)} triples from subgraph")
    return top, hop_counts

def triple_centric_retrieval(query_triples: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[List[float]]]:
    triples_map: Dict[str, Dict[str, Any]] = {}
    q_trip_embs: List[List[float]] = []
    for qt in query_triples:
        try:
            txt = query_triple_to_text(qt)
            emb = embed_text(txt)
            q_trip_embs.append(emb)
        except Exception as ex:
            _log(f"[TripleRetrieval] Embedding failed for query triple '{qt}': {ex}")
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
    _log(f"[TripleRetrieval] Collected {len(merged)} unique KG triples across {len(q_trip_embs)} query triple(s)")
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
            _log(f"[ChunkRerank] Embedding failed for chunk {key}: {ex}")
            continue
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
def agent2_answer(
    query_original: str,
    context: str,
    guidance: Optional[str],
    output_lang: str = "id"
) -> Tuple[str, float]:
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
    _log(f"[Agent 2] Prompt size: {len(prompt)} chars, est_tokens≈{est_tokens}")
    answer, took = safe_generate_text(prompt, max_tokens=ANSWER_MAX_TOKENS, temperature=0.2)
    _log(f"[Agent 2] Answer length={len(answer)} | {took:.0f} ms")
    return answer, took

# ----------------- Per-question processing -----------------
def process_question(qid: int, question_text: str, answers_file: Path):
    global _shared_chunk_store
    _reset_cypher_cumulative()
    set_question_logger(qid, LOGS_DIR)

    try:
        _log(f"=== Question processing started ===")
        chunk_store = _get_shared_chunk_store()

        user_lang = detect_user_language(question_text)
        _log(f"[Language] Detected: {user_lang}")

        # Step 0: Embed whole user query
        t0 = now_ms()
        q_emb_query = embed_text(question_text)
        t_step0 = dur_ms(t0)
        _log(f"[Step 0] Embed query | duration_ms={t_step0:.0f}")

        # Step 1: Agent 1 – extract triples and derive entities
        t1 = now_ms()
        query_triples, ents = agent1_extract_triples_and_entities(question_text)
        t_step1 = dur_ms(t1)
        _log(f"[Step 1] Agent 1 | duration_ms={t_step1:.0f} | triples={len(query_triples)} | entities={len(ents)}")

        # Step 2: Triple-centric retrieval
        t2 = now_ms()
        ctx2_triples, q_trip_embs = triple_centric_retrieval(query_triples)
        t_step2 = dur_ms(t2)
        _log(f"[Step 2] Triple-centric retrieval | duration_ms={t_step2:.0f} | ctx2_triples={len(ctx2_triples)}")

        # Step 3: Entity-centric retrieval
        t3 = now_ms()
        ctx1_triples, hop_counts = entity_centric_retrieval(ents, q_trip_embs=q_trip_embs, q_emb_fallback=q_emb_query)
        t_step3 = dur_ms(t3)
        for hop_num, count in sorted(hop_counts.items()):
            _log(f"[Step 3] Entity-centric hop {hop_num}: {count} entities")
        _log(f"[Step 3] Entity-centric retrieval | duration_ms={t_step3:.0f} | ctx1_triples={len(ctx1_triples)}")

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
        t_step4 = dur_ms(t4)
        _log(f"[Step 4] Merge triples | duration_ms={t_step4:.0f} | merged={len(merged_triples)}")

        # Step 5: Gather chunks and rerank
        t5 = now_ms()
        chunk_records = collect_chunks_for_triples(merged_triples, chunk_store)
        _log(f"[Step 5] Collected {len(chunk_records)} chunk candidates (pre-rerank)")
        chunks_ranked = rerank_chunks_by_query(chunk_records, q_emb_query, top_k=MAX_CHUNKS_FINAL)
        t_step5 = dur_ms(t5)
        chunk_ids_sent = [str(k[1]) for k, _, _ in chunk_records]
        _log(f"[Step 5] Chunk rerank | duration_ms={t_step5:.0f} | selected={len(chunks_ranked)}")
        _log(f"[Agent 2] Chunk IDs sent to LLM: {chunk_ids_sent}")

        # Step 6: Rerank triples
        t6 = now_ms()
        triples_ranked = rerank_triples_by_query_triples(
            merged_triples, q_trip_embs=q_trip_embs, q_emb_fallback=q_emb_query, top_k=MAX_TRIPLES_FINAL
        )
        t_step6 = dur_ms(t6)
        _log(f"[Step 6] Triple rerank | duration_ms={t_step6:.0f} | selected={len(triples_ranked)}")

        # Build combined context text
        context_text, context_summary, _ = build_combined_context_text(triples_ranked, chunks_ranked)
        _log(f"[Agent 2] Triples sent to LLM: {['{} [{}] {}'.format(x.get('subject'), x.get('predicate'), x.get('object')) for x in triples_ranked]}")

        # Step 7: Agent 2 – Answer
        t7 = now_ms()
        answer, _ = agent2_answer(question_text, context_text, guidance=None, output_lang=user_lang)
        t_step7 = dur_ms(t7)
        _log(f"[Step 7] Agent 2 answer | duration_ms={t_step7:.0f}")

        # Durations summary
        total_ms = dur_ms(t0)
        cypher_cumulative = _get_cypher_cumulative()
        _log(f"[Durations] Step 0={t_step0:.0f}ms | Step 1={t_step1:.0f}ms | Step 2={t_step2:.0f}ms | Step 3={t_step3:.0f}ms | Step 4={t_step4:.0f}ms | Step 5={t_step5:.0f}ms | Step 6={t_step6:.0f}ms | Step 7={t_step7:.0f}ms")
        _log(f"[Durations] Cypher queries cumulative={cypher_cumulative:.0f}ms | Total={total_ms:.0f}ms")
        _log(f"=== Final Answer ===\n{answer}")
        _log(f"=== Question processing completed ===")

        # Write answer to JSONL immediately
        record = {"question_id": qid, "question": question_text, "generated_answer": answer}
        with _answers_lock:
            with open(answers_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()

        return {
            "question_id": qid,
            "final_answer": answer,
            "total_duration_ms": total_ms,
            "step_durations": {
                "step0_embed": t_step0,
                "step1_agent1": t_step1,
                "step2_triple_centric": t_step2,
                "step3_entity_centric": t_step3,
                "step4_merge": t_step4,
                "step5_chunks": t_step5,
                "step6_triple_rerank": t_step6,
                "step7_agent2": t_step7,
            },
            "cypher_cumulative_ms": cypher_cumulative,
        }
    finally:
        clear_question_logger()

# ----------------- Worker function (semaphore-wrapped) -----------------
def _worker(qid: int, question_text: str, answers_file: Path):
    _semaphore.acquire()
    try:
        return process_question(qid, question_text, answers_file)
    finally:
        _semaphore.release()

# ----------------- Main batch processor -----------------
def main():
    # Ensure output directories exist
    ANSWERS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    # Load questions
    if not Path(QA_FILE).exists():
        print(f"ERROR: Question file not found: {QA_FILE}")
        return

    questions: List[Tuple[int, str]] = []
    with open(QA_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            qid = record.get("question_id", len(questions))
            q_text = record.get("question", "")
            if q_text:
                questions.append((qid, q_text))

    print(f"Loaded {len(questions)} questions from {QA_FILE}")
    print(f"LLM Concurrency: {LLM_CONCURRENCY}")
    print(f"Output directory: {ANSWERS_DIR}")
    print(f"Log directory: {LOGS_DIR}")

    # Generate answers file name with timestamp
    ts = time.strftime("%Y%m%d-%H%M%S")
    answers_file = ANSWERS_DIR / f"graph_rag_answers_{ts}.jsonl"

    # Process all questions with concurrency
    results: List[Dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=LLM_CONCURRENCY) as executor:
        futures = {
            executor.submit(_worker, qid, q_text, answers_file): (qid, q_text)
            for qid, q_text in questions
        }
        for future in concurrent.futures.as_completed(futures):
            qid, q_text = futures[future]
            try:
                result = future.result()
                results.append(result)
                print(f"[Completed] qid={qid} | total={result['total_duration_ms']:.0f}ms | {len(results)}/{len(questions)} done")
            except Exception as e:
                print(f"[ERROR] qid={qid} failed: {e}")

    print(f"\nAll done. Processed {len(results)}/{len(questions)} questions.")
    print(f"Answers written to: {answers_file}")

    # Summary
    if results:
        durations = [r["total_duration_ms"] for r in results]
        print(f"\nDuration stats: min={min(durations):.0f}ms | max={max(durations):.0f}ms | avg={sum(durations)/len(durations):.0f}ms")

if __name__ == "__main__":
    try:
        main()
    finally:
        try:
            driver.close()
        except Exception:
            pass
