# -*- coding: utf-8 -*-
# naive_pipeline_local.py
# Local LMStudio version of naive_pipeline.py — uses:
#   - BAAI/bge-m3 via SentenceTransformers for embeddings (no Google API key needed)
#   - OpenAI-compatible API (LMStudio) for LLM generation (if used)
#   - Semaphore-based LLM_CONCURRENCY scheduling (N parallel workers at a time)
# The core indexing process (embed chunks → upsert to Neo4j) is unchanged.

import os, json, hashlib, time, threading, pickle, math, re, random
from collections import deque
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, Future

from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from neo4j import GraphDatabase

# ----------------- Thread-safe logging with timestamps -----------------
_PRINT_LOCK = threading.Lock()

def _ts() -> str:
    now = time.time()
    lt = time.localtime(now)
    ms = int((now - int(now)) * 1000)
    return time.strftime("%Y-%m-%d %H:%M:%S", lt) + f".{ms:03d}"

def _fmt_prefix(level: str) -> str:
    return f"[{_ts()}] [{level}]"

def log_info(msg: str) -> None:
    with _PRINT_LOCK:
        print(f"{_fmt_prefix('INFO')} {msg}")

def log_warn(msg: str) -> None:
    with _PRINT_LOCK:
        print(f"{_fmt_prefix('WARN')} {msg}")

def log_error(msg: str) -> None:
    with _PRINT_LOCK:
        print(f"{_fmt_prefix('ERROR')} {msg}")

# ----------------- Load .env from the parent directory -----------------
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

# ----------------- Config from env with sensible defaults -----------------
# LMStudio local server (OpenAI-compatible)
LMSTUDIO_BASE_URL = os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
LOCAL_GEN_MODEL   = os.getenv("LOCAL_GEN_MODEL",   "qwen/qwen3.5-9b")
LOCAL_EMBED_MODEL = os.getenv("LOCAL_EMBED_MODEL",  "BAAI/bge-m3")

# BGE-M3 output dimension (dense vector) is 1024 by default
EMBED_DIM = int(os.getenv("EMBED_DIM", "1024"))

# Per-request timeout in seconds for LMStudio LLM calls.
LLM_REQUEST_TIMEOUT = float(os.getenv("LLM_REQUEST_TIMEOUT", "1200"))

# Directory of LangChain per-document pickle files
_IS_SAMPLE = os.getenv("IS_SAMPLE", "false").strip().lower() in ("1", "true", "yes", "y", "on")
_ROOT = Path(__file__).resolve().parents[2]  # myrag/
if _IS_SAMPLE:
    LANGCHAIN_DIR = (_ROOT / "dataset" / "samples" / "3_indexing" / "3a_langchain_results").resolve()
else:
    LANGCHAIN_DIR = (_ROOT / "dataset" / "3_indexing" / "3a_langchain_results").resolve()

# Neo4j
NEO4J_URI  = os.getenv("NEO4J_URI",  "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASS", "@ik4nkus")

# Client-side RPM limiter (calls per minute) — embedding calls through bge-m3 are local,
# but the limiter is kept for parity and future use.
LLM_MAX_CALLS_PER_MIN = int(os.getenv("LLM_MAX_CALLS_PER_MIN", "1000000000"))

# Number of batches that may be processed concurrently.
# N=1 → fully sequential. N>1 → sliding window via semaphore: as soon as any
# in-flight batch finishes, the next queued batch starts immediately.
LLM_CONCURRENCY = int(os.getenv("LLM_CONCURRENCY", "1"))

# API budget controls
def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "y", "on")

ENFORCE_API_BUDGET = _env_bool("ENFORCE_API_BUDGET", True)
API_BUDGET_TOTAL = int(os.getenv("API_BUDGET_TOTAL", "100000000000"))
COUNT_EMBEDDINGS_IN_BUDGET = _env_bool("COUNT_EMBEDDINGS_IN_BUDGET", True)  # RAG indexing uses embeddings only

# Files to skip explicitly
SKIP_FILES = {"all_langchain_documents.pkl"}

# Chunk packing
PRACTICAL_MAX_ITEMS_PER_BATCH = int(os.getenv("PRACTICAL_MAX_ITEMS_PER_BATCH", "1"))

# Max characters of text to embed per chunk (safety clamp)
EMBED_TEXT_MAX_CHARS = int(os.getenv("EMBED_TEXT_MAX_CHARS", "6000"))

# Dedup by content hash (avoid re-embedding identical text within a run)
DEDUP_BY_CONTENT = _env_bool("DEDUP_BY_CONTENT", True)

# ----------------- Initialize SDKs -----------------
# LMStudio client (OpenAI-compatible) — available if LLM generation is needed
lmstudio_client = OpenAI(base_url=LMSTUDIO_BASE_URL, api_key="lm-studio", timeout=LLM_REQUEST_TIMEOUT)
log_info(f"LMStudio generation model: {LOCAL_GEN_MODEL}")
log_info(f"LMStudio base URL: {LMSTUDIO_BASE_URL}")

# BGE-M3 embedding model (loaded once, thread-safe for inference)
log_info(f"Loading embedding model via SentenceTransformers: {LOCAL_EMBED_MODEL} ...")
_bge_model = SentenceTransformer(LOCAL_EMBED_MODEL, trust_remote_code=True)
log_info(f"Embedding model loaded. Embed dim: {EMBED_DIM}")
log_info(f"Source LANGCHAIN_DIR: {LANGCHAIN_DIR} (IS_SAMPLE={_IS_SAMPLE})")

# ----------------- Neo4j -----------------
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

# ----------------- Rate Limiter -----------------
class RateLimiter:
    def __init__(self, max_calls: int, period_sec: float):
        self.max_calls = max_calls
        self.period = period_sec
        self.calls = deque()
        self.lock = threading.Lock()

    def acquire(self):
        with self.lock:
            now = time.time()
            while self.calls and (now - self.calls[0]) >= self.period:
                self.calls.popleft()
            wait_for = 0.0
            if len(self.calls) >= self.max_calls:
                wait_for = self.period - (now - self.calls[0])
        if wait_for > 0:
            log_info(f"[RateLimiter] Sleeping {wait_for:.2f}s to respect RPM limit")
            time.sleep(wait_for)
        with self.lock:
            self.calls.append(time.time())

EMBED_RATE_LIMITER = RateLimiter(max_calls=LLM_MAX_CALLS_PER_MIN, period_sec=60.0)

# ----------------- Global API Budget -----------------
class ApiBudget:
    def __init__(self, total: int, enforce: bool, count_embeddings: bool):
        self.total = total
        self.enforce = enforce
        self.count_embeddings = count_embeddings
        self.used = 0
        self.lock = threading.Lock()
        self.resource_usage = {"embed": 0}

    def _should_count(self, kind: str) -> bool:
        if kind == "embed":
            return self.count_embeddings
        return False

    def will_allow(self, kind: str, n: int = 1) -> bool:
        if not self.enforce or not self._should_count(kind):
            return True
        with self.lock:
            return (self.used + n) <= self.total

    def register(self, kind: str, n: int = 1):
        if not self.enforce or not self._should_count(kind):
            return
        with self.lock:
            if (self.used + n) > self.total:
                raise RuntimeError(f"API budget exceeded: used={self.used}, trying={n}, total={self.total}")
            self.used += n
            self.resource_usage[kind] = self.resource_usage.get(kind, 0) + n

API_BUDGET = ApiBudget(total=API_BUDGET_TOTAL, enforce=ENFORCE_API_BUDGET, count_embeddings=COUNT_EMBEDDINGS_IN_BUDGET)

# ----------------- Helpers -----------------
def sanitize_filename_component(s: str) -> str:
    s = (s or "").strip().lower().replace(" ", "-")
    s = re.sub(r"[^a-z0-9_\-+]", "", s)
    s = re.sub(r"-{2,}", "-", s)
    return s or "unknown"

def list_pickles(dir_path: Path) -> List[Path]:
    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {dir_path}")
    return sorted([p for p in dir_path.glob("*.pkl") if p.name not in SKIP_FILES])

def load_chunks_from_file(pkl_path: Path) -> List[Any]:
    with open(pkl_path, "rb") as f:
        chunks = pickle.load(f)
    if not isinstance(chunks, list):
        raise ValueError(f"Unexpected pickle content in {pkl_path}")
    log_info(f"  - {pkl_path.name}: {len(chunks)} chunks")
    return chunks

def chunk_text_for_embedding(text: str) -> str:
    t = text or ""
    if len(t) > EMBED_TEXT_MAX_CHARS:
        return t[:EMBED_TEXT_MAX_CHARS]
    return t

def content_sha256(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()

def make_chunk_key(meta: Dict[str, Any], idx: int, text: str) -> str:
    chunk_id = meta.get("chunk_id")
    doc_id = meta.get("document_id") or "unknown-doc"
    base = chunk_id or f"{doc_id}::chunk_{idx}"
    if DEDUP_BY_CONTENT:
        return f"{base}::{content_sha256(text)[:12]}"
    return base

# ----------------- Embeddings (local BGE-M3) -----------------
_emb_cache_lock = threading.Lock()
_emb_cache: Dict[str, List[float]] = {}  # key := content_sha256

def embed_text(text: str) -> Tuple[List[float], float]:
    if ENFORCE_API_BUDGET and not API_BUDGET.will_allow("embed", 1):
        raise RuntimeError("API budget would be exceeded by another embedding call; stopping embedding.")

    # Dedup embed calls by content hash within a run
    key = content_sha256(text)
    with _emb_cache_lock:
        if key in _emb_cache:
            return _emb_cache[key], 0.0

    EMBED_RATE_LIMITER.acquire()
    start = time.time()
    try:
        # BGE-M3 via SentenceTransformers — runs locally, no remote API call
        vec_array = _bge_model.encode(text, normalize_embeddings=True)
        vec: List[float] = vec_array.tolist()
    except Exception:
        raise

    dur = time.time() - start

    API_BUDGET.register("embed", 1)
    with _emb_cache_lock:
        _emb_cache[key] = vec

    return vec, dur

# ----------------- Neo4j storage (unchanged from naive_pipeline.py) -----------------
def upsert_chunk(tx, *, chunk_key: str, chunk_id: str, doc_id: str, uu_number: Optional[str],
                 pages: Any, content: str, content_hash: str, embedding: List[float]) -> float:
    start = time.time()
    cypher = """
    MERGE (c:TextChunk {key:$chunk_key})
      ON CREATE SET c.createdAt=timestamp()
    SET c.chunk_id=$chunk_id,
        c.document_id=$doc_id,
        c.uu_number=$uu_number,
        c.pages=$pages,
        c.content=$content,
        c.content_sha256=$content_hash,
        c.embedding=$embedding

    MERGE (d:Document {document_id:$doc_id})
      ON CREATE SET d.createdAt=timestamp()
    SET d.lastIndexedAt=timestamp()

    MERGE (d)-[:HAS_CHUNK]->(c)
    """
    tx.run(
        cypher,
        chunk_key=chunk_key,
        chunk_id=chunk_id,
        doc_id=doc_id,
        uu_number=uu_number,
        pages=pages if isinstance(pages, (list, tuple)) else pages,
        content=content,
        content_hash=content_hash,
        embedding=embedding,
    )
    return time.time() - start

def write_one_item(item: Dict[str, Any], idx_in_file: int) -> Tuple[bool, float, float]:
    meta = item["meta"]
    text = item["text"] or ""
    text_emb = chunk_text_for_embedding(text)
    c_hash = content_sha256(text_emb)
    chunk_key = make_chunk_key(meta, idx_in_file, text_emb)
    chunk_id = meta.get("chunk_id") or chunk_key
    doc_id = meta.get("document_id") or "unknown-doc"
    uu_number = meta.get("uu_number")
    pages = meta.get("pages")

    vec, emb_dur = embed_text(text_emb)
    with driver.session() as session:
        neo_dur = session.execute_write(
            upsert_chunk,
            chunk_key=chunk_key, chunk_id=chunk_id, doc_id=doc_id,
            uu_number=uu_number, pages=pages, content=text, content_hash=c_hash,
            embedding=vec
        )
    return True, emb_dur, neo_dur

# ----------------- Batch packing (unchanged from naive_pipeline.py) -----------------
def pack_batches(use_items: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    batches: List[List[Dict[str, Any]]] = []
    for i in range(0, len(use_items), PRACTICAL_MAX_ITEMS_PER_BATCH):
        batches.append(use_items[i:i+PRACTICAL_MAX_ITEMS_PER_BATCH])
    return batches

# ----------------- Main pipeline (semaphore-based concurrency, like lexidkg_pipeline_local.py) -----------------
def run_rag_index_over_folder(
    dir_path: Path,
    max_files: Optional[int] = None,
    max_chunks_per_file: Optional[int] = None
):
    pkls = list_pickles(dir_path)
    if max_files is not None:
        pkls = pkls[:max_files]

    raw_items_all: List[Dict[str, Any]] = []
    total_chunks_planned = 0

    log_info(f"Scanning pickle files in {dir_path}")
    for file_idx, pkl in enumerate(pkls, 1):
        try:
            chunks = load_chunks_from_file(pkl)
        except Exception as e:
            log_warn(f"Failed to load {pkl.name}: {e}")
            continue

        limit = len(chunks) if max_chunks_per_file is None else min(len(chunks), max_chunks_per_file)
        total_chunks_planned += limit

        for idx, ch in enumerate(chunks[:limit]):
            meta_source = getattr(ch, "metadata", {}) if hasattr(ch, "metadata") else {}
            meta = {
                "document_id": meta_source.get("document_id"),
                "chunk_id": meta_source.get("chunk_id"),
                "uu_number": meta_source.get("uu_number"),
                "pages": meta_source.get("pages"),
            }
            text = getattr(ch, "page_content", str(ch))
            if not meta.get("chunk_id"):
                meta["chunk_id"] = f"{pkl.stem}_chunk_{idx}"
            raw_items_all.append({"meta": meta, "text": text, "idx_in_file": idx})

    if not raw_items_all:
        log_info("No chunks found. Exiting.")
        return

    # Budget-aware truncation (embeddings only)
    if ENFORCE_API_BUDGET and COUNT_EMBEDDINGS_IN_BUDGET:
        allowed = max(0, API_BUDGET.total - API_BUDGET.used)
        if allowed < len(raw_items_all):
            log_warn(f"Budget cap: truncating items from {len(raw_items_all)} to {allowed}")
            raw_items_all = raw_items_all[:allowed]

    batches = pack_batches(raw_items_all)
    total_batches = len(batches)

    log_info(f"Prepared {total_batches} batch(es); chunks planned: {len(raw_items_all)}")
    log_info(f"API budget: enforce={API_BUDGET.enforce}, total={API_BUDGET.total}, used={API_BUDGET.used}")
    log_info(f"Embedding RPM limit: {LLM_MAX_CALLS_PER_MIN} calls/min")
    log_info(f"LLM concurrency (parallel batches): {LLM_CONCURRENCY}")

    total_done = 0
    total_emb_time = 0.0
    total_neo_time = 0.0
    overall_start = time.time()
    batches_completed = 0
    _results_lock = threading.Lock()

    # Semaphore limits how many batches are in-flight simultaneously.
    # When LLM_CONCURRENCY=1 this degenerates to purely sequential behaviour.
    _semaphore = threading.Semaphore(LLM_CONCURRENCY)

    def process_batch(batch_items: List[Dict[str, Any]]) -> Tuple[int, float, float]:
        done = 0
        emb_sum = 0.0
        neo_sum = 0.0

        for it in batch_items:
            # Budget gate (double-check at call-time)
            if ENFORCE_API_BUDGET and COUNT_EMBEDDINGS_IN_BUDGET and not API_BUDGET.will_allow("embed", 1):
                log_warn("Budget exhausted during batch; stopping batch early.")
                break

            try:
                ok, e_dur, n_dur = write_one_item(it, it["idx_in_file"])
            except Exception as e:
                msg = str(e).lower()
                if "quota" in msg:
                    log_warn(f"Quota error while embedding; stopping batch early: {e}")
                    break
                if "rate" in msg or "429" in msg or "resource exhausted" in msg:
                    sleep_for = random.uniform(5.0, 12.0)
                    log_warn(f"Rate-limit detected; sleeping {sleep_for:.1f}s and retrying this chunk once.")
                    time.sleep(sleep_for)
                    try:
                        ok, e_dur, n_dur = write_one_item(it, it["idx_in_file"])
                    except Exception as e2:
                        log_warn(f"Chunk failed after retry; skipping. Reason: {e2}")
                        continue
                else:
                    log_warn(f"Chunk failed; skipping. Reason: {e}")
                    continue

            if ok:
                done += 1
                emb_sum += e_dur
                neo_sum += n_dur

        return done, emb_sum, neo_sum

    def _run_batch(b_idx: int, items_batch: List[Dict[str, Any]]) -> None:
        """Worker executed in a thread pool slot. Releases the semaphore when done."""
        nonlocal total_done, total_emb_time, total_neo_time, batches_completed

        submit_time = time.time()
        try:
            done_count, emb_sum, neo_sum = process_batch(items_batch)
        except Exception as e:
            log_error(f"[Batch {b_idx+1}/{total_batches}] failed with unhandled error: {e}")
            with _results_lock:
                batches_completed += 1
                log_info(f"    - Batches completed: {batches_completed}/{total_batches}")
            _semaphore.release()
            return

        _semaphore.release()  # free the slot as soon as this batch is done

        wall = time.time() - submit_time
        with _results_lock:
            total_done += done_count
            total_emb_time += emb_sum
            total_neo_time += neo_sum
            batches_completed += 1

            log_info(f"✓ [Batch {b_idx+1}/{total_batches}] "
                     f"chunks={len(items_batch)} | done={done_count} | "
                     f"embed={emb_sum:.2f}s | neo4j={neo_sum:.2f}s | wall={wall:.2f}s")
            log_info(f"    - Batches completed: {batches_completed}/{total_batches}")

    with ThreadPoolExecutor(max_workers=LLM_CONCURRENCY) as executor:
        futures: List[Future] = []
        for b_idx, items_batch in enumerate(batches):
            # Block here until a concurrency slot becomes available.
            # The slot is released inside _run_batch once that batch finishes.
            _semaphore.acquire()

            fut = executor.submit(_run_batch, b_idx, items_batch)
            futures.append(fut)
        # executor.__exit__ waits for all submitted futures to complete

    total_wall = time.time() - overall_start
    sequential_est = total_emb_time + total_neo_time
    speedup = (sequential_est / total_wall) if total_wall > 0 else float('inf')

    log_info("Summary")
    log_info(f"- Chunks planned: {total_chunks_planned}")
    log_info(f"- Chunks processed: {total_done}/{len(raw_items_all)}")
    log_info(f"- API budget used: {API_BUDGET.resource_usage.get('embed', 0)}/{API_BUDGET.total} (embeddings counted={COUNT_EMBEDDINGS_IN_BUDGET})")
    log_info(f"- Total Embedding time (sum): {total_emb_time:.2f}s")
    log_info(f"- Total Neo4j time (sum): {total_neo_time:.2f}s")
    log_info(f"- Total wall time: {total_wall:.2f}s")
    log_info(f"- Sequential time estimate (components sum): {sequential_est:.2f}s")
    log_info(f"- Speedup vs sequential: {speedup:.2f}×")

if __name__ == "__main__":
    run_rag_index_over_folder(
        LANGCHAIN_DIR,
        max_files=10000,
        max_chunks_per_file=100000000000,
    )
