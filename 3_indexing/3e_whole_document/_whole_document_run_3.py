#!/usr/bin/env python3
# doc_index_pipeline.py
# Multithreaded document-level indexer: embed each LangChain Document (full or split)
# and store to Neo4j under a separate schema from your chunk graph.

import os, json, hashlib, time, threading, pickle, math, re, random
from collections import deque
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

from dotenv import load_dotenv
import google.generativeai as genai
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
GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY_3"]

def _normalize_model_name(m: str) -> str:
    if not m:
        return m
    m = m.strip()
    if m.startswith("models/") or m.startswith("tunedModels/"):
        return m
    return f"models/{m}"

EMBED_MODEL = _normalize_model_name(os.getenv("EMBED_MODEL", "text-embedding-004"))

# Directory of LangChain per-document pickle files (runner patches this)
DEFAULT_LANGCHAIN_DIR = Path(r"C:\Users\ROKADE\Documents\Joel TA\myrag\dataset\3_indexing\3f1_langchain_whole_document_batches\3").resolve()
LANGCHAIN_DIR = DEFAULT_LANGCHAIN_DIR

# Neo4j
NEO4J_URI  = os.getenv("NEO4J_URI",  "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASS", "neo4j")

# Client-side RPM limiter (calls per minute)
LLM_MAX_CALLS_PER_MIN = int(os.getenv("LLM_MAX_CALLS_PER_MIN", "90"))

# Parallelism
INDEX_WORKERS = int(os.getenv("INDEX_WORKERS", "20"))

# Stagger worker starts (seconds) - randomized uniformly for ramp-up parity
STAGGER_WORKER_SECONDS = random.uniform(7.0, 17.0)

# API budget controls
def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "y", "on")

ENFORCE_API_BUDGET = _env_bool("ENFORCE_API_BUDGET", True)
API_BUDGET_TOTAL = int(os.getenv("API_BUDGET_TOTAL", "100000"))
COUNT_EMBEDDINGS_IN_BUDGET = _env_bool("COUNT_EMBEDDINGS_IN_BUDGET", True)  # we embed only

# Files to skip explicitly
SKIP_FILES = {"all_langchain_documents.pkl", "all_full_langchain_documents.pkl"}

# Embedding dimensions for text-embedding-004 (reference only)
EMBED_DIM = 768

# Batch sizing
PRACTICAL_MAX_ITEMS_PER_BATCH = int(os.getenv("PRACTICAL_MAX_ITEMS_PER_BATCH", "200"))

# Max characters of text to embed per document (safety clamp)
EMBED_TEXT_MAX_CHARS = int(os.getenv("EMBED_TEXT_MAX_CHARS", "600000000000000000"))

# Dedup by content hash (avoid re-embedding identical text within a run)
DEDUP_BY_CONTENT = _env_bool("DEDUP_BY_CONTENT", True)

# ----------------- Initialize SDKs -----------------
genai.configure(api_key=GOOGLE_API_KEY)
log_info(f"Using embedding model: {EMBED_MODEL}")

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
def list_pickles(dir_path: Path) -> List[Path]:
    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {dir_path}")
    return sorted([p for p in dir_path.glob("*.pkl") if p.name not in SKIP_FILES])

def load_documents_from_file(pkl_path: Path) -> List[Any]:
    with open(pkl_path, "rb") as f:
        docs = pickle.load(f)
    if not isinstance(docs, list):
        raise ValueError(f"Unexpected pickle content in {pkl_path} (expected list[Document])")
    log_info(f"  - {pkl_path.name}: {len(docs)} document(s)")
    return docs

def doc_text_for_embedding(text: str) -> str:
    t = text or ""
    if len(t) > EMBED_TEXT_MAX_CHARS:
        return t[:EMBED_TEXT_MAX_CHARS]
    return t

def content_sha256(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()

def make_doc_key(meta: Dict[str, Any], idx: int, text_for_embed: str) -> str:
    """
    Stable unique key for MERGE:
      - Prefer file_id (unique per generated file).
      - Else chunk_id (from generator metadata).
      - Else document_id + idx fallback.
      - Optionally add content hash suffix if DEDUP_BY_CONTENT.
    """
    file_id = meta.get("file_id")
    chunk_id = meta.get("chunk_id")
    doc_id = meta.get("document_id") or "unknown-doc"
    base = file_id or chunk_id or f"{doc_id}::part_{idx}"
    if DEDUP_BY_CONTENT:
        return f"{base}::{content_sha256(text_for_embed)[:12]}"
    return base

# ----------------- Embeddings -----------------
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
        res = genai.embed_content(model=EMBED_MODEL, content=text)
    except Exception as e:
        raise

    dur = time.time() - start

    vec: Optional[List[float]] = None
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

    API_BUDGET.register("embed", 1)
    with _emb_cache_lock:
        _emb_cache[key] = vec

    return vec, dur

# ----------------- Neo4j storage (DocBundle/DocPart) -----------------
def upsert_docpart(tx, *, doc_key: str, file_id: str, bundle_id: str, parent_id: Optional[str],
                   uu_number: Optional[str], title: Optional[str], tanggal_berlaku: Optional[str],
                   pages: Any, split_idx: int, split_total: int,
                   content: str, content_hash: str, embedding: List[float]) -> float:
    start = time.time()
    cypher = """
    MERGE (p:DocPart {key:$doc_key})
      ON CREATE SET p.createdAt=timestamp()
    SET p.file_id=$file_id,
        p.document_id=$bundle_id,
        p.parent_document_id=$parent_id,
        p.uu_number=$uu_number,
        p.title=$title,
        p.tanggal_berlaku=$tanggal_berlaku,
        p.pages=$pages,
        p.split_index=$split_idx,
        p.split_total=$split_total,
        p.content=$content,
        p.content_sha256=$content_hash,
        p.embedding=$embedding,
        p.lastIndexedAt=timestamp()

    MERGE (b:DocBundle {document_id:$bundle_id})
      ON CREATE SET b.createdAt=timestamp()
    SET b.lastIndexedAt=timestamp()

    MERGE (b)-[:HAS_PART]->(p)
    """
    tx.run(
        cypher,
        doc_key=doc_key,
        file_id=file_id,
        bundle_id=bundle_id,
        parent_id=parent_id,
        uu_number=uu_number,
        title=title,
        tanggal_berlaku=tanggal_berlaku,
        pages=pages if isinstance(pages, (list, tuple)) else pages,
        split_idx=split_idx,
        split_total=split_total,
        content=content,
        content_hash=content_hash,
        embedding=embedding,
    )
    return time.time() - start

def write_one_item(item: Dict[str, Any], idx_in_file: int) -> Tuple[bool, float, float]:
    meta = item["meta"]
    text = item["text"] or ""
    text_emb = doc_text_for_embedding(text)
    c_hash = content_sha256(text_emb)

    doc_key = make_doc_key(meta, idx_in_file, text_emb)
    file_id = meta.get("file_id") or doc_key
    bundle_id = meta.get("document_id") or "unknown-doc"
    parent_id = meta.get("parent_document_id")  # optional
    uu_number = meta.get("uu_number")
    title = meta.get("title")
    tanggal_berlaku = meta.get("tanggal_berlaku")
    pages = meta.get("pages")
    split_idx = int(meta.get("chunk_index", 0))
    split_total = int(meta.get("total_chunks", 1))

    vec, emb_dur = embed_text(text_emb)
    with driver.session() as session:
        neo_dur = session.execute_write(
            upsert_docpart,
            doc_key=doc_key, file_id=file_id, bundle_id=bundle_id, parent_id=parent_id,
            uu_number=uu_number, title=title, tanggal_berlaku=tanggal_berlaku,
            pages=pages, split_idx=split_idx, split_total=split_total,
            content=text, content_hash=c_hash, embedding=vec
        )
    return True, emb_dur, neo_dur

# ----------------- Batch packing -----------------
def pack_batches(use_items: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    batches: List[List[Dict[str, Any]]] = []
    for i in range(0, len(use_items), PRACTICAL_MAX_ITEMS_PER_BATCH):
        batches.append(use_items[i:i+PRACTICAL_MAX_ITEMS_PER_BATCH])
    return batches

# ----------------- Main pipeline (parallel, RPM-limited) -----------------
def run_doc_index_over_folder(
    dir_path: Path,
    max_files: Optional[int] = None,
    max_docs_per_file: Optional[int] = None
):
    pkls = list_pickles(dir_path)
    if max_files is not None:
        pkls = pkls[:max_files]

    raw_items_all: List[Dict[str, Any]] = []
    total_docs_planned = 0

    log_info(f"Scanning pickle files in {dir_path}")
    for file_idx, pkl in enumerate(pkls, 1):
        try:
            docs = load_documents_from_file(pkl)
        except Exception as e:
            log_warn(f"Failed to load {pkl.name}: {e}")
            continue

        limit = len(docs) if max_docs_per_file is None else min(len(docs), max_docs_per_file)
        total_docs_planned += limit

        for idx, d in enumerate(docs[:limit]):
            meta_source = getattr(d, "metadata", {}) if hasattr(d, "metadata") else {}
            meta = {
                # Identification
                "file_id": meta_source.get("file_id"),
                "document_id": meta_source.get("document_id"),
                "parent_document_id": meta_source.get("parent_document_id"),

                # Compatibility / grouping
                "chunk_id": meta_source.get("chunk_id"),
                "pages": meta_source.get("pages"),
                "chunk_index": meta_source.get("chunk_index"),
                "total_chunks": meta_source.get("total_chunks"),

                # Searchable fields
                "uu_number": meta_source.get("uu_number"),
                "title": meta_source.get("title"),
                "tanggal_berlaku": meta_source.get("tanggal_berlaku"),
            }
            text = getattr(d, "page_content", str(d))
            if not meta.get("file_id"):
                # Fall back to a stable identifier derived from filename and index
                meta["file_id"] = f"{pkl.stem}__doc_{idx}"
            raw_items_all.append({"meta": meta, "text": text, "idx_in_file": idx})

    if not raw_items_all:
        log_info("No documents found. Exiting.")
        return

    # Budget-aware truncation (embeddings only)
    if ENFORCE_API_BUDGET and COUNT_EMBEDDINGS_IN_BUDGET:
        allowed = max(0, API_BUDGET.total - API_BUDGET.used)
        if allowed < len(raw_items_all):
            log_warn(f"Budget cap: truncating items from {len(raw_items_all)} to {allowed}")
            raw_items_all = raw_items_all[:allowed]

    batches = pack_batches(raw_items_all)
    log_info(f"Prepared {len(batches)} batch(es); documents planned: {len(raw_items_all)}")
    log_info(f"API budget: enforce={API_BUDGET.enforce}, total={API_BUDGET.total}, used={API_BUDGET.used}")
    log_info(f"Embedding RPM limit: {LLM_MAX_CALLS_PER_MIN} calls/min")
    log_info(f"Parallel workers: {INDEX_WORKERS}")
    if STAGGER_WORKER_SECONDS > 0:
        log_info(f"Worker ramp-up: start 1 worker, add 1 every {STAGGER_WORKER_SECONDS:.3f}s (up to {INDEX_WORKERS}).")
    else:
        log_info("Worker ramp-up disabled (all workers may start immediately).")

    total_done = 0
    total_emb_time = 0.0
    total_neo_time = 0.0
    overall_start = time.time()

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
                    # Gentle client backoff
                    sleep_for = random.uniform(5.0, 12.0)
                    log_warn(f"Rate-limit detected; sleeping {sleep_for:.1f}s and retrying this document once.")
                    time.sleep(sleep_for)
                    try:
                        ok, e_dur, n_dur = write_one_item(it, it["idx_in_file"])
                    except Exception as e2:
                        log_warn(f"Document failed after retry; skipping. Reason: {e2}")
                        continue
                else:
                    log_warn(f"Document failed; skipping. Reason: {e}")
                    continue

            if ok:
                done += 1
                emb_sum += e_dur
                neo_sum += n_dur

        return done, emb_sum, neo_sum

    # Staggered worker scheduling (same as your chunk indexer)
    with ThreadPoolExecutor(max_workers=INDEX_WORKERS) as executor:
        ramp_start = time.time()

        def allowed_workers_now() -> int:
            if STAGGER_WORKER_SECONDS <= 0:
                return INDEX_WORKERS
            elapsed = time.time() - ramp_start
            allowed = 1 + int(elapsed // STAGGER_WORKER_SECONDS)
            return min(INDEX_WORKERS, max(1, allowed))

        futures_set = set()
        meta_map: Dict[Any, Tuple[int, List[Dict[str, Any]], float]] = {}
        next_i = 0
        total_batches = len(batches)
        batches_completed = 0

        def time_until_next_ramp() -> float:
            if STAGGER_WORKER_SECONDS <= 0:
                return 1.0
            elapsed = time.time() - ramp_start
            steps_completed = int(elapsed // STAGGER_WORKER_SECONDS)
            next_step_time = (steps_completed + 1) * STAGGER_WORKER_SECONDS
            return max(0.0, next_step_time - elapsed)

        while True:
            target = allowed_workers_now()
            while next_i < total_batches and len(futures_set) < target:
                bi = next_i
                items_batch = batches[bi]
                submit_time = time.time()
                fut = executor.submit(process_batch, items_batch)
                futures_set.add(fut)
                meta_map[fut] = (bi, items_batch, submit_time)
                next_i += 1

            if not futures_set and next_i >= total_batches:
                break

            timeout = min(1.0, time_until_next_ramp())
            done, _ = wait(futures_set, timeout=timeout, return_when=FIRST_COMPLETED)

            for fut in list(done):
                bi, items_batch, submit_time = meta_map.pop(fut)
                futures_set.remove(fut)

                try:
                    done_count, emb_sum, neo_sum = fut.result()
                except Exception as e:
                    log_error(f"[Batch {bi+1}/{total_batches}] failed: {e}")
                    batches_completed += 1
                    continue

                wall = time.time() - submit_time
                total_done += done_count
                total_emb_time += emb_sum
                total_neo_time += neo_sum

                log_info(f"✓ [Batch {bi+1}/{total_batches}] "
                         f"docs={len(items_batch)} | done={done_count} | "
                         f"embed={emb_sum:.2f}s | neo4j={neo_sum:.2f}s | wall={wall:.2f}s")
                batches_completed += 1
                log_info(f"    - Batches completed: {batches_completed}/{total_batches}")

    total_wall = time.time() - overall_start
    sequential_est = total_emb_time + total_neo_time
    speedup = (sequential_est / total_wall) if total_wall > 0 else float('inf')

    log_info("Summary")
    log_info(f"- Documents planned: {total_docs_planned}")
    log_info(f"- Documents processed: {total_done}/{len(raw_items_all)}")
    log_info(f"- API budget used: {API_BUDGET.resource_usage.get('embed', 0)}/{API_BUDGET.total} (embeddings counted={COUNT_EMBEDDINGS_IN_BUDGET})")
    log_info(f"- Total Embedding time (sum): {total_emb_time:.2f}s")
    log_info(f"- Total Neo4j time (sum): {total_neo_time:.2f}s")
    log_info(f"- Total wall time: {total_wall:.2f}s")
    log_info(f"- Sequential time estimate (components sum): {sequential_est:.2f}s")
    log_info(f"- Speedup vs sequential: {speedup:.2f}×")

if __name__ == "__main__":
    run_doc_index_over_folder(
        LANGCHAIN_DIR,
        max_files=10000,
        max_docs_per_file=100000000000,
    )