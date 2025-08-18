import os, json, hashlib, time, threading, pickle, math
from collections import deque
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
from tenacity import retry, wait_exponential, stop_after_attempt, RetryError
from neo4j import GraphDatabase

# ----------------- Load .env from the parent directory -----------------
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

# ----------------- Config from env with sensible defaults -----------------
GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]

GEN_MODEL = os.getenv("GEN_MODEL", "models/gemini-2.5-flash-lite")
EMBED_MODEL = os.getenv("EMBED_MODEL", "models/text-embedding-004")

# Directory of LangChain per-document pickle files
DEFAULT_LANGCHAIN_DIR = (Path(__file__).resolve().parent / "../dataset/langchain-results/samples").resolve()
LANGCHAIN_DIR = DEFAULT_LANGCHAIN_DIR
# Neo4j
NEO4J_URI  = os.getenv("NEO4J_URI",  "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASS", "@ik4nkus")

# LLM rate limiter (calls per minute)
LLM_MAX_CALLS_PER_MIN = int(os.getenv("LLM_MAX_CALLS_PER_MIN", "13"))

# Parallelism
# Set to 13 as requested (still gated by the rate limiter)
INDEX_WORKERS = int(os.getenv("INDEX_WORKERS", "13"))

# API budget controls
def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "y", "on")

ENFORCE_API_BUDGET = _env_bool("ENFORCE_API_BUDGET", True)
# Set the default budget to 20 as requested
API_BUDGET_TOTAL = int(os.getenv("API_BUDGET_TOTAL", "150"))
COUNT_EMBEDDINGS_IN_BUDGET = _env_bool("COUNT_EMBEDDINGS_IN_BUDGET", False)

# Files to skip explicitly
SKIP_FILES = {"all_langchain_documents.pkl"}

# Embedding dimensions for text-embedding-004 (for reference)
EMBED_DIM = 768

# Prompt token control (heuristic)
PROMPT_TOKEN_LIMIT = int(os.getenv("PROMPT_TOKEN_LIMIT", "8000"))
PRACTICAL_MAX_ITEMS_PER_BATCH = int(os.getenv("PRACTICAL_MAX_ITEMS_PER_BATCH", "40"))

# ----------------- Initialize SDKs -----------------
genai.configure(api_key=GOOGLE_API_KEY)
try:
    gen_model = genai.GenerativeModel(GEN_MODEL)
    print(f"Using generation model: {GEN_MODEL}")
except Exception as e:
    raise RuntimeError(f"Failed to initialize GenerativeModel {GEN_MODEL}: {e}")

# ----------------- Indonesian vocab -----------------
LEGAL_ENTITY_TYPES = [
    "UU", "PASAL", "AYAT", "INSTANSI", "ORANG", "ISTILAH", "SANKSI", "NOMINAL", "TANGGAL"
]
LEGAL_PREDICATES = [
    "mendefinisikan",
    "mengubah",
    "mencabut",
    "mulai_berlaku",
    "mewajibkan",
    "melarang",
    "memberikan_sanksi",
    "berlaku_untuk",
    "termuat_dalam",
    "mendelegasikan_kepada",
    "berjumlah",
    "berdurasi"
]

# Schemas
TRIPLE_SCHEMA = {
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
              "type": {"type": "string"},
              "canonical_id": {"type": "string"}
            },
            "required": ["text"]
          },
          "predicate": {"type": "string"},
          "object": {
            "type": "object",
            "properties": {
              "text": {"type": "string"},
              "type": {"type": "string"},
              "canonical_id": {"type": "string"}
            },
            "required": ["text"]
          },
          "evidence": {
            "type": "object",
            "properties": {
              "quote": {"type": "string"},
              "char_start": {"type": "integer"},
              "char_end": {"type": "integer"},
              "article_ref": {"type": "string"}
            }
          },
          "confidence": {"type": "number"}
        },
        "required": ["subject", "predicate", "object"]
      }
    }
  },
  "required": ["triples"]
}

BATCH_SCHEMA = {
  "type": "object",
  "properties": {
    "results": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "chunk_id": {"type": "string"},
          "triples": TRIPLE_SCHEMA["properties"]["triples"]
        },
        "required": ["chunk_id", "triples"]
      }
    }
  },
  "required": ["results"]
}

SYSTEM_HINT = f"""
You extract knowledge graph triples from Indonesian legal text (Undang-Undang).

Output requirements:
- subject.type, object.type, and predicate MUST be Indonesian strings only.
- Allowed entity types: {", ".join(LEGAL_ENTITY_TYPES)}.
- Prefer predicates from: {", ".join(LEGAL_PREDICATES)} (snake_case where applicable).
- Use 'UU' for the Law, 'PASAL' for Article, 'AYAT' for Clause, 'INSTANSI' for institutions/agencies, 'ISTILAH' for defined terms.

Rules:
- Be precise and strictly grounded in the chunk; if unsupported, omit the triple.
- Include a short evidence.quote, and article_ref like 'Pasal X ayat (Y)' when present.
- Normalize predicate to a lowercase Indonesian single-verb phrase (use the list above when possible).
- Keep numbers/dates verbatim from the text.
"""

# ----------------- Rate Limiter (LLM) -----------------
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
            wait = 0.0
            if len(self.calls) >= self.max_calls:
                wait = self.period - (now - self.calls[0])
        if wait > 0:
            print(f"    [RateLimiter] Sleeping {wait:.2f}s to respect rate limit")
            time.sleep(wait)
        with self.lock:
            self.calls.append(time.time())

LLM_RATE_LIMITER = RateLimiter(max_calls=LLM_MAX_CALLS_PER_MIN, period_sec=60.0)

# ----------------- Global API Budget -----------------
class ApiBudget:
    def __init__(self, total: int, enforce: bool, count_embeddings: bool):
        self.total = total
        self.enforce = enforce
        self.count_embeddings = count_embeddings
        self.used = 0
        self.lock = threading.Lock()
        self.resource_usage = {"llm": 0, "embed": 0}

    def _should_count(self, kind: str) -> bool:
        if kind == "embed" and not self.count_embeddings:
            return False
        return True

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
def _slug(s: str) -> str:
    return " ".join((s or "").strip().split()).lower()

def normalize_entity_key(text: str, etype: str, uu_number: Optional[str] = None) -> str:
    if etype in ("UU", "PASAL", "AYAT") and uu_number:
        return f"{etype}::{_slug(uu_number)}::{_slug(text)}"
    return f"{etype}::{_slug(text)}"

def deterministic_triple_uid(
    subject_key: str, predicate: str, object_key: str, doc_id: Optional[str], span: Optional[Tuple[int,int]]
) -> str:
    h = hashlib.sha256()
    payload = "|".join([
        subject_key, (predicate or "").strip().lower(), object_key, doc_id or "",
        str(span[0]) if span else "", str(span[1]) if span else ""
    ])
    h.update(payload.encode("utf-8"))
    return h.hexdigest()

def estimate_tokens_for_text(text: str) -> int:
    # Rough heuristic: ~3.5 chars/token to be conservative
    return max(1, int(len(text) / 3.5))

# ----------------- Prompt builders -----------------
def build_single_prompt(meta: Dict[str, Any], chunk_text: str) -> str:
    return f"""
{SYSTEM_HINT}

Chunk metadata:
- document_id: {meta.get('document_id')}
- chunk_id: {meta.get('chunk_id')}
- uu_number: {meta.get('uu_number')}
- pages: {meta.get('pages')}

Text:
\"\"\"{chunk_text}\"\"\"
"""

def build_batch_prompt(items: List[Dict[str, Any]]) -> str:
    intro = f"{SYSTEM_HINT}\n\nYou will receive several chunks. For each chunk, return an object with 'chunk_id' and 'triples'."
    lines = []
    for it in items:
        m = it["meta"]
        lines.append(
            f"- chunk_id: {m.get('chunk_id')} | document_id: {m.get('document_id')} | uu_number: {m.get('uu_number')} | pages: {m.get('pages')}\n"
            f"TEXT:\n\"\"\"{it['text']}\"\"\""
        )
    return intro + "\n\nChunks:\n" + "\n\n".join(lines)

# ----------------- LLM Extraction -----------------
@retry(wait=wait_exponential(multiplier=1, min=2, max=30), stop=stop_after_attempt(5), reraise=True)
def extract_triples_from_chunk(chunk_text: str, meta: Dict[str, Any], prompt_override: Optional[str] = None) -> Tuple[List[Dict[str, Any]], float]:
    if not API_BUDGET.will_allow("llm", 1):
        raise RuntimeError("API budget would be exceeded by another LLM call; stopping extraction.")
    LLM_RATE_LIMITER.acquire()
    API_BUDGET.register("llm", 1)

    cfg = GenerationConfig(
        temperature=0,
        response_mime_type="application/json",
        response_schema=TRIPLE_SCHEMA,
    )
    prompt = prompt_override or build_single_prompt(meta, chunk_text)

    gemini_start = time.time()
    resp = gen_model.generate_content(prompt, generation_config=cfg)
    gemini_duration = time.time() - gemini_start

    raw = None
    try:
        raw = resp.text
        data = json.loads(raw)
    except Exception:
        try:
            raw = resp.candidates[0].content.parts[0].text
            data = json.loads(raw)
        except Exception as e:
            preview = raw[:200] if isinstance(raw, str) else "None"
            raise RuntimeError(f"Failed to parse model JSON: {e}; raw={preview}")

    uu_number = meta.get("uu_number")
    triples: List[Dict[str, Any]] = []
    for t in data.get("triples", []):
        subj = t["subject"]; obj = t["object"]; pred = (t["predicate"] or "").strip().lower()
        s_type = subj.get("type") or "ISTILAH"
        o_type = obj.get("type") or "ISTILAH"
        s_key = normalize_entity_key(subj["text"], s_type, uu_number)
        o_key = normalize_entity_key(obj["text"],  o_type,  uu_number)

        ev = t.get("evidence") or {}
        span = (ev["char_start"], ev["char_end"]) if ev.get("char_start") is not None and ev.get("char_end") is not None else None
        triple_uid = deterministic_triple_uid(s_key, pred, o_key, meta.get("document_id"), span)

        triples.append({
            "triple_uid": triple_uid,
            "subject": {"text": subj["text"], "type": s_type, "key": s_key},
            "predicate": pred,
            "object": {"text": obj["text"], "type": o_type, "key": o_key},
            "evidence": {
                "quote": ev.get("quote"),
                "char_start": ev.get("char_start"),
                "char_end": ev.get("char_end"),
                "article_ref": ev.get("article_ref"),
            },
            "confidence": float(t.get("confidence", 0.0)),
            "provenance": {
                "document_id": meta.get("document_id"),
                "chunk_id": meta.get("chunk_id"),
                "uu_number": uu_number,
                "pages": meta.get("pages"),
            }
        })
    return triples, gemini_duration

@retry(wait=wait_exponential(multiplier=1, min=2, max=30), stop=stop_after_attempt(5), reraise=True)
def extract_triples_from_chunks_batch(items: List[Dict[str, Any]], prompt: str) -> Tuple[Dict[str, List[Dict[str, Any]]], float]:
    if not API_BUDGET.will_allow("llm", 1):
        raise RuntimeError("API budget would be exceeded by another LLM call; stopping extraction.")
    LLM_RATE_LIMITER.acquire()
    API_BUDGET.register("llm", 1)

    cfg = GenerationConfig(
        temperature=0,
        response_mime_type="application/json",
        response_schema=BATCH_SCHEMA,
    )

    gemini_start = time.time()
    resp = gen_model.generate_content(prompt, generation_config=cfg)
    gemini_duration = time.time() - gemini_start

    raw = None
    try:
        raw = resp.text
        data = json.loads(raw)
    except Exception:
        try:
            raw = resp.candidates[0].content.parts[0].text
            data = json.loads(raw)
        except Exception as e:
            preview = raw[:200] if isinstance(raw, str) else "None"
            raise RuntimeError(f"Failed to parse batch JSON: {e}; raw={preview}")

    results_map: Dict[str, List[Dict[str, Any]]] = {}
    meta_map = {it["meta"]["chunk_id"]: it["meta"] for it in items}

    for res in data.get("results", []):
        cid = res.get("chunk_id")
        if not cid or cid not in meta_map:
            continue
        meta = meta_map[cid]
        uu_number = meta.get("uu_number")
        triples_for_chunk: List[Dict[str, Any]] = []
        for t in res.get("triples", []):
            subj = t["subject"]; obj = t["object"]; pred = (t["predicate"] or "").strip().lower()
            s_type = subj.get("type") or "ISTILAH"
            o_type = obj.get("type") or "ISTILAH"
            s_key = normalize_entity_key(subj["text"], s_type, uu_number)
            o_key = normalize_entity_key(obj["text"],  o_type,  uu_number)

            ev = t.get("evidence") or {}
            span = (ev["char_start"], ev["char_end"]) if ev.get("char_start") is not None and ev.get("char_end") is not None else None
            triple_uid = deterministic_triple_uid(s_key, pred, o_key, meta.get("document_id"), span)

            triples_for_chunk.append({
                "triple_uid": triple_uid,
                "subject": {"text": subj["text"], "type": s_type, "key": s_key},
                "predicate": pred,
                "object": {"text": obj["text"], "type": o_type, "key": o_key},
                "evidence": {
                    "quote": ev.get("quote"),
                    "char_start": ev.get("char_start"),
                    "char_end": ev.get("char_end"),
                    "article_ref": ev.get("article_ref"),
                },
                "confidence": float(t.get("confidence", 0.0)),
                "provenance": {
                    "document_id": meta.get("document_id"),
                    "chunk_id": meta.get("chunk_id"),
                    "uu_number": uu_number,
                    "pages": meta.get("pages"),
                }
            })
        results_map[cid] = triples_for_chunk

    return results_map, gemini_duration

# ----------------- Embeddings -----------------
def embed_text(text: str) -> Tuple[List[float], float]:
    if not API_BUDGET.will_allow("embed", 1):
        raise RuntimeError("API budget would be exceeded by another embedding call; stopping embedding.")
    start = time.time()
    res = genai.embed_content(model=EMBED_MODEL, content=text)
    API_BUDGET.register("embed", 1)
    dur = time.time() - start

    if isinstance(res, dict):
        emb = res.get("embedding")
        if isinstance(emb, dict) and "values" in emb:
            return emb["values"], dur
        if isinstance(emb, list):
            return emb, dur
    try:
        return res.embedding.values, dur  # type: ignore[attr-defined]
    except Exception:
        pass
    raise RuntimeError("Unexpected embedding response shape")

def node_embedding_text(name: str, etype: str) -> str:
    return f"{(name or '').strip()} | {etype}"

def triple_embedding_text(t: Dict[str, Any]) -> str:
    s = t["subject"]["text"]; p = t["predicate"]; o = t["object"]["text"]
    uu = t["provenance"].get("uu_number") or ""
    art = (t.get("evidence") or {}).get("article_ref") or ""
    return f"{s} [{p}] {o} | {uu} | {art}"

# ----------------- Neo4j Storage -----------------
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

_entity_emb_cache: dict[str, List[float]] = {}
_triple_emb_cache: dict[str, List[float]] = {}
_entity_emb_cache_lock = threading.Lock()
_triple_emb_cache_lock = threading.Lock()

def _get_entity_emb(name: str, etype: str, key: str) -> Tuple[List[float], float]:
    with _entity_emb_cache_lock:
        cached = _entity_emb_cache.get(key)
    if cached is not None:
        return cached, 0.0
    vec, dur = embed_text(node_embedding_text(name, etype))
    with _entity_emb_cache_lock:
        _entity_emb_cache[key] = vec
    return vec, dur

def _get_triple_emb(triple_uid: str, t: Dict[str, Any]) -> Tuple[List[float], float]:
    with _triple_emb_cache_lock:
        cached = _triple_emb_cache.get(triple_uid)
    if cached is not None:
        return cached, 0.0
    vec, dur = embed_text(triple_embedding_text(t))
    with _triple_emb_cache_lock:
        _triple_emb_cache[triple_uid] = vec
    return vec, dur

def upsert_triple(tx, t: Dict[str, Any], s_emb: List[float], o_emb: List[float], tr_emb: List[float]) -> float:
    s, o = t["subject"], t["object"]
    s_name, s_type, s_key = s["text"], s["type"], s["key"]
    o_name, o_type, o_key = o["text"], o["type"], o["key"]
    pred = t["predicate"]
    triple_uid = t["triple_uid"]

    prov = t["provenance"]
    ev = t.get("evidence") or {}
    confidence = float(t.get("confidence", 0.0))

    start = time.time()
    cypher = """
    MERGE (s:Entity {key:$s_key})
      ON CREATE SET s.name=$s_name, s.type=$s_type, s.createdAt=timestamp()
    SET s.embedding=$s_emb

    MERGE (o:Entity {key:$o_key})
      ON CREATE SET o.name=$o_name, o.type=$o_type, o.createdAt=timestamp()
    SET o.embedding=$o_emb

    MERGE (tr:Triple {triple_uid:$triple_uid})
      ON CREATE SET tr.createdAt=timestamp()
    SET tr.predicate=$pred,
        tr.embedding=$tr_emb,
        tr.document_id=$doc_id,
        tr.chunk_id=$chunk_id,
        tr.uu_number=$uu_number,
        tr.pages=$pages,
        tr.evidence_quote=$evidence_quote,
        tr.evidence_char_start=$evidence_char_start,
        tr.evidence_char_end=$evidence_char_end,
        tr.evidence_article_ref=$evidence_article_ref,
        tr.confidence=$confidence

    MERGE (tr)-[:SUBJECT]->(s)
    MERGE (tr)-[:OBJECT]->(o)

    MERGE (s)-[r:REL {triple_uid:$triple_uid}]->(o)
    SET r.predicate=$pred, r.chunk_id=$chunk_id, r.document_id=$doc_id
    """
    tx.run(
        cypher,
        s_key=s_key, s_name=s_name, s_type=s_type, s_emb=s_emb,
        o_key=o_key, o_name=o_name, o_type=o_type, o_emb=o_emb,
        triple_uid=triple_uid, pred=pred, tr_emb=tr_emb,
        doc_id=prov.get("document_id"),
        chunk_id=prov.get("chunk_id"),
        uu_number=prov.get("uu_number"),
        pages=prov.get("pages", []),
        evidence_quote=ev.get("quote"),
        evidence_char_start=ev.get("char_start"),
        evidence_char_end=ev.get("char_end"),
        evidence_article_ref=ev.get("article_ref"),
        confidence=confidence,
    )
    return time.time() - start

def write_triples_for_chunk(triples: List[Dict[str, Any]]) -> Tuple[int, float, float]:
    if not triples:
        return 0, 0.0, 0.0

    total_emb_dur = 0.0
    total_neo4j_dur = 0.0
    written = 0

    with driver.session() as session:
        for t in triples:
            s = t["subject"]; o = t["object"]
            s_emb, d1 = _get_entity_emb(s["text"], s["type"], s["key"])
            o_emb, d2 = _get_entity_emb(o["text"], o["type"], o["key"])
            tr_emb, d3 = _get_triple_emb(t["triple_uid"], t)
            total_emb_dur += (d1 + d2 + d3)

            neo4j_dur = session.execute_write(upsert_triple, t, s_emb, o_emb, tr_emb)
            total_neo4j_dur += neo4j_dur
            written += 1

    return written, total_emb_dur, total_neo4j_dur

# ----------------- Folder utilities -----------------
def list_pickles(dir_path: Path) -> List[Path]:
    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {dir_path}")
    return sorted([p for p in dir_path.glob("*.pkl") if p.name not in SKIP_FILES])

def load_chunks_from_file(pkl_path: Path) -> List[Any]:
    with open(pkl_path, "rb") as f:
        chunks = pickle.load(f)
    if not isinstance(chunks, list):
        raise ValueError(f"Unexpected pickle content in {pkl_path}")
    print(f"  - {pkl_path.name}: {len(chunks)} chunks")
    return chunks

# ----------------- Greedy and budget-aware batch builders -----------------
def build_greedy_batch(use_items: List[Dict[str, Any]], start_idx: int) -> Tuple[List[Dict[str, Any]], int, int]:
    items_batch: List[Dict[str, Any]] = []
    idx = start_idx
    est_tokens = 0

    while idx < len(use_items) and len(items_batch) < PRACTICAL_MAX_ITEMS_PER_BATCH:
        candidate = use_items[idx]
        tentative = items_batch + [candidate]
        prompt = build_batch_prompt(tentative) if len(tentative) > 1 else build_single_prompt(candidate["meta"], candidate["text"])
        tokens = estimate_tokens_for_text(prompt)
        if tokens <= PROMPT_TOKEN_LIMIT or len(items_batch) == 0:
            items_batch = tentative
            est_tokens = tokens
            idx += 1
        else:
            break

    return items_batch, idx, est_tokens

def pack_batches(use_items: List[Dict[str, Any]]) -> List[Tuple[List[Dict[str, Any]], int]]:
    batches: List[Tuple[List[Dict[str, Any]], int]] = []
    i = 0
    while i < len(use_items):
        items_batch, next_i, est_tokens = build_greedy_batch(use_items, i)
        if not items_batch:
            i += 1
            continue
        batches.append((items_batch, est_tokens))
        i = next_i
    return batches

def try_merge_batches(b1: Tuple[List[Dict[str, Any]], int], b2: Tuple[List[Dict[str, Any]], int]) -> Optional[Tuple[List[Dict[str, Any]], int]]:
    items1, _ = b1
    items2, _ = b2
    merged = items1 + items2
    if len(merged) > PRACTICAL_MAX_ITEMS_PER_BATCH:
        return None
    # recompute token estimate for merged
    merged_prompt = build_batch_prompt(merged) if len(merged) > 1 else build_single_prompt(merged[0]["meta"], merged[0]["text"])
    tokens = estimate_tokens_for_text(merged_prompt)
    if tokens <= PROMPT_TOKEN_LIMIT:
        return (merged, tokens)
    return None

def pack_batches_with_cap(use_items: List[Dict[str, Any]], max_batches_allowed: Optional[int]) -> Tuple[List[Tuple[List[Dict[str, Any]], int]], int]:
    """
    Build batches greedily and then enforce a hard cap on number of batches.
    If we exceed the cap, attempt to merge adjacent batches within token limits.
    If still over the cap, truncate extra batches (defer remaining chunks).

    Returns:
      (batches_to_run, deferred_chunks_count)
    """
    batches = pack_batches(use_items)
    if max_batches_allowed is None:
        return batches, 0

    if len(batches) <= max_batches_allowed:
        return batches, 0

    # Try to merge adjacent batches to reduce count
    changed = True
    while len(batches) > max_batches_allowed and changed:
        changed = False
        merged_list: List[Tuple[List[Dict[str, Any]], int]] = []
        i = 0
        while i < len(batches):
            if i < len(batches) - 1:
                merged = try_merge_batches(batches[i], batches[i+1])
                if merged:
                    merged_list.append(merged)
                    i += 2
                    changed = True
                    continue
            merged_list.append(batches[i])
            i += 1
        batches = merged_list

    # Enforce hard cap by truncation if needed
    if len(batches) > max_batches_allowed:
        to_run = batches[:max_batches_allowed]
        deferred = batches[max_batches_allowed:]
        deferred_chunks = sum(len(b[0]) for b in deferred)
        return to_run, deferred_chunks

    return batches, 0

# ----------------- Processing helper (with recursive split on failure) -----------------
def process_items(items: List[Dict[str, Any]]) -> Tuple[int, int, float, Dict[str, Dict[str, float]]]:
    """
    Processes a list of items (chunks) either as a single batch call or by splitting recursively on failure.

    Returns:
    - num_chunks_processed
    - num_triples_stored
    - gemini_duration_total
    - per_chunk_stats: {chunk_id: {"gemini": x, "embed": y, "neo4j": z, "triples": n}}
    """
    if not items:
        return 0, 0, 0.0, {}

    if ENFORCE_API_BUDGET and not API_BUDGET.will_allow("llm", 1):
        print("    ! Stopping: API budget for LLM calls would be exceeded.")
        return 0, 0, 0.0, {}

    prompt = build_batch_prompt(items) if len(items) > 1 else build_single_prompt(items[0]["meta"], items[0]["text"])

    try:
        if len(items) == 1:
            triples, gemini_dur = extract_triples_from_chunk(items[0]["text"], items[0]["meta"], prompt_override=prompt)
            written, emb_dur, neo4j_dur = write_triples_for_chunk(triples)
            cid = items[0]["meta"]["chunk_id"]
            per_chunk = {
                cid: {"gemini": gemini_dur, "embed": emb_dur, "neo4j": neo4j_dur, "triples": written}
            }
            return 1, written, gemini_dur, per_chunk

        results_map, gemini_dur = extract_triples_from_chunks_batch(items, prompt)

        per_chunk_gemini = gemini_dur / max(1, len(items))
        num_chunks_processed = 0
        num_triples_stored = 0
        per_chunk: Dict[str, Dict[str, float]] = {}

        for it in items:
            cid = it["meta"]["chunk_id"]
            triples = results_map.get(cid, [])
            written, emb_dur, neo4j_dur = write_triples_for_chunk(triples)
            per_chunk[cid] = {"gemini": per_chunk_gemini, "embed": emb_dur, "neo4j": neo4j_dur, "triples": written}
            num_chunks_processed += 1
            num_triples_stored += written

        return num_chunks_processed, num_triples_stored, gemini_dur, per_chunk

    except RetryError:
        if len(items) == 1:
            cid = items[0]["meta"]["chunk_id"]
            print(f"    ! Single-chunk extraction failed after retries (chunk_id={cid}). Skipping this chunk.")
            return 0, 0, 0.0, {}
        mid = len(items) // 2
        left = items[:mid]
        right = items[mid:]
        print(f"    ! Batch failed (size={len(items)}). Splitting into {len(left)} + {len(right)}.")
        l_chunks, l_triples, l_gemini, l_stats = process_items(left)
        r_chunks, r_triples, r_gemini, r_stats = process_items(right)
        chunks = l_chunks + r_chunks
        triples = l_triples + r_triples
        gemini_total = l_gemini + r_gemini
        l_stats.update(r_stats)
        return chunks, triples, gemini_total, l_stats
    except Exception as e:
        if len(items) == 1:
            cid = items[0]["meta"]["chunk_id"]
            print(f"    ! Unexpected error on single chunk (chunk_id={cid}): {e}. Skipping.")
            return 0, 0, 0.0, {}
        mid = len(items) // 2
        left = items[:mid]
        right = items[mid:]
        print(f"    ! Batch error (size={len(items)}): {e}. Splitting into {len(left)} + {len(right)}.")
        l_chunks, l_triples, l_gemini, l_stats = process_items(left)
        r_chunks, r_triples, r_gemini, r_stats = process_items(right)
        chunks = l_chunks + r_chunks
        triples = l_triples + r_triples
        gemini_total = l_gemini + r_gemini
        l_stats.update(r_stats)
        return chunks, triples, gemini_total, l_stats

# ----------------- Main pipeline (parallel, budget-capped batching) -----------------
def run_kg_pipeline_over_folder(
    dir_path: Path,
    max_files: Optional[int] = None,
    max_chunks_per_file: Optional[int] = None
):
    pkls = list_pickles(dir_path)
    if max_files is not None:
        pkls = pkls[:max_files]

    total_chunks_planned = 0
    for p in pkls:
        try:
            with open(p, "rb") as f:
                chunks = pickle.load(f)
            n = len(chunks)
            total_chunks_planned += n if max_chunks_per_file is None else min(n, max_chunks_per_file)
        except Exception:
            continue

    print(f"Found {len(pkls)} pickle files in {dir_path} (skipping: {', '.join(SKIP_FILES) or 'none'})")
    print(f"Total chunks planned: {total_chunks_planned}")
    if ENFORCE_API_BUDGET:
        allowed_calls_remaining = max(0, API_BUDGET.total - API_BUDGET.used)
        print(f"API budget: enforce={API_BUDGET.enforce}, total={API_BUDGET.total}, used={API_BUDGET.used}, allowed_calls_remaining={allowed_calls_remaining}")
    else:
        print(f"API budget: enforce={API_BUDGET.enforce} (unlimited LLM calls)")
    print(f"LLM rate limit: {LLM_MAX_CALLS_PER_MIN} calls/minute")
    print(f"Greedy batching target: <= {PRACTICAL_MAX_ITEMS_PER_BATCH} items, est tokens <= {PROMPT_TOKEN_LIMIT}")
    print(f"Parallel workers: {INDEX_WORKERS}")

    # Build 'raw_items_all' list across all files
    raw_items_all: List[Dict[str, Any]] = []
    for file_idx, pkl in enumerate(pkls, 1):
        print(f"[{file_idx}/{len(pkls)}] Scanning {pkl.name}")
        try:
            chunks = load_chunks_from_file(pkl)
        except Exception as e:
            print(f"    ! Failed to load {pkl.name}: {e}")
            continue

        for idx, ch in enumerate(chunks):
            if max_chunks_per_file is not None and idx >= max_chunks_per_file:
                break
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
            raw_items_all.append({"chunk_id": meta["chunk_id"], "text": text, "meta": meta})

    if not raw_items_all:
        print("No chunks found. Exiting.")
        return

    # Determine max number of batches allowed by budget (at most remaining LLM calls)
    max_batches_allowed = None
    if ENFORCE_API_BUDGET:
        max_batches_allowed = max(0, API_BUDGET.total - API_BUDGET.used)

    # Pack into batches and enforce a hard cap by budget
    all_batches_capped, deferred_chunks = pack_batches_with_cap(raw_items_all, max_batches_allowed)
    total_batches_planned = len(all_batches_capped)
    print(f"Packed {len(raw_items_all)} chunks into {total_batches_planned} batch(es) (budget-capped).")
    for i, (batch_items, est_tokens) in enumerate(all_batches_capped, 1):
        print(f"  • Batch {i}: chunks={len(batch_items)}, est_tokens≈{est_tokens}")
    if deferred_chunks > 0:
        print(f"Deferring {deferred_chunks} chunk(s) to future runs due to API budget cap of {max_batches_allowed} batch(es).")

    # Stats
    total_triples_stored = 0
    total_chunks_done = 0

    total_gemini_duration = 0.0
    total_embedding_duration = 0.0
    total_neo4j_duration = 0.0

    per_batch_component_sums: Dict[int, float] = {}
    per_batch_wall_times: Dict[int, float] = {}
    per_batch_sizes: Dict[int, int] = {}

    overall_start = time.time()

    def batch_desc(idx: int, size: int, tokens: int) -> str:
        return f"[Batch {idx+1}/{total_batches_planned}] size={size}, est_tokens≈{tokens}"

    # Budget-aware lazy scheduling (still guarded inside extract calls)
    next_idx = 0
    futures_set = set()
    futures_meta: Dict[Any, Tuple[int, int, List[Dict[str, Any]], float]] = {}

    def can_submit_more() -> bool:
        if not ENFORCE_API_BUDGET:
            return True
        return API_BUDGET.will_allow("llm", 1)

    def remaining_budget() -> int:
        if not ENFORCE_API_BUDGET:
            return 1_000_000_000
        return max(0, API_BUDGET.total - API_BUDGET.used)

    with ThreadPoolExecutor(max_workers=INDEX_WORKERS) as executor:
        initial_cap = INDEX_WORKERS
        if ENFORCE_API_BUDGET:
            initial_cap = min(initial_cap, remaining_budget(), total_batches_planned)

        while next_idx < total_batches_planned and len(futures_set) < initial_cap and can_submit_more():
            items_batch, est_tokens = all_batches_capped[next_idx]
            submit_time = time.time()
            fut = executor.submit(process_items, items_batch)
            futures_set.add(fut)
            futures_meta[fut] = (next_idx, est_tokens, items_batch, submit_time)
            next_idx += 1

        while futures_set:
            for fut in as_completed(futures_set, timeout=None):
                b_idx, est_tokens, items_batch, submit_time = futures_meta.pop(fut)
                futures_set.remove(fut)

                start_label = batch_desc(b_idx, len(items_batch), est_tokens)
                try:
                    chunks_processed, triples_stored, gemini_dur, per_chunk_stats = fut.result()
                except Exception as e:
                    print(f"    ! {start_label} failed with error: {e}")
                    # Try to submit the next batch if any remain and budget allows
                    while next_idx < total_batches_planned and len(futures_set) < INDEX_WORKERS and can_submit_more():
                        nb_items, nb_tokens = all_batches_capped[next_idx]
                        nb_submit_time = time.time()
                        nfut = executor.submit(process_items, nb_items)
                        futures_set.add(nfut)
                        futures_meta[nfut] = (next_idx, nb_tokens, nb_items, nb_submit_time)
                        next_idx += 1
                    break

                end_time = time.time()
                wall_time = end_time - submit_time

                embed_total = sum(stats['embed'] for stats in per_chunk_stats.values())
                neo4j_total = sum(stats['neo4j'] for stats in per_chunk_stats.values())
                comp_sum = gemini_dur + embed_total + neo4j_total

                total_chunks_done += chunks_processed
                total_triples_stored += triples_stored
                total_gemini_duration += gemini_dur
                total_embedding_duration += embed_total
                total_neo4j_duration += neo4j_total

                per_batch_component_sums[b_idx] = comp_sum
                per_batch_wall_times[b_idx] = wall_time
                per_batch_sizes[b_idx] = len(items_batch)

                if per_chunk_stats:
                    for it in items_batch:
                        cid = it["meta"]["chunk_id"]
                        stats = per_chunk_stats.get(cid)
                        if stats:
                            print(f"      · Chunk {cid}: Gemini={stats['gemini']:.2f}s | Embedding={stats['embed']:.2f}s | Neo4j={stats['neo4j']:.2f}s | Triples={int(stats['triples'])}")

                overhead = wall_time - comp_sum
                if chunks_processed == 0 and comp_sum == 0.0:
                    print(f"    • {start_label} skipped (budget exhausted). Batch wall time: {wall_time:.2f}s")
                else:
                    print(f"    ✓ {start_label}")
                    print(f"        - KG extraction (LLM): {gemini_dur:.2f}s")
                    print(f"        - Embedding total:     {embed_total:.2f}s")
                    print(f"        - Neo4j insert total:  {neo4j_total:.2f}s")
                    print(f"        - Component sum:       {comp_sum:.2f}s")
                    print(f"        - Batch wall time:     {wall_time:.2f}s")
                    print(f"        - Overhead (wall - components): {overhead:+.2f}s")
                    print(f"        - Chunks processed: {chunks_processed} | Triples stored: {triples_stored}")

                # Submit next batch if any remain and budget allows
                while next_idx < total_batches_planned:
                    if not can_submit_more():
                        break
                    if ENFORCE_API_BUDGET and len(futures_set) >= min(INDEX_WORKERS, remaining_budget()):
                        break
                    if len(futures_set) >= INDEX_WORKERS:
                        break
                    nb_items, nb_tokens = all_batches_capped[next_idx]
                    nb_submit_time = time.time()
                    nfut = executor.submit(process_items, nb_items)
                    futures_set.add(nfut)
                    futures_meta[nfut] = (next_idx, nb_tokens, nb_items, nb_submit_time)
                    next_idx += 1

                break  # re-enter as_completed with updated futures_set

    total_time_real = time.time() - overall_start
    total_llm_calls_used = API_BUDGET.resource_usage.get("llm", 0)

    sequential_estimate = sum(per_batch_component_sums.values())
    speedup = (sequential_estimate / total_time_real) if total_time_real > 0 else float('inf')

    print("\nSummary")
    print(f"- Batches planned (capped): {total_batches_planned}")
    print(f"- Chunks processed: {total_chunks_done}/{total_chunks_planned}")
    print(f"- Triples stored: {total_triples_stored}")
    print(f"- LLM calls used: {total_llm_calls_used}{' (budget enforced)' if ENFORCE_API_BUDGET else ''}")
    print(f"- Total Gemini time (sum of batch LLM times): {total_gemini_duration:.2f}s")
    print(f"- Total Embedding time (sum): {total_embedding_duration:.2f}s")
    print(f"- Total Neo4j time (sum): {total_neo4j_duration:.2f}s")
    print(f"- Total real wall time: {total_time_real:.2f}s")
    print(f"- Sequential time estimate (components sum): {sequential_estimate:.2f}s")
    print(f"- Speedup vs sequential: {speedup:.2f}× faster")
    if ENFORCE_API_BUDGET:
        print(f"- API used: {API_BUDGET.used}/{API_BUDGET.total} (LLM calls and {'embeddings' if COUNT_EMBEDDINGS_IN_BUDGET else 'no embeddings'} counted)")

if __name__ == "__main__":
    run_kg_pipeline_over_folder(
        LANGCHAIN_DIR,
        max_files=40,          # set to 10 as requested
        max_chunks_per_file=None
    )