# agentic_graph_rag.py
import os, time, json, math, pickle, re
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, Set

from dotenv import load_dotenv
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
from neo4j import GraphDatabase

# ----------------- Load .env (parent directory of this file) -----------------
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

# ----------------- Config -----------------
# Credentials and endpoints should stay in env for safety.
GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]
NEO4J_URI  = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASS", "password")

# Gemini models (can be overridden via env if desired)
GEN_MODEL = os.getenv("GEN_MODEL", "models/gemini-2.5-flash")
EMBED_MODEL = os.getenv("EMBED_MODEL", "models/text-embedding-004")

# Dataset folder for original chunk pickles (same as ingestion)
DEFAULT_LANGCHAIN_DIR = (Path(__file__).resolve().parent / "../dataset/samples/langchain-results/").resolve()
LANGCHAIN_DIR = Path(os.getenv("LANGCHAIN_DIR") or str(DEFAULT_LANGCHAIN_DIR))
SKIP_FILES = {"all_langchain_documents.pkl"}

# ----------------- Retrieval/agent parameters (hardcoded constants) -----------------
# Independent hop-depth and top-k per step (each "n" is independent as requested)
# Entity-centric path
ENTITY_MATCH_TOP_K = 8                 # top similar KG entities per extracted query entity
ENTITY_SUBGRAPH_HOPS = 1               # hop-depth for subgraph expansion from matched entities
ENTITY_SUBGRAPH_PER_HOP_LIMIT = 2000   # per-hop expansion limit
SUBGRAPH_TRIPLES_TOP_K = 30            # top triples selected from subgraph after triple-vs-triple similarity

# Triple-centric path
QUERY_TRIPLE_MATCH_TOP_K_PER = 10      # per query-triple, top similar KG triples

# Final context combination and reranking
MAX_TRIPLES_FINAL = 60                 # final number of triples after reranking
MAX_CHUNKS_FINAL = 40                  # final number of chunks after reranking
CHUNK_RERANK_CAND_LIMIT = 200          # cap chunk candidates before embedding/reranking to control cost

# Agent loop and output
ANSWER_MAX_TOKENS = 4096
MAX_ITERS = 3

# Language setting
OUTPUT_LANG = "id"  # retained for compatibility; we auto-detect based on query

# ----------------- Initialize SDKs -----------------
genai.configure(api_key=GOOGLE_API_KEY)
gen_model = genai.GenerativeModel(GEN_MODEL)

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

# ----------------- Logger -----------------
class FileLogger:
    def __init__(self, file_path: Path, also_console: bool = True):
        self.file_path = file_path
        self.also_console = also_console
        self._fh = open(file_path, "w", encoding="utf-8")

    def log(self, msg: str = ""):
        if not isinstance(msg, str):
            try:
                msg = json.dumps(msg, ensure_ascii=False, default=str)
            except Exception:
                msg = str(msg)
        self._fh.write(msg + "\n")
        self._fh.flush()
        if self.also_console:
            print(msg)

    def close(self):
        try:
            self._fh.flush()
            self._fh.close()
        except Exception:
            pass

_LOGGER: Optional[FileLogger] = None

def log(msg: str = ""):
    global _LOGGER
    if _LOGGER is not None:
        _LOGGER.log(msg)
    else:
        print(msg)

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

def estimate_tokens_for_text(text: str) -> int:
    # Quick heuristic: ~4 characters per token
    return max(1, int(len(text) / 4))

def embed_text(text: str) -> List[float]:
    res = genai.embed_content(model=EMBED_MODEL, content=text)
    if isinstance(res, dict):
        emb = res.get("embedding")
        if isinstance(emb, dict) and "values" in emb:
            return emb["values"]
        if isinstance(emb, list):
            return emb
    try:
        return res.embedding.values  # type: ignore[attr-defined]
    except Exception:
        pass
    raise RuntimeError("Unexpected embedding response shape for embeddings")

def cos_sim(a: List[float], b: List[float]) -> float:
    import math
    dot = sum(x*y for x,y in zip(a,b))
    na = math.sqrt(sum(x*x for x in a))
    nb = math.sqrt(sum(y*y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)

# ----------------- Simple language detection (ID vs EN) -----------------
def detect_user_language(text: str) -> str:
    """
    Lightweight heuristic to detect Indonesian ('id') vs English ('en')
    for the purpose of matching the user's query language in the final answer.
    """
    t = (text or "").lower()

    # Strong hints
    if re.search(r"\b(pasal|undang[- ]?undang|uu\s*\d|peraturan|menteri|ayat|bab|bagian|paragraf|ketentuan|sebagaimana|dimaksud)\b", t):
        return "id"
    if re.search(r"\b(article|act|law|regulation|minister|section|paragraph|chapter|pursuant|provided that)\b", t):
        return "en"

    # Token-based scoring
    id_tokens = {
        "yang","dan","atau","tidak","adalah","berdasarkan","sebagaimana","pada","dalam","dapat","harus","wajib",
        "pasal","undang","peraturan","menteri","ayat","bab","bagian","paragraf","ketentuan","pengundangan","apabila","jika"
    }
    en_tokens = {
        "the","and","or","not","is","based","as","provided","pursuant","in","may","must","shall",
        "article","act","law","regulation","minister","section","paragraph","chapter","whereas"
    }
    words = re.findall(r"[a-z]+", t)
    score_id = sum(1 for w in words if w in id_tokens)
    score_en = sum(1 for w in words if w in en_tokens)
    if score_id > score_en:
        return "id"
    if score_en > score_id:
        return "en"

    # Fallback: prefer English for safety
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
    resp = gen_model.generate_content(prompt, generation_config=cfg)
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
        log(f"[LLM JSON parse warning] No JSON content returned. Diagnostics: {info}. Error: {e}")
        try:
            return json.loads("{}")
        except Exception:
            return {}

def safe_generate_text(prompt: str, max_tokens: int, temperature: float = 0.2) -> str:
    cfg = GenerationConfig(temperature=temperature, max_output_tokens=max_tokens)
    resp = gen_model.generate_content(prompt, generation_config=cfg)
    text = extract_text_from_response(resp)
    if text is not None and text.strip():
        return text.strip()
    info = get_finish_info(resp)
    log(f"[LLM text warning] No text returned. Diagnostics: {info}")
    return f"(Model returned no text. finish_info={info})"

# ----------------- Agent 1: entity/predicate extraction -----------------
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
          "type": {"type": "string"}   # Indonesian type if known, else ""
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
    data = safe_generate_json(prompt, QUERY_SCHEMA, temp=0.0)
    if "entities" not in data: data["entities"] = []
    if "predicates" not in data: data["predicates"] = []
    log(f"[Agent 1] Output: entities={ [e.get('text') for e in data['entities']] }, predicates={ data['predicates'] }")
    return data

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
    out = safe_generate_json(prompt, QUERY_TRIPLES_SCHEMA, temp=0.0)
    triples = out.get("triples", []) if isinstance(out, dict) else []
    # sanitize minimal fields
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
    o = ((t.get("object") or {}).get("text") or "").strip()
    return f"{s} [{p}] {o}"

# ----------------- Neo4j vector search helpers -----------------
def _vector_query_nodes(index_name: str, q_emb: List[float], k: int) -> List[Dict[str, Any]]:
    """
    Query a specific vector index and return rows as dictionaries with common fields.
    """
    cypher = """
    CALL db.index.vector.queryNodes($index_name, $k, $q_emb) YIELD node AS n, score
    RETURN n, score
    ORDER BY score DESC
    LIMIT $k
    """
    with driver.session() as session:
        res = session.run(cypher, index_name=index_name, k=k, q_emb=q_emb)
        rows = []
        for r in res:
            n = r["n"]
            rows.append({
                "key": n.get("key"),
                "name": n.get("name"),
                "type": n.get("type"),
                "score": r["score"],
            })
        return rows

def search_similar_entities_by_embedding(q_emb: List[float], k: int) -> List[Dict[str, Any]]:
    """
    Search across entity-like indices and return top-k merged results by score.
    """
    candidates: List[Dict[str, Any]] = []
    try:
        candidates.extend(_vector_query_nodes("document_vec", q_emb, k))
    except Exception as e:
        log(f"[Warn] document_vec query failed: {e}")
    try:
        candidates.extend(_vector_query_nodes("content_vec", q_emb, k))
    except Exception as e:
        log(f"[Warn] content_vec query failed: {e}")
    try:
        candidates.extend(_vector_query_nodes("expression_vec", q_emb, k))
    except Exception as e:
        log(f"[Warn] expression_vec query failed: {e}")

    # Deduplicate by (key, type) while keeping the best score
    best: Dict[Tuple[Optional[str], Optional[str]], Dict[str, Any]] = {}
    for row in candidates:
        key = (row.get("key"), row.get("type"))
        if key not in best or (row.get("score", -1) > best[key].get("score", -1)):
            best[key] = row

    merged = list(best.values())
    merged.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    return merged[:k]

def search_similar_triples_by_embedding(q_emb: List[float], k: int) -> List[Dict[str, Any]]:
    """
    Query triple_vec; return dict rows with triple + subject/object if available.
    """
    cypher = """
    CALL db.index.vector.queryNodes('triple_vec', $k, $q_emb) YIELD node AS tr, score
    OPTIONAL MATCH (tr)-[:SUBJECT]->(s)
    OPTIONAL MATCH (tr)-[:OBJECT]->(o)
    RETURN tr, s, o, score
    ORDER BY score DESC
    LIMIT $k
    """
    with driver.session() as session:
        res = session.run(cypher, k=k, q_emb=q_emb)
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

# ----------------- Graph expansion (as in LexID) -----------------
def expand_from_entities(entity_keys: List[str], hops: int, per_hop_limit: int) -> List[Dict[str, Any]]:
    """
    Traverse edges that carry r.triple_uid, fetch Triple nodes and endpoints.
    """
    triples: Dict[str, Dict[str, Any]] = {}
    current_seeds = list(set(k for k in entity_keys if k))
    for _ in range(hops):
        if not current_seeds:
            break
        cypher = """
        UNWIND $keys AS k
        MATCH (e {key:k})-[r]->(nbr)
        WHERE r.triple_uid IS NOT NULL
        WITH DISTINCT r LIMIT $limit
        MATCH (tr:Triple {triple_uid:r.triple_uid})
        OPTIONAL MATCH (tr)-[:SUBJECT]->(s)
        OPTIONAL MATCH (tr)-[:OBJECT]->(o)
        RETURN tr, s, o
        """
        with driver.session() as session:
            res = session.run(cypher, keys=current_seeds, limit=per_hop_limit)
            next_seeds: Set[str] = set()
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
                if s and s.get("key"): next_seeds.add(s.get("key"))
                if o and o.get("key"): next_seeds.add(o.get("key"))
        current_seeds = list(next_seeds)
    return list(triples.values())

# ----------------- Chunk store -----------------
class ChunkStore:
    def __init__(self, root: Path, skip: Set[str]):
        self.root = root
        self.skip = skip
        self._index: Dict[Tuple[Any, Any], str] = {}
        self._loaded_files: Set[Path] = set()
        self._built = False

    def _build_index(self):
        if self._built:
            return
        start = now_ms()
        pkls = [p for p in self.root.glob("*.pkl") if p.name not in self.skip]
        for pkl in pkls:
            try:
                with open(pkl, "rb") as f:
                    chunks = pickle.load(f)
                for ch in chunks:
                    meta = getattr(ch, "metadata", {}) if hasattr(ch, "metadata") else {}
                    doc_id = meta.get("document_id")
                    chunk_id = meta.get("chunk_id")
                    text = getattr(ch, "page_content", None)
                    if doc_id is not None and chunk_id is not None and isinstance(text, str):
                        self._index[(doc_id, chunk_id)] = text
                self._loaded_files.add(pkl)
            except Exception:
                continue
        elapsed = dur_ms(start)
        log(f"[ChunkStore] Indexed {len(self._index)} chunks from {len(self._loaded_files)} files in {elapsed:.0f} ms")
        self._built = True

    def get_chunk(self, document_id: Any, chunk_id: Any) -> Optional[str]:
        if not self._built:
            self._build_index()
        # Direct lookup
        val = self._index.get((document_id, chunk_id))
        if val is not None:
            return val
        # Fallback: if chunk_id has split suffix like "::part1", try base chunk_id
        if isinstance(chunk_id, str) and "::" in chunk_id:
            base_id = chunk_id.split("::", 1)[0]
            val = self._index.get((document_id, base_id))
            if val is not None:
                return val
        return None

# ----------------- Scoring helpers -----------------
def mean_similarity_to_query_triples(triple_emb: Optional[List[float]], q_trip_embs: List[List[float]]) -> float:
    if not isinstance(triple_emb, list) or not q_trip_embs:
        return 0.0
    sims = [cos_sim(triple_emb, q) for q in q_trip_embs]
    if not sims:
        return 0.0
    return sum(sims) / len(sims)

# ----------------- New retrieval pipeline pieces -----------------
def entity_centric_retrieval(
    query_entities: List[Dict[str, Any]],
    q_trip_embs: List[List[float]],
    q_emb_fallback: Optional[List[float]] = None
) -> List[Dict[str, Any]]:
    """
    For each extracted entity:
      - embed entity text
      - find top-K similar KG entities
    Expand subgraphs from the union of matched entity keys (independent hop setting).
    Score subgraph triples by mean similarity to query triple embeddings,
    fallback to query embedding if no query triples.
    Return top SUBGRAPH_TRIPLES_TOP_K triples.
    """
    # 1) Match entities
    all_matched_keys: Set[str] = set()
    for e in query_entities:
        text = (e.get("text") or "").strip()
        if not text:
            continue
        try:
            e_emb = embed_text(text)
        except Exception as ex:
            log(f"[EntityRetrieval] Embedding failed for entity '{text}': {ex}")
            continue
        matches = search_similar_entities_by_embedding(e_emb, k=ENTITY_MATCH_TOP_K)
        keys = [m.get("key") for m in matches if m.get("key")]
        all_matched_keys.update(keys)
        log(f"[EntityRetrieval] '{text}' -> matched {len(keys)} KG keys (sample: {keys[:3]})")

    if not all_matched_keys:
        log("[EntityRetrieval] No KG entity matches found from query entities.")
        return []

    # 2) Expand from matched entities
    expanded_triples = expand_from_entities(list(all_matched_keys), hops=ENTITY_SUBGRAPH_HOPS, per_hop_limit=ENTITY_SUBGRAPH_PER_HOP_LIMIT)
    log(f"[EntityRetrieval] Expanded subgraph triples: {len(expanded_triples)}")

    if not expanded_triples:
        return []

    # 3) Score triples by mean similarity to query triples (fallback to query embedding)
    def score(t: Dict[str, Any]) -> float:
        emb = t.get("embedding")
        if q_trip_embs:
            return mean_similarity_to_query_triples(emb, q_trip_embs)
        # fallback to whole-query embedding
        if q_emb_fallback and isinstance(emb, list):
            return cos_sim(q_emb_fallback, emb)
        return 0.0

    ranked = sorted(expanded_triples, key=score, reverse=True)
    top = ranked[:SUBGRAPH_TRIPLES_TOP_K]
    log(f"[EntityRetrieval] Selected top-{len(top)} triples from subgraph")
    return top

def triple_centric_retrieval(query_triples: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[List[float]]]:
    """
    For each extracted query triple:
      - embed "s [p] o"
      - search top-K similar KG triples
    Return merged, deduped triples and the list of query triple embeddings used.
    """
    triples_map: Dict[str, Dict[str, Any]] = {}
    q_trip_embs: List[List[float]] = []
    for qt in query_triples:
        try:
            txt = query_triple_to_text(qt)
            emb = embed_text(txt)
            q_trip_embs.append(emb)
        except Exception as ex:
            log(f"[TripleRetrieval] Embedding failed for query triple '{qt}': {ex}")
            continue

        matches = search_similar_triples_by_embedding(emb, k=QUERY_TRIPLE_MATCH_TOP_K_PER)
        for m in matches:
            uid = m.get("triple_uid")
            if uid:
                if uid not in triples_map:
                    triples_map[uid] = m
                else:
                    # keep the better scored one (optional)
                    if m.get("score", 0.0) > triples_map[uid].get("score", 0.0):
                        triples_map[uid] = m

    merged = list(triples_map.values())
    log(f"[TripleRetrieval] Collected {len(merged)} unique KG triples across {len(q_trip_embs)} query triple(s)")
    return merged, q_trip_embs

def collect_chunks_for_triples(triples: List[Dict[str, Any]], chunk_store: ChunkStore) -> List[Tuple[Tuple[Any, Any], str, Dict[str, Any]]]:
    """
    From triples, gather unique (doc_id, chunk_id) and load texts.
    Returns list of (key_pair, text, triple) for each chunk instance.
    """
    seen_pairs: Set[Tuple[Any, Any]] = set()
    out: List[Tuple[Tuple[Any, Any], str, Dict[str, Any]]] = []
    for t in triples:
        doc_id = t.get("document_id")
        chunk_id = t.get("chunk_id")
        if doc_id is None or chunk_id is None:
            # fallback to evidence quote if no chunk available
            quote = t.get("evidence_quote")
            if quote:
                key = (t.get("triple_uid"), "quote")
                if key not in seen_pairs:
                    out.append((key, quote, t))
                    seen_pairs.add(key)
            continue
        key = (doc_id, chunk_id)
        if key in seen_pairs:
            continue
        text = chunk_store.get_chunk(doc_id, chunk_id)
        if text:
            out.append((key, text, t))
            seen_pairs.add(key)
        else:
            # fallback to evidence quote
            quote = t.get("evidence_quote")
            if quote:
                key2 = (t.get("triple_uid"), "quote")
                if key2 not in seen_pairs:
                    out.append((key2, quote, t))
                    seen_pairs.add(key2)
    return out

def rerank_chunks_by_query(
    chunk_records: List[Tuple[Tuple[Any, Any], str, Dict[str, Any]]],
    q_emb_query: List[float],
    top_k: int
) -> List[Tuple[Tuple[Any, Any], str, Dict[str, Any], float]]:
    """
    Embed chunk texts and score similarity to whole user query.
    Returns list of records augmented with score, sorted desc.
    """
    # Optionally cap to limit embedding cost
    cand = chunk_records[:CHUNK_RERANK_CAND_LIMIT]
    scored: List[Tuple[Tuple[Any, Any], str, Dict[str, Any], float]] = []
    for key, text, t in cand:
        try:
            emb = embed_text(text)
            s = cos_sim(q_emb_query, emb)
            scored.append((key, text, t, s))
        except Exception as ex:
            log(f"[ChunkRerank] Embedding failed for chunk {key}: {ex}")
            continue
    scored.sort(key=lambda x: x[3], reverse=True)
    return scored[:top_k]

def rerank_triples_by_query_triples(
    triples: List[Dict[str, Any]],
    q_trip_embs: List[List[float]],
    q_emb_fallback: Optional[List[float]],
    top_k: int
) -> List[Dict[str, Any]]:
    """
    Sort triples by mean similarity to all query triple embeddings.
    If no query triple embeddings, fallback to whole-query embedding similarity.
    """
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
    """
    Create a readable context text with:
      - Triple summary (top up to 50)
      - Selected chunk texts
    Returns (context_text, summary_text, list_of_chunk_records_dict)
    """
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
        lines.append(f"[Chunk {idx}] doc={doc_id} chunk={chunk_id} | {uu} | score={score:.3f}\n{text}")
    context = "\n".join(lines)

    chunk_records = [{"key": key, "text": text, "triple": t, "score": score} for key, text, t, score in chunks_ranked]
    return context, summary_text, chunk_records

# ----------------- Agent 2 (Intermediate answer) -----------------
def agent2_answer(query_original: str, context: str, guidance: Optional[str], output_lang: str = "id") -> str:
    instructions = (
        "Answer concisely and accurately based strictly on the provided context. "
        "Cite UU/Article references when they are clear. "
        "Respond in the same language as the user's question."
    )

    guidance_text = guidance.strip() if isinstance(guidance, str) and guidance.strip() else "(No additional guidance from the Judge in the previous iteration.)"

    prompt = f"""
You are Agent 2 (Answerer). Task: provide an answer based on the context only.

Core instructions:
{instructions}

Additional guidance from Judge (if any):
\"\"\"{guidance_text}\"\"\"

Original user question:
\"\"\"{query_original}\"\"\"

Context:
\"\"\"{context}\"\"\"
"""
    est_tokens = estimate_tokens_for_text(prompt)
    log("\n[Agent 2] Prompt:")
    log(prompt)
    log(f"[Agent 2] Prompt size: {len(prompt)} chars, est_tokens≈{est_tokens}")

    answer = safe_generate_text(prompt, max_tokens=ANSWER_MAX_TOKENS, temperature=0.2)
    log("[Agent 2] Intermediate answer:")
    log(answer)
    return answer

# ----------------- Agent 3 (Judge) -----------------
JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "is_sufficient": {"type": "boolean"},
        "rationale": {"type": "string"},
        "issues_found": {
            "type": "array",
            "items": {"type": "string"}
        },
        "rewritten_query": {"type": "string"},
        "guidance_next": {"type": "string"}
    },
    "required": ["is_sufficient", "rationale"]
}

def agent3_judge(query_original: str, intermediate_answer: str, context_summary: str, output_lang: str = "id") -> Dict[str, Any]:
    prompt = f"""
You are Agent 3 (Judge). Task: evaluate whether the intermediate answer is sufficient and grounded in the context.

Sufficiency criteria:
- Clearly and correctly addresses the core of the question.
- Grounded in the provided context (triple summary).
- Cites UU/Article references when conclusive.
- Does not fabricate information outside the context.

If NOT sufficient:
- Provide "rewritten_query": a sharper, more targeted version of the user question.
- Provide "guidance_next": specific operational guidance for the next Agent 2 iteration
  (e.g., keywords to look for in the context, main theme, which UU/articles to prioritize, or expected answer format).

Original user question:
\"\"\"{query_original}\"\"\"

Context summary (triples):
\"\"\"{context_summary}\"\"\"

Intermediate answer:
\"\"\"{intermediate_answer}\"\"\"
"""
    est_tokens = estimate_tokens_for_text(prompt)
    log("\n[Agent 3] Prompt:")
    log(prompt)
    log(f"[Agent 3] Prompt size: {len(prompt)} chars, est_tokens≈{est_tokens}")

    out = safe_generate_json(prompt, JUDGE_SCHEMA, temp=0.0)
    out["is_sufficient"] = bool(out.get("is_sufficient", False))
    out["rationale"] = out.get("rationale", "")
    if "issues_found" not in out or not isinstance(out["issues_found"], list):
        out["issues_found"] = []
    out["rewritten_query"] = out.get("rewritten_query", "").strip()
    out["guidance_next"] = out.get("guidance_next", "").strip()

    log("[Agent 3] Judgment output:")
    log(json.dumps(out, ensure_ascii=False, indent=2))
    return out

# ----------------- Iterative GraphRAG with Agents (updated retrieval) -----------------
def agentic_graph_rag(query_original: str) -> Dict[str, Any]:
    global _LOGGER
    ts_name = make_timestamp_name()
    log_file = Path.cwd() / f"{ts_name}.txt"
    _LOGGER = FileLogger(log_file, also_console=True)

    try:
        log("=== Agentic GraphRAG run started ===")
        log(f"Log file: {log_file}")
        log(f"Original Query: {query_original}")
        log(f"Parameters: ENTITY_MATCH_TOP_K={ENTITY_MATCH_TOP_K}, ENTITY_SUBGRAPH_HOPS={ENTITY_SUBGRAPH_HOPS}, "
            f"ENTITY_SUBGRAPH_PER_HOP_LIMIT={ENTITY_SUBGRAPH_PER_HOP_LIMIT}, SUBGRAPH_TRIPLES_TOP_K={SUBGRAPH_TRIPLES_TOP_K}, "
            f"QUERY_TRIPLE_MATCH_TOP_K_PER={QUERY_TRIPLE_MATCH_TOP_K_PER}, MAX_TRIPLES_FINAL={MAX_TRIPLES_FINAL}, "
            f"MAX_CHUNKS_FINAL={MAX_CHUNKS_FINAL}, CHUNK_RERANK_CAND_LIMIT={CHUNK_RERANK_CAND_LIMIT}, "
            f"ANSWER_MAX_TOKENS={ANSWER_MAX_TOKENS}, MAX_ITERS={MAX_ITERS}, OUTPUT_LANG={OUTPUT_LANG}")

        chunk_store = ChunkStore(LANGCHAIN_DIR, set(SKIP_FILES))

        final_answer = None
        judge_reports: List[Dict[str, Any]] = []
        guidance_prev = None
        query_for_iter = query_original

        # Detect user language once, reuse per iteration
        user_lang = detect_user_language(query_original)
        log(f"[Language] Detected user language: {user_lang}")

        for it in range(1, MAX_ITERS + 1):
            log(f"\n--- Iteration {it}/{MAX_ITERS} ---")

            # Step 0: Embed whole user query (for chunk reranking and fallback)
            t0 = now_ms()
            q_emb_query = embed_text(query_for_iter)
            t_embed = dur_ms(t0)
            log(f"[Step 0] Whole-query embedding in {t_embed:.0f} ms")
            log(f"[Step 0] Query used this iteration: {query_for_iter}")

            # Step 1: Agent 1 – extract entities/predicates from query_for_iter
            t1 = now_ms()
            extraction = agent1_extract_entities_predicates(query_for_iter)
            ents = extraction.get("entities", [])
            preds = extraction.get("predicates", [])
            t_extract = dur_ms(t1)
            log(f"[Step 1] Entity/Predicate extraction done in {t_extract:.0f} ms")

            # Step 1b: Agent 1b – extract triples from query_for_iter
            t1b = now_ms()
            query_triples = agent1b_extract_query_triples(query_for_iter)
            t_extract_tr = dur_ms(t1b)
            log(f"[Step 1b] Query triple extraction done in {t_extract_tr:.0f} ms")

            # Step 2: Triple-centric retrieval (per query triple)
            t2 = now_ms()
            ctx2_triples, q_trip_embs = triple_centric_retrieval(query_triples)
            t_triple = dur_ms(t2)
            log(f"[Step 2] Triple-centric retrieval in {t_triple:.0f} ms; ctx2_triples={len(ctx2_triples)}, q_trip_embs={len(q_trip_embs)}")

            # Step 3: Entity-centric retrieval (entity→KG entities→expand→triples scored vs query triples)
            t3 = now_ms()
            ctx1_triples = entity_centric_retrieval(ents, q_trip_embs=q_trip_embs, q_emb_fallback=q_emb_query)
            t_entity = dur_ms(t3)
            log(f"[Step 3] Entity-centric retrieval in {t_entity:.0f} ms; ctx1_triples={len(ctx1_triples)}")

            # Step 4: Merge contexts, dedupe triples
            triple_map: Dict[str, Dict[str, Any]] = {}
            for t in ctx1_triples + ctx2_triples:
                uid = t.get("triple_uid")
                if uid:
                    # keep best by aggregated score proxy if available (use existing score if present)
                    prev = triple_map.get(uid)
                    if prev is None or (t.get("score", 0.0) > prev.get("score", 0.0)):
                        triple_map[uid] = t
            merged_triples = list(triple_map.values())
            log(f"[Step 4] Merged triples from contexts: {len(merged_triples)}")

            # Step 5: Gather chunks from merged triples, dedupe, rerank by whole-query similarity
            t5 = now_ms()
            chunk_records = collect_chunks_for_triples(merged_triples, chunk_store)
            log(f"[Step 5] Collected {len(chunk_records)} chunk candidates (pre-rerank)")
            chunks_ranked = rerank_chunks_by_query(chunk_records, q_emb_query, top_k=MAX_CHUNKS_FINAL)
            t_chunks = dur_ms(t5)
            log(f"[Step 5] Chunk rerank done in {t_chunks:.0f} ms; selected {len(chunks_ranked)}")

            # Step 6: Rerank triples by mean similarity to query triples (fallback to query embedding)
            t6 = now_ms()
            triples_ranked = rerank_triples_by_query_triples(merged_triples, q_trip_embs=q_trip_embs, q_emb_fallback=q_emb_query, top_k=MAX_TRIPLES_FINAL)
            t_rerank = dur_ms(t6)
            log(f"[Step 6] Triple rerank done in {t_rerank:.0f} ms; selected {len(triples_ranked)}")

            # Build combined context text (summary + chunks)
            context_text, context_summary, _ = build_combined_context_text(triples_ranked, chunks_ranked)
            log("\n[Context summary for this iteration]:")
            log(context_summary)

            # Step 7: Agent 2 – Answer
            t7 = now_ms()
            intermediate_answer = agent2_answer(query_original, context_text, guidance_prev, output_lang=user_lang)
            t_answer = dur_ms(t7)
            log(f"[Step 7] Intermediate answer generated in {t_answer:.0f} ms")

            # Step 8: Agent 3 – Judge
            t8 = now_ms()
            judge = agent3_judge(query_original, intermediate_answer, context_summary, output_lang=user_lang)
            t_judge = dur_ms(t8)
            log(f"[Step 8] Judgment done in {t_judge:.0f} ms")

            judge_reports.append(judge)

            if judge.get("is_sufficient", False):
                final_answer = intermediate_answer
                log("\n[Judge] Verdict: Sufficient. Using intermediate answer as final.")
                break
            else:
                log("\n[Judge] Verdict: Not sufficient.")
                rq = judge.get("rewritten_query", "").strip()
                gn = judge.get("guidance_next", "").strip()
                if rq:
                    query_for_iter = rq
                    log(f"[Judge] Rewritten query for next iteration: {rq}")
                else:
                    query_for_iter = query_original
                    log("[Judge] No rewritten query provided; will reuse original query.")
                if gn:
                    guidance_prev = gn
                    log(f"[Judge] Guidance for next iteration:\n{gn}")
                else:
                    guidance_prev = None
                    log("[Judge] No guidance provided for next iteration.")

        if final_answer is None:
            final_answer = intermediate_answer
            log("\n[Loop] Reached max iterations. Returning the last intermediate answer as final.")

        # Summary
        log("\n=== Agentic GraphRAG summary ===")
        log(f"- Iterations used: {len(judge_reports)}")
        if judge_reports:
            last = judge_reports[-1]
            log(f"- Last judge verdict: sufficient={last.get('is_sufficient')}")
        log("\n=== Final Answer ===")
        log(final_answer)
        log(f"\nLogs saved to: {log_file}")

        return {
            "final_answer": final_answer,
            "log_file": str(log_file),
            "iterations": len(judge_reports),
            "judge_reports": judge_reports
        }
    finally:
        if _LOGGER is not None:
            _LOGGER.close()

# ----------------- Main -----------------
if __name__ == "__main__":
    try:
        user_query = input("Enter your query: ").strip()
        if not user_query:
            print("Empty query. Exiting.")
        else:
            agentic_graph_rag(user_query)
    finally:
        try:
            driver.close()
        except Exception:
            pass