import os, time, json, math, pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, Set

from dotenv import load_dotenv
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
from neo4j import GraphDatabase

# ----------------- Load .env (parent directory of this file) -----------------
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

# ----------------- Config -----------------
GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]

GEN_MODEL = os.getenv("GEN_MODEL", "models/gemini-2.5-flash")
EMBED_MODEL = os.getenv("EMBED_MODEL", "models/text-embedding-004")

# Neo4j
NEO4J_URI  = os.getenv("NEO4J_URI",  "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASS", "password")

# Dataset folder for original chunk pickles (same as ingestion)
DEFAULT_LANGCHAIN_DIR = (Path(__file__).resolve().parent / "../dataset/langchain-results/samples").resolve()
LANGCHAIN_DIR = Path(os.getenv("LANGCHAIN_DIR") or str(DEFAULT_LANGCHAIN_DIR))
SKIP_FILES = {"all_langchain_documents.pkl"}

# Retrieval parameters (edit or set via env)
TOP_K_ENTITIES = int(os.getenv("TOP_K_ENTITIES", "10"))
TOP_K_TRIPLES = int(os.getenv("TOP_K_TRIPLES", "15"))

N_HOPS = int(os.getenv("N_HOPS", "1"))                  # 1 or 2 is typical
PER_HOP_LIMIT = int(os.getenv("PER_HOP_LIMIT", "2000")) # guardrail per hop
MAX_EDGES = int(os.getenv("MAX_EDGES", "200"))          # cap on triples/edges after rerank
MAX_CHUNKS = int(os.getenv("MAX_CHUNKS", "40"))         # cap on chunks included in context

OUTPUT_LANG = os.getenv("OUTPUT_LANG", "id")            # "id" or "en"
ANSWER_MAX_TOKENS = int(os.getenv("ANSWER_MAX_TOKENS", "4096"))  # default 4096 as requested

# ----------------- Initialize SDKs -----------------
genai.configure(api_key=GOOGLE_API_KEY)
gen_model = genai.GenerativeModel(GEN_MODEL)

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

# ----------------- Simple logger (file + console) -----------------
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
    # Example: 20250811-010512-123 (local time, ms)
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
    # Try the quick accessor
    try:
        text = resp.text
        if isinstance(text, str) and text.strip():
            return text.strip()
    except Exception:
        pass
    # Fall back to iterating candidates/parts
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

def safe_generate_json(prompt: str, schema: Dict[str, Any]) -> Dict[str, Any]:
    cfg = GenerationConfig(
        temperature=0,
        response_mime_type="application/json",
        response_schema=schema,
    )
    resp = gen_model.generate_content(prompt, generation_config=cfg)
    # Try resp.text
    try:
        if isinstance(resp.text, str) and resp.text.strip():
            return json.loads(resp.text)
    except Exception:
        pass
    # Try candidates
    try:
        raw = resp.candidates[0].content.parts[0].text
        return json.loads(raw)
    except Exception as e:
        info = get_finish_info(resp)
        log(f"[LLM JSON parse warning] No JSON content returned. Diagnostics: {info}. Error: {e}")
        if "results" in schema:
            return {"results": []}
        if "triples" in schema:
            return {"triples": []}
        return {"entities": [], "predicates": []}

def safe_generate_text(prompt: str, max_tokens: int, temperature: float = 0.2) -> str:
    cfg = GenerationConfig(temperature=temperature, max_output_tokens=max_tokens)
    # Try non-streaming first
    resp = gen_model.generate_content(prompt, generation_config=cfg)
    text = extract_text_from_response(resp)
    if text is not None and text.strip():
        return text.strip()
    # If still nothing, print diagnostics and return a helpful message
    info = get_finish_info(resp)
    log(f"[LLM text warning] No text returned. Diagnostics: {info}")
    return f"(Model returned no text. finish_info={info})"

# ----------------- Query parsing (LLM) -----------------
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

def extract_entities_predicates_from_query(query: str) -> Dict[str, Any]:
    prompt = f"""
You will analyze a user query about Indonesian legal matters.

Goal:
- Extract any entities and legal predicates referenced or implied by the query.

Output:
- JSON with arrays: entities (objects with text and optional Indonesian type) and predicates (Indonesian strings, snake_case where possible).

Guidelines:
- Entity types, when present, MUST be Indonesian from this list: {", ".join(LEGAL_ENTITY_TYPES)}.
- Predicates should be Indonesian strings, preferably one of: {", ".join(LEGAL_PREDICATES)}.
- Be concise; if unsure about type, leave it empty.

User query:
\"\"\"{query}\"\"\"
"""
    data = safe_generate_json(prompt, QUERY_SCHEMA)
    if "entities" not in data: data["entities"] = []
    if "predicates" not in data: data["predicates"] = []
    return data

# ----------------- Vector search in Neo4j -----------------
def search_similar_entities(q_emb: List[float], k: int) -> List[Dict[str, Any]]:
    cypher = """
    CALL db.index.vector.queryNodes('entity_vec', $k, $q_emb) YIELD node AS e, score
    RETURN e, score
    ORDER BY score DESC
    LIMIT $k
    """
    with driver.session() as session:
        res = session.run(cypher, k=k, q_emb=q_emb)
        rows = []
        for r in res:
            e = r["e"]
            rows.append({
                "key": e.get("key"),
                "name": e.get("name"),
                "type": e.get("type"),
                "score": r["score"],
            })
        return rows

def search_similar_triples(q_emb: List[float], k: int) -> List[Dict[str, Any]]:
    cypher = """
    CALL db.index.vector.queryNodes('triple_vec', $k, $q_emb) YIELD node AS tr, score
    OPTIONAL MATCH (tr)-[:SUBJECT]->(s:Entity)
    OPTIONAL MATCH (tr)-[:OBJECT]->(o:Entity)
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

# ----------------- Graph expansion -----------------
def expand_from_entities(entity_keys: List[str], hops: int, per_hop_limit: int) -> List[Dict[str, Any]]:
    triples: Dict[str, Dict[str, Any]] = {}
    current_seeds = list(set(k for k in entity_keys if k))
    for _ in range(hops):
        if not current_seeds:
            break
        cypher = """
        UNWIND $keys AS k
        MATCH (e:Entity {key:k})-[r:REL]->(nbr:Entity)
        WITH DISTINCT r LIMIT $limit
        MATCH (tr:Triple {triple_uid:r.triple_uid})
        OPTIONAL MATCH (tr)-[:SUBJECT]->(s:Entity)
        OPTIONAL MATCH (tr)-[:OBJECT]->(o:Entity)
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
                if s and s.get("key"):
                    next_seeds.add(s.get("key"))
                if o and o.get("key"):
                    next_seeds.add(o.get("key"))
        current_seeds = list(next_seeds)
    return list(triples.values())

def expand_from_triples(triple_ids: List[str], hops: int, per_hop_limit: int) -> List[Dict[str, Any]]:
    triples: Dict[str, Dict[str, Any]] = {}
    with driver.session() as session:
        res = session.run("""
        UNWIND $uids AS uid
        MATCH (tr:Triple {triple_uid:uid})-[:SUBJECT]->(s:Entity)
        MATCH (tr)-[:OBJECT]->(o:Entity)
        RETURN DISTINCT s.key AS s_key, o.key AS o_key
        """, uids=triple_ids)
        seeds = set()
        for r in res:
            if r["s_key"]: seeds.add(r["s_key"])
            if r["o_key"]: seeds.add(r["o_key"])
    expanded = expand_from_entities(list(seeds), hops, per_hop_limit)
    for t in expanded:
        triples[t["triple_uid"]] = t
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
                # Skip problematic file
                continue
        elapsed = dur_ms(start)
        log(f"[ChunkStore] Indexed {len(self._index)} chunks from {len(self._loaded_files)} files in {elapsed:.0f} ms")
        self._built = True

    def get_chunk(self, document_id: Any, chunk_id: Any) -> Optional[str]:
        if not self._built:
            self._build_index()
        return self._index.get((document_id, chunk_id))

# ----------------- Context builder -----------------
def build_context_from_triples(triples: List[Dict[str, Any]], chunk_store: ChunkStore, max_chunks: int, q_emb: List[float]) -> Tuple[str, List[Dict[str, Any]]]:
    # Rerank triples by similarity using stored triple embeddings
    def triple_score(t: Dict[str, Any]) -> float:
        emb = t.get("embedding")
        return cos_sim(q_emb, emb) if isinstance(emb, list) else 0.0
    ranked = sorted(triples, key=triple_score, reverse=True)

    # Collect chunks by (doc_id, chunk_id), preserving rank
    seen_pairs: Set[Tuple[Any, Any]] = set()
    selected_chunks: List[Tuple[Tuple[Any, Any], str, Dict[str, Any]]] = []
    for t in ranked:
        doc_id = t.get("document_id")
        chunk_id = t.get("chunk_id")
        if doc_id is None or chunk_id is None:
            quote = t.get("evidence_quote")
            if quote:
                key = (t.get("triple_uid"), "quote")
                if key not in seen_pairs:
                    selected_chunks.append((key, quote, t))
                    seen_pairs.add(key)
            continue
        key = (doc_id, chunk_id)
        if key in seen_pairs:
            continue
        text = chunk_store.get_chunk(doc_id, chunk_id)
        if text:
            selected_chunks.append((key, text, t))
            seen_pairs.add(key)
        else:
            quote = t.get("evidence_quote")
            if quote:
                key2 = (t.get("triple_uid"), "quote")
                if key2 not in seen_pairs:
                    selected_chunks.append((key2, quote, t))
                    seen_pairs.add(key2)
        if len(selected_chunks) >= max_chunks:
            break

    # Compose textual context
    lines = []
    lines.append("Ringkasan triple yang relevan:")
    for t in ranked[:min(50, len(ranked))]:
        s = t.get("subject"); p = t.get("predicate"); o = t.get("object")
        uu = t.get("uu_number") or ""
        art = t.get("evidence_article_ref") or ""
        quote = (t.get("evidence_quote") or "")[:300]
        lines.append(f"- {s} [{p}] {o} | {uu} | {art} | bukti: {quote}")

    lines.append("\nPotongan teks terkait (chunk):")
    for idx, (key, text, t) in enumerate(selected_chunks, start=1):
        doc_id = t.get("document_id")
        chunk_id = t.get("chunk_id")
        uu = t.get("uu_number") or ""
        lines.append(f"[Chunk {idx}] doc={doc_id} chunk={chunk_id} | {uu}\n{text}")

    context = "\n".join(lines)
    chunk_records = [{"key": key, "text": text, "triple": t} for key, text, t in selected_chunks]
    return context, chunk_records

# ----------------- Final answer via LLM (with full prompt print) -----------------
def answer_with_context(query: str, context: str, output_lang: str = "id") -> str:
    instr_id = "Jawab pertanyaan pengguna dalam Bahasa Indonesia secara ringkas, akurat, dan berdasarkan konteks. Cantumkan rujukan UU/Pasal bila jelas."
    instr_en = "Answer concisely and accurately based strictly on the provided context. Cite UU/Article references when clear."
    instructions = instr_id if output_lang.lower().startswith("id") else instr_en

    prompt = f"""
{instructions}

User query:
\"\"\"{query}\"\"\"

Context:
\"\"\"{context}\"\"\"
"""

    est_tokens = estimate_tokens_for_text(prompt)
    log("\n=== Full prompt sent to LLM (query + context) ===")
    log(prompt)
    log(f"=== End prompt ===\n(Prompt size: {len(prompt)} chars, est_tokens≈{est_tokens})\n")

    answer = safe_generate_text(prompt, max_tokens=ANSWER_MAX_TOKENS, temperature=0.2)
    return answer

# ----------------- GraphRAG main -----------------
def graph_rag_answer(query: str):
    global _LOGGER
    # Create timestamp-named log file at the moment the query is submitted
    ts_name = make_timestamp_name()
    log_file = Path.cwd() / f"{ts_name}.txt"
    _LOGGER = FileLogger(log_file, also_console=True)

    try:
        log("=== GraphRAG run started ===")
        log(f"Log file: {log_file}")
        log(f"Query: {query}")
        log(f"Parameters: TOP_K_ENTITIES={TOP_K_ENTITIES}, TOP_K_TRIPLES={TOP_K_TRIPLES}, N_HOPS={N_HOPS}, "
            f"PER_HOP_LIMIT={PER_HOP_LIMIT}, MAX_EDGES={MAX_EDGES}, MAX_CHUNKS={MAX_CHUNKS}, "
            f"ANSWER_MAX_TOKENS={ANSWER_MAX_TOKENS}")

        # Step 0: Embed query
        t0 = now_ms()
        q_emb = embed_text(query)
        t_embed = dur_ms(t0)
        log(f"[Step 0] Query embedding complete in {t_embed:.0f} ms")

        # Step 1: Extract entities/predicates from query
        t1 = now_ms()
        extraction = extract_entities_predicates_from_query(query)
        t_extract = dur_ms(t1)
        ents = extraction.get("entities", [])
        preds = extraction.get("predicates", [])
        log(f"[Step 1] Extraction done in {t_extract:.0f} ms")
        log(f"  - Entities: {[e.get('text') for e in ents]}")
        log(f"  - Predicates: {preds}")

        # Step 2: Similar entities and triples
        t2 = now_ms()
        sim_entities = search_similar_entities(q_emb, k=TOP_K_ENTITIES)
        sim_triples  = search_similar_triples(q_emb, k=TOP_K_TRIPLES)
        t_search = dur_ms(t2)
        log(f"[Step 2] Vector search done in {t_search:.0f} ms")
        log(f"  - Top entities: {len(sim_entities)}; sample: {sim_entities[:3]}")
        log(f"  - Top triples: {len(sim_triples)}; sample: "
            f"{[{k:v for k,v in x.items() if k in ('subject','predicate','object','uu_number','score')} for x in sim_triples[:3]]}")

        # Step 3: n-hop expansion
        seed_entity_keys = [e["key"] for e in sim_entities if e.get("key")]
        seed_triple_uids = [t["triple_uid"] for t in sim_triples if t.get("triple_uid")]

        t3 = now_ms()
        expanded_from_ents = expand_from_entities(seed_entity_keys, N_HOPS, PER_HOP_LIMIT)
        expanded_from_trs  = expand_from_triples(seed_triple_uids, N_HOPS, PER_HOP_LIMIT)
        t_expand = dur_ms(t3)

        # Merge and deduplicate triples
        triple_map: Dict[str, Dict[str, Any]] = {}
        for t in sim_triples + expanded_from_ents + expanded_from_trs:
            triple_map[t["triple_uid"]] = t
        expanded_triples = list(triple_map.values())

        log(f"[Step 3] Expansion done in {t_expand:.0f} ms")
        log(f"  - Expanded triples (unique): {len(expanded_triples)}")

        # Step 4: Rerank and cap to MAX_EDGES using triple embedding
        t4 = now_ms()
        def score_triple(t: Dict[str, Any]) -> float:
            emb = t.get("embedding")
            return cos_sim(q_emb, emb) if isinstance(emb, list) else 0.0
        ranked_triples = sorted(expanded_triples, key=score_triple, reverse=True)
        selected_triples = ranked_triples[:MAX_EDGES]
        t_rerank = dur_ms(t4)

        # Count nodes/edges selected
        node_keys: Set[str] = set()
        for t in selected_triples:
            if t.get("subject_key"): node_keys.add(t["subject_key"])
            if t.get("object_key"):  node_keys.add(t["object_key"])
        log(f"[Step 4] Rerank+cap done in {t_rerank:.0f} ms")
        log(f"  - Selected edges/triples: {len(selected_triples)}")
        log(f"  - Selected nodes (unique keys): {len(node_keys)}")

        # Step 5: Retrieve chunks and build context
        t5 = now_ms()
        chunk_store = ChunkStore(LANGCHAIN_DIR, set(SKIP_FILES))
        context_text, selected_chunks = build_context_from_triples(selected_triples, chunk_store, MAX_CHUNKS, q_emb)
        t_chunks = dur_ms(t5)
        log(f"[Step 5] Chunk retrieval and context build in {t_chunks:.0f} ms")
        log(f"  - Chunks selected: {len(selected_chunks)}")

        # Step 6: LLM final answer
        t6 = now_ms()
        answer = answer_with_context(query, context_text, output_lang=OUTPUT_LANG)
        t_answer = dur_ms(t6)
        log(f"[Step 6] Final LLM answer in {t_answer:.0f} ms")

        # Summary
        log("\n=== GraphRAG summary ===")
        log(f"- Entities extracted: {len(ents)}; Predicates extracted: {len(preds)}")
        log(f"- Top-k entities: {len(sim_entities)}; Top-k triples: {len(sim_triples)}")
        log(f"- Expanded triples unique: {len(expanded_triples)}")
        log(f"- Selected triples (MAX_EDGES={MAX_EDGES}): {len(selected_triples)}")
        log(f"- Selected nodes (unique): {len(node_keys)}")
        log(f"- Chunks included (MAX_CHUNKS={MAX_CHUNKS}): {len(selected_chunks)}")
        log(f"- Timings (ms): embed={t_embed:.0f}, extract={t_extract:.0f}, search={t_search:.0f}, "
            f"expand={t_expand:.0f}, rerank={t_rerank:.0f}, chunks={t_chunks:.0f}, answer={t_answer:.0f}")

        log("\n=== Final Answer ===")
        log(answer)
        log(f"\nLogs saved to: {log_file}")
        return {
            "answer": answer,
            "log_file": str(log_file),
            "stats": {
                "entities_extracted": ents,
                "predicates_extracted": preds,
                "top_entities": sim_entities,
                "top_triples": sim_triples,
                "expanded_triples": len(expanded_triples),
                "selected_triples": len(selected_triples),
                "selected_nodes": len(node_keys),
                "chunks": len(selected_chunks),
                "timings_ms": {
                    "embed": t_embed, "extract": t_extract, "search": t_search,
                    "expand": t_expand, "rerank": t_rerank, "chunks": t_chunks, "answer": t_answer
                }
            }
        }
    finally:
        if _LOGGER is not None:
            _LOGGER.close()
            _LOGGER = None

# ----------------- CLI -----------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="GraphRAG over Neo4j-backed KG")
    parser.add_argument("--query", type=str, required=False, help="User query to answer")
    parser.add_argument("--n_hops", type=int, default=N_HOPS)
    parser.add_argument("--top_k_entities", type=int, default=TOP_K_ENTITIES)
    parser.add_argument("--top_k_triples", type=int, default=TOP_K_TRIPLES)
    parser.add_argument("--max_edges", type=int, default=MAX_EDGES)
    parser.add_argument("--max_chunks", type=int, default=MAX_CHUNKS)
    args = parser.parse_args()

    # allow runtime overrides
    if args.n_hops: N_HOPS = args.n_hops
    if args.top_k_entities: TOP_K_ENTITIES = args.top_k_entities
    if args.top_k_triples: TOP_K_TRIPLES = args.top_k_triples
    if args.max_edges: MAX_EDGES = args.max_edges
    if args.max_chunks: MAX_CHUNKS = args.max_chunks

    q = args.query or input("Enter your query: ").strip()
    graph_rag_answer(q)