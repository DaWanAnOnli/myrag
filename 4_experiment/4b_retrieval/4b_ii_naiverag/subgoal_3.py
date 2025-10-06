#!/usr/bin/env python3
# agentic_rag.py
# Agentic RAG with Subgoals and Aggregator:
# - Detect user language
# - Subgoal Generator agent: decomposes the query into independent, parallelizable subgoals (bounded by MAX_SUBGOALS)
# - Answer each subgoal in parallel using existing GraphRAG pipeline (embed -> vector search -> answerer)
# - Aggregator agent: synthesizes final answer from original query + subgoal Q/A pairs
# - Logging to a timestamped file (and console)
# - Rate limiting: LLM_CALLS_PER_MINUTE applies ONLY to text generation (not embeddings). Thread-safe.

import os, time, json, pickle, re, random, threading
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
from neo4j import GraphDatabase

# ----------------- Load .env -----------------
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent.parent.parent / ".env")

# ----------------- Config -----------------
GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]
NEO4J_URI  = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASS", "password")

GEN_MODEL   = os.getenv("GEN_MODEL", "models/gemini-2.5-flash")
EMBED_MODEL = os.getenv("EMBED_MODEL", "models/text-embedding-004")

# Retrieval params (naive vector search over chunks)
TOP_K_CHUNKS = int(os.getenv("TOP_K_CHUNKS", "40"))          # initial k from vector index
MAX_CHUNKS_FINAL = int(os.getenv("MAX_CHUNKS_FINAL", "40"))  # chunks kept for final context
CHUNK_TEXT_CLAMP = int(os.getenv("CHUNK_TEXT_CLAMP", "1000000")) # max chars per chunk included in context

# Agent loop (Answerer)
ANSWER_MAX_TOKENS = int(os.getenv("ANSWER_MAX_TOKENS", "4096"))
MAX_ITERS = 1  # We maintain single-pass for answerer per prompt

# Subgoals + Aggregator
MAX_SUBGOALS = int(os.getenv("MAX_SUBGOALS", "2"))  # hard cap on subgoals
SUBGOAL_TOP_K = int(os.getenv("SUBGOAL_TOP_K", str(TOP_K_CHUNKS)))  # top-k per subgoal retrieval
SUBGOAL_MAX_WORKERS = int(os.getenv("SUBGOAL_MAX_WORKERS", "2"))  # parallelism for subgoal answering
AGGREGATOR_MAX_TOKENS = int(os.getenv("AGGREGATOR_MAX_TOKENS", "4096"))

# LLM generation rate limit (calls per minute). Applies ONLY to generation (not embeddings).
LLM_CALLS_PER_MINUTE = int(os.getenv("LLM_CALLS_PER_MINUTE", "13"))

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
        self._lock = threading.Lock()

    def log(self, msg: str = ""):
        if not isinstance(msg, str):
            try:
                msg = json.dumps(msg, ensure_ascii=False, default=str)
            except Exception:
                msg = str(msg)
        with self._lock:
            self._fh.write(msg + "\n")
            self._fh.flush()
            if self.also_console:
                print(msg)

    def close(self):
        try:
            with self._lock:
                self._fh.flush(); self._fh.close()
        except Exception:
            pass

_LOGGER: Optional[FileLogger] = None
def log(msg: str = ""):
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
def estimate_tokens_for_text(text: str) -> int:
    return max(1, int(len(text) / 4))

# --- Thread-safe per-process rate limiter (calls/minute) for LLM generations ---
class RateLimiter:
    def __init__(self, calls_per_minute: int):
        self.calls_per_minute = max(0, int(calls_per_minute))
        self.window = deque()  # monotonic timestamps of recent calls
        self.window_seconds = 60.0
        self._lock = threading.Lock()

    def wait_for_slot(self):
        if self.calls_per_minute <= 0:
            return  # disabled
        while True:
            now = time.monotonic()
            with self._lock:
                # evict old timestamps
                while self.window and (now - self.window[0]) >= self.window_seconds:
                    self.window.popleft()
                if len(self.window) < self.calls_per_minute:
                    self.window.append(now)
                    return
                sleep_time = self.window_seconds - (now - self.window[0])
            if sleep_time > 0:
                log(f"[LLM RateLimit] Sleeping {sleep_time:.2f}s to respect LLM_CALLS_PER_MINUTE={self.calls_per_minute}")
                time.sleep(sleep_time)
            else:
                time.sleep(0.01)

_LLM_RATE_LIMITER = RateLimiter(LLM_CALLS_PER_MINUTE)

def _rand_wait_seconds() -> float:
    return random.uniform(5.0, 20.0)

# --- Retry helpers split by API type ---
def _llm_call_with_retry(func, *args, **kwargs):
    # For text generation only (rate-limited)
    while True:
        try:
            _LLM_RATE_LIMITER.wait_for_slot()
            return func(*args, **kwargs)
        except Exception as e:
            wait_s = _rand_wait_seconds()
            log(f"[Retry] LLM call failed: {e}. Retrying in {wait_s:.1f}s.")
            time.sleep(wait_s)

def _embedding_call_with_retry(func, *args, **kwargs):
    # For embeddings: NO LLM rate limiter
    while True:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            wait_s = _rand_wait_seconds()
            log(f"[Retry] Embedding call failed: {e}. Retrying in {wait_s:.1f}s.")
            time.sleep(wait_s)

def embed_text(text: str) -> List[float]:
    res = _embedding_call_with_retry(genai.embed_content, model=EMBED_MODEL, content=text)
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
    raise RuntimeError("Unexpected embedding response shape")

def run_cypher_with_retry(cypher: str, params: Dict[str, Any]) -> List[Any]:
    while True:
        try:
            with driver.session() as session:
                res = session.run(cypher, **params)
                return list(res)
        except Exception as e:
            wait_s = _rand_wait_seconds()
            log(f"[Retry] Neo4j query failed: {e}. Retrying in {wait_s:.1f}s.")
            time.sleep(wait_s)

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
                    if isinstance(t, str): buf.append(t)
                if buf: return "".join(buf).strip()
    except Exception:
        pass
    return None

def safe_generate_json(prompt: str, schema: Dict[str, Any], temp: float = 0.0) -> Dict[str, Any]:
    cfg = GenerationConfig(temperature=temp, response_mime_type="application/json", response_schema=schema)
    resp = _llm_call_with_retry(gen_model.generate_content, prompt, generation_config=cfg)
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
        return {}

def safe_generate_text(prompt: str, max_tokens: int, temperature: float = 0.2) -> str:
    cfg = GenerationConfig(temperature=temperature, max_output_tokens=max_tokens)
    resp = _llm_call_with_retry(gen_model.generate_content, prompt, generation_config=cfg)
    text = extract_text_from_response(resp)
    if text: return text
    info = get_finish_info(resp)
    log(f"[LLM text warning] No text returned. Diagnostics: {info}")
    return f"(Model returned no text. finish_info={info})"

# ----------------- Agent 1 / 1b (kept for structure, optional signals) -----------------
LEGAL_ENTITY_TYPES = ["UU","PASAL","AYAT","INSTANSI","ORANG","ISTILAH","SANKSI","NOMINAL","TANGGAL"]
LEGAL_PREDICATES = ["mendefinisikan","mengubah","mencabut","mulai_berlaku","mewajibkan","melarang","memberikan_sanksi","berlaku_untuk","termuat_dalam","mendelegasikan_kepada","berjumlah","berdurasi"]

QUERY_SCHEMA = {
  "type": "object",
  "properties": {
    "entities": {"type": "array","items":{"type":"object","properties":{"text":{"type":"string"},"type":{"type":"string"}},"required":["text"]}},
    "predicates": {"type": "array", "items": {"type": "string"}}
  },
  "required": ["entities","predicates"]
}

def agent1_extract_entities_predicates(query: str) -> Dict[str, Any]:
    prompt = f"""
You are Agent 1. Extract legal entities and predicates mentioned or implied by the user's question.

Output JSON:
  - entities: array of {{text, type(optional in {LEGAL_ENTITY_TYPES})}}
  - predicates: array of strings (Indonesian; snake_case if multiword)

User question:
\"\"\"{query}\"\"\"
"""
    est = estimate_tokens_for_text(prompt)
    log("[Agent 1] Prompt:")
    log(prompt)
    log(f"[Agent 1] Prompt size: {len(prompt)} chars, est_tokens≈{est}")
    out = safe_generate_json(prompt, QUERY_SCHEMA, temp=0.0)
    if "entities" not in out: out["entities"] = []
    if "predicates" not in out: out["predicates"] = []
    log(f"[Agent 1] entities={[e.get('text') for e in out['entities']]}, predicates={out['predicates']}")
    return out

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
You are Agent 1b. Extract explicit or implied triples from the user's question in the form: subject — predicate — object.
Use short literal texts as they appear. Predicates: lowercase, snake_case if multiword.

Return JSON with key "triples".
User question:
\"\"\"{query}\"\"\"
"""
    est = estimate_tokens_for_text(prompt)
    log("[Agent 1b] Prompt:")
    log(prompt)
    log(f"[Agent 1b] Prompt size: {len(prompt)} chars, est_tokens≈{est}")
    out = safe_generate_json(prompt, QUERY_TRIPLES_SCHEMA, temp=0.0) or {}
    triples = out.get("triples", [])
    clean = []
    for t in triples or []:
        try:
            s = (t.get("subject") or {}).get("text","").strip()
            p = (t.get("predicate") or "").strip()
            o = (t.get("object")  or {}).get("text","").strip()
            if s and p and o:
                clean.append({"subject":{"text":s,"type":(t.get("subject") or {}).get("type","").strip()},
                              "predicate":p,
                              "object":{"text":o,"type":(t.get("object") or {}).get("type","").strip()}})
        except Exception:
            pass
    formatted = [f"{x['subject']['text']} [{x['predicate']}] {x['object']['text']}" for x in clean]
    log(f"[Agent 1b] Extracted query triples: {formatted}")
    return clean

# ----------------- Vector search over TextChunk -----------------
def vector_query_chunks(q_emb: List[float], k: int) -> List[Dict[str, Any]]:
    """
    Query Neo4j vector index 'chunk_embedding_index' over (:TextChunk {embedding}) property.
    Returns list of dicts with node properties and score.
    """
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

# ----------------- Build context -----------------
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
        lines.append(f"[Chunk {i}] doc={doc} chunk={cid} | {uu} | pages={pg} | score={c.get('score'):.3f}\n{txt}\n")
    return "".join(lines)

# ----------------- Agent 2 (Answerer) -----------------
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
    log("[Agent 2] Prompt:")
    log(prompt)
    log(f"[Agent 2] Prompt size: {len(prompt)} chars, est_tokens≈{est}")
    answer = safe_generate_text(prompt, max_tokens=ANSWER_MAX_TOKENS, temperature=0.2)
    log("[Agent 2] Answer:")
    log(answer)
    return answer

# ----------------- Subgoal Generator (new) -----------------
SUBGOALS_SCHEMA = {
  "type": "object",
  "properties": {
    "subgoals": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "id": {"type": "string"},
          "question": {"type": "string"},
          "rationale": {"type": "string"}
        },
        "required": ["question"]
      }
    }
  },
  "required": ["subgoals"]
}

def _normalize_question(q: str) -> str:
    q = (q or "").strip().lower()
    q = re.sub(r"\s+", " ", q)
    return q

def generate_subgoals(query_original: str, max_subgoals: int, output_lang: str = "en") -> List[Dict[str, str]]:
    """
    Generate up to max_subgoals independent, parallelizable subgoals.
    Fallback: one subgoal identical to the original query if decomposition isn't needed.
    """
    instruction = (
        "Decompose the user's query into independent, parallelizable sub-questions only if it improves clarity or answerability. "
        f"Return at most {max_subgoals} subgoals. "
        "If decomposition is unnecessary, return exactly one subgoal identical to the original query. "
        "Write subgoals in the same language as the user's query."
    )
    prompt = f"""
You are the Subgoal Generator.

Instruction:
{instruction}

User query:
\"\"\"{query_original}\"\"\"

Output JSON schema:
- subgoals: array of items with fields:
  - id (string; optional; you may leave blank)
  - question (string; REQUIRED; an independently answerable sub-question)
  - rationale (string; optional; why this subgoal helps)
"""
    est = estimate_tokens_for_text(prompt)
    log("=== Subgoal Generation ===")
    log("[SubgoalGen] Prompt:")
    log(prompt)
    log(f"[SubgoalGen] Prompt size: {len(prompt)} chars, est_tokens≈{est}")

    out = safe_generate_json(prompt, SUBGOALS_SCHEMA, temp=0.0) or {}
    raw_subgoals = out.get("subgoals", []) or []

    # Cleanup, deduplicate, clamp
    cleaned = []
    seen = set()
    for i, sg in enumerate(raw_subgoals, 1):
        q = (sg.get("question") or "").strip()
        if not q:
            continue
        key = _normalize_question(q)
        if key in seen:
            continue
        seen.add(key)
        rid = (sg.get("id") or "").strip() or f"SG{i}"
        rationale = (sg.get("rationale") or "").strip()
        cleaned.append({"id": rid, "question": q, "rationale": rationale})
        if len(cleaned) >= max_subgoals:
            break

    # Fallback if nothing valid
    if not cleaned:
        cleaned = [{"id": "SG1", "question": query_original.strip(), "rationale": "Decomposition not needed; answer the original query directly."}]

    log(f"[SubgoalGen] Produced {len(cleaned)} subgoal(s):")
    for sg in cleaned:
        log(f"  - {sg['id']}: {sg['question']} (rationale: {sg.get('rationale','')})")

    return cleaned

# ----------------- Subgoal Answering (parallel GraphRAG) -----------------
def _format_context_preview(ctx: str, max_lines: int = 30) -> str:
    lines = ctx.splitlines()
    return "\n".join(lines[:max_lines])

def answer_single_subgoal(subgoal: Dict[str, str], user_lang: str) -> Dict[str, Any]:
    """
    Answer a single subgoal using the existing RAG pipeline.
    Returns a dict with id, question, answer, retrieval_meta.
    """
    sg_id = subgoal.get("id", "")
    q = subgoal.get("question", "").strip()
    start = time.time()
    log(f"[Subgoal {sg_id}] Start answering")

    try:
        t0 = time.time()
        q_emb = embed_text(q)
        log(f"[Subgoal {sg_id}] Embedded in {(time.time()-t0)*1000:.0f} ms")

        t1 = time.time()
        candidates = vector_query_chunks(q_emb, k=SUBGOAL_TOP_K)
        log(f"[Subgoal {sg_id}] Vector search returned {len(candidates)} candidates in {(time.time()-t1)*1000:.0f} ms")

        if not candidates:
            context_text = "Tidak ada potongan teks yang ditemukan." if user_lang == "id" else "No relevant chunks found."
        else:
            context_text = build_context_from_chunks(candidates, max_chunks=MAX_CHUNKS_FINAL)
            log(f"[Subgoal {sg_id}] Context preview:\n{_format_context_preview(context_text, max_lines=20)}")

        t2 = time.time()
        ans = agent2_answer(q, context_text, guidance=None, output_lang=user_lang)
        log(f"[Subgoal {sg_id}] Answer generated in {(time.time()-t2)*1000:.0f} ms")

        # Build retrieval meta (lightweight)
        meta_top = []
        for c in (candidates[:5] if candidates else []):
            meta_top.append({
                "document_id": c.get("document_id"),
                "chunk_id": c.get("chunk_id"),
                "uu_number": c.get("uu_number"),
                "pages": c.get("pages"),
                "score": c.get("score"),
            })

        total_ms = (time.time() - start) * 1000.0
        log(f"[Subgoal {sg_id}] Done in {total_ms:.0f} ms")

        return {
            "id": sg_id,
            "question": q,
            "answer": ans,
            "retrieval_meta": {
                "num_candidates": len(candidates) if candidates else 0,
                "top_chunks": meta_top
            }
        }
    except Exception as e:
        log(f"[Subgoal {sg_id}] ERROR: {e}")
        return {
            "id": sg_id,
            "question": q,
            "answer": "",
            "error": str(e),
            "retrieval_meta": {
                "num_candidates": 0,
                "top_chunks": []
            }
        }

def answer_subgoals_parallel(subgoals: List[Dict[str, str]], user_lang: str) -> List[Dict[str, Any]]:
    """
    Answer all subgoals in parallel and return list of Q/A dicts in the same order as input.
    """
    log("=== Subgoal Answering (Parallel) ===")
    results: List[Optional[Dict[str, Any]]] = [None] * len(subgoals)
    with ThreadPoolExecutor(max_workers=SUBGOAL_MAX_WORKERS) as executor:
        futures = {}
        for idx, sg in enumerate(subgoals):
            futures[executor.submit(answer_single_subgoal, sg, user_lang)] = idx
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                res = fut.result()
            except Exception as e:
                log(f"[Subgoal index {idx}] Future error: {e}")
                sg = subgoals[idx]
                res = {"id": sg.get("id",""), "question": sg.get("question",""), "answer": "", "error": str(e), "retrieval_meta": {"num_candidates": 0, "top_chunks": []}}
            results[idx] = res
    # Fill any gaps (safety)
    return [r if r is not None else {"id": subgoals[i].get("id",""), "question": subgoals[i].get("question",""), "answer": "", "retrieval_meta": {"num_candidates": 0, "top_chunks": []}} for i, r in enumerate(results)]

# ----------------- Aggregator Agent (final synthesis) -----------------
def aggregate_answers(original_query: str, subgoal_qas: List[Dict[str, Any]], output_lang: str = "en") -> str:
    """
    Aggregate subgoal answers into a final answer to the original query.
    """
    # Build a compact list of Q/A pairs
    lines = []
    for i, qa in enumerate(subgoal_qas, 1):
        sgid = qa.get("id", f"SG{i}")
        q = qa.get("question", "").strip()
        a = (qa.get("answer", "") or "").strip()
        if not a:
            a = "(No answer or retrieval failed.)"
        lines.append(f"[{sgid}] Question: {q}\n[{sgid}] Answer: {a}\n")
    qa_block = "\n".join(lines)

    instruction = (
        "Synthesize a final, concise, and accurate answer to the original query using ONLY the information contained in the subgoal answers. "
        "If there are conflicts among subgoal answers, resolve them explicitly and explain your reasoning briefly. "
        "If evidence is missing or inconclusive, state the limitation clearly. "
        "Preserve any UU/Article references or citations mentioned in the subgoal answers. "
        "Respond in the same language as the user's question."
    )

    prompt = f"""
You are the Final Aggregator.

Original user query:
\"\"\"{original_query}\"\"\"

Subgoal Q/A pairs:
\"\"\"{qa_block}\"\"\"

Instructions:
{instruction}
"""
    est = estimate_tokens_for_text(prompt)
    log("=== Aggregation ===")
    log("[Aggregator] Prompt:")
    log(prompt)
    log(f"[Aggregator] Prompt size: {len(prompt)} chars, est_tokens≈{est}")
    out = safe_generate_text(prompt, max_tokens=AGGREGATOR_MAX_TOKENS, temperature=0.2)
    log("[Aggregator] Final Answer:")
    log(out)
    return out

# ----------------- Agentic RAG main (with Subgoals + Aggregator) -----------------
def agentic_rag(query_original: str) -> Dict[str, Any]:
    global _LOGGER
    ts_name = make_timestamp_name()
    log_file = Path.cwd() / f"{ts_name}.txt"
    _LOGGER = FileLogger(log_file, also_console=True)

    try:
        log("=== Agentic RAG run started ===")
        log(f"Log file: {log_file}")
        log(f"Original Query: {query_original}")
        log(f"Parameters: TOP_K_CHUNKS={TOP_K_CHUNKS}, MAX_CHUNKS_FINAL={MAX_CHUNKS_FINAL}, CHUNK_TEXT_CLAMP={CHUNK_TEXT_CLAMP}, "
            f"ANSWER_MAX_TOKENS={ANSWER_MAX_TOKENS}, AGGREGATOR_MAX_TOKENS={AGGREGATOR_MAX_TOKENS}, MAX_SUBGOALS={MAX_SUBGOALS}, "
            f"SUBGOAL_TOP_K={SUBGOAL_TOP_K}, SUBGOAL_MAX_WORKERS={SUBGOAL_MAX_WORKERS}, MAX_ITERS={MAX_ITERS}, "
            f"LLM_CALLS_PER_MINUTE={LLM_CALLS_PER_MINUTE}")

        user_lang = detect_user_language(query_original)
        log(f"[Language] Detected user language: {user_lang}")

        # Subgoal Generation
        subgoals = generate_subgoals(query_original, MAX_SUBGOALS, output_lang=user_lang)

        # Optional: Agent 1 & 1b on the original query (kept for structure)
        t1 = time.time()
        _ = agent1_extract_entities_predicates(query_original)
        _ = agent1b_extract_query_triples(query_original)
        log(f"[Step] Entity/Triple extraction on original query in {(time.time()-t1)*1000:.0f} ms")

        # Answer subgoals in parallel
        t2 = time.time()
        subgoal_qas = answer_subgoals_parallel(subgoals, user_lang)
        log(f"[Step] Subgoal answering completed in {(time.time()-t2)*1000:.0f} ms")

        # Aggregate into final answer
        t3 = time.time()
        final_answer = aggregate_answers(query_original, subgoal_qas, output_lang=user_lang)
        log(f"[Step] Aggregation completed in {(time.time()-t3)*1000:.0f} ms")

        # Summary
        log("=== Agentic RAG summary ===")
        log(f"- Subgoals: {len(subgoals)}")
        for sg in subgoals:
            log(f"  • {sg['id']}: {sg['question']}")
        log("=== Final Answer ===")
        log(final_answer or "")
        log(f"Logs saved to: {log_file}")

        return {
            "final_answer": final_answer or "",
            "subgoals": subgoals,
            "subgoal_answers": subgoal_qas,
            "log_file": str(log_file),
            "iterations": 1,
            "judge_reports": []  # kept for downstream compatibility; empty
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
            agentic_rag(user_query)
    finally:
        try:
            driver.close()
        except Exception:
            pass