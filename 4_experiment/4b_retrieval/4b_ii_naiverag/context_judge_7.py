#!/usr/bin/env python3
# agentic_rag.py
# Agentic RAG with Context Judge + Query Modifier:
# - Detect user language
# - Retrieval over graph chunks (embed -> vector search -> build context)
# - Context Judge agent: evaluates if retrieved context is sufficient/relevant for the current query,
#   using prior query-feedback history; if insufficient, produces concrete feedback on how to improve the query
# - Query Modifier agent: rewrites the query based on judge feedback and prior history to improve retrieval
# - Iterative loop with a hard cap on iterations; if the cap is reached, skip judge/modifier and go straight to Answerer
# - The existing Answerer produces the final answer strictly from the provided context
# - Entity extractor (Agent 1/1b) runs each iteration for analysis/logging signals
# - Logging to a timestamped file (and console)
# - Rate limiting: LLM_CALLS_PER_MINUTE applies ONLY to text generation (not embeddings). Thread-safe.

import os, time, json, re, random, threading
from pathlib import Path
from typing import Any, Dict, List, Optional
from collections import deque

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
TOP_K_CHUNKS = int(os.getenv("TOP_K_CHUNKS", "40"))               # initial k from vector index
MAX_CHUNKS_FINAL = int(os.getenv("MAX_CHUNKS_FINAL", "40"))       # chunks kept for final context
CHUNK_TEXT_CLAMP = int(os.getenv("CHUNK_TEXT_CLAMP", "1000000"))  # max chars per chunk included in context
CHUNK_DIGEST_CLAMP = int(os.getenv("CHUNK_DIGEST_CLAMP", "5000000000"))  # clamp per-chunk content for judge prompts
JUDGE_MAX_CHUNK_DIGESTS = int(os.getenv("JUDGE_MAX_CHUNK_DIGESTS", "40"))  # max chunk digests passed to judge

# Answerer
ANSWER_MAX_TOKENS = int(os.getenv("ANSWER_MAX_TOKENS", "4096"))

# Iterative judge+modifier settings
MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "4"))  # hard cap on judge+modifier iterations
JUDGE_MAX_TOKENS = int(os.getenv("JUDGE_MAX_TOKENS", "4096"))
MODIFIER_MAX_TOKENS = int(os.getenv("MODIFIER_MAX_TOKENS", "4096"))

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
    return random.uniform(50.0, 80.0)

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
            wait_s = random.uniform(5.0, 15.0)
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

# ----------------- Build context and digests -----------------
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

def build_chunk_digests_for_judge(chunks: List[Dict[str, Any]], max_items: int, clamp_chars: int) -> str:
    chosen = chunks[:max_items]
    lines = []
    for i, c in enumerate(chosen, 1):
        doc = c.get("document_id")
        cid = c.get("chunk_id")
        uu = c.get("uu_number") or ""
        pg = c.get("pages")
        txt = clamp(c.get("content") or "", clamp_chars)
        lines.append(f"[C{i}] doc={doc} chunk={cid} | {uu} | pages={pg} | score={c.get('score'):.3f}\n{txt}")
    return "\n".join(lines) if lines else "(No chunks retrieved)"

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

# ----------------- Context Judge Agent -----------------
JUDGE_SCHEMA = {
  "type": "object",
  "properties": {
    "sufficient": {"type": "boolean"},
    "reasoning": {"type": "string"},
    "feedback": {
      "type": "object",
      "properties": {
        "problem": {"type": "string"},
        "suggestion": {"type": "string"}
      }
    }
  },
  "required": ["sufficient"]
}

def context_judge(current_query: str,
                  chunk_digests: str,
                  query_feedback_history: List[Dict[str, Any]],
                  user_lang: str = "en") -> Dict[str, Any]:
    """
    Judge whether the retrieved context is sufficient/relevant to answer the current query.
    If insufficient, provide concrete feedback on how to modify the query to improve retrieval.
    """
    # Build prior query-feedback block
    lines = []
    for i, item in enumerate(query_feedback_history, 1):
        q = (item.get("query") or "").strip()
        fb = item.get("feedback") or {}
        prob = (fb.get("problem") or "").strip()
        sug = (fb.get("suggestion") or "").strip()
        lines.append(f"[Hist {i}] Query: {q}\n[Hist {i}] Problem: {prob}\n[Hist {i}] Suggestion: {sug}")
    history_block = "\n".join(lines) if lines else "(No prior query-feedback pairs)"

    pipeline_brief = (
        "GraphRAG pipeline overview:\n"
        "- The system embeds the current query using an embedding model and performs a vector search over graph chunk embeddings.\n"
        "- It retrieves the top-K chunks by similarity.\n"
        "- The Answerer is constrained to produce answers only from the retrieved chunks' content.\n"
        "- Better retrieval often requires: narrowing scope (specific UU/Article), adding entities, using synonyms or domain terms, adding dates/periods, or clarifying the requested relationship."
    )

    instruction = (
        "Task: Evaluate whether the retrieved chunks are sufficient and relevant to answer the current query.\n"
        "- If sufficient: set sufficient=true and briefly explain why in 'reasoning'.\n"
        "- If insufficient: set sufficient=false and provide:\n"
        "  * feedback.problem: What seems wrong with the query vis-à-vis the retrieved chunks (too broad, wrong entity, missing UU number, missing timeframe, etc.).\n"
        "  * feedback.suggestion: A concrete suggestion to modify the query to improve retrieval (e.g., add specific article, include key entity names, add synonyms, specify timeframe).\n"
        "Respond in the same language as the user's query."
    )

    prompt = f"""
You are the Context Judge.

{pipeline_brief}

Current query:
\"\"\"{current_query}\"\"\"

Retrieved chunk digests (subset):
\"\"\"{chunk_digests}\"\"\"

Previous query-feedback pairs:
\"\"\"{history_block}\"\"\"

Instructions:
{instruction}

Return JSON with keys: sufficient (bool), reasoning (string), and feedback (object with problem, suggestion) when insufficient.
"""
    est = estimate_tokens_for_text(prompt)
    log("=== Context Judge ===")
    log("[Judge] Prompt:")
    log(prompt)
    log(f"[Judge] Prompt size: {len(prompt)} chars, est_tokens≈{est}")

    out = safe_generate_json(prompt, JUDGE_SCHEMA, temp=0.0) or {}
    sufficient = bool(out.get("sufficient", False))
    reasoning = (out.get("reasoning") or "").strip()
    feedback = out.get("feedback") or {}
    problem = (feedback.get("problem") or "").strip()
    suggestion = (feedback.get("suggestion") or "").strip()

    log(f"[Judge] sufficient={sufficient}")
    if reasoning:
        log(f"[Judge] reasoning: {reasoning}")
    if not sufficient:
        log(f"[Judge] feedback.problem: {problem}")
        log(f"[Judge] feedback.suggestion: {suggestion}")

    return {
        "sufficient": sufficient,
        "reasoning": reasoning,
        "feedback": {"problem": problem, "suggestion": suggestion}
    }

# ----------------- Query Modifier Agent -----------------
MODIFIER_SCHEMA = {
  "type": "object",
  "properties": {
    "modified_query": {"type": "string"},
    "rationale": {"type": "string"}
  },
  "required": ["modified_query"]
}

def query_modifier(current_query: str,
                   judge_feedback: Dict[str, str],
                   query_feedback_history: List[Dict[str, Any]],
                   user_lang: str = "en") -> Dict[str, str]:
    """
    Rewrite the query based on the judge's feedback and prior history to improve retrieval.
    Must respect the GraphRAG pipeline (vector search over chunk embeddings).
    """
    lines = []
    for i, item in enumerate(query_feedback_history, 1):
        q = (item.get("query") or "").strip()
        fb = item.get("feedback") or {}
        prob = (fb.get("problem") or "").strip()
        sug = (fb.get("suggestion") or "").strip()
        lines.append(f"[Hist {i}] Query: {q}\n[Hist {i}] Problem: {prob}\n[Hist {i}] Suggestion: {sug}")
    history_block = "\n".join(lines) if lines else "(No prior query-feedback pairs)"

    pipeline_brief = (
        "GraphRAG pipeline constraints to consider:\n"
        "- Retrieval is driven by semantic similarity of the embedded query to chunk embeddings.\n"
        "- Adding specific entities (e.g., UU numbers, articles, named institutions), synonyms, key legal terms, or timeframe can significantly affect matches.\n"
        "- Avoid inventing facts; modify only the query phrasing/scope to guide retrieval."
    )

    instruction = (
        "Rewrite the current query to improve the chance of retrieving relevant chunks, using the judge's feedback and prior history.\n"
        "- Keep the user's language.\n"
        "- Be precise and concise; prefer adding concrete identifiers (UU/Article, dates, entities) and domain terms.\n"
        "- Do not add new claims; only rephrase or focus the query.\n"
        "Return JSON with 'modified_query' and optional 'rationale'."
    )

    problem = (judge_feedback.get("problem") or "").strip()
    suggestion = (judge_feedback.get("suggestion") or "").strip()

    prompt = f"""
You are the Query Modifier.

{pipeline_brief}

Current query:
\"\"\"{current_query}\"\"\"

Judge feedback:
- Problem: {problem}
- Suggestion: {suggestion}

Previous query-feedback pairs:
\"\"\"{history_block}\"\"\"

Instructions:
{instruction}
"""
    est = estimate_tokens_for_text(prompt)
    log("=== Query Modifier ===")
    log("[Modifier] Prompt:")
    log(prompt)
    log(f"[Modifier] Prompt size: {len(prompt)} chars, est_tokens≈{est}")

    out = safe_generate_json(prompt, MODIFIER_SCHEMA, temp=0.0) or {}
    mod_q = (out.get("modified_query") or "").strip()
    rationale = (out.get("rationale") or "").strip()
    if not mod_q:
        mod_q = current_query  # fallback
        log("[Modifier] No modified query produced; using current query as-is.")
    else:
        log(f"[Modifier] Modified query: {mod_q}")
        if rationale:
            log(f"[Modifier] Rationale: {rationale}")

    return {"modified_query": mod_q, "rationale": rationale}

# ----------------- Agentic RAG main (Iterative Judge + Modifier) -----------------
def agentic_rag(query_original: str) -> Dict[str, Any]:
    global _LOGGER
    ts_name = make_timestamp_name()
    log_file = Path.cwd() / f"{ts_name}.txt"
    _LOGGER = FileLogger(log_file, also_console=True)

    try:
        log("=== Agentic RAG (Context Judge + Query Modifier) run started ===")
        log(f"Log file: {log_file}")
        log(f"Original Query: {query_original}")
        log(f"Parameters: TOP_K_CHUNKS={TOP_K_CHUNKS}, MAX_CHUNKS_FINAL={MAX_CHUNKS_FINAL}, CHUNK_TEXT_CLAMP={CHUNK_TEXT_CLAMP}, "
            f"ANSWER_MAX_TOKENS={ANSWER_MAX_TOKENS}, MAX_ITERATIONS={MAX_ITERATIONS}, "
            f"JUDGE_MAX_TOKENS={JUDGE_MAX_TOKENS}, MODIFIER_MAX_TOKENS={MODIFIER_MAX_TOKENS}, "
            f"LLM_CALLS_PER_MINUTE={LLM_CALLS_PER_MINUTE}")

        user_lang = detect_user_language(query_original)
        log(f"[Language] Detected user language: {user_lang}")

        # Iterative loop state
        current_query = query_original.strip()
        query_feedback_history: List[Dict[str, Any]] = []  # list of {query, feedback{problem,suggestion}, iteration}
        judge_decisions: List[Dict[str, Any]] = []         # list of per-iteration judge outputs
        final_answer = ""
        final_context = ""
        final_iteration_index = 0
        final_query_used = ""

        for it in range(1, MAX_ITERATIONS + 1):
            log(f"\n=== Iteration {it}/{MAX_ITERATIONS} ===")
            log(f"[Iter {it}] Current query: {current_query}")

            # Optional analysis: Agent 1 & 1b on the current query (signals only)
            t_a = time.time()
            _ = agent1_extract_entities_predicates(current_query)
            _ = agent1b_extract_query_triples(current_query)
            log(f"[Iter {it}] Entity/Triple extraction in {(time.time()-t_a)*1000:.0f} ms")

            # Retrieval for current query
            t_r = time.time()
            q_emb = embed_text(current_query)
            log(f"[Iter {it}] Embedded query in {(time.time()-t_r)*1000:.0f} ms")

            t_v = time.time()
            candidates = vector_query_chunks(q_emb, k=TOP_K_CHUNKS)
            log(f"[Iter {it}] Vector search returned {len(candidates)} candidates in {(time.time()-t_v)*1000:.0f} ms")

            if not candidates:
                context_text = "Tidak ada potongan teks yang ditemukan." if user_lang == "id" else "No relevant chunks found."
            else:
                context_text = build_context_from_chunks(candidates, max_chunks=MAX_CHUNKS_FINAL)
                prev = "\n".join(context_text.splitlines()[:20])
                log(f"[Iter {it}] Context preview:\n{prev}")

            # If iteration limit reached, skip judge/modifier and answer directly
            if it >= MAX_ITERATIONS:
                log(f"[Iter {it}] Reached MAX_ITERATIONS={MAX_ITERATIONS}. Skipping Judge/Modifier and going straight to Answerer.")
                final_answer = agent2_answer(current_query, context_text, guidance=None, output_lang=user_lang)
                final_context = context_text
                final_iteration_index = it
                final_query_used = current_query
                break

            # Judge: decide if context is sufficient
            chunk_digests = build_chunk_digests_for_judge(candidates, JUDGE_MAX_CHUNK_DIGESTS, CHUNK_DIGEST_CLAMP)
            judge_res = context_judge(current_query, chunk_digests, query_feedback_history, user_lang=user_lang)
            judge_decisions.append({"iteration": it, **judge_res})

            if judge_res.get("sufficient"):
                log(f"[Iter {it}] Judge deemed context sufficient. Proceeding to Answerer.")
                final_answer = agent2_answer(current_query, context_text, guidance=None, output_lang=user_lang)
                final_context = context_text
                final_iteration_index = it
                final_query_used = current_query
                break
            else:
                # Store feedback and modify query for next iteration
                fb = judge_res.get("feedback") or {}
                query_feedback_history.append({
                    "iteration": it,
                    "query": current_query,
                    "feedback": {"problem": fb.get("problem",""), "suggestion": fb.get("suggestion","")}
                })
                log(f"[Iter {it}] Judge suggests modifying query. Running Query Modifier.")
                mod_out = query_modifier(current_query, fb, query_feedback_history, user_lang=user_lang)
                next_query = (mod_out.get("modified_query") or "").strip()
                if not next_query:
                    next_query = current_query  # fallback safety
                if next_query == current_query:
                    log(f"[Iter {it}] Modified query is identical to current query; will retry retrieval next iteration.")
                else:
                    log(f"[Iter {it}] Next query: {next_query}")
                current_query = next_query
                # continue loop

        # If loop ended without setting final_answer (edge cases), answer with last context
        if not final_answer:
            log("[Final] No earlier conclusion; answering with latest context.")
            final_answer = agent2_answer(current_query, context_text, guidance=None, output_lang=user_lang)
            final_context = context_text
            final_iteration_index = min(MAX_ITERATIONS, max(1, final_iteration_index or 1))
            final_query_used = current_query

        # Summary
        log("\n=== Agentic RAG (Judge + Modifier) summary ===")
        log(f"- Iterations executed: {final_iteration_index}")
        for h in query_feedback_history:
            log(f"  • Iter {h['iteration']}: Query='{h['query']}' | Problem='{h['feedback'].get('problem','')}' | Suggestion='{h['feedback'].get('suggestion','')}'")
        log("=== Final Query Used ===")
        log(final_query_used)
        log("=== Final Answer ===")
        log(final_answer or "")
        log(f"Logs saved to: {log_file}")

        return {
            "final_answer": final_answer or "",
            "final_query": final_query_used,
            "iterations": final_iteration_index,
            "judge_decisions": judge_decisions,
            "query_feedback_history": query_feedback_history,
            "log_file": str(log_file),
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