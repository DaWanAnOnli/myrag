#!/usr/bin/env python3
# agentic_rag.py
# Agentic RAG with Intermediate Questions (IQ) and Query Modifier:
# - Detect user language
# - Intermediate Question Generator: produces a sequential, dependent plan of questions (bounded by MAX_INTERMEDIATE_QUESTIONS)
# - Sequential loop: for each IQ, optionally enrich via Query Modifier (except the first), then run retrieval + answerer
# - Query Modifier: completes/enriches the next IQ using all previous IQ Q/A pairs
# - The last IQ's answer is the final answer (no aggregator)
# - Logging to a timestamped file (and console); logs include which IQ is being processed, e.g., 1/2, 2/2
# - Rate limiting: LLM_CALLS_PER_MINUTE applies ONLY to text generation (not embeddings). Thread-safe.

import os, time, json, pickle, re, random, threading
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed  # not used for IQs but kept if needed elsewhere

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

# Answerer
ANSWER_MAX_TOKENS = int(os.getenv("ANSWER_MAX_TOKENS", "4096"))

# Intermediate Questions settings
MAX_INTERMEDIATE_QUESTIONS = int(os.getenv("MAX_INTERMEDIATE_QUESTIONS", "5"))
IQ_GENERATOR_MAX_TOKENS = int(os.getenv("IQ_GENERATOR_MAX_TOKENS", "4096"))
IQ_MODIFIER_MAX_TOKENS  = int(os.getenv("IQ_MODIFIER_MAX_TOKENS", "4096"))

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
    return random.uniform(50, 80)

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
            wait_s = random.uniform(5.0, 20.0)
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

# ----------------- Intermediate Question Generator -----------------
IQ_PLAN_SCHEMA = {
  "type": "object",
  "properties": {
    "iqs": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "id": {"type": "string"},
          "question": {"type": "string"},
          "rationale": {"type": "string"},
          "depends_on": {"type": "array", "items": {"type": "string"}}
        },
        "required": ["question"]
      }
    }
  },
  "required": ["iqs"]
}

def _normalize_question(q: str) -> str:
    q = (q or "").strip().lower()
    q = re.sub(r"\s+", " ", q)
    return q

def generate_intermediate_questions(query_original: str, max_iqs: int, output_lang: str = "en") -> List[Dict[str, Any]]:
    """
    Generate up to max_iqs sequential, dependent intermediate questions (IQs).
    Fallback: one IQ identical to the original query if decomposition isn't needed.
    """
    instruction = (
        "Plan a short sequence of dependent intermediate questions (IQs) to answer the user's query. "
        "Make later IQs depend on earlier answers only if that helps. "
        f"Return at most {max_iqs} IQs. If decomposition isn't needed, return exactly one IQ identical to the original query. "
        "Write IQs in the same language as the user's query."
    )
    prompt = f"""
You are the Intermediate Question Planner.

Instruction:
{instruction}

User query:
\"\"\"{query_original}\"\"\"

Output JSON schema:
- iqs: array of items with fields:
  - id (string; optional; you may leave blank)
  - question (string; REQUIRED; the intermediate question)
  - rationale (string; optional; why this IQ is needed)
  - depends_on (array of prior IQ ids, optional)
"""
    est = estimate_tokens_for_text(prompt)
    log("=== IQ Generation ===")
    log("[IQ Planner] Prompt:")
    log(prompt)
    log(f"[IQ Planner] Prompt size: {len(prompt)} chars, est_tokens≈{est}")

    out = safe_generate_json(prompt, IQ_PLAN_SCHEMA, temp=0.0) or {}
    raw_iqs = out.get("iqs", []) or []

    # Cleanup, deduplicate, clamp
    cleaned: List[Dict[str, Any]] = []
    seen = set()
    for i, iq in enumerate(raw_iqs, 1):
        q = (iq.get("question") or "").strip()
        if not q:
            continue
        key = _normalize_question(q)
        if key in seen:
            continue
        seen.add(key)
        rid = (iq.get("id") or "").strip() or f"IQ{i}"
        rationale = (iq.get("rationale") or "").strip()
        deps = iq.get("depends_on") or []
        cleaned.append({"id": rid, "question": q, "rationale": rationale, "depends_on": deps})
        if len(cleaned) >= max_iqs:
            break

    if not cleaned:
        cleaned = [{"id": "IQ1", "question": query_original.strip(), "rationale": "Decomposition not needed; answer the original query directly.", "depends_on": []}]

    log(f"[IQ Planner] Planned {len(cleaned)} IQ(s):")
    for iq in cleaned:
        log(f"  - {iq['id']}: {iq['question']} (rationale: {iq.get('rationale','')})")
    return cleaned

# ----------------- Query Modifier (enrich the next IQ using previous Q/As) -----------------
IQ_MODIFY_SCHEMA = {
  "type": "object",
  "properties": {
    "enriched_question": {"type": "string"}
  },
  "required": ["enriched_question"]
}

def modify_next_iq(prev_qas: List[Dict[str, str]], planned_iq: Dict[str, Any], output_lang: str = "en") -> str:
    """
    Enrich/complete the next planned IQ using all previous IQ question-answer pairs.
    If enrichment is unnecessary, may return the planned IQ unchanged.
    """
    # Build prior Q/A block
    lines = []
    for i, qa in enumerate(prev_qas, 1):
        pid = qa.get("iq_id", f"IQ{i}")
        pq = qa.get("question", "").strip()
        pa = (qa.get("answer", "") or "").strip()
        lines.append(f"[{pid}] Question: {pq}\n[{pid}] Answer: {pa}")
    prev_block = "\n".join(lines) if lines else "(No previous Q/A pairs)"

    next_id = planned_iq.get("id", "")
    next_q  = planned_iq.get("question", "").strip()

    instruction = (
        "Using only the information contained in the previous Q/A pairs and the planned next IQ, rewrite the next IQ as a fully specified, concrete question. "
        "Resolve placeholders (e.g., 'the result of previous step') with actual prior answers. "
        "Do not introduce new facts beyond prior answers; if data is missing, keep the question as-is while clarifying the missing piece. "
        "Return the enriched question in the same language as the user's query."
    )

    prompt = f"""
You are the Query Modifier.

Previous Q/A pairs:
\"\"\"{prev_block}\"\"\"

Planned next IQ ({next_id}):
\"\"\"{next_q}\"\"\"

Instructions:
{instruction}

Return JSON with key "enriched_question".
"""
    est = estimate_tokens_for_text(prompt)
    log("[Query Modifier] Prompt:")
    log(prompt)
    log(f"[Query Modifier] Prompt size: {len(prompt)} chars, est_tokens≈{est}")

    out = safe_generate_json(prompt, IQ_MODIFY_SCHEMA, temp=0.0) or {}
    enriched = (out.get("enriched_question") or "").strip()
    if not enriched:
        # Fallback to planned question
        enriched = next_q
        log("[Query Modifier] No enrichment produced; using planned IQ as-is.")
    else:
        log(f"[Query Modifier] Enriched IQ: {enriched}")
    return enriched

# ----------------- Agentic RAG main (Sequential IQs + Query Modifier) -----------------
def agentic_rag(query_original: str) -> Dict[str, Any]:
    global _LOGGER
    ts_name = make_timestamp_name()
    log_file = Path.cwd() / f"{ts_name}.txt"
    _LOGGER = FileLogger(log_file, also_console=True)

    try:
        log("=== Agentic RAG (Sequential IQs) run started ===")
        log(f"Log file: {log_file}")
        log(f"Original Query: {query_original}")
        log(f"Parameters: TOP_K_CHUNKS={TOP_K_CHUNKS}, MAX_CHUNKS_FINAL={MAX_CHUNKS_FINAL}, CHUNK_TEXT_CLAMP={CHUNK_TEXT_CLAMP}, "
            f"ANSWER_MAX_TOKENS={ANSWER_MAX_TOKENS}, MAX_INTERMEDIATE_QUESTIONS={MAX_INTERMEDIATE_QUESTIONS}, "
            f"IQ_GENERATOR_MAX_TOKENS={IQ_GENERATOR_MAX_TOKENS}, IQ_MODIFIER_MAX_TOKENS={IQ_MODIFIER_MAX_TOKENS}, "
            f"LLM_CALLS_PER_MINUTE={LLM_CALLS_PER_MINUTE}")

        user_lang = detect_user_language(query_original)
        log(f"[Language] Detected user language: {user_lang}")

        # Step 0: Generate IQ plan
        t0 = time.time()
        iq_plan = generate_intermediate_questions(query_original, MAX_INTERMEDIATE_QUESTIONS, output_lang=user_lang)
        log(f"[Step 0] IQ planning in {(time.time()-t0)*1000:.0f} ms")

        # Sequential execution over IQs
        iq_runs: List[Dict[str, Any]] = []
        total_iqs = len(iq_plan)
        log("=== IQ Loop (Sequential) ===")

        for idx, planned in enumerate(iq_plan, 1):
            iq_id = planned.get("id", f"IQ{idx}")
            planned_q = planned.get("question", "").strip()

            log(f"--- IQ {idx}/{total_iqs} ---")
            log(f"[{iq_id}] Planned IQ: {planned_q}")

            # Enrich next IQ if not the first one
            if idx == 1:
                used_q = planned_q
                log(f"[{iq_id}] Using planned IQ as-is (first step).")
            else:
                t_mod = time.time()
                used_q = modify_next_iq(
                    prev_qas=[{"iq_id": run.get("iq_id",""), "question": run.get("used_question",""), "answer": run.get("answer","")} for run in iq_runs],
                    planned_iq=planned,
                    output_lang=user_lang
                )
                log(f"[{iq_id}] Enrichment done in {(time.time()-t_mod)*1000:.0f} ms")
                log(f"[{iq_id}] Enriched IQ: {used_q}")

            # Optional analysis: Agent 1 & 1b on the used IQ
            t_a = time.time()
            _ = agent1_extract_entities_predicates(used_q)
            _ = agent1b_extract_query_triples(used_q)
            log(f"[{iq_id}] Entity/Triple extraction in {(time.time()-t_a)*1000:.0f} ms")

            # Retrieval + Answerer for this IQ
            t_r = time.time()
            q_emb = embed_text(used_q)
            log(f"[{iq_id}] Embedded used IQ in {(time.time()-t_r)*1000:.0f} ms")

            t_v = time.time()
            candidates = vector_query_chunks(q_emb, k=TOP_K_CHUNKS)
            log(f"[{iq_id}] Vector search returned {len(candidates)} candidates in {(time.time()-t_v)*1000:.0f} ms")

            if not candidates:
                context_text = "Tidak ada potongan teks yang ditemukan." if user_lang == "id" else "No relevant chunks found."
            else:
                context_text = build_context_from_chunks(candidates, max_chunks=MAX_CHUNKS_FINAL)
                # Short preview to keep logs readable
                prev = "\n".join(context_text.splitlines()[:20])
                log(f"[{iq_id}] Context preview:\n{prev}")

            t_ans = time.time()
            answer = agent2_answer(used_q, context_text, guidance=None, output_lang=user_lang)
            log(f"[{iq_id}] Answer generated in {(time.time()-t_ans)*1000:.0f} ms")

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

            iq_runs.append({
                "step_index": idx,
                "iq_id": iq_id,
                "planned_question": planned_q,
                "used_question": used_q,
                "answer": answer,
                "retrieval_meta": {
                    "num_candidates": len(candidates) if candidates else 0,
                    "top_chunks": meta_top
                }
            })

        # Final answer is the last IQ's answer
        final_answer = iq_runs[-1]["answer"] if iq_runs else ""

        # Summary
        log("=== Agentic RAG (Sequential IQs) summary ===")
        log(f"- IQs executed: {len(iq_runs)}")
        for run in iq_runs:
            log(f"  • [{run['iq_id']}] {run['used_question'][:120]}...")  # short preview
        log("=== Final Answer ===")
        log(final_answer or "")
        log(f"Logs saved to: {log_file}")

        return {
            "final_answer": final_answer or "",
            "iq_plan": iq_plan,
            "iq_runs": iq_runs,
            "iterations": len(iq_runs),
            "log_file": str(log_file),
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