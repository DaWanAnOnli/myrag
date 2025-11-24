#!/usr/bin/env python3
# agentic_rag.py
# Agentic RAG with Answer Judge + Query Modifier (iterative):
# - Detect user language
# - Retrieval (embed -> vector search over chunks)
# - Answerer agent generates an answer from retrieved context
# - Answer Judge agent: inspects the current query, the generated answer, and prior query–answer–feedback history,
#   decides if the answer is acceptable; if not, diagnoses the problem and suggests how to modify the query
# - Query Modifier agent: takes the query, the answer, the judge's feedback, and the history; outputs a modified query
# - Loop continues until the judge deems the answer acceptable OR the maximum number of iterations is reached
# - If the iteration limit is reached, skip judge/modifier and return the most recent answer (best effort)
# - The Answerer uses only the provided context to answer
# - Entity extractor (Agent 1 + 1b) runs each iteration on the current query before retrieval
# - Logging to a timestamped file (and console); logs include iteration markers like 1/4, 2/4, etc.
# - Rate limiting: LLM_CALLS_PER_MINUTE applies ONLY to text generation (not embeddings). Thread-safe.

import os, time, json, pickle, re, random, threading
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
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
TOP_K_CHUNKS = int(os.getenv("TOP_K_CHUNKS", "40"))                # initial k from vector index
MAX_CHUNKS_FINAL = int(os.getenv("MAX_CHUNKS_FINAL", "40"))        # chunks kept for final context
CHUNK_TEXT_CLAMP = int(os.getenv("CHUNK_TEXT_CLAMP", "1000000"))   # max chars per chunk included in context

# Answerer
ANSWER_MAX_TOKENS = int(os.getenv("ANSWER_MAX_TOKENS", "4096"))

# Iterative Answer Judge/Modifier settings
MAX_ANSWER_JUDGE_ITERATIONS = int(os.getenv("MAX_ANSWER_JUDGE_ITERATIONS", "2"))   # hard cap on loops
ANSWER_JUDGE_MAX_TOKENS = int(os.getenv("ANSWER_JUDGE_MAX_TOKENS", "4096"))
QUERY_MODIFIER_MAX_TOKENS = int(os.getenv("QUERY_MODIFIER_MAX_TOKENS", "4096"))

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
    return random.uniform(90.0, 150.0)

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
            wait_s = random.uniform(5.0, 20.0)
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

# ----------------- Agent 1 / 1b (entity extractor; optional signals) -----------------
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

# ----------------- Answer Judge -----------------
ANSWER_JUDGE_SCHEMA = {
  "type": "object",
  "properties": {
    "decision": {"type": "string", "enum": ["acceptable", "insufficient"]},
    "reasoning": {"type": "string"},
    "problem": {"type": "string"},
    "suggested_solution": {"type": "string"}
  },
  "required": ["decision", "reasoning"]
}

def answer_judge(current_query: str, current_answer: str, qaf_history: List[Dict[str, Any]], output_lang: str = "en") -> Dict[str, Any]:
    """
    Decide whether the generated answer is acceptable for the query.
    If insufficient, explain the problem and suggest how to modify the query so retrieval/answering improves.
    """
    # Format prior query–answer–feedback triples
    hist_lines = []
    for i, item in enumerate(qaf_history, 1):
        qq = (item.get("query") or "").strip()
        qa = (item.get("answer") or "").strip()
        fb = (item.get("feedback") or {}) or {}
        pr = (fb.get("problem","") or "").strip()
        ss = (fb.get("suggested_solution","") or "").strip()
        hist_lines.append(
            f"[Iter {i}] Query: {qq}\n[Iter {i}] Answer: {qa}\n[Iter {i}] Feedback.problem: {pr}\n[Iter {i}] Feedback.suggested_solution: {ss}"
        )
    history_block = "\n".join(hist_lines) if hist_lines else "(no prior query–answer–feedback history)"

    instructions = (
        "You are the Answer Judge for a GraphRAG pipeline.\n"
        "- The pipeline embeds the query, performs vector similarity search over chunked legal documents, "
        "builds a context from top-k chunks, and the Answerer responds strictly from that context.\n"
        "- Your job: judge whether the current answer adequately addresses the current query (relevance, completeness, specificity, legal grounding). "
        "Consider whether a typical GraphRAG retrieval likely captured the necessary passages, and whether the answer shows signs of vagueness or lack of citations.\n"
        "- If acceptable: set decision='acceptable' and explain briefly why.\n"
        "- If insufficient: set decision='insufficient' and provide:\n"
        "  • problem: concise diagnosis (e.g., 'query ambiguous', 'missing statute number/article/year', 'wrong entity/jurisdiction/timeframe', 'needs synonyms/canonical name').\n"
        "  • suggested_solution: concrete query changes to improve retrieval (add identifiers, constrain timeframe/jurisdiction, disambiguate entities, add synonyms/terms).\n"
        "- Use prior query–answer–feedback history to avoid repeating failed strategies.\n"
        "- Respond in the same language as the user's question."
    )

    prompt = f"""
You are the Answer Judge.

Current query:
\"\"\"{current_query}\"\"\"

Generated answer:
\"\"\"{current_answer}\"\"\"

Prior query–answer–feedback history:
\"\"\"{history_block}\"\"\"

Instructions:
{instructions}

Return JSON with keys: decision ('acceptable'|'insufficient'), reasoning, and if insufficient also problem and suggested_solution.
"""
    est = estimate_tokens_for_text(prompt)
    log("=== Answer Judge ===")
    log("[Judge] Prompt:")
    log(prompt)
    log(f"[Judge] Prompt size: {len(prompt)} chars, est_tokens≈{est}")

    out = safe_generate_json(prompt, ANSWER_JUDGE_SCHEMA, temp=0.0) or {}
    decision = (out.get("decision") or "").strip().lower()
    reasoning = (out.get("reasoning") or "").strip()
    problem = (out.get("problem") or "").strip()
    solution = (out.get("suggested_solution") or "").strip()

    if decision not in ("acceptable","insufficient"):
        decision = "insufficient"  # conservative default
    log(f"[Judge] Decision: {decision.upper()} | Reasoning: {reasoning}")
    if decision == "insufficient":
        log(f"[Judge] Problem: {problem}")
        log(f"[Judge] Suggested solution: {solution}")

    return {"decision": decision, "reasoning": reasoning, "problem": problem, "suggested_solution": solution}

# ----------------- Query Modifier -----------------
QUERY_MODIFIER_SCHEMA = {
  "type": "object",
  "properties": {
    "modified_query": {"type": "string"},
    "notes": {"type": "string"}
  },
  "required": ["modified_query"]
}

def query_modifier(current_query: str, current_answer: str, judge_feedback: Dict[str, Any], qaf_history: List[Dict[str, Any]], output_lang: str = "en") -> Dict[str, Any]:
    """
    Modify/enrich the current query based on the answer judge's feedback and prior query–answer–feedback history.
    The modifier understands the GraphRAG pipeline and crafts a retrieval-friendly query.
    """
    # Format prior query–answer–feedback triples
    hist_lines = []
    for i, item in enumerate(qaf_history, 1):
        qq = (item.get("query") or "").strip()
        qa = (item.get("answer") or "").strip()
        fb = (item.get("feedback") or {}) or {}
        pr = (fb.get("problem","") or "").strip()
        ss = (fb.get("suggested_solution","") or "").strip()
        hist_lines.append(
            f"[Iter {i}] Query: {qq}\n[Iter {i}] Answer: {qa}\n[Iter {i}] Feedback.problem: {pr}\n[Iter {i}] Feedback.suggested_solution: {ss}"
        )
    history_block = "\n".join(hist_lines) if hist_lines else "(no prior query–answer–feedback history)"

    problem = (judge_feedback.get("problem") or "").strip()
    solution = (judge_feedback.get("suggested_solution") or "").strip()

    instructions = (
        "You are the Query Modifier for a GraphRAG pipeline.\n"
        "- The pipeline embeds the query and runs vector search over chunked legal texts; better queries produce better chunks.\n"
        "- Use the judge's problem diagnosis and suggested_solution plus prior history to rewrite the query so retrieval improves.\n"
        "- Keep the user's intent intact. Prefer adding exact identifiers (law number, article, year), synonyms, constraints (jurisdiction/timeframe), "
        "and canonical names. Be precise, avoid hallucinating facts. If information is missing, propose a neutral phrasing that still narrows retrieval.\n"
        "- Respond in the same language as the user's question."
    )

    prompt = f"""
You are the Query Modifier.

Current query:
\"\"\"{current_query}\"\"\"

Current answer:
\"\"\"{current_answer}\"\"\"

Answer judge feedback:
- problem: {problem}
- suggested_solution: {solution}

Prior query–answer–feedback history:
\"\"\"{history_block}\"\"\"

Instructions:
{instructions}

Return JSON with:
- modified_query: the improved query (same language as the user)
- notes: brief rationale of the changes (optional)
"""
    est = estimate_tokens_for_text(prompt)
    log("=== Query Modifier ===")
    log("[Modifier] Prompt:")
    log(prompt)
    log(f"[Modifier] Prompt size: {len(prompt)} chars, est_tokens≈{est}")

    out = safe_generate_json(prompt, QUERY_MODIFIER_SCHEMA, temp=0.0) or {}
    modified_query = (out.get("modified_query") or "").strip()
    notes = (out.get("notes") or "").strip()
    if not modified_query:
        modified_query = current_query  # fallback to current query if modification fails
        notes = notes or "(No modification produced; using current query as-is.)"
    log(f"[Modifier] Modified query: {modified_query}")
    if notes:
        log(f"[Modifier] Notes: {notes}")
    return {"modified_query": modified_query, "notes": notes}

# ----------------- Agentic RAG main (Answer Judge + Modifier loop) -----------------
def agentic_rag(query_original: str) -> Dict[str, Any]:
    global _LOGGER
    ts_name = make_timestamp_name()
    log_file = Path.cwd() / f"{ts_name}.txt"
    _LOGGER = FileLogger(log_file, also_console=True)

    try:
        log("=== Agentic RAG (Answer Judge + Modifier) run started ===")
        log(f"Log file: {log_file}")
        log(f"Original Query: {query_original}")
        log(f"Parameters: TOP_K_CHUNKS={TOP_K_CHUNKS}, MAX_CHUNKS_FINAL={MAX_CHUNKS_FINAL}, CHUNK_TEXT_CLAMP={CHUNK_TEXT_CLAMP}, "
            f"ANSWER_MAX_TOKENS={ANSWER_MAX_TOKENS}, MAX_ANSWER_JUDGE_ITERATIONS={MAX_ANSWER_JUDGE_ITERATIONS}, "
            f"ANSWER_JUDGE_MAX_TOKENS={ANSWER_JUDGE_MAX_TOKENS}, QUERY_MODIFIER_MAX_TOKENS={QUERY_MODIFIER_MAX_TOKENS}, "
            f"LLM_CALLS_PER_MINUTE={LLM_CALLS_PER_MINUTE}")

        user_lang = detect_user_language(query_original)
        log(f"[Language] Detected user language: {user_lang}")

        # Iterative loop state
        current_query = query_original.strip()
        qaf_history: List[Dict[str, Any]] = []     # list of {iteration, query, answer, feedback{problem,suggested_solution}}
        iteration_runs: List[Dict[str, Any]] = []  # per-iteration trace
        final_answer = ""

        for it in range(1, MAX_ANSWER_JUDGE_ITERATIONS + 1):
            log(f"--- Iteration {it}/{MAX_ANSWER_JUDGE_ITERATIONS} ---")
            log(f"[Iter {it}] Current query: {current_query}")

            # Entity extraction (Agent 1 & 1b) before retrieval each iteration
            t_a = time.time()
            _ = agent1_extract_entities_predicates(current_query)
            _ = agent1b_extract_query_triples(current_query)
            log(f"[Iter {it}] Entity/Triple extraction in {(time.time()-t_a)*1000:.0f} ms")

            # Retrieval for current query
            t_e = time.time()
            q_emb = embed_text(current_query)
            log(f"[Iter {it}] Embedded query in {(time.time()-t_e)*1000:.0f} ms")

            t_v = time.time()
            candidates = vector_query_chunks(q_emb, k=TOP_K_CHUNKS)
            log(f"[Iter {it}] Vector search returned {len(candidates)} candidates in {(time.time()-t_v)*1000:.0f} ms")

            if not candidates:
                context_text = "Tidak ada potongan teks yang ditemukan." if user_lang == "id" else "No relevant chunks found."
            else:
                context_text = build_context_from_chunks(candidates, max_chunks=MAX_CHUNKS_FINAL)
                preview = "\n".join(context_text.splitlines()[:20])
                log(f"[Iter {it}] Context preview:\n{preview}")

            # Answerer
            t_ans = time.time()
            answer = agent2_answer(current_query, context_text, guidance=None, output_lang=user_lang)
            log(f"[Iter {it}] Answer generated in {(time.time()-t_ans)*1000:.0f} ms")

            # If iteration limit reached, skip judge/modifier and finalize this answer
            if it >= MAX_ANSWER_JUDGE_ITERATIONS:
                log(f"[Iter {it}] Iteration limit reached. Skipping Answer Judge/Query Modifier and finalizing current answer.")
                final_answer = answer
                meta_top = []
                for c in (candidates[:5] if candidates else []):
                    meta_top.append({
                        "document_id": c.get("document_id"),
                        "chunk_id": c.get("chunk_id"),
                        "uu_number": c.get("uu_number"),
                        "pages": c.get("pages"),
                        "score": c.get("score"),
                    })
                iteration_runs.append({
                    "iteration": it,
                    "query": current_query,
                    "answer": answer,
                    "judge": {"decision": "skipped_limit"},
                    "modified_query": None,
                    "retrieval_meta": {
                        "num_candidates": len(candidates) if candidates else 0,
                        "top_chunks": meta_top
                    }
                })
                break

            # Answer Judge
            judge = answer_judge(current_query, answer, qaf_history, output_lang=user_lang)
            decision = judge.get("decision","insufficient")

            if decision == "acceptable":
                log(f"[Iter {it}] Judge deemed answer ACCEPTABLE. Finalizing.")
                final_answer = answer
                meta_top = []
                for c in (candidates[:5] if candidates else []):
                    meta_top.append({
                        "document_id": c.get("document_id"),
                        "chunk_id": c.get("chunk_id"),
                        "uu_number": c.get("uu_number"),
                        "pages": c.get("pages"),
                        "score": c.get("score"),
                    })
                iteration_runs.append({
                    "iteration": it,
                    "query": current_query,
                    "answer": answer,
                    "judge": judge,
                    "modified_query": None,
                    "retrieval_meta": {
                        "num_candidates": len(candidates) if candidates else 0,
                        "top_chunks": meta_top
                    }
                })
                break
            else:
                # Insufficient: store Q-A-F, then modify query
                fb = {"problem": judge.get("problem",""), "suggested_solution": judge.get("suggested_solution","")}
                qaf_history.append({"iteration": it, "query": current_query, "answer": answer, "feedback": fb})
                log(f"[Iter {it}] Judge indicates insufficiency; invoking Query Modifier.")
                mod = query_modifier(current_query, answer, fb, qaf_history, output_lang=user_lang)
                new_query = mod.get("modified_query", "").strip() or current_query
                notes = mod.get("notes","")
                log(f"[Iter {it}] Modified query -> {new_query}")
                if notes:
                    log(f"[Iter {it}] Modifier notes: {notes}")

                # Trace this iteration (no final answer yet)
                iteration_runs.append({
                    "iteration": it,
                    "query": current_query,
                    "answer": answer,
                    "judge": judge,
                    "modified_query": new_query,
                    "retrieval_meta": {
                        "num_candidates": len(candidates) if candidates else 0
                    }
                })

                # Prepare for next iteration
                current_query = new_query
                continue

        # Summary
        log("=== Agentic RAG (Answer Judge + Modifier) summary ===")
        log(f"- Iterations executed: {len(iteration_runs)}")
        for run in iteration_runs:
            q_preview = (run.get("query","") or "")[:120]
            status = (run.get("judge",{}) or {}).get("decision","")
            log(f"  • Iter {run['iteration']}: decision={status} | query='{q_preview}...'")
        log("=== Final Answer ===")
        log(final_answer or "")
        log(f"Logs saved to: {log_file}")

        return {
            "final_answer": final_answer or "",
            "iterations": len(iteration_runs),
            "iteration_runs": iteration_runs,
            "query_answer_feedback_history": qaf_history,
            "log_file": str(log_file),
            "judge_reports": []  # kept for compatibility; empty
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