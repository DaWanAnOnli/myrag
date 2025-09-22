#!/usr/bin/env python3
# agentic_rag.py
# Agentic RAG with Context Judge and Query Modifier
#
# Pipeline:
# - Detect user language
# - Iterative retrieval-improvement loop (hardcapped by MAX_CTX_ITERS):
#     * Embed + retrieve chunks from Neo4j
#     * Context Judge evaluates if retrieved context is sufficient/relevant for the query,
#       using the query, the retrieved chunk previews, and the history of query–feedback pairs.
#     * If sufficient OR iteration cap reached: proceed to Answerer with current context.
#     * Else: Query Modifier takes the query + judge feedback + history and outputs a modified query.
#       Store the query–feedback pair and iterate again with the modified query.
# - Answerer (Agent 2) produces the final answer based on the last context; optionally evaluated by Judge (Agent 3).
#
# Notes:
# - All query–feedback pairs are stored and passed to the Context Judge and Query Modifier each iteration.
# - If the context-judging loop hits the hard limit, we proceed to Answerer even if the Context Judge says the context is insufficient.

import os, time, json, re, random, threading
from pathlib import Path
from typing import Any, Dict, List, Optional

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
TOP_K_CHUNKS = int(os.getenv("TOP_K_CHUNKS", "30"))              # initial k from vector index
MAX_CHUNKS_FINAL = int(os.getenv("MAX_CHUNKS_FINAL", "40"))      # chunks kept for final context (Answerer)
CHUNK_TEXT_CLAMP = int(os.getenv("CHUNK_TEXT_CLAMP", "1000000")) # max chars per chunk included in context for Answerer

# Answerer loop (optional inner refinement via Agent 3; kept from original)
ANSWER_MAX_TOKENS = int(os.getenv("ANSWER_MAX_TOKENS", "4096"))
MAX_ITERS = int(os.getenv("MAX_ITERS", "3"))

# New: Context-judge iteration cap (hardcoded as requested)
MAX_CTX_ITERS = 5  # maximum retrieval-improvement iterations before forcing Answerer

# New: Context Judge preview sizing
CTX_JUDGE_MAX_CHUNKS = 15        # max chunks to preview for the Context Judge
CTX_JUDGE_CHUNK_CLAMP = 1200     # max chars per chunk in judge preview
FEEDBACK_HISTORY_MAX_ITEMS = 8   # max previous query-feedback records included in judge/modifier prompts
FEEDBACK_HISTORY_CLAMP = 1200    # max chars per answer/rationale in history

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
                self._fh.flush()
                self._fh.close()
        except Exception:
            pass

_LOGGER: Optional[FileLogger] = None
def log(msg: str = ""):
    if _LOGGER is not None:
        _LOGGER.log(msg)
    else:
        print(msg)

def log_with_prefix(prefix: str, msg: str = ""):
    if not isinstance(msg, str):
        log(f"{prefix} {msg}")
        return
    lines = msg.splitlines() or [""]
    for line in lines:
        log(f"{prefix} {line}")

def make_timestamp_name() -> str:
    t = time.time()
    base = time.strftime("%Y%m%d-%H%M%S", time.localtime(t))
    ms = int((t % 1) * 1000)
    return f"{base}-{ms:03d}"

# ----------------- Utilities -----------------
def estimate_tokens_for_text(text: str) -> int:
    return max(1, int(len(text) / 4))

def clamp(s: Optional[str], n: int) -> str:
    t = (s or "").strip()
    return t[:n]

def _rand_wait_seconds() -> float:
    return random.uniform(5.0, 20.0)

def _api_call_with_retry(func, *args, **kwargs):
    while True:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            wait_s = _rand_wait_seconds()
            log(f"[Retry] API call failed: {e}. Retrying in {wait_s:.1f}s.")
            time.sleep(wait_s)

# ----------------- Embedding + Neo4j -----------------
def embed_text(text: str) -> List[float]:
    res = _api_call_with_retry(genai.embed_content, model=EMBED_MODEL, content=text)
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
                if buf: return "\n".join(buf).strip()
    except Exception:
        pass
    return None

def safe_generate_json(prompt: str, schema: Dict[str, Any], temp: float = 0.0) -> Dict[str, Any]:
    cfg = GenerationConfig(temperature=temp, response_mime_type="application/json", response_schema=schema)
    resp = _api_call_with_retry(gen_model.generate_content, prompt, generation_config=cfg)
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
    resp = _api_call_with_retry(gen_model.generate_content, prompt, generation_config=cfg)
    text = extract_text_from_response(resp)
    if text: return text
    info = get_finish_info(resp)
    log(f"[LLM text warning] No text returned. Diagnostics: {info}")
    return f"(Model returned no text. finish_info={info})"

# ----------------- Agent 1 / 1b (optional signals) -----------------
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
    log("\n[Agent 1] Prompt:")
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
    log("\n[Agent 1b] Prompt:")
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

# ----------------- Context builders -----------------
def build_context_from_chunks(chunks: List[Dict[str, Any]], max_chunks: int) -> str:
    chosen = chunks[:max_chunks]
    lines = ["Potongan teks terkait (chunk):"]
    for i, c in enumerate(chosen, 1):
        doc = c.get("document_id")
        cid = c.get("chunk_id")
        uu = c.get("uu_number") or ""
        pg = c.get("pages")
        txt = clamp(c.get("content") or "", CHUNK_TEXT_CLAMP)
        lines.append(f"[Chunk {i}] doc={doc} chunk={cid} | {uu} | pages={pg} | score={c.get('score'):.3f}\n{txt}")
    return "\n".join(lines)

def build_judge_context_preview(chunks: List[Dict[str, Any]], max_chunks: int, per_chunk_clamp: int) -> str:
    chosen = chunks[:max_chunks]
    lines = ["Retrieved chunk preview (for Context Judge):"]
    for i, c in enumerate(chosen, 1):
        doc = c.get("document_id")
        cid = c.get("chunk_id")
        uu = c.get("uu_number") or ""
        pg = c.get("pages")
        snippet = clamp(c.get("content") or "", per_chunk_clamp)
        lines.append(f"[C{i}] doc={doc} chunk={cid} | {uu} | pages={pg} | score={c.get('score'):.3f}\n{snippet}")
    return "\n".join(lines)

def describe_pipeline(user_lang: str) -> str:
    # Short brief to inform Judge/Modifier how the pipeline works
    bullet = "-" if user_lang == "en" else "-"
    return (
        f"{bullet} Embedding model: {EMBED_MODEL}\n"
        f"{bullet} Vector search over Neo4j index 'chunk_embedding_index' using db.index.vector.queryNodes\n"
        f"{bullet} Retrieval params: TOP_K_CHUNKS={TOP_K_CHUNKS}, MAX_CHUNKS_FINAL={MAX_CHUNKS_FINAL}\n"
        f"{bullet} Context = top chunks with metadata (document_id, uu_number, pages, score)\n"
        f"{bullet} Common failure modes: query too broad; missing identifiers (UU/PP/Permen numbers, article/ayat); wrong language/terminology; no timeframe/agency; synonyms/aliases not included; overly strict or too loose constraints.\n"
        f"{bullet} Remedies: add law number/article; specify agency/sector; include date range; use Indonesian legal terms; add synonyms/aliases; clarify scope.\n"
        f"{bullet} Language handling: respond in user's language; retrieval language may matter.\n"
    )

# ----------------- Agent 2 (Answerer) -----------------
def agent2_answer(query_original: str, context: str, guidance: Optional[str], output_lang: str = "id") -> str:
    instructions = (
        "Answer concisely and accurately based strictly on the provided context. "
        "Cite UU/Article references when they are clear. "
        "Respond in the same language as the user's question."
    )
    guidance_text = guidance.strip() if isinstance(guidance, str) and guidance.strip() else "(No additional guidance from the Judge.)"
    prompt = f"""
You are Agent 2 (Answerer). Provide an answer based on the context only.

Core instructions:
{instructions}

Additional guidance from Judge (if any):
\"\"\"{guidance_text}\"\"\"

Original user question:
\"\"\"{query_original}\"\"\"

Context:
\"\"\"{context}\"\"\"
"""
    est = estimate_tokens_for_text(prompt)
    log("\n[Agent 2] Prompt:")
    log(prompt)
    log(f"[Agent 2] Prompt size: {len(prompt)} chars, est_tokens≈{est}")
    answer = safe_generate_text(prompt, max_tokens=ANSWER_MAX_TOKENS, temperature=0.2)
    log("[Agent 2] Answer:")
    log(answer)
    return answer

# ----------------- Agent 3 (Answer-quality Judge; optional) -----------------
JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "is_sufficient": {"type": "boolean"},
        "rationale": {"type": "string"},
        "issues_found": {"type": "array", "items": {"type": "string"}},
        "rewritten_query": {"type": "string"},
        "guidance_next": {"type": "string"}
    },
    "required": ["is_sufficient","rationale"]
}

def agent3_judge(query_original: str, intermediate_answer: str, context_full: str, output_lang: str = "id") -> Dict[str, Any]:
    prompt = f"""
You are Agent 3 (Judge). Evaluate whether the answer is sufficient and grounded in the context.

Sufficiency criteria:
- Correctly addresses the user's question.
- Grounded in the provided context (chunks).
- Cites UU/Article references when conclusive.
- No fabrication beyond the context.

Original user question:
\"\"\"{query_original}\"\"\"

Context (full chunk text):
\"\"\"{context_full}\"\"\"

Answer:
\"\"\"{intermediate_answer}\"\"\"
"""
    est = estimate_tokens_for_text(prompt)
    log("\n[Agent 3] Prompt:")
    log(prompt)
    log(f"[Agent 3] Prompt size: {len(prompt)} chars, est_tokens≈{est}")
    out = safe_generate_json(prompt, JUDGE_SCHEMA, temp=0.0) or {}
    out["is_sufficient"] = bool(out.get("is_sufficient", False))
    out["rationale"] = out.get("rationale", "")
    if "issues_found" not in out or not isinstance(out["issues_found"], list): out["issues_found"] = []
    out["rewritten_query"] = (out.get("rewritten_query") or "").strip()
    out["guidance_next"] = (out.get("guidance_next") or "").strip()
    log("[Agent 3] Judgment output:")
    log(json.dumps(out, ensure_ascii=False, indent=2))
    return out

# ----------------- Context Judge (new) -----------------
CONTEXT_JUDGE_SCHEMA = {
  "type": "object",
  "properties": {
    "is_sufficient": {"type": "boolean"},
    "rationale": {"type": "string"},
    "problems": {"type": "array", "items": {"type": "string"}},
    "suggestions": {"type": "array", "items": {"type": "string"}},
    "suggested_query_preview": {"type": "string"}
  },
  "required": ["is_sufficient", "rationale"]
}

def format_history_for_prompts(history: List[Dict[str, Any]]) -> str:
    if not history:
        return "(none)"
    # Use only last FEEDBACK_HISTORY_MAX_ITEMS items
    recent = history[-FEEDBACK_HISTORY_MAX_ITEMS:]
    lines = []
    for rec in recent:
        it = rec.get("iteration")
        qb = clamp(rec.get("query_before", ""), 1500)
        j = rec.get("judge", {}) or {}
        j_prob = "; ".join(j.get("problems") or [])[:FEEDBACK_HISTORY_CLAMP]
        j_sugg = "; ".join(j.get("suggestions") or [])[:FEEDBACK_HISTORY_CLAMP]
        mq = clamp(rec.get("modified_query", "") or "(no modification)", 1500)
        notes = clamp(rec.get("modifier_notes", "") or "", 600)
        lines.append(f"- Iter {it}\n  Query: {qb}\n  Judge problems: {j_prob}\n  Judge suggestions: {j_sugg}\n  Modified query: {mq}\n  Notes: {notes}")
    return "\n".join(lines)

def context_judge_evaluate(current_query: str,
                           context_preview: str,
                           history: List[Dict[str, Any]],
                           user_lang: str) -> Dict[str, Any]:
    pipeline_brief = describe_pipeline(user_lang)
    history_text = format_history_for_prompts(history)
    instructions = (
        "You are the Context Judge. Decide if the retrieved chunks are sufficient and relevant to answer the query. "
        "If not sufficient, diagnose why retrieval is weak and propose concrete, retrieval-aware improvements to the query. "
        "You understand the GraphRAG pipeline described below (embedding, vector search, chunk metadata and limits)."
    )
    prompt = f"""
{instructions}

GraphRAG pipeline brief:
\"\"\"{pipeline_brief}\"\"\"

Current query:
\"\"\"{current_query}\"\"\"

Retrieved chunk preview:
\"\"\"{context_preview}\"\"\"

Previous query–feedback history:
\"\"\"{history_text}\"\"\"

Return JSON with:
- is_sufficient: boolean
- rationale: brief explanation
- problems: list of concrete issues (e.g., missing UU number/article, too broad scope, wrong language)
- suggestions: list of actionable changes to the query (add identifiers, synonyms, timeframes, agency names, etc.)
- suggested_query_preview: optional draft of a better query (do not invent facts)
"""
    est = estimate_tokens_for_text(prompt)
    log("\n[Context Judge] Prompt:")
    log(prompt)
    log(f"[Context Judge] Prompt size: {len(prompt)} chars, est_tokens≈{est}")
    out = safe_generate_json(prompt, CONTEXT_JUDGE_SCHEMA, temp=0.1) or {}
    # Normalize
    out["is_sufficient"] = bool(out.get("is_sufficient", False))
    out["rationale"] = clamp(out.get("rationale", "") or "", 2000)
    if not isinstance(out.get("problems"), list):
        out["problems"] = []
    if not isinstance(out.get("suggestions"), list):
        out["suggestions"] = []
    out["suggested_query_preview"] = clamp(out.get("suggested_query_preview", "") or "", 2000)
    log("[Context Judge] Output:")
    log(json.dumps(out, ensure_ascii=False, indent=2))
    return out

# ----------------- Query Modifier (new) -----------------
QUERY_MODIFIER_SCHEMA = {
  "type": "object",
  "properties": {
    "modified_query": {"type": "string"},
    "notes": {"type": "string"}
  },
  "required": ["modified_query"]
}

def query_modifier_apply_feedback(original_query: str,
                                  current_query: str,
                                  judge_feedback: Dict[str, Any],
                                  history: List[Dict[str, Any]],
                                  user_lang: str) -> Dict[str, Any]:
    pipeline_brief = describe_pipeline(user_lang)
    history_text = format_history_for_prompts(history)
    problems = judge_feedback.get("problems") or []
    suggestions = judge_feedback.get("suggestions") or []
    suggested_preview = judge_feedback.get("suggested_query_preview") or ""
    rules = (
        "Modify the current query to improve retrieval according to the judge feedback. "
        "Do not introduce new factual claims; you may add identifiers, synonyms, translations, date ranges, or scope qualifiers only as suggested. "
        "Keep the query self-contained, precise, and in the user's language."
    )
    prompt = f"""
You are the Query Modifier. You understand how this GraphRAG pipeline retrieves context. Apply the Context Judge's feedback to produce a better query.

GraphRAG pipeline brief:
\"\"\"{pipeline_brief}\"\"\"

Original user question:
\"\"\"{original_query}\"\"\"

Current query (to be modified):
\"\"\"{current_query}\"\"\"

Context Judge feedback:
- Problems: {problems}
- Suggestions: {suggestions}
- Suggested query preview (optional): \"\"\"{suggested_preview}\"\"\"

Previous query–feedback history:
\"\"\"{history_text}\"\"\"

Rules:
{rules}

Return JSON:
- modified_query: string (the improved query)
- notes: brief explanation of how you applied the suggestions
"""
    est = estimate_tokens_for_text(prompt)
    log("\n[Query Modifier] Prompt:")
    log(prompt)
    log(f"[Query Modifier] Prompt size: {len(prompt)} chars, est_tokens≈{est}")
    out = safe_generate_json(prompt, QUERY_MODIFIER_SCHEMA, temp=0.2) or {}
    modified_query = clamp(out.get("modified_query", "") or "", 3000)
    notes = clamp(out.get("notes", "") or "", 1000)
    if not modified_query:
        modified_query = current_query  # Fallback: keep current query
        notes = "Fallback: modifier returned empty; keeping current query."
    log("[Query Modifier] Output:")
    log(json.dumps({"modified_query": modified_query, "notes": notes}, ensure_ascii=False, indent=2))
    return {"modified_query": modified_query, "notes": notes}

# ----------------- Orchestrator -----------------
def agentic_rag(query_original: str) -> Dict[str, Any]:
    global _LOGGER
    ts_name = make_timestamp_name()
    log_file = Path.cwd() / f"{ts_name}.txt"
    _LOGGER = FileLogger(log_file, also_console=True)

    try:
        log("=== Agentic RAG (Context Judge + Query Modifier) run started ===")
        log(f"Log file: {log_file}")
        log(f"Original Query: {query_original}")
        log("Parameters:")
        log(f"- TOP_K_CHUNKS={TOP_K_CHUNKS}, MAX_CHUNKS_FINAL={MAX_CHUNKS_FINAL}, CHUNK_TEXT_CLAMP={CHUNK_TEXT_CLAMP}")
        log(f"- ANSWER_MAX_TOKENS={ANSWER_MAX_TOKENS}, MAX_ITERS={MAX_ITERS}")
        log(f"- MAX_CTX_ITERS (hardcoded)={MAX_CTX_ITERS}, CTX_JUDGE_MAX_CHUNKS={CTX_JUDGE_MAX_CHUNKS}, CTX_JUDGE_CHUNK_CLAMP={CTX_JUDGE_CHUNK_CLAMP}")

        user_lang = detect_user_language(query_original)
        log(f"[Language] Detected user language: {user_lang}")

        current_query = query_original
        query_feedback_history: List[Dict[str, Any]] = []

        final_context_text = ""
        final_candidates: List[Dict[str, Any]] = []
        final_query_used = current_query
        context_judge_last: Dict[str, Any] = {}

        for ctx_iter in range(1, MAX_CTX_ITERS + 1):
            log(f"\n=== Retrieval-Improvement Iteration {ctx_iter}/{MAX_CTX_ITERS} ===")
            log(f"[Iter {ctx_iter}] Current query: {current_query}")

            # Step 0: Embed
            t0 = time.time()
            q_emb = embed_text(current_query)
            log(f"[Iter {ctx_iter}] Embedded query in {(time.time()-t0)*1000:.0f} ms")

            # Step 1: Optional entity/triple extraction
            t1 = time.time()
            _ = agent1_extract_entities_predicates(current_query)
            _ = agent1b_extract_query_triples(current_query)
            log(f"[Iter {ctx_iter}] Entity/Triple extraction in {(time.time()-t1)*1000:.0f} ms")

            # Step 2: Retrieval
            t2 = time.time()
            candidates = vector_query_chunks(q_emb, k=TOP_K_CHUNKS)
            log(f"[Iter {ctx_iter}] Vector search returned {len(candidates)} candidates in {(time.time()-t2)*1000:.0f} ms")

            # Build previews/contexts
            judge_preview = build_judge_context_preview(candidates, max_chunks=CTX_JUDGE_MAX_CHUNKS, per_chunk_clamp=CTX_JUDGE_CHUNK_CLAMP)
            answer_context = build_context_from_chunks(candidates, max_chunks=MAX_CHUNKS_FINAL)

            # Step 3: Context Judge
            judge_out = context_judge_evaluate(current_query, judge_preview, query_feedback_history, user_lang)
            context_judge_last = judge_out

            # Store preliminary history record (before modification)
            history_record = {
                "iteration": ctx_iter,
                "query_before": current_query,
                "judge": judge_out
            }

            if judge_out.get("is_sufficient", False):
                log(f"[Iter {ctx_iter}] Context judged sufficient. Proceeding to Answerer.")
                final_context_text = answer_context
                final_candidates = candidates
                final_query_used = current_query
                query_feedback_history.append(history_record)
                break

            # If insufficient and we still have iterations left, apply Query Modifier
            if ctx_iter < MAX_CTX_ITERS:
                qm = query_modifier_apply_feedback(query_original, current_query, judge_out, query_feedback_history, user_lang)
                modified_query = qm.get("modified_query") or current_query
                history_record["modified_query"] = modified_query
                history_record["modifier_notes"] = qm.get("notes", "")
                query_feedback_history.append(history_record)
                # Update query and continue loop
                current_query = modified_query
                continue
            else:
                # Iteration cap reached: proceed to Answerer anyway (even if insufficient)
                log(f"[Iter {ctx_iter}] Iteration cap reached. Proceeding to Answerer regardless of sufficiency.")
                final_context_text = answer_context
                final_candidates = candidates
                final_query_used = current_query
                history_record["modified_query"] = current_query
                history_record["modifier_notes"] = "(No modification; iteration cap reached)"
                query_feedback_history.append(history_record)
                break

        # Safety fallback if no iterations ran (should not happen)
        if not final_context_text:
            # Use last known retrieval (if any), else minimal message
            final_context_text = build_context_from_chunks(final_candidates, max_chunks=MAX_CHUNKS_FINAL) if final_candidates else (
                "Tidak ada potongan teks yang ditemukan." if user_lang == "id" else "No relevant chunks found."
            )

        # Answerer phase (single pass; optional answer-quality Judge)
        final_answer = agent2_answer(query_original, final_context_text, guidance=None, output_lang=user_lang)
        answer_quality_judge = agent3_judge(query_original, final_answer, final_context_text, output_lang=user_lang)

        # Summary
        log("\n=== Context-Judge RAG summary ===")
        log(f"- Retrieval-improvement iterations used: {len(query_feedback_history)} (hard cap {MAX_CTX_ITERS})")
        log(f"- Final query used for retrieval: {final_query_used}")
        log("\n=== Final Answer ===")
        log(final_answer)
        log(f"\nLogs saved to: {log_file}")

        return {
            "final_answer": final_answer,
            "final_query_used": final_query_used,
            "iterations_ctx": len(query_feedback_history),
            "context_judge_last": context_judge_last,
            "answer_quality_judge": answer_quality_judge,
            "query_feedback_history": query_feedback_history,
            "log_file": str(log_file)
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
            result = agentic_rag(user_query)
            print("\n----- FINAL ANSWER -----\n")
            print(result["final_answer"])
    finally:
        try:
            driver.close()
        except Exception:
            pass