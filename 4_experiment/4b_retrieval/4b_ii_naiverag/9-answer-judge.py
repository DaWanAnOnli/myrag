#!/usr/bin/env python3
# agentic_rag.py
# Agentic RAG with Answer Judge (post-answer) and Query Modifier loop
#
# Pipeline per iteration (hardcapped by MAX_QA_JUDGE_ITERS):
#   - Embed current query and retrieve chunks from Neo4j
#   - Build context from chunks
#   - Answerer generates an answer to the current query using only the retrieved context
#   - Answer Judge evaluates the answer quality (given the query and QA-feedback history)
#       * If sufficient: return the answer as final
#       * If not: produce problems + retrieval-aware suggestions to improve the query
#   - Query Modifier applies the judge’s feedback to produce a modified query for next iteration
#   - Store query–answer–feedback triples in history
# If iteration cap is reached, return the latest answer even if judged insufficient.

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
MAX_CHUNKS_FINAL = int(os.getenv("MAX_CHUNKS_FINAL", "40"))      # chunks kept for final context
CHUNK_TEXT_CLAMP = int(os.getenv("CHUNK_TEXT_CLAMP", "1000000")) # max chars per chunk included in context

# Answerer generation
ANSWER_MAX_TOKENS = int(os.getenv("ANSWER_MAX_TOKENS", "4096"))

# New: Judge-guided outer loop (hardcoded cap as requested)
MAX_QA_JUDGE_ITERS = 5

# Prompt size controls
CTX_PREVIEW_CHUNKS = 20         # how many chunks to preview in logs (not sent to LLM)
CTX_PREVIEW_CLAMP = 800         # max chars per chunk in preview (logs)
QA_HISTORY_MAX_ITEMS = 8        # max previous QA-feedback items included in prompts
QA_HISTORY_CLAMP_ANSWER = 6000  # clamp stored answer in history prompts
QA_HISTORY_CLAMP_TEXT = 1200    # clamp rationale/notes in history prompts

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
    for line in (msg.splitlines() or [""]):
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
                    if isinstance(t, str):
                        buf.append(t)
                if buf:
                    return "\n".join(buf).strip()
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
    if text:
        return text
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

def build_context_preview(chunks: List[Dict[str, Any]], max_chunks: int, per_chunk_clamp: int) -> str:
    chosen = chunks[:max_chunks]
    lines = ["[Context preview]"]
    for i, c in enumerate(chosen, 1):
        doc = c.get("document_id")
        cid = c.get("chunk_id")
        uu = c.get("uu_number") or ""
        pg = c.get("pages")
        snippet = clamp(c.get("content") or "", per_chunk_clamp)
        lines.append(f"[C{i}] doc={doc} chunk={cid} | {uu} | pages={pg} | score={c.get('score'):.3f}\n{snippet}")
    return "\n".join(lines)

def describe_pipeline(user_lang: str) -> str:
    # Brief given to Judge/Modifier explaining the pipeline mechanics and common issues
    dash = "-"  # keep simple to avoid locale complexity
    return (
        f"{dash} Embedding model: {EMBED_MODEL}\n"
        f"{dash} Vector search over Neo4j index 'chunk_embedding_index' (db.index.vector.queryNodes)\n"
        f"{dash} Retrieval params: TOP_K_CHUNKS={TOP_K_CHUNKS}, MAX_CHUNKS_FINAL={MAX_CHUNKS_FINAL}\n"
        f"{dash} Chunks include metadata (document_id, uu_number, pages, score)\n"
        f"{dash} Common failure modes: query too broad; missing identifiers (UU/PP/Permen numbers, article/ayat); wrong language/terminology; missing timeframe/agency; synonyms/aliases not included; over/under constrained queries.\n"
        f"{dash} Remedies: add law/article numbers; specify agency/sector; include date range; use Indonesian legal terms; add synonyms/aliases; clarify scope.\n"
        f"{dash} Answerer only uses retrieved context; modifying the query changes retrieval.\n"
    )

# ----------------- Agent 2 (Answerer) -----------------
def agent2_answer(query_for_this_iter: str, context: str, guidance: Optional[str], output_lang: str = "id") -> str:
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

Original query for this iteration:
\"\"\"{query_for_this_iter}\"\"\"

Additional guidance:
\"\"\"{guidance_text}\"\"\"

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

# ----------------- Answer Judge (post-answer, controls loop) -----------------
ANSWER_JUDGE_SCHEMA = {
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

def format_qa_history_for_prompts(history: List[Dict[str, Any]]) -> str:
    if not history:
        return "(none)"
    recent = history[-QA_HISTORY_MAX_ITEMS:]
    lines = []
    for rec in recent:
        it = rec.get("iteration")
        qb = clamp(rec.get("query_before", ""), 1500)
        ans = clamp(rec.get("answer", ""), QA_HISTORY_CLAMP_ANSWER)
        j = rec.get("judge", {}) or {}
        j_r = clamp(j.get("rationale", "") or "", QA_HISTORY_CLAMP_TEXT)
        j_prob = "; ".join(j.get("problems") or [])[:QA_HISTORY_CLAMP_TEXT]
        j_sugg = "; ".join(j.get("suggestions") or [])[:QA_HISTORY_CLAMP_TEXT]
        mq = clamp(rec.get("modified_query", "") or "(no modification)", 1500)
        mn = clamp(rec.get("modifier_notes", "") or "", QA_HISTORY_CLAMP_TEXT)
        lines.append(
            f"- Iter {it}\n"
            f"  Query: {qb}\n"
            f"  Answer: {ans}\n"
            f"  Judge rationale: {j_r}\n"
            f"  Problems: {j_prob}\n"
            f"  Suggestions: {j_sugg}\n"
            f"  Modified query: {mq}\n"
            f"  Notes: {mn}"
        )
    return "\n".join(lines)

def answer_judge_evaluate(current_query: str,
                          generated_answer: str,
                          history: List[Dict[str, Any]],
                          user_lang: str) -> Dict[str, Any]:
    pipeline_brief = describe_pipeline(user_lang)
    history_text = format_qa_history_for_prompts(history)
    instructions = (
        "You are the Answer Judge. Decide whether the generated answer sufficiently addresses the current query. "
        "If not sufficient, diagnose why (likely retrieval/query issues) and propose concrete, retrieval-aware changes to the query. "
        "You understand the GraphRAG pipeline described below (embedding, vector search, chunk limits, and typical failure modes)."
    )
    prompt = f"""
{instructions}

GraphRAG pipeline brief:
\"\"\"{pipeline_brief}\"\"\"

Current query:
\"\"\"{current_query}\"\"\"

Generated answer:
\"\"\"{generated_answer}\"\"\"

Previous query–answer–feedback history:
\"\"\"{history_text}\"\"\"

Return JSON with:
- is_sufficient: boolean
- rationale: brief explanation
- problems: list of concrete issues
- suggestions: list of actionable query changes (identifiers, synonyms, timeframe, agency, scope, language)
- suggested_query_preview: optional draft; do not invent facts not implied by the query/history
"""
    est = estimate_tokens_for_text(prompt)
    log("\n[Answer Judge] Prompt:")
    log(prompt)
    log(f"[Answer Judge] Prompt size: {len(prompt)} chars, est_tokens≈{est}")
    out = safe_generate_json(prompt, ANSWER_JUDGE_SCHEMA, temp=0.1) or {}
    # Normalize
    out["is_sufficient"] = bool(out.get("is_sufficient", False))
    out["rationale"] = clamp(out.get("rationale", "") or "", 2000)
    if not isinstance(out.get("problems"), list):
        out["problems"] = []
    if not isinstance(out.get("suggestions"), list):
        out["suggestions"] = []
    out["suggested_query_preview"] = clamp(out.get("suggested_query_preview", "") or "", 2000)
    log("[Answer Judge] Output:")
    log(json.dumps(out, ensure_ascii=False, indent=2))
    return out

# ----------------- Query Modifier (applies judge feedback) -----------------
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
                                  generated_answer: str,
                                  judge_feedback: Dict[str, Any],
                                  history: List[Dict[str, Any]],
                                  user_lang: str) -> Dict[str, Any]:
    pipeline_brief = describe_pipeline(user_lang)
    history_text = format_qa_history_for_prompts(history)
    problems = judge_feedback.get("problems") or []
    suggestions = judge_feedback.get("suggestions") or []
    suggested_preview = judge_feedback.get("suggested_query_preview") or ""
    rules = (
        "Modify the current query to improve future retrieval in line with the judge's feedback. "
        "Do not invent new facts; you may add identifiers, synonyms, translations, date ranges, agency names, or scope qualifiers only as suggested. "
        "Keep the query self-contained, unambiguous, and in the user's language."
    )
    prompt = f"""
You are the Query Modifier. You understand the GraphRAG pipeline. Apply the Answer Judge's feedback to produce a better query.

GraphRAG pipeline brief:
\"\"\"{pipeline_brief}\"\"\"

Original user question:
\"\"\"{original_query}\"\"\"

Current query (to be modified):
\"\"\"{current_query}\"\"\"

Generated answer (for reference; do not copy content into the query):
\"\"\"{generated_answer}\"\"\"

Answer Judge feedback:
- Problems: {problems}
- Suggestions: {suggestions}
- Suggested query preview (optional): \"\"\"{suggested_preview}\"\"\"

Previous query–answer–feedback history:
\"\"\"{history_text}\"\"\"

Rules:
{rules}

Return JSON:
- modified_query: string (the improved query)
- notes: brief explanation of how suggestions were applied
"""
    est = estimate_tokens_for_text(prompt)
    log("\n[Query Modifier] Prompt:")
    log(prompt)
    log(f"[Query Modifier] Prompt size: {len(prompt)} chars, est_tokens≈{est}")
    out = safe_generate_json(prompt, QUERY_MODIFIER_SCHEMA, temp=0.2) or {}
    modified_query = clamp(out.get("modified_query", "") or "", 3000)
    notes = clamp(out.get("notes", "") or "", 1000)
    if not modified_query:
        modified_query = current_query
        notes = "Fallback: modifier returned empty; keeping current query."
    log("[Query Modifier] Output:")
    log(json.dumps({"modified_query": modified_query, "notes": notes}, ensure_ascii=False, indent=2))
    return {"modified_query": modified_query, "notes": notes}

# ----------------- Orchestrator -----------------
def agentic_rag(user_query: str) -> Dict[str, Any]:
    global _LOGGER
    ts_name = make_timestamp_name()
    log_file = Path.cwd() / f"{ts_name}.txt"
    _LOGGER = FileLogger(log_file, also_console=True)

    try:
        log("=== Agentic RAG (Answer Judge + Query Modifier) run started ===")
        log(f"Log file: {log_file}")
        log(f"Original User Query: {user_query}")
        log("Parameters:")
        log(f"- TOP_K_CHUNKS={TOP_K_CHUNKS}, MAX_CHUNKS_FINAL={MAX_CHUNKS_FINAL}, CHUNK_TEXT_CLAMP={CHUNK_TEXT_CLAMP}")
        log(f"- ANSWER_MAX_TOKENS={ANSWER_MAX_TOKENS}")
        log(f"- MAX_QA_JUDGE_ITERS (hardcoded)={MAX_QA_JUDGE_ITERS}")

        user_lang = detect_user_language(user_query)
        log(f"[Language] Detected user language: {user_lang}")

        current_query = user_query
        qa_feedback_history: List[Dict[str, Any]] = []

        final_answer: str = ""
        final_query_used: str = current_query
        iterations_used: int = 0

        for it in range(1, MAX_QA_JUDGE_ITERS + 1):
            iterations_used = it
            log(f"\n=== Iteration {it}/{MAX_QA_JUDGE_ITERS} ===")
            log(f"[Iter {it}] Current query: {current_query}")

            # Step 0: Embed
            t0 = time.time()
            q_emb = embed_text(current_query)
            log(f"[Iter {it}] Embedded query in {(time.time()-t0)*1000:.0f} ms")

            # Step 1: Optional extraction signals
            t1 = time.time()
            _ = agent1_extract_entities_predicates(current_query)
            _ = agent1b_extract_query_triples(current_query)
            log(f"[Iter {it}] Entity/Triple extraction in {(time.time()-t1)*1000:.0f} ms")

            # Step 2: Retrieval
            t2 = time.time()
            candidates = vector_query_chunks(q_emb, k=TOP_K_CHUNKS)
            log(f"[Iter {it}] Vector search returned {len(candidates)} candidates in {(time.time()-t2)*1000:.0f} ms")

            # Log preview (for developer visibility)
            preview = build_context_preview(candidates, max_chunks=CTX_PREVIEW_CHUNKS, per_chunk_clamp=CTX_PREVIEW_CLAMP)
            log(preview)

            # Step 3: Build answer context and answer
            context_text = build_context_from_chunks(candidates, max_chunks=MAX_CHUNKS_FINAL)
            answer = agent2_answer(current_query, context_text, guidance=None, output_lang=user_lang)

            # Step 4: Answer Judge
            judge_out = answer_judge_evaluate(current_query, answer, qa_feedback_history, user_lang)

            # Record this iteration in history
            record = {
                "iteration": it,
                "query_before": current_query,
                "answer": answer,
                "judge": judge_out
            }

            # If sufficient or cap reached, finish
            if judge_out.get("is_sufficient", False):
                log(f"[Iter {it}] Judge accepted the answer. Stopping.")
                qa_feedback_history.append(record)
                final_answer = answer
                final_query_used = current_query
                break

            if it == MAX_QA_JUDGE_ITERS:
                log(f"[Iter {it}] Iteration cap reached. Returning latest answer even if insufficient.")
                record["modified_query"] = current_query
                record["modifier_notes"] = "(No modification; iteration cap reached)"
                qa_feedback_history.append(record)
                final_answer = answer
                final_query_used = current_query
                break

            # Step 5: Query Modifier to improve the query for next iteration
            qm = query_modifier_apply_feedback(user_query, current_query, answer, judge_out, qa_feedback_history, user_lang)
            modified_query = qm.get("modified_query") or current_query
            record["modified_query"] = modified_query
            record["modifier_notes"] = qm.get("notes", "")
            qa_feedback_history.append(record)

            # Update query and continue
            current_query = modified_query

        # Safety fallback if loop didn't set final_answer
        if not final_answer:
            final_answer = "(No answer generated.)"
            final_query_used = current_query

        # Summary
        log("\n=== QA-Judge RAG summary ===")
        log(f"- Iterations used: {iterations_used} (cap {MAX_QA_JUDGE_ITERS})")
        log(f"- Final query used for retrieval: {final_query_used}")
        log("\n=== Final Answer ===")
        log(final_answer)
        log(f"\nLogs saved to: {log_file}")

        return {
            "final_answer": final_answer,
            "final_query_used": final_query_used,
            "iterations_used": iterations_used,
            "qa_feedback_history": qa_feedback_history,
            "log_file": str(log_file)
        }
    finally:
        if _LOGGER is not None:
            _LOGGER.close()

# ----------------- Main -----------------
if __name__ == "__main__":
    try:
        q = input("Enter your query: ").strip()
        if not q:
            print("Empty query. Exiting.")
        else:
            res = agentic_rag(q)
            print("\n----- FINAL ANSWER -----\n")
            print(res["final_answer"])
    finally:
        try:
            driver.close()
        except Exception:
            pass