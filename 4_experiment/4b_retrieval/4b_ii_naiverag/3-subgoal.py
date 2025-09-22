#!/usr/bin/env python3
# agentic_rag.py
# Agentic RAG with Subgoal Generator and Final Aggregator
#
# Pipeline:
# - Detect user language
# - Subgoal Generator produces up to SUBGOAL_MAX independent, parallelizable subgoals
#   (falls back to a single subgoal identical to the original query if decomposition isn't needed)
# - Execute each subgoal through the existing RAG loop (Agent 2 Answerer + Agent 3 Judge) in parallel
# - Final Aggregator synthesizes the final answer from the original query + all subgoal Q/A pairs
# - Verbose, thread-safe logging to a timestamped file (and console)

import os, time, json, pickle, re, random, threading
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
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
TOP_K_CHUNKS = int(os.getenv("TOP_K_CHUNKS", "30"))          # initial k from vector index
MAX_CHUNKS_FINAL = int(os.getenv("MAX_CHUNKS_FINAL", "40"))  # chunks kept for final context
CHUNK_TEXT_CLAMP = int(os.getenv("CHUNK_TEXT_CLAMP", "1000000")) # max chars per chunk included in context

# Agent loop
ANSWER_MAX_TOKENS = int(os.getenv("ANSWER_MAX_TOKENS", "4096"))
MAX_ITERS = int(os.getenv("MAX_ITERS", "3"))

# Subgoal + Aggregator (new)
# Hardcoded maximum number of subgoals as requested (not env-driven).
SUBGOAL_MAX = 4
# Concurrency cap for parallel subgoal execution (kept conservative).
SUBGOAL_MAX_PARALLEL = min(SUBGOAL_MAX, 4)
AGGREGATOR_MAX_TOKENS = int(os.getenv("AGGREGATOR_MAX_TOKENS", str(ANSWER_MAX_TOKENS)))

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

# ----------------- Build context -----------------
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
    log("[Agent 2] Intermediate answer:")
    log(answer)
    return answer

# ----------------- Agent 3 (Judge) -----------------
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
You are Agent 3 (Judge). Evaluate whether the intermediate answer is sufficient and grounded in the context.

Sufficiency criteria:
- Correctly addresses the core of the question.
- Grounded in the provided context (chunks).
- Cites UU/Article references when conclusive.
- No fabrication beyond the context.

If NOT sufficient:
- Provide "rewritten_query": a sharper, more targeted version of the user question.
- Provide "guidance_next": specific guidance for the next Agent 2 iteration.

Original user question:
\"\"\"{query_original}\"\"\"

Context (full chunk text):
\"\"\"{context_full}\"\"\"

Intermediate answer:
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

# ----------------- Subgoal Generator (new) -----------------
SUBGOAL_LIST_SCHEMA = {
  "type": "object",
  "properties": {
    "subgoals": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "question": {"type": "string"},
          "reason": {"type": "string"}
        },
        "required": ["question"]
      }
    }
  },
  "required": ["subgoals"]
}

def generate_subgoals(query_original: str, user_lang: str) -> List[Dict[str, Any]]:
    prompt = f"""
You are the Subgoal Generator.
Task: Decompose the user's question into up to {SUBGOAL_MAX} independent, parallelizable subgoals.
- If decomposition is unnecessary, return exactly one subgoal identical to the original question.
- Each subgoal must be self-contained, answerable independently, and avoid referencing other subgoals.
- Keep them minimal and non-overlapping.
- Do NOT answer any subgoal; only generate them.
- Respond in the same language as the user's question.

Original user question:
\"\"\"{query_original}\"\"\"

Return JSON with key "subgoals": list of objects:
  - question: string (the subgoal question)
  - reason: brief string (why this subgoal helps)
"""
    est = estimate_tokens_for_text(prompt)
    log("\n[Subgoal Generator] Prompt:")
    log(prompt)
    log(f"[Subgoal Generator] Prompt size: {len(prompt)} chars, est_tokens≈{est}")
    out = safe_generate_json(prompt, SUBGOAL_LIST_SCHEMA, temp=0.2) or {}
    raw = out.get("subgoals", []) or []
    # Normalize, deduplicate, clamp, and fallback
    seen = set()
    cleaned = []
    for sg in raw:
        q = clamp((sg.get("question") or "").strip(), 2000)
        if not q:
            continue
        key = re.sub(r"\s+", " ", q.lower())
        if key in seen:
            continue
        seen.add(key)
        cleaned.append({"question": q, "reason": clamp((sg.get("reason") or "").strip(), 500)})
        if len(cleaned) >= SUBGOAL_MAX:
            break
    if not cleaned:
        cleaned = [{"question": query_original, "reason": "Fallback: decomposition not required or failed; use original question as single subgoal."}]
    # Attach ids
    for i, sg in enumerate(cleaned, 1):
        sg["id"] = i
    log("[Subgoal Generator] Subgoals:")
    log(json.dumps(cleaned, ensure_ascii=False, indent=2))
    return cleaned

# ----------------- Subgoal Executor (wrapper over existing loop) -----------------
def run_agentic_loop_for_query(query_original: str, user_lang: str, log_prefix: str = "") -> Dict[str, Any]:
    """
    Executes the existing Answerer+Judge iterative loop for a single query (used per subgoal).
    Returns a dict with the final answer, judge reports, and stats.
    """
    if log_prefix:
        lp = lambda m: log_with_prefix(log_prefix, m)
    else:
        lp = log

    final_answer: Optional[str] = None
    judge_reports: List[Dict[str, Any]] = []
    guidance_prev = None
    query_for_iter = query_original
    intermediate_answer = ""
    last_candidates_count = 0

    for it in range(1, MAX_ITERS + 1):
        lp(f"\n--- Iteration {it}/{MAX_ITERS} ---")

        # Step 0: whole-query embedding
        t0 = time.time()
        q_emb = embed_text(query_for_iter)
        lp(f"[Step 0] Embedded query in {(time.time()-t0)*1000:.0f} ms")
        lp(f"[Step 0] Query used this iteration: {query_for_iter}")

        # Agent 1 & 1b (optional signals)
        t1 = time.time()
        extraction = agent1_extract_entities_predicates(query_for_iter)
        _ = agent1b_extract_query_triples(query_for_iter)
        lp(f"[Step 1] Entity/Triple extraction in {(time.time()-t1)*1000:.0f} ms")

        # Retrieval: vector search over TextChunk
        t2 = time.time()
        candidates = vector_query_chunks(q_emb, k=TOP_K_CHUNKS)
        last_candidates_count = len(candidates)
        lp(f"[Step 2] Vector search returned {len(candidates)} candidates in {(time.time()-t2)*1000:.0f} ms")

        if not candidates:
            context_text = "Tidak ada potongan teks yang ditemukan." if user_lang == "id" else "No relevant chunks found."
            intermediate_answer = agent2_answer(query_for_iter, context_text, guidance_prev, output_lang=user_lang)
            judge = agent3_judge(query_for_iter, intermediate_answer, context_text, output_lang=user_lang)
            judge_reports.append(judge)
            final_answer = intermediate_answer
            break

        # Build context from top chunks
        top_context = build_context_from_chunks(candidates, max_chunks=MAX_CHUNKS_FINAL)
        lp("\n[Context preview]:")
        lp("\n".join(top_context.splitlines()[:30]))  # preview at most 30 lines

        # Agent 2: Answer
        t3 = time.time()
        intermediate_answer = agent2_answer(query_for_iter, top_context, guidance_prev, output_lang=user_lang)
        lp(f"[Step 3] Intermediate answer in {(time.time()-t3)*1000:.0f} ms")

        # Agent 3: Judge
        t4 = time.time()
        judge = agent3_judge(query_for_iter, intermediate_answer, top_context, output_lang=user_lang)
        lp(f"[Step 4] Judge in {(time.time()-t4)*1000:.0f} ms")
        judge_reports.append(judge)

        if judge.get("is_sufficient", False):
            final_answer = intermediate_answer
            lp("\n[Judge] Verdict: Sufficient. Using intermediate answer as final.")
            break
        else:
            lp("\n[Judge] Verdict: Not sufficient.")
            rq = (judge.get("rewritten_query") or "").strip()
            gn = (judge.get("guidance_next") or "").strip()
            if rq:
                query_for_iter = rq
                lp(f"[Judge] Rewritten query for next iteration: {rq}")
            else:
                query_for_iter = query_original
                lp("[Judge] No rewritten query provided; will reuse original query.")
            guidance_prev = gn if gn else None
            if gn:
                lp(f"[Judge] Guidance for next iteration:\n{gn}")

    if final_answer is None:
        final_answer = intermediate_answer
        lp("\n[Loop] Reached max iterations. Returning the last intermediate answer as final.")

    return {
        "final_answer": final_answer,
        "iterations": len(judge_reports),
        "judge_reports": judge_reports,
        "retrieval_candidates": last_candidates_count
    }

def run_subgoal(subgoal: Dict[str, Any], user_lang: str) -> Dict[str, Any]:
    sid = subgoal.get("id", 0)
    q = subgoal.get("question", "")
    reason = subgoal.get("reason", "")
    prefix = f"[Subgoal {sid}]"
    log_with_prefix(prefix, f"Question: {q}")
    if reason:
        log_with_prefix(prefix, f"Reason: {reason}")
    try:
        result = run_agentic_loop_for_query(q, user_lang=user_lang, log_prefix=prefix)
        return {
            "id": sid,
            "question": q,
            "reason": reason,
            "answer": result["final_answer"],
            "iterations": result["iterations"],
            "judge_reports": result["judge_reports"],
            "judge_sufficient": bool(result["judge_reports"] and result["judge_reports"][-1].get("is_sufficient", False)),
            "retrieval_candidates": result.get("retrieval_candidates", 0),
            "error": None
        }
    except Exception as e:
        log_with_prefix(prefix, f"[Error] Subgoal execution failed: {e}")
        return {
            "id": sid,
            "question": q,
            "reason": reason,
            "answer": f"(Subgoal failed with error: {e})",
            "iterations": 0,
            "judge_reports": [],
            "judge_sufficient": False,
            "retrieval_candidates": 0,
            "error": str(e)
        }

# ----------------- Final Aggregator (new) -----------------
def aggregate_subgoal_answers(original_query: str, subgoal_results: List[Dict[str, Any]], user_lang: str) -> str:
    # Prepare a compact, structured summary of subgoal Q/A pairs
    lines = []
    for r in sorted(subgoal_results, key=lambda x: x.get("id", 0)):
        sid = r.get("id")
        lines.append(f"--- Subgoal S{sid} ---")
        lines.append(f"Question: {r.get('question')}")
        lines.append(f"Judge sufficient: {bool(r.get('judge_sufficient'))}; iterations: {r.get('iterations')}")
        if r.get("error"):
            lines.append(f"Error: {r.get('error')}")
        ans = r.get("answer") or ""
        # Clamp extremely long subgoal answers to keep aggregator prompt manageable
        lines.append("Answer:\n" + clamp(ans, 12000))
    subgoals_summary = "\n".join(lines)

    instructions = (
        "You are Agent 4 (Final Aggregator). Synthesize the best possible final answer to the user's original question "
        "using ONLY the information contained in the subgoal answers below. Do not invent facts. "
        "Resolve any conflicts by prioritizing answers that were marked as 'Judge sufficient: True' and that provide clearer citations. "
        "When relevant, include concrete UU/Article references that appear in the subgoal answers. "
        "Respond concisely in the same language as the user's question."
    )

    prompt = f"""
{instructions}

Original user question:
\"\"\"{original_query}\"\"\"

Subgoal Q/A pairs:
\"\"\"{subgoals_summary}\"\"\"

Now produce the final answer to the original question. Provide only the final answer (no preamble).
"""
    est = estimate_tokens_for_text(prompt)
    log("\n[Aggregator] Prompt:")
    log(prompt)
    log(f"[Aggregator] Prompt size: {len(prompt)} chars, est_tokens≈{est}")
    final_answer = safe_generate_text(prompt, max_tokens=AGGREGATOR_MAX_TOKENS, temperature=0.2)
    log("\n[Aggregator] Final synthesized answer:")
    log(final_answer)
    return final_answer

# ----------------- Agentic RAG main -----------------
def agentic_rag(query_original: str) -> Dict[str, Any]:
    global _LOGGER
    ts_name = make_timestamp_name()
    log_file = Path.cwd() / f"{ts_name}.txt"
    _LOGGER = FileLogger(log_file, also_console=True)

    try:
        log("=== Agentic RAG run started ===")
        log(f"Log file: {log_file}")
        log(f"Original Query: {query_original}")
        log("Parameters:")
        log(f"- TOP_K_CHUNKS={TOP_K_CHUNKS}, MAX_CHUNKS_FINAL={MAX_CHUNKS_FINAL}, CHUNK_TEXT_CLAMP={CHUNK_TEXT_CLAMP}")
        log(f"- ANSWER_MAX_TOKENS={ANSWER_MAX_TOKENS}, MAX_ITERS={MAX_ITERS}")
        log(f"- SUBGOAL_MAX (hardcoded)={SUBGOAL_MAX}, SUBGOAL_MAX_PARALLEL={SUBGOAL_MAX_PARALLEL}, AGGREGATOR_MAX_TOKENS={AGGREGATOR_MAX_TOKENS}")

        user_lang = detect_user_language(query_original)
        log(f"[Language] Detected user language: {user_lang}")

        # Step A: Generate subgoals
        subgoals = generate_subgoals(query_original, user_lang)

        # Step B: Execute subgoals in parallel
        log("\n=== Executing subgoals in parallel ===")
        subgoal_results: List[Dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=SUBGOAL_MAX_PARALLEL) as ex:
            futures = {ex.submit(run_subgoal, sg, user_lang): sg for sg in subgoals}
            for fut in as_completed(futures):
                res = fut.result()
                subgoal_results.append(res)

        # Keep results ordered by id
        subgoal_results.sort(key=lambda r: r.get("id", 0))

        # Step C: Aggregate into final answer
        log("\n=== Aggregating subgoal answers into final answer ===")
        final_answer = aggregate_subgoal_answers(query_original, subgoal_results, user_lang)

        # Summary
        sufficient_count = sum(1 for r in subgoal_results if r.get("judge_sufficient"))
        total_iters = sum(r.get("iterations", 0) for r in subgoal_results)
        aggregator_notes = f"Synthesized from {len(subgoal_results)} subgoals; {sufficient_count} marked sufficient by Judge; total iterations across subgoals: {total_iters}."

        log("\n=== Agentic RAG summary ===")
        log(f"- Subgoals generated: {len(subgoals)}")
        log(f"- Subgoals executed: {len(subgoal_results)} (sufficient={sufficient_count})")
        log(f"- Total iterations (across subgoals): {total_iters}")
        log("\n=== Final Answer ===")
        log(final_answer)
        log(f"\nLogs saved to: {log_file}")

        return {
            "final_answer": final_answer,
            "log_file": str(log_file),
            "subgoals": subgoals,
            "subgoal_results": subgoal_results,
            "aggregator_notes": aggregator_notes,
            "iterations_total": total_iters
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
            # Optionally, print final answer to console explicitly:
            print("\n----- FINAL ANSWER -----\n")
            print(result["final_answer"])
    finally:
        try:
            driver.close()
        except Exception:
            pass