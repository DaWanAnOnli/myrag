#!/usr/bin/env python3
# agentic_rag.py
#
# Agentic RAG (10 agents + tools) for Indonesian/English legal Q&A
# - HyDE-driven retrieval w/ optional hybrid recall
# - Unsupervised reranking (RRF; optional MMR)
# - Multi-level judging and adaptive loops
# - Evidence map with per-claim chunk citations
# - Full ablation: ENABLE_AGENT_1..ENABLE_AGENT_10, ENABLE_RETRIEVER, ENABLE_RERANKER
#
# Notes:
# - Uses Neo4j vector index 'chunk_embedding_index' over (:TextChunk {embedding})
# - Chunk ref format: CHUNK-<document_id>-<chunk_id>
# - Final answer contains citations and a "not legal advice" disclaimer
#
# Environment variables (see "Config" below for full list and defaults)

import os, time, json, pickle, re, random, math, uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, Iterable

from dotenv import load_dotenv
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
from neo4j import GraphDatabase

# ----------------- Load .env -----------------
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent.parent.parent / ".env")

# ----------------- Config -----------------
def _as_bool(x: Optional[str], default: bool) -> bool:
    if x is None: return default
    return x.strip().lower() in {"1","true","yes","y","on"}

def _as_int(x: Optional[str], default: int) -> int:
    try: return int(x)
    except Exception: return default

def _as_float(x: Optional[str], default: float) -> float:
    try: return float(x)
    except Exception: return default

# API / Backends
GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]
NEO4J_URI  = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASS", "password")

GEN_MODEL   = os.getenv("GEN_MODEL", "models/gemini-2.5-flash-lite")
EMBED_MODEL = os.getenv("EMBED_MODEL", "models/text-embedding-004")

# Retrieval (low-level)
RETRIEVE_TOP_K_PER_VARIANT = _as_int(os.getenv("RETRIEVE_TOP_K_PER_VARIANT"), 20)
HYBRID_RETRIEVAL = _as_bool(os.getenv("HYBRID_RETRIEVAL"), True)

# Reranking (unsupervised)
RRF_ENABLED = _as_bool(os.getenv("RRF_ENABLED"), True)
RRF_K = _as_int(os.getenv("RRF_K"), 60)
MMR_ENABLED = _as_bool(os.getenv("MMR_ENABLED"), False)
MMR_LAMBDA = _as_float(os.getenv("MMR_LAMBDA"), 0.7)
RERANKER_KEEP_N = _as_int(os.getenv("RERANKER_KEEP_N"), 12)

# Context and citations
MAX_CHUNKS_PER_IQ = _as_int(os.getenv("MAX_CHUNKS_PER_IQ"), 8)
MAX_CHARS_PER_CHUNK_IN_CONTEXT = _as_int(os.getenv("MAX_CHARS_PER_CHUNK_IN_CONTEXT"), 2000)
REQUIRE_CITATIONS = _as_bool(os.getenv("REQUIRE_CITATIONS"), True)

# HyDE
HYDE_ENABLED = _as_bool(os.getenv("HYDE_ENABLED"), True)
HYDE_VARIANTS = min(max(_as_int(os.getenv("HYDE_VARIANTS"), 2), 1), 3)
HYDE_TARGET_LENGTH_CHARS = _as_int(os.getenv("HYDE_TARGET_LENGTH_CHARS"), 1200)
HYDE_DIVERSITY = _as_bool(os.getenv("HYDE_DIVERSITY"), True)

# Judges and temperatures
JUDGE_CONFIDENCE_THRESHOLD = _as_float(os.getenv("JUDGE_CONFIDENCE_THRESHOLD"), 0.7)
JUDGE_MAX_CRITIQUE_LEN = _as_int(os.getenv("JUDGE_MAX_CRITIQUE_LEN"), 600)

ANSWER_TEMPERATURES = {
    "planner": _as_float(os.getenv("TEMP_PLANNER"), 0.2),
    "iq": _as_float(os.getenv("TEMP_IQ"), 0.2),
    "hyde": _as_float(os.getenv("TEMP_HYDE"), 0.3),
    "judge": _as_float(os.getenv("TEMP_JUDGE"), 0.0),
    "ia": _as_float(os.getenv("TEMP_IA"), 0.1),
    "sa": _as_float(os.getenv("TEMP_SA"), 0.1),
    "fa": _as_float(os.getenv("TEMP_FA"), 0.2),
    "modifier": _as_float(os.getenv("TEMP_MODIFIER"), 0.2),
}

# Loop guards and budgets
MAX_GLOBAL_PASSES = _as_int(os.getenv("MAX_GLOBAL_PASSES"), 1)
MAX_SUBGOAL_REFINEMENTS = _as_int(os.getenv("MAX_SUBGOAL_REFINEMENTS"), 2)
MAX_HYDE_TRIES = _as_int(os.getenv("MAX_HYDE_TRIES"), 3)
MAX_IQ_REWRITES = _as_int(os.getenv("MAX_IQ_REWRITES"), 2)
NO_OP_GUARD = _as_bool(os.getenv("NO_OP_GUARD"), True)
CONFIDENCE_GATES = _as_bool(os.getenv("CONFIDENCE_GATES"), True)

# Language
LANGUAGE_MODE = os.getenv("LANGUAGE_MODE", "auto").lower()  # auto/id/en

# Telemetry / logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "verbose")  # minimal/verbose
TRACE_PROMPTS = _as_bool(os.getenv("TRACE_PROMPTS"), True)
STORE_EVIDENCE_MAP = _as_bool(os.getenv("STORE_EVIDENCE_MAP"), True)

# Ablation flags (agents & tools)
ENABLE_AGENT_1 = _as_bool(os.getenv("ENABLE_AGENT_1"), True)   # Subgoal Generator
ENABLE_AGENT_2 = _as_bool(os.getenv("ENABLE_AGENT_2"), True)   # Intermediate Question Generator
ENABLE_AGENT_3 = _as_bool(os.getenv("ENABLE_AGENT_3"), True)   # HyDE Generator
ENABLE_AGENT_4 = _as_bool(os.getenv("ENABLE_AGENT_4"), True)   # Intermediate Context Judge
ENABLE_AGENT_5 = _as_bool(os.getenv("ENABLE_AGENT_5"), True)   # Intermediate Answerer
ENABLE_AGENT_6 = _as_bool(os.getenv("ENABLE_AGENT_6"), True)   # Subgoal Context Judge
ENABLE_AGENT_7 = _as_bool(os.getenv("ENABLE_AGENT_7"), True)   # Subgoal Answerer
ENABLE_AGENT_8 = _as_bool(os.getenv("ENABLE_AGENT_8"), True)   # Subgoal Modifier
ENABLE_AGENT_9 = _as_bool(os.getenv("ENABLE_AGENT_9"), True)   # Final Context Judge
ENABLE_AGENT_10 = _as_bool(os.getenv("ENABLE_AGENT_10"), True) # Final Answer Generator

ENABLE_RETRIEVER = _as_bool(os.getenv("ENABLE_RETRIEVER"), True)
ENABLE_RERANKER = _as_bool(os.getenv("ENABLE_RERANKER"), True)

ROUTING_MODE = os.getenv("ROUTING_MODE", "sequential")  # sequential/parallel (parallel not implemented, placeholder)
SEED = _as_int(os.getenv("SEED"), 42)

# ----------------- Initialize SDKs -----------------
random.seed(SEED)
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
            self._fh.flush(); self._fh.close()
        except Exception:
            pass

_LOGGER: Optional[FileLogger] = None
def log(msg: str = "", level: str = "verbose"):
    if LOG_LEVEL == "minimal" and level == "verbose":
        return
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

def detect_user_language(text: str) -> Tuple[str, float]:
    t = (text or "").lower()
    if re.search(r"\b(pasal|undang[- ]?undang|uu\s*\d|peraturan|menteri|ayat|bab|bagian|paragraf|ketentuan|sebagaimana|dimaksud)\b", t):
        return "id", 0.9
    if re.search(r"\b(article|act|law|regulation|minister|section|paragraph|chapter|pursuant|provided that)\b", t):
        return "en", 0.9
    id_tokens = {"yang","dan","atau","tidak","adalah","berdasarkan","sebagaimana","pada","dalam","dapat","harus","wajib","pasal","undang","peraturan","menteri","ayat","bab","bagian","paragraf","ketentuan","pengundangan","apabila","jika"}
    en_tokens = {"the","and","or","not","is","based","as","provided","pursuant","in","may","must","shall","article","act","law","regulation","minister","section","paragraph","chapter","whereas"}
    words = re.findall(r"[a-z]+", t)
    score_id = sum(1 for w in words if w in id_tokens)
    score_en = sum(1 for w in words if w in en_tokens)
    if score_id > score_en: return "id", 0.6
    if score_en > score_id: return "en", 0.6
    return "en", 0.5

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

# --- Replace your safe_generate_json with this version (plus helper) ---

def _extract_first_json_object(s: str) -> Optional[str]:
    if not s: 
        return None
    # Strip markdown fences if present
    t = s.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\n", "", t)
        if t.endswith("```"):
            t = t[:-3].strip()
    # Find the first balanced {...}
    start = t.find("{")
    if start == -1:
        return None
    depth = 0
    for i, ch in enumerate(t[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return t[start:i+1]
    return None

def safe_generate_json(prompt: str, schema: Dict[str, Any], temp: float = 0.0, max_tokens: int = 512, retries: int = 2) -> Dict[str, Any]:
    if TRACE_PROMPTS:
        log("\n[LLM JSON Prompt]:", "verbose")
        log(prompt, "verbose")

    def _try_parse(payload: str) -> Optional[Dict[str, Any]]:
        if not payload:
            return None
        # direct attempt
        try:
            return json.loads(payload)
        except Exception:
            pass
        # fenced / substring attempt
        js = _extract_first_json_object(payload)
        if js:
            try:
                return json.loads(js)
            except Exception:
                return None
        return None

    # First attempt: strict JSON mode with schema
    cfg = GenerationConfig(
        temperature=temp,
        response_mime_type="application/json",
        response_schema=schema,
        max_output_tokens=max_tokens
    )
    resp = _api_call_with_retry(gen_model.generate_content, prompt, generation_config=cfg)
    texts: List[str] = []
    try:
        if isinstance(resp.text, str) and resp.text.strip():
            texts.append(resp.text)
    except Exception:
        pass
    try:
        cand0 = resp.candidates[0].content.parts[0].text
        if isinstance(cand0, str) and cand0.strip():
            texts.append(cand0)
    except Exception:
        pass

    for t in texts:
        parsed = _try_parse(t)
        if parsed is not None:
            return parsed

    # Repair/retry loop: ask for minified JSON only
    for attempt in range(retries):
        repair_prompt = (
            "Return ONLY minified JSON that conforms to this schema. "
            "No prose, no explanations, no markdown, no code fences.\n\n"
            f"Schema:\n{json.dumps(schema, ensure_ascii=False)}\n\n"
            "Now produce the JSON for this task:\n"
            f"{prompt}"
        )
        if TRACE_PROMPTS:
            log("\n[LLM JSON Repair Prompt]:", "verbose")
            log(repair_prompt[:4000], "verbose")

        cfg2 = GenerationConfig(
            temperature=0.0,
            response_mime_type="application/json",
            response_schema=schema,
            max_output_tokens=max_tokens
        )
        resp2 = _api_call_with_retry(gen_model.generate_content, repair_prompt, generation_config=cfg2)
        try:
            t2 = resp2.text if isinstance(resp2.text, str) else ""
        except Exception:
            t2 = ""
        if not t2:
            try:
                t2 = resp2.candidates[0].content.parts[0].text
            except Exception:
                t2 = ""
        parsed = _try_parse(t2)
        if parsed is not None:
            return parsed

    info = get_finish_info(resp)
    log(f"[LLM JSON parse warning] Unable to parse JSON after retries. Diagnostics: {info}")
    return {}

def safe_generate_text(prompt: str, max_tokens: int, temperature: float = 0.2) -> str:
    if TRACE_PROMPTS:
        log("\n[LLM Text Prompt]:", "verbose")
        log(prompt, "verbose")
    cfg = GenerationConfig(temperature=temperature, max_output_tokens=max_tokens)
    resp = _api_call_with_retry(gen_model.generate_content, prompt, generation_config=cfg)
    text = extract_text_from_response(resp)
    if text: return text
    info = get_finish_info(resp)
    log(f"[LLM text warning] No text returned. Diagnostics: {info}")
    return f"(Model returned no text. finish_info={info})"

# ----------------- Data Model -----------------
@dataclass
class IQObject:
    iq_id: str
    text: str
    status: str = "pending"  # pending/retrieving/answering/done
    hyde_docs: List[str] = field(default_factory=list)  # per variant
    retrievals: List[List[Dict[str, Any]]] = field(default_factory=list)  # per hyde variant (or IQ raw)
    reranked_chunks: List[Dict[str, Any]] = field(default_factory=list)
    context_judge: Dict[str, Any] = field(default_factory=dict)
    intermediate_answer: str = ""
    citations: List[str] = field(default_factory=list)
    confidence: float = 0.0
    tries_hyde: int = 0
    rewrites: int = 0

@dataclass
class Subgoal:
    subgoal_id: str
    text: str
    status: str = "pending"  # pending/answering/done
    dependencies: List[str] = field(default_factory=list)
    intermediate_questions: List[IQObject] = field(default_factory=list)
    intermediate_answers: List[Dict[str, Any]] = field(default_factory=list)
    subgoal_answer: str = ""
    citations: List[str] = field(default_factory=list)
    confidence: float = 0.0
    judge_reports: List[Dict[str, Any]] = field(default_factory=list)
    history: List[Dict[str, Any]] = field(default_factory=list)  # Agent 8 modifications

@dataclass
class EvidenceMap:
    answers_to_chunks: Dict[str, List[str]] = field(default_factory=dict)
    chunk_metadata_index: Dict[str, Dict[str, Any]] = field(default_factory=dict)

@dataclass
class RunState:
    run_id: str
    user_query: str
    user_language: str
    config_snapshot: Dict[str, Any]
    subgoals: List[Subgoal] = field(default_factory=list)
    final: Dict[str, Any] = field(default_factory=lambda: {"status":"pending","final_answer":"","confidence":0.0,"judge_reports":[]})
    telemetry: Dict[str, Any] = field(default_factory=dict)
    evidence: EvidenceMap = field(default_factory=EvidenceMap)
    counters: Dict[str, int] = field(default_factory=lambda: {"sg":0, "iq":0, "ia":0, "sa":0})
    guidance_from_agent9: str = ""

# ----------------- Telemetry helpers -----------------
class Timer:
    def __init__(self, name: str, state: RunState):
        self.name = name
        self.state = state
        self.t0 = None
    def __enter__(self):
        self.t0 = time.time()
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        dt = (time.time() - self.t0) * 1000.0
        self.state.telemetry.setdefault("timings_ms", {}).setdefault(self.name, 0.0)
        self.state.telemetry["timings_ms"][self.name] += dt

def snapshot_config() -> Dict[str, Any]:
    return {
        "RETRIEVE_TOP_K_PER_VARIANT": RETRIEVE_TOP_K_PER_VARIANT,
        "HYBRID_RETRIEVAL": HYBRID_RETRIEVAL,
        "RRF_ENABLED": RRF_ENABLED,
        "RRF_K": RRF_K,
        "MMR_ENABLED": MMR_ENABLED,
        "MMR_LAMBDA": MMR_LAMBDA,
        "RERANKER_KEEP_N": RERANKER_KEEP_N,
        "MAX_CHUNKS_PER_IQ": MAX_CHUNKS_PER_IQ,
        "MAX_CHARS_PER_CHUNK_IN_CONTEXT": MAX_CHARS_PER_CHUNK_IN_CONTEXT,
        "REQUIRE_CITATIONS": REQUIRE_CITATIONS,
        "HYDE_ENABLED": HYDE_ENABLED,
        "HYDE_VARIANTS": HYDE_VARIANTS,
        "HYDE_TARGET_LENGTH_CHARS": HYDE_TARGET_LENGTH_CHARS,
        "HYDE_DIVERSITY": HYDE_DIVERSITY,
        "JUDGE_CONFIDENCE_THRESHOLD": JUDGE_CONFIDENCE_THRESHOLD,
        "JUDGE_MAX_CRITIQUE_LEN": JUDGE_MAX_CRITIQUE_LEN,
        "ANSWER_TEMPERATURES": ANSWER_TEMPERATURES,
        "MAX_GLOBAL_PASSES": MAX_GLOBAL_PASSES,
        "MAX_SUBGOAL_REFINEMENTS": MAX_SUBGOAL_REFINEMENTS,
        "MAX_HYDE_TRIES": MAX_HYDE_TRIES,
        "MAX_IQ_REWRITES": MAX_IQ_REWRITES,
        "NO_OP_GUARD": NO_OP_GUARD,
        "CONFIDENCE_GATES": CONFIDENCE_GATES,
        "LANGUAGE_MODE": LANGUAGE_MODE,
        "LOG_LEVEL": LOG_LEVEL,
        "TRACE_PROMPTS": TRACE_PROMPTS,
        "STORE_EVIDENCE_MAP": STORE_EVIDENCE_MAP,
        "ENABLE_FLAGS": {
            "ENABLE_AGENT_1": ENABLE_AGENT_1,
            "ENABLE_AGENT_2": ENABLE_AGENT_2,
            "ENABLE_AGENT_3": ENABLE_AGENT_3,
            "ENABLE_AGENT_4": ENABLE_AGENT_4,
            "ENABLE_AGENT_5": ENABLE_AGENT_5,
            "ENABLE_AGENT_6": ENABLE_AGENT_6,
            "ENABLE_AGENT_7": ENABLE_AGENT_7,
            "ENABLE_AGENT_8": ENABLE_AGENT_8,
            "ENABLE_AGENT_9": ENABLE_AGENT_9,
            "ENABLE_AGENT_10": ENABLE_AGENT_10,
            "ENABLE_RETRIEVER": ENABLE_RETRIEVER,
            "ENABLE_RERANKER": ENABLE_RERANKER,
            "ROUTING_MODE": ROUTING_MODE
        }
    }

# ----------------- Retrieval helpers -----------------
def make_chunk_ref(c: Dict[str, Any]) -> str:
    return f"CHUNK-{c.get('document_id')}-{c.get('chunk_id')}"

def vector_query_chunks(q_emb: List[float], k: int) -> List[Dict[str, Any]]:
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
        item = {
            "key": c.get("key"),
            "chunk_id": c.get("chunk_id"),
            "document_id": c.get("document_id"),
            "uu_number": c.get("uu_number"),
            "pages": c.get("pages"),
            "content": c.get("content"),
            "score": r["score"],
        }
        item["chunk_ref"] = make_chunk_ref(item)
        out.append(item)
    return out

def retrieve_by_text(text: str, top_k: int) -> List[Dict[str, Any]]:
    q_emb = embed_text(text)
    return vector_query_chunks(q_emb, k=top_k)

# ----------------- Reranker: RRF and optional MMR -----------------
def rrf_fuse(result_lists: List[List[Dict[str, Any]]], k: int = RRF_K) -> Dict[str, float]:
    fused: Dict[str, float] = {}
    for lst in result_lists:
        for rank, item in enumerate(lst, start=1):
            ref = item["chunk_ref"]
            fused[ref] = fused.get(ref, 0.0) + 1.0 / (k + rank)
    return fused

_CHUNK_EMBED_CACHE: Dict[str, List[float]] = {}
def _chunk_embedding(ref: str, content: str) -> List[float]:
    if ref in _CHUNK_EMBED_CACHE:
        return _CHUNK_EMBED_CACHE[ref]
    emb = embed_text(content[:1000])  # cap for speed
    _CHUNK_EMBED_CACHE[ref] = emb
    return emb

def cosine(a: List[float], b: List[float]) -> float:
    if not a or not b: return 0.0
    s = sum(x*y for x,y in zip(a,b))
    na = math.sqrt(sum(x*x for x in a))
    nb = math.sqrt(sum(y*y for y in b))
    if na == 0 or nb == 0: return 0.0
    return s/(na*nb)

def mmr_select(candidates: List[Dict[str, Any]], query_vec: List[float], lam: float, keep_n: int) -> List[Dict[str, Any]]:
    # Simple MMR: relevance = normalized score; diversity = cosine among chunk embeddings
    if not candidates:
        return []
    # normalize relevance by rank (higher better)
    scores = {c["chunk_ref"]: float(c.get("score", 0.0)) for c in candidates}
    if scores:
        mx = max(scores.values())
        if mx > 0:
            for k_ in scores:
                scores[k_] /= mx
    selected: List[Dict[str, Any]] = []
    remaining = candidates[:]
    selected_refs: set = set()
    while remaining and len(selected) < keep_n:
        best = None
        best_val = -1e9
        for c in remaining:
            ref = c["chunk_ref"]
            rel = scores.get(ref, 0.0)
            # Diversity penalty: max sim to already selected
            if not selected:
                div = 0.0
            else:
                c_emb = _chunk_embedding(ref, c.get("content",""))
                max_sim = 0.0
                for s in selected:
                    s_emb = _chunk_embedding(s["chunk_ref"], s.get("content",""))
                    max_sim = max(max_sim, cosine(c_emb, s_emb))
                div = max_sim
            val = lam * rel - (1.0 - lam) * div
            if val > best_val:
                best_val = val
                best = c
        if best is None:
            break
        selected.append(best)
        selected_refs.add(best["chunk_ref"])
        remaining = [x for x in remaining if x["chunk_ref"] not in selected_refs]
    return selected

def unsupervised_rerank(result_sets: List[List[Dict[str, Any]]],
                        keep_n: int = RERANKER_KEEP_N,
                        query_text: Optional[str] = None) -> List[Dict[str, Any]]:
    # RRF fusion
    if not result_sets or not any(result_sets):
        return []
    fused_scores = rrf_fuse(result_sets, k=RRF_K) if RRF_ENABLED else {}
    merged: Dict[str, Dict[str, Any]] = {}
    for lst in result_sets:
        for item in lst:
            ref = item["chunk_ref"]
            if ref not in merged:
                merged[ref] = item.copy()
            # maintain best raw score as well
            merged[ref]["_max_raw_score"] = max(merged[ref].get("_max_raw_score", 0.0), float(item.get("score", 0.0)))
    # attach fused score
    for ref, it in merged.items():
        it["_fused_score"] = fused_scores.get(ref, 0.0)
    # rank by fused first then raw
    ranked = sorted(merged.values(), key=lambda x: (x.get("_fused_score", 0.0), x.get("_max_raw_score", 0.0)), reverse=True)
    # optional MMR
    if ENABLE_RERANKER and MMR_ENABLED and query_text:
        q_vec = embed_text(query_text)
        mmr_ranked = mmr_select(ranked, q_vec, lam=MMR_LAMBDA, keep_n=keep_n)
        return mmr_ranked[:keep_n]
    return ranked[:keep_n]

# ----------------- Context builder -----------------
def clamp(s: Optional[str], n: int) -> str:
    t = (s or "").strip()
    return t[:n]

def context_from_chunks(chunks: List[Dict[str, Any]], max_chars_per_chunk: int, user_language: str) -> str:
    lines = ["Context chunks:" if user_language == "en" else "Potongan konteks:"]
    for i, c in enumerate(chunks, 1):
        meta = f"{c.get('uu_number') or ''} | pages={c.get('pages')} | score={c.get('score'):.3f}"
        txt = clamp(c.get("content") or "", max_chars_per_chunk)
        lines.append(f"[{c['chunk_ref']}] doc={c.get('document_id')} chunk={c.get('chunk_id')} | {meta}\n{txt}")
    return "\n".join(lines)

# ----------------- Evidence map helpers -----------------
def add_chunks_to_evidence(state: RunState, chunks: List[Dict[str, Any]]):
    for c in chunks:
        ref = c["chunk_ref"]
        if ref not in state.evidence.chunk_metadata_index:
            state.evidence.chunk_metadata_index[ref] = {
                "doc_id": c.get("document_id"),
                "uu_number": c.get("uu_number"),
                "pages": c.get("pages"),
                "score": c.get("score"),
                "snippet": clamp(c.get("content",""), 400)
            }

def record_answer_citations(state: RunState, answer_id: str, citations: List[str]):
    state.evidence.answers_to_chunks[answer_id] = list(dict.fromkeys([c for c in citations if isinstance(c, str) and c.startswith("CHUNK-")]))

# ----------------- Agent schemas -----------------
JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "decision": {"type": "boolean"},
        "rationale": {"type": "string"},
        "guidance_next": {"type": "string"},
        "confidence": {"type": "number"},
        "missing_items": {"type": "array", "items": {"type": "string"}}
    },
    "required": ["decision","rationale"]
}

SUBGOAL_PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "subgoals": {"type":"array","items":{"type":"string"}},
        "rationale": {"type":"string"},
        "confidence": {"type":"number"}
    },
    "required": ["subgoals"]
}

IQ_SCHEMA = {
    "type": "object",
    "properties": {
        "intermediate_questions": {"type":"array","items":{"type":"string"}},
        "coverage_note": {"type":"string"},
        "confidence": {"type":"number"}
    },
    "required": ["intermediate_questions"]
}

HYDE_SCHEMA = {
    "type": "object",
    "properties": {
        "hyde_docs": {"type":"array","items":{"type":"string"}},
        "generation_notes": {"type":"string"},
        "confidence": {"type":"number"}
    },
    "required": ["hyde_docs"]
}

INTERMEDIATE_ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type":"string"},
        "citations": {"type":"array","items":{"type":"string"}},
        "confidence": {"type":"number"}
    },
    "required": ["answer","citations"]
}

SUBGOAL_ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type":"string"},
        "citations": {"type":"array","items":{"type":"string"}},
        "confidence": {"type":"number"}
    },
    "required": ["answer","citations"]
}

FINAL_ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "final_answer": {"type":"string"},
        "citations": {"type":"array","items":{"type":"string"}},
        "confidence": {"type":"number"},
        "notes": {"type":"string"}
    },
    "required": ["final_answer","citations"]
}

# ----------------- Agent prompts -----------------
def lang_tag(user_lang: str) -> str:
    return "Indonesian" if user_lang == "id" else "English"

def agent1_subgoal_generator(state: RunState) -> Tuple[List[str], Dict[str, Any]]:
    if not ENABLE_AGENT_1:
        return [state.user_query], {"rationale":"Ablated: using original query as single subgoal.","confidence":1.0}
    feedback = state.guidance_from_agent9 or ""
    prompt = f"""
You are Agent 1 (Subgoal Generator). Plan a sequential set of dependent subgoals to answer the user's legal question. If single-step, return the original query as the sole subgoal.

Return JSON with keys: subgoals (ordered list of question-form strings), rationale, confidence (0-1).

User language: {lang_tag(state.user_language)}
User question:
\"\"\"{state.user_query}\"\"\"

Feedback from Final Context Judge (if any):
\"\"\"{feedback[:JUDGE_MAX_CRITIQUE_LEN]}\"\"\"
"""
    out = safe_generate_json(prompt, SUBGOAL_PLAN_SCHEMA, temp=ANSWER_TEMPERATURES["planner"])
    subgoals = [s for s in (out.get("subgoals") or []) if isinstance(s, str) and s.strip()]
    if not subgoals:
        subgoals = [state.user_query]
    return subgoals, {"rationale": out.get("rationale",""), "confidence": float(out.get("confidence", 0.6))}

def agent2_iq_generator(state: RunState, sg: Subgoal, guidance: str = "") -> Tuple[List[str], Dict[str, Any]]:
    if not ENABLE_AGENT_2:
        return [sg.text], {"coverage_note":"Ablated: single IQ = subgoal.","confidence":1.0}
    prompt = f"""
You are Agent 2 (Intermediate Question Generator). For the subgoal below, produce independent, parallelizable intermediate questions (IQs). If decomposition isn't needed, return the subgoal itself as a single IQ.

Return JSON with keys: intermediate_questions (1-3), coverage_note, confidence (0-1).

User language: {lang_tag(state.user_language)}
Original user query:
\"\"\"{state.user_query}\"\"\"

Subgoal:
\"\"\"{sg.text}\"\"\"

Guidance (if any):
\"\"\"{guidance[:JUDGE_MAX_CRITIQUE_LEN]}\"\"\"
"""
    out = safe_generate_json(prompt, IQ_SCHEMA, temp=ANSWER_TEMPERATURES["iq"])
    iqs = [q for q in (out.get("intermediate_questions") or []) if isinstance(q,str) and q.strip()]
    if not iqs:
        iqs = [sg.text]
    return iqs, {"coverage_note": out.get("coverage_note",""), "confidence": float(out.get("confidence", 0.6))}

def agent3_hyde_generator(state: RunState, iq: IQObject, guidance: str = "") -> Tuple[List[str], Dict[str, Any]]:
    if not ENABLE_AGENT_3 or not HYDE_ENABLED:
        return [f"A hypothetical legal summary and relevant statutory excerpts that would answer: {iq.text}"], {"generation_notes":"Ablated HyDE: simple template.","confidence":1.0}
    prompt = f"""
You are Agent 3 (HyDE Generator). Write {HYDE_VARIANTS} hypothetical short documents that could plausibly exist in Indonesian/English legal corpora and directly answer the Intermediate Question. Each should be ~{HYDE_TARGET_LENGTH_CHARS} characters and vary in style/scope for diversity. Avoid hallucinating exact citations; prefer plausible structure and terminology.

Return JSON with keys: hyde_docs (array of strings), generation_notes, confidence.

User language: {lang_tag(state.user_language)}
Intermediate Question:
\"\"\"{iq.text}\"\"\"

Guidance from Context Judge (if any):
\"\"\"{guidance[:JUDGE_MAX_CRITIQUE_LEN]}\"\"\"
"""
    out = safe_generate_json(prompt, HYDE_SCHEMA, temp=ANSWER_TEMPERATURES["hyde"], max_tokens=8192)
    docs = [d for d in (out.get("hyde_docs") or []) if isinstance(d, str) and d.strip()]
    if not docs:
        docs = [f"Answer narrative likely mentions statute articles and definitions relevant to: {iq.text}."]
    # enforce variant count
    docs = docs[:HYDE_VARIANTS]
    return docs, {"generation_notes":out.get("generation_notes",""), "confidence": float(out.get("confidence", 0.6))}

def agent4_context_judge(state: RunState, iq: IQObject, chunks: List[Dict[str, Any]], hyde_docs: List[str]) -> Dict[str, Any]:
    if not ENABLE_AGENT_4:
        return {"decision": bool(chunks), "rationale":"Ablated: accept if any chunks.", "guidance_next":"", "confidence":1.0, "missing_items":[]}
    # Summarize top chunk headers for compactness
    headers = "\n".join([f"{i+1}. {c['chunk_ref']} | uu={c.get('uu_number','')} | pages={c.get('pages')}" for i,c in enumerate(chunks[:MAX_CHUNKS_PER_IQ])])
    prompt = f"""
You are Agent 4 (Intermediate Context Judge). Decide if the reranked chunks are relevant and sufficient to answer the Intermediate Question. If not, provide concrete guidance to regenerate HyDE or refine the IQ.

Criteria:
- Relevance: majority of top chunks directly address the IQ.
- Sufficiency: enough information to answer without guessing.
- Grounding: identifiable citations exist.

Return JSON: decision (true/false), rationale, guidance_next, confidence (0-1), missing_items (array).

User language: {lang_tag(state.user_language)}
Intermediate Question:
\"\"\"{iq.text}\"\"\"

HyDE summaries (first {min(2,len(hyde_docs))} shown):
\"\"\"{chr(10).join(hyde_docs[:2])[:1000]}\"\"\"

Top reranked chunk headers:
\"\"\"{headers}\"\"\"
"""
    out = safe_generate_json(prompt, JUDGE_SCHEMA, temp=ANSWER_TEMPERATURES["judge"])
    out["decision"] = bool(out.get("decision", False))
    out["rationale"] = out.get("rationale","")
    out["guidance_next"] = out.get("guidance_next","")
    out["confidence"] = float(out.get("confidence", 0.6))
    out["missing_items"] = out.get("missing_items", []) or []
    return out

def agent5_intermediate_answerer(state: RunState, iq: IQObject, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not ENABLE_AGENT_5:
        # simple extractive baseline
        text = (chunks[0]["content"] if chunks else "")
        ans = text[:600] if text else ("No evidence found." if state.user_language=="en" else "Tidak ada bukti yang ditemukan.")
        cits = [chunks[0]["chunk_ref"]] if chunks else []
        return {"answer": ans, "citations": cits, "confidence": 0.4}
    ctx = context_from_chunks(chunks[:MAX_CHUNKS_PER_IQ], MAX_CHARS_PER_CHUNK_IN_CONTEXT, state.user_language)
    cite_instr = "Use bracket citations like [CHUNK-<doc_id>-<chunk_id>] for each claim." if REQUIRE_CITATIONS else "Citations optional."
    prompt = f"""
You are Agent 5 (Intermediate Answerer). Answer the Intermediate Question strictly from the provided context chunks. Do not invent facts not in the chunks. {cite_instr}

Return JSON: answer (concise, 3-8 sentences), citations (unique list of chunk_refs you used), confidence (0-1).

User language: {lang_tag(state.user_language)}
Intermediate Question:
\"\"\"{iq.text}\"\"\"

{ctx}
"""
    out = safe_generate_json(prompt, INTERMEDIATE_ANSWER_SCHEMA, temp=ANSWER_TEMPERATURES["ia"], max_tokens=8192)
    ans = out.get("answer","").strip()
    cits = [c for c in (out.get("citations") or []) if isinstance(c,str) and c.startswith("CHUNK-")]
    conf = float(out.get("confidence", 0.6))
    # safety net: extract citations found in text if missing
    if REQUIRE_CITATIONS and not cits and ans:
        cits = sorted(set(re.findall(r"\[(CHUNK-[^\]]+)\]", ans)))
    return {"answer": ans, "citations": cits, "confidence": conf}

def agent6_subgoal_context_judge(state: RunState, sg: Subgoal, coverage_note: str) -> Dict[str, Any]:
    if not ENABLE_AGENT_6:
        return {"decision": True, "rationale":"Ablated: accept subgoal.", "guidance_next":"", "confidence":1.0, "contradictions":[]}
    # Build summary of IQs and answers
    lines = []
    for iq in sg.intermediate_questions:
        lines.append(f"- {iq.iq_id}: {iq.text}\n  Answer: {iq.intermediate_answer[:300]}\n  Citations: {', '.join(iq.citations)}")
    summary = "\n".join(lines)
    schema = {
        "type":"object",
        "properties":{
            "decision":{"type":"boolean"},
            "rationale":{"type":"string"},
            "guidance_next":{"type":"string"},
            "confidence":{"type":"number"},
            "contradictions":{"type":"array","items":{"type":"string"}}
        },
        "required":["decision","rationale"]
    }
    prompt = f"""
You are Agent 6 (Subgoal Context Judge). Determine whether the set of intermediate answers fully and consistently resolves the subgoal.

Criteria:
- Coverage: all aspects in coverage_note are answered.
- Consistency: no contradictions across intermediate answers.
- Readiness: clear path to compose a subgoal answer now.

Return JSON: decision (true/false), rationale, guidance_next, confidence (0-1), contradictions (array).

User language: {lang_tag(state.user_language)}
Subgoal:
\"\"\"{sg.text}\"\"\"

Coverage note:
\"\"\"{coverage_note}\"\"\"

Intermediate answers:
\"\"\"{summary[:2500]}\"\"\"
"""
    out = safe_generate_json(prompt, schema, temp=ANSWER_TEMPERATURES["judge"])
    out["decision"] = bool(out.get("decision", False))
    out["rationale"] = out.get("rationale","")
    out["guidance_next"] = out.get("guidance_next","")
    out["confidence"] = float(out.get("confidence", 0.6))
    out["contradictions"] = out.get("contradictions", []) or []
    return out

def agent7_subgoal_answerer(state: RunState, sg: Subgoal) -> Dict[str, Any]:
    if not ENABLE_AGENT_7:
        # Concatenate IAs
        text = "\n\n".join([iq.intermediate_answer for iq in sg.intermediate_questions if iq.intermediate_answer])
        cits = sorted({c for iq in sg.intermediate_questions for c in (iq.citations or [])})
        return {"answer": text, "citations": cits, "confidence": 0.5}
    # Build stitchable material with citations
    bundle = []
    for iq in sg.intermediate_questions:
        bundle.append(f"[{iq.iq_id}] {iq.intermediate_answer}\nCitations: {', '.join(iq.citations)}")
    src = "\n\n".join(bundle)
    prompt = f"""
You are Agent 7 (Subgoal Answerer). Compose a coherent, concise answer to the subgoal using the intermediate answers. Include bracket citations referencing the chunk_refs used across intermediate answers. Do not add new facts.

Return JSON: answer (5-10 sentences), citations (unique list of chunk_refs used), confidence (0-1).

User language: {lang_tag(state.user_language)}
Subgoal:
\"\"\"{sg.text}\"\"\"

Intermediate materials:
\"\"\"{src[:4000]}\"\"\"
"""
    out = safe_generate_json(prompt, SUBGOAL_ANSWER_SCHEMA, temp=ANSWER_TEMPERATURES["sa"], max_tokens=8192)
    ans = out.get("answer","").strip()
    cits = [c for c in (out.get("citations") or []) if isinstance(c,str) and c.startswith("CHUNK-")]
    if not cits and ans:
        cits = sorted(set(re.findall(r"\[(CHUNK-[^\]]+)\]", ans)))
    return {"answer": ans, "citations": cits, "confidence": float(out.get("confidence", 0.7))}

def agent8_subgoal_modifier(state: RunState, completed_sg: Subgoal, pending: List[Subgoal]) -> Tuple[List[str], Dict[str, Any]]:
    if not ENABLE_AGENT_8 or not pending:
        return [sg.text for sg in pending], {"change_log":"Ablated/no pending changes.","confidence":1.0}
    pending_texts = "\n".join([f"- {sg.subgoal_id}: {sg.text}" for sg in pending])
    prompt = f"""
You are Agent 8 (Subgoal Modifier). Based on the completed subgoal answer below, propose modifications to the remaining pending subgoals (add, alter, remove, enrich). Preserve completed subgoals. Only modify pending ones.

Return JSON with key: modified_subgoals (ordered array of question-form strings), change_log (brief), confidence (0-1).

User language: {lang_tag(state.user_language)}
Original user query:
\"\"\"{state.user_query}\"\"\"

Completed subgoal and answer:
ID: {completed_sg.subgoal_id}
Text: \"\"\"{completed_sg.text}\"\"\"
Answer: \"\"\"{completed_sg.subgoal_answer[:1500]}\"\"\"

Pending subgoals:
{pending_texts}
"""
    schema = {
        "type":"object",
        "properties":{
            "modified_subgoals":{"type":"array","items":{"type":"string"}},
            "change_log":{"type":"string"},
            "confidence":{"type":"number"}
        },
        "required":["modified_subgoals"]
    }
    out = safe_generate_json(prompt, schema, temp=ANSWER_TEMPERATURES["modifier"])
    mods = [s for s in (out.get("modified_subgoals") or []) if isinstance(s, str) and s.strip()]
    if not mods:
        mods = [sg.text for sg in pending]
    return mods, {"change_log": out.get("change_log",""), "confidence": float(out.get("confidence", 0.6))}

def agent9_final_context_judge(state: RunState) -> Dict[str, Any]:
    if not ENABLE_AGENT_9:
        return {"decision": True, "rationale":"Ablated: accept finalization.", "guidance_next":"", "confidence":1.0, "missing_items":[]}
    # Compose overview of subgoal answers
    lines = []
    for sg in state.subgoals:
        lines.append(f"{sg.subgoal_id}: {sg.text}\nAnswer: {sg.subgoal_answer[:600]}\nCitations: {', '.join(sg.citations)}")
    overview = "\n\n".join(lines)
    prompt = f"""
You are Agent 9 (Final Context Judge). Determine whether the set of subgoal answers adequately answers the original query.

Criteria:
- Completeness: all user requirements satisfied.
- Coherence: subgoal answers integrate smoothly without contradictions.
- Auditability: key claims trace to chunk_ids via citations.

Return JSON: decision (true/false), rationale, guidance_next, confidence (0-1), missing_items (array).

User language: {lang_tag(state.user_language)}
Original question:
\"\"\"{state.user_query}\"\"\"

Subgoal answers:
\"\"\"{overview[:6000]}\"\"\"
"""
    out = safe_generate_json(prompt, JUDGE_SCHEMA, temp=ANSWER_TEMPERATURES["judge"])
    out["decision"] = bool(out.get("decision", False))
    out["rationale"] = out.get("rationale","")
    out["guidance_next"] = out.get("guidance_next","")
    out["confidence"] = float(out.get("confidence", 0.6))
    out["missing_items"] = out.get("missing_items", []) or []
    return out

def agent10_final_answer_generator(state: RunState) -> Dict[str, Any]:
    if not ENABLE_AGENT_10:
        # Concatenate subgoal answers
        text = "\n\n".join([sg.subgoal_answer for sg in state.subgoals if sg.subgoal_answer])
        cits = sorted({c for sg in state.subgoals for c in (sg.citations or [])})
        disclaimer = "This is general information and not legal advice." if state.user_language=="en" else "Ini adalah informasi umum, bukan nasihat hukum."
        return {"final_answer": text + ("\n\n" + disclaimer if text else disclaimer), "citations": cits, "confidence": 0.5, "notes": disclaimer}
    # Build stitched material
    bundle = []
    for sg in state.subgoals:
        bundle.append(f"[{sg.subgoal_id}] {sg.subgoal_answer}\nCitations: {', '.join(sg.citations)}")
    src = "\n\n".join(bundle)
    disclaimer = "Include a brief 'not legal advice' disclaimer at the end."
    prompt = f"""
You are Agent 10 (Final Answer Generator). Produce the final answer using subgoal answers. Keep it clear, concise, and ensure citations remain as [CHUNK-...] where used. Add a brief 'not legal advice' disclaimer at the end.

Return JSON: final_answer, citations (unique chunk_refs), confidence (0-1), notes (caveats/assumptions).

User language: {lang_tag(state.user_language)}
Original question:
\"\"\"{state.user_query}\"\"\"

Subgoal materials:
\"\"\"{src[:7000]}\"\"\"
"""
    out = safe_generate_json(prompt, FINAL_ANSWER_SCHEMA, temp=ANSWER_TEMPERATURES["fa"], max_tokens=8192)
    final_answer = out.get("final_answer","").strip()
    cits = [c for c in (out.get("citations") or []) if isinstance(c,str) and c.startswith("CHUNK-")]
    if not cits and final_answer:
        cits = sorted(set(re.findall(r"\[(CHUNK-[^\]]+)\]", final_answer)))
    notes = out.get("notes","")
    return {"final_answer": final_answer, "citations": cits, "confidence": float(out.get("confidence", 0.7)), "notes": notes}

# ----------------- Core pipeline -----------------
def new_run_state(user_query: str, user_lang: str) -> RunState:
    return RunState(
        run_id=make_timestamp_name() + "-" + uuid.uuid4().hex[:6],
        user_query=user_query,
        user_language=user_lang,
        config_snapshot=snapshot_config(),
    )

def next_id(state: RunState, kind: str) -> str:
    state.counters[kind] += 1
    if kind == "sg": return f"SG-{state.counters[kind]}"
    if kind == "iq": return f"IQ-{state.counters[kind]}"
    if kind == "ia": return f"IA-{state.counters[kind]}"
    if kind == "sa": return f"SA-{state.counters[kind]}"
    return f"{kind.upper()}-{state.counters[kind]}"

def retrieve_for_iq(state: RunState, iq: IQObject, hyde_docs: List[str]) -> Tuple[List[List[Dict[str, Any]]], List[Dict[str, Any]]]:
    result_sets: List[List[Dict[str, Any]]] = []
    # HyDE variants
    for hd in hyde_docs:
        if not ENABLE_RETRIEVER:
            result_sets.append([])
            continue
        chunks = retrieve_by_text(hd, top_k=RETRIEVE_TOP_K_PER_VARIANT)
        result_sets.append(chunks)
    # Hybrid: raw IQ retrieval as an additional set
    if HYBRID_RETRIEVAL and ENABLE_RETRIEVER:
        result_sets.append(retrieve_by_text(iq.text, top_k=RETRIEVE_TOP_K_PER_VARIANT))
    # Rerank
    reranked = unsupervised_rerank(result_sets, keep_n=RERANKER_KEEP_N, query_text=iq.text)
    return result_sets, reranked

def process_intermediate_question(state: RunState, sg: Subgoal, iq: IQObject) -> None:
    # Loop: HyDE tries and optional IQ rewrites guided by Agent 4
    guidance = ""
    last_guidance = None
    no_op_strikes = 0
    while True:
        # 1) HyDE generation
        iq.tries_hyde += 1
        hyde_docs, hyde_meta = agent3_hyde_generator(state, iq, guidance)
        iq.hyde_docs = hyde_docs

        # 2) Retrieval + rerank
        retrieval_sets, reranked = retrieve_for_iq(state, iq, hyde_docs)
        iq.retrievals = retrieval_sets
        iq.reranked_chunks = reranked[:MAX_CHUNKS_PER_IQ]
        add_chunks_to_evidence(state, iq.reranked_chunks)

        # 3) Judge context sufficiency
        ctx_judgment = agent4_context_judge(state, iq, iq.reranked_chunks, hyde_docs)
        iq.context_judge = ctx_judgment

        if ctx_judgment.get("decision", False) or iq.tries_hyde >= MAX_HYDE_TRIES:
            # Proceed to answer (even if not ideal after exhausting tries)
            ia = agent5_intermediate_answerer(state, iq, iq.reranked_chunks)
            iq.intermediate_answer = ia.get("answer","")
            iq.citations = list(dict.fromkeys(ia.get("citations", [])))
            iq.confidence = float(ia.get("confidence", 0.6))
            ans_id = next_id(state, "ia")
            record_answer_citations(state, ans_id, iq.citations)
            return

        # Not sufficient; consider guidance and possibly rewrite IQ
        guidance = ctx_judgment.get("guidance_next","") or ""
        if NO_OP_GUARD:
            if last_guidance == guidance or not guidance.strip():
                no_op_strikes += 1
            else:
                no_op_strikes = 0
        last_guidance = guidance

        # Decide whether to rewrite IQ if HyDE tries are used
        if iq.tries_hyde >= MAX_HYDE_TRIES and iq.rewrites < MAX_IQ_REWRITES and ENABLE_AGENT_2:
            # Rewrite IQ using Agent 2 with guidance
            iq.rewrites += 1
            iqs, _meta = agent2_iq_generator(state, sg, guidance)
            # Replace IQ text with the first rewritten item
            if iqs:
                iq.text = iqs[0]
            # reset hyde tries
            iq.tries_hyde = 0
            continue

        if NO_OP_GUARD and no_op_strikes >= 2:
            # Escalate: stop trying
            ia = agent5_intermediate_answerer(state, iq, iq.reranked_chunks)
            iq.intermediate_answer = ia.get("answer","")
            iq.citations = list(dict.fromkeys(ia.get("citations", [])))
            iq.confidence = float(ia.get("confidence", 0.5))
            ans_id = next_id(state, "ia")
            record_answer_citations(state, ans_id, iq.citations)
            return
        # else: loop again with new HyDE attempts

def process_subgoal(state: RunState, sg: Subgoal) -> None:
    refinements = 0
    coverage_note = ""
    guidance = ""
    while True:
        # Agent 2: IQ generation
        iqs, meta = agent2_iq_generator(state, sg, guidance)
        coverage_note = meta.get("coverage_note","")
        sg.intermediate_questions = []
        for q in iqs:
            iq = IQObject(iq_id=next_id(state, "iq"), text=q)
            sg.intermediate_questions.append(iq)

        # For each IQ
        for iq in sg.intermediate_questions:
            process_intermediate_question(state, sg, iq)

        # Agent 6: Judge subgoal readiness
        judg = agent6_subgoal_context_judge(state, sg, coverage_note)
        sg.judge_reports.append(judg)
        if judg.get("decision", False) and (not CONFIDENCE_GATES or judg.get("confidence", 0.0) >= JUDGE_CONFIDENCE_THRESHOLD):
            # Agent 7: Compose subgoal answer
            sa = agent7_subgoal_answerer(state, sg)
            sg.subgoal_answer = sa.get("answer","")
            sg.citations = list(dict.fromkeys(sa.get("citations", [])))
            sg.confidence = float(sa.get("confidence", 0.7))
            ans_id = next_id(state, "sa")
            record_answer_citations(state, ans_id, sg.citations)
            sg.status = "done"
            return
        # Not ready: refine loop or stop
        refinements += 1
        if refinements >= MAX_SUBGOAL_REFINEMENTS:
            # Best-effort composition anyway
            sa = agent7_subgoal_answerer(state, sg)
            sg.subgoal_answer = sa.get("answer","")
            sg.citations = list(dict.fromkeys(sa.get("citations", [])))
            sg.confidence = float(sa.get("confidence", 0.5))
            ans_id = next_id(state, "sa")
            record_answer_citations(state, ans_id, sg.citations)
            sg.status = "done"
            return
        guidance = judg.get("guidance_next","")

def agentic_rag(user_query: str) -> Dict[str, Any]:
    global _LOGGER
    ts_name = make_timestamp_name()
    log_file = Path.cwd() / f"{ts_name}.txt"
    _LOGGER = FileLogger(log_file, also_console=True)
    try:
        log("=== Agentic RAG (10-agent) started ===")
        log(f"Log file: {log_file}")

        # Phase 0: Initialization
        if LANGUAGE_MODE == "auto":
            user_lang, lang_conf = detect_user_language(user_query)
        else:
            user_lang, lang_conf = (LANGUAGE_MODE, 1.0)
        state = new_run_state(user_query, user_lang)
        state.telemetry["language_detector"] = {"language": user_lang, "confidence": lang_conf}
        log(f"[Language] Detected: {user_lang} (confidence={lang_conf:.2f})")

        # Phase 1..3 with global passes
        for gpass in range(1, MAX_GLOBAL_PASSES+1):
            log(f"\n--- Global Pass {gpass}/{MAX_GLOBAL_PASSES} ---")
            # Agent 1: Planning
            subgoals_text, plan_meta = agent1_subgoal_generator(state)
            state.subgoals = []
            for t in subgoals_text:
                sg = Subgoal(subgoal_id=next_id(state, "sg"), text=t, dependencies=[])
                state.subgoals.append(sg)
            log(f"[Plan] Subgoals: {[sg.subgoal_id + ' ' + sg.text for sg in state.subgoals]}")

            # Phase 2: Per subgoal
            for i, sg in enumerate(state.subgoals):
                log(f"\n[Subgoal {sg.subgoal_id}] {sg.text}")
                with Timer(f"subgoal_{sg.subgoal_id}", state):
                    process_subgoal(state, sg)

                # Agent 8: Modify pending subgoals based on completed sg
                pending = [x for x in state.subgoals if x.status != "done" and x.subgoal_id != sg.subgoal_id]
                if pending:
                    mods, mod_meta = agent8_subgoal_modifier(state, sg, pending)
                    # Apply modifications preserving order where possible
                    # Map new texts to existing pending IDs in order
                    new_list = []
                    for idx, text in enumerate(mods):
                        if idx < len(pending):
                            # reuse object but update text
                            pending[idx].text = text
                            new_list.append(pending[idx])
                        else:
                            # append new subgoal
                            new_sg = Subgoal(subgoal_id=next_id(state, "sg"), text=text)
                            new_list.append(new_sg)
                    # Replace pending portion of state.subgoals
                    done_part = [x for x in state.subgoals if x.status == "done"]
                    state.subgoals = done_part + new_list
                    sg.history.append({"modified_pending": [x.subgoal_id for x in new_list], "change_log": mod_meta.get("change_log","")})

            # Phase 3: Finalization
            judge_final = agent9_final_context_judge(state)
            state.final["judge_reports"].append(judge_final)
            if judge_final.get("decision", False) and (not CONFIDENCE_GATES or judge_final.get("confidence",0.0) >= JUDGE_CONFIDENCE_THRESHOLD):
                fa = agent10_final_answer_generator(state)
                state.final.update({"status":"done","final_answer":fa.get("final_answer",""),"confidence":fa.get("confidence",0.7)})
                final_cits = list(dict.fromkeys(fa.get("citations", [])))
                state.final["citations"] = final_cits
                # add disclaimer if missing
                if state.user_language == "en":
                    disclaimer = "This is general information and not legal advice."
                else:
                    disclaimer = "Ini adalah informasi umum, bukan nasihat hukum."
                if "legal advice" not in state.final["final_answer"].lower() and "nasihat hukum" not in state.final["final_answer"].lower():
                    state.final["final_answer"] = (state.final["final_answer"].rstrip() + "\n\n" + disclaimer).strip()
                break
            else:
                # Need re-planning if budget allows
                state.guidance_from_agent9 = judge_final.get("guidance_next","")
                if gpass >= MAX_GLOBAL_PASSES:
                    # produce best-effort final regardless
                    fa = agent10_final_answer_generator(state)
                    state.final.update({"status":"done","final_answer":fa.get("final_answer",""),"confidence":fa.get("confidence",0.5)})
                    final_cits = list(dict.fromkeys(fa.get("citations", [])))
                    state.final["citations"] = final_cits
                    if state.user_language == "en":
                        disclaimer = "This is general information and not legal advice."
                    else:
                        disclaimer = "Ini adalah informasi umum, bukan nasihat hukum."
                    if "legal advice" not in state.final["final_answer"].lower() and "nasihat hukum" not in state.final["final_answer"].lower():
                        state.final["final_answer"] = (state.final["final_answer"].rstrip() + "\n\n" + disclaimer).strip()
                    break
                else:
                    log("[Final Judge] Re-planning requested. Looping to Agent 1 with guidance.")
                    continue

        # Telemetry and summary
        log("\n=== Evidence Map ===")
        if STORE_EVIDENCE_MAP:
            log(json.dumps({"answers_to_chunks": state.evidence.answers_to_chunks,
                            "chunk_metadata_index": state.evidence.chunk_metadata_index}, ensure_ascii=False)[:4000] + (" ... (truncated)" if len(json.dumps(state.evidence.chunk_metadata_index))>4000 else ""))

        log("\n=== Run Summary ===")
        log(f"Run ID: {state.run_id}")
        log(f"Language: {state.user_language}")
        log(f"Subgoals: {[sg.subgoal_id for sg in state.subgoals]}")
        log(f"Final decision confidence: {state.final.get('confidence')}")
        log("\n=== Final Answer ===")
        log(state.final.get("final_answer",""))
        log(f"\nLogs saved to: {log_file}")

        return {
            "run_id": state.run_id,
            "final_answer": state.final.get("final_answer",""),
            "citations": state.final.get("citations", []),
            "confidence": state.final.get("confidence", 0.0),
            "subgoals": [asdict(sg) for sg in state.subgoals],
            "evidence": asdict(state.evidence) if STORE_EVIDENCE_MAP else {},
            "telemetry": state.telemetry,
            "log_file": str(log_file),
            "judge_reports": state.final.get("judge_reports", [])
        }
    finally:
        if _LOGGER is not None:
            _LOGGER.close()

# ----------------- Main -----------------
if __name__ == "__main__":
    try:
        print("Agentic RAG (10-agent) — Legal Q&A (id/en)")
        user_query = input("Enter your query: ").strip()
        if not user_query:
            print("Empty query. Exiting.")
        else:
            out = agentic_rag(user_query)
            # Pretty-print concise results to console
            print("\n--- Final Answer ---\n")
            print(out.get("final_answer",""))
            if out.get("citations"):
                print("\nCitations:", ", ".join(out["citations"]))
            print(f"\nConfidence: {out.get('confidence')}")
            print(f"Run ID: {out.get('run_id')}")
            print(f"Log file: {out.get('log_file')}")
    finally:
        try:
            driver.close()
        except Exception:
            pass