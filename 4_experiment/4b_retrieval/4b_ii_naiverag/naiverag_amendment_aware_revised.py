#!/usr/bin/env python3
# agentic_rag_with_amendments.py
# Amendment-Aware Agentic RAG for Indonesian Legal Documents
# Handles amendments, repeals, and partial modifications

import os, time, json, pickle, re, random
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
from collections import deque

from dotenv import load_dotenv
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
from neo4j import GraphDatabase
import numpy as np

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
TOP_K_CHUNKS = int(os.getenv("TOP_K_CHUNKS", "40"))
MAX_CHUNKS_FINAL = int(os.getenv("MAX_CHUNKS_FINAL", "40"))
CHUNK_TEXT_CLAMP = int(os.getenv("CHUNK_TEXT_CLAMP", "1000000"))

# Agent loop
ANSWER_MAX_TOKENS = int(os.getenv("ANSWER_MAX_TOKENS", "4096"))
MAX_ITERS = 1

# Hardcoded LLM rate limit
LLM_CALLS_PER_MINUTE = 13

# ----------------- Initialize SDKs -----------------
genai.configure(api_key=GOOGLE_API_KEY)
gen_model = genai.GenerativeModel(GEN_MODEL)
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

# ----------------- LLM Call/Token Tracking -----------------
class LLMUsageTracker:
    def __init__(self):
        self.total_calls = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.call_details = []
    
    def record_call(self, call_type: str, prompt_tokens: int, completion_tokens: int):
        self.total_calls += 1
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.call_details.append({
            "type": call_type,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "timestamp": time.time()
        })
    
    def get_summary(self) -> Dict[str, Any]:
        return {
            "total_llm_calls": self.total_calls,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_prompt_tokens + self.total_completion_tokens,
            "call_details": self.call_details
        }

_USAGE_TRACKER = LLMUsageTracker()

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

# --- Simple per-process rate limiter (calls/minute) ---
class RateLimiter:
    def __init__(self, calls_per_minute: int, name: str = "LLM"):
        self.calls_per_minute = max(0, int(calls_per_minute))
        self.name = name
        self.window = deque()
        self.window_seconds = 60.0

    def wait_for_slot(self):
        if self.calls_per_minute <= 0:
            return
        while True:
            now = time.monotonic()
            while self.window and (now - self.window[0]) >= self.window_seconds:
                self.window.popleft()
            if len(self.window) < self.calls_per_minute:
                self.window.append(now)
                return
            sleep_time = self.window_seconds - (now - self.window[0])
            if sleep_time > 0:
                log(f"[RateLimit:{self.name}] Sleeping {sleep_time:.2f}s to respect {self.name}_CALLS_PER_MINUTE={self.calls_per_minute}")
                time.sleep(sleep_time)
            else:
                time.sleep(0.01)

EMBEDDING_CALLS_PER_MINUTE = int(os.getenv("EMBEDDING_CALLS_PER_MINUTE", "0"))

_LLM_RATE_LIMITER = RateLimiter(LLM_CALLS_PER_MINUTE)
_EMBED_RATE_LIMITER = RateLimiter(EMBEDDING_CALLS_PER_MINUTE)

def _rand_wait_seconds() -> float:
    return random.uniform(5.0, 20.0)

def _api_call_with_retry(func, *args, **kwargs):
    while True:
        try:
            _LLM_RATE_LIMITER.wait_for_slot()
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
        return res.embedding.values
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

def safe_generate_json(prompt: str, schema: Dict[str, Any], temp: float = 0.0, call_type: str = "json") -> Dict[str, Any]:
    cfg = GenerationConfig(temperature=temp, response_mime_type="application/json", response_schema=schema)
    resp = _api_call_with_retry(gen_model.generate_content, prompt, generation_config=cfg)
    
    # Track usage
    prompt_tokens = estimate_tokens_for_text(prompt)
    try:
        if isinstance(resp.text, str) and resp.text.strip():
            completion_tokens = estimate_tokens_for_text(resp.text)
            _USAGE_TRACKER.record_call(call_type, prompt_tokens, completion_tokens)
            return json.loads(resp.text)
    except Exception:
        pass
    try:
        raw = resp.candidates[0].content.parts[0].text
        completion_tokens = estimate_tokens_for_text(raw)
        _USAGE_TRACKER.record_call(call_type, prompt_tokens, completion_tokens)
        return json.loads(raw)
    except Exception as e:
        info = get_finish_info(resp)
        log(f"[LLM JSON parse warning] No JSON content returned. Diagnostics: {info}. Error: {e}")
        _USAGE_TRACKER.record_call(call_type, prompt_tokens, 0)
        return {}

def safe_generate_text(prompt: str, max_tokens: int, temperature: float = 0.2, call_type: str = "text") -> str:
    cfg = GenerationConfig(temperature=temperature, max_output_tokens=max_tokens)
    resp = _api_call_with_retry(gen_model.generate_content, prompt, generation_config=cfg)
    
    # Track usage
    prompt_tokens = estimate_tokens_for_text(prompt)
    text = extract_text_from_response(resp)
    if text:
        completion_tokens = estimate_tokens_for_text(text)
        _USAGE_TRACKER.record_call(call_type, prompt_tokens, completion_tokens)
        return text
    
    info = get_finish_info(resp)
    log(f"[LLM text warning] No text returned. Diagnostics: {info}")
    _USAGE_TRACKER.record_call(call_type, prompt_tokens, 0)
    return f"(Model returned no text. finish_info={info})"

# ----------------- UU Format Conversion -----------------
def extract_number_year_from_uu(uu_text: str) -> Optional[Tuple[int, int]]:
    """
    Extract number and year from any UU format.
    Returns (number, year) or None
    """
    if not uu_text:
        return None
    
    # Handle AMD_ prefix
    if uu_text.startswith("AMD_"):
        uu_text = uu_text.replace("AMD_", "")
    
    # If already in X_Y format
    match = re.match(r'^(\d+)_(\d{4})$', uu_text)
    if match:
        return (int(match.group(1)), int(match.group(2)))
    
    # Extract from full format
    patterns = [
        r'(?:UU|Undang-undang|Undang-Undang)\s*(?:\(UU\))?\s*(?:Nomor|No\.?)\s*(\d+)\s*Tahun\s*(\d{4})',
        r'(?:UU|Undang-undang|Undang-Undang)\s*(\d+)/(\d{4})',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, uu_text, re.IGNORECASE)
        if match:
            return (int(match.group(1)), int(match.group(2)))
    
    return None

def convert_to_db_formats(uu_identifier: str) -> List[str]:
    """
    Convert any UU identifier format to the two database formats:
    - "Undang-undang (UU) Nomor X Tahun Y"
    - "Undang-undang (UU) No. X Tahun Y"
    
    Args:
        uu_identifier: Can be "AMD_16_2025", "16_2025", or any other format
    
    Returns:
        List of both possible database format strings
    """
    result = extract_number_year_from_uu(uu_identifier)
    if not result:
        return []
    
    number, year = result
    return [
        f"Undang-undang (UU) Nomor {number} Tahun {year}",
        f"Undang-undang (UU) No. {number} Tahun {year}"
    ]

# ----------------- UU Reference Extraction (LLM-based) -----------------
UU_REFERENCE_SCHEMA = {
    "type": "object",
    "properties": {
        "references": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "number": {"type": "integer"},
                    "year": {"type": "integer"},
                    "context": {"type": "string"}
                },
                "required": ["number", "year"]
            }
        }
    },
    "required": ["references"]
}

def extract_uu_references_llm(answer_text: str) -> List[Dict[str, Any]]:
    """
    Extract UU references from answer text using LLM.
    Returns list of {number, year, context}
    """
    prompt = f"""
Extract all Indonesian law (Undang-undang/UU) references from the following text.

For each reference, identify:
- number: The UU number (integer)
- year: The year (integer)
- context: Brief context about what aspect this UU addresses (optional)

Common formats include:
- "UU No. 16 Tahun 2025"
- "Undang-undang Nomor 16 Tahun 2025"
- "UU 16/2025"
- "Undang-Undang (UU) No. 16 Tahun 2025"

Return JSON with "references" array containing objects with number, year, and optional context.

Text:
\"\"\"
{answer_text}
\"\"\"
"""
    
    log("[UU Reference Extractor] Extracting references from initial answer...")
    result = safe_generate_json(prompt, UU_REFERENCE_SCHEMA, temp=0.0, call_type="reference_extraction")
    
    references = result.get("references", [])
    log(f"[UU Reference Extractor] Found {len(references)} UU references: {[(r.get('number'), r.get('year')) for r in references]}")
    
    return references

# ----------------- Amendment Chain Traversal -----------------
def normalize_uu_identifier(number: int, year: int) -> str:
    """Convert number and year to AMD graph format: AMD_X_Y"""
    return f"AMD_{number}_{year}"

def get_outgoing_amendments(uu_key: str) -> List[Dict[str, Any]]:
    """
    Get all outgoing amendment relationships from a UU node.
    Returns list of {target_key, relationship_type}
    """
    cypher = """
    MATCH (source:AMD_UndangUndang {key: $uu_key})-[r]->(target:AMD_UndangUndang)
    WHERE type(r) IN ['AMD_DIUBAH_DENGAN', 'AMD_DIUBAH_SEBAGIAN_DENGAN', 
                       'AMD_DICABUT_DENGAN', 'AMD_DICABUT_SEBAGIAN_DENGAN']
    RETURN target.key AS target_key, 
           target.number AS target_number,
           target.year AS target_year,
           type(r) AS relationship_type
    """
    
    rows = run_cypher_with_retry(cypher, {"uu_key": uu_key})
    
    results = []
    for row in rows:
        results.append({
            "target_key": row["target_key"],
            "target_number": row["target_number"],
            "target_year": row["target_year"],
            "relationship_type": row["relationship_type"]
        })
    
    return results

def traverse_amendment_chain_with_reset(start_number: int, start_year: int) -> Dict[str, Any]:
    """
    Traverse amendment chain with repeal reset logic.
    
    Rules:
    - Collect all nodes in chain
    - If AMD_DICABUT_DENGAN encountered: discard all previous, start fresh from target
    - Continue traversing downstream
    
    Returns:
    {
        "original_uu": "AMD_X_Y",
        "relevant_uus": ["AMD_A_B", "AMD_C_D", ...],  # UUs to retrieve from
        "amendment_info": [
            {"from": "AMD_X_Y", "to": "AMD_A_B", "type": "AMD_DIUBAH_DENGAN"},
            ...
        ],
        "has_amendments": bool
    }
    """
    start_key = normalize_uu_identifier(start_number, start_year)
    
    log(f"[Amendment Traversal] Starting from {start_key} (UU {start_number}/{start_year})")
    
    # Check if node exists in amendment graph
    check_cypher = "MATCH (u:AMD_UndangUndang {key: $key}) RETURN u"
    exists = run_cypher_with_retry(check_cypher, {"key": start_key})
    
    if not exists:
        log(f"[Amendment Traversal] {start_key} not found in amendment graph. No amendments.")
        return {
            "original_uu": start_key,
            "relevant_uus": [start_key],
            "amendment_info": [],
            "has_amendments": False
        }
    
    # Traverse chain with reset logic
    relevant_uus = [start_key]
    amendment_info = []
    visited = set()
    queue = [start_key]
    
    while queue:
        current = queue.pop(0)
        
        if current in visited:
            continue
        visited.add(current)
        
        outgoing = get_outgoing_amendments(current)
        
        if not outgoing:
            # Terminal node
            continue
        
        for rel in outgoing:
            target = rel["target_key"]
            rel_type = rel["relationship_type"]
            
            amendment_info.append({
                "from": current,
                "to": target,
                "type": rel_type,
                "target_number": rel["target_number"],
                "target_year": rel["target_year"]
            })
            
            if rel_type == "AMD_DICABUT_DENGAN":
                # RESET: Discard all previous UUs, start fresh from target
                log(f"[Amendment Traversal] REPEAL encountered: {current} → {target}. Resetting collection.")
                relevant_uus = [target]
            else:
                # AMD_DIUBAH_DENGAN, AMD_DIUBAH_SEBAGIAN_DENGAN, AMD_DICABUT_SEBAGIAN_DENGAN
                if target not in relevant_uus:
                    relevant_uus.append(target)
            
            queue.append(target)
    
    has_amendments = len(relevant_uus) > 1 or len(amendment_info) > 0
    
    log(f"[Amendment Traversal] Relevant UUs for retrieval: {relevant_uus}")
    log(f"[Amendment Traversal] Amendment relationships found: {len(amendment_info)}")
    
    return {
        "original_uu": start_key,
        "relevant_uus": relevant_uus,
        "amendment_info": amendment_info,
        "has_amendments": has_amendments
    }

# ----------------- Currency Warning Generation -----------------
AMENDMENT_TYPE_NAMES = {
    "AMD_DIUBAH_DENGAN": "diubah sepenuhnya dengan",
    "AMD_DIUBAH_SEBAGIAN_DENGAN": "diubah sebagian dengan",
    "AMD_DICABUT_DENGAN": "dicabut dengan",
    "AMD_DICABUT_SEBAGIAN_DENGAN": "dicabut sebagian dengan"
}

def format_uu_display(uu_key: str) -> str:
    """Convert AMD_16_2025 to display format 'UU No. 16 Tahun 2025'"""
    parts = uu_key.replace("AMD_", "").split("_")
    if len(parts) == 2:
        return f"UU No. {parts[0]} Tahun {parts[1]}"
    return uu_key

def generate_currency_warning(chain_results: List[Dict[str, Any]]) -> str:
    """
    Generate currency warning text for amended UUs.
    
    Input: List of chain traversal results
    Output: Formatted warning text
    """
    warnings = []
    
    for chain in chain_results:
        if not chain["has_amendments"]:
            continue
        
        original = format_uu_display(chain["original_uu"])
        
        # Group amendments by type
        amendments_by_type = {}
        for amd in chain["amendment_info"]:
            rel_type = amd["type"]
            target_display = f"UU No. {amd['target_number']} Tahun {amd['target_year']}"
            
            if rel_type not in amendments_by_type:
                amendments_by_type[rel_type] = []
            amendments_by_type[rel_type].append(target_display)
        
        # Build warning text
        amendment_descriptions = []
        for rel_type, targets in amendments_by_type.items():
            type_name = AMENDMENT_TYPE_NAMES.get(rel_type, rel_type)
            targets_str = ", ".join(targets)
            amendment_descriptions.append(f"{type_name} {targets_str}")
        
        warning = f"⚠️ {original} telah {'; '.join(amendment_descriptions)}. Memeriksa ketentuan terbaru..."
        warnings.append(warning)
    
    if warnings:
        return "\n".join(warnings)
    else:
        return ""

# ----------------- Vector search with filtering (Two-Stage Approach) -----------------
def vector_query_chunks_filtered(q_emb: List[float], k: int, uu_filters: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """
    Query Neo4j vector index over TextChunk with optional UU filtering.
    
    Uses two-stage approach when filtering:
    1. Filter chunks by uu_number first (MATCH query)
    2. Manually compute cosine similarity scores
    3. Return top-k by score
    
    This avoids the limitation where vector index WHERE clause filters AFTER top-k retrieval.
    
    Args:
        q_emb: Query embedding
        k: Number of results to return
        uu_filters: List of UU identifiers in any format (e.g., ["AMD_16_2025", "AMD_5_2023"])
                   These will be converted to database formats for filtering
    
    Returns:
        List of chunk dictionaries with scores
    """
    
    if uu_filters:
        # Convert all filter identifiers to database formats
        db_format_filters = []
        for uu_id in uu_filters:
            formats = convert_to_db_formats(uu_id)
            db_format_filters.extend(formats)
        
        log(f"[Vector Filter] Input filters: {uu_filters}")
        log(f"[Vector Filter] Converted to DB formats: {db_format_filters}")
        
        # Stage 1: Get all chunks from target UUs with their embeddings
        cypher = """
        MATCH (c:TextChunk)
        WHERE c.uu_number IN $uu_filters
        RETURN c AS node, c.embedding AS embedding
        """
        rows = run_cypher_with_retry(cypher, {"uu_filters": db_format_filters})
        
        log(f"[Vector Filter] Found {len(rows)} chunks from target UUs")
        
        if not rows:
            log(f"[Vector Filter] No chunks found for filters: {db_format_filters}")
            return []
        
        # Stage 2: Manually compute cosine similarity scores
        results = []
        q_array = np.array(q_emb, dtype=np.float32)
        q_norm = np.linalg.norm(q_array)
        
        chunks_without_embedding = 0
        
        for r in rows:
            node = r["node"]
            embedding = r.get("embedding")
            
            if embedding is None or len(embedding) == 0:
                chunks_without_embedding += 1
                continue
            
            # Compute cosine similarity
            try:
                emb_array = np.array(embedding, dtype=np.float32)
                emb_norm = np.linalg.norm(emb_array)
                
                if emb_norm == 0 or q_norm == 0:
                    score = 0.0
                else:
                    score = float(np.dot(q_array, emb_array) / (q_norm * emb_norm))
                
                results.append({
                    "node": node,
                    "score": score
                })
            except Exception as e:
                log(f"[Vector Filter] Error computing similarity for chunk {node.get('chunk_id')}: {e}")
                continue
        
        if chunks_without_embedding > 0:
            log(f"[Vector Filter] Warning: {chunks_without_embedding} chunks had no embedding")
        
        # Stage 3: Sort by score and take top-k
        results.sort(key=lambda x: x["score"], reverse=True)
        results = results[:k]
        
        log(f"[Vector Filter] Returning top {len(results)} chunks by similarity score")
        
        # Format output
        out = []
        for r in results:
            c = r["node"]
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
    
    else:
        # No filtering - use vector index directly
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
        
        log(f"[Vector Search] Retrieved {len(out)} chunks from vector index")
        return out

# ----------------- Build context -----------------
def clamp(s: Optional[str], n: int) -> str:
    t = (s or "").strip()
    return t[:n]

def build_context_from_chunks(chunks: List[Dict[str, Any]], max_chunks: int) -> str:
    chosen = chunks[:max_chunks]
    lines = ["Potongan teks terkait (chunk):\n"]
    for i, c in enumerate(chosen, 1):
        doc = c.get("document_id")
        cid = c.get("chunk_id")
        uu = c.get("uu_number") or ""
        pg = c.get("pages")
        txt = clamp(c.get("content") or "", CHUNK_TEXT_CLAMP)
        lines.append(f"\n[Chunk {i}] doc={doc} chunk={cid} | {uu} | pages={pg} | score={c.get('score'):.3f}\n{txt}\n")
    return "".join(lines)

# ----------------- Agent 2 (Initial Answerer) -----------------
def agent2_initial_answer(query_original: str, context: str, output_lang: str = "id") -> str:
    instructions = (
        "Answer concisely and accurately based strictly on the provided context. "
        "IMPORTANT: You MUST cite specific UU references (e.g., 'UU No. 16 Tahun 2025' or 'Undang-undang Nomor 16 Tahun 2025') "
        "when they are mentioned in the context. This is critical for amendment checking. "
        "Respond in the same language as the user's question."
    )
    
    prompt = f"""
You are an Indonesian legal document answering agent. Provide an answer based on the context only.

Core instructions:
{instructions}

Original user question:
\"\"\"{query_original}\"\"\"

Context:
\"\"\"{context}\"\"\"
"""
    est = estimate_tokens_for_text(prompt)
    log("[Initial Answerer] Prompt:")
    log(prompt[:1000] + "..." if len(prompt) > 1000 else prompt)
    log(f"[Initial Answerer] Prompt size: {len(prompt)} chars, est_tokens≈{est}")
    
    answer = safe_generate_text(prompt, max_tokens=ANSWER_MAX_TOKENS, temperature=0.2, call_type="initial_answer")
    log("[Initial Answerer] Answer:")
    log(answer)
    return answer

# ----------------- Relevance Judge Agent -----------------
RELEVANCE_JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "is_relevant": {"type": "boolean"},
        "affected_aspects": {
            "type": "array",
            "items": {"type": "string"}
        },
        "reasoning": {"type": "string"}
    },
    "required": ["is_relevant", "affected_aspects", "reasoning"]
}

def judge_amendment_relevance(query_original: str, initial_answer: str, new_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Judge whether new chunks from amending documents are relevant to the original question.
    
    Returns:
    {
        "is_relevant": bool,
        "affected_aspects": [list of aspects],
        "reasoning": str
    }
    """
    new_context = build_context_from_chunks(new_chunks, max_chunks=MAX_CHUNKS_FINAL)
    
    prompt = f"""
You are a legal relevance judge. Determine if the new legal provisions from amending documents 
contain information that addresses or modifies the answer to the user's original question.

Original user question:
\"\"\"{query_original}\"\"\"

Initial answer (based on older law):
\"\"\"{initial_answer}\"\"\"

New provisions from amending documents:
\"\"\"{new_context}\"\"\"

Task:
1. Determine if the new provisions are relevant to the original question
2. If relevant, identify which aspects of the initial answer are affected
3. Provide brief reasoning

Output JSON with:
- is_relevant: boolean (true if new provisions address the question)
- affected_aspects: array of strings (which parts of initial answer are affected)
- reasoning: string (brief explanation)
"""
    
    log("[Relevance Judge] Evaluating amendment relevance...")
    result = safe_generate_json(prompt, RELEVANCE_JUDGE_SCHEMA, temp=0.0, call_type="relevance_judge")
    
    is_relevant = result.get("is_relevant", False)
    affected_aspects = result.get("affected_aspects", [])
    reasoning = result.get("reasoning", "")
    
    log(f"[Relevance Judge] Is relevant: {is_relevant}")
    log(f"[Relevance Judge] Affected aspects: {affected_aspects}")
    log(f"[Relevance Judge] Reasoning: {reasoning}")
    
    return result

# ----------------- Amendment Integration Agent -----------------
def integrate_amendments(query_original: str, initial_answer: str, new_context: str, 
                        amendment_info: List[Dict[str, Any]], relevance_result: Dict[str, Any],
                        output_lang: str = "id") -> str:
    """
    Integrate new information from amending documents into the initial answer.
    
    The LLM should:
    - Add new relevant information from new context
    - Specify document sources for each aspect
    - Mark which information is currently operative
    - Handle partial amendments (only some aspects affected)
    """
    
    # Format amendment information
    amendment_descriptions = []
    for amd in amendment_info:
        from_uu = format_uu_display(amd["from"])
        to_uu = format_uu_display(amd["to"])
        rel_type_display = AMENDMENT_TYPE_NAMES.get(amd["type"], amd["type"])
        amendment_descriptions.append(f"- {from_uu} {rel_type_display} {to_uu}")
    
    amendments_text = "\n".join(amendment_descriptions) if amendment_descriptions else "Tidak ada amandemen"
    
    affected_aspects_text = ", ".join(relevance_result.get("affected_aspects", [])) if relevance_result.get("affected_aspects") else "Semua aspek"
    
    prompt = f"""
You are a legal amendment integration agent. Your task is to update the initial answer with 
new information from amending documents.

Original user question:
\"\"\"{query_original}\"\"\"

Initial answer (from older law):
\"\"\"{initial_answer}\"\"\"

Amendment information:
{amendments_text}

Affected aspects identified:
{affected_aspects_text}

New provisions from amending documents:
\"\"\"{new_context}\"\"\"

Task:
1. Integrate the new information into the initial answer
2. Clearly mark which parts come from which document
3. Specify which provisions are currently operative
4. If only some aspects are affected, clearly indicate which parts of the initial answer remain valid
5. Use format like: "[Aspek X berdasarkan UU A/B]: ... [DIPERBARUI oleh UU C/D]: ..."

Important:
- Be precise about which specific aspects are updated vs. unchanged
- Cite document sources clearly
- Make it clear which law is currently in effect for each aspect
- Respond in {output_lang}
"""
    
    log("[Amendment Integrator] Integrating amendments into final answer...")
    log(f"[Amendment Integrator] Prompt size: {len(prompt)} chars")
    
    integrated_answer = safe_generate_text(prompt, max_tokens=ANSWER_MAX_TOKENS, temperature=0.2, call_type="amendment_integration")
    
    log("[Amendment Integrator] Integrated answer:")
    log(integrated_answer)
    
    return integrated_answer

# ----------------- Main Amendment-Aware RAG -----------------
def agentic_rag_with_amendments(query_original: str) -> Dict[str, Any]:
    global _LOGGER, _USAGE_TRACKER
    
    # Reset tracker for this query
    _USAGE_TRACKER = LLMUsageTracker()
    
    ts_name = make_timestamp_name()
    log_file = Path.cwd() / f"{ts_name}.txt"
    _LOGGER = FileLogger(log_file, also_console=True)

    try:
        log("=== Amendment-Aware Agentic RAG run started ===")
        log(f"Log file: {log_file}")
        log(f"Original Query: {query_original}")
        log(f"Parameters: TOP_K_CHUNKS={TOP_K_CHUNKS}, MAX_CHUNKS_FINAL={MAX_CHUNKS_FINAL}")
        log("")

        user_lang = detect_user_language(query_original)
        log(f"[Language] Detected user language: {user_lang}")
        log("")

        # ============================================================
        # STEP 1: Initial RAG Pass
        # ============================================================
        log("=" * 60)
        log("STEP 1: Initial RAG Pass")
        log("=" * 60)
        
        t0 = time.time()
        q_emb = embed_text(query_original)
        log(f"[Step 1.1] Embedded query in {(time.time()-t0)*1000:.0f} ms")
        
        t1 = time.time()
        initial_candidates = vector_query_chunks_filtered(q_emb, k=TOP_K_CHUNKS, uu_filters=None)
        log(f"[Step 1.2] Vector search returned {len(initial_candidates)} candidates in {(time.time()-t1)*1000:.0f} ms")
        
        if not initial_candidates:
            final_answer = "Tidak ada potongan teks yang ditemukan." if user_lang == "id" else "No relevant chunks found."
            log("No chunks found. Returning empty result.")
            
            usage_summary = _USAGE_TRACKER.get_summary()
            log("")
            log("=" * 60)
            log("LLM USAGE SUMMARY")
            log("=" * 60)
            log(json.dumps(usage_summary, indent=2, ensure_ascii=False))
            
            return {
                "final_answer": final_answer,
                "has_amendments": False,
                "currency_warnings": "",
                "log_file": str(log_file),
                "llm_usage": usage_summary
            }
        
        # Build context and generate initial answer
        initial_context = build_context_from_chunks(initial_candidates, max_chunks=MAX_CHUNKS_FINAL)
        
        t2 = time.time()
        initial_answer = agent2_initial_answer(query_original, initial_context, output_lang=user_lang)
        log(f"[Step 1.3] Initial answer generated in {(time.time()-t2)*1000:.0f} ms")
        log("")
        
        # ============================================================
        # STEP 2: Extract UU References from Initial Answer
        # ============================================================
        log("=" * 60)
        log("STEP 2: Extract UU References")
        log("=" * 60)
        
        t3 = time.time()
        uu_references = extract_uu_references_llm(initial_answer)
        log(f"[Step 2] Reference extraction completed in {(time.time()-t3)*1000:.0f} ms")
        log("")
        
        if not uu_references:
            log("[Step 2] No UU references found in initial answer. Using initial answer as final.")
            final_answer = initial_answer
            
            usage_summary = _USAGE_TRACKER.get_summary()
            log("")
            log("=" * 60)
            log("LLM USAGE SUMMARY")
            log("=" * 60)
            log(json.dumps(usage_summary, indent=2, ensure_ascii=False))
            
            return {
                "final_answer": final_answer,
                "has_amendments": False,
                "currency_warnings": "",
                "log_file": str(log_file),
                "llm_usage": usage_summary
            }
        
        # ============================================================
        # STEP 3: Check for Amendments
        # ============================================================
        log("=" * 60)
        log("STEP 3: Check for Amendments")
        log("=" * 60)
        
        t4 = time.time()
        chain_results = []
        all_amending_uus = set()
        all_amendment_info = []
        
        for ref in uu_references:
            number = ref.get("number")
            year = ref.get("year")
            
            if number is None or year is None:
                log(f"[Step 3] Skipping invalid reference: {ref}")
                continue
            
            chain_result = traverse_amendment_chain_with_reset(number, year)
            chain_results.append(chain_result)
            
            if chain_result["has_amendments"]:
                # Collect amending UUs (excluding the original)
                for uu in chain_result["relevant_uus"]:
                    if uu != chain_result["original_uu"]:
                        # Keep in AMD_ format; filtering function will convert to DB formats
                        all_amending_uus.add(uu)
                
                all_amendment_info.extend(chain_result["amendment_info"])
        
        log(f"[Step 3] Amendment check completed in {(time.time()-t4)*1000:.0f} ms")
        log(f"[Step 3] Found amendments: {len(all_amending_uus) > 0}")
        log(f"[Step 3] Amending UUs: {list(all_amending_uus)}")
        log("")
        
        # Generate currency warning
        currency_warning = generate_currency_warning(chain_results)
        if currency_warning:
            log("=" * 60)
            log("CURRENCY WARNING")
            log("=" * 60)
            log(currency_warning)
            log("")
        
        # ============================================================
        # STEP 4: Handle Amendments (if any)
        # ============================================================
        if not all_amending_uus:
            log("[Step 4] No amendments found. Using initial answer as final.")
            final_answer = initial_answer
        else:
            log("=" * 60)
            log("STEP 4: Retrieve from Amending Documents")
            log("=" * 60)
            
            t5 = time.time()
            amending_chunks = vector_query_chunks_filtered(q_emb, k=TOP_K_CHUNKS, uu_filters=list(all_amending_uus))
            log(f"[Step 4.1] Retrieved {len(amending_chunks)} chunks from amending documents in {(time.time()-t5)*1000:.0f} ms")
            
            if not amending_chunks:
                log("[Step 4.1] No chunks found in amending documents.")
                final_answer = initial_answer + "\n" + currency_warning if currency_warning else initial_answer
                final_answer += f"\nCatatan: Undang-undang yang dirujuk telah mengalami amandemen, namun ketentuan yang mengatur pertanyaan Anda tidak mengalami perubahan."
            else:
                # ============================================================
                # STEP 5: Judge Relevance
                # ============================================================
                log("=" * 60)
                log("STEP 5: Judge Amendment Relevance")
                log("=" * 60)
                
                t6 = time.time()
                relevance_result = judge_amendment_relevance(query_original, initial_answer, amending_chunks)
                log(f"[Step 5] Relevance judging completed in {(time.time()-t6)*1000:.0f} ms")
                log("")
                
                if not relevance_result.get("is_relevant", False):
                    log("[Step 5] Amendments not relevant to the question. Using initial answer.")
                    final_answer = initial_answer + "\n" + currency_warning if currency_warning else initial_answer
                    final_answer += f"\nCatatan: Undang-undang yang dirujuk telah mengalami amandemen, namun amandemen tersebut tidak mempengaruhi ketentuan yang menjawab pertanyaan Anda. Jawaban di atas tetap berlaku berdasarkan ketentuan yang masih aktif."
                else:
                    # ============================================================
                    # STEP 6: Integrate Amendments
                    # ============================================================
                    log("=" * 60)
                    log("STEP 6: Integrate Amendments into Final Answer")
                    log("=" * 60)
                    
                    new_context = build_context_from_chunks(amending_chunks, max_chunks=MAX_CHUNKS_FINAL)
                    
                    t7 = time.time()
                    final_answer = integrate_amendments(
                        query_original, 
                        initial_answer, 
                        new_context, 
                        all_amendment_info,
                        relevance_result,
                        output_lang=user_lang
                    )
                    log(f"[Step 6] Amendment integration completed in {(time.time()-t7)*1000:.0f} ms")
                    
                    # Prepend currency warning
                    if currency_warning:
                        final_answer = currency_warning + "\n" + final_answer
        
        # ============================================================
        # Final Summary
        # ============================================================
        usage_summary = _USAGE_TRACKER.get_summary()
        
        log("")
        log("=" * 60)
        log("FINAL ANSWER")
        log("=" * 60)
        log(final_answer)
        log("")
        log("=" * 60)
        log("LLM USAGE SUMMARY")
        log("=" * 60)
        log(json.dumps(usage_summary, indent=2, ensure_ascii=False))
        log("")
        log(f"Logs saved to: {log_file}")

        return {
            "final_answer": final_answer,
            "has_amendments": len(all_amending_uus) > 0,
            "currency_warnings": currency_warning,
            "amendment_info": all_amendment_info,
            "log_file": str(log_file),
            "llm_usage": usage_summary
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
            result = agentic_rag_with_amendments(user_query)
            print("\n" + "=" * 60)
            print("RESULT")
            print("=" * 60)
            print(f"Final Answer:\n{result['final_answer']}")
            print(f"\nHas Amendments: {result['has_amendments']}")
            print(f"Total LLM Calls: {result['llm_usage']['total_llm_calls']}")
            print(f"Total Tokens: {result['llm_usage']['total_tokens']}")
    finally:
        try:
            driver.close()
        except Exception:
            pass