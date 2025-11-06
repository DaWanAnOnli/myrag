#!/usr/bin/env python3
# agentic_rag_amendment_aware.py
# Amendment-aware Agentic RAG for Indonesian Legal Documents
# Multi-iteration pipeline: target date → baseline → amendment discovery → consolidation → final answer
# All extractions are LLM-based (no regex); infinite retry on API errors; extensive logging
# UPDATES (2025-10-28):
# - Iteration 3 uses the preliminary answer (Iteration 1) as semantic anchor,
#   after an extra LLM pass that removes UU/citation refs.
# - safe_generate_json_via_text retries JSON parsing up to 10 times before fallback.
# - Iteration 3 retrieval now uses the AMENDING UU only (not the base UU),
#   and the enriched query uses long form: "Undang Undang (UU) Nomor <number> Tahun <year>".

import os, time, json, re, random
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

GEN_MODEL   = os.getenv("GEN_MODEL", "models/gemini-2.0-flash-exp")
EMBED_MODEL = os.getenv("EMBED_MODEL", "models/text-embedding-004")

# Retrieval params
TOP_K_CHUNKS = int(os.getenv("TOP_K_CHUNKS", "40"))
MAX_CHUNKS_FINAL = int(os.getenv("MAX_CHUNKS_FINAL", "40"))
CHUNK_TEXT_CLAMP = int(os.getenv("CHUNK_TEXT_CLAMP", "1000000"))

# Amendment discovery params
AMENDMENT_SEARCH_K = int(os.getenv("AMENDMENT_SEARCH_K", "12"))  # K per expansion query

# Hardcoded amendment expansion phrases (no LLM generation)
AMENDMENT_EXPANSIONS = [
    "perubahan",
    "dicabut",
    "tetap berlaku sepanjang tidak bertentangan dan belum diganti"
]

# Consolidation params
CONSOLIDATION_SEARCH_K = int(os.getenv("CONSOLIDATION_SEARCH_K", "15"))  # K per article search
BASE_CONTENT_EXCERPT_WORDS = int(os.getenv("BASE_CONTENT_EXCERPT_WORDS", "150"))

# Agent params
ANSWER_MAX_TOKENS = int(os.getenv("ANSWER_MAX_TOKENS", "4096"))
AGENT_TEMP = float(os.getenv("AGENT_TEMP", "0.1"))

# Rate limit
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

def _rand_wait_seconds(min_s: float = 5.0, max_s: float = 20.0) -> float:
    return random.uniform(min_s, max_s)

def first_n_words(text: str, n_words: int) -> str:
    if not text:
        return ""
    words = text.split()
    return " ".join(words[:max(0, n_words)])

# ----------------- UU Number Normalization -----------------
def normalize_uu_number_for_filter(uu_short: str) -> tuple:
    """
    Convert short format like "32/2009" or "11/2020" 
    to (number, year) tuple for flexible regex matching in database.
    
    Returns: (number: str, year: str) or original string if unparseable
    """
    uu_short = (uu_short or "").strip()
    if not uu_short:
        return ""
    
    # Try to parse X/YYYY format
    parts = uu_short.split("/")
    if len(parts) == 2:
        number = parts[0].strip()
        year = parts[1].strip()
        return (number, year)
    
    # If already in long format or unparseable, return as-is
    return uu_short

def build_full_uu_formats(uu_short: str) -> List[str]:
    """
    Convert short format like "32/2009" to both full database formats:
    - "Undang-undang (UU) Nomor 32 Tahun 2009"
    - "Undang-undang (UU) No. 32 Tahun 2009"
    
    Returns list of both formats for embedding.
    """
    normalized = normalize_uu_number_for_filter(uu_short)
    if isinstance(normalized, tuple):
        number, year = normalized
        return [
            f"Undang-undang (UU) Nomor {number} Tahun {year}",
            f"Undang-undang (UU) No. {number} Tahun {year}"
        ]
    return [uu_short]

def build_uu_nomor_long(uu_short: str) -> str:
    """
    Return exactly: "Undang Undang (UU) Nomor <number> Tahun <year>"
    using the short form input "X/YYYY".
    """
    normalized = normalize_uu_number_for_filter(uu_short)
    if isinstance(normalized, tuple):
        number, year = normalized
        return f"Undang Undang (UU) Nomor {number} Tahun {year}"
    return uu_short

# ----------------- Rate Limiter -----------------
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
                log(f"[RateLimit:{self.name}] Sleeping {sleep_time:.2f}s to respect {self.calls_per_minute} calls/min")
                time.sleep(sleep_time)
            else:
                time.sleep(0.01)

_LLM_RATE_LIMITER = RateLimiter(LLM_CALLS_PER_MINUTE)

# ----------------- Infinite Retry Helpers -----------------
def _api_call_with_infinite_retry(func, *args, **kwargs):
    """Retry indefinitely with exponential backoff (capped) on API errors/exceptions."""
    attempt = 0
    while True:
        try:
            _LLM_RATE_LIMITER.wait_for_slot()
            return func(*args, **kwargs)
        except Exception as e:
            attempt += 1
            wait_base = min(60.0, 5.0 * (1.5 ** min(attempt - 1, 8)))
            wait_s = _rand_wait_seconds(wait_base, wait_base * 1.5)
            log(f"[Retry #{attempt}] API call failed: {e}. Retrying in {wait_s:.1f}s.")
            time.sleep(wait_s)

def embed_text(text: str) -> List[float]:
    res = _api_call_with_infinite_retry(genai.embed_content, model=EMBED_MODEL, content=text)
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
    attempt = 0
    while True:
        try:
            with driver.session() as session:
                res = session.run(cypher, **params)
                return list(res)
        except Exception as e:
            attempt += 1
            wait_s = _rand_wait_seconds()
            log(f"[Retry #{attempt}] Neo4j query failed: {e}. Retrying in {wait_s:.1f}s.")
            time.sleep(wait_s)

# ----------------- Safe LLM Generation (text-based JSON) -----------------
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

def safe_generate_text(prompt: str, max_tokens: int, temperature: float = 0.2) -> str:
    """Generate text with infinite retry on API errors."""
    cfg = GenerationConfig(temperature=temperature, max_output_tokens=max_tokens)
    resp = _api_call_with_infinite_retry(gen_model.generate_content, prompt, generation_config=cfg)
    text = extract_text_from_response(resp)
    if text:
        return text
    log(f"[LLM text warning] No text returned from model")
    return ""

def safe_generate_json_via_text(prompt: str, max_tokens: int = 2048, temperature: float = 0.1, parse_retries: int = 10) -> Dict[str, Any]:
    """
    Generate JSON by prompting for strict JSON output, then parse.
    - Infinite retry on API errors (handled inside safe_generate_text/_api_call_with_infinite_retry)
    - If JSON parse fails, repeat the LLM call up to `parse_retries` times before falling back to {}
    """
    for attempt in range(1, max(1, parse_retries) + 1):
        json_prompt = (
            prompt
            + "\nYou MUST respond with valid JSON only. No explanations, no markdown, no code fences. Strict JSON object only."
        )
        text = safe_generate_text(json_prompt, max_tokens=max_tokens, temperature=temperature)

        # Try to extract JSON from code fences if present, though we asked not to include them
        t = text.strip()
        if t.startswith("```json"):
            t = t[7:]
        elif t.startswith("```"):
            t = t[3:]
        if t.endswith("```"):
            t = t[:-3]
        t = t.strip()

        try:
            return json.loads(t)
        except json.JSONDecodeError as e:
            log(f"[JSON Parse Warning] Attempt {attempt}/{parse_retries}: Failed to parse JSON: {e}")
            log(f"[JSON Parse Warning] Raw text (first 500 chars): {t[:500]}")
            if attempt < parse_retries:
                time.sleep(_rand_wait_seconds(0.2, 0.8))
                continue
            else:
                log("[JSON Parse Warning] Giving up after max parse attempts; returning empty dict {}.")
                return {}

# ----------------- Anchor Cleaning (LLM-based) -----------------
def build_semantic_anchor_from_answer(answer_text: str, user_language: str = "id") -> str:
    """
    Use an LLM call to remove any Indonesian legal document references/citations
    from the preliminary answer so the anchor carries only substantive topic terms.
    Returns a single-line cleaned text suitable for retrieval queries.
    """
    if not (answer_text or "").strip():
        return ""

    if user_language == "id":
        prompt = f"""
Anda akan membersihkan jawaban berikut agar menjadi jangkar semantik untuk pencarian.
Tujuan: HAPUS semua rujukan dokumen hukum Indonesia dan sitiran formal, misalnya:
- "UU 32/2009", "Undang-Undang Nomor 32 Tahun 2009", "UU No. 11 Tahun 2020", "Perppu 2/2022", "PP 43/2008",
  "Peraturan Menteri ...", "SE ...", serta frasa "sebagaimana diubah oleh ..."
- penyebutan "Pasal", "ayat (...)", "huruf", "Bab", "Bagian", "Paragraf", "lampiran" beserta nomornya.
Pertahankan hanya kata/frasa tematik, konsep, istilah kunci, dan ringkasan substansi yang relevan.
Keluarkan SATU baris teks biasa tanpa tanda kutip dan TANPA sitiran apa pun.

Jawaban awal:
\"\"\"{answer_text}\"\"\"
"""
    else:
        prompt = f"""
You will clean the following answer so it can serve as a semantic search anchor.
Goal: REMOVE all references/citations to specific Indonesian legal documents, e.g.:
- "UU 32/2009", "Undang-Undang Nomor 32 Tahun 2009", "UU No. 11 Tahun 2020", "Perppu 2/2022", "PP 43/2008",
  "Ministerial Regulation ...", "as amended by ...", etc.
- also remove explicit mentions like "Article", "paragraph", "section", "chapter" with their numbers/letters.
Keep only the conceptual substance, topic phrases, and key terms relevant to the inquiry.
Output ONE single line of plain text without quotes and WITHOUT citations.

Original answer:
\"\"\"{answer_text}\"\"\"
"""

    cleaned = safe_generate_text(prompt, max_tokens=512, temperature=0.1).strip()
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned

# ----------------- Language Detection -----------------
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

# ----------------- Vector Search -----------------
def vector_query_chunks(q_emb: List[float], k: int, uu_filter: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Query Neo4j vector index 'chunk_embedding_index' over (:TextChunk {embedding}).
    Optionally filter by uu_number using flexible pattern matching to handle
    both "No." and "Nomor" variants in database.
    """
    if uu_filter:
        normalized = normalize_uu_number_for_filter(uu_filter)
        if isinstance(normalized, tuple):
            number, year = normalized
            cypher = """
            WITH $q AS q
            CALL db.index.vector.queryNodes('chunk_embedding_index', $k * 3, q)
            YIELD node, score
            WHERE node.uu_number =~ $uu_pattern
            RETURN node AS c, score
            ORDER BY score DESC
            LIMIT $k
            """
            uu_pattern = f".*No\\.?\\s+{number}\\s+Tahun\\s+{year}.*|.*Nomor\\s+{number}\\s+Tahun\\s+{year}.*"
            rows = run_cypher_with_retry(cypher, {"q": q_emb, "k": k, "uu_pattern": uu_pattern})
        else:
            cypher = """
            WITH $q AS q
            CALL db.index.vector.queryNodes('chunk_embedding_index', $k * 3, q)
            YIELD node, score
            WHERE node.uu_number = $uu_filter
            RETURN node AS c, score
            ORDER BY score DESC
            LIMIT $k
            """
            rows = run_cypher_with_retry(cypher, {"q": q_emb, "k": k, "uu_filter": normalized})
    else:
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

# ----------------- Context Building -----------------
def clamp(s: Optional[str], n: int) -> str:
    t = (s or "").strip()
    return t[:n]

def build_context_from_chunks(chunks: List[Dict[str, Any]], max_chunks: int, label: str = "Chunk") -> str:
    chosen = chunks[:max_chunks]
    lines = [f"=== {label} Context ==="]
    for i, c in enumerate(chosen, 1):
        doc = c.get("document_id", "?")
        cid = c.get("chunk_id", "?")
        uu = c.get("uu_number") or "?"
        pg = c.get("pages", "?")
        txt = clamp(c.get("content") or "", CHUNK_TEXT_CLAMP)
        lines.append(f"\n[{label} {i}] doc={doc} chunk={cid} | UU={uu} | pages={pg} | score={c.get('score', 0):.3f}")
        lines.append(txt)
    lines.append(f"\n=== End {label} Context ===\n")
    return "\n".join(lines)

# ----------------- PHASE 0: Target Date Gate -----------------
def phase0_target_date_gate(query: str) -> Dict[str, Any]:
    log("\n" + "="*80)
    log("PHASE 0: TARGET DATE GATE")
    log("="*80)
    
    user_lang = detect_user_language(query)
    log(f"[Phase 0] Detected user language: {user_lang}")
    
    prompt = f"""
You are a legal research assistant. Analyze the user's query to determine their temporal intent.

User query: "{query}"

Determine:
1. Does the user want the LATEST/CURRENT version of the law (as of today, October 28, 2025)?
2. Or does the user want the law AS OF a specific date/year in the past?

Extract:
- mode: "latest" or "as_of"
- target_date: "YYYY-MM-DD" (use 2025-10-28 for latest; for as_of, infer from query; if only year given, use YYYY-12-31)
- user_language: "id" or "en"

Examples:
- "Apa tugas DPRD?" → mode: "latest", target_date: "2025-10-28"
- "Apa tugas DPRD per 2020?" → mode: "as_of", target_date: "2020-12-31"
- "What were the provisions as of January 2019?" → mode: "as_of", target_date: "2019-01-31"

Respond with strict JSON:
{{
  "mode": "latest" or "as_of",
  "target_date": "YYYY-MM-DD",
  "user_language": "id" or "en"
}}
"""
    log("[Phase 0] Prompt:")
    log(prompt)
    log(f"[Phase 0] Prompt size: {len(prompt)} chars, est_tokens≈{estimate_tokens_for_text(prompt)}")
    
    result = safe_generate_json_via_text(prompt, max_tokens=512, temperature=AGENT_TEMP)
    
    if "mode" not in result:
        result["mode"] = "latest"
    if "target_date" not in result:
        result["target_date"] = "2025-10-28"
    if "user_language" not in result:
        result["user_language"] = user_lang
    
    log(f"[Phase 0] Result: {json.dumps(result, ensure_ascii=False)}")
    return result

# ----------------- ITERATION 1: Baseline Retrieval + Reference Set -----------------
def iteration1_baseline_retrieval(query: str, target_info: Dict[str, Any]) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
    log("\n" + "="*80)
    log("ITERATION 1: BASELINE RETRIEVAL + REFERENCE SET EXTRACTION")
    log("="*80)
    
    log("[Iteration 1] Embedding query...")
    t0 = time.time()
    q_emb = embed_text(query)
    log(f"[Iteration 1] Embedded query in {(time.time()-t0)*1000:.0f} ms")
    
    log(f"[Iteration 1] Retrieving top {TOP_K_CHUNKS} chunks from vector index...")
    t1 = time.time()
    candidates = vector_query_chunks(q_emb, k=TOP_K_CHUNKS)
    log(f"[Iteration 1] Retrieved {len(candidates)} candidates in {(time.time()-t1)*1000:.0f} ms")
    
    if not candidates:
        log("[Iteration 1] No chunks found!")
        return "Tidak ada dokumen relevan yang ditemukan.", [], {"laws": [], "question_focus": ""}
    
    context_text = build_context_from_chunks(candidates, max_chunks=MAX_CHUNKS_FINAL, label="Base")
    
    log("[Iteration 1] Generating preliminary answer...")
    answer_prompt = f"""
You are a legal research assistant for Indonesian law. Answer the user's question based strictly on the provided context.

User question: "{query}"

Target date: {target_info.get('target_date', '2025-10-28')} (mode: {target_info.get('mode', 'latest')})

{context_text}

Provide a concise preliminary answer. You MUST ALSO Cite relevant UU numbers and pasal/ayat.
"""
    log(f"[Iteration 1] Answer prompt size: {len(answer_prompt)} chars, est_tokens≈{estimate_tokens_for_text(answer_prompt)}")
    preliminary_answer = safe_generate_text(answer_prompt, max_tokens=ANSWER_MAX_TOKENS, temperature=0.2)
    log("[Iteration 1] Preliminary Answer:")
    log(preliminary_answer)
    
    log("[Iteration 1] Extracting reference set...")
    ref_prompt = f"""
You are a legal research assistant. Extract structured information about which laws and provisions are discussed in the context and answer.

User question: "{query}"

Preliminary answer:
{preliminary_answer}



Extract:
- laws: list of laws mentioned, each with:
  - uu_number: e.g., "32/2009", "11/2020"
  - title: full title if available
  - doc_id: document_id if available
  - source_chunk_ids: list of chunk_ids from which this law was cited
- question_focus: brief description of what the user is asking about (1 sentence)

Respond with strict JSON:
{{
  "laws": [
    {{
      "uu_number": "X/YYYY",
      "title": "...",
      "doc_id": "...",
      "source_chunk_ids": ["...", "..."]
    }}
  ],
  "question_focus": "..."
}}
"""
    log(f"[Iteration 1] Reference extraction prompt size: {len(ref_prompt)} chars")
    reference_set = safe_generate_json_via_text(ref_prompt, max_tokens=2048, temperature=AGENT_TEMP)
    
    if "laws" not in reference_set:
        reference_set["laws"] = []
    if "question_focus" not in reference_set:
        reference_set["question_focus"] = ""
    
    log(f"[Iteration 1] Reference Set: {json.dumps(reference_set, ensure_ascii=False, indent=2)}")
    
    return preliminary_answer, candidates[:MAX_CHUNKS_FINAL], reference_set

# ----------------- ITERATION 2: Amendment Discovery (Hardcoded Expansions) -----------------
def iteration2_amendment_discovery(reference_set: Dict[str, Any], target_info: Dict[str, Any]) -> Dict[str, Any]:
    log("\n" + "="*80)
    log("ITERATION 2: AMENDMENT DISCOVERY (HARDCODED EXPANSIONS)")
    log("="*80)
    
    laws = reference_set.get("laws", [])
    if not laws:
        log("[Iteration 2] No laws in reference set; skipping amendment discovery")
        return {"edges": []}
    
    target_date = target_info.get("target_date", "2025-10-28")
    log(f"[Iteration 2] Target date: {target_date}")
    log(f"[Iteration 2] Discovering amendments for {len(laws)} law(s)...")
    log(f"[Iteration 2] Using hardcoded expansions: {AMENDMENT_EXPANSIONS}")
    
    all_edges = []
    
    for law_idx, law in enumerate(laws, 1):
        uu_number = law.get("uu_number", "?")
        log(f"\n[Iteration 2.{law_idx}] Processing UU {uu_number}")
        
        full_formats = build_full_uu_formats(uu_number)
        log(f"[Iteration 2.{law_idx}] Full UU formats: {full_formats}")
        
        expansion_queries = []
        for phrase in AMENDMENT_EXPANSIONS:
            for full_format in full_formats:
                expansion_queries.append({
                    "query": f"{full_format} {phrase}",
                    "phrase": phrase,
                    "format": full_format
                })
        
        log(f"[Iteration 2.{law_idx}] Generated {len(expansion_queries)} expansion queries")
        
        log(f"[Iteration 2.{law_idx}] Searching for amendment chunks...")
        amendment_candidates = []
        seen_chunk_ids = set()
        
        for exp_idx, exp_item in enumerate(expansion_queries, 1):
            exp_query = exp_item["query"]
            exp_phrase = exp_item["phrase"]
            
            log(f"[Iteration 2.{law_idx}.{exp_idx}] Searching: '{exp_query}'")
            exp_emb = embed_text(exp_query)
            chunks = vector_query_chunks(exp_emb, k=AMENDMENT_SEARCH_K)
            
            for c in chunks:
                cid = c.get("chunk_id")
                if cid not in seen_chunk_ids:
                    seen_chunk_ids.add(cid)
                    c["expansion_phrase"] = exp_phrase
                    amendment_candidates.append(c)
            
            log(f"[Iteration 2.{law_idx}.{exp_idx}] Found {len(chunks)} chunks (total unique: {len(amendment_candidates)})")
        
        if not amendment_candidates:
            log(f"[Iteration 2.{law_idx}] No amendment candidates found for UU {uu_number}")
            continue
        
        log(f"[Iteration 2.{law_idx}] Extracting amendment edges from {len(amendment_candidates)} candidates...")
        log(f"\n[Iteration 2.{law_idx}] CANDIDATE PREVIEW (first 150 words each):")
        log("="*80)
        for cand_idx, cand in enumerate(amendment_candidates, 1):
            chunk_id = cand.get("chunk_id", "?")
            uu = cand.get("uu_number", "?")
            doc_id = cand.get("document_id", "?")
            pages = cand.get("pages", "?")
            expansion = cand.get("expansion_phrase", "?")
            score = cand.get("score", 0)
            content = cand.get("content", "")
            words = content.split()[:150]
            preview = " ".join(words)
            if len(words) < len(content.split()):
                preview += " [...]"
            log(f"\n--- Candidate {cand_idx}/{len(amendment_candidates)} ---")
            log(f"Chunk ID: {chunk_id}")
            log(f"UU: {uu}")
            log(f"Doc ID: {doc_id}")
            log(f"Pages: {pages}")
            log(f"Expansion phrase: {expansion}")
            log(f"Score: {score:.4f}")
            log(f"Preview (150 words):")
            log(preview)
        log("="*80)
        log(f"[Iteration 2.{law_idx}] End of candidate preview\n")
        
        candidates_context = build_context_from_chunks(amendment_candidates, len(amendment_candidates), "Amendment")
        
        extract_prompt = f"""
You are a legal research assistant. Analyze the provided chunks to find amendments to UU {uu_number}.

Base UU: {uu_number}
Target date: {target_date}

{candidates_context}

Extract all amendment relationships. For each amendment found:
- base_uu: "{uu_number}"
- amending_uu: the UU number that amends/modifies/repeals the base (e.g., "11/2020" - use SHORT format X/YYYY)
- action: one of ["mengubah", "mengganti", "menambah", "mencabut", "mencabut_sebagian", "tetap_berlaku_dengan_syarat"]
- effective_date: "YYYY-MM-DD" if mentioned; otherwise null
- evidence: list of chunk_ids where this amendment is mentioned

IMPORTANT: 
- For "tetap berlaku sepanjang tidak bertentangan" phrases, use action "tetap_berlaku_dengan_syarat"
- Do NOT extract specific pasal/ayat targets - we only need to know WHICH UU amends WHICH UU
- The specific provisions affected will be determined in the next iteration

Filter OUT any amendments with effective_date AFTER {target_date}.

Respond with strict JSON:
{{
  "edges": [
    {{
      "base_uu": "{uu_number}",
      "amending_uu": "X/YYYY",
      "action": "mengubah",
      "effective_date": "YYYY-MM-DD",
      "evidence": ["chunk_id_1", "chunk_id_2"]
    }}
  ]
}}
"""
        log(f"[Iteration 2.{law_idx}] Amendment extraction prompt size: {len(extract_prompt)} chars")
        edges_result = safe_generate_json_via_text(extract_prompt, max_tokens=4096, temperature=AGENT_TEMP)
        
        edges = edges_result.get("edges", [])
        
        for edge in edges:
            evidence_chunk_ids = edge.get("evidence", [])
            expansion_phrases = set()
            for ac in amendment_candidates:
                if ac.get("chunk_id") in evidence_chunk_ids:
                    expansion_phrases.add(ac.get("expansion_phrase", "unknown"))
            edge["expansion_phrases"] = list(expansion_phrases)
        
        log(f"[Iteration 2.{law_idx}] Extracted {len(edges)} amendment edge(s)")
        for edge in edges:
            log(f"[Iteration 2.{law_idx}]   - {edge.get('base_uu')} ← [{edge.get('action')}] ← {edge.get('amending_uu')} (effective: {edge.get('effective_date', 'unknown')}, expansions: {edge.get('expansion_phrases', [])})")
        
        all_edges.extend(edges)
    
    amendment_overlay = {"edges": all_edges}
    log(f"\n[Iteration 2] Amendment Discovery Complete: {len(all_edges)} total edge(s)")
    log(f"[Iteration 2] Amendment Overlay: {json.dumps(amendment_overlay, ensure_ascii=False, indent=2)}")
    return amendment_overlay

# ----------------- ITERATION 3: Consolidation (with Relevance Check) -----------------
def iteration3_consolidation(
    reference_set: Dict[str, Any],
    amendment_overlay: Dict[str, Any],
    base_chunks: List[Dict[str, Any]],
    target_info: Dict[str, Any],
    preliminary_answer: str
) -> Dict[str, Any]:
    """
    Returns: consolidated_context = {items: [...]}
    
    - Semantic anchor is derived from the Iteration 1 preliminary answer,
      cleaned via an LLM pass to remove UU/citation references.
    - Retrieval for consolidation uses ONLY the AMENDING UU, not the base UU.
    - Enriched query is built with long form: "Undang Undang (UU) Nomor <number> Tahun <year>".
    """
    log("\n" + "="*80)
    log("ITERATION 3: CONSOLIDATION OF OPERATIVE TEXT (WITH RELEVANCE CHECK)")
    log("="*80)
    
    laws = reference_set.get("laws", [])
    edges = amendment_overlay.get("edges", [])
    target_date = target_info.get("target_date", "2025-10-28")
    question_focus = reference_set.get("question_focus", "")
    user_language = target_info.get("user_language", "id")
    
    if not laws:
        log("[Iteration 3] No laws to consolidate")
        return {"items": []}
    
    log(f"[Iteration 3] Consolidating {len(laws)} law(s) with {len(edges)} amendment edge(s)")
    log(f"[Iteration 3] Target date: {target_date}")
    log(f"[Iteration 3] User language: {user_language}")
    log(f"[Iteration 3] Question focus: {question_focus}")
    
    # Build semantic anchor from preliminary answer (cleaned)
    log("[Iteration 3] Building semantic anchor from preliminary answer (LLM-cleaned to remove UU/citation references)...")
    semantic_anchor_raw = build_semantic_anchor_from_answer(preliminary_answer, user_language=user_language)
    if not semantic_anchor_raw:
        semantic_anchor_raw = question_focus or " ".join([c.get("content", "")[:300] for c in base_chunks[:1]])
    semantic_anchor = first_n_words(semantic_anchor_raw, BASE_CONTENT_EXCERPT_WORDS)
    log(f"[Iteration 3] Semantic anchor (first {BASE_CONTENT_EXCERPT_WORDS} words): {semantic_anchor[:200]}...")
    
    consolidated_items = []
    
    for law_idx, law in enumerate(laws, 1):
        uu_number = law.get("uu_number", "?")
        source_chunk_ids = law.get("source_chunk_ids", [])
        
        log(f"\n[Iteration 3.{law_idx}] Base UU under consideration: {uu_number}")
        built_uu_numbers = build_full_uu_formats(uu_number)
        built_uu_number_1, built_uu_number_2 = built_uu_numbers[0], built_uu_numbers[1]
        
        # Base chunks (for context/reference only)
        relevant_base_chunks = [c for c in base_chunks if c.get("uu_number") == built_uu_number_1 or c.get("uu_number") == built_uu_number_2 or c.get("chunk_id") in source_chunk_ids]
        
        if not relevant_base_chunks:
            log(f"[Iteration 3.{law_idx}] No base chunks found for UU {uu_number}; skipping")
            continue
        
        log(f"[Iteration 3.{law_idx}] Found {len(relevant_base_chunks)} relevant base chunk(s) for context")
        
        # Find applicable amendments for this base UU
        applicable_amendments = [e for e in edges if e.get("base_uu") == uu_number]
        if not applicable_amendments:
            log(f"[Iteration 3.{law_idx}] No amendments found for UU {uu_number}; using base text as operative")
            for base_chunk in relevant_base_chunks[:3]:
                item = {
                    "uu_number": uu_number,
                    "article": {"pasal": "multiple", "ayat": None},
                    "status": "operative",
                    "base_chunk_content": clamp(base_chunk.get("content", ""), CHUNK_TEXT_CLAMP),
                    "operative_text_snippet": clamp(base_chunk.get("content", ""), CHUNK_TEXT_CLAMP),
                    "cites": [{"uu_number": uu_number, "pasal": None, "ayat": None, "chunk_id": base_chunk.get("chunk_id")}]
                }
                consolidated_items.append(item)
            continue
        
        log(f"[Iteration 3.{law_idx}] Found {len(applicable_amendments)} applicable amendment(s)")
        
        # For each amendment, retrieve from the AMENDING UU only
        for amend_idx, amendment in enumerate(applicable_amendments, 1):
            amending_uu = amendment.get("amending_uu", "?")
            action = amendment.get("action", "?")
            expansion_phrases = amendment.get("expansion_phrases", [])
            
            log(f"[Iteration 3.{law_idx}.{amend_idx}] Amending UU: {amending_uu} ({action})")
            log(f"[Iteration 3.{law_idx}.{amend_idx}]   Expansion phrases: {expansion_phrases}")
            
            needs_relevance_check = (
                "perubahan" in expansion_phrases or 
                "tetap berlaku sepanjang tidak bertentangan dan belum diganti" in expansion_phrases or
                action == "tetap_berlaku_dengan_syarat"
            )
            if needs_relevance_check:
                log(f"[Iteration 3.{law_idx}.{amend_idx}] ⚠️ Relevance check REQUIRED")
            
            # Build enriched query from the AMENDING UU in long form
            long_amending = build_uu_nomor_long(amending_uu)  # "Undang Undang (UU) Nomor X Tahun Y"
            enriched_query_amending = f"{long_amending} {semantic_anchor}"
            log(f"[Iteration 3.{law_idx}.{amend_idx}] Enriched query (amending UU): {enriched_query_amending[:200]}...")
            
            # Regex logging for uu_filter
            normalized = normalize_uu_number_for_filter(amending_uu)
            if isinstance(normalized, tuple):
                number, year = normalized
                log(f"[Iteration 3.{law_idx}.{amend_idx}] Filtering to amending UU by pattern No./Nomor {number} Tahun {year}")
            
            # Vector search restricted to the AMENDING UU only
            enriched_emb = embed_text(enriched_query_amending)
            amending_chunks = vector_query_chunks(enriched_emb, k=CONSOLIDATION_SEARCH_K, uu_filter=amending_uu)
            
            log(f"[Iteration 3.{law_idx}.{amend_idx}] Retrieved {len(amending_chunks)} chunk(s) FROM AMENDING UU {amending_uu}")
            if amending_chunks:
                log(f"[Iteration 3.{law_idx}.{amend_idx}] Sample retrieved UU number format: '{amending_chunks[0].get('uu_number')}'")
            else:
                log(f"[Iteration 3.{law_idx}.{amend_idx}] ⚠️ No chunks found for amending UU {amending_uu}")
                for base_chunk in relevant_base_chunks[:2]:
                    item = {
                        "uu_number": uu_number,
                        "article": {"pasal": "multiple", "ayat": None},
                        "status": "operative",
                        "base_chunk_content": clamp(base_chunk.get("content", ""), CHUNK_TEXT_CLAMP),
                        "operative_text_snippet": clamp(base_chunk.get("content", ""), CHUNK_TEXT_CLAMP),
                        "cites": [{"uu_number": uu_number, "pasal": None, "ayat": None, "chunk_id": base_chunk.get("chunk_id")}],
                        "note": f"Amendment reference found ({amending_uu}) but no amending text retrieved"
                    }
                    consolidated_items.append(item)
                continue
            
            # Relevance check when needed
            if needs_relevance_check:
                log(f"[Iteration 3.{law_idx}.{amend_idx}] 🔍 Performing relevance check...")
                relevance_context = build_context_from_chunks(relevant_base_chunks[:2], 2, "Base") + "\n" + build_context_from_chunks(amending_chunks[:3], 3, "Amending")
                relevance_prompt = f"""
You are a legal research assistant. Determine if the amending UU actually addresses the user's question.

User's question focus: "{question_focus}"

Base UU: {uu_number}
Amending UU: {amending_uu}
Action: {action}

{relevance_context}

Analyze:
1. Does the amending UU contain provisions that actually MODIFY or REPLACE the specific matter the user is asking about?
2. Or does the amending UU address DIFFERENT matters, leaving the base UU's provisions on this topic unchanged?

IMPORTANT: 
- If the amendment is about "perubahan" (change) or "tetap berlaku..." (remains valid with conditions), 
  it might NOT change the specific provision the user is asking about.
- Only mark as "relevant" if the amending text DIRECTLY addresses the user's question focus.

Respond with strict JSON:
{{
  "is_relevant": true or false,
  "reasoning": "brief explanation (1-2 sentences)"
}}
"""
                log(f"[Iteration 3.{law_idx}.{amend_idx}] Relevance check prompt size: {len(relevance_prompt)} chars")
                relevance_result = safe_generate_json_via_text(relevance_prompt, max_tokens=512, temperature=AGENT_TEMP)
                is_relevant = relevance_result.get("is_relevant", True)
                reasoning = relevance_result.get("reasoning", "")
                log(f"[Iteration 3.{law_idx}.{amend_idx}] Relevance result: {is_relevant}")
                log(f"[Iteration 3.{law_idx}.{amend_idx}] Reasoning: {reasoning}")
                
                if not is_relevant:
                    log(f"[Iteration 3.{law_idx}.{amend_idx}] ✓ Amendment NOT relevant; keeping base text operative")
                    for base_chunk in relevant_base_chunks[:2]:
                        item = {
                            "uu_number": uu_number,
                            "article": {"pasal": "multiple", "ayat": None},
                            "status": "operative",
                            "base_chunk_content": clamp(base_chunk.get("content", ""), CHUNK_TEXT_CLAMP),
                            "operative_text_snippet": clamp(base_chunk.get("content", ""), CHUNK_TEXT_CLAMP),
                            "cites": [{"uu_number": uu_number, "pasal": None, "ayat": None, "chunk_id": base_chunk.get("chunk_id")}],
                            "note": f"Amendment {amending_uu} exists but does not modify this specific provision. Reasoning: {reasoning}"
                        }
                        consolidated_items.append(item)
                    continue
                else:
                    log(f"[Iteration 3.{law_idx}.{amend_idx}] ✓ Amendment IS relevant; proceeding with consolidation")
            
            # Consolidation: determine operative text
            consolidation_context = build_context_from_chunks(relevant_base_chunks[:2], 2, "Base") + "\n" + build_context_from_chunks(amending_chunks[:3], 3, "Amending")
            consolidation_prompt = f"""
You are a legal research assistant. Determine the operative text for UU {uu_number} after amendments.

Base UU: {uu_number}
Amending UU: {amending_uu}
Action: {action}
Target date: {target_date}
User's question focus: "{question_focus}"

{consolidation_context}

Determine:
1. What is the status of the provisions relevant to the user's question?
   - "operative": unchanged, still valid
   - "replaced": replaced by new text in amending UU
   - "repealed": explicitly repealed/canceled
   - "supplemented": new provision added alongside original

2. Extract the OPERATIVE TEXT SNIPPET (the text that should be used to answer the user's question).
   - If replaced: extract the replacement text from the amending UU chunks
   - If repealed: state "Dicabut" or "Repealed"
   - If supplemented: include both base and supplement
   - If operative: use base text

3. Provide citations: which UU, pasal, ayat, and chunk_id contains this text?

Respond with strict JSON:
{{
  "items": [
    {{
      "uu_number": "{uu_number}",
      "article": {{"pasal": "<pasal_number>", "ayat": "(<ayat_number>)"}},
      "status": "replaced",
      "base_chunk_content": "original text from base UU",
      "operative_text_snippet": "text from amending UU that replaces it",
      "cites": [
        {{"uu_number": "{amending_uu}", "pasal": "<pasal_number>", "ayat": "(<ayat_number>)", "chunk_id": "<chunk_id>"}}
      ]
    }}
  ]
}}
"""
            log(f"[Iteration 3.{law_idx}.{amend_idx}] Consolidation prompt size: {len(consolidation_prompt)} chars")
            consolidation_result = safe_generate_json_via_text(consolidation_prompt, max_tokens=4096, temperature=AGENT_TEMP)
            items = consolidation_result.get("items", [])
            log(f"[Iteration 3.{law_idx}.{amend_idx}] Consolidated {len(items)} item(s)")
            for item in items:
                log(f"[Iteration 3.{law_idx}.{amend_idx}]   - UU {item.get('uu_number')} Pasal {item.get('article', {}).get('pasal')} → {item.get('status')}")
            consolidated_items.extend(items)
    
    consolidated_context = {"items": consolidated_items}
    log(f"\n[Iteration 3] Consolidation Complete: {len(consolidated_items)} consolidated item(s)")
    log(f"[Iteration 3] Consolidated Context: {json.dumps(consolidated_context, ensure_ascii=False, indent=2)}")
    return consolidated_context

# ----------------- ITERATION 4: Final Answer -----------------
def iteration4_final_answer(
    query: str,
    consolidated_context: Dict[str, Any],
    target_info: Dict[str, Any]
) -> str:
    log("\n" + "="*80)
    log("ITERATION 4: FINAL ANSWER")
    log("="*80)
    
    items = consolidated_context.get("items", [])
    target_date = target_info.get("target_date", "2025-10-28")
    mode = target_info.get("mode", "latest")
    user_language = target_info.get("user_language", "id")
    
    if not items:
        log("[Iteration 4] No consolidated items; returning base answer")
        if user_language == "id":
            return f"Tidak ditemukan ketentuan yang relevan per {target_date}."
        else:
            return f"No relevant provisions found as of {target_date}."
    
    log(f"[Iteration 4] Generating final answer from {len(items)} consolidated item(s)")
    
    context_lines = ["=== CONSOLIDATED OPERATIVE CONTEXT ===\n"]
    for i, item in enumerate(items, 1):
        uu = item.get("uu_number", "?")
        article = item.get("article", {})
        pasal = article.get("pasal", "?")
        ayat = article.get("ayat")
        status = item.get("status", "?")
        operative_text = item.get("operative_text_snippet", "")
        cites = item.get("cites", [])
        note = item.get("note", "")
        
        context_lines.append(f"\n[Item {i}] UU {uu}, Pasal {pasal}" + (f" ayat {ayat}" if ayat else ""))
        context_lines.append(f"Status: {status}")
        if note:
            context_lines.append(f"Note: {note}")
        context_lines.append(f"Operative text:\n{operative_text}")
        if cites:
            cite_strs = [f"UU {c.get('uu_number', '?')}" + (f" Pasal {c.get('pasal')}" if c.get('pasal') else "") + (f" ayat {c.get('ayat')}" if c.get('ayat') else "") for c in cites]
            context_lines.append(f"Citations: {'; '.join(cite_strs)}")
    context_lines.append("\n=== END CONSOLIDATED CONTEXT ===\n")
    context_text = "\n".join(context_lines)
    
    log("[Iteration 4] Consolidated context for final answer:")
    log(context_text[:2000] + "..." if len(context_text) > 2000 else context_text)
    
    version_note_id = f"Per {target_date} ({'versi terbaru' if mode == 'latest' else 'versi pada tanggal tersebut'}):"
    version_note_en = f"As of {target_date} ({'latest version' if mode == 'latest' else 'version at that date'}):"
    version_note = version_note_id if user_language == "id" else version_note_en
    
    final_prompt = f"""
You are a legal research assistant for Indonesian law. Provide a comprehensive, accurate answer to the user's question using ONLY the consolidated operative context provided.

CRITICAL INSTRUCTIONS:
1. Answer STRICTLY from the consolidated context below
2. When base and amended texts conflict, ALWAYS prefer the amended text
3. If a provision is marked "replaced", use the replacement text, NOT the base text
4. If a provision is marked "repealed", state clearly that it has been repealed
5. If a note indicates an amendment exists but doesn't modify the specific provision, use the base text and mention this
6. Cite both the base UU (for article identity) and amending UU (for authority) where applicable
   Example: "Pasal 15 UU 32/2009 sebagaimana telah diubah oleh UU 11/2020, Pasal 25"
7. Include version note at the start: "{version_note}"
8. Respond in {"Indonesian" if user_language == "id" else "English"}

User question: "{query}"

{context_text}

Provide your answer with proper legal citations.
"""
    log(f"[Iteration 4] Final answer prompt size: {len(final_prompt)} chars, est_tokens≈{estimate_tokens_for_text(final_prompt)}")
    final_answer = safe_generate_text(final_prompt, max_tokens=ANSWER_MAX_TOKENS, temperature=0.2)
    
    log("\n" + "="*80)
    log("FINAL ANSWER")
    log("="*80)
    log(final_answer)
    return final_answer

# ----------------- Main Pipeline -----------------
def agentic_rag_amendment_aware(query_original: str) -> Dict[str, Any]:
    global _LOGGER
    ts_name = make_timestamp_name()
    log_file = Path.cwd() / f"rag_amendment_{ts_name}.txt"
    _LOGGER = FileLogger(log_file, also_console=True)
    
    try:
        log("="*80)
        log("AMENDMENT-AWARE AGENTIC RAG - START")
        log("="*80)
        log(f"Log file: {log_file}")
        log(f"Original Query: {query_original}")
        log(f"Parameters:")
        log(f"  - TOP_K_CHUNKS: {TOP_K_CHUNKS}")
        log(f"  - MAX_CHUNKS_FINAL: {MAX_CHUNKS_FINAL}")
        log(f"  - AMENDMENT_SEARCH_K: {AMENDMENT_SEARCH_K}")
        log(f"  - CONSOLIDATION_SEARCH_K: {CONSOLIDATION_SEARCH_K}")
        log(f"  - AMENDMENT_EXPANSIONS: {AMENDMENT_EXPANSIONS}")
        log(f"  - BASE_CONTENT_EXCERPT_WORDS: {BASE_CONTENT_EXCERPT_WORDS}")
        log(f"  - LLM_CALLS_PER_MINUTE: {LLM_CALLS_PER_MINUTE}")
        
        target_info = phase0_target_date_gate(query_original)
        preliminary_answer, base_chunks, reference_set = iteration1_baseline_retrieval(query_original, target_info)
        amendment_overlay = iteration2_amendment_discovery(reference_set, target_info)
        
        # Iteration 3 uses cleaned preliminary answer as semantic anchor; retrieval from AMENDING UU only
        consolidated_context = iteration3_consolidation(reference_set, amendment_overlay, base_chunks, target_info, preliminary_answer)
        final_answer = iteration4_final_answer(query_original, consolidated_context, target_info)
        
        log("\n" + "="*80)
        log("AMENDMENT-AWARE AGENTIC RAG - COMPLETE")
        log("="*80)
        log(f"Logs saved to: {log_file}")
        
        return {
            "final_answer": final_answer,
            "preliminary_answer": preliminary_answer,
            "log_file": str(log_file),
            "target_info": target_info,
            "reference_set": reference_set,
            "amendment_overlay": amendment_overlay,
            "consolidated_context": consolidated_context
        }
    
    except Exception as e:
        log(f"\n{'='*80}")
        log(f"FATAL ERROR: {e}")
        log(f"{'='*80}")
        import traceback
        log(traceback.format_exc())
        raise
    finally:
        if _LOGGER is not None:
            _LOGGER.close()

# ----------------- Main Entry Point -----------------
if __name__ == "__main__":
    try:
        user_query = input("Enter your query: ").strip()
        if not user_query:
            print("Empty query. Exiting.")
        else:
            result = agentic_rag_amendment_aware(user_query)
            print("\n" + "="*80)
            print("FINAL ANSWER:")
            print("="*80)
            print(result["final_answer"])
            print(f"Full log saved to: {result['log_file']}")
    finally:
        try:
            driver.close()
        except Exception:
            pass