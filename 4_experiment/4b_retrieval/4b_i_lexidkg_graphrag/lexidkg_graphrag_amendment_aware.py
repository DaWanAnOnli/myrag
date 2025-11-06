# lexidkg_graphrag_agentic.py
# Single-pass Agentic GraphRAG with version-aware law resolution:
# - As-of intent parsing (latest vs historical date)
# - Two retrieval branches:
#   A) Core topical GraphRAG (triple-centric + entity-centric)
#   B) Version-awareness (mod detection), itself with two sub-approaches:
#       B1) Predicate-driven (AMENDS/REPEALS/etc.)
#       B2) Node-type–driven (LawAmendment/LawModification/LawAddition/LawDeletion)
# - Temporal filtering: exclude repealed content as of as-of date (keep mod-evidence; exclusions logged)
# - Temporal-aware re-ranking with reserved budget for version-lineage chunks
# - Short version lineage summary for the Answerer
# - Indefinite retry for LLM generation (JSON/text) and embeddings

import os, time, json, math, pickle, re, random, datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, Set
from threading import Lock

from dotenv import load_dotenv
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
from neo4j import GraphDatabase

# ----------------- Load .env (parent directory of this file) -----------------
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent.parent.parent / ".env")

# ----------------- Config -----------------
GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]
NEO4J_URI  = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASS", "password")

# Neo4j retry/timeout controls
NEO4J_TX_TIMEOUT_S = float(os.getenv("NEO4J_TX_TIMEOUT_S", "60"))
NEO4J_MAX_ATTEMPTS = int(os.getenv("NEO4J_MAX_ATTEMPTS", "10"))

# Gemini models
GEN_MODEL = os.getenv("GEN_MODEL", "models/gemini-2.5-flash")
EMBED_MODEL = os.getenv("EMBED_MODEL", "models/text-embedding-004")

# Dataset folder for original chunk pickles (same as ingestion)
DEFAULT_LANGCHAIN_DIR = (Path(__file__).resolve().parent / "../../../dataset/3_indexing/3a_langchain_results/").resolve()
LANGCHAIN_DIR = Path(os.getenv("LANGCHAIN_DIR") or str(DEFAULT_LANGCHAIN_DIR))
SKIP_FILES = {"all_langchain_documents.pkl"}

# ----------------- Retrieval/agent parameters -----------------
# Core entity path
ENTITY_MATCH_TOP_K = 15
ENTITY_SUBGRAPH_HOPS = 5
ENTITY_SUBGRAPH_PER_HOP_LIMIT = 2000
SUBGRAPH_TRIPLES_TOP_K = 30

# Triple-centric path
QUERY_TRIPLE_MATCH_TOP_K_PER = 20

# Final context combination and reranking
MAX_TRIPLES_FINAL = 60
MAX_CHUNKS_FINAL = 40
CHUNK_RERANK_CAND_LIMIT = 200

# Agent loop and output
ANSWER_MAX_TOKENS = 4096
MAX_ITERS = 3  # Ignored in single-pass

# Language setting
OUTPUT_LANG = "id"  # retained for compatibility

# ----------------- Version-aware settings -----------------
# Predicate-driven mod detection
MOD_DOC_PREDICATES = {
    # English normalized
    "amends", "amended_by", "repeals",
    # Indonesian normalized variants
    "mengubah", "diubah", "mencabut", "dicabut", "tidak_berlaku"
}
MOD_CONTENT_PREDICATES = {
    "modifies", "adds", "deletes",
    "law_modification", "law_addition", "law_deletion",
    "mengubah_ketentuan", "menambahkan", "menyisipkan", "menghapus"
}
VERSION_PREDICATES_ALLOWED = MOD_DOC_PREDICATES.union(MOD_CONTENT_PREDICATES)

# Node-type–driven mod detection
MOD_NODE_TYPES = {"LawAmendment", "LawModification", "LawAddition", "LawDeletion"}

# Predicates that may indicate effective dates (best-effort parse)
DATE_PREDICATES = {
    "mulai_berlaku", "berlaku_mulai", "berlaku_sejak", "effective_from", "effective_date",
    "has_enaction_date", "tanggal_diundangkan", "tanggal_berlaku"
}

# Guardrails / weights
VERSION_RESERVED_CHUNKS_MIN = int(os.getenv("VERSION_RESERVED_CHUNKS_MIN", "6"))
EXCLUDE_REPEALED_CONTENT = os.getenv("EXCLUDE_REPEALED_CONTENT", "1").strip().lower() in ("1", "true", "yes", "y", "on")
TEMPORAL_BOOST_FOR_CURRENT = float(os.getenv("TEMPORAL_BOOST_FOR_CURRENT", "0.10"))
TEMPORAL_DOWNWEIGHT_SUPERSEDED = float(os.getenv("TEMPORAL_DOWNWEIGHT_SUPERSEDED", "0.10"))

# ----------------- Initialize SDKs -----------------
genai.configure(api_key=GOOGLE_API_KEY)
gen_model = genai.GenerativeModel(GEN_MODEL)
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

# ----------------- Logger -----------------
def _now_ts() -> str:
    t = time.time()
    base = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t))
    ms = int((t % 1) * 1000)
    return f"{base}.{ms:03d}"

def _pid() -> int:
    try:
        return os.getpid()
    except Exception:
        return -1

def _prefix(level: str = "INFO") -> str:
    return f"[{_now_ts()}] [{level}] [pid={_pid()}]"

class FileLogger:
    def __init__(self, file_path: Path, also_console: bool = True):
        self.file_path = file_path
        self.also_console = also_console
        self._fh = open(file_path, "w", encoding="utf-8")

    def log(self, msg: str = ""):
        self._fh.write(msg + "\n")
        self._fh.flush()
        if self.also_console:
            print(msg, flush=True)

    def close(self):
        try:
            self._fh.flush()
            self._fh.close()
        except Exception:
            pass

_LOGGER: Optional[FileLogger] = None

def _fmt_msg(msg: Any, level: str) -> str:
    if not isinstance(msg, str):
        try:
            msg = json.dumps(msg, ensure_ascii=False, default=str)
        except Exception:
            msg = str(msg)
    lines = str(msg).splitlines() or [str(msg)]
    prefixed = [f"{_prefix(level)} {line}" for line in lines] if lines else [f"{_prefix(level)}"]
    return "\n".join(prefixed)

def log(msg: Any = "", level: str = "INFO"):
    global _LOGGER
    out = _fmt_msg(msg, level)
    if _LOGGER is not None:
        _LOGGER.log(out)
    else:
        print(out, flush=True)

def make_timestamp_name() -> str:
    t = time.time()
    base = time.strftime("%Y%m%d-%H%M%S", time.localtime(t))
    ms = int((t % 1) * 1000)
    return f"{base}-{ms:03d}"

# ----------------- Utilities -----------------
def now_ms() -> float:
    return time.time()

def dur_ms(start: float) -> float:
    return (time.time() - start) * 1000.0

def _norm_id(x) -> str:
    return str(x).strip() if x is not None else ""

def estimate_tokens_for_text(text: str) -> int:
    return max(1, int(len(text) / 4))

def _as_float_list(vec) -> List[float]:
    if vec is None:
        return []
    try:
        if hasattr(vec, "tolist"):
            vec = vec.tolist()
    except Exception:
        pass
    try:
        return [float(x) for x in list(vec)]
    except Exception:
        try:
            return [float(vec)]
        except Exception:
            return []

def cos_sim(a: List[float], b: List[float]) -> float:
    a = _as_float_list(a)
    b = _as_float_list(b)
    dot = sum(x*y for x,y in zip(a,b))
    na = math.sqrt(sum(x*x for x in a))
    nb = math.sqrt(sum(y*y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)

def normalize_predicate_for_test(p: Optional[str]) -> str:
    if not p:
        return ""
    p = p.strip().lower()
    p = p.replace("-", "_").replace(" ", "_")
    p = re.sub(r"[^a-z0-9_]", "", p)
    return p

# ----------------- Language detection -----------------
def detect_user_language(text: str) -> str:
    t = (text or "").lower()
    if re.search(r"\b(pasal|undang[- ]?undang|uu\s*\d|peraturan|menteri|ayat|bab|bagian|paragraf|ketentuan|sebagaimana|dimaksud)\b", t):
        return "id"
    if re.search(r"\b(article|act|law|regulation|minister|section|paragraph|chapter|pursuant|provided that)\b", t):
        return "en"
    id_tokens = {"yang","dan","atau","tidak","adalah","berdasarkan","sebagaimana","pada","dalam","dapat","harus","wajib",
                 "pasal","undang","peraturan","menteri","ayat","bab","bagian","paragraf","ketentuan","pengundangan","apabila","jika"}
    en_tokens = {"the","and","or","not","is","based","as","provided","pursuant","in","may","must","shall",
                 "article","act","law","regulation","minister","section","paragraph","chapter","whereas"}
    words = re.findall(r"[a-z]+", t)
    score_id = sum(1 for w in words if w in id_tokens)
    score_en = sum(1 for w in words if w in en_tokens)
    if score_id > score_en:
        return "id"
    if score_en > score_id:
        return "en"
    return "en"

# ----------------- As-of parsing helpers -----------------
_ID_MONTHS = {
    "januari":1, "jan":1, "februari":2, "feb":2, "maret":3, "mar":3,
    "april":4, "apr":4, "mei":5, "juni":6, "jun":6, "juli":7, "jul":7,
    "agustus":8, "agu":8, "agt":8, "aug":8, "september":9, "sep":9, "sept":9,
    "oktober":10, "okt":10, "oct":10, "november":11, "nov":11, "desember":12, "des":12, "dec":12
}
_EN_MONTHS = {
    "january":1, "jan":1, "february":2, "feb":2, "march":3, "mar":3,
    "april":4, "apr":4, "may":5, "june":6, "jun":6, "july":7, "jul":7,
    "august":8, "aug":8, "september":9, "sep":9, "sept":9, "october":10, "oct":10,
    "november":11, "nov":11, "december":12, "dec":12
}

def _try_parse_int(s: str) -> Optional[int]:
    try:
        return int(s)
    except Exception:
        return None

def _parse_date_tokens(day: str, month: str, year: str) -> Optional[datetime.date]:
    d = _try_parse_int(re.sub(r"[^\d]", "", day))
    y = _try_parse_int(re.sub(r"[^\d]", "", year))
    if not d or not y:
        return None
    m = None
    m_txt = (month or "").lower()
    if m_txt.isdigit():
        m = _try_parse_int(m_txt)
    if m is None:
        m = _ID_MONTHS.get(m_txt) or _EN_MONTHS.get(m_txt)
    if not m or m < 1 or m > 12:
        return None
    try:
        return datetime.date(y, m, d)
    except Exception:
        return None

_DATE_PATTERNS = [
    re.compile(r"\b(\d{1,2})\s+([A-Za-z\.]+)\s+(\d{4})\b"),  # 1 Januari 2020
    re.compile(r"\b(\d{1,2})[-/](\d{1,2})[-/](\d{2,4})\b"),   # 01-01-2020 or 01/01/2020
    re.compile(r"\b(\d{4})[-/](\d{1,2})[-/](\d{1,2})\b"),     # 2020-01-31
]

def parse_date_any(text: str) -> Optional[datetime.date]:
    if not text:
        return None
    t = text.strip()
    for pat in _DATE_PATTERNS:
        m = pat.search(t)
        if not m:
            continue
        if pat is _DATE_PATTERNS[0]:
            d, mon, y = m.group(1), m.group(2), m.group(3)
            dt = _parse_date_tokens(d, mon, y)
            if dt: return dt
        elif pat is _DATE_PATTERNS[1]:
            d, mth, y = m.group(1), m.group(2), m.group(3)
            yy = _try_parse_int(y)
            if yy and yy < 100:
                yy = 2000 + yy
            try:
                return datetime.date(int(yy), int(mth), int(d))
            except Exception:
                pass
        else:
            y, mth, d = m.group(1), m.group(2), m.group(3)
            try:
                return datetime.date(int(y), int(mth), int(d))
            except Exception:
                pass
    return None

def parse_as_of_intent(query: str) -> Tuple[str, datetime.date]:
    today = datetime.date.today()
    q = (query or "").lower()
    if re.search(r"\b(latest|current|today|now|terkini|terbaru|paling\s+baru|sekarang)\b", q):
        return "latest", today
    m = re.search(r"\b(per|as of|per\s*tanggal|per\s*tgl)\b", q)
    if m:
        window = q[m.end():m.end()+50]
        dt = parse_date_any(window)
        if dt:
            return "as_of", dt
    dt2 = parse_date_any(q)
    if dt2:
        return "as_of", dt2
    return "latest", today

def extract_uu_number_from_text(text: Optional[str]) -> Optional[str]:
    if not text: return None
    m = re.search(r"\buu\s*([0-9]{1,4}\s*/\s*[0-9]{4})\b", text.lower().replace("-", " "))
    if m:
        return m.group(1).replace(" ", "")
    m2 = re.search(r"(?:undang[- ]?undang|uu)\s*(?:nomor|no\.?)\s*([0-9]{1,4})\s*(?:tahun)\s*([0-9]{4})", text.lower())
    if m2:
        return f"{m2.group(1)}/{m2.group(2)}"
    return None

# ----------------- Safe LLM helpers (with indefinite retry) -----------------
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
        text = resp.text
        if isinstance(text, str) and text.strip():
            return text.strip()
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

def _rand_wait_seconds() -> float:
    # Using the same backoff window as your original embed retry
    return random.uniform(80.0, 120.0)

def _api_call_with_retry(func, *args, **kwargs):
    while True:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            wait_s = _rand_wait_seconds()
            log(f"[Retry] API call failed: {e}. Retrying in {wait_s:.1f}s.", level="WARN")
            time.sleep(wait_s)

def safe_generate_json(prompt: str, schema: Dict[str, Any], temp: float = 0.0) -> Dict[str, Any]:
    cfg = GenerationConfig(
        temperature=temp,
        response_mime_type="application/json",
        response_schema=schema,
    )
    t0 = now_ms()
    # Indefinite retry on errors
    resp = _api_call_with_retry(gen_model.generate_content, prompt, generation_config=cfg)
    log(f"[LLM JSON] call completed in {dur_ms(t0):.0f} ms", level="DEBUG")
    try:
        if isinstance(resp.text, str) and resp.text.strip():
            return json.loads(resp.text)
    except Exception:
        pass
    try:
        raw = resp.candidates[0].content.parts[0].text  # type: ignore[attr-defined]
        return json.loads(raw)
    except Exception as e:
        info = get_finish_info(resp)
        log(f"[LLM JSON parse warning] No JSON content returned. Diagnostics: {info}. Error: {e}", level="WARN")
        try:
            return json.loads("{}")
        except Exception:
            return {}

def safe_generate_text(prompt: str, max_tokens: int, temperature: float = 0.2) -> str:
    cfg = GenerationConfig(temperature=temperature, max_output_tokens=max_tokens)
    t0 = now_ms()
    # Indefinite retry on errors
    resp = _api_call_with_retry(gen_model.generate_content, prompt, generation_config=cfg)
    took = dur_ms(t0)
    text = extract_text_from_response(resp)
    if text is not None and text.strip():
        log(f"[LLM TEXT] call completed in {took:.0f} ms, len={len(text)}", level="DEBUG")
        return text.strip()
    info = get_finish_info(resp)
    log(f"[LLM text warning] No text returned. Diagnostics: {info}. Took={took:.0f} ms", level="WARN")
    return f"(Model returned no text. finish_info={info})"

# ----------------- Agent 1: entity/predicate extraction -----------------
LEGAL_ENTITY_TYPES = [
    "UU", "PASAL", "AYAT", "INSTANSI", "ORANG", "ISTILAH", "SANKSI", "NOMINAL", "TANGGAL"
]
LEGAL_PREDICATES = [
    "mendefinisikan", "mengubah", "mencabut", "mulai_berlaku", "mewajibkan",
    "melarang", "memberikan_sanksi", "berlaku_untuk", "termuat_dalam",
    "mendelegasikan_kepada", "berjumlah", "berdurasi"
]

QUERY_SCHEMA = {
  "type": "object",
  "properties": {
    "entities": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "text": {"type": "string"},
          "type": {"type": "string"}
        },
        "required": ["text"]
      }
    },
    "predicates": {
      "type": "array",
      "items": {"type": "string"}
    }
  },
  "required": ["entities", "predicates"]
}

def agent1_extract_entities_predicates(query: str) -> Dict[str, Any]:
    prompt = f"""
You are Agent 1. Task: extract the legal entities and predicates referenced or implied by the user's question.

Output format:
- JSON with:
  - "entities": array of objects with fields {{text, type(optional)}}
  - "predicates": array of strings (Indonesian, snake_case when applicable)
- If entity type is provided, it MUST be one of: {", ".join(LEGAL_ENTITY_TYPES)}.
- Predicates should ideally be one of: {", ".join(LEGAL_PREDICATES)}.

User question:
\"\"\"{query}\"\"\"
"""
    est_tokens = estimate_tokens_for_text(prompt)
    log("\n[Agent 1] Prompt:")
    log(prompt)
    log(f"[Agent 1] Prompt size: {len(prompt)} chars, est_tokens≈{est_tokens}")
    t0 = now_ms()
    data = safe_generate_json(prompt, QUERY_SCHEMA, temp=0.0)
    took = dur_ms(t0)
    if "entities" not in data: data["entities"] = []
    if "predicates" not in data: data["predicates"] = []
    log(f"[Agent 1] Output: entities={ [e.get('text') for e in data['entities']] }, predicates={ data['predicates'] } | {took:.0f} ms")
    return data

# ----------------- Agent 1b: query triple extraction -----------------
QUERY_TRIPLES_SCHEMA = {
    "type": "object",
    "properties": {
        "triples": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "subject": {"type": "object","properties":{"text":{"type":"string"},"type":{"type":"string"}},"required":["text"]},
                    "predicate": {"type": "string"},
                    "object": {"type": "object","properties":{"text":{"type":"string"},"type":{"type":"string"}},"required":["text"]}
                },
                "required": ["subject", "predicate", "object"]
            }
        }
    },
    "required": ["triples"]
}

def agent1b_extract_query_triples(query: str) -> List[Dict[str, Any]]:
    prompt = f"""
You are Agent 1b. Task: extract explicit or implied triples from the user's question in the form:
subject — predicate — object.

Rules:
- Use short, literal subject/object texts as they appear in the question.
- Predicates should be concise (lowercase, snake_case if multiword).
- If type is unknown, leave it blank.
- Do not invent or speculate; extract only what is clearly suggested by the question.

Return JSON with a key "triples" as specified.

User question:
\"\"\"{query}\"\"\"
"""
    est_tokens = estimate_tokens_for_text(prompt)
    log("\n[Agent 1b] Prompt:")
    log(prompt)
    log(f"[Agent 1b] Prompt size: {len(prompt)} chars, est_tokens≈{est_tokens}")
    t0 = now_ms()
    out = safe_generate_json(prompt, QUERY_TRIPLES_SCHEMA, temp=0.0)
    triples = out.get("triples", []) if isinstance(out, dict) else []
    clean: List[Dict[str, Any]] = []
    for t in triples:
        try:
            s = t.get("subject", {}) or {}
            o = t.get("object", {}) or {}
            p = (t.get("predicate") or "").strip()
            if (s.get("text") and o.get("text") and p):
                clean.append({
                    "subject": {"text": s.get("text","").strip(), "type": (s.get("type") or "").strip()},
                    "predicate": p,
                    "object":  {"text": o.get("text","").strip(), "type": (o.get("type") or "").strip()},
                })
        except Exception:
            continue
    took = dur_ms(t0)
    log(f"[Agent 1b] Extracted query triples: {['{} [{}] {}'.format(x['subject']['text'], x['predicate'], x['object']['text']) for x in clean]} | {took:.0f} ms")
    return clean

def query_triple_to_text(t: Dict[str, Any]) -> str:
    s = ((t.get("subject") or {}).get("text") or "").strip()
    p = (t.get("predicate") or "").strip()
    o = ((t.get("object") or "").get("text") or "").strip() if isinstance(t.get("object"), dict) else ((t.get("object") or "").strip())
    return f"{s} [{p}] {o}"

# ----------------- Neo4j vector search helpers -----------------
_NEO4J_QUERY_SEQ = 0
_NEO4J_QUERY_LOCK = Lock()

def _next_query_id() -> int:
    global _NEO4J_QUERY_SEQ
    with _NEO4J_QUERY_LOCK:
        _NEO4J_QUERY_SEQ += 1
        return _NEO4J_QUERY_SEQ

def _summarize_params(params: Dict[str, Any]) -> str:
    parts = []
    for k, v in (params or {}).items():
        try:
            if k.lower() in ("q_emb", "embedding", "emb"):
                if isinstance(v, (list, tuple)):
                    parts.append(f"{k}=list(len={len(v)})")
                else:
                    parts.append(f"{k}=vector")
            elif isinstance(v, list):
                parts.append(f"{k}=list(len={len(v)})")
            elif isinstance(v, str) and len(v) > 60:
                parts.append(f"{k}=str(len={len(v)})")
            else:
                parts.append(f"{k}={v}")
        except Exception:
            parts.append(f"{k}=<?>")
    return ", ".join(parts)

def run_cypher_with_retry(cypher: str, params: Dict[str, Any]) -> List[Any]:
    attempts = 0
    last_e: Optional[Exception] = None
    qid = _next_query_id()
    preview = " ".join((cypher or "").split())
    if len(preview) > 220:
        preview = preview[:220] + "..."
    param_summary = _summarize_params(params)
    while attempts < max(1, NEO4J_MAX_ATTEMPTS):
        attempts += 1
        t0 = now_ms()
        log(f"[Neo4j] Attempt {attempts}/{NEO4J_MAX_ATTEMPTS} | qid={qid} | timeout={NEO4J_TX_TIMEOUT_S:.1f}s | Cypher=\"{preview}\" | Params: {param_summary}")
        try:
            with driver.session() as session:
                res = session.run(cypher, **params, timeout=NEO4J_TX_TIMEOUT_S)
                records = list(res)
            took = dur_ms(t0)
            log(f"[Neo4j] Success | qid={qid} | rows={len(records)} | {took:.0f} ms")
            return records
        except Exception as e:
            took = dur_ms(t0)
            last_e = e
            if attempts >= NEO4J_MAX_ATTEMPTS:
                break
            wait_s = random.uniform(5, 10)
            log(f"[Neo4j] Failure | qid={qid} | attempt={attempts}/{NEO4J_MAX_ATTEMPTS} | {took:.0f} ms | error={e}. Retrying in {wait_s:.1f}s.", level="WARN")
            time.sleep(wait_s)
    raise RuntimeError(f"Neo4j query failed after {NEO4J_MAX_ATTEMPTS} attempts (qid={qid}): {last_e}")

def _vector_query_nodes(index_name: str, q_emb: List[float], k: int) -> List[Dict[str, Any]]:
    q_emb = _as_float_list(q_emb)
    cypher = """
    CALL db.index.vector.queryNodes($index_name, $k, $q_emb) YIELD node AS n, score
    RETURN n, score, elementId(n) AS elem_id
    ORDER BY score DESC
    LIMIT $k
    """
    res = run_cypher_with_retry(cypher, {"index_name": index_name, "k": k, "q_emb": q_emb})
    rows = []
    for r in res:
        n = r["n"]
        rows.append({
            "key": n.get("key"),
            "elem_id": r["elem_id"],
            "name": n.get("name"),
            "type": n.get("type"),
            "score": r["score"],
        })
    return rows

def search_similar_entities_by_embedding(q_emb: List[float], k: int) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    try:
        candidates.extend(_vector_query_nodes("document_vec", q_emb, k))
    except Exception as e:
        log(f"[Warn] document_vec query failed: {e}", level="WARN")
    try:
        candidates.extend(_vector_query_nodes("content_vec", q_emb, k))
    except Exception as e:
        log(f"[Warn] content_vec query failed: {e}", level="WARN")
    try:
        candidates.extend(_vector_query_nodes("expression_vec", q_emb, k))
    except Exception as e:
        log(f"[Warn] expression_vec query failed: {e}", level="WARN")

    best: Dict[str, Dict[str, Any]] = {}
    for row in candidates:
        dedup_key = row.get("elem_id") or f"{row.get('key')}|{row.get('type')}"
        if dedup_key not in best or (row.get("score", -1) > best[dedup_key].get("score", -1)):
            best[dedup_key] = row

    merged = list(best.values())
    merged.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    return merged[:k]

def search_similar_triples_by_embedding(q_emb: List[float], k: int) -> List[Dict[str, Any]]:
    q_emb = _as_float_list(q_emb)
    cypher = """
    CALL db.index.vector.queryNodes('triple_vec', $k, $q_emb) YIELD node AS tr, score
    OPTIONAL MATCH (tr)-[:SUBJECT]->(s)
    OPTIONAL MATCH (tr)-[:OBJECT]->(o)
    RETURN tr, s, o, score
    ORDER BY score DESC
    LIMIT $k
    """
    res = run_cypher_with_retry(cypher, {"k": k, "q_emb": q_emb})
    rows = []
    for r in res:
        tr = r["tr"]; s = r["s"]; o = r["o"]
        rows.append({
            "triple_uid": tr.get("triple_uid"),
            "predicate": tr.get("predicate"),
            "uu_number": tr.get("uu_number"),
            "evidence_quote": tr.get("evidence_quote"),
            "subject": s.get("name") if s else None,
            "subject_key": s.get("key") if s else None,
            "subject_type": s.get("type") if s else None,
            "object": o.get("name") if o else None,
            "object_key": o.get("key") if o else None,
            "object_type": o.get("type") if o else None,
            "score": r["score"],
            "embedding": tr.get("embedding"),
            "document_id": tr.get("document_id"),
            "chunk_id": tr.get("chunk_id"),
            "evidence_article_ref": tr.get("evidence_article_ref"),
        })
    return rows

# ----------------- Graph expansion (predicate and node-type filters supported) -----------------
def expand_from_entities(
    entity_keys: List[str],
    hops: int,
    per_hop_limit: int,
    entity_elem_ids: Optional[List[str]] = None,
    allowed_predicates: Optional[List[str]] = None,
    allowed_node_types: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """
    Expand outward and collect triples. Apply filters:
      - allowed_predicates: only include triples whose Triple.predicate is in list
      - allowed_node_types: only include triples where subject.type or object.type is in list
    Note: We seed next-hop only from triples that pass filters to constrain expansion when filters are used.
    """
    triples: Dict[str, Dict[str, Any]] = {}
    current_ids: Set[str] = set(x for x in (entity_elem_ids or []) if x)
    current_keys: Set[str] = set(x for x in (entity_keys or []) if x)

    where_pred = ""
    params_extra: Dict[str, Any] = {}
    # We can pre-filter on predicate in Cypher to reduce volume when allowed_predicates provided
    if allowed_predicates:
        where_pred = " AND r.predicate IN $allowed_preds "
        params_extra["allowed_preds"] = [normalize_predicate_for_test(p) for p in allowed_predicates]

    allowed_types_set: Optional[Set[str]] = set(allowed_node_types) if allowed_node_types else None

    for _ in range(hops):
        if not current_ids and not current_keys:
            break

        if current_ids:
            cypher = f"""
            UNWIND $ids AS eid
            MATCH (e) WHERE elementId(e) = eid
            MATCH (e)-[r:PREDICATE]->()
            WHERE 1=1 {where_pred}
            WITH DISTINCT r.triple_uid AS uid LIMIT $limit
            MATCH (tr:Triple {{triple_uid: uid}})
            OPTIONAL MATCH (tr)-[:SUBJECT]->(s)
            OPTIONAL MATCH (tr)-[:OBJECT]->(o)
            RETURN tr, s, o, elementId(s) AS s_id, elementId(o) AS o_id
            """
            params = {"ids": list(current_ids), "limit": per_hop_limit, **params_extra}
        else:
            cypher = f"""
            UNWIND $keys AS k
            MATCH (e:Entity {{key:k}})-[r:PREDICATE]->()
            WHERE 1=1 {where_pred}
            WITH DISTINCT r.triple_uid AS uid LIMIT $limit
            MATCH (tr:Triple {{triple_uid: uid}})
            OPTIONAL MATCH (tr)-[:SUBJECT]->(s)
            OPTIONAL MATCH (tr)-[:OBJECT]->(o)
            RETURN tr, s, o, elementId(s) AS s_id, elementId(o) AS o_id
            """
            params = {"keys": list(current_keys), "limit": per_hop_limit, **params_extra}

        res = run_cypher_with_retry(cypher, params)

        next_ids: Set[str] = set()
        next_keys: Set[str] = set()
        for r in res:
            tr = r["tr"]; s = r["s"]; o = r["o"]
            uid = tr.get("triple_uid")
            pred = normalize_predicate_for_test(tr.get("predicate"))
            s_type = s.get("type") if s else None
            o_type = o.get("type") if o else None

            keep = True
            if allowed_types_set is not None:
                keep = ((s_type in allowed_types_set) or (o_type in allowed_types_set))
            # allowed_predicates was already applied in Cypher WHERE, but we guard as well
            if allowed_predicates is not None and pred not in [normalize_predicate_for_test(p) for p in allowed_predicates]:
                keep = False

            if keep:
                if uid not in triples:
                    triples[uid] = {
                        "triple_uid": uid,
                        "predicate": tr.get("predicate"),
                        "uu_number": tr.get("uu_number"),
                        "evidence_quote": tr.get("evidence_quote"),
                        "embedding": tr.get("embedding"),
                        "document_id": tr.get("document_id"),
                        "chunk_id": tr.get("chunk_id"),
                        "evidence_article_ref": tr.get("evidence_article_ref"),
                        "subject": s.get("name") if s else None,
                        "subject_key": s.get("key") if s else None,
                        "subject_type": s.get("type") if s else None,
                        "object": o.get("name") if o else None,
                        "object_key": o.get("key") if o else None,
                        "object_type": o.get("type") if o else None,
                    }
                # Seed next hop only from kept triples to constrain expansion when filters are active
                if s and s.get("key"): next_keys.add(s.get("key"))
                if o and o.get("key"): next_keys.add(o.get("key"))
                s_id = r.get("s_id");  o_id = r.get("o_id")
                if s_id: next_ids.add(s_id)
                if o_id: next_ids.add(o_id)

        current_ids = next_ids if next_ids else set()
        current_keys = set() if next_ids else next_keys

    return list(triples.values())

# ----------------- Chunk store -----------------
class ChunkStore:
    def __init__(self, root: Path, skip: Set[str]):
        self.root = root
        self.skip = skip
        self._index: Dict[Tuple[str, str], str] = {}
        self._by_chunk: Dict[str, List[Tuple[str, str]]] = {}
        self._loaded_files: Set[Path] = set()
        self._built = False

    def _build_index(self):
        if self._built:
            return
        start = time.monotonic()
        log(f"[ChunkStore] Building index from {self.root}...")
        pkls = [p for p in self.root.glob("*.pkl") if p.name not in self.skip]

        total_chunks_indexed = 0
        for pkl in pkls:
            try:
                with open(pkl, "rb") as f:
                    chunks = pickle.load(f)
                loaded_count = 0
                for ch in chunks:
                    meta = getattr(ch, "metadata", {}) if hasattr(ch, "metadata") else {}
                    doc_id = _norm_id(meta.get("document_id"))
                    chunk_id = _norm_id(meta.get("chunk_id"))
                    text = getattr(ch, "page_content", None)
                    if doc_id and chunk_id and isinstance(text, str):
                        self._index[(doc_id, chunk_id)] = text
                        self._by_chunk.setdefault(chunk_id, []).append((doc_id, chunk_id))
                        loaded_count += 1
                self._loaded_files.add(pkl)
                total_chunks_indexed += loaded_count
                log(f"[ChunkStore] Loaded {loaded_count} chunks from {pkl.name}")
            except Exception as e:
                log(f"[ChunkStore] Failed to load or process {pkl.name}: {e}")
                continue

        elapsed = time.monotonic() - start
        log(f"[ChunkStore] Index built. Total chunks indexed: {total_chunks_indexed} from {len(self._loaded_files)} files in {elapsed:.3f}s.")
        self._built = True

    def get_chunk(self, document_id: Any, chunk_id: Any) -> Optional[str]:
        if not self._built:
            self._build_index()
        doc_id_s = _norm_id(document_id)
        chunk_id_s = _norm_id(chunk_id)
        val = self._index.get((doc_id_s, chunk_id_s))
        if val is not None:
            log(f"[ChunkStore] HIT exact: doc={doc_id_s} chunk={chunk_id_s} len={len(val)}")
            return val
        if "::" in chunk_id_s:
            base_id = chunk_id_s.split("::", 1)[0]
            val = self._index.get((doc_id_s, base_id))
            if val is not None:
                log(f"[ChunkStore] HIT base-id: doc={doc_id_s} chunk={chunk_id_s} -> base={base_id} len={len(val)}")
                return val
        matches = self._by_chunk.get(chunk_id_s)
        if matches:
            chosen_doc, chosen_chunk = matches[0]
            val = self._index.get((chosen_doc, chosen_chunk))
            if val is not None:
                note = "" if len(matches) == 1 else f" (warn: chunk_id occurs in {len(matches)} docs; chose doc={chosen_doc})"
                log(f"[ChunkStore] HIT by chunk_id only: requested doc={doc_id_s} chunk={chunk_id_s}; using doc={chosen_doc}{note}. len={len(val)}")
                return val
        log(f"[ChunkStore] MISS: doc={doc_id_s} chunk={chunk_id_s} (no exact/base-id/chunk-id-only match)")
        return None

# ----------------- Scoring helpers -----------------
def mean_similarity_to_query_triples(triple_emb: Optional[List[float]], q_trip_embs: List[List[float]]) -> float:
    if not isinstance(triple_emb, list) and triple_emb is not None:
        triple_emb = _as_float_list(triple_emb)
    if not isinstance(triple_emb, list) or not q_trip_embs:
        return 0.0
    sims = [cos_sim(triple_emb, q) for q in q_trip_embs]
    if not sims:
        return 0.0
    return sum(sims) / len(sims)

# ----------------- Core retrieval: entity- and triple-centric -----------------
def entity_centric_retrieval(
    query_entities: List[Dict[str, Any]],
    q_trip_embs: List[List[float]],
    q_emb_fallback: Optional[List[float]] = None
) -> List[Dict[str, Any]]:
    all_matched_keys: Set[str] = set()
    all_matched_ids: Set[str] = set()
    for e in query_entities:
        text = (e.get("text") or "").strip()
        if not text:
            continue
        try:
            e_emb = embed_text(text)
        except Exception as ex:
            log(f"[EntityRetrieval] Embedding failed for entity '{text}': {ex}", level="WARN")
            continue
        matches = search_similar_entities_by_embedding(e_emb, k=ENTITY_MATCH_TOP_K)
        keys = [m.get("key") for m in matches if m.get("key")]
        ids  = [m.get("elem_id") for m in matches if m.get("elem_id")]
        all_matched_keys.update(keys); all_matched_ids.update(ids)

    if not (all_matched_keys or all_matched_ids):
        log("[EntityRetrieval] No KG entity matches found from query entities.")
        return []

    t0 = now_ms()
    expanded_triples = expand_from_entities(
        list(all_matched_keys),
        hops=ENTITY_SUBGRAPH_HOPS,
        per_hop_limit=ENTITY_SUBGRAPH_PER_HOP_LIMIT,
        entity_elem_ids=list(all_matched_ids) if all_matched_ids else None
    )
    log(f"[EntityRetrieval] Expanded subgraph triples: {len(expanded_triples)} | {dur_ms(t0):.0f} ms")
    if not expanded_triples:
        return []

    def score(t: Dict[str, Any]) -> float:
        emb = t.get("embedding")
        if q_trip_embs:
            return mean_similarity_to_query_triples(emb, q_trip_embs)
        if q_emb_fallback and isinstance(emb, list):
            return cos_sim(q_emb_fallback, emb)
        return 0.0

    ranked = sorted(expanded_triples, key=score, reverse=True)
    top = ranked[:SUBGRAPH_TRIPLES_TOP_K]
    log(f"[EntityRetrieval] Selected top-{len(top)} triples from subgraph")
    return top

def triple_centric_retrieval(query_triples: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[List[float]]]:
    triples_map: Dict[str, Dict[str, Any]] = {}
    q_trip_embs: List[List[float]] = []
    for qt in query_triples:
        try:
            txt = query_triple_to_text(qt)
            emb = embed_text(txt)
            q_trip_embs.append(emb)
        except Exception as ex:
            log(f"[TripleRetrieval] Embedding failed for query triple '{qt}': {ex}", level="WARN")
            continue
        matches = search_similar_triples_by_embedding(emb, k=QUERY_TRIPLE_MATCH_TOP_K_PER)
        for m in matches:
            uid = m.get("triple_uid")
            if uid:
                if uid not in triples_map or (m.get("score", 0.0) > triples_map[uid].get("score", 0.0)):
                    triples_map[uid] = m
    merged = list(triples_map.values())
    log(f"[TripleRetrieval] Collected {len(merged)} unique KG triples across {len(q_trip_embs)} query triple(s)")
    return merged, q_trip_embs

# ----------------- Version-aware retrieval (Branch B): predicate + node-type driven -----------------
def mod_centric_retrieval(
    query_entities: List[Dict[str, Any]],
    query_triples: List[Dict[str, Any]],
    q_trip_embs_existing: List[List[float]],
    q_emb_fallback: Optional[List[float]] = None
) -> Tuple[List[Dict[str, Any]], List[List[float]], Dict[str, int]]:
    """
    Returns (merged_mod_triples, all_q_trip_embs_used_for_mod, counters).
    Combines:
      B1 (predicate-driven):
        - triple-centric filter by VERSION_PREDICATES_ALLOWED
        - entity-centric expansion constrained by allowed predicates
      B2 (node-type–driven):
        - triple-centric: subject_type/object_type in MOD_NODE_TYPES
        - entity-centric expansion constrained by node types
    """
    counters = {"B1_triple":0, "B1_entity":0, "B2_triple":0, "B2_entity":0}
    mod_map: Dict[str, Dict[str, Any]] = {}
    q_embs: List[List[float]] = list(q_trip_embs_existing)

    # Helper to add if better
    def _add(m: Dict[str, Any]):
        uid = m.get("triple_uid")
        if not uid:
            return
        if uid not in mod_map or (m.get("score", 0.0) > mod_map[uid].get("score", 0.0)):
            mod_map[uid] = m

    # B1a: Predicate-driven triple-centric
    for qt in query_triples:
        try:
            txt = query_triple_to_text(qt)
            emb = embed_text(txt)
            q_embs.append(emb)
        except Exception as ex:
            log(f"[ModRetrieval:B1a] Embedding failed for query triple '{qt}': {ex}", level="WARN")
            continue
        matches = search_similar_triples_by_embedding(emb, k=QUERY_TRIPLE_MATCH_TOP_K_PER * 2)
        hit = 0
        for m in matches:
            pred = normalize_predicate_for_test(m.get("predicate"))
            if pred in VERSION_PREDICATES_ALLOWED:
                _add(m); hit += 1
        counters["B1_triple"] += hit

    # B1b: Predicate-driven entity-centric expansion
    all_matched_keys: Set[str] = set(); all_matched_ids: Set[str] = set()
    for e in query_entities:
        text = (e.get("text") or "").strip()
        if not text:
            continue
        try:
            e_emb = embed_text(text)
        except Exception as ex:
            log(f"[ModRetrieval:B1b] Embedding failed for entity '{text}': {ex}", level="WARN")
            continue
        matches = search_similar_entities_by_embedding(e_emb, k=ENTITY_MATCH_TOP_K)
        for m in matches:
            if m.get("key"): all_matched_keys.add(m.get("key"))
            if m.get("elem_id"): all_matched_ids.add(m.get("elem_id"))
    if all_matched_keys or all_matched_ids:
        expanded = expand_from_entities(
            list(all_matched_keys),
            hops=ENTITY_SUBGRAPH_HOPS,
            per_hop_limit=ENTITY_SUBGRAPH_PER_HOP_LIMIT,
            entity_elem_ids=list(all_matched_ids) if all_matched_ids else None,
            allowed_predicates=list(VERSION_PREDICATES_ALLOWED),
            allowed_node_types=None
        )
        for m in expanded:
            _add(m)
        counters["B1_entity"] += len(expanded)

    # B2a: Node-type–driven triple-centric
    for qt in query_triples:
        try:
            txt = query_triple_to_text(qt)
            emb = embed_text(txt)
            q_embs.append(emb)
        except Exception as ex:
            log(f"[ModRetrieval:B2a] Embedding failed for query triple '{qt}': {ex}", level="WARN")
            continue
        matches = search_similar_triples_by_embedding(emb, k=QUERY_TRIPLE_MATCH_TOP_K_PER * 2)
        hit = 0
        for m in matches:
            st = m.get("subject_type")
            ot = m.get("object_type")
            if (st in MOD_NODE_TYPES) or (ot in MOD_NODE_TYPES):
                _add(m); hit += 1
        counters["B2_triple"] += hit

    # B2b: Node-type–driven entity-centric expansion
    if all_matched_keys or all_matched_ids:
        expanded_nt = expand_from_entities(
            list(all_matched_keys),
            hops=ENTITY_SUBGRAPH_HOPS,
            per_hop_limit=ENTITY_SUBGRAPH_PER_HOP_LIMIT,
            entity_elem_ids=list(all_matched_ids) if all_matched_ids else None,
            allowed_predicates=None,
            allowed_node_types=list(MOD_NODE_TYPES)
        )
        for m in expanded_nt:
            _add(m)
        counters["B2_entity"] += len(expanded_nt)

    merged_mod = list(mod_map.values())
    log(f"[ModRetrieval] Combined mod triples: {len(merged_mod)} | counters={counters}")
    return merged_mod, q_embs, counters

# ----------------- Collect chunks -----------------
def collect_chunks_for_triples(
    triples: List[Dict[str, Any]],
    chunk_store: ChunkStore
) -> List[Tuple[Tuple[Any, Any], str, Dict[str, Any]]]:
    seen_pairs: Set[Tuple[Any, Any]] = set()
    out: List[Tuple[Tuple[Any, Any], str, Dict[str, Any]]] = []
    for t in triples:
        doc_id = t.get("document_id"); chunk_id = t.get("chunk_id")
        if doc_id is None or chunk_id is None:
            quote = t.get("evidence_quote")
            if quote:
                key = (t.get("triple_uid"), "quote")
                if key not in seen_pairs:
                    t["_is_quote_fallback"] = True
                    out.append((key, quote, t))
                    seen_pairs.add(key)
            continue
        norm_key = (_norm_id(doc_id), _norm_id(chunk_id))
        if norm_key in seen_pairs:
            continue
        text = chunk_store.get_chunk(doc_id, chunk_id)
        if text:
            t["_is_quote_fallback"] = False
            out.append((norm_key, text, t))
            seen_pairs.add(norm_key)
        else:
            quote = t.get("evidence_quote")
            if quote:
                key2 = (t.get("triple_uid"), "quote")
                if key2 not in seen_pairs:
                    t["_is_quote_fallback"] = True
                    log(f"[ChunkStore] FALLBACK to quote for doc={_norm_id(doc_id)} chunk={_norm_id(chunk_id)}", level="WARN")
                    out.append((key2, quote, t))
                    seen_pairs.add(key2)
    return out

# ----------------- Temporal resolver -----------------
def _triple_effective_date_guess(t: Dict[str, Any]) -> Optional[datetime.date]:
    for field in [t.get("evidence_quote"), t.get("evidence_article_ref"), t.get("object"), t.get("subject")]:
        if isinstance(field, str):
            dt = parse_date_any(field)
            if dt:
                return dt
    uu = (t.get("uu_number") or "").strip()
    m = re.match(r"(\d{1,4})/(\d{4})", uu)
    if m:
        try:
            return datetime.date(int(m.group(2)), 1, 1)
        except Exception:
            return None
    return None

def build_version_index(mod_triples: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Lightweight version index keyed by UU number 'X/YYYY'.
    Only explicit doc-level amend/repeal predicates influence status.
    """
    index: Dict[str, Dict[str, Any]] = {}
    def _uu_bucket(u: Optional[str]) -> Optional[str]:
        if not u: return None
        return u.replace(" ", "")

    for t in mod_triples:
        pred = normalize_predicate_for_test(t.get("predicate"))
        subj_uu = extract_uu_number_from_text(t.get("subject") or "")
        obj_uu  = extract_uu_number_from_text(t.get("object") or "")
        triple_doc_uu = (t.get("uu_number") or "").strip() or None
        eff_date = _triple_effective_date_guess(t)

        if pred in {"amends", "mengubah", "diubah"}:
            amender = _uu_bucket(subj_uu or triple_doc_uu)
            amended = _uu_bucket(obj_uu)
            if amended and amender:
                d = index.setdefault(amended, {"amended_by": [], "repealed_by": []})
                d["amended_by"].append({"uu": amender, "date": eff_date, "triple_uid": t.get("triple_uid")})
        elif pred in {"repeals", "mencabut", "dicabut", "tidak_berlaku"}:
            repealer = _uu_bucket(subj_uu or triple_doc_uu)
            repealed = _uu_bucket(obj_uu)
            if repealed and repealer:
                d = index.setdefault(repealed, {"amended_by": [], "repealed_by": []})
                d["repealed_by"].append({"uu": repealer, "date": eff_date, "triple_uid": t.get("triple_uid")})
        else:
            # Node-type–driven content changes are not treated as full doc repeal/amend without explicit doc-level links
            pass
    return index

def resolve_doc_status(uu: str, idx: Dict[str, Dict[str, Any]], as_of_date: datetime.date) -> Dict[str, Any]:
    info = idx.get(uu, {"amended_by": [], "repealed_by": []})
    rep_candidates = [e for e in info.get("repealed_by", []) if e.get("date") is None or e["date"] <= as_of_date]
    repealed = None
    if rep_candidates:
        repealed = sorted(rep_candidates, key=lambda x: (x["date"] or datetime.date.min))[0]
    latest_amend = None
    if not repealed:
        amend_candidates = [e for e in info.get("amended_by", []) if e.get("date") is None or e["date"] <= as_of_date]
        if amend_candidates:
            latest_amend = sorted(amend_candidates, key=lambda x: (x["date"] or datetime.date.min))[-1]
    return {
        "uu": uu,
        "is_repealed": repealed is not None,
        "repealed_by": repealed,
        "latest_amended_by": latest_amend
    }

def summarize_lineage_for_docs(target_uu_list: List[str], idx: Dict[str, Dict[str, Any]], as_of_date: datetime.date, lang: str) -> str:
    lines: List[str] = []
    for uu in sorted(set([u for u in target_uu_list if u])):
        status = resolve_doc_status(uu, idx, as_of_date)
        if status["is_repealed"]:
            rb = status["repealed_by"]
            when = rb.get("date").strftime("%d-%m-%Y") if rb and rb.get("date") else "tanggal tidak jelas"
            line = f"UU {uu} telah dicabut oleh UU {rb.get('uu')} (efektif {when})."
        else:
            am = status["latest_amended_by"]
            if am:
                when = am.get("date").strftime("%d-%m-%Y") if am and am.get("date") else "tanggal tidak jelas"
                line = f"UU {uu} (versi berlaku) terakhir diubah oleh UU {am.get('uu')} (efektif {when})."
            else:
                line = f"UU {uu} (tidak terdeteksi perubahan atau pencabutan hingga tanggal acuan)."
        lines.append(line)
    if not lines:
        return "(Tidak ada garis keturunan versi yang relevan terdeteksi.)" if lang == "id" else "(No relevant version lineage detected.)"
    return " ".join(lines)

# ----------------- Temporal-aware filtering and reweighting -----------------
def temporal_filter_and_weight_chunks(
    chunk_records: List[Tuple[Tuple[Any, Any], str, Dict[str, Any]]],
    version_index: Dict[str, Dict[str, Any]],
    as_of_date: datetime.date,
    exclude_repealed: bool = True
) -> Tuple[List[Tuple[Tuple[Any, Any], str, Dict[str, Any]]], List[Tuple[Tuple[Any, Any], str, Dict[str, Any], str]]]:
    kept: List[Tuple[Tuple[Any, Any], str, Dict[str, Any]]] = []
    excluded: List[Tuple[Tuple[Any, Any], str, Dict[str, Any], str]] = []
    for rec in chunk_records:
        key, text, t = rec
        pred = normalize_predicate_for_test(t.get("predicate"))
        uu = (t.get("uu_number") or "").strip()
        if not uu:
            kept.append(rec)
            continue
        status = resolve_doc_status(uu, version_index, as_of_date)
        if exclude_repealed and status["is_repealed"]:
            if pred in VERSION_PREDICATES_ALLOWED or (t.get("subject_type") in MOD_NODE_TYPES) or (t.get("object_type") in MOD_NODE_TYPES):
                kept.append(rec)
            else:
                excluded.append((key, text, t, f"Excluded repealed UU {uu} as of {as_of_date.isoformat()}"))
        else:
            kept.append(rec)
    return kept, excluded

def temporal_reweight_chunk_score(base_score: float, t: Dict[str, Any], version_index: Dict[str, Dict[str, Any]], as_of_date: datetime.date) -> float:
    uu = (t.get("uu_number") or "").strip()
    if not uu:
        return base_score
    status = resolve_doc_status(uu, version_index, as_of_date)
    score = base_score
    if status["is_repealed"]:
        score -= TEMPORAL_DOWNWEIGHT_SUPERSEDED
    else:
        if status["latest_amended_by"]:
            score += TEMPORAL_BOOST_FOR_CURRENT
    return score

def rerank_chunks_by_query(
    chunk_records: List[Tuple[Tuple[Any, Any], str, Dict[str, Any]]],
    q_emb_query: List[float],
    top_k: int,
    version_index: Optional[Dict[str, Dict[str, Any]]] = None,
    as_of_date: Optional[datetime.date] = None
) -> List[Tuple[Tuple[Any, Any], str, Dict[str, Any], float]]:
    cand = chunk_records[:CHUNK_RERANK_CAND_LIMIT]
    t0 = now_ms()
    scored: List[Tuple[Tuple[Any, Any], str, Dict[str, Any], float]] = []
    for key, text, t in cand:
        try:
            emb = embed_text(text)
            s = cos_sim(q_emb_query, emb)
            if version_index is not None and as_of_date is not None:
                s = temporal_reweight_chunk_score(s, t, version_index, as_of_date)
            scored.append((key, text, t, s))
        except Exception as ex:
            log(f"[ChunkRerank] Embedding failed for chunk {key}: {ex}", level="WARN")
            continue
    scored.sort(key=lambda x: x[3], reverse=True)
    took = dur_ms(t0)
    log(f"[ChunkRerank] Scored {len(scored)} candidates | picked top {min(top_k, len(scored))} | {took:.0f} ms")
    return scored[:top_k]

def rerank_triples_by_query_triples(
    triples: List[Dict[str, Any]],
    q_trip_embs: List[List[float]],
    q_emb_fallback: Optional[List[float]],
    top_k: int
) -> List[Dict[str, Any]]:
    t0 = now_ms()
    def score(t: Dict[str, Any]) -> float:
        emb = t.get("embedding")
        if q_trip_embs:
            return mean_similarity_to_query_triples(emb, q_trip_embs)
        if q_emb_fallback and isinstance(emb, list):
            return cos_sim(q_emb_fallback, emb)
        return 0.0
    ranked = sorted(triples, key=score, reverse=True)
    took = dur_ms(t0)
    log(f"[TripleRerank] Input={len(triples)} | Output={min(top_k, len(ranked))} | {took:.0f} ms")
    return ranked[:top_k]

# ----------------- Build combined context and lineage -----------------
def build_combined_context_text(
    triples_ranked: List[Dict[str, Any]],
    chunks_ranked: List[Tuple[Tuple[Any, Any], str, Dict[str, Any], float]],
    version_lineage_summary: str
) -> Tuple[str, str, List[Dict[str, Any]]]:
    summary_lines = []
    summary_lines.append("Ringkasan triple yang relevan:")
    for t in triples_ranked[:min(50, len(triples_ranked))]:
        s = t.get("subject"); p = t.get("predicate"); o = t.get("object")
        uu = t.get("uu_number") or ""
        art = t.get("evidence_article_ref") or ""
        quote = (t.get("evidence_quote") or "")[:300]
        summary_lines.append(f"- {s} [{p}] {o} | {uu} | {art} | bukti: {quote}")
    summary_text = "\n".join(summary_lines)

    lines = []
    lines.append("Version lineage (ringkas):")
    lines.append(version_lineage_summary.strip() or "(n/a)")
    lines.append("")
    lines.append(summary_text)
    lines.append("\nPotongan teks terkait (chunk):")
    for idx, (key, text, t, score) in enumerate(chunks_ranked, start=1):
        doc_id = t.get("document_id")
        chunk_id = t.get("chunk_id")
        uu = t.get("uu_number") or ""
        fb = " | quote-fallback" if t.get("_is_quote_fallback") else ""
        lines.append(f"[Chunk {idx}] doc={doc_id} chunk={chunk_id} | {uu} | score={score:.3f}{fb}\n{text}")
    context = "\n".join(lines)

    chunk_records = [{"key": key, "text": text, "triple": t, "score": score} for key, text, t, score in chunks_ranked]
    return context, summary_text, chunk_records

# ----------------- Agent 2 (Answerer) -----------------
def agent2_answer(
    query_original: str,
    context: str,
    guidance: Optional[str],
    output_lang: str,
    as_of_date: datetime.date,
    version_lineage_summary: str
) -> str:
    instructions = (
        "Answer concisely and accurately based strictly on the provided context. "
        "Use the version of the law that is effective as of the specified date. "
        "If newer amending or repealing provisions exist as of that date, they take precedence over older text. "
        "If the question is historical (as-of a past date), use the provisions effective on that date and note later changes if relevant. "
        "Cite UU/Article references when they are clear. "
        "Respond in the same language as the user's question."
    )
    guidance_text = guidance.strip() if isinstance(guidance, str) and guidance.strip() else "(No additional guidance from the Judge in the previous iteration.)"
    as_of_line = f"As-of date: {as_of_date.strftime('%d-%m-%Y')}"
    lineage_line = f"Version lineage (short): {version_lineage_summary}"

    prompt = f"""
You are Agent 2 (Answerer). Task: provide an answer based on the context only.

Core instructions:
{instructions}

Additional guidance from Judge (if any):
\"\"\"{guidance_text}\"\"\"

{as_of_line}
{lineage_line}

Original user question:
\"\"\"{query_original}\"\"\"

Context:
\"\"\"{context}\"\"\"
"""
    est_tokens = estimate_tokens_for_text(prompt)
    log("\n[Agent 2] Prompt:")
    log(prompt)
    log(f"[Agent 2] Prompt size: {len(prompt)} chars, est_tokens≈{est_tokens}")

    t0 = now_ms()
    answer = safe_generate_text(prompt, max_tokens=ANSWER_MAX_TOKENS, temperature=0.2)
    took = dur_ms(t0)
    log(f"[Agent 2] Intermediate answer length={len(answer)} | {took:.0f} ms")
    return answer

# ----------------- Embedding with indefinite retry -----------------
def embed_text(text: str) -> List[float]:
    t0 = now_ms()
    res = _api_call_with_retry(genai.embed_content, model=EMBED_MODEL, content=text)
    vec = None
    if isinstance(res, dict):
        emb = res.get("embedding")
        if isinstance(emb, dict) and "values" in emb:
            vec = emb["values"]
        elif isinstance(emb, list):
            vec = emb
    if vec is None:
        try:
            vec = res.embedding.values  # type: ignore[attr-defined]
        except Exception:
            pass
    if vec is None:
        raise RuntimeError("Unexpected embedding response shape for embeddings")
    out = _as_float_list(vec)
    log(f"[Embed] text_len={len(text)} -> vec_len={len(out)} | {dur_ms(t0):.0f} ms", level="DEBUG")
    return out

# ----------------- Single-pass GraphRAG with Version-aware logic -----------------
def agentic_graph_rag(query_original: str) -> Dict[str, Any]:
    global _LOGGER
    ts_name = make_timestamp_name()
    log_file = Path.cwd() / f"{ts_name}.txt"
    _LOGGER = FileLogger(log_file, also_console=True)

    t_all = now_ms()
    try:
        log("=== Agentic GraphRAG run started ===")
        log(f"Process info: pid={_pid()}")
        log(f"Log file: {log_file}")
        log(f"Original Query: {query_original}")
        log(f"Parameters: ENTITY_MATCH_TOP_K={ENTITY_MATCH_TOP_K}, ENTITY_SUBGRAPH_HOPS={ENTITY_SUBGRAPH_HOPS}, "
            f"ENTITY_SUBGRAPH_PER_HOP_LIMIT={ENTITY_SUBGRAPH_PER_HOP_LIMIT}, SUBGRAPH_TRIPLES_TOP_K={SUBGRAPH_TRIPLES_TOP_K}, "
            f"QUERY_TRIPLE_MATCH_TOP_K_PER={QUERY_TRIPLE_MATCH_TOP_K_PER}, MAX_TRIPLES_FINAL={MAX_TRIPLES_FINAL}, "
            f"MAX_CHUNKS_FINAL={MAX_CHUNKS_FINAL}, CHUNK_RERANK_CAND_LIMIT={CHUNK_RERANK_CAND_LIMIT}, "
            f"ANSWER_MAX_TOKENS={ANSWER_MAX_TOKENS}, MAX_ITERS={MAX_ITERS}, OUTPUT_LANG={OUTPUT_LANG}, "
            f"VERSION_RESERVED_CHUNKS_MIN={VERSION_RESERVED_CHUNKS_MIN}, EXCLUDE_REPEALED_CONTENT={EXCLUDE_REPEALED_CONTENT}")
        log(f"Neo4j retry/timeout: NEO4J_MAX_ATTEMPTS={NEO4J_MAX_ATTEMPTS}, NEO4J_TX_TIMEOUT_S={NEO4J_TX_TIMEOUT_S:.1f}")
        log("Mode: Single-pass (Agents 1 & 2 only; no Judge, no iterations)")

        chunk_store = ChunkStore(LANGCHAIN_DIR, set(SKIP_FILES))

        # Language detection
        user_lang = detect_user_language(query_original)
        log(f"[Language] Detected user language: {user_lang}")

        # As-of parsing
        version_intent, as_of_date = parse_as_of_intent(query_original)
        log(f"[As-Of] Intent={version_intent} | as_of_date={as_of_date.isoformat()}")

        # Step 0: Whole-query embedding
        t0 = now_ms()
        q_emb_query = embed_text(query_original)
        log(f"[Step 0] Whole-query embedding in {dur_ms(t0):.0f} ms")

        # Step 1: Agent 1 – extract entities/predicates
        t1 = now_ms()
        extraction = agent1_extract_entities_predicates(query_original)
        ents = extraction.get("entities", [])
        _ = extraction.get("predicates", [])
        log(f"[Step 1] Entity/Predicate extraction done in {dur_ms(t1):.0f} ms")

        # Step 1b: extract query triples
        t1b = now_ms()
        query_triples = agent1b_extract_query_triples(query_original)
        log(f"[Step 1b] Query triple extraction done in {dur_ms(t1b):.0f} ms")

        # Branch A: Core topical retrieval
        t2 = now_ms()
        ctx2_triples, q_trip_embs = triple_centric_retrieval(query_triples)
        log(f"[Branch A] Triple-centric retrieval in {dur_ms(t2):.0f} ms; ctx2_triples={len(ctx2_triples)}, q_trip_embs={len(q_trip_embs)}")

        t3 = now_ms()
        ctx1_triples = entity_centric_retrieval(ents, q_trip_embs=q_trip_embs, q_emb_fallback=q_emb_query)
        log(f"[Branch A] Entity-centric retrieval in {dur_ms(t3):.0f} ms; ctx1_triples={len(ctx1_triples)}")

        # Branch B: Version-aware retrieval (predicate + node-type)
        t3b = now_ms()
        mod_triples, q_trip_embs_mod, mod_counters = mod_centric_retrieval(ents, query_triples, q_trip_embs, q_emb_fallback=q_emb_query)
        log(f"[Branch B] Version-aware retrieval in {dur_ms(t3b):.0f} ms; mod_triples={len(mod_triples)} | counters={mod_counters}")

        # Build version index from mod_triples (doc-level status via explicit predicates)
        version_index = build_version_index(mod_triples)

        # Merge A + B triples
        t4 = now_ms()
        triple_map: Dict[str, Dict[str, Any]] = {}
        for t in ctx1_triples + ctx2_triples + mod_triples:
            uid = t.get("triple_uid")
            if uid:
                prev = triple_map.get(uid)
                if prev is None or (t.get("score", 0.0) > prev.get("score", 0.0)):
                    triple_map[uid] = t
        merged_triples = list(triple_map.values())
        log(f"[Step 4] Merged triples from branches: {len(merged_triples)} | {dur_ms(t4):.0f} ms")

        # Step 5: Gather chunks
        t5 = now_ms()
        all_chunk_records = collect_chunks_for_triples(merged_triples, chunk_store)
        log(f"[Step 5] Collected {len(all_chunk_records)} chunk candidates (pre-temporal-filter)")

        # Temporal filter: exclude repealed content (except mod-evidence)
        kept_chunks, excluded_chunks = temporal_filter_and_weight_chunks(
            all_chunk_records, version_index, as_of_date, exclude_repealed=EXCLUDE_REPEALED_CONTENT
        )
        if excluded_chunks:
            log("[Temporal] Excluded repealed-content chunks (kept out of final context):")
            for (key, _text, t, reason) in excluded_chunks[:100]:
                log(f"  - key={key} uu={t.get('uu_number')} pred={t.get('predicate')} reason={reason}")

        # Reserve min version-lineage chunks from Branch B
        mod_uids = {t.get("triple_uid") for t in mod_triples}
        kept_mod_chunks = [rec for rec in kept_chunks if (rec[2].get("triple_uid") in mod_uids)]
        kept_main_chunks = [rec for rec in kept_chunks if (rec[2].get("triple_uid") not in mod_uids)]
        log(f"[Temporal] Kept chunks: total={len(kept_chunks)} | mod={len(kept_mod_chunks)} | main={len(kept_main_chunks)}")

        # Rerank with temporal weights
        chunks_ranked_mod = rerank_chunks_by_query(
            kept_mod_chunks, q_emb_query, top_k=min(VERSION_RESERVED_CHUNKS_MIN, MAX_CHUNKS_FINAL),
            version_index=version_index, as_of_date=as_of_date
        )
        chunks_ranked_main = rerank_chunks_by_query(
            kept_main_chunks, q_emb_query,
            top_k=max(1, MAX_CHUNKS_FINAL - min(VERSION_RESERVED_CHUNKS_MIN, len(kept_mod_chunks))),
            version_index=version_index, as_of_date=as_of_date
        )

        final_chunks_ranked = (chunks_ranked_mod[:VERSION_RESERVED_CHUNKS_MIN]) + chunks_ranked_main
        final_chunks_ranked = final_chunks_ranked[:MAX_CHUNKS_FINAL]
        log(f"[Step 5] Chunk rerank: selected={len(final_chunks_ranked)} (reserved mod={min(VERSION_RESERVED_CHUNKS_MIN, len(chunks_ranked_mod))}) | took {dur_ms(t5):.0f} ms")

        # Step 6: Rerank triples (for summary)
        t6 = now_ms()
        triples_ranked = rerank_triples_by_query_triples(merged_triples, q_trip_embs=q_trip_embs_mod, q_emb_fallback=q_emb_query, top_k=MAX_TRIPLES_FINAL)
        log(f"[Step 6] Triple rerank done in {dur_ms(t6):.0f} ms; selected {len(triples_ranked)}")

        # Lineage summary for UUs in scope
        candidate_uu: Set[str] = set()
        for e in ents:
            uu = extract_uu_number_from_text(e.get("text"))
            if uu: candidate_uu.add(uu)
        for _key, _text, t in kept_chunks:
            uu = (t.get("uu_number") or "").strip()
            if uu: candidate_uu.add(uu)
        version_lineage_summary = summarize_lineage_for_docs(list(candidate_uu), version_index, as_of_date, user_lang)

        # Build combined context
        t_ctx = now_ms()
        context_text, context_summary, _ = build_combined_context_text(triples_ranked, final_chunks_ranked, version_lineage_summary)
        log(f"[Context] Built in {dur_ms(t_ctx):.0f} ms")
        log("\n[Context summary for this pass]:")
        log(context_summary)

        # Step 7: Answer
        t7 = now_ms()
        intermediate_answer = agent2_answer(
            query_original, context_text, guidance=None, output_lang=user_lang,
            as_of_date=as_of_date, version_lineage_summary=version_lineage_summary
        )
        log(f"[Step 7] Answer generated in {dur_ms(t7):.0f} ms")

        final_answer = intermediate_answer

        total_ms = dur_ms(t_all)
        log("\n=== Agentic GraphRAG summary ===")
        log(f"- Iterations used: 1")
        log(f"- Total runtime: {total_ms:.0f} ms")
        log("\n=== Final Answer ===")
        log(final_answer)
        log(f"\nLogs saved to: {log_file}")

        return {
            "final_answer": final_answer,
            "log_file": str(log_file),
            "iterations": 1,
            "judge_reports": [],
            "as_of_date": as_of_date.isoformat(),
            "version_lineage": version_lineage_summary
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
            result = agentic_graph_rag(user_query)
            if isinstance(result, dict):
                lineage = result.get("version_lineage") or ""
                asof = result.get("as_of_date") or ""
                print(f"\n[Version summary] as-of={asof}\n{lineage}")
    finally:
        try:
            driver.close()
        except Exception:
            pass