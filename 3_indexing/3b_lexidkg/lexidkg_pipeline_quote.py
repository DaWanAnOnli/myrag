import os, json, hashlib, time, threading, pickle, math, re, random
from collections import deque
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED

from dotenv import load_dotenv
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
from neo4j import GraphDatabase

# ----------------- Thread-safe logging with timestamps and attempt IDs -----------------
_PRINT_LOCK = threading.Lock()

def _ts() -> str:
    now = time.time()
    lt = time.localtime(now)
    ms = int((now - int(now)) * 1000)
    return time.strftime("%Y-%m-%d %H:%M:%S", lt) + f".{ms:03d}"

def _fmt_prefix(level: str, attempt_id: Optional[int]) -> str:
    ts = _ts()
    if attempt_id is None:
        return f"[{ts}] [{level}]"
    return f"[{ts}] [{level}] [attempt_id={attempt_id}]"

def log_info(msg: str, attempt_id: Optional[int] = None) -> None:
    with _PRINT_LOCK:
        print(f"{_fmt_prefix('INFO', attempt_id)} {msg}")

def log_warn(msg: str, attempt_id: Optional[int] = None) -> None:
    with _PRINT_LOCK:
        print(f"{_fmt_prefix('WARN', attempt_id)} {msg}")

def log_error(msg: str, attempt_id: Optional[int] = None) -> None:
    with _PRINT_LOCK:
        print(f"{_fmt_prefix('ERROR', attempt_id)} {msg}")

# ----------------- Load .env from the parent directory -----------------
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

# ----------------- Config from env with sensible defaults -----------------
GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]

def _normalize_model_name(m: str) -> str:
    if not m:
        return m
    m = m.strip()
    if m.startswith("models/") or m.startswith("tunedModels/"):
        return m
    return f"models/{m}"

GEN_MODEL = _normalize_model_name(os.getenv("GEN_MODEL", "gemini-2.5-flash-lite"))
EMBED_MODEL = _normalize_model_name(os.getenv("EMBED_MODEL", "text-embedding-004"))

# Directory of LangChain per-document pickle files
DEFAULT_LANGCHAIN_DIR = (Path(__file__).resolve().parent / "../dataset/langchain-results/samples").resolve()
LANGCHAIN_DIR = DEFAULT_LANGCHAIN_DIR

# Neo4j
NEO4J_URI  = os.getenv("NEO4J_URI",  "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASS", "@ik4nkus")

# LLM rate limiter (calls per minute)
LLM_MAX_CALLS_PER_MIN = int(os.getenv("LLM_MAX_CALLS_PER_MIN", "8"))

# Parallelism
INDEX_WORKERS = int(os.getenv("INDEX_WORKERS", "8"))

# Stagger worker starts (seconds) - randomized uniformly in [7.0, 17.0]
STAGGER_WORKER_SECONDS = random.uniform(7.0, 17.0)

# API budget controls
def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "y", "on")

ENFORCE_API_BUDGET = _env_bool("ENFORCE_API_BUDGET", True)
API_BUDGET_TOTAL = int(os.getenv("API_BUDGET_TOTAL", "10000"))
COUNT_EMBEDDINGS_IN_BUDGET = _env_bool("COUNT_EMBEDDINGS_IN_BUDGET", False)

# Files to skip explicitly
SKIP_FILES = {"all_langchain_documents.pkl"}

# Embedding dimensions for text-embedding-004 (for reference)
EMBED_DIM = 768

# Prompt token control (heuristic)
PROMPT_TOKEN_LIMIT = int(os.getenv("PROMPT_TOKEN_LIMIT", "8000"))
PRACTICAL_MAX_ITEMS_PER_BATCH = int(os.getenv("PRACTICAL_MAX_ITEMS_PER_BATCH", "40"))

# Single-chunk parse error split threshold
SINGLE_PARSE_SPLIT_AFTER = int(os.getenv("SINGLE_PARSE_SPLIT_AFTER", "2"))

# JSON output directory (must already exist; do not create it)
JSON_OUTPUT_DIR = Path(os.getenv("JSON_OUTPUT_DIR", str((Path(__file__).resolve().parent / "../../dataset/llm-json-outputs").resolve())))
if not JSON_OUTPUT_DIR.exists() or not JSON_OUTPUT_DIR.is_dir():
    raise FileNotFoundError(f"JSON_OUTPUT_DIR does not exist or is not a directory: {JSON_OUTPUT_DIR}. Please create it or set JSON_OUTPUT_DIR.")

# NEW: Evidence quote limit (hardcoded constant, easy to change)
QUOTE_MAX_WORDS = int(os.getenv("QUOTE_MAX_WORDS", "60"))

# ----------------- Initialize SDKs -----------------
genai.configure(api_key=GOOGLE_API_KEY)
try:
    gen_model = genai.GenerativeModel(GEN_MODEL)
    log_info(f"Using generation model: {GEN_MODEL}")
    log_info(f"Using embedding model:  {EMBED_MODEL}")
    log_info(f"LLM JSON outputs directory: {JSON_OUTPUT_DIR}")
except Exception as e:
    raise RuntimeError(f"Failed to initialize GenerativeModel {GEN_MODEL}: {e}")

# ----------------- LexID ontology constants -----------------
# LexID node labels
LEGAL_NODE_LABELS = [
    "LegalDocument", "LegalDocumentContent", "RuleExpression",
    "LawAmendment", "PlaceOfPromulgation", "Person", "Office", "City"
]

# LexID document types (with LegalDocument label)
LEGAL_DOCUMENT_TYPES = [
    "Constitution", "AmendmentToTheConstitution", "PeoplesConsultativeAssemblyResolution",
    "Act", "GovernmentRegulationInLieuOfAct", "GovernmentRegulation", "PresidentialRegulation",
    "PresidentialDecree", "PresidentialInstruction", "MinisterialRegulation",
    "MinisterialDecision", "MinisterialInstruction", "ProvincialRegulation",
    "RegencyRegulation", "MunicipalRegulation", "VillageRegulation"
]

# LexID content types (with LegalDocumentContent label)
LEGAL_CONTENT_TYPES = [
    "Chapter", "Part", "Paragraph", "Article", "Section", "Item"
]

# LexID expression types (with RuleExpression label)
RULE_EXPRESSION_TYPES = [
    "Norm", "RuleAct", "Concept", "CompoundExpression",
    "AndExpression", "OrExpression", "XorExpression"
]

# LexID amendment types (with LawAmendment label)
LAW_AMENDMENT_TYPES = [
    "LawAmendment", "LawAddition", "LawModification", "LawDeletion"
]

# LexID relationship types (uppercased, snake-case)
LEXID_RELATIONSHIPS = [
    # General Metadata
    "CONSIDERS", "HAS_DESCRIPTION", "HAS_NAME", "SAME_AS", "HAS_CREATOR",
    "HAS_DICTUM", "HAS_ENACTION_DATE", "HAS_ENACTION_OFFICIAL",
    "HAS_PROMULGATION_PLACE", "HAS_REGULATION_NUMBER", "HAS_REGULATION_YEAR", "HAS_LABEL",

    # Inter-Document
    "AMENDS", "AMENDED_BY", "IMPLEMENTS", "HAS_LEGAL_BASIS", "REPEALS",

    # Document Structure
    "HAS_CONTENT", "HAS_PART", "IS_CONTENT_OF", "IS_PART_OF",

    # Content Changes
    "ADDS", "DELETES", "MODIFIES", "HAS_ADDITION_CONTENT", "HAS_MODIFICATION_TARGET",

    # Semantic
    "HAS_ACT", "HAS_ACT_TYPE", "HAS_OBJECT", "HAS_SUBJECT", "HAS_CONDITION",
    "HAS_MODALITY", "HAS_QUALIFIER", "HAS_QUALIFIER_TYPE", "HAS_QUALIFIER_VALUE",
    "HAS_RULE", "REFERS_TO", "HAS_ELEMENT"
]

# Most common predicates (for prompt guidance)
COMMON_PREDICATES = [
    "mendefinisikan", "mengubah", "mencabut", "mulai_berlaku",
    "mewajibkan", "melarang", "memberikan_sanksi", "berlaku_untuk",
    "termuat_dalam", "mendelegasikan_kepada", "memiliki_bagian",
    "merujuk_pada", "merupakan_isi_dari", "menambahkan", "menghapus"
]

# Build a comprehensive allowed node types list for the JSON schema enum
ALLOWED_NODE_TYPES: List[str] = sorted(set(
    ["LegalDocument", "LegalDocumentContent", "RuleExpression", "LawAmendment",
     "PlaceOfPromulgation", "Person", "Office", "City"] +
    LEGAL_DOCUMENT_TYPES +
    LEGAL_CONTENT_TYPES +
    RULE_EXPRESSION_TYPES +
    LAW_AMENDMENT_TYPES
))

# ----------------- SYSTEM HINT (LexID ontology) -----------------
SYSTEM_HINT = f"""
You are an expert in Indonesian legal knowledge graphs. You extract structured information from legal documents (Undang-Undang) to build a knowledge graph following the LexID ontology.

## COMPLETE LEXID ONTOLOGY SPECIFICATION

### 1. NODE TYPES (LABELS)

#### Top-Level Node Labels:
- LegalDocument
- LegalDocumentContent
- RuleExpression
- LawAmendment
- PlaceOfPromulgation
- Person
- Office
- City

#### Legal Document Types (subtypes of LegalDocument):
- Constitution
- AmendmentToTheConstitution
- PeoplesConsultativeAssemblyResolution
- Act
- GovernmentRegulationInLieuOfAct
- GovernmentRegulation
- PresidentialRegulation
- PresidentialDecree
- PresidentialInstruction
- MinisterialRegulation
- MinisterialDecision
- MinisterialInstruction
- ProvincialRegulation
- RegencyRegulation
- MunicipalRegulation
- VillageRegulation

#### Document Content Types (subtypes of LegalDocumentContent):
- Chapter (Bab)
- Part (Bagian)
- Paragraph (Paragraf)
- Article (Pasal)
- Section (Ayat)
- Item (Sub-clauses: a., b., 1., 2.)

#### Rule Expression Types (subtypes of RuleExpression):
- Norm
- RuleAct
- Concept
- CompoundExpression (AndExpression, OrExpression, XorExpression)

#### Amendment Types (subtypes of LawAmendment):
- LawAddition, LawModification, LawDeletion

### 2. RELATIONSHIP TYPES
(As in the LexID list above.)

## EXTRACTION PATTERNS AND EXAMPLES
Examples below are simplified to the following shape (no canonical_id, no char spans):

- subject: {{text, type}}
- predicate: string (lower_snake_case)
- object: {{text, type}}
- evidence: {{quote}}

### 1. Document Identification
MATCH: "UNDANG-UNDANG REPUBLIK INDONESIA NOMOR 12 TAHUN 2011"
EXTRACT:
{{
  "subject": {{"text": "UU 12/2011", "type": "Act"}},
  "predicate": "has_regulation_number",
  "object": {{"text": "12", "type": "Concept"}},
  "evidence": {{"quote": "UNDANG-UNDANG REPUBLIK INDONESIA NOMOR 12 TAHUN 2011"}}
}}

### 2. Document Structure
MATCH: "BAB I KETENTUAN UMUM"
EXTRACT:
{{
  "subject": {{"text": "UU 12/2011", "type": "Act"}},
  "predicate": "has_content",
  "object": {{"text": "BAB I", "type": "Chapter"}},
  "evidence": {{"quote": "BAB I KETENTUAN UMUM"}}
}}

### 3. Definitions
MATCH: "Pembentukan Peraturan Perundang-undangan adalah pembuatan Peraturan Perundang-undangan yang mencakup tahapan perencanaan, penyusunan, pembahasan, pengesahan atau penetapan, dan pengundangan."
EXTRACT:
{{
  "subject": {{"text": "Pembentukan Peraturan Perundang-undangan", "type": "Concept"}},
  "predicate": "mendefinisikan",
  "object": {{"text": "pembuatan Peraturan Perundang-undangan yang mencakup tahapan perencanaan, penyusunan, pembahasan, pengesahan atau penetapan, dan pengundangan.", "type": "Concept"}},
  "evidence": {{"quote": "Pembentukan Peraturan Perundang-undangan adalah pembuatan Peraturan Perundang-undangan yang mencakup tahapan perencanaan, penyusunan, pembahasan, pengesahan atau penetapan, dan pengundangan."}}
}}

### 4. Legal Norms
MATCH: "Menteri harus menetapkan peraturan pelaksanaan undang-undang ini."
EXTRACT:
{{
  "subject": {{"text": "Norm about ministerial obligation", "type": "Norm"}},
  "predicate": "has_act",
  "object": {{"text": "menetapkan peraturan pelaksanaan", "type": "RuleAct"}},
  "evidence": {{"quote": "Menteri harus menetapkan peraturan pelaksanaan undang-undang ini."}}
}}

### 5. References
MATCH: "Materi Muatan sebagaimana dimaksud pada Pasal 5"
EXTRACT:
{{
  "subject": {{"text": "Materi Muatan", "type": "Concept"}},
  "predicate": "refers_to",
  "object": {{"text": "Pasal 5", "type": "Article"}},
  "evidence": {{"quote": "Materi Muatan sebagaimana dimaksud pada Pasal 5"}}
}}

### 6. Amendments
Context: we are currently inside a chunk from Undang-unddang nomor 12 tahun 2011
MATCH: "Pasal 15 Undang-Undang Nomor 10 Tahun 2004 diubah sehingga berbunyi sebagai berikut:"
EXTRACT:
{{
  "subject": {{"text": "UU 12/2011", "type": "Act"}},
  "predicate": "amends",
  "object": {{"text": "UU 10/2004", "type": "Act"}},
  "evidence": {{"quote": "Pasal 15 Undang-Undang Nomor 10 Tahun 2004 diubah sehingga berbunyi sebagai berikut: "}}
}}

### 7. Conditions
MATCH: "Jika diperlukan, Menteri dapat menetapkan Peraturan"
EXTRACT:
{{
  "subject": {{"text": "Norm about ministerial permission", "type": "Norm"}},
  "predicate": "has_modality",
  "object": {{"text": "dapat", "type": "Concept"}},
  "evidence": {{"quote": "Jika diperlukan, Menteri dapat menetapkan Peraturan"}}
}}

### 8. Compound Expressions
MATCH: "perencanaan, penyusunan, pembahasan, pengesahan atau penetapan, dan pengundangan"
EXTRACT:
{{
  "subject": {{"text": "stages of law formation", "type": "CompoundExpression"}},
  "predicate": "has_element",
  "object": {{"text": "perencanaan", "type": "Concept"}},
  "evidence": {{"quote": "perencanaan, penyusunan, pembahasan, pengesahan atau penetapan, dan pengundangan"}}
}}

OUTPUT REQUIREMENTS:
- Extract triples as JSON. Every triple must include:
  - subject: {{text, type}}
  - predicate: relationship type (lowercase, snake_case)
  - object: {{text, type}}
  - evidence: {{quote}}
- Limit evidence.quote to at most {QUOTE_MAX_WORDS} words. The quote may include one or multiple (possibly non-consecutive) sentences from the chunk.
- Use only the provided chunk text; do not invent information.
- Normalize predicates to lowercase Indonesian phrases in snake_case.
- Use specific node types rather than generic ones.

EXTRACTION PRIORITIES:
1. Document metadata (number, year, creator)
2. Document structure (chapters, articles, sections)
3. Inter-document relationships (amends, repeals)
4. Legal norms (obligations, permissions, prohibitions)
5. Semantic relationships (subject-action-object)
6. Concept definitions and references
"""

def clamp_words(text: Optional[str], max_words: int) -> str:
    """Clamp text to at most max_words words, appending '...' if truncated."""
    if not text:
        return ""
    words = re.findall(r"\S+", text.strip())
    if len(words) <= max_words:
        return text.strip()
    return " ".join(words[:max_words]).strip() + "..."

def build_single_prompt(meta: Dict[str, Any], chunk_text: str) -> str:
    return f"""
{SYSTEM_HINT}

Chunk metadata:
- document_id: {meta.get('document_id')}
- chunk_id: {meta.get('chunk_id')}
- uu_number: {meta.get('uu_number')}
- pages: {meta.get('pages')}

Extract knowledge graph triples using the LexID ontology described above.
Return JSON with a top-level key "triples" (array). Each triple must be:
{{
  "subject": {{"text": "string", "type": "one of the allowed node types described above"}},
  "predicate": "string (lower_snake_case)",
  "object":  {{"text": "string", "type": "one of the allowed node types described above"}},
  "evidence": {{"quote": "string (around {QUOTE_MAX_WORDS} words)"}},
  "confidence": 0.0
}}

Text:
\"\"\"{chunk_text}\"\"\"
"""

def build_batch_prompt(items: List[Dict[str, Any]]) -> str:
    intro = f"{SYSTEM_HINT}\n\n" + f"""
You will receive several chunks. Return JSON with a top-level key 'results' (array).
Each element must be an object with the following shape:
{{
  "chunk_id": "<chunk id string>",
  "triples": [
    {{
      "subject": {{"text": "string", "type": "one of the allowed node types described above"}},
      "predicate": "string (lower_snake_case)",
      "object":  {{"text": "string", "type": "one of the allowed node types described above"}},
      "evidence": {{"quote": "string (around {QUOTE_MAX_WORDS} words)"}},
      "confidence": 0.0
    }}
  ]
}}
Notes:
- 'triples' must follow the LexID ontology and constraints above.
- Use only the provided chunk text; do not invent information.
- Limit evidence.quote to at most {QUOTE_MAX_WORDS} words.
"""

    lines = []
    for it in items:
        m = it["meta"]
        lines.append(
            f"- chunk_id: {m.get('chunk_id')} | document_id: {m.get('document_id')} | uu_number: {m.get('uu_number')} | pages: {m.get('pages')}\n"
            f"TEXT:\n\"\"\"{it['text']}\"\"\""
        )
    return intro + "\n\nChunks:\n" + "\n\n".join(lines)

# ----------------- JSON Schemas -----------------
TRIPLE_SCHEMA = {
  "type": "object",
  "properties": {
    "triples": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "subject": {
            "type": "object",
            "properties": {
              "text": {"type": "string"},
              "type": {"type": "string", "enum": ALLOWED_NODE_TYPES}
            },
            "required": ["text", "type"]
          },
          "predicate": {"type": "string"},
          "object": {
            "type": "object",
            "properties": {
              "text": {"type": "string"},
              "type": {"type": "string", "enum": ALLOWED_NODE_TYPES}
            },
            "required": ["text", "type"]
          },
          "evidence": {
            "type": "object",
            "properties": {
              "quote": {"type": "string"}
            },
            "required": ["quote"]
          },
          "confidence": {"type": "number"}
        },
        "required": ["subject", "predicate", "object", "evidence"]
      }
    }
  },
  "required": ["triples"]
}

# Batch variant: results = [{chunk_id, triples:[... as above ...]}]
BATCH_TRIPLE_SCHEMA = {
  "type": "object",
  "properties": {
    "results": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "chunk_id": {"type": "string"},
          "triples": TRIPLE_SCHEMA["properties"]["triples"]
        },
        "required": ["chunk_id", "triples"]
      }
    }
  },
  "required": ["results"]
}

# ----------------- Attempt ID generator -----------------
_attempt_lock = threading.Lock()
_attempt_counter = 0

def next_attempt_id() -> int:
    global _attempt_counter
    with _attempt_lock:
        aid = _attempt_counter
        _attempt_counter += 1
        return aid

# ----------------- Rate Limiter (LLM) -----------------
class RateLimitError(Exception):
    def __init__(self, message: str, attempt_id: int, category: str = "unknown", reason: str = "", json_path: Optional[str] = None):
        super().__init__(message)
        self.attempt_id = attempt_id
        self.category = category  # 'rpm', 'quota', or 'unknown'
        self.reason = reason
        self.json_path = json_path

class JsonParseError(Exception):
    def __init__(self, message: str, attempt_id: int, json_path: Optional[str] = None):
        super().__init__(message)
        self.attempt_id = attempt_id
        self.json_path = json_path

class LlmCallError(RuntimeError):
    def __init__(self, message: str, attempt_id: int, json_path: Optional[str] = None):
        super().__init__(message)
        self.attempt_id = attempt_id
        self.json_path = json_path

class RateLimiter:
    def __init__(self, max_calls: int, period_sec: float):
        self.max_calls = max_calls
        self.period = period_sec
        self.calls = deque()
        self.lock = threading.Lock()

    def acquire(self, attempt_id: Optional[int] = None):
        with self.lock:
            now = time.time()
            while self.calls and (now - self.calls[0]) >= self.period:
                self.calls.popleft()
            wait = 0.0
            if len(self.calls) >= self.max_calls:
                wait = self.period - (now - self.calls[0])
        if wait > 0:
            log_info(f"[RateLimiter] Sleeping {wait:.2f}s to respect client-side RPM limit", attempt_id=attempt_id)
            time.sleep(wait)
        with self.lock:
            self.calls.append(time.time())

LLM_RATE_LIMITER = RateLimiter(max_calls=LLM_MAX_CALLS_PER_MIN, period_sec=60.0)

# ----------------- Global API Budget -----------------
class ApiBudget:
    def __init__(self, total: int, enforce: bool, count_embeddings: bool):
        self.total = total
        self.enforce = enforce
        self.count_embeddings = count_embeddings
        self.used = 0
        self.lock = threading.Lock()
        self.resource_usage = {"llm": 0, "embed": 0}

    def _should_count(self, kind: str) -> bool:
        if kind == "embed" and not self.count_embeddings:
            return False
        return True

    def will_allow(self, kind: str, n: int = 1) -> bool:
        if not self.enforce or not self._should_count(kind):
            return True
        with self.lock:
            return (self.used + n) <= self.total

    def register(self, kind: str, n: int = 1):
        if not self.enforce or not self._should_count(kind):
            return
        with self.lock:
            if (self.used + n) > self.total:
                raise RuntimeError(f"API budget exceeded: used={self.used}, trying={n}, total={self.total}")
            self.used += n
            self.resource_usage[kind] = self.resource_usage.get(kind, 0) + n

API_BUDGET = ApiBudget(total=API_BUDGET_TOTAL, enforce=ENFORCE_API_BUDGET, count_embeddings=COUNT_EMBEDDINGS_IN_BUDGET)

# ----------------- Helpers -----------------
def _slug(s: str) -> str:
    return " ".join((s or "").strip().split()).lower()

def sanitize_filename_component(s: str) -> str:
    s = (s or "").strip().lower().replace(" ", "-")
    s = re.sub(r"[^a-z0-9_\-+]", "", s)
    s = re.sub(r"-{2,}", "-", s)
    return s or "unknown"

def normalize_entity_key(text: str, etype: str, uu_number: Optional[str] = None) -> str:
    """Generate a canonical key ID based on type and text (LexID-friendly)."""
    text_slug = re.sub(r'[^\w\s-]', '', (text or "").strip().lower())
    text_slug = re.sub(r'\s+', '-', text_slug)

    # Document nodes (prefer uu_number if present)
    if etype in ("LegalDocument",) or etype in LEGAL_DOCUMENT_TYPES:
        if uu_number:
            return f"{etype}::uu-{uu_number.replace('/', '-')}"
        return f"{etype}::{text_slug}"

    # Structural nodes (Article, Section, etc.)
    if etype in LEGAL_CONTENT_TYPES or etype == "LegalDocumentContent":
        if uu_number:
            return f"{etype}::uu-{uu_number.replace('/', '-')}::{text_slug}"
        return f"{etype}::{text_slug}"

    # Semantic nodes
    if etype in RULE_EXPRESSION_TYPES or etype == "RuleExpression":
        return f"{etype}::{text_slug}"

    # Amendments
    if etype in LAW_AMENDMENT_TYPES or etype == "LawAmendment":
        prefix = f"uu-{uu_number.replace('/', '-')}-" if uu_number else ""
        return f"{etype}::{prefix}{text_slug}"

    # Other entity types
    return f"{etype}::{text_slug}"

def deterministic_triple_uid(
    subject_key: str, predicate: str, object_key: str, doc_id: Optional[str], chunk_id: Optional[str]
) -> str:
    """Generate a deterministic unique ID for a triple (predicate snake_case), keyed per chunk and document."""
    h = hashlib.sha256()
    payload = "|".join([
        subject_key,
        (predicate or "").strip().lower().replace(" ", "_"),
        object_key,
        doc_id or "",
        chunk_id or ""
    ])
    h.update(payload.encode("utf-8"))
    return h.hexdigest()

def estimate_tokens_for_text(text: str) -> int:
    # Rough heuristic: ~3.5 chars/token to be conservative
    return max(1, int(len(text) / 3.5))

def aggregate_pages(items_metas: List[Dict[str, Any]]) -> List[int]:
    pages: set[int] = set()
    for m in items_metas:
        p = m.get("pages")
        if isinstance(p, (list, tuple, set)):
            for x in p:
                if isinstance(x, int):
                    pages.add(x)
                elif isinstance(x, str):
                    for tok in re.split(r"[,\s\-]+", x):
                        if tok.isdigit():
                            pages.add(int(tok))
        elif isinstance(p, int):
            pages.add(p)
        elif isinstance(p, str):
            for tok in re.split(r"[,\s\-]+", p):
                if tok.isdigit():
                    pages.add(int(tok))
    return sorted(pages)

def build_json_output_filename(context: Dict[str, Any]) -> Path:
    """
    Build filename. For multi-UU batches, include every UU and its page range appended with '+'.
    Example: uu-12-2011__p-10-13+uu-10-2004__p-5-6__batch__5__20250828-...json
    """
    kind = context.get("kind", "batch")
    items_metas: List[Dict[str, Any]] = context.get("items_metas", [])
    items_count = context.get("items_count", len(items_metas) or "?")

    # Aggregate pages per UU
    by_uu: Dict[str, List[int]] = {}
    for m in items_metas:
        uu = m.get("uu_number") or "unknown"
        by_uu.setdefault(uu, [])
        # Aggregate pages from this meta only
        ps = m.get("pages")
        page_set: set[int] = set()
        if isinstance(ps, (list, tuple, set)):
            for x in ps:
                if isinstance(x, int):
                    page_set.add(x)
                elif isinstance(x, str):
                    for tok in re.split(r"[,\s\-]+", x):
                        if tok.isdigit():
                            page_set.add(int(tok))
        elif isinstance(ps, int):
            page_set.add(ps)
        elif isinstance(ps, str):
            for tok in re.split(r"[,\s\-]+", ps):
                if tok.isdigit():
                    page_set.add(int(tok))
        by_uu[uu].extend(sorted(page_set))

    def pages_label(pages: List[int]) -> str:
        if not pages:
            return "p-unknown"
        sp = sorted(set(pages))
        if len(sp) == 1:
            return f"p-{sp[0]}"
        return f"p-{sp[0]}-{sp[-1]}"

    if not by_uu:
        uu_lab = "uu-unknown"
    else:
        segs = []
        for uu, pages in sorted(by_uu.items(), key=lambda kv: sanitize_filename_component(str(kv[0]))):
            uu_norm = sanitize_filename_component(str(uu))
            segs.append(f"uu-{uu_norm}__{pages_label(pages)}")
        uu_lab = "+".join(segs)

    chunk_ids = [m.get("chunk_id") or "" for m in items_metas]
    short = hashlib.sha1("|".join(sorted(chunk_ids)).encode("utf-8")).hexdigest()[:8]
    ts = time.strftime("%Y%m%d-%H%M%S")

    fname = f"{uu_lab}__{kind}__{items_count}__{ts}_{short}.json"
    return (JSON_OUTPUT_DIR / fname)

def save_llm_json_output(data: Dict[str, Any], context: Dict[str, Any], attempt_id: int) -> Path:
    """
    Save successful LLM JSON output and return the full path.
    """
    try:
        out_path = build_json_output_filename(context)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        log_info(f"[LLM] Saved JSON output: {out_path}", attempt_id=attempt_id)
        return out_path
    except Exception as e:
        log_warn(f"[LLM] Warning: failed to save JSON output: {e}", attempt_id=attempt_id)
        raise

def save_error_json_output(error_stub: Dict[str, Any], context: Dict[str, Any], attempt_id: int, err_type: str) -> Path:
    """
    Save an error JSON output with a prefixed filename: "[ERROR errortype] <normal-filename>".
    """
    base_path = build_json_output_filename(context)
    err_type_sanitized = sanitize_filename_component(err_type)
    prefixed_name = f"[ERROR {err_type_sanitized}] {base_path.name}"
    out_path = base_path.with_name(prefixed_name)
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(error_stub, f, ensure_ascii=False, indent=2)
        log_warn(f"[LLM] Saved ERROR JSON output ({err_type}): {out_path}", attempt_id=attempt_id)
        return out_path
    except Exception as e:
        log_warn(f"[LLM] Warning: failed to save ERROR JSON output: {e}", attempt_id=attempt_id)
        return out_path

def split_text_in_two(text: str) -> Tuple[str, str]:
    """
    Split text into two halves near the middle, preferring paragraph/sentence/space boundaries.
    """
    n = len(text)
    if n <= 1:
        return text, ""
    mid = n // 2
    window = 300
    start = max(0, mid - window)
    end = min(n, mid + window)
    seps = ["\n\n", "\n", ". ", " "]
    best_idx = None
    for sep in seps:
        li = text.rfind(sep, 0, mid)
        ri = text.find(sep, mid, end)
        candidates = []
        if li != -1:
            candidates.append((abs(mid - (li + len(sep)//2)), li + len(sep)))
        if ri != -1:
            candidates.append((abs(ri - mid), ri + len(sep)))
        if candidates:
            candidates.sort(key=lambda x: x[0])
            best_idx = candidates[0][1]
            break
    if best_idx is None:
        best_idx = mid
    left = text[:best_idx].strip()
    right = text[best_idx:].strip()
    return left, right

# ----------------- Predicate and label helpers -----------------
def normalize_predicate(pred: Optional[str]) -> str:
    p = (pred or "").strip().toLower() if False else (pred or "").strip().lower()  # keep original behavior; safety
    p = p.replace("-", "_").replace(" ", "_")
    p = re.sub(r"[^a-z0-9_]", "", p)
    return p or "rel"

def rel_type_from_pred(pred: str) -> str:
    """
    Turn a normalized predicate (lower_snake_case) into a safe relationship label (UPPER_SNAKE_CASE).
    Does not enforce the LEXID_RELATIONSHIPS set (to avoid dropping data),
    but sanitizes to [A-Z0-9_]+ and falls back to REL if empty.
    """
    rel = (pred or "").strip().lower()
    rel = rel.replace("-", "_").replace(" ", "_")
    rel = re.sub(r"[^a-z0-9_]", "", rel)
    rel = rel.upper() or "REL"
    return rel

def get_base_label(specific_type: str) -> str:
    """Map specific types to their base label."""
    if specific_type in LEGAL_DOCUMENT_TYPES or specific_type == "LegalDocument":
        return "LegalDocument"
    if specific_type in LEGAL_CONTENT_TYPES or specific_type == "LegalDocumentContent":
        return "LegalDocumentContent"
    if specific_type in RULE_EXPRESSION_TYPES or specific_type == "RuleExpression":
        return "RuleExpression"
    if specific_type in LAW_AMENDMENT_TYPES or specific_type == "LawAmendment":
        return "LawAmendment"
    return specific_type

def labels_for_type(specific_type: str) -> str:
    """Return a Cypher label string e.g., 'Act:LegalDocument' (dedup base if same)."""
    base = get_base_label(specific_type)
    if base == specific_type:
        return specific_type
    return f"{specific_type}:{base}"

# ----------------- LLM call wrapper (logs every attempt) -----------------
def is_rate_limit_error(e: Exception) -> bool:
    msg = str(e).lower()
    if "429" in msg or "rate limit" in msg or "too many requests" in msg or "quota" in msg or "resourceexhausted" in msg or "resource exhausted" in msg:
        return True
    code = getattr(e, "code", None)
    if code == 429:
        return True
    status = getattr(e, "status", None) or getattr(e, "reason", None)
    if isinstance(status, str) and ("exhausted" in status.lower() or "429" in status):
        return True
    return False

def classify_rate_limit_error(e: Exception) -> Tuple[str, str]:
    """
    Returns (category, reason) where category in {'rpm','quota','unknown'}.
    Heuristics based on message text from the Gemini API.
    """
    msg = (str(e) or "").lower()
    if "quota" in msg or "exceeded your current quota" in msg or "billing" in msg:
        return "quota", "quota exceeded"
    if "rate" in msg or "too many requests" in msg or "resource exhausted" in msg or "exhausted" in msg:
        return "rpm", "request rate exceeded"
    return "unknown", "unclassified 429"

def run_llm_json(prompt: str, schema: dict, context: Dict[str, Any]) -> Tuple[Dict[str, Any], float, int, Optional[str]]:
    """
    Makes a single LLM call with JSON schema and returns parsed JSON, duration, attempt_id, and output path.
    On failures, writes an ERROR JSON file and raises a typed exception carrying attempt_id and json_path.
    Budget note: budget is charged ONLY after successful call and JSON parsing.
    Also saves the successful JSON output to a file (one file per batch).
    """
    if not API_BUDGET.will_allow("llm", 1):
        raise RuntimeError("API budget would be exceeded by another LLM call; stopping extraction.")

    attempt_id = next_attempt_id()

    LLM_RATE_LIMITER.acquire(attempt_id=attempt_id)

    items_metas: List[Dict[str, Any]] = context.get("items_metas", [])
    chunk_ids = [m.get("chunk_id") for m in items_metas if m.get("chunk_id")]
    uu_numbers = list(sorted({m.get("uu_number") for m in items_metas if m.get("uu_number")}))
    doc_ids = list(sorted({m.get("document_id") for m in items_metas if m.get("document_id")}))
    try:
        kind = context.get("kind", "unknown")
        items_count = context.get("items_count", "?")
        token_est = estimate_tokens_for_text(prompt)
        log_info(
            f"[LLM] Attempt: model={GEN_MODEL} | kind={kind} | items={items_count} | est_tokens~{token_est} | "
            f"chunk_ids={chunk_ids} | docs={doc_ids} | uu={uu_numbers}",
            attempt_id=attempt_id
        )
    except Exception:
        log_info(f"[LLM] Attempt: model={GEN_MODEL}", attempt_id=attempt_id)

    cfg = GenerationConfig(
        temperature=0,
        response_mime_type="application/json",
        response_schema=schema,
    )

    start = time.time()
    try:
        resp = gen_model.generate_content(prompt, generation_config=cfg)
    except Exception as e:
        dur = time.time() - start
        err_cat, err_reason = ("unknown", "unclassified")
        if is_rate_limit_error(e):
            err_cat, err_reason = classify_rate_limit_error(e)
            err_type = f"ratelimit-{err_cat}"
        else:
            err_type = "llmcall"
        error_stub = {
            "error": {"type": err_type, "category": err_cat, "reason": err_reason, "message": str(e)},
            "attempt_id": attempt_id,
            "context": {"kind": context.get("kind"), "items_count": context.get("items_count"), "chunk_ids": chunk_ids, "uu_numbers": uu_numbers, "document_ids": doc_ids}
        }
        error_path = save_error_json_output(error_stub, context, attempt_id=attempt_id, err_type=err_type)
        if is_rate_limit_error(e):
            cat, reason = classify_rate_limit_error(e)
            raise RateLimitError(f"LLM rate limit error ({cat}): {e}", attempt_id=attempt_id, category=cat, reason=reason, json_path=str(error_path)) from e
        raise LlmCallError(f"LLM call failed: {e}", attempt_id=attempt_id, json_path=str(error_path)) from e

    dur = time.time() - start

    raw = None
    try:
        raw = resp.text
        data = json.loads(raw)
    except Exception:
        try:
            raw = resp.candidates[0].content.parts[0].text
            data = json.loads(raw)
        except Exception as e:
            preview = raw[:500] if isinstance(raw, str) else "None"
            error_stub = {
                "error": {"type": "jsonparseerror", "message": str(e), "raw_preview": preview},
                "attempt_id": attempt_id,
                "context": {"kind": context.get("kind"), "items_count": context.get("items_count"), "chunk_ids": chunk_ids, "uu_numbers": uu_numbers, "document_ids": doc_ids}
            }
            error_path = save_error_json_output(error_stub, context, attempt_id=attempt_id, err_type="jsonparseerror")
            raise JsonParseError(f"Failed to parse model JSON: {e}; raw_preview={preview[:200]}", attempt_id=attempt_id, json_path=str(error_path)) from e

    API_BUDGET.register("llm", 1)

    out_path = save_llm_json_output(data, context, attempt_id=attempt_id)

    return data, dur, attempt_id, str(out_path)

# ----------------- LLM Extraction (single attempt) -----------------
def extract_triples_from_chunk(chunk_text: str, meta: Dict[str, Any], prompt_override: Optional[str] = None) -> Tuple[List[Dict[str, Any]], float, int, Optional[str]]:
    prompt = prompt_override or build_single_prompt(meta, chunk_text)
    data, gemini_duration, attempt_id, json_path = run_llm_json(
        prompt, TRIPLE_SCHEMA,
        context={"kind": "single", "items_count": 1, "chunk_id": meta.get("chunk_id"), "items_metas": [meta]}
    )

    uu_number = meta.get("uu_number")
    triples: List[Dict[str, Any]] = []
    for t in data.get("triples", []):
        subj = t["subject"]; obj = t["object"]; pred_raw = t.get("predicate")
        pred = normalize_predicate(pred_raw)
        s_type = subj.get("type") or "Concept"
        o_type = obj.get("type") or "Concept"
        s_key = normalize_entity_key(subj["text"], s_type, uu_number)
        o_key = normalize_entity_key(obj["text"],  o_type,  uu_number)

        ev = t.get("evidence") or {}
        quote_raw = ev.get("quote") or ""
        quote = clamp_words(quote_raw, QUOTE_MAX_WORDS)

        triple_uid = deterministic_triple_uid(
            s_key, pred, o_key, meta.get("document_id"), meta.get("chunk_id")
        )

        triples.append({
            "triple_uid": triple_uid,
            "subject": {"text": subj["text"], "type": s_type, "key": s_key},
            "predicate": pred,
            "object": {"text": obj["text"], "type": o_type, "key": o_key},
            "evidence": {
                "quote": quote
            },
            "confidence": float(t.get("confidence", 0.0)),
            "provenance": {
                "document_id": meta.get("document_id"),
                "chunk_id": meta.get("chunk_id"),
                "uu_number": uu_number,
                "pages": meta.get("pages"),
            }
        })
    return triples, gemini_duration, attempt_id, json_path

def extract_triples_from_chunks_batch(items: List[Dict[str, Any]], prompt: str) -> Tuple[Dict[str, List[Dict[str, Any]]], float, int, Optional[str]]:
    metas = [it["meta"] for it in items]
    data, gemini_duration, attempt_id, json_path = run_llm_json(
        prompt, BATCH_TRIPLE_SCHEMA,   # <-- use batch schema here
        context={"kind": "batch", "items_count": len(items), "items_metas": metas}
    )

    results_map: Dict[str, List[Dict[str, Any]]] = {}
    meta_map = {it["meta"]["chunk_id"]: it["meta"] for it in items}

    for res in data.get("results", []):
        cid = res.get("chunk_id")
        if not cid or cid not in meta_map:
            continue
        meta = meta_map[cid]
        uu_number = meta.get("uu_number")
        triples_for_chunk: List[Dict[str, Any]] = []
        for t in res.get("triples", []):
            subj = t["subject"]; obj = t["object"]; pred_raw = t.get("predicate")
            pred = normalize_predicate(pred_raw)
            s_type = subj.get("type") or "Concept"
            o_type = obj.get("type") or "Concept"
            s_key = normalize_entity_key(subj["text"], s_type, uu_number)
            o_key = normalize_entity_key(obj["text"],  o_type,  uu_number)

            ev = t.get("evidence") or {}
            quote_raw = ev.get("quote") or ""
            quote = clamp_words(quote_raw, QUOTE_MAX_WORDS)

            triple_uid = deterministic_triple_uid(
                s_key, pred, o_key, meta.get("document_id"), meta.get("chunk_id")
            )

            triples_for_chunk.append({
                "triple_uid": triple_uid,
                "subject": {"text": subj["text"], "type": s_type, "key": s_key},
                "predicate": pred,
                "object": {"text": obj["text"], "type": o_type, "key": o_key},
                "evidence": {
                    "quote": quote
                },
                "confidence": float(t.get("confidence", 0.0)),
                "provenance": {
                    "document_id": meta.get("document_id"),
                    "chunk_id": meta.get("chunk_id"),
                    "uu_number": uu_number,
                    "pages": meta.get("pages"),
                }
            })
        results_map[cid] = triples_for_chunk

    return results_map, gemini_duration, attempt_id, json_path

# ----------------- Embeddings -----------------
def embed_text(text: str) -> Tuple[List[float], float]:
    if not API_BUDGET.will_allow("embed", 1):
        raise RuntimeError("API budget would be exceeded by another embedding call; stopping embedding.")
    start = time.time()
    try:
        res = genai.embed_content(model=EMBED_MODEL, content=text)
    except Exception as e:
        msg = str(e)
        if "EmbedContentRequest.model" in msg and "unexpected model name format" in msg.lower():
            raise RuntimeError(f"Embedding model name invalid: {EMBED_MODEL}. Use 'models/text-embedding-004' or set EMBED_MODEL accordingly.") from e
        raise
    dur = time.time() - start

    vec: Optional[List[float]] = None

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
        raise RuntimeError("Unexpected embedding response shape")

    API_BUDGET.register("embed", 1)

    return vec, dur

def node_embedding_text(name: str, etype: str) -> str:
    return f"{(name or '').strip()} | {etype}"

def triple_embedding_text(t: Dict[str, Any]) -> str:
    # NEW: Only "subject predicate object" (no brackets, no uu, no article ref)
    s = t["subject"]["text"]
    p = t["predicate"]
    o = t["object"]["text"]
    return f"{s} {p} {o}"

# ----------------- Neo4j Storage -----------------
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

_entity_emb_cache: dict[str, List[float]] = {}
_triple_emb_cache: dict[str, List[float]] = {}
_entity_emb_cache_lock = threading.Lock()
_triple_emb_cache_lock = threading.Lock()

def _get_entity_emb(name: str, etype: str, key: str) -> Tuple[List[float], float]:
    with _entity_emb_cache_lock:
        cached = _entity_emb_cache.get(key)
    if cached is not None:
        return cached, 0.0
    vec, dur = embed_text(node_embedding_text(name, etype))
    with _entity_emb_cache_lock:
        _entity_emb_cache[key] = vec
    return vec, dur

def _get_triple_emb(triple_uid: str, t: Dict[str, Any]) -> Tuple[List[float], float]:
    with _triple_emb_cache_lock:
        cached = _triple_emb_cache.get(triple_uid)
    if cached is not None:
        return cached, 0.0
    vec, dur = embed_text(triple_embedding_text(t))
    with _triple_emb_cache_lock:
        _triple_emb_cache[triple_uid] = vec
    return vec, dur

def upsert_triple(tx, t: Dict[str, Any], s_emb: List[float], o_emb: List[float], tr_emb: List[float]) -> float:
    s, o = t["subject"], t["object"]
    s_name, s_type, s_key = s["text"], s["type"], s["key"]
    o_name, o_type, o_key = o["text"], o["type"], o["key"]
    pred = t["predicate"]  # normalized lower_snake_case
    triple_uid = t["triple_uid"]

    prov = t["provenance"]
    ev = t.get("evidence") or {}
    confidence = float(t.get("confidence", 0.0))

    # Label expansion (e.g., Act:LegalDocument)
    s_label_str = labels_for_type(s_type)
    o_label_str = labels_for_type(o_type)

    # Relationship label (UPPER_SNAKE_CASE, sanitized)
    rel_label = rel_type_from_pred(pred)

    start = time.time()
    cypher = f"""
    MERGE (s:{s_label_str} {{key:$s_key}})
      ON CREATE SET s.name=$s_name, s.type=$s_type, s.createdAt=timestamp()
    SET s.embedding=$s_emb

    MERGE (o:{o_label_str} {{key:$o_key}})
      ON CREATE SET o.name=$o_name, o.type=$o_type, o.createdAt=timestamp()
    SET o.embedding=$o_emb

    MERGE (tr:Triple {{triple_uid:$triple_uid}})
      ON CREATE SET tr.createdAt=timestamp()
    SET tr.predicate=$pred,
        tr.embedding=$tr_emb,
        tr.document_id=$doc_id,
        tr.chunk_id=$chunk_id,
        tr.uu_number=$uu_number,
        tr.pages=$pages,
        tr.evidence_quote=$evidence_quote,
        tr.confidence=$confidence

    MERGE (tr)-[:SUBJECT]->(s)
    MERGE (tr)-[:OBJECT]->(o)

    MERGE (s)-[r:{rel_label} {{triple_uid:$triple_uid}}]->(o)
    SET r.predicate=$pred, r.chunk_id=$chunk_id, r.document_id=$doc_id
    """
    tx.run(
        cypher,
        s_key=s_key, s_name=s_name, s_type=s_type, s_emb=s_emb,
        o_key=o_key, o_name=o_name, o_type=o_type, o_emb=o_emb,
        triple_uid=triple_uid, pred=pred, tr_emb=tr_emb,
        doc_id=prov.get("document_id"),
        chunk_id=prov.get("chunk_id"),
        uu_number=prov.get("uu_number"),
        pages=prov.get("pages", []),
        evidence_quote=ev.get("quote"),
        confidence=confidence,
    )
    return time.time() - start

def write_triples_for_chunk(triples: List[Dict[str, Any]]) -> Tuple[int, float, float, Dict[str, Any]]:
    """
    Return: written_count, total_embedding_duration, total_neo4j_duration, embed_stats
    embed_stats: {
      'entity_new_calls': int, 'entity_cached': int, 'entity_new_dur': float,
      'triple_new_calls': int, 'triple_cached': int, 'triple_new_dur': float
    }
    """
    if not triples:
        return 0, 0.0, 0.0, {'entity_new_calls':0,'entity_cached':0,'entity_new_dur':0.0,'triple_new_calls':0,'triple_cached':0,'triple_new_dur':0.0}

    total_emb_dur = 0.0
    total_neo4j_dur = 0.0
    written = 0

    embed_stats = {'entity_new_calls':0,'entity_cached':0,'entity_new_dur':0.0,'triple_new_calls':0,'triple_cached':0,'triple_new_dur':0.0}

    with driver.session() as session:
        for t in triples:
            s = t["subject"]; o = t["object"]
            s_emb, d1 = _get_entity_emb(s["text"], s["type"], s["key"])
            if d1 > 0:
                embed_stats['entity_new_calls'] += 1
                embed_stats['entity_new_dur'] += d1
            else:
                embed_stats['entity_cached'] += 1

            o_emb, d2 = _get_entity_emb(o["text"], o["type"], o["key"])
            if d2 > 0:
                embed_stats['entity_new_calls'] += 1
                embed_stats['entity_new_dur'] += d2
            else:
                embed_stats['entity_cached'] += 1

            tr_emb, d3 = _get_triple_emb(t["triple_uid"], t)
            if d3 > 0:
                embed_stats['triple_new_calls'] += 1
                embed_stats['triple_new_dur'] += d3
            else:
                embed_stats['triple_cached'] += 1

            total_emb_dur += (d1 + d2 + d3)

            neo4j_dur = session.execute_write(upsert_triple, t, s_emb, o_emb, tr_emb)
            total_neo4j_dur += neo4j_dur
            written += 1

    return written, total_emb_dur, total_neo4j_dur, embed_stats

# ----------------- Folder utilities -----------------
def list_pickles(dir_path: Path) -> List[Path]:
    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {dir_path}")
    return sorted([p for p in dir_path.glob("*.pkl") if p.name not in SKIP_FILES])

def load_chunks_from_file(pkl_path: Path) -> List[Any]:
    with open(pkl_path, "rb") as f:
        chunks = pickle.load(f)
    if not isinstance(chunks, list):
        raise ValueError(f"Unexpected pickle content in {pkl_path}")
    log_info(f"  - {pkl_path.name}: {len(chunks)} chunks")
    return chunks

# ----------------- Greedy and budget-aware batch builders -----------------
def build_greedy_batch(use_items: List[Dict[str, Any]], start_idx: int) -> Tuple[List[Dict[str, Any]], int, int]:
    items_batch: List[Dict[str, Any]] = []
    idx = start_idx
    est_tokens = 0

    while idx < len(use_items) and len(items_batch) < PRACTICAL_MAX_ITEMS_PER_BATCH:
        candidate = use_items[idx]
        tentative = items_batch + [candidate]
        prompt = build_batch_prompt(tentative) if len(tentative) > 1 else build_single_prompt(candidate["meta"], candidate["text"])
        tokens = estimate_tokens_for_text(prompt)
        if tokens <= PROMPT_TOKEN_LIMIT or len(items_batch) == 0:
            items_batch = tentative
            est_tokens = tokens
            idx += 1
        else:
            break

    return items_batch, idx, est_tokens

def pack_batches(use_items: List[Dict[str, Any]]) -> List[Tuple[List[Dict[str, Any]], int]]:
    batches: List[Tuple[List[Dict[str, Any]], int]] = []
    i = 0
    while i < len(use_items):
        items_batch, next_i, est_tokens = build_greedy_batch(use_items, i)
        if not items_batch:
            i += 1
            continue
        batches.append((items_batch, est_tokens))
        i = next_i
    return batches

def try_merge_batches(b1: Tuple[List[Dict[str, Any]], int], b2: Tuple[List[Dict[str, Any]], int]) -> Optional[Tuple[List[Dict[str, Any]], int]]:
    items1, _ = b1
    items2, _ = b2
    merged = items1 + items2
    if len(merged) > PRACTICAL_MAX_ITEMS_PER_BATCH:
        return None
    merged_prompt = build_batch_prompt(merged) if len(merged) > 1 else build_single_prompt(merged[0]["meta"], merged[0]["text"])
    tokens = estimate_tokens_for_text(merged_prompt)
    if tokens <= PROMPT_TOKEN_LIMIT:
        return (merged, tokens)
    return None

def pack_batches_with_cap(use_items: List[Dict[str, Any]], max_batches_allowed: Optional[int]) -> Tuple[List[Tuple[List[Dict[str, Any]], int]], int]:
    """
    Build batches greedily and then enforce a hard cap on number of batches.
    If we exceed the cap, attempt to merge adjacent batches within token limits.
    If still over the cap, truncate extra batches (defer remaining chunks).

    Returns:
      (batches_to_run, deferred_chunks_count)
    """
    batches = pack_batches(use_items)
    if max_batches_allowed is None:
        return batches, 0

    if len(batches) <= max_batches_allowed:
        return batches, 0

    changed = True
    while len(batches) > max_batches_allowed and changed:
        changed = False
        merged_list: List[Tuple[List[Dict[str, Any]], int]] = []
        i = 0
        while i < len(batches):
            if i < len(batches) - 1:
                merged = try_merge_batches(batches[i], batches[i+1])
                if merged:
                    merged_list.append(merged)
                    i += 2
                    changed = True
                    continue
            merged_list.append(batches[i])
            i += 1
        batches = merged_list

    if len(batches) > max_batches_allowed:
        to_run = batches[:max_batches_allowed]
        deferred = batches[max_batches_allowed:]
        deferred_chunks = sum(len(b[0]) for b in deferred)
        return to_run, deferred_chunks

    return batches, 0

# ----------------- Processing helper (rate-limit retry + parse-error split) -----------------
def process_items(items: List[Dict[str, Any]]) -> Tuple[int, int, float, Dict[str, Dict[str, float]], Dict[str, Any]]:
    """
    Processes a list of items (chunks). Implements:
    - Rate-limit errors (429): retry the same batch indefinitely with backoff and classification (rpm vs quota).
    - JSON parse errors: split strategy, tagged with attempt_id.
    - All logs timestamped; LLM attempts and related logs tagged with attempt_id.

    Returns:
    - num_chunks_processed
    - num_triples_stored
    - gemini_duration_total (sum of LLM durations across unique attempts in this call)
    - per_chunk_stats: {
        chunk_id: {
          "gemini": x,
          "embed": y,
          "neo4j": z,
          "triples": n,
          "attempt_id": id,
          "json_path": path,
          "document_id": doc,
          "uu_number": uu,
          "embed_counts": {...}
        }
      }
    - extra_counters: {
        "final_batches": b, "rate_limit_retries": r, "json_parse_errors": j, "llm_calls": c,
        "attempts_info": [ {attempt_id, duration, chunk_ids, uu_numbers, document_ids, json_path} ]
      }
    """
    if not items:
        return 0, 0, 0.0, {}, {"final_batches": 0, "rate_limit_retries": 0, "json_parse_errors": 0, "llm_calls": 0, "attempts_info": []}

    if ENFORCE_API_BUDGET and not API_BUDGET.will_allow("llm", 1):
        log_warn("Stopping: API budget for LLM calls would be exceeded.", attempt_id=None)
        return 0, 0, 0.0, {}, {"final_batches": 0, "rate_limit_retries": 0, "json_parse_errors": 0, "llm_calls": 0, "attempts_info": []}

    rate_limit_retries = 0
    json_parse_errors = 0
    llm_calls = 0
    attempts_info: List[Dict[str, Any]] = []

    # Single item case
    if len(items) == 1:
        item = items[0]
        meta = item["meta"]
        text = item["text"]
        chunk_id = meta.get("chunk_id")
        per_chunk: Dict[str, Dict[str, float]] = {}
        backoff = random.uniform(2.0, 7.0)  # seconds, exponential for rate-limit
        single_parse_err_count = 0
        last_attempt_id: Optional[int] = None
        last_json_path: Optional[str] = None

        while True:
            if ENFORCE_API_BUDGET and not API_BUDGET.will_allow("llm", 1):
                log_warn(f"Stopping (budget exhausted) before LLM call for single chunk {chunk_id}", attempt_id=last_attempt_id)
                return 0, 0, 0.0, {}, {"final_batches": 0, "rate_limit_retries": rate_limit_retries, "json_parse_errors": json_parse_errors, "llm_calls": llm_calls, "attempts_info": attempts_info}

            prompt = build_single_prompt(meta, text)
            llm_calls += 1
            try:
                triples, gemini_dur, attempt_id, json_path = extract_triples_from_chunk(text, meta, prompt_override=prompt)
                last_attempt_id = attempt_id
                last_json_path = json_path
                written, emb_dur, neo4j_dur, embed_stats = write_triples_for_chunk(triples)

                per_chunk[chunk_id] = {
                    "gemini": gemini_dur,  # single-attempt leader gets full LLM time
                    "embed": emb_dur,
                    "neo4j": neo4j_dur,
                    "triples": written,
                    "attempt_id": attempt_id,
                    "json_path": json_path or "",
                    "document_id": meta.get("document_id") or "",
                    "uu_number": meta.get("uu_number") or "",
                    "embed_counts": embed_stats
                }

                attempts_info.append({
                    "attempt_id": attempt_id,
                    "duration": gemini_dur,
                    "chunk_ids": [chunk_id],
                    "uu_numbers": [meta.get("uu_number") or ""],
                    "document_ids": [meta.get("document_id") or ""],
                    "json_path": json_path or ""
                })

                log_info(
                    f"Attempt completed successfully: chunks={[chunk_id]} | uu={[meta.get('uu_number')]} | json={json_path} | "
                    f"triples_written={written} | times: llm={gemini_dur:.2f}s embed={emb_dur:.2f}s neo4j={neo4j_dur:.2f}s",
                    attempt_id=attempt_id
                )

                return 1, written, gemini_dur, per_chunk, {
                    "final_batches": 1,
                    "rate_limit_retries": rate_limit_retries,
                    "json_parse_errors": json_parse_errors,
                    "llm_calls": llm_calls,
                    "attempts_info": attempts_info
                }
            except RateLimitError as rle:
                last_attempt_id = rle.attempt_id
                last_json_path = getattr(rle, "json_path", None)
                rate_limit_retries += 1
                sleep_for = min(backoff, random.uniform(50.0, 80.0))
                log_warn(f"Rate-limit ({rle.category}) on single chunk {chunk_id}. Reason: {rle.reason}. "
                         f"(doc={meta.get('document_id')}, uu={meta.get('uu_number')}, json={last_json_path or 'n/a'}) "
                         f"Retrying after {sleep_for:.1f}s (retry #{rate_limit_retries}).", attempt_id=last_attempt_id)
                log_info(f"attempt id {last_attempt_id} failed", attempt_id=last_attempt_id)
                time.sleep(sleep_for)
                backoff = min(backoff * 2.0, random.uniform(50.0, 80.0))
            except JsonParseError as e:
                last_attempt_id = e.attempt_id
                last_json_path = getattr(e, "json_path", None)
                json_parse_errors += 1
                single_parse_err_count += 1
                if single_parse_err_count >= SINGLE_PARSE_SPLIT_AFTER:
                    left_text, right_text = split_text_in_two(text)
                    if not right_text:
                        log_warn(f"JSON parse error on single chunk {chunk_id}, could not split further. "
                                 f"(doc={meta.get('document_id')}, uu={meta.get('uu_number')}, json={last_json_path or 'n/a'}) "
                                 f"Continuing retries. Detail: {e}", attempt_id=last_attempt_id)
                        log_info(f"attempt id {last_attempt_id} failed", attempt_id=last_attempt_id)
                        time.sleep(1.0)
                        continue
                    left_meta = dict(meta); left_meta["chunk_id"] = f"{chunk_id}::part1"
                    right_meta = dict(meta); right_meta["chunk_id"] = f"{chunk_id}::part2"
                    log_warn(f"JSON parse error repeated {single_parse_err_count} on {chunk_id}. Splitting into 2 parts: {len(left_text)} + {len(right_text)} chars. "
                             f"(doc={meta.get('document_id')}, uu={meta.get('uu_number')}, json={last_json_path or 'n/a'})", attempt_id=last_attempt_id)
                    log_info(f"attempt id {last_attempt_id} failed", attempt_id=last_attempt_id)
                    l_chunks, l_triples, l_gemini, l_stats, l_extra = process_items([{"text": left_text, "meta": left_meta}])
                    r_chunks, r_triples, r_gemini, r_stats, r_extra = process_items([{"text": right_text, "meta": right_meta}])

                    per_chunk_agg: Dict[str, Dict[str, float]] = {}
                    per_chunk_agg.update(l_stats)
                    per_chunk_agg.update(r_stats)

                    attempts_info_agg = []
                    attempts_info_agg.extend(l_extra.get("attempts_info", []))
                    attempts_info_agg.extend(r_extra.get("attempts_info", []))

                    return (
                        l_chunks + r_chunks,
                        l_triples + r_triples,
                        l_gemini + r_gemini,
                        per_chunk_agg,
                        {
                            "final_batches": l_extra.get("final_batches", 0) + r_extra.get("final_batches", 0),
                            "rate_limit_retries": rate_limit_retries + l_extra.get("rate_limit_retries", 0) + r_extra.get("rate_limit_retries", 0),
                            "json_parse_errors": json_parse_errors + l_extra.get("json_parse_errors", 0) + r_extra.get("json_parse_errors", 0),
                            "llm_calls": llm_calls + l_extra.get("llm_calls", 0) + r_extra.get("llm_calls", 0),
                            "attempts_info": attempts_info_agg
                        }
                    )
                else:
                    log_warn(f"JSON parse error on single chunk {chunk_id}. "
                             f"(doc={meta.get('document_id')}, uu={meta.get('uu_number')}, json={last_json_path or 'n/a'}) "
                             f"Retrying. Detail: {e}", attempt_id=last_attempt_id)
                    log_info(f"attempt id {last_attempt_id} failed", attempt_id=last_attempt_id)
                    time.sleep(1.0)
            except LlmCallError as e:
                last_attempt_id = e.attempt_id
                last_json_path = getattr(e, "json_path", None)
                json_parse_errors += 1
                log_warn(f"Runtime error on single chunk {chunk_id}. "
                         f"(doc={meta.get('document_id')}, uu={meta.get('uu_number')}, json={last_json_path or 'n/a'}) "
                         f"Treating as parse error and retrying. Detail: {e}", attempt_id=last_attempt_id)
                log_info(f"attempt id {last_attempt_id} failed", attempt_id=last_attempt_id)
                time.sleep(1.0)
            except RuntimeError as e:
                if "API budget would be exceeded" in str(e):
                    log_warn(f"Budget exhausted while processing single chunk {chunk_id}", attempt_id=last_attempt_id)
                    return 0, 0, 0.0, {}, {"final_batches": 0, "rate_limit_retries": rate_limit_retries, "json_parse_errors": json_parse_errors, "llm_calls": llm_calls, "attempts_info": attempts_info}
                json_parse_errors += 1
                log_warn(f"Runtime error on single chunk {chunk_id}. "
                         f"(doc={meta.get('document_id')}, uu={meta.get('uu_number')}, json={last_json_path or 'n/a'}) "
                         f"Treating as parse error and retrying. Detail: {e}", attempt_id=last_attempt_id)
                if last_attempt_id is not None:
                    log_info(f"attempt id {last_attempt_id} failed", attempt_id=last_attempt_id)
                time.sleep(1.0)
            except Exception as e:
                json_parse_errors += 1
                log_warn(f"Unexpected error on single chunk {chunk_id}. "
                         f"(doc={meta.get('document_id')}, uu={meta.get('uu_number')}, json={last_json_path or 'n/a'}) "
                         f"Treating as parse error and retrying. Detail: {e}", attempt_id=last_attempt_id)
                if last_attempt_id is not None:
                    log_info(f"attempt id {last_attempt_id} failed", attempt_id=last_attempt_id)
                time.sleep(1.0)

    # Batch case
    else:
        per_chunk: Dict[str, Dict[str, float]] = {}
        backoff = random.uniform(2.0, 7.0)  # seconds, exponential for rate-limit
        last_attempt_id: Optional[int] = None
        last_json_path: Optional[str] = None

        while True:
            if ENFORCE_API_BUDGET and not API_BUDGET.will_allow("llm", 1):
                log_warn("Stopping: API budget for LLM calls would be exceeded (batch).", attempt_id=last_attempt_id)
                return 0, 0, 0.0, {}, {"final_batches": 0, "rate_limit_retries": 0, "json_parse_errors": 0, "llm_calls": 0, "attempts_info": attempts_info}

            prompt = build_batch_prompt(items)
            llm_calls += 1
            try:
                results_map, gemini_dur, attempt_id, json_path = extract_triples_from_chunks_batch(items, prompt)
                last_attempt_id = attempt_id
                last_json_path = json_path

                num_chunks_processed = 0
                num_triples_stored = 0

                leader_cid = items[0]["meta"]["chunk_id"] if items else None
                emb_total = 0.0
                neo_total = 0.0

                for it in items:
                    cid = it["meta"]["chunk_id"]
                    triples = results_map.get(cid, [])
                    written, emb_dur, neo4j_dur, embed_stats = write_triples_for_chunk(triples)

                    gem_for_chunk = gemini_dur if cid == leader_cid else 0.0

                    per_chunk[cid] = {
                        "gemini": gem_for_chunk,
                        "embed": emb_dur,
                        "neo4j": neo4j_dur,
                        "triples": written,
                        "attempt_id": attempt_id,
                        "json_path": json_path or "",
                        "document_id": it["meta"].get("document_id") or "",
                        "uu_number": it["meta"].get("uu_number") or "",
                        "embed_counts": embed_stats
                    }

                    num_chunks_processed += 1
                    num_triples_stored += written
                    emb_total += emb_dur
                    neo_total += neo4j_dur

                attempts_info.append({
                    "attempt_id": attempt_id,
                    "duration": gemini_dur,
                    "chunk_ids": [it["meta"]["chunk_id"] for it in items],
                    "uu_numbers": list(sorted({it["meta"].get("uu_number") or "" for it in items})),
                    "document_ids": list(sorted({it["meta"].get("document_id") or "" for it in items})),
                    "json_path": json_path or ""
                })

                log_info(
                    f"Attempt completed successfully: chunks={[it['meta']['chunk_id'] for it in items]} | "
                    f"uu={list(sorted({it['meta'].get('uu_number') for it in items}))} | json={json_path} | "
                    f"triples_written={num_triples_stored} | times: llm={gemini_dur:.2f}s embed={emb_total:.2f}s neo4j={neo_total:.2f}s",
                    attempt_id=attempt_id
                )

                return num_chunks_processed, num_triples_stored, gemini_dur, per_chunk, {
                    "final_batches": 1,
                    "rate_limit_retries": rate_limit_retries,
                    "json_parse_errors": json_parse_errors,
                    "llm_calls": llm_calls,
                    "attempts_info": attempts_info
                }

            except RateLimitError as rle:
                last_attempt_id = rle.attempt_id
                last_json_path = getattr(rle, "json_path", None)
                rate_limit_retries += 1
                sleep_for = min(backoff, random.uniform(50.0, 80.0))
                log_warn(f"Rate-limit ({rle.category}) on batch (size={len(items)}). Reason: {rle.reason}. "
                         f"chunks={[it['meta']['chunk_id'] for it in items]} docs={[it['meta'].get('document_id') for it in items]} "
                         f"uu={list(sorted({it['meta'].get('uu_number') for it in items}))} json={last_json_path or 'n/a'}. "
                         f"Retrying after {sleep_for:.1f}s (retry #{rate_limit_retries}).", attempt_id=last_attempt_id)
                log_info(f"attempt id {last_attempt_id} failed", attempt_id=last_attempt_id)
                time.sleep(sleep_for)
                backoff = min(backoff * 2.0, random.uniform(50.0, 80.0))
                continue
            except JsonParseError as e:
                last_attempt_id = e.attempt_id
                last_json_path = getattr(e, "json_path", None)
                json_parse_errors += 1
                mid = max(1, len(items) // 2)
                left = items[:mid]
                right = items[mid:]
                log_warn(f"JSON parse error on batch (size={len(items)}). Splitting into {len(left)} + {len(right)}. "
                         f"chunks={[it['meta']['chunk_id'] for it in items]} docs={[it['meta'].get('document_id') for it in items]} "
                         f"uu={list(sorted({it['meta'].get('uu_number') for it in items}))} json={last_json_path or 'n/a'}. Detail: {e}", attempt_id=last_attempt_id)
                log_info(f"attempt id {last_attempt_id} failed", attempt_id=last_attempt_id)
                l_chunks, l_triples, l_gemini, l_stats, l_extra = process_items(left)
                r_chunks, r_triples, r_gemini, r_stats, r_extra = process_items(right)

                per_chunk_agg = {}
                per_chunk_agg.update(l_stats)
                per_chunk_agg.update(r_stats)

                attempts_info_agg = []
                attempts_info_agg.extend(l_extra.get("attempts_info", []))
                attempts_info_agg.extend(r_extra.get("attempts_info", []))

                return (l_chunks + r_chunks,
                        l_triples + r_triples,
                        l_gemini + r_gemini,
                        per_chunk_agg,
                        {
                            "final_batches": l_extra.get("final_batches", 0) + r_extra.get("final_batches", 0),
                            "rate_limit_retries": rate_limit_retries + l_extra.get("rate_limit_retries", 0) + r_extra.get("rate_limit_retries", 0),
                            "json_parse_errors": json_parse_errors + l_extra.get("json_parse_errors", 0) + r_extra.get("json_parse_errors", 0),
                            "llm_calls": llm_calls + l_extra.get("llm_calls", 0) + r_extra.get("llm_calls", 0),
                            "attempts_info": attempts_info_agg
                        })
            except LlmCallError as e:
                last_attempt_id = e.attempt_id
                last_json_path = getattr(e, "json_path", None)
                json_parse_errors += 1
                mid = max(1, len(items) // 2)
                left = items[:mid]
                right = items[mid:]
                log_warn(f"Runtime error on batch (size={len(items)}). Splitting into {len(left)} + {len(right)}. "
                         f"chunks={[it['meta']['chunk_id'] for it in items]} docs={[it['meta'].get('document_id') for it in items]} "
                         f"uu={list(sorted({it['meta'].get('uu_number') for it in items}))} json={last_json_path or 'n/a'}. Detail: {e}", attempt_id=last_attempt_id)
                log_info(f"attempt id {last_attempt_id} failed", attempt_id=last_attempt_id)
                l_chunks, l_triples, l_gemini, l_stats, l_extra = process_items(left)
                r_chunks, r_triples, r_gemini, r_stats, r_extra = process_items(right)

                per_chunk_agg = {}
                per_chunk_agg.update(l_stats)
                per_chunk_agg.update(r_stats)

                attempts_info_agg = []
                attempts_info_agg.extend(l_extra.get("attempts_info", []))
                attempts_info_agg.extend(r_extra.get("attempts_info", []))

                return (l_chunks + r_chunks,
                        l_triples + r_triples,
                        l_gemini + r_gemini,
                        per_chunk_agg,
                        {
                            "final_batches": l_extra.get("final_batches", 0) + r_extra.get("final_batches", 0),
                            "rate_limit_retries": rate_limit_retries + l_extra.get("rate_limit_retries", 0) + r_extra.get("rate_limit_retries", 0),
                            "json_parse_errors": json_parse_errors + l_extra.get("json_parse_errors", 0) + r_extra.get("json_parse_errors", 0),
                            "llm_calls": llm_calls + l_extra.get("llm_calls", 0) + r_extra.get("llm_calls", 0),
                            "attempts_info": attempts_info_agg
                        })
            except RuntimeError as e:
                if "API budget would be exceeded" in str(e):
                    log_warn("Budget exhausted during batch processing.", attempt_id=last_attempt_id)
                    return 0, 0, 0.0, {}, {"final_batches": 0, "rate_limit_retries": rate_limit_retries, "json_parse_errors": json_parse_errors, "llm_calls": llm_calls, "attempts_info": attempts_info}
                json_parse_errors += 1
                mid = max(1, len(items) // 2)
                left = items[:mid]
                right = items[mid:]
                log_warn(f"Runtime error on batch (size={len(items)}). Splitting into {len(left)} + {len(right)}. "
                         f"chunks={[it['meta']['chunk_id'] for it in items]} docs={[it['meta'].get('document_id') for it in items]} "
                         f"uu={list(sorted({it['meta'].get('uu_number') for it in items}))} json={last_json_path or 'n/a'}. Detail: {e}", attempt_id=last_attempt_id)
                if last_attempt_id is not None:
                    log_info(f"attempt id {last_attempt_id} failed", attempt_id=last_attempt_id)
                l_chunks, l_triples, l_gemini, l_stats, l_extra = process_items(left)
                r_chunks, r_triples, r_gemini, r_stats, r_extra = process_items(right)

                per_chunk_agg = {}
                per_chunk_agg.update(l_stats)
                per_chunk_agg.update(r_stats)

                attempts_info_agg = []
                attempts_info_agg.extend(l_extra.get("attempts_info", []))
                attempts_info_agg.extend(r_extra.get("attempts_info", []))

                return (l_chunks + r_chunks,
                        l_triples + r_triples,
                        l_gemini + r_gemini,
                        per_chunk_agg,
                        {
                            "final_batches": l_extra.get("final_batches", 0) + r_extra.get("final_batches", 0),
                            "rate_limit_retries": rate_limit_retries + l_extra.get("rate_limit_retries", 0) + r_extra.get("rate_limit_retries", 0),
                            "json_parse_errors": json_parse_errors + l_extra.get("json_parse_errors", 0) + r_extra.get("json_parse_errors", 0),
                            "llm_calls": llm_calls + l_extra.get("llm_calls", 0) + r_extra.get("llm_calls", 0),
                            "attempts_info": attempts_info_agg
                        })
            except Exception as e:
                json_parse_errors += 1
                mid = max(1, len(items) // 2)
                left = items[:mid]
                right = items[mid:]
                log_warn(f"Unexpected error on batch (size={len(items)}). Splitting into {len(left)} + {len(right)}. "
                         f"chunks={[it['meta']['chunk_id'] for it in items]} docs={[it['meta'].get('document_id') for it in items]} "
                         f"uu={list(sorted({it['meta'].get('uu_number') for it in items}))} json={last_json_path or 'n/a'}. Detail: {e}", attempt_id=last_attempt_id)
                if last_attempt_id is not None:
                    log_info(f"attempt id {last_attempt_id} failed", attempt_id=last_attempt_id)
                l_chunks, l_triples, l_gemini, l_stats, l_extra = process_items(left)
                r_chunks, r_triples, r_gemini, r_stats, r_extra = process_items(right)

                per_chunk_agg = {}
                per_chunk_agg.update(l_stats)
                per_chunk_agg.update(r_stats)

                attempts_info_agg = []
                attempts_info_agg.extend(l_extra.get("attempts_info", []))
                attempts_info_agg.extend(r_extra.get("attempts_info", []))

                return (l_chunks + r_chunks,
                        l_triples + r_triples,
                        l_gemini + r_gemini,
                        per_chunk_agg,
                        {
                            "final_batches": l_extra.get("final_batches", 0) + r_extra.get("final_batches", 0),
                            "rate_limit_retries": rate_limit_retries + l_extra.get("rate_limit_retries", 0) + r_extra.get("rate_limit_retries", 0),
                            "json_parse_errors": json_parse_errors + l_extra.get("json_parse_errors", 0) + r_extra.get("json_parse_errors", 0),
                            "llm_calls": llm_calls + l_extra.get("llm_calls", 0) + r_extra.get("llm_calls", 0),
                            "attempts_info": attempts_info_agg
                        })

# ----------------- Main pipeline (parallel, budget-capped batching) -----------------
def run_kg_pipeline_over_folder(
    dir_path: Path,
    max_files: Optional[int] = None,
    max_chunks_per_file: Optional[int] = None
):
    pkls = list_pickles(dir_path)
    if max_files is not None:
        pkls = pkls[:max_files]

    total_chunks_planned = 0
    for p in pkls:
        try:
            with open(p, "rb") as f:
                chunks = pickle.load(f)
            n = len(chunks)
            total_chunks_planned += n if max_chunks_per_file is None else min(n, max_chunks_per_file)
        except Exception:
            continue

    log_info(f"Found {len(pkls)} pickle files in {dir_path} (skipping: {', '.join(SKIP_FILES) or 'none'})")
    log_info(f"Total chunks planned: {total_chunks_planned}")
    if ENFORCE_API_BUDGET:
        allowed_calls_remaining = max(0, API_BUDGET.total - API_BUDGET.used)
        log_info(f"API budget: enforce={API_BUDGET.enforce}, total={API_BUDGET.total}, used={API_BUDGET.used}, allowed_calls_remaining={allowed_calls_remaining}")
    else:
        log_info(f"API budget: enforce={API_BUDGET.enforce} (unlimited LLM calls)")
    log_info(f"LLM rate limit: {LLM_MAX_CALLS_PER_MIN} calls/minute")
    log_info(f"Greedy batching target: <= {PRACTICAL_MAX_ITEMS_PER_BATCH} items, est tokens <= {PROMPT_TOKEN_LIMIT}")
    log_info(f"Parallel workers: {INDEX_WORKERS}")
    if STAGGER_WORKER_SECONDS > 0:
        log_info(f"Worker ramp-up enabled: start 1 worker, add 1 every {STAGGER_WORKER_SECONDS:.3f}s (up to {INDEX_WORKERS}).")
    else:
        log_info("Worker ramp-up disabled (all workers may start immediately).")

    raw_items_all: List[Dict[str, Any]] = []
    for file_idx, pkl in enumerate(pkls, 1):
        log_info(f"[{file_idx}/{len(pkls)}] Scanning {pkl.name}")
        try:
            chunks = load_chunks_from_file(pkl)
        except Exception as e:
            log_warn(f"Failed to load {pkl.name}: {e}")
            continue

        for idx, ch in enumerate(chunks):
            if max_chunks_per_file is not None and idx >= max_chunks_per_file:
                break
            meta_source = getattr(ch, "metadata", {}) if hasattr(ch, "metadata") else {}
            meta = {
                "document_id": meta_source.get("document_id"),
                "chunk_id": meta_source.get("chunk_id"),
                "uu_number": meta_source.get("uu_number"),
                "pages": meta_source.get("pages"),
            }
            text = getattr(ch, "page_content", str(ch))
            if not meta.get("chunk_id"):
                meta["chunk_id"] = f"{pkl.stem}_chunk_{idx}"
            raw_items_all.append({"chunk_id": meta["chunk_id"], "text": text, "meta": meta})

    if not raw_items_all:
        log_info("No chunks found. Exiting.")
        return

    max_batches_allowed = None
    if ENFORCE_API_BUDGET:
        max_batches_allowed = max(0, API_BUDGET.total - API_BUDGET.used)

    all_batches_capped, deferred_chunks = pack_batches_with_cap(raw_items_all, max_batches_allowed)
    total_batches_planned = len(all_batches_capped)
    log_info(f"Packed {len(raw_items_all)} chunks into {total_batches_planned} batch(es) (budget-capped).")
    for i, (batch_items, est_tokens) in enumerate(all_batches_capped, 1):
        log_info(f"  • Batch {i}: chunks={len(batch_items)}, est_tokens~{est_tokens}")
    if deferred_chunks > 0:
        log_warn(f"Deferring {deferred_chunks} chunk(s) to future runs due to API budget cap of {max_batches_allowed} batch(es).")

    total_triples_stored = 0
    total_chunks_done = 0

    total_gemini_duration = 0.0
    total_embedding_duration = 0.0
    total_neo4j_duration = 0.0

    per_batch_component_sums: Dict[int, float] = {}
    per_batch_wall_times: Dict[int, float] = {}
    per_batch_sizes: Dict[int, int] = {}

    global_final_batches = 0
    global_rate_limit_retries = 0
    global_json_parse_errors = 0
    global_llm_call_attempts = 0

    overall_start = time.time()

    def batch_desc(idx: int, size: int, tokens: int) -> str:
        return f"[Batch {idx+1}/{total_batches_planned}] size={size}, est_tokens~{tokens}"

    def remaining_budget() -> int:
        if not ENFORCE_API_BUDGET:
            return 1_000_000_000
        return max(0, API_BUDGET.total - API_BUDGET.used)

    batches_completed_so_far = 0

    with ThreadPoolExecutor(max_workers=INDEX_WORKERS) as executor:
        ramp_start = time.time()

        def allowed_workers_now() -> int:
            if STAGGER_WORKER_SECONDS <= 0:
                return INDEX_WORKERS
            elapsed = time.time() - ramp_start
            allowed = 1 + int(elapsed // STAGGER_WORKER_SECONDS)
            return min(INDEX_WORKERS, max(1, allowed))

        futures_set = set()
        futures_meta: Dict[Any, Tuple[int, int, List[Dict[str, Any]], float]] = {}
        next_idx = 0

        def time_until_next_ramp() -> float:
            if STAGGER_WORKER_SECONDS <= 0:
                return 1.0
            elapsed = time.time() - ramp_start
            steps_completed = int(elapsed // STAGGER_WORKER_SECONDS)
            next_step_time = (steps_completed + 1) * STAGGER_WORKER_SECONDS
            return max(0.0, next_step_time - elapsed)

        while True:
            target = allowed_workers_now()
            if ENFORCE_API_BUDGET:
                target = min(target, remaining_budget())

            while next_idx < total_batches_planned and len(futures_set) < target:
                items_batch, est_tokens = all_batches_capped[next_idx]
                submit_time = time.time()
                fut = executor.submit(process_items, items_batch)
                futures_set.add(fut)
                futures_meta[fut] = (next_idx, est_tokens, items_batch, submit_time)
                next_idx += 1

            if not futures_set and next_idx >= total_batches_planned:
                break

            timeout = min(1.0, time_until_next_ramp())
            done, _ = wait(futures_set, timeout=timeout, return_when=FIRST_COMPLETED)

            for fut in list(done):
                b_idx, est_tokens, items_batch, submit_time = futures_meta.pop(fut)
                futures_set.remove(fut)

                start_label = batch_desc(b_idx, len(items_batch), est_tokens)
                try:
                    chunks_processed, triples_stored, gemini_dur, per_chunk_stats, extra = fut.result()
                except Exception as e:
                    log_error(f"{start_label} failed with error: {e}")
                    batches_completed_so_far += 1
                    continue

                end_time = time.time()
                wall_time = end_time - submit_time

                embed_total = sum(stats['embed'] for stats in per_chunk_stats.values())
                neo4j_total = sum(stats['neo4j'] for stats in per_chunk_stats.values())
                comp_sum = gemini_dur + embed_total + neo4j_total

                total_chunks_done += chunks_processed
                total_triples_stored += triples_stored
                total_gemini_duration += gemini_dur
                total_embedding_duration += embed_total
                total_neo4j_duration += neo4j_total

                final_batches = extra.get("final_batches", 0)
                rate_limit_retries = extra.get("rate_limit_retries", 0)
                json_parse_errors = extra.get("json_parse_errors", 0)
                llm_calls = extra.get("llm_calls", 0)
                attempts_info = extra.get("attempts_info", [])

                global_final_batches += final_batches
                global_rate_limit_retries += rate_limit_retries
                global_json_parse_errors += json_parse_errors
                global_llm_call_attempts += llm_calls

                per_batch_component_sums[b_idx] = comp_sum
                per_batch_wall_times[b_idx] = wall_time
                per_batch_sizes[b_idx] = len(items_batch)

                if per_chunk_stats:
                    for it in items_batch:
                        cid = it["meta"]["chunk_id"]
                        stats = per_chunk_stats.get(cid)
                        if stats:
                            json_base = Path(stats.get('json_path') or "").name
                            uu_num = stats.get('uu_number') or "unknown"
                            attempt_id = int(stats.get('attempt_id')) if stats.get('attempt_id') is not None else None
                            gem_disp = f"{stats['gemini']:.2f}s" if stats['gemini'] > 0 else "0.00s (shared)"
                            emb_counts = stats.get("embed_counts") or {}
                            emb_detail = ""
                            if emb_counts:
                                emb_detail = f" (new e={emb_counts.get('entity_new_calls',0)},t={emb_counts.get('triple_new_calls',0)} | cached e={emb_counts.get('entity_cached',0)},t={emb_counts.get('triple_cached',0)})"
                            log_info(f"      - Chunk {cid} (uu={uu_num}, attempt={attempt_id}, json={json_base}) "
                                     f"Gemini={gem_disp} | Embedding={stats['embed']:.2f}s{emb_detail} | Neo4j={stats['neo4j']:.2f}s | Triples={int(stats['triples'])}")

                attempt_ids_list = [a.get("attempt_id") for a in attempts_info] if attempts_info else []
                log_info(f"    ✓ {start_label}")
                log_info(f"        - KG extraction (LLM): {gemini_dur:.2f}s across {max(1, len(attempt_ids_list))} call(s), attempt_ids={attempt_ids_list}")
                log_info(f"        - Embedding total:     {embed_total:.2f}s")
                log_info(f"        - Neo4j insert total:  {neo4j_total:.2f}s")
                log_info(f"        - Component sum:       {comp_sum:.2f}s")
                log_info(f"        - Batch wall time:     {wall_time:.2f}s")
                log_info(f"        - Overhead (wall - components): {wall_time - comp_sum:+.2f}s")
                log_info(f"        - Chunks processed: {chunks_processed} | Triples stored: {triples_stored}")

                batches_completed_so_far += 1
                log_info(f"        - Batches completed: {batches_completed_so_far}/{total_batches_planned}")

    total_time_real = time.time() - overall_start
    total_llm_calls_used = API_BUDGET.resource_usage.get("llm", 0)

    sequential_estimate = sum(per_batch_component_sums.values())
    speedup = (sequential_estimate / total_time_real) if total_time_real > 0 else float('inf')

    log_info("Summary")
    log_info(f"- Batches planned (initial, capped): {total_batches_planned}")
    log_info(f"- Batches processed after JSON-split: {global_final_batches} (extra from splits: {max(0, global_final_batches - total_batches_planned)})")
    log_info(f"- Chunks processed: {total_chunks_done}/{total_chunks_planned}")
    log_info(f"- Triples stored: {total_triples_stored}")
    log_info(f"- LLM calls attempted (total): {global_llm_call_attempts}")
    log_info(f"- Rate-limit retries (429): {global_rate_limit_retries}")
    log_info(f"- JSON parse errors encountered: {global_json_parse_errors}")
    log_info(f"- LLM calls used (budget counter): {total_llm_calls_used}{' (budget enforced)' if ENFORCE_API_BUDGET else ''}")
    log_info(f"- Total Gemini time (sum of batch LLM times): {total_gemini_duration:.2f}s")
    log_info(f"- Total Embedding time (sum): {total_embedding_duration:.2f}s")
    log_info(f"- Total Neo4j time (sum): {total_neo4j_duration:.2f}s")
    log_info(f"- Total real wall time: {total_time_real:.2f}s")
    log_info(f"- Sequential time estimate (components sum): {sequential_estimate:.2f}s")
    log_info(f"- Speedup vs sequential: {speedup:.2f}× faster")
    if ENFORCE_API_BUDGET:
        log_info(f"- API used: {API_BUDGET.used}/{API_BUDGET.total} (LLM calls and {'embeddings' if COUNT_EMBEDDINGS_IN_BUDGET else 'no embeddings'} counted)")

if __name__ == "__main__":
    run_kg_pipeline_over_folder(
        LANGCHAIN_DIR,
        max_files=1000,
        max_chunks_per_file=None
    )