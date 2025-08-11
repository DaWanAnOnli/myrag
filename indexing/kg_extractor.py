import os, json, hashlib, time, threading
from collections import deque
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
from tenacity import retry, wait_exponential, stop_after_attempt
from dotenv import load_dotenv
import google.generativeai as genai
from google.generativeai.types import GenerationConfig

# Load .env from the parent directory of this file
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]
GEN_MODEL = os.getenv("GEN_MODEL", "models/gemini-2.5-flash")
EMBED_MODEL = os.getenv("EMBED_MODEL", "models/text-embedding-004")

genai.configure(api_key=GOOGLE_API_KEY)

# Dimensions for text-embedding-004 (as of now)
EMBED_DIM = 768

# Indonesian vocabulary
LEGAL_ENTITY_TYPES = [
    "UU", "PASAL", "AYAT", "INSTANSI", "ORANG", "ISTILAH", "SANKSI", "NOMINAL", "TANGGAL"
]
LEGAL_PREDICATES = [
    "mendefinisikan",
    "mengubah",
    "mencabut",
    "mulai_berlaku",
    "mewajibkan",
    "melarang",
    "memberikan_sanksi",
    "berlaku_untuk",
    "termuat_dalam",
    "mendelegasikan_kepada",
    "berjumlah",
    "berdurasi"
]

# JSON schema (keys in English, values in Indonesian where applicable)
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
              "type": {"type": "string"},       # must be Indonesian (e.g., "UU", "PASAL")
              "canonical_id": {"type": "string"}
            },
            "required": ["text"]
          },
          "predicate": {"type": "string"},       # must be Indonesian (from LEGAL_PREDICATES)
          "object": {
            "type": "object",
            "properties": {
              "text": {"type": "string"},
              "type": {"type": "string"},       # must be Indonesian
              "canonical_id": {"type": "string"}
            },
            "required": ["text"]
          },
          "evidence": {
            "type": "object",
            "properties": {
              "quote": {"type": "string"},
              "char_start": {"type": "integer"},
              "char_end": {"type": "integer"},
              "article_ref": {"type": "string"}  # e.g., "Pasal 5 ayat (2)"
            }
          },
          "confidence": {"type": "number"}
        },
        "required": ["subject", "predicate", "object"]
      }
    }
  },
  "required": ["triples"]
}

# English instructions; force Indonesian output values
SYSTEM_HINT = f"""
You extract knowledge graph triples from Indonesian legal text (Undang-Undang).
Output requirements:
- subject.type, object.type, and predicate MUST be Indonesian strings only.
- Allowed entity types: {", ".join(LEGAL_ENTITY_TYPES)}.
- Prefer predicates from: {", ".join(LEGAL_PREDICATES)} (all Indonesian, snake_case where applicable).
- Use 'UU' for the Law, 'PASAL' for Article, 'AYAT' for Clause, 'INSTANSI' for institutions/agencies, 'ISTILAH' for defined terms.

Rules:
- Be precise and strictly grounded in the chunk; if unsupported, omit the triple.
- Include a short evidence.quote, and article_ref like 'Pasal X ayat (Y)' when present.
- Normalize predicate to a lowercase Indonesian single-verb phrase (use the list above when possible).
- Keep numbers/dates verbatim from the text.
"""

gen_model = genai.GenerativeModel(GEN_MODEL)

# ------------- Rate Limiter (max 8 LLM calls per minute) -------------
class RateLimiter:
    def __init__(self, max_calls: int, period_sec: float):
        self.max_calls = max_calls
        self.period = period_sec
        self.calls = deque()
        self.lock = threading.Lock()

    def acquire(self):
        with self.lock:
            now = time.time()
            # drop old timestamps
            while self.calls and (now - self.calls[0]) >= self.period:
                self.calls.popleft()

            if len(self.calls) >= self.max_calls:
                sleep_for = self.period - (now - self.calls[0])
            else:
                sleep_for = 0

        if sleep_for > 0:
            time.sleep(sleep_for)

        with self.lock:
            self.calls.append(time.time())

# Global limiter for LLM calls (not embeddings)
LLM_RATE_LIMITER = RateLimiter(max_calls=8, period_sec=60.0)

# ----------------------------------------------------------------------

def _slug(s: str) -> str:
    return " ".join((s or "").strip().split()).lower()

def normalize_entity_key(text: str, etype: str, uu_number: Optional[str] = None) -> str:
    # Tie UU/PASAL/AYAT to UU number when available to avoid collisions
    if etype in ("UU", "PASAL", "AYAT") and uu_number:
        return f"{etype}::{_slug(uu_number)}::{_slug(text)}"
    return f"{etype}::{_slug(text)}"

def deterministic_triple_uid(
    subject_key: str, predicate: str, object_key: str, doc_id: Optional[str], span: Optional[Tuple[int,int]]
) -> str:
    h = hashlib.sha256()
    payload = "|".join([
        subject_key, (predicate or "").strip().lower(), object_key, doc_id or "",
        str(span[0]) if span else "", str(span[1]) if span else ""
    ])
    h.update(payload.encode("utf-8"))
    return h.hexdigest()

@retry(wait=wait_exponential(multiplier=1, min=2, max=30), stop=stop_after_attempt(5))
def extract_triples_from_chunk(chunk_text: str, meta: Dict[str, Any]) -> List[Dict[str, Any]]:
    # Respect global rate limit for LLM calls
    LLM_RATE_LIMITER.acquire()

    cfg = GenerationConfig(
        temperature=0,
        response_mime_type="application/json",
        response_schema=TRIPLE_SCHEMA,
    )
    prompt = f"""
{SYSTEM_HINT}

Chunk metadata:
- document_id: {meta.get('document_id')}
- chunk_id: {meta.get('chunk_id')}
- uu_number: {meta.get('uu_number')}
- pages: {meta.get('pages')}

Text:
\"\"\"{chunk_text}\"\"\"
"""
    resp = gen_model.generate_content(prompt, generation_config=cfg)

    # Robust JSON parsing (SDK responses vary slightly)
    raw = None
    try:
        raw = resp.text
        data = json.loads(raw)
    except Exception:
        try:
            raw = resp.candidates[0].content.parts[0].text
            data = json.loads(raw)
        except Exception as e:
            preview = raw[:200] if isinstance(raw, str) else "None"
            raise RuntimeError(f"Failed to parse model JSON: {e}; raw={preview}")

    triples: List[Dict[str, Any]] = []
    uu_number = meta.get("uu_number")
    for t in data.get("triples", []):
        subj = t["subject"]; obj = t["object"]; pred = (t["predicate"] or "").strip().lower()

        # Default types in Indonesian
        s_type = subj.get("type") or "ISTILAH"
        o_type = obj.get("type") or "ISTILAH"

        s_key = normalize_entity_key(subj["text"], s_type, uu_number)
        o_key = normalize_entity_key(obj["text"],  o_type,  uu_number)

        ev = t.get("evidence") or {}
        span = None
        if ev.get("char_start") is not None and ev.get("char_end") is not None:
            span = (ev["char_start"], ev["char_end"])
        triple_uid = deterministic_triple_uid(s_key, pred, o_key, meta.get("document_id"), span)

        triples.append({
            "triple_uid": triple_uid,
            "subject": {"text": subj["text"], "type": s_type, "key": s_key},
            "predicate": pred,
            "object": {"text": obj["text"], "type": o_type, "key": o_key},
            "evidence": {
                "quote": ev.get("quote"),
                "char_start": ev.get("char_start"),
                "char_end": ev.get("char_end"),
                "article_ref": ev.get("article_ref"),
            },
            "confidence": float(t.get("confidence", 0.0)),
            "provenance": {
                "document_id": meta.get("document_id"),
                "chunk_id": meta.get("chunk_id"),
                "uu_number": uu_number,
                "pages": meta.get("pages"),
            }
        })
    return triples

def embed_text(text: str) -> List[float]:
    # Embeddings are not rate-limited per your request
    res = genai.embed_content(model=EMBED_MODEL, content=text)
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

def node_embedding_text(name: str, etype: str) -> str:
    return f"{(name or '').strip()} | {etype}"

def triple_embedding_text(t: Dict[str, Any]) -> str:
    s = t["subject"]["text"]; p = t["predicate"]; o = t["object"]["text"]
    uu = t["provenance"].get("uu_number") or ""
    art = (t.get("evidence") or {}).get("article_ref") or ""
    return f"{s} [{p}] {o} | {uu} | {art}"