import os, json
from typing import Dict, Any, List
from pathlib import Path
from neo4j import GraphDatabase
from dotenv import load_dotenv
from kg_extractor import embed_text, node_embedding_text, triple_embedding_text

# Load .env from the parent directory of this file
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASS", "password")

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

# Simple in-memory caches to reduce embedding calls during a run
_entity_emb_cache: dict[str, List[float]] = {}
_triple_emb_cache: dict[str, List[float]] = {}

def _get_entity_emb(name: str, etype: str, key: str) -> List[float]:
    if key in _entity_emb_cache:
        return _entity_emb_cache[key]
    vec = embed_text(node_embedding_text(name, etype))
    _entity_emb_cache[key] = vec
    return vec

def _get_triple_emb(triple_uid: str, t: Dict[str, Any]) -> List[float]:
    if triple_uid in _triple_emb_cache:
        return _triple_emb_cache[triple_uid]
    vec = embed_text(triple_embedding_text(t))
    _triple_emb_cache[triple_uid] = vec
    return vec

def upsert_triple(tx, t: Dict[str, Any]):
    s, o = t["subject"], t["object"]
    s_name, s_type, s_key = s["text"], s["type"], s["key"]
    o_name, o_type, o_key = o["text"], o["type"], o["key"]
    pred = t["predicate"]
    triple_uid = t["triple_uid"]

    prov = t["provenance"]
    evidence = t.get("evidence") or {}
    confidence = float(t.get("confidence", 0.0))

    # Embeddings (cached)
    s_emb = _get_entity_emb(s_name, s_type, s_key)
    o_emb = _get_entity_emb(o_name, o_type, o_key)
    tr_emb = _get_triple_emb(triple_uid, t)

    cypher = """
    // Entities
    MERGE (s:Entity {key:$s_key})
      ON CREATE SET s.name=$s_name, s.type=$s_type, s.createdAt=timestamp()
    SET s.embedding=$s_emb

    MERGE (o:Entity {key:$o_key})
      ON CREATE SET o.name=$o_name, o.type=$o_type, o.createdAt=timestamp()
    SET o.embedding=$o_emb

    // Triple (reified)
    MERGE (tr:Triple {triple_uid:$triple_uid})
      ON CREATE SET tr.createdAt=timestamp()
    SET tr.predicate=$pred,
        tr.embedding=$tr_emb,
        tr.document_id=$doc_id,
        tr.chunk_id=$chunk_id,
        tr.uu_number=$uu_number,
        tr.pages=$pages,
        tr.evidence=$evidence,
        tr.confidence=$confidence

    // Link triple to entities
    MERGE (tr)-[:SUBJECT]->(s)
    MERGE (tr)-[:OBJECT]->(o)

    // Optional direct relationship for faster traversal
    MERGE (s)-[r:REL {triple_uid:$triple_uid}]->(o)
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
        evidence=json.dumps(evidence, ensure_ascii=False),
        confidence=confidence,
    )

def write_triples_to_neo4j(triples: List[Dict[str, Any]]):
    if not triples:
        return
    with driver.session() as session:
        session.execute_write(lambda tx: [upsert_triple(tx, t) for t in triples])