import os
from typing import List, Dict, Any
from pathlib import Path
from neo4j import GraphDatabase
from dotenv import load_dotenv
from kg_extractor import embed_text

# Load .env from the parent directory of this file
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASS", "password")

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

def search_triples(query: str, k: int = 20) -> List[Dict[str, Any]]:
    q_emb = embed_text(query)
    cypher = """
    CALL db.index.vector.queryNodes('triple_vec', $k, $q_emb) YIELD node AS tr, score
    OPTIONAL MATCH (tr)-[:SUBJECT]->(s:Entity)
    OPTIONAL MATCH (tr)-[:OBJECT]->(o:Entity)
    RETURN tr, s, o, score
    ORDER BY score DESC   // cosine similarity: higher is better
    LIMIT $k
    """
    with driver.session() as session:
        res = session.run(cypher, k=k, q_emb=q_emb)
        rows = []
        for r in res:
            tr = r["tr"]; s = r["s"]; o = r["o"]
            rows.append({
                "predicate": tr.get("predicate"),
                "triple_uid": tr.get("triple_uid"),
                "subject": s.get("name") if s else None,
                "subject_type": s.get("type") if s else None,
                "object": o.get("name") if o else None,
                "object_type": o.get("type") if o else None,
                "uu_number": tr.get("uu_number"),
                "evidence": tr.get("evidence"),
                "score": r["score"],
            })
        return rows

def expand_neighborhood(triple_uid: str, limit: int = 50):
    cypher = """
    MATCH (tr:Triple {triple_uid:$uid})-[:SUBJECT]->(s:Entity)
    MATCH (tr)-[:OBJECT]->(o:Entity)
    OPTIONAL MATCH (s)-[r:REL]->(nbr:Entity)
    RETURN s, o, collect({rel:properties(r), neighbor:nbr})[0..$limit] AS neighbors
    """
    with driver.session() as session:
        res = session.run(cypher, uid=triple_uid, limit=limit)
        return [r.data() for r in res]

if __name__ == "__main__":
    hits = search_triples("tanggal berlaku undang-undang pajak dan sanksi", k=10)
    for h in hits:
        print(h)