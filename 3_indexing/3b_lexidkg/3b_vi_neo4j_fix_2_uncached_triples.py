# -*- coding: utf-8 -*-
# Fix script — patches Triple nodes with NULL s_name/o_name properties.
# The two problematic triples have no :SUBJECT or :OBJECT relationships
# pointing to valid entity nodes — they reference ghost keys that no longer exist.
# We set placeholder values so the covering index can still serve the queries.
#
# Usage: python 3b_vi_neo4j_fix_2_uncached_triples.py

import os
from pathlib import Path
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent.parent / ".env")

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASS", "@ik4nkus")

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

print("=" * 60)
print("Fixing uncached Triple nodes...")
print("=" * 60)

with driver.session() as s:
    # Find all triples where s_name or o_name is NULL
    uncached = list(s.run("""
        MATCH (tr:Triple)
        WHERE tr.s_name IS NULL OR tr.o_name IS NULL
        RETURN tr.triple_uid AS uid
    """))
    print(f"\nFound {len(uncached)} triple(s) with NULL s_name/o_name:")
    for r in uncached:
        print(f"  - uid={r['uid']}")

    fixed = 0
    still_null = 0
    for r in uncached:
        uid = r["uid"]
        print(f"\nProcessing {uid[:16]}...")

        # Try to find the subject and object entities via the graph relationships
        result = s.run("""
            MATCH (tr:Triple {triple_uid: $uid})
            OPTIONAL MATCH (tr)-[:SUBJECT]->(s)
            OPTIONAL MATCH (tr)-[:OBJECT]->(o)
            RETURN s.name AS s_name, s.key AS s_key, s.type AS s_type,
                   o.name AS o_name, o.key AS o_key, o.type AS o_type
        """, uid=uid).single()

        s_name = result["s_name"] if result else None
        s_key  = result["s_key"]  if result else None
        s_type = result["s_type"] if result else None
        o_name = result["o_name"] if result else None
        o_key  = result["o_key"]  if result else None
        o_type = result["o_type"] if result else None

        print(f"  Found via relationship: s={s_name}, o={o_name}")

        if s_name is None:
            # Try to reconstruct from stored predicate/document context
            # Look up if there's a fallback — e.g. the entity might have been deleted
            # We use the predicate and document_id as placeholder context
            ctx = s.run("""
                MATCH (tr:Triple {triple_uid: $uid})
                RETURN tr.predicate AS pred, tr.document_id AS doc_id, tr.chunk_id AS chunk_id
            """, uid=uid).single()
            pred = ctx["pred"] if ctx else "unknown"
            doc  = ctx["doc_id"] if ctx else "unknown"
            print(f"  Subject entity not found via relationship — using placeholder")
            s_name = f"[entity:{pred}:{doc}]"
            s_key  = s_name
            s_type = "Concept"
        if o_name is None:
            ctx = s.run("""
                MATCH (tr:Triple {triple_uid: $uid})
                RETURN tr.predicate AS pred, tr.document_id AS doc_id
            """, uid=uid).single()
            pred = ctx["pred"] if ctx else "unknown"
            doc  = ctx["doc_id"] if ctx else "unknown"
            print(f"  Object entity not found via relationship — using placeholder")
            o_name = f"[entity:{pred}:{doc}]"
            o_key  = o_name
            o_type = "Concept"

        # Update the Triple node
        s.run("""
            MATCH (tr:Triple {triple_uid: $uid})
            SET tr.s_name = $s_name,
                tr.s_key  = $s_key,
                tr.s_type = $s_type,
                tr.o_name = $o_name,
                tr.o_key  = $o_key,
                tr.o_type = $o_type
        """, uid=uid, s_name=s_name, s_key=s_key, s_type=s_type,
           o_name=o_name, o_key=o_key, o_type=o_type)
        print(f"  Set: s_name={s_name}, o_name={o_name}")
        fixed += 1

    print(f"\nUpdated {fixed} triples.")

    # Verify
    remaining = s.run("""
        MATCH (tr:Triple)
        WHERE tr.s_name IS NULL OR tr.o_name IS NULL
        RETURN count(tr) AS remaining
    """).single()
    print(f"Remaining uncached triples: {remaining['remaining']}")

driver.close()
print("\nDone.")
