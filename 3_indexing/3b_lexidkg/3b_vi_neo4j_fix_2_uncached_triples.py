# -*- coding: utf-8 -*-
# Fix script — patches the 2 Triple nodes missing cached s_name/o_name properties.
# Run ONCE after 3b_iv_neo4j_lexidkg_post_indexing_optimize.txt.
#
# These 2 triples have NULL s_name/o_name because the APOC migration ran
# before the triple nodes were fully populated, or the entity nodes they
# reference were themselves missing properties at the time of the migration.
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
print("Fixing 2 uncached Triple nodes...")
print("=" * 60)

with driver.session() as s:
    # Identify the 2 uncached triples
    uncached = list(s.run("""
        MATCH (tr:Triple)
        WHERE tr.s_name IS NULL OR tr.o_name IS NULL
        RETURN tr.triple_uid AS uid, tr.subject AS sub_key, tr.object AS obj_key
    """))
    print(f"\nFound {len(uncached)} uncached triples:")
    for r in uncached:
        print(f"  - uid={r['uid'][:16]}... | s={r['sub_key']} | o={r['obj_key']}")

    # Fix each by re-joining to the actual entity nodes
    fixed = 0
    for r in uncached:
        uid = r["uid"]
        print(f"\nFixing {uid[:16]}...")
        # The triple's stored subject/object keys are in tr.subject / tr.object
        # Re-match the entities and set the cached properties
        result = s.run("""
            MATCH (tr:Triple {triple_uid: $uid})
            OPTIONAL MATCH (tr)-[:SUBJECT]->(s)
            OPTIONAL MATCH (tr)-[:OBJECT]->(o)
            SET tr.s_name = s.name,
                tr.s_key  = s.key,
                tr.s_type = s.type,
                tr.o_name = o.name,
                tr.o_key  = o.key,
                tr.o_type = o.type
            RETURN tr.triple_uid AS uid
        """, uid=uid).single()
        if result:
            print(f"  Fixed: {result['uid'][:16]}...")
            fixed += 1

    print(f"\nFixed {fixed}/{len(uncached)} triples.")

    # Verify
    remaining = s.run("""
        MATCH (tr:Triple)
        WHERE tr.s_name IS NULL OR tr.o_name IS NULL
        RETURN count(tr) AS remaining
    """).single()
    print(f"Remaining uncached triples: {remaining['remaining']}")

driver.close()
print("\nDone.")
