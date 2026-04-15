# -*- coding: utf-8 -*-
# Verification script — run ONCE to confirm the Neo4j graph is ready
# for the optimized GraphRAG retrieval pipeline.
#
# Usage: python 3b_v_neo4j_graphrag_retrieval_verify.py
#
# Expected output:
#   - Uncached triples: 0
#   - All indexes ONLINE

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
print("GraphRAG Retrieval Optimization — DB Verification")
print("=" * 60)

try:
    with driver.session() as s:
        # 1. Check for uncached triples
        print("\n[1] Checking for uncached Triple nodes...")
        result = s.run("""
            MATCH (tr:Triple)
            WHERE tr.s_name IS NULL AND tr.o_name IS NULL
            RETURN count(tr) AS uncached_triples
        """).single()
        uncached = result["uncached_triples"] if result else -1
        print(f"    Triples with both s_name AND o_name NULL: {uncached}")
        # Also check individually
        result2 = s.run("""
            MATCH (tr:Triple)
            WHERE tr.s_name IS NULL OR tr.o_name IS NULL
            RETURN count(tr) AS partial_uncached
        """).single()
        partial = result2["partial_uncached"] if result2 else -1
        print(f"    Triples with either s_name OR o_name NULL: {partial}")
        if uncached == 0 and partial == 0:
            print("    PASS — all triples have cached s_name/o_name properties")
        elif uncached == 0:
            print("    PASS — no triples have both s_name AND o_name NULL (OK)")
        else:
            print(f"    WARN — {uncached} triples missing cached properties!")
            print("    Run 3b_vi_neo4j_fix_2_uncached_triples.py to fix.")

        # 2. Check indexes are online
        print("\n[2] Checking indexes status...")
        indexes = list(s.run("SHOW INDEXES").data())
        online = [i for i in indexes if i.get("state") == "ONLINE"]
        offline = [i for i in indexes if i.get("state") != "ONLINE"]
        print(f"    Total indexes: {len(indexes)}")
        print(f"    ONLINE: {len(online)}")
        print(f"    Other state: {len(offline)}")
        for idx in indexes:
            status = "ONLINE" if idx["state"] == "ONLINE" else f"STATE={idx['state']}"
            print(f"    [{status}] {idx['name']} — {idx.get('labelsOrTypes', idx.get('relationshipType', '?'))}")
        if offline:
            print("    WARN — some indexes are not ONLINE. Retrieval may be slow.")
            print("    Wait for indexes to come ONLINE or recreate them.")

        # 3. Key retrieval-specific indexes
        print("\n[3] Verifying retrieval-critical indexes...")
        retrieval_indexes = {
            "triple_vec": False,
            "document_vec": False,
            "content_vec": False,
            "expression_vec": False,
        }
        for idx in indexes:
            name = idx.get("name", "").lower()
            for key in retrieval_indexes:
                if key in name and idx.get("state") == "ONLINE":
                    retrieval_indexes[key] = True
        for name, found in retrieval_indexes.items():
            status = "FOUND" if found else "MISSING"
            print(f"    {name}: {status}")
        missing = [k for k, v in retrieval_indexes.items() if not v]
        if missing:
            print(f"    WARN — missing vector indexes: {missing}")
        else:
            print("    PASS — all vector indexes present and ONLINE")

        # 4. Summary
        print("\n" + "=" * 60)
        if uncached == 0 and not offline and not missing:
            print("VERIFICATION PASSED — graph is ready for optimized retrieval")
        else:
            print("VERIFICATION WARNINGS — see above")
        print("=" * 60)

finally:
    driver.close()
