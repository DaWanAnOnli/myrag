import pandas as pd
from neo4j import GraphDatabase
import hashlib
from typing import List, Dict
from collections import defaultdict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
INPUT_CSV = "../dataset/5_amendments/csv/passive_regulations_relationships.csv"
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER= "neo4j"
NEO4J_PASSWORD= "@ik4nkus"
BATCH_SIZE = 1000

# Relationship type mapping (CSV value -> Neo4j relationship type with AMD_ prefix)
RELATIONSHIP_TYPE_MAPPING = {
    'diubah_dengan': 'AMD_DIUBAH_DENGAN',
    'dicabut_dengan': 'AMD_DICABUT_DENGAN',
    'diubah_sebagian_dengan': 'AMD_DIUBAH_SEBAGIAN_DENGAN',
    'dicabut_sebagian_dengan': 'AMD_DICABUT_SEBAGIAN_DENGAN'
}


class Neo4jAmendmentLoader:
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        
    def close(self):
        self.driver.close()
    
    def create_constraints(self):
        """Create uniqueness constraints for AMD_UndangUndang nodes"""
        constraints = [
            """
            CREATE CONSTRAINT amd_undang_undang_key_unique IF NOT EXISTS
            FOR (u:AMD_UndangUndang) REQUIRE u.key IS UNIQUE
            """,
            """
            CREATE CONSTRAINT amd_triple_uid_unique IF NOT EXISTS
            FOR (t:AMD_Triple) REQUIRE t.triple_uid IS UNIQUE
            """
        ]
        
        with self.driver.session() as session:
            for constraint in constraints:
                try:
                    session.run(constraint)
                    logger.info(f"Constraint created or already exists")
                except Exception as e:
                    logger.error(f"Error creating constraint: {e}")
    
    def parse_undang_undang_id(self, uu_id: str) -> Dict[str, any]:
        """
        Parse undang-undang identifier like '16_2025' into components
        Returns: {key, number, year}
        """
        parts = uu_id.split('_')
        if len(parts) == 2:
            return {
                'key': f"AMD_{uu_id}",
                'number': int(parts[0]),
                'year': int(parts[1]),
                'original_id': uu_id
            }
        else:
            logger.warning(f"Unexpected format for ID: {uu_id}")
            return {
                'key': f"AMD_{uu_id}",
                'original_id': uu_id
            }
    
    def generate_triple_uid(self, source_key: str, rel_type: str, target_key: str) -> str:
        """Generate unique identifier for triple"""
        triple_string = f"{source_key}|{rel_type}|{target_key}"
        return hashlib.sha256(triple_string.encode()).hexdigest()
    
    def load_batch_for_relationship_type(self, batch: List[Dict], rel_type: str):
        """Load a batch of relationships of a specific type into Neo4j"""
        
        # Create a Cypher query specific to this relationship type
        cypher_queries = {
            'AMD_DIUBAH_DENGAN': """
                UNWIND $batch AS row
                
                MERGE (source:AMD_UndangUndang {key: row.source.key})
                ON CREATE SET 
                    source.number = row.source.number,
                    source.year = row.source.year,
                    source.original_id = row.source.original_id,
                    source.created_at = timestamp()
                
                MERGE (target:AMD_UndangUndang {key: row.target.key})
                ON CREATE SET 
                    target.number = row.target.number,
                    target.year = row.target.year,
                    target.original_id = row.target.original_id,
                    target.created_at = timestamp()
                
                MERGE (source)-[r:AMD_DIUBAH_DENGAN]->(target)
                ON CREATE SET 
                    r.triple_uid = row.triple_uid, 
                    r.created_at = timestamp()
                
                MERGE (t:AMD_Triple {triple_uid: row.triple_uid})
                ON CREATE SET
                    t.source_key = row.source.key,
                    t.relationship_type = row.relationship_type,
                    t.target_key = row.target.key,
                    t.created_at = timestamp()
                
                RETURN count(*) as processed
            """,
            'AMD_DICABUT_DENGAN': """
                UNWIND $batch AS row
                
                MERGE (source:AMD_UndangUndang {key: row.source.key})
                ON CREATE SET 
                    source.number = row.source.number,
                    source.year = row.source.year,
                    source.original_id = row.source.original_id,
                    source.created_at = timestamp()
                
                MERGE (target:AMD_UndangUndang {key: row.target.key})
                ON CREATE SET 
                    target.number = row.target.number,
                    target.year = row.target.year,
                    target.original_id = row.target.original_id,
                    target.created_at = timestamp()
                
                MERGE (source)-[r:AMD_DICABUT_DENGAN]->(target)
                ON CREATE SET 
                    r.triple_uid = row.triple_uid, 
                    r.created_at = timestamp()
                
                MERGE (t:AMD_Triple {triple_uid: row.triple_uid})
                ON CREATE SET
                    t.source_key = row.source.key,
                    t.relationship_type = row.relationship_type,
                    t.target_key = row.target.key,
                    t.created_at = timestamp()
                
                RETURN count(*) as processed
            """,
            'AMD_DIUBAH_SEBAGIAN_DENGAN': """
                UNWIND $batch AS row
                
                MERGE (source:AMD_UndangUndang {key: row.source.key})
                ON CREATE SET 
                    source.number = row.source.number,
                    source.year = row.source.year,
                    source.original_id = row.source.original_id,
                    source.created_at = timestamp()
                
                MERGE (target:AMD_UndangUndang {key: row.target.key})
                ON CREATE SET 
                    target.number = row.target.number,
                    target.year = row.target.year,
                    target.original_id = row.target.original_id,
                    target.created_at = timestamp()
                
                MERGE (source)-[r:AMD_DIUBAH_SEBAGIAN_DENGAN]->(target)
                ON CREATE SET 
                    r.triple_uid = row.triple_uid, 
                    r.created_at = timestamp()
                
                MERGE (t:AMD_Triple {triple_uid: row.triple_uid})
                ON CREATE SET
                    t.source_key = row.source.key,
                    t.relationship_type = row.relationship_type,
                    t.target_key = row.target.key,
                    t.created_at = timestamp()
                
                RETURN count(*) as processed
            """,
            'AMD_DICABUT_SEBAGIAN_DENGAN': """
                UNWIND $batch AS row
                
                MERGE (source:AMD_UndangUndang {key: row.source.key})
                ON CREATE SET 
                    source.number = row.source.number,
                    source.year = row.source.year,
                    source.original_id = row.source.original_id,
                    source.created_at = timestamp()
                
                MERGE (target:AMD_UndangUndang {key: row.target.key})
                ON CREATE SET 
                    target.number = row.target.number,
                    target.year = row.target.year,
                    target.original_id = row.target.original_id,
                    target.created_at = timestamp()
                
                MERGE (source)-[r:AMD_DICABUT_SEBAGIAN_DENGAN]->(target)
                ON CREATE SET 
                    r.triple_uid = row.triple_uid, 
                    r.created_at = timestamp()
                
                MERGE (t:AMD_Triple {triple_uid: row.triple_uid})
                ON CREATE SET
                    t.source_key = row.source.key,
                    t.relationship_type = row.relationship_type,
                    t.target_key = row.target.key,
                    t.created_at = timestamp()
                
                RETURN count(*) as processed
            """
        }
        
        cypher = cypher_queries.get(rel_type)
        if not cypher:
            logger.error(f"Unknown relationship type: {rel_type}")
            return None
        
        with self.driver.session() as session:
            try:
                result = session.run(cypher, batch=batch)
                summary = result.consume()
                return summary.counters
            except Exception as e:
                logger.error(f"Error loading batch for {rel_type}: {e}")
                raise
    
    def load_amendments(self, csv_file: str):
        """Main method to load all amendments from CSV"""
        logger.info(f"Reading CSV file: {csv_file}")
        df = pd.read_csv(csv_file)
        
        logger.info(f"Total rows to process: {len(df)}")
        
        # Group data by relationship type for efficient processing
        batches_by_type = defaultdict(list)
        
        for idx, row in df.iterrows():
            source_info = self.parse_undang_undang_id(row['source'])
            target_info = self.parse_undang_undang_id(row['target'])
            relationship_type = RELATIONSHIP_TYPE_MAPPING.get(
                row['relationship'], 
                f"AMD_{row['relationship'].upper()}"
            )
            
            triple_uid = self.generate_triple_uid(
                source_info['key'],
                relationship_type,
                target_info['key']
            )
            
            row_data = {
                'source': source_info,
                'target': target_info,
                'relationship_type': relationship_type,
                'triple_uid': triple_uid
            }
            
            batches_by_type[relationship_type].append(row_data)
        
        # Process batches by relationship type
        total_processed = 0
        total_nodes_created = 0
        total_rels_created = 0
        
        for rel_type, rows in batches_by_type.items():
            logger.info(f"Processing {len(rows)} rows for relationship type: {rel_type}")
            
            # Process in batches of BATCH_SIZE
            for i in range(0, len(rows), BATCH_SIZE):
                batch = rows[i:i + BATCH_SIZE]
                logger.info(f"  Batch {i//BATCH_SIZE + 1}: Processing {len(batch)} rows")
                
                counters = self.load_batch_for_relationship_type(batch, rel_type)
                
                if counters:
                    total_nodes_created += counters.nodes_created
                    total_rels_created += counters.relationships_created
                    logger.info(f"  Nodes created: {counters.nodes_created}, "
                               f"Relationships created: {counters.relationships_created}")
                
                total_processed += len(batch)
        
        logger.info(f"Total rows processed: {total_processed}")
        logger.info(f"Total nodes created: {total_nodes_created}")
        logger.info(f"Total relationships created: {total_rels_created}")
        
        # Get final statistics
        self.print_statistics()
    
    def print_statistics(self):
        """Print statistics about loaded data"""
        with self.driver.session() as session:
            # Count nodes
            node_count = session.run(
                "MATCH (u:AMD_UndangUndang) RETURN count(u) as count"
            ).single()['count']
            
            # Count relationships by type
            rel_stats = session.run("""
                MATCH ()-[r]->()
                WHERE type(r) STARTS WITH 'AMD_'
                RETURN type(r) as rel_type, count(r) as count
                ORDER BY count DESC
            """).data()
            
            # Count triples
            triple_count = session.run(
                "MATCH (t:AMD_Triple) RETURN count(t) as count"
            ).single()['count']
            
            logger.info("=" * 60)
            logger.info("LOADING COMPLETE - STATISTICS")
            logger.info("=" * 60)
            logger.info(f"Total AMD_UndangUndang nodes: {node_count}")
            logger.info(f"Total AMD_Triple nodes: {triple_count}")
            logger.info("Relationships by type:")
            for rel in rel_stats:
                logger.info(f"  {rel['rel_type']}: {rel['count']}")
            logger.info("=" * 60)


def main():
    loader = Neo4jAmendmentLoader(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    
    try:
        logger.info("Creating constraints...")
        loader.create_constraints()
        
        logger.info("Loading amendments...")
        loader.load_amendments(INPUT_CSV)
        
        logger.info("Process completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during loading: {e}")
        raise
    finally:
        loader.close()


if __name__ == "__main__":
    main()