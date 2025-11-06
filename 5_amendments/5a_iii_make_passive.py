import pandas as pd

# Constants
INPUT_FILE = "../dataset/5_amendments/csv/transformed_regulations_relationships.csv"
OUTPUT_FILE = "../dataset/5_amendments/csv/passive_regulations_relationships.csv"

# Mapping from active to passive relationships
ACTIVE_TO_PASSIVE = {
    'mencabut': 'dicabut_dengan',
    'mencabut_sebagian': 'dicabut_sebagian_dengan',
    'mengubah': 'diubah_dengan',
    'mengubah_sebagian': 'diubah_sebagian_dengan'
}

def main():
    # Read the 6-column CSV
    df = pd.read_csv(INPUT_FILE)
    
    result_rows = []
    
    for _, row in df.iterrows():
        # Check if active and passive columns are filled
        has_active = pd.notna(row['source_active']) and row['source_active'] != ''
        has_passive = pd.notna(row['source_passive']) and row['source_passive'] != ''
        
        if has_active and has_passive:
            # Complete pair: keep only passive
            result_rows.append({
                'source': row['source_passive'],
                'relationship': row['relationship_passive'],
                'target': row['target_passive']
            })
        elif has_active and not has_passive:
            # Active only: transform to passive by swapping source/target
            result_rows.append({
                'source': row['target_active'],  # Target becomes source
                'relationship': ACTIVE_TO_PASSIVE[row['relationship_active']],
                'target': row['source_active']  # Source becomes target
            })
        elif has_passive and not has_active:
            # Passive only: keep as is
            result_rows.append({
                'source': row['source_passive'],
                'relationship': row['relationship_passive'],
                'target': row['target_passive']
            })
    
    # Create DataFrame
    result_df = pd.DataFrame(result_rows)
    
    # Remove duplicates
    result_df_unique = result_df.drop_duplicates()
    
    # Save to CSV
    result_df_unique.to_csv(OUTPUT_FILE, index=False)
    
    print(f"Transformation complete!")
    print(f"Input rows: {len(df)}")
    print(f"Output rows (before deduplication): {len(result_df)}")
    print(f"Output rows (after deduplication): {len(result_df_unique)}")
    print(f"Duplicates removed: {len(result_df) - len(result_df_unique)}")

if __name__ == "__main__":
    main()