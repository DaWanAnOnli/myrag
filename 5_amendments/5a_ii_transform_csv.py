import pandas as pd

# Constants
INPUT_FILE = "../dataset/5_amendments/csv/regulations_relationships.csv"
OUTPUT_FILE = "../dataset/5_amendments/csv/transformed_regulations_relationships.csv"

# Mapping between active and passive relationships
ACTIVE_TO_PASSIVE = {
    'mencabut': 'dicabut_dengan',
    'mencabut_sebagian': 'dicabut_sebagian_dengan',
    'mengubah': 'diubah_dengan',
    'mengubah_sebagian': 'diubah_sebagian_dengan'
}

PASSIVE_TO_ACTIVE = {v: k for k, v in ACTIVE_TO_PASSIVE.items()}

def main():
    # Read the CSV file
    df = pd.read_csv(INPUT_FILE)
    
    # Separate active and passive relationships
    active_relationships = df[df['relationship'].isin(ACTIVE_TO_PASSIVE.keys())].copy()
    passive_relationships = df[df['relationship'].isin(PASSIVE_TO_ACTIVE.keys())].copy()
    
    # Create result lists
    matched_rows = []
    unmatched_active_rows = []
    matched_passive_indices = set()
    
    # Process active relationships and find their passive counterparts
    for idx, active_row in active_relationships.iterrows():
        source = active_row['source']
        rel_active = active_row['relationship']
        target = active_row['target']
        
        # Expected passive relationship
        rel_passive = ACTIVE_TO_PASSIVE[rel_active]
        
        # Find matching passive relationship (where source and target are swapped)
        matching_passive = passive_relationships[
            (passive_relationships['source'] == target) &
            (passive_relationships['relationship'] == rel_passive) &
            (passive_relationships['target'] == source)
        ]
        
        if not matching_passive.empty:
            # Found a match
            passive_row = matching_passive.iloc[0]
            matched_passive_indices.add(matching_passive.index[0])
            
            matched_rows.append({
                'source_active': source,
                'relationship_active': rel_active,
                'target_active': target,
                'source_passive': passive_row['source'],
                'relationship_passive': passive_row['relationship'],
                'target_passive': passive_row['target']
            })
        else:
            # No match found - add to unmatched active
            unmatched_active_rows.append({
                'source_active': source,
                'relationship_active': rel_active,
                'target_active': target,
                'source_passive': '',
                'relationship_passive': '',
                'target_passive': ''
            })
    
    # Add unmatched passive relationships
    unmatched_passive_rows = []
    unmatched_passive = passive_relationships[~passive_relationships.index.isin(matched_passive_indices)]
    for _, passive_row in unmatched_passive.iterrows():
        unmatched_passive_rows.append({
            'source_active': '',
            'relationship_active': '',
            'target_active': '',
            'source_passive': passive_row['source'],
            'relationship_passive': passive_row['relationship'],
            'target_passive': passive_row['target']
        })
    
    # Combine all rows: matched first, then unmatched
    all_rows = matched_rows + unmatched_active_rows + unmatched_passive_rows
    
    # Create output DataFrame
    result_df = pd.DataFrame(all_rows)
    
    # Save to CSV
    result_df.to_csv(OUTPUT_FILE, index=False)
    
    print(f"Transformation complete!")
    print(f"Input rows: {len(df)}")
    print(f"Output rows: {len(result_df)}")
    print(f"Matched pairs: {len(matched_rows)}")
    print(f"Unmatched active: {len(unmatched_active_rows)}")
    print(f"Unmatched passive: {len(unmatched_passive_rows)}")

if __name__ == "__main__":
    main()