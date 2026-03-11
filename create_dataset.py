"""
Split embeddings CSV into train, valid, and test sets based on IDs.
"""

import json
import pandas as pd
import os
from pathlib import Path


def split_embeddings():
    # Paths
    base_dir = Path(__file__).parent
    ids_path = base_dir.parent / "alignn_test" / "prepared_data_merged" / "ids_train_val_test.json"
    csv_path = base_dir / "output" / "embeddings_bert-base-uncased_chemnlp_0_210579_skip_none_210579.csv"
    output_dir = base_dir / "dataset"
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    
    # Load IDs
    print(f"Loading IDs from: {ids_path}")
    with open(ids_path, 'r') as f:
        ids = json.load(f)
    
    train_ids = set(ids['id_train'])
    val_ids = set(ids['id_val'])
    test_ids = set(ids['id_test'])
    
    print(f"ID counts: train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}")
    
    # Load embeddings CSV
    print(f"Loading embeddings from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Total rows in CSV: {len(df)}")
    
    # Get ID column (first column)
    id_col = df.columns[0]
    print(f"ID column: {id_col}")
    
    # Split dataframe
    train_df = df[df[id_col].isin(train_ids)]
    val_df = df[df[id_col].isin(val_ids)]
    test_df = df[df[id_col].isin(test_ids)]
    
    print(f"Split counts: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    
    # Save split files
    train_path = output_dir / "train.csv"
    val_path = output_dir / "valid.csv"
    test_path = output_dir / "test.csv"
    
    print(f"Saving train.csv...")
    train_df.to_csv(train_path, index=False)
    
    print(f"Saving valid.csv...")
    val_df.to_csv(val_path, index=False)
    
    print(f"Saving test.csv...")
    test_df.to_csv(test_path, index=False)
    
    print(f"\nDone! Files saved to: {output_dir}")
    print(f"  - train.csv: {len(train_df)} rows")
    print(f"  - valid.csv: {len(val_df)} rows")
    print(f"  - test.csv: {len(test_df)} rows")


if __name__ == "__main__":
    split_embeddings()
