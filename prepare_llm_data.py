"""
Prepare text data for LLM training.

This script:
1. Loads CSV data with text descriptions
2. Matches with structure data IDs
3. Tokenizes text using GPT-2 tokenizer
4. Saves tokenized data to disk
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
from transformers import GPT2Tokenizer
from tqdm import tqdm
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Label mapping for mp_ordering
ORDERING_LABELS = {
    0: "NM",      # Non-magnetic
    1: "FM",      # Ferromagnetic
    2: "AFM",     # Antiferromagnetic
    3: "FiM",     # Ferrimagnetic
}
NUM_LABELS = 4


def load_prepared_data(filepath: str) -> list:
    """Load prepared JSON data file with structure data."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data


def load_csv_data(csv_path: str, text_column: str = 'text', 
                  label_column: str = 'mp_ordering',
                  id_column: str = 'mp_id') -> pd.DataFrame:
    """Load CSV file with text descriptions."""
    logger.info(f"Loading CSV data from {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Filter valid numeric labels (0, 1, 2, 3)
    df = df[df[label_column].apply(lambda x: str(x).isdigit())]
    df[label_column] = df[label_column].astype(int)
    df = df[df[label_column].isin([0, 1, 2, 3])]
    
    # Remove samples with empty text
    df = df[df[text_column].notna()]
    df = df[df[text_column].str.strip() != '']
    
    logger.info(f"Loaded {len(df)} valid samples from CSV")
    return df


def match_datasets(csv_df: pd.DataFrame, 
                   structure_data: list,
                   id_column: str = 'mp_id') -> tuple:
    """Match samples between CSV and structure data by ID."""
    # Create lookup for structure data
    structure_lookup = {item['id']: item for item in structure_data}
    
    # Filter CSV to only include samples with matching structures
    matched_ids = set(csv_df[id_column].values) & set(structure_lookup.keys())
    logger.info(f"Found {len(matched_ids)} matched samples")
    
    # Filter both datasets
    matched_csv = csv_df[csv_df[id_column].isin(matched_ids)].copy()
    matched_csv = matched_csv.reset_index(drop=True)
    
    return matched_csv, matched_ids


def tokenize_and_save(df: pd.DataFrame, tokenizer, output_path: str,
                      text_column: str, label_column: str, id_column: str,
                      max_length: int = 512):
    """Tokenize texts and save to disk."""
    logger.info(f"Tokenizing {len(df)} samples...")
    
    input_ids_list = []
    attention_mask_list = []
    labels_list = []
    ids_list = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Tokenizing"):
        text = str(row[text_column])
        label = int(row[label_column])
        sample_id = row[id_column]
        
        encoding = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        input_ids_list.append(encoding['input_ids'].squeeze(0))
        attention_mask_list.append(encoding['attention_mask'].squeeze(0))
        labels_list.append(label)
        ids_list.append(sample_id)
    
    # Stack tensors
    input_ids = torch.stack(input_ids_list)
    attention_mask = torch.stack(attention_mask_list)
    labels = torch.tensor(labels_list, dtype=torch.long)
    
    # Save to disk
    data = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
        'ids': ids_list,
    }
    
    torch.save(data, output_path)
    logger.info(f"Saved tokenized data to {output_path}")
    
    return len(ids_list)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Prepare text data for LLM training'
    )
    
    # Data arguments
    parser.add_argument('--input_csv', type=str,
                        default='output/chemnlp_0_210579_skip_none.csv',
                        help='Path to input CSV file with text descriptions')
    parser.add_argument('--train_structures', type=str,
                        default='../alignn_test/prepared_data_merged/train_data.json',
                        help='Path to training structure JSON')
    parser.add_argument('--val_structures', type=str,
                        default='../alignn_test/prepared_data_merged/val_data.json',
                        help='Path to validation structure JSON')
    parser.add_argument('--test_structures', type=str,
                        default='../alignn_test/prepared_data_merged/test_data.json',
                        help='Path to test structure JSON')
    parser.add_argument('--text_column', type=str, default='text',
                        help='Name of text column in CSV')
    parser.add_argument('--label_column', type=str, default='mp_ordering',
                        help='Name of label column in CSV')
    parser.add_argument('--id_column', type=str, default='mp_id',
                        help='Name of ID column in CSV')
    
    # Tokenizer arguments
    parser.add_argument('--gpt2_model', type=str, default='gpt2',
                        help='GPT-2 model name (gpt2, gpt2-medium, etc.)')
    parser.add_argument('--max_length', type=int, default=512,
                        help='Maximum text sequence length')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='llm_gnn_data',
                        help='Output directory for prepared data')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load CSV data
    csv_df = load_csv_data(
        args.input_csv,
        args.text_column,
        args.label_column,
        args.id_column
    )
    
    # Load structure data to get IDs
    logger.info("Loading structure data...")
    train_structures = load_prepared_data(args.train_structures)
    val_structures = load_prepared_data(args.val_structures)
    test_structures = load_prepared_data(args.test_structures)
    all_structures = train_structures + val_structures + test_structures
    
    # Match CSV with structures
    matched_csv, matched_ids = match_datasets(csv_df, all_structures, args.id_column)
    
    # Get IDs from prepared data splits
    train_ids = set(item['id'] for item in train_structures)
    val_ids = set(item['id'] for item in val_structures)
    test_ids = set(item['id'] for item in test_structures)
    
    # Split CSV data based on structure splits
    train_csv = matched_csv[matched_csv[args.id_column].isin(train_ids)].reset_index(drop=True)
    val_csv = matched_csv[matched_csv[args.id_column].isin(val_ids)].reset_index(drop=True)
    test_csv = matched_csv[matched_csv[args.id_column].isin(test_ids)].reset_index(drop=True)
    
    logger.info(f"Train: {len(train_csv)}, Val: {len(val_csv)}, Test: {len(test_csv)}")
    
    # Load tokenizer
    logger.info(f"Loading tokenizer: {args.gpt2_model}")
    tokenizer = GPT2Tokenizer.from_pretrained(args.gpt2_model)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Tokenize and save each split
    for split_name, split_df in [('train', train_csv), ('val', val_csv), ('test', test_csv)]:
        output_path = os.path.join(args.output_dir, f'{split_name}_llm_data.pt')
        n_samples = tokenize_and_save(
            split_df, tokenizer, output_path,
            args.text_column, args.label_column, args.id_column,
            args.max_length
        )
        logger.info(f"Saved {n_samples} {split_name} samples")
    
    # Save metadata
    metadata = {
        'gpt2_model': args.gpt2_model,
        'max_length': args.max_length,
        'text_column': args.text_column,
        'label_column': args.label_column,
        'id_column': args.id_column,
        'num_labels': NUM_LABELS,
        'label_mapping': ORDERING_LABELS,
        'train_samples': len(train_csv),
        'val_samples': len(val_csv),
        'test_samples': len(test_csv),
        'timestamp': datetime.now().isoformat(),
    }
    
    metadata_path = os.path.join(args.output_dir, 'llm_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved metadata to {metadata_path}")
    
    logger.info("\nLLM data preparation completed!")
    logger.info(f"Output directory: {args.output_dir}")
    
    return metadata


if __name__ == '__main__':
    main()
