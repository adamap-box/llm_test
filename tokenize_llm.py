"""
Generate LLM embeddings from text representations in CSV file.
Based on preprocess.py from ALIGNN-BERT-TL.
"""

import numpy as np
import pandas as pd
import torch
from transformers import (
    AutoTokenizer, 
    GPT2Model, 
    BertModel, 
    BertTokenizerFast,
    AutoModelForCausalLM,
)
try:
    from transformers import BitsAndBytesConfig
    BITSANDBYTES_AVAILABLE = True
except ImportError:
    BITSANDBYTES_AVAILABLE = False
from tqdm import tqdm
import argparse
import logging
import os

logging.basicConfig(level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(description='Generate LLM embeddings from CSV text data')
    parser.add_argument('--input_csv', type=str, 
                        default='output/chemnlp_0_210579_skip_none.csv',
                        help='Path to input CSV file with text column')
    parser.add_argument('--text_column', type=str, default='text',
                        help='Name of the column containing text')
    parser.add_argument('--id_column', type=str, default='mp_id',
                        help='Name of the column containing sample IDs')
    parser.add_argument('--llm', type=str, default='gpt2',
                        help='Pre-trained LLM to use (gpt2, bert-base-uncased, matbert-base-cased, etc.)')
    parser.add_argument('--output_dir', type=str, default='output',
                        help='Directory to save output embeddings')
    parser.add_argument('--output_name', type=str, default=None,
                        help='Output filename (auto-generated if not specified)')
    parser.add_argument('--max_length', type=int, default=512,
                        help='Maximum token length for truncation')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for processing (currently only 1 supported)')
    parser.add_argument('--cpu', action='store_true',
                        help='Use CPU only')
    parser.add_argument('--matbert_path', type=str, default='/data/yll6162/tf_llm',
                        help='Path to MatBERT model if using matbert-base-cased')
    args = parser.parse_args()
    return args


def load_model_and_tokenizer(llm, device, matbert_path=None):
    """Load the specified LLM and tokenizer."""
    logging.info(f"Loading model: {llm}")
    
    if llm == "matbert-base-cased":
        model_path = os.path.join(matbert_path, llm)
        tokenizer = BertTokenizerFast.from_pretrained(model_path, do_lower_case=False)
        model = BertModel.from_pretrained(model_path)
    elif "gpt2" in llm.lower():
        tokenizer = AutoTokenizer.from_pretrained(llm)
        model = GPT2Model.from_pretrained(llm)
    elif "bert" in llm.lower():
        tokenizer = AutoTokenizer.from_pretrained(llm)
        model = BertModel.from_pretrained(llm)
    elif "opt" in llm.lower():
        tokenizer = AutoTokenizer.from_pretrained(llm)
        if BITSANDBYTES_AVAILABLE:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True, 
                llm_int8_enable_fp32_cpu_offload=True
            )
            model = AutoModelForCausalLM.from_pretrained(
                llm, 
                device_map='auto',
                quantization_config=quantization_config
            )
        else:
            logging.warning("BitsAndBytesConfig not available (requires PyTorch >= 2.4). Loading OPT without quantization.")
            model = AutoModelForCausalLM.from_pretrained(llm)
            model.to(device)
        model.eval()
        return model, tokenizer  # OPT handles its own device mapping when quantized
    else:
        # Generic fallback
        tokenizer = AutoTokenizer.from_pretrained(llm)
        model = GPT2Model.from_pretrained(llm)
    
    model.to(device)
    model.eval()
    return model, tokenizer


def extract_embeddings(model, tokenizer, texts, ids, device, max_length=512):
    """
    Extract mean-pooled last hidden state embeddings for each text.
    
    Args:
        model: The LLM model
        tokenizer: The tokenizer
        texts: List of text strings
        ids: List of sample IDs
        device: torch device
        max_length: Maximum token length
        
    Returns:
        embeddings: numpy array of shape (n_samples, embedding_dim)
        valid_ids: List of IDs for successfully processed samples
    """
    embeddings = []
    valid_ids = []
    max_token_length = model.config.max_position_embeddings
    logging.info(f"Model max position embeddings: {max_token_length}")
    
    for idx, (text, sample_id) in enumerate(tqdm(zip(texts, ids), total=len(texts), desc="Extracting embeddings")):
        try:
            # Handle NaN or empty text
            if pd.isna(text) or text == '':
                logging.warning(f"Skipping {sample_id}: empty or NaN text")
                continue
                
            # Tokenize
            inputs = tokenizer(
                text, 
                max_length=max_length, 
                truncation=True, 
                return_tensors="pt"
            ).to(device)
            
            # Check token length
            if len(inputs['input_ids'][0]) > max_token_length:
                logging.warning(f"Skipping {sample_id}: token length {len(inputs['input_ids'][0])} exceeds max {max_token_length}")
                continue
            
            # Extract embeddings
            with torch.no_grad():
                if device.type == 'cuda':
                    with torch.cuda.amp.autocast():
                        output = model(**inputs)
                else:
                    output = model(**inputs)
            
            # Mean pool the last hidden state
            last_hidden_state = output.last_hidden_state  # (1, seq_len, hidden_dim)
            embedding = last_hidden_state.mean(dim=1)  # (1, hidden_dim)
            
            # Move to CPU and convert to numpy
            if device.type == 'cuda':
                embedding = embedding.cpu().numpy().flatten()
            else:
                embedding = embedding.numpy().flatten()
            
            embeddings.append(embedding)
            valid_ids.append(sample_id)
            
        except Exception as e:
            logging.error(f"Error processing {sample_id}: {e}")
            continue
    
    if len(embeddings) > 0:
        embeddings = np.vstack(embeddings)
    else:
        embeddings = np.array([])
        
    return embeddings, valid_ids


def save_embeddings(embeddings, ids, output_path):
    """Save embeddings to CSV file."""
    df = pd.DataFrame(embeddings, index=ids)
    df.index.name = 'id'
    df.to_csv(output_path)
    logging.info(f"Saved {len(ids)} embeddings to {output_path}")


def main():
    args = parse_args()
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    logging.info(f"Using device: {device}")
    
    # Load input CSV
    logging.info(f"Loading input CSV: {args.input_csv}")
    df = pd.read_csv(args.input_csv)
    logging.info(f"Loaded {len(df)} samples")
    
    # Extract text and IDs
    texts = df[args.text_column].tolist()
    ids = df[args.id_column].tolist()
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(
        args.llm, 
        device, 
        matbert_path=args.matbert_path
    )
    
    # Extract embeddings
    embeddings, valid_ids = extract_embeddings(
        model, 
        tokenizer, 
        texts, 
        ids, 
        device, 
        max_length=args.max_length
    )
    
    logging.info(f"Generated embeddings for {len(valid_ids)} / {len(ids)} samples")
    logging.info(f"Embedding dimension: {embeddings.shape[1] if len(embeddings) > 0 else 'N/A'}")
    
    # Generate output path
    if args.output_name:
        output_filename = args.output_name
    else:
        input_basename = os.path.splitext(os.path.basename(args.input_csv))[0]
        llm_name = args.llm.replace('/', '_')
        output_filename = f"embeddings_{llm_name}_{input_basename}_{len(valid_ids)}.csv"
    
    output_path = os.path.join(args.output_dir, output_filename)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save embeddings
    save_embeddings(embeddings, valid_ids, output_path)
    
    logging.info("Done!")


if __name__ == "__main__":
    main()
