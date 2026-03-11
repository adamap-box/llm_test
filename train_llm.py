"""
Train GPT-2 based model to predict mp_ordering from chemical text descriptions.

This script:
1. Loads text data from CSV file
2. Fine-tunes GPT-2 for sequence classification (4 classes: 0, 1, 2, 3)
3. Evaluates on train/val/test splits
4. Saves the trained model and metrics
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    GPT2Tokenizer,
    GPT2ForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
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


class ChemTextDataset(Dataset):
    """Dataset for chemical text classification."""
    
    def __init__(self, texts, labels, tokenizer, max_length=512):
        """
        Args:
            texts: List of text strings
            labels: List of integer labels
            tokenizer: GPT-2 tokenizer
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenize with padding and truncation
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def load_and_preprocess_data(csv_path, text_column='text', label_column='mp_ordering'):
    """Load CSV and filter valid samples."""
    logger.info(f"Loading data from {csv_path}")
    df = pd.read_csv(csv_path)
    
    logger.info(f"Total samples: {len(df)}")
    
    # Filter valid numeric labels (0, 1, 2, 3)
    df = df[df[label_column].apply(lambda x: str(x).isdigit())]
    df[label_column] = df[label_column].astype(int)
    df = df[df[label_column].isin([0, 1, 2, 3])]
    
    # Remove samples with empty text
    df = df[df[text_column].notna()]
    df = df[df[text_column].str.strip() != '']
    
    logger.info(f"Valid samples after filtering: {len(df)}")
    
    # Log class distribution
    class_dist = df[label_column].value_counts().sort_index()
    for label, count in class_dist.items():
        logger.info(f"  Class {label} ({ORDERING_LABELS.get(label, 'Unknown')}): {count}")
    
    return df


def create_data_splits(df, text_column, label_column, test_size=0.1, val_size=0.1, random_state=42):
    """Create train/val/test splits with stratification."""
    texts = df[text_column].tolist()
    labels = df[label_column].tolist()
    
    # First split: train+val vs test
    train_val_texts, test_texts, train_val_labels, test_labels = train_test_split(
        texts, labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels
    )
    
    # Second split: train vs val
    val_ratio = val_size / (1 - test_size)
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_val_texts, train_val_labels,
        test_size=val_ratio,
        random_state=random_state,
        stratify=train_val_labels
    )
    
    logger.info(f"Data splits - Train: {len(train_texts)}, Val: {len(val_texts)}, Test: {len(test_texts)}")
    
    return {
        'train': (train_texts, train_labels),
        'val': (val_texts, val_labels),
        'test': (test_texts, test_labels)
    }


def compute_class_weights(labels):
    """Compute class weights for imbalanced data."""
    class_counts = np.bincount(labels, minlength=NUM_LABELS)
    total = len(labels)
    weights = total / (NUM_LABELS * class_counts + 1e-6)
    # Normalize weights
    weights = weights / weights.sum() * NUM_LABELS
    return torch.tensor(weights, dtype=torch.float32)


def train_epoch(model, dataloader, optimizer, scheduler, device, class_weights=None):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    # Loss function with class weights
    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    else:
        criterion = nn.CrossEntropyLoss()
    
    progress_bar = tqdm(dataloader, desc="Training")
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        
        logits = outputs.logits
        loss = criterion(logits, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
        
        progress_bar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    return avg_loss, accuracy, f1


def evaluate(model, dataloader, device, class_weights=None):
    """Evaluate the model."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    else:
        criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            
            logits = outputs.logits
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    return avg_loss, accuracy, f1, all_preds, all_labels


def parse_args():
    parser = argparse.ArgumentParser(description='Train GPT-2 for mp_ordering prediction')
    
    # Data arguments
    parser.add_argument('--input_csv', type=str,
                        default='output/chemnlp_0_210579_skip_none.csv',
                        help='Path to input CSV file')
    parser.add_argument('--text_column', type=str, default='text',
                        help='Name of text column')
    parser.add_argument('--label_column', type=str, default='mp_ordering',
                        help='Name of label column')
    
    # Model arguments
    parser.add_argument('--model_name', type=str, default='gpt2',
                        help='Pre-trained model name (gpt2, gpt2-medium, etc.)')
    parser.add_argument('--max_length', type=int, default=512,
                        help='Maximum sequence length')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                        help='Learning rate')
    parser.add_argument('--warmup_ratio', type=float, default=0.1,
                        help='Warmup ratio for scheduler')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('--use_class_weights', action='store_true',
                        help='Use class weights for imbalanced data')
    
    # Split arguments
    parser.add_argument('--test_size', type=float, default=0.1,
                        help='Test set ratio')
    parser.add_argument('--val_size', type=float, default=0.1,
                        help='Validation set ratio')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random seed')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='llm_output',
                        help='Output directory for model and results')
    parser.add_argument('--save_model', action='store_true',
                        help='Save the trained model')
    
    # Device arguments
    parser.add_argument('--cpu', action='store_true',
                        help='Use CPU only')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set device
    if args.cpu:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load and preprocess data
    df = load_and_preprocess_data(
        args.input_csv,
        args.text_column,
        args.label_column
    )
    
    # Create data splits
    splits = create_data_splits(
        df, args.text_column, args.label_column,
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.random_state
    )
    
    # Load tokenizer
    logger.info(f"Loading tokenizer: {args.model_name}")
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_name)
    
    # GPT-2 doesn't have a pad token by default, use eos_token
    tokenizer.pad_token = tokenizer.eos_token
    
    # Create datasets
    train_dataset = ChemTextDataset(
        splits['train'][0], splits['train'][1],
        tokenizer, args.max_length
    )
    val_dataset = ChemTextDataset(
        splits['val'][0], splits['val'][1],
        tokenizer, args.max_length
    )
    test_dataset = ChemTextDataset(
        splits['test'][0], splits['test'][1],
        tokenizer, args.max_length
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=0
    )
    
    # Calculate class weights
    class_weights = None
    if args.use_class_weights:
        class_weights = compute_class_weights(splits['train'][1])
        logger.info(f"Class weights: {class_weights.tolist()}")
    
    # Load model
    logger.info(f"Loading model: {args.model_name}")
    model = GPT2ForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=NUM_LABELS
    )
    
    # Set pad token id in model config
    model.config.pad_token_id = tokenizer.pad_token_id
    
    model.to(device)
    
    # Setup optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    logger.info(f"Total training steps: {total_steps}, Warmup steps: {warmup_steps}")
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'train_f1': [],
        'val_loss': [],
        'val_acc': [],
        'val_f1': [],
    }
    
    best_val_f1 = 0
    best_model_state = None
    
    # Training loop
    logger.info("Starting training...")
    for epoch in range(args.epochs):
        logger.info(f"\n{'='*50}")
        logger.info(f"Epoch {epoch + 1}/{args.epochs}")
        logger.info(f"{'='*50}")
        
        # Train
        train_loss, train_acc, train_f1 = train_epoch(
            model, train_loader, optimizer, scheduler,
            device, class_weights
        )
        
        # Validate
        val_loss, val_acc, val_f1, _, _ = evaluate(
            model, val_loader, device, class_weights
        )
        
        # Log metrics
        logger.info(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
        logger.info(f"Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_f1'].append(train_f1)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        
        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_state = model.state_dict().copy()
            logger.info(f"New best model saved! Val F1: {val_f1:.4f}")
    
    # Load best model for final evaluation
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Final evaluation on test set
    logger.info("\n" + "="*50)
    logger.info("Final Evaluation on Test Set")
    logger.info("="*50)
    
    test_loss, test_acc, test_f1, test_preds, test_labels = evaluate(
        model, test_loader, device, class_weights
    )
    
    logger.info(f"Test - Loss: {test_loss:.4f}, Acc: {test_acc:.4f}, F1: {test_f1:.4f}")
    
    # Detailed classification report
    target_names = [f"{i} ({ORDERING_LABELS[i]})" for i in range(NUM_LABELS)]
    report = classification_report(
        test_labels, test_preds,
        target_names=target_names,
        digits=4
    )
    logger.info(f"\nClassification Report:\n{report}")
    
    # Confusion matrix
    cm = confusion_matrix(test_labels, test_preds)
    logger.info(f"\nConfusion Matrix:\n{cm}")
    
    # Save results
    results = {
        'args': vars(args),
        'history': history,
        'test_metrics': {
            'loss': test_loss,
            'accuracy': test_acc,
            'f1_weighted': test_f1,
        },
        'classification_report': classification_report(
            test_labels, test_preds,
            target_names=target_names,
            output_dict=True
        ),
        'confusion_matrix': cm.tolist(),
        'timestamp': datetime.now().isoformat(),
    }
    
    results_path = os.path.join(args.output_dir, 'training_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {results_path}")
    
    # Save history
    history_path = os.path.join(args.output_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    logger.info(f"History saved to {history_path}")
    
    # Save model
    if args.save_model:
        model_dir = os.path.join(args.output_dir, 'model')
        os.makedirs(model_dir, exist_ok=True)
        model.save_pretrained(model_dir)
        tokenizer.save_pretrained(model_dir)
        logger.info(f"Model saved to {model_dir}")
    
    logger.info("\nTraining completed!")
    
    return results


if __name__ == '__main__':
    main()
