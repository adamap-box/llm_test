"""
Train MLP model combining GNN and LLM features to predict mp_ordering.

This script:
1. Loads GNN hidden features (256-dim) and LLM embeddings (768-dim)
2. Concatenates them by mp_id to form 1024-dim input
3. Trains an MLP classifier for 4-class mp_ordering prediction
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score, 
    f1_score, 
    classification_report,
    confusion_matrix
)
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CombinedFeaturesDataset(Dataset):
    """Dataset combining GNN and LLM features by mp_id."""
    
    def __init__(self, gnn_json_path: str, llm_csv_path: str):
        """
        Args:
            gnn_json_path: Path to GNN JSON file with hidden_features
            llm_csv_path: Path to LLM CSV file with embeddings
        """
        # Load GNN data
        logger.info(f"Loading GNN data from {gnn_json_path}")
        with open(gnn_json_path, 'r') as f:
            gnn_data = json.load(f)
        
        # Create dict: mp_id -> (hidden_features, mp_ordering)
        self.gnn_dict = {}
        for item in gnn_data:
            mp_id = item.get('mp_id') or item.get('id')
            hidden_features = item.get('hidden_features')
            mp_ordering = item.get('mp_ordering')
            if hidden_features is not None and mp_ordering is not None:
                self.gnn_dict[mp_id] = {
                    'hidden_features': np.array(hidden_features, dtype=np.float32),
                    'mp_ordering': int(mp_ordering)
                }
        
        logger.info(f"Loaded {len(self.gnn_dict)} GNN samples")
        
        # Load LLM data
        logger.info(f"Loading LLM data from {llm_csv_path}")
        llm_df = pd.read_csv(llm_csv_path)
        
        # Create dict: mp_id -> embedding
        self.llm_dict = {}
        id_col = 'id' if 'id' in llm_df.columns else llm_df.columns[0]
        embedding_cols = [col for col in llm_df.columns if col != id_col]
        
        for _, row in llm_df.iterrows():
            mp_id = row[id_col]
            embedding = row[embedding_cols].values.astype(np.float32)
            self.llm_dict[mp_id] = embedding
        
        logger.info(f"Loaded {len(self.llm_dict)} LLM samples")
        
        # Find common mp_ids
        common_ids = set(self.gnn_dict.keys()) & set(self.llm_dict.keys())
        self.mp_ids = sorted(list(common_ids))
        
        logger.info(f"Found {len(self.mp_ids)} common samples")
        
        if len(self.mp_ids) == 0:
            raise ValueError("No common mp_ids found between GNN and LLM datasets!")
        
        # Get feature dimensions
        sample_gnn = self.gnn_dict[self.mp_ids[0]]['hidden_features']
        sample_llm = self.llm_dict[self.mp_ids[0]]
        self.gnn_dim = len(sample_gnn)
        self.llm_dim = len(sample_llm)
        self.total_dim = self.gnn_dim + self.llm_dim
        
        logger.info(f"Feature dims - GNN: {self.gnn_dim}, LLM: {self.llm_dim}, Total: {self.total_dim}")
    
    def __len__(self):
        return len(self.mp_ids)
    
    def __getitem__(self, idx):
        mp_id = self.mp_ids[idx]
        
        # Get GNN features and target
        gnn_data = self.gnn_dict[mp_id]
        gnn_features = gnn_data['hidden_features']
        target = gnn_data['mp_ordering']
        
        # Get LLM features
        llm_features = self.llm_dict[mp_id]
        
        # Concatenate features
        combined_features = np.concatenate([gnn_features, llm_features])
        
        return (
            torch.tensor(combined_features, dtype=torch.float32),
            torch.tensor(target, dtype=torch.long),
            mp_id
        )


class MLP(nn.Module):
    """Multi-layer perceptron for classification."""
    
    def __init__(
        self, 
        input_dim: int, 
        hidden_dims: list = [512, 256, 128],
        num_classes: int = 4,
        dropout: float = 0.3
    ):
        super(MLP, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    
    for batch_features, batch_targets, _ in tqdm(dataloader, desc="Training", leave=False):
        batch_features = batch_features.to(device)
        batch_targets = batch_targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_features)
        loss = criterion(outputs, batch_targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * batch_features.size(0)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(batch_targets.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader.dataset)
    accuracy = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average='macro')
    
    return avg_loss, accuracy, f1


def evaluate(model, dataloader, criterion, device):
    """Evaluate the model."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    all_ids = []
    
    with torch.no_grad():
        for batch_features, batch_targets, batch_ids in tqdm(dataloader, desc="Evaluating", leave=False):
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device)
            
            outputs = model(batch_features)
            loss = criterion(outputs, batch_targets)
            
            total_loss += loss.item() * batch_features.size(0)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(batch_targets.cpu().numpy())
            all_ids.extend(batch_ids)
    
    avg_loss = total_loss / len(dataloader.dataset)
    accuracy = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average='macro')
    
    return avg_loss, accuracy, f1, all_preds, all_targets, all_ids


def main():
    parser = argparse.ArgumentParser(description='Train MLP on combined GNN+LLM features')
    
    # Data paths
    parser.add_argument('--gnn_dir', type=str, default='gnn_dataset',
                        help='Directory containing GNN JSON files')
    parser.add_argument('--llm_dir', type=str, default='llm_dataset',
                        help='Directory containing LLM CSV files')
    parser.add_argument('--output_dir', type=str, default='mlp_output',
                        help='Output directory for models and results')
    
    # Model parameters
    parser.add_argument('--hidden_dims', type=str, default='512,256,128',
                        help='Hidden layer dimensions (comma-separated)')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout rate')
    parser.add_argument('--num_classes', type=int, default=4,
                        help='Number of output classes')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay (L2 regularization)')
    parser.add_argument('--patience', type=int, default=15,
                        help='Early stopping patience')
    
    # Other
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of data loading workers')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Parse hidden dimensions
    hidden_dims = [int(d) for d in args.hidden_dims.split(',')]
    
    # Get script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    gnn_dir = os.path.join(script_dir, args.gnn_dir) if not os.path.isabs(args.gnn_dir) else args.gnn_dir
    llm_dir = os.path.join(script_dir, args.llm_dir) if not os.path.isabs(args.llm_dir) else args.llm_dir
    output_dir = os.path.join(script_dir, args.output_dir) if not os.path.isabs(args.output_dir) else args.output_dir
    
    # Load datasets
    logger.info("=" * 60)
    logger.info("Loading datasets...")
    logger.info("=" * 60)
    
    train_dataset = CombinedFeaturesDataset(
        gnn_json_path=os.path.join(gnn_dir, 'new_train_data.json'),
        llm_csv_path=os.path.join(llm_dir, 'train.csv')
    )
    
    val_dataset = CombinedFeaturesDataset(
        gnn_json_path=os.path.join(gnn_dir, 'new_val_data.json'),
        llm_csv_path=os.path.join(llm_dir, 'valid.csv')
    )
    
    test_dataset = CombinedFeaturesDataset(
        gnn_json_path=os.path.join(gnn_dir, 'new_test_data.json'),
        llm_csv_path=os.path.join(llm_dir, 'test.csv')
    )
    
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")
    logger.info(f"Test samples: {len(test_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # Create model
    input_dim = train_dataset.total_dim
    model = MLP(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        num_classes=args.num_classes,
        dropout=args.dropout
    )
    model.to(device)
    
    logger.info(f"\nModel architecture:")
    logger.info(f"Input dim: {input_dim}")
    logger.info(f"Hidden dims: {hidden_dims}")
    logger.info(f"Output classes: {args.num_classes}")
    logger.info(f"Dropout: {args.dropout}")
    logger.info(f"\n{model}")
    
    # Calculate class weights for imbalanced data
    train_targets = [train_dataset[i][1].item() for i in range(len(train_dataset))]
    class_counts = np.bincount(train_targets, minlength=args.num_classes)
    class_weights = 1.0 / (class_counts + 1e-6)
    class_weights = class_weights / class_weights.sum() * args.num_classes
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    
    logger.info(f"\nClass distribution in training set:")
    for i, count in enumerate(class_counts):
        logger.info(f"  Class {i}: {count} samples (weight: {class_weights[i]:.4f})")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )
    
    # Training loop
    logger.info("\n" + "=" * 60)
    logger.info("Starting training...")
    logger.info("=" * 60)
    
    best_val_f1 = 0.0
    best_epoch = 0
    patience_counter = 0
    history = {'train_loss': [], 'train_acc': [], 'train_f1': [],
               'val_loss': [], 'val_acc': [], 'val_f1': []}
    
    for epoch in range(1, args.epochs + 1):
        logger.info(f"\nEpoch {epoch}/{args.epochs}")
        
        # Train
        train_loss, train_acc, train_f1 = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validate
        val_loss, val_acc, val_f1, _, _, _ = evaluate(
            model, val_loader, criterion, device
        )
        
        # Update scheduler
        scheduler.step(val_f1)
        
        # Log metrics
        logger.info(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
        logger.info(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_f1'].append(train_f1)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        
        # Check for best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            patience_counter = 0
            
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_f1,
                'val_acc': val_acc,
            }, os.path.join(output_dir, 'best_model.pt'))
            logger.info(f"  New best model saved! (F1: {val_f1:.4f})")
        else:
            patience_counter += 1
            logger.info(f"  No improvement. Patience: {patience_counter}/{args.patience}")
        
        # Early stopping
        if patience_counter >= args.patience:
            logger.info(f"\nEarly stopping triggered at epoch {epoch}")
            break
    
    # Save training history
    with open(os.path.join(output_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    # Load best model for final evaluation
    logger.info("\n" + "=" * 60)
    logger.info(f"Loading best model from epoch {best_epoch}")
    logger.info("=" * 60)
    
    checkpoint = torch.load(os.path.join(output_dir, 'best_model.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Final evaluation on test set
    test_loss, test_acc, test_f1, test_preds, test_targets, test_ids = evaluate(
        model, test_loader, criterion, device
    )
    
    logger.info(f"\nTest Results:")
    logger.info(f"  Loss: {test_loss:.4f}")
    logger.info(f"  Accuracy: {test_acc:.4f}")
    logger.info(f"  Macro F1: {test_f1:.4f}")
    
    # Detailed classification report
    logger.info("\nClassification Report:")
    report = classification_report(test_targets, test_preds, digits=4)
    logger.info(f"\n{report}")
    
    # Confusion matrix
    cm = confusion_matrix(test_targets, test_preds)
    logger.info("\nConfusion Matrix:")
    logger.info(f"\n{cm}")
    
    # Save test results
    test_results = {
        'test_loss': test_loss,
        'test_accuracy': test_acc,
        'test_f1': test_f1,
        'best_epoch': best_epoch,
        'classification_report': classification_report(test_targets, test_preds, output_dict=True),
        'confusion_matrix': cm.tolist()
    }
    
    with open(os.path.join(output_dir, 'test_results.json'), 'w') as f:
        json.dump(test_results, f, indent=2)
    
    # Save predictions
    predictions_df = pd.DataFrame({
        'mp_id': test_ids,
        'true_ordering': test_targets,
        'predicted_ordering': test_preds
    })
    predictions_df.to_csv(os.path.join(output_dir, 'test_predictions.csv'), index=False)
    
    logger.info(f"\nResults saved to {output_dir}")
    logger.info("Training completed!")


if __name__ == '__main__':
    main()
