"""
Test a trained GPT-2 + ALIGNN model on the test dataset.

Loads the best model from output directory and evaluates on test set.
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Model
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from tqdm import tqdm
import logging
from datetime import datetime
from functools import partial
from typing import Dict, List

# Add alignn paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'alignn'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'alignn_test'))

import dgl

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


# ============================================================================
# Data Loading
# ============================================================================

def load_llm_data(data_dir: str, split: str) -> Dict:
    """Load pre-tokenized LLM data."""
    path = os.path.join(data_dir, f'{split}_llm_data.pt')
    logger.info(f"Loading LLM data from {path}")
    data = torch.load(path)
    return data


def load_gnn_metadata(data_dir: str) -> Dict:
    """Load GNN metadata to get chunk information."""
    path = os.path.join(data_dir, 'gnn_metadata.json')
    with open(path, 'r') as f:
        return json.load(f)


def load_single_gnn_chunk(data_dir: str, chunk_file: str) -> Dict:
    """Load a single GNN chunk."""
    graph_path = os.path.join(data_dir, f'{chunk_file}_graphs.bin')
    line_graph_path = os.path.join(data_dir, f'{chunk_file}_line_graphs.bin')
    meta_path = os.path.join(data_dir, f'{chunk_file}_meta.pt')
    
    graphs, _ = dgl.load_graphs(graph_path)
    line_graphs, _ = dgl.load_graphs(line_graph_path)
    meta_data = torch.load(meta_path)
    
    return {
        'graphs': graphs,
        'line_graphs': line_graphs,
        'labels': meta_data['labels'],
        'ids': meta_data['ids'],
    }


# ============================================================================
# Dataset
# ============================================================================

class ChunkBasedDataset(Dataset):
    """Dataset for a single chunk."""
    
    def __init__(self, llm_data: Dict, gnn_chunk: Dict):
        llm_id_to_idx = {id_: idx for idx, id_ in enumerate(llm_data['ids'])}
        
        self.samples = []
        for gnn_idx, gnn_id in enumerate(gnn_chunk['ids']):
            if gnn_id in llm_id_to_idx:
                llm_idx = llm_id_to_idx[gnn_id]
                self.samples.append({
                    'llm_idx': llm_idx,
                    'gnn_idx': gnn_idx,
                    'id': gnn_id,
                })
        
        self.input_ids = llm_data['input_ids']
        self.attention_mask = llm_data['attention_mask']
        self.llm_labels = llm_data['labels']
        
        self.graphs = gnn_chunk['graphs']
        self.line_graphs = gnn_chunk['line_graphs']
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        llm_idx = sample['llm_idx']
        gnn_idx = sample['gnn_idx']
        
        return {
            'input_ids': self.input_ids[llm_idx],
            'attention_mask': self.attention_mask[llm_idx],
            'graph': self.graphs[gnn_idx],
            'line_graph': self.line_graphs[gnn_idx],
            'labels': self.llm_labels[llm_idx],
            'id': sample['id'],
        }


def collate_multimodal(batch, line_graph=True):
    """Collate function for multimodal batches."""
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    ids = [item['id'] for item in batch]
    
    graphs = [item['graph'] for item in batch]
    batched_graph = dgl.batch(graphs)
    
    if line_graph:
        line_graphs = [item['line_graph'] for item in batch]
        batched_lg = dgl.batch(line_graphs)
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'graph': batched_graph,
            'line_graph': batched_lg,
            'labels': labels,
            'ids': ids,
        }
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'graph': batched_graph,
        'labels': labels,
        'ids': ids,
    }


# ============================================================================
# Model Components
# ============================================================================

class RBFExpansion(nn.Module):
    def __init__(self, vmin=0, vmax=8.0, bins=80, gamma=None):
        super().__init__()
        self.vmin = vmin
        self.vmax = vmax
        self.bins = bins
        self.register_buffer("centers", torch.linspace(vmin, vmax, bins))
        if gamma is None:
            gamma = 0.5 / ((vmax - vmin) / bins) ** 2
        self.gamma = gamma
    
    def forward(self, x):
        return torch.exp(-self.gamma * (x.unsqueeze(-1) - self.centers) ** 2)


class MLPLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.LayerNorm(out_features),
            nn.SiLU(),
        )
    
    def forward(self, x):
        return self.net(x)


class EdgeGatedGraphConv(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.src_gate = nn.Linear(in_features, out_features)
        self.dst_gate = nn.Linear(in_features, out_features)
        self.edge_gate = nn.Linear(in_features, out_features)
        self.bn_edges = nn.LayerNorm(out_features)
        
        self.src_update = nn.Linear(in_features, out_features)
        self.dst_update = nn.Linear(in_features, out_features)
        self.bn_nodes = nn.LayerNorm(out_features)
    
    def forward(self, g, node_feats, edge_feats):
        g = g.local_var()
        
        g.ndata["e_src"] = self.src_gate(node_feats)
        g.ndata["e_dst"] = self.dst_gate(node_feats)
        g.apply_edges(lambda edges: {"sigma_e": edges.src["e_src"] + edges.dst["e_dst"]})
        g.edata["sigma_e"] = g.edata["sigma_e"] + self.edge_gate(edge_feats)
        g.edata["sigma_e"] = torch.sigmoid(g.edata["sigma_e"])
        
        g.edata["e"] = edge_feats
        g.apply_edges(lambda edges: {"e": edges.data["sigma_e"] * edges.data["e"]})
        edge_feats_out = self.bn_edges(g.edata.pop("e"))
        
        g.ndata["Bh"] = self.dst_update(node_feats)
        g.update_all(
            dgl.function.u_mul_e("Bh", "sigma_e", "m"),
            dgl.function.sum("m", "sum_sigma_h")
        )
        g.update_all(
            dgl.function.copy_e("sigma_e", "m"),
            dgl.function.sum("m", "sum_sigma")
        )
        g.ndata["h"] = g.ndata["sum_sigma_h"] / (g.ndata["sum_sigma"] + 1e-8)
        node_feats_out = self.src_update(node_feats) + g.ndata.pop("h")
        node_feats_out = self.bn_nodes(node_feats_out)
        
        return node_feats_out, edge_feats_out


class ALIGNNConv(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.edge_update = EdgeGatedGraphConv(in_features, out_features)
        self.node_update = EdgeGatedGraphConv(in_features, out_features)
        self.fc = nn.Linear(in_features, out_features)
    
    def forward(self, g, lg, x, y, z):
        y, z = self.edge_update(lg, y, z)
        x, y = self.node_update(g, x, y)
        return x, y, z


class ALIGNNEncoder(nn.Module):
    def __init__(
        self,
        atom_input_features: int = 92,
        edge_input_features: int = 80,
        triplet_input_features: int = 40,
        embedding_features: int = 64,
        hidden_features: int = 256,
        alignn_layers: int = 4,
        gcn_layers: int = 4,
    ):
        super().__init__()
        
        self.atom_embedding = MLPLayer(atom_input_features, hidden_features)
        
        self.edge_embedding = nn.Sequential(
            RBFExpansion(vmin=0, vmax=8.0, bins=edge_input_features),
            MLPLayer(edge_input_features, embedding_features),
            MLPLayer(embedding_features, hidden_features),
        )
        
        self.angle_embedding = nn.Sequential(
            RBFExpansion(vmin=-1, vmax=1.0, bins=triplet_input_features),
            MLPLayer(triplet_input_features, embedding_features),
            MLPLayer(embedding_features, hidden_features),
        )
        
        self.alignn_layers = nn.ModuleList([
            ALIGNNConv(hidden_features, hidden_features)
            for _ in range(alignn_layers)
        ])
        
        self.gcn_layers = nn.ModuleList([
            EdgeGatedGraphConv(hidden_features, hidden_features)
            for _ in range(gcn_layers)
        ])
        
        self.readout = dgl.nn.AvgPooling()
        self.hidden_features = hidden_features
    
    def forward(self, g, lg):
        lg = lg.local_var()
        z = self.angle_embedding(lg.edata.pop("h"))
        
        g = g.local_var()
        x = g.ndata.pop("atom_features")
        x = self.atom_embedding(x)
        
        bondlength = g.edata["bondlength"]
        y = self.edge_embedding(bondlength)
        
        for alignn_layer in self.alignn_layers:
            x, y, z = alignn_layer(g, lg, x, y, z)
        
        for gcn_layer in self.gcn_layers:
            x, y = gcn_layer(g, x, y)
        
        h = self.readout(g, x)
        return h


class GPT2Encoder(nn.Module):
    def __init__(self, model_name: str = "gpt2", hidden_size: int = 768, 
                 freeze: bool = False, gradient_checkpointing: bool = False):
        super().__init__()
        self.gpt2 = GPT2Model.from_pretrained(model_name)
        self.hidden_size = self.gpt2.config.hidden_size
        
        if gradient_checkpointing:
            self.gpt2.gradient_checkpointing_enable()
        
        if freeze:
            for param in self.gpt2.parameters():
                param.requires_grad = False
        
        if self.hidden_size != hidden_size:
            self.proj = nn.Linear(self.hidden_size, hidden_size)
        else:
            self.proj = nn.Identity()
    
    def forward(self, input_ids, attention_mask):
        outputs = self.gpt2(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        
        mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        mean_pooled = sum_embeddings / sum_mask
        
        return self.proj(mean_pooled)


class MultimodalClassifier(nn.Module):
    def __init__(
        self,
        gpt2_model_name: str = "gpt2",
        num_classes: int = 4,
        hidden_features: int = 256,
        fusion_type: str = "concat",
        alignn_config: Dict = None,
        freeze_gpt2: bool = False,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        
        if alignn_config is None:
            alignn_config = {}
        self.alignn_encoder = ALIGNNEncoder(
            hidden_features=hidden_features,
            **alignn_config
        )
        
        self.gpt2_encoder = GPT2Encoder(
            gpt2_model_name, 
            hidden_features,
            freeze=freeze_gpt2,
            gradient_checkpointing=gradient_checkpointing
        )
        
        self.fusion_type = fusion_type
        
        if fusion_type == "concat":
            self.classifier = nn.Sequential(
                nn.Linear(hidden_features * 2, hidden_features),
                nn.LayerNorm(hidden_features),
                nn.SiLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_features, num_classes),
            )
        elif fusion_type == "add":
            self.classifier = nn.Sequential(
                nn.Linear(hidden_features, hidden_features),
                nn.LayerNorm(hidden_features),
                nn.SiLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_features, num_classes),
            )
        elif fusion_type == "attention":
            self.attn = nn.MultiheadAttention(hidden_features, num_heads=4, batch_first=True)
            self.classifier = nn.Sequential(
                nn.Linear(hidden_features, hidden_features),
                nn.LayerNorm(hidden_features),
                nn.SiLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_features, num_classes),
            )
        
        self.hidden_features = hidden_features
    
    def forward(self, input_ids, attention_mask, graph, line_graph):
        text_emb = self.gpt2_encoder(input_ids, attention_mask)
        graph_emb = self.alignn_encoder(graph, line_graph)
        
        if self.fusion_type == "concat":
            fused = torch.cat([text_emb, graph_emb], dim=-1)
        elif self.fusion_type == "add":
            fused = text_emb + graph_emb
        elif self.fusion_type == "attention":
            combined = torch.stack([text_emb, graph_emb], dim=1)
            attn_out, _ = self.attn(combined, combined, combined)
            fused = attn_out.mean(dim=1)
        
        logits = self.classifier(fused)
        return logits


# ============================================================================
# Testing Functions
# ============================================================================

def test_chunked(model, data_dir: str, llm_data: Dict, chunk_files: List[str],
                 device, batch_size: int, line_graph=True):
    """Test the model chunk by chunk, returning detailed results."""
    model.eval()
    all_preds = []
    all_labels = []
    all_ids = []
    all_probs = []
    
    collate_fn = partial(collate_multimodal, line_graph=line_graph)
    
    with torch.no_grad():
        for chunk_idx, chunk_file in enumerate(tqdm(chunk_files, desc="Testing chunks")):
            gnn_chunk = load_single_gnn_chunk(data_dir, chunk_file)
            chunk_dataset = ChunkBasedDataset(llm_data, gnn_chunk)
            
            if len(chunk_dataset) == 0:
                del gnn_chunk, chunk_dataset
                continue
            
            chunk_loader = DataLoader(
                chunk_dataset, batch_size=batch_size,
                shuffle=False, num_workers=0,
                collate_fn=collate_fn
            )
            
            for batch in chunk_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                graph = batch['graph'].to(device)
                labels = batch['labels'].to(device)
                ids = batch['ids']
                
                if line_graph:
                    line_graph_batch = batch['line_graph'].to(device)
                else:
                    line_graph_batch = None
                
                logits = model(input_ids, attention_mask, graph, line_graph_batch)
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)
                
                all_preds.extend(preds.cpu().numpy().tolist())
                all_labels.extend(labels.cpu().numpy().tolist())
                all_ids.extend(ids)
                all_probs.extend(probs.cpu().numpy().tolist())
            
            del gnn_chunk, chunk_dataset, chunk_loader
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    return all_preds, all_labels, all_ids, all_probs


def parse_args():
    parser = argparse.ArgumentParser(
        description='Test trained GPT-2 + ALIGNN model on test dataset'
    )
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='llm_gnn_data',
                        help='Directory containing prepared LLM and GNN data')
    parser.add_argument('--model_dir', type=str, default='llm_gnn_output',
                        help='Directory containing trained model')
    parser.add_argument('--model_file', type=str, default='best_model.pt',
                        help='Model file name (default: best_model.pt)')
    
    # Model arguments (used if model_config.json is not found)
    parser.add_argument('--gpt2_model', type=str, default='gpt2',
                        help='GPT-2 model name')
    parser.add_argument('--hidden_features', type=int, default=256,
                        help='Hidden feature dimension')
    parser.add_argument('--fusion_type', type=str, default='concat',
                        choices=['concat', 'add', 'attention'],
                        help='Fusion type')
    parser.add_argument('--alignn_layers', type=int, default=4,
                        help='Number of ALIGNN layers')
    parser.add_argument('--gcn_layers', type=int, default=4,
                        help='Number of GCN layers')
    
    # Test arguments
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for testing')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'val', 'test'],
                        help='Which split to test on')
    
    # Output arguments
    parser.add_argument('--output_file', type=str, default=None,
                        help='Output CSV file for predictions (default: test_predictions.csv in model_dir)')
    
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
    
    # Load model config (or use command line args as fallback)
    config_path = os.path.join(args.model_dir, 'model_config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            model_config = json.load(f)
        logger.info(f"Loaded model config from {config_path}")
    else:
        logger.warning(f"Model config not found at {config_path}, using command line arguments")
        model_config = {
            'gpt2_model': args.gpt2_model,
            'hidden_features': args.hidden_features,
            'fusion_type': args.fusion_type,
            'alignn_layers': args.alignn_layers,
            'gcn_layers': args.gcn_layers,
            'num_classes': NUM_LABELS,
        }
    
    logger.info(f"Model config: {model_config}")
    
    # Load model
    model_path = os.path.join(args.model_dir, args.model_file)
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        return
    
    logger.info(f"Loading model from {model_path}")
    
    alignn_config = {
        'alignn_layers': model_config.get('alignn_layers', 4),
        'gcn_layers': model_config.get('gcn_layers', 4),
    }
    
    model = MultimodalClassifier(
        gpt2_model_name=model_config.get('gpt2_model', 'gpt2'),
        num_classes=model_config.get('num_classes', NUM_LABELS),
        hidden_features=model_config.get('hidden_features', 256),
        fusion_type=model_config.get('fusion_type', 'concat'),
        alignn_config=alignn_config,
    )
    
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    logger.info("Model loaded successfully")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total parameters: {total_params:,}")
    
    # Load test data
    logger.info(f"Loading {args.split} data...")
    test_llm = load_llm_data(args.data_dir, args.split)
    
    gnn_metadata = load_gnn_metadata(args.data_dir)
    test_chunk_files = gnn_metadata['splits'][args.split]['chunk_files']
    
    logger.info(f"Found {len(test_chunk_files)} chunks for {args.split} split")
    
    # Run testing
    logger.info("Running inference...")
    preds, labels, ids, probs = test_chunked(
        model, args.data_dir, test_llm, test_chunk_files,
        device, args.batch_size, line_graph=True
    )
    
    # Calculate metrics
    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted', zero_division=0)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Test Results on {args.split} split")
    logger.info(f"{'='*60}")
    logger.info(f"Samples: {len(preds)}")
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"F1 (weighted): {f1:.4f}")
    
    # Per-class metrics
    target_names = [f"{i} ({ORDERING_LABELS[i]})" for i in range(NUM_LABELS)]
    report = classification_report(
        labels, preds,
        target_names=target_names,
        digits=4,
        zero_division=0
    )
    logger.info(f"\nClassification Report:\n{report}")
    
    # Confusion matrix
    cm = confusion_matrix(labels, preds)
    logger.info(f"\nConfusion Matrix:\n{cm}")
    
    # Save predictions to CSV
    output_file = args.output_file
    if output_file is None:
        output_file = os.path.join(args.model_dir, f'{args.split}_predictions.csv')
    
    import pandas as pd
    
    results_df = pd.DataFrame({
        'id': ids,
        'true_label': labels,
        'predicted_label': preds,
        'true_name': [ORDERING_LABELS[l] for l in labels],
        'predicted_name': [ORDERING_LABELS[p] for p in preds],
        'correct': [l == p for l, p in zip(labels, preds)],
        'prob_NM': [p[0] for p in probs],
        'prob_FM': [p[1] for p in probs],
        'prob_AFM': [p[2] for p in probs],
        'prob_FiM': [p[3] for p in probs],
    })
    
    results_df.to_csv(output_file, index=False)
    logger.info(f"\nPredictions saved to {output_file}")
    
    # Save summary results
    summary = {
        'split': args.split,
        'model_file': args.model_file,
        'num_samples': len(preds),
        'accuracy': accuracy,
        'f1_weighted': f1,
        'classification_report': classification_report(
            labels, preds,
            target_names=target_names,
            output_dict=True,
            zero_division=0
        ),
        'confusion_matrix': cm.tolist(),
        'timestamp': datetime.now().isoformat(),
    }
    
    summary_path = os.path.join(args.model_dir, f'{args.split}_results.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary saved to {summary_path}")
    
    logger.info("\nTesting completed!")
    
    return summary


if __name__ == '__main__':
    main()
