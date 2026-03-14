"""
Train a combined GPT-2 + ALIGNN model from prepared data.

This script:
1. Loads pre-tokenized LLM data
2. Loads pre-built GNN graphs
3. Creates combined dataset matching LLM and GNN data
4. Trains the multimodal classifier
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
from typing import Dict, List, Optional

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
    
    # Load graphs
    graphs, _ = dgl.load_graphs(graph_path)
    line_graphs, _ = dgl.load_graphs(line_graph_path)
    
    # Load metadata
    meta_data = torch.load(meta_path)
    
    return {
        'graphs': graphs,
        'line_graphs': line_graphs,
        'labels': meta_data['labels'],
        'ids': meta_data['ids'],
    }


def load_gnn_data_chunked(data_dir: str, split: str, metadata: Dict = None) -> Dict:
    """Load all pre-built GNN graphs from chunks (for validation/test)."""
    if metadata is None:
        metadata = load_gnn_metadata(data_dir)
    
    split_info = metadata['splits'][split]
    chunk_files = split_info['chunk_files']
    
    logger.info(f"Loading {split} GNN data from {len(chunk_files)} chunks...")
    
    all_graphs = []
    all_line_graphs = []
    all_labels = []
    all_ids = []
    
    for chunk_file in tqdm(chunk_files, desc=f"Loading {split} chunks"):
        chunk_data = load_single_gnn_chunk(data_dir, chunk_file)
        all_graphs.extend(chunk_data['graphs'])
        all_line_graphs.extend(chunk_data['line_graphs'])
        all_labels.extend(chunk_data['labels'])
        all_ids.extend(chunk_data['ids'])
    
    logger.info(f"Loaded {len(all_graphs)} {split} graphs")
    
    return {
        'graphs': all_graphs,
        'line_graphs': all_line_graphs,
        'labels': all_labels,
        'ids': all_ids,
    }


# ============================================================================
# Combined Dataset
# ============================================================================

class PreparedMultimodalDataset(Dataset):
    """Dataset combining pre-tokenized text and pre-built graph data."""
    
    def __init__(self, llm_data: Dict, gnn_data: Dict):
        # Create lookup from ID to index for both datasets
        llm_id_to_idx = {id_: idx for idx, id_ in enumerate(llm_data['ids'])}
        gnn_id_to_idx = {id_: idx for idx, id_ in enumerate(gnn_data['ids'])}
        
        # Find matching IDs
        matched_ids = set(llm_data['ids']) & set(gnn_data['ids'])
        logger.info(f"Found {len(matched_ids)} matched samples between LLM and GNN data")
        
        # Build aligned indices
        self.llm_indices = []
        self.gnn_indices = []
        self.matched_ids = []
        
        for id_ in matched_ids:
            self.llm_indices.append(llm_id_to_idx[id_])
            self.gnn_indices.append(gnn_id_to_idx[id_])
            self.matched_ids.append(id_)
        
        self.input_ids = llm_data['input_ids']
        self.attention_mask = llm_data['attention_mask']
        self.llm_labels = llm_data['labels']
        
        self.graphs = gnn_data['graphs']
        self.line_graphs = gnn_data['line_graphs']
        self.gnn_labels = gnn_data['labels']
        
        logger.info(f"Created dataset with {len(self)} samples")
    
    def __len__(self):
        return len(self.matched_ids)
    
    def __getitem__(self, idx):
        llm_idx = self.llm_indices[idx]
        gnn_idx = self.gnn_indices[idx]
        
        return {
            'input_ids': self.input_ids[llm_idx],
            'attention_mask': self.attention_mask[llm_idx],
            'graph': self.graphs[gnn_idx],
            'line_graph': self.line_graphs[gnn_idx],
            'labels': self.llm_labels[llm_idx],
        }


class ChunkBasedDataset(Dataset):
    """Dataset for a single chunk - used for chunk-by-chunk training."""
    
    def __init__(self, llm_data: Dict, gnn_chunk: Dict):
        """
        Args:
            llm_data: Full LLM data dict with 'input_ids', 'attention_mask', 'labels', 'ids'
            gnn_chunk: Single chunk GNN data with 'graphs', 'line_graphs', 'labels', 'ids'
        """
        # Create lookup from ID to LLM index
        llm_id_to_idx = {id_: idx for idx, id_ in enumerate(llm_data['ids'])}
        
        # Match chunk GNN IDs with LLM IDs
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
        }
    
    def get_labels(self) -> List[int]:
        """Return labels for class weight calculation."""
        return [self.llm_labels[s['llm_idx']].item() for s in self.samples]


def collate_multimodal(batch, line_graph=True):
    """Collate function for multimodal batches."""
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    
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
            'labels': labels
        }
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'graph': batched_graph,
        'labels': labels
    }


# ============================================================================
# Model Components
# ============================================================================

class RBFExpansion(nn.Module):
    """Expand interatomic distances with radial basis functions."""
    
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
    """MLP with SiLU activation and layer norm."""
    
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
    """Edge-gated graph convolution."""
    
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
    """ALIGNN convolution layer."""
    
    def __init__(self, in_features, out_features):
        super().__init__()
        self.edge_update = EdgeGatedGraphConv(in_features, out_features)
        self.node_update = EdgeGatedGraphConv(in_features, out_features)
        self.fc = nn.Linear(in_features, out_features)
    
    def forward(self, g, lg, x, y, z):
        # Line graph update (edge-angles)
        y, z = self.edge_update(lg, y, z)
        
        # Crystal graph update
        x, y = self.node_update(g, x, y)
        
        return x, y, z


class ALIGNNEncoder(nn.Module):
    """ALIGNN encoder that outputs graph embeddings."""
    
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
        """Forward pass returning graph embedding."""
        lg = lg.local_var()
        z = self.angle_embedding(lg.edata.pop("h"))
        
        g = g.local_var()
        x = g.ndata.pop("atom_features")
        x = self.atom_embedding(x)
        
        bondlength = g.edata["bondlength"]
        y = self.edge_embedding(bondlength)
        
        # ALIGNN updates
        for alignn_layer in self.alignn_layers:
            x, y, z = alignn_layer(g, lg, x, y, z)
        
        # GCN updates
        for gcn_layer in self.gcn_layers:
            x, y = gcn_layer(g, x, y)
        
        # Readout
        h = self.readout(g, x)
        return h


class GPT2Encoder(nn.Module):
    """GPT-2 encoder that outputs text embeddings."""
    
    def __init__(self, model_name: str = "gpt2", hidden_size: int = 768, 
                 freeze: bool = False, gradient_checkpointing: bool = False):
        super().__init__()
        self.gpt2 = GPT2Model.from_pretrained(model_name)
        self.hidden_size = self.gpt2.config.hidden_size
        
        # Enable gradient checkpointing to save memory
        if gradient_checkpointing:
            self.gpt2.gradient_checkpointing_enable()
        
        # Freeze GPT-2 parameters to save memory
        if freeze:
            for param in self.gpt2.parameters():
                param.requires_grad = False
        
        # Project to target hidden size if needed
        if self.hidden_size != hidden_size:
            self.proj = nn.Linear(self.hidden_size, hidden_size)
        else:
            self.proj = nn.Identity()
    
    def forward(self, input_ids, attention_mask):
        outputs = self.gpt2(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use the last token's representation (similar to classification)
        # Or mean pooling over all tokens
        last_hidden_state = outputs.last_hidden_state
        
        # Mean pooling with attention mask
        mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        mean_pooled = sum_embeddings / sum_mask
        
        return self.proj(mean_pooled)


class MultimodalClassifier(nn.Module):
    """Combined GPT-2 + ALIGNN model for classification."""
    
    def __init__(
        self,
        gpt2_model_name: str = "gpt2",
        num_classes: int = 4,
        hidden_features: int = 256,
        fusion_type: str = "concat",  # "concat", "attention", "add"
        alignn_config: Dict = None,
        freeze_gpt2: bool = False,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        
        # ALIGNN encoder
        if alignn_config is None:
            alignn_config = {}
        self.alignn_encoder = ALIGNNEncoder(
            hidden_features=hidden_features,
            **alignn_config
        )
        
        # GPT-2 encoder
        self.gpt2_encoder = GPT2Encoder(
            gpt2_model_name, 
            hidden_features,
            freeze=freeze_gpt2,
            gradient_checkpointing=gradient_checkpointing
        )
        
        self.fusion_type = fusion_type
        
        # Fusion and classification layers
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
        # Get text embedding
        text_emb = self.gpt2_encoder(input_ids, attention_mask)
        
        # Get graph embedding
        graph_emb = self.alignn_encoder(graph, line_graph)
        
        # Fusion
        if self.fusion_type == "concat":
            fused = torch.cat([text_emb, graph_emb], dim=-1)
        elif self.fusion_type == "add":
            fused = text_emb + graph_emb
        elif self.fusion_type == "attention":
            # Stack embeddings as sequence
            combined = torch.stack([text_emb, graph_emb], dim=1)
            attn_out, _ = self.attn(combined, combined, combined)
            fused = attn_out.mean(dim=1)
        
        # Classification
        logits = self.classifier(fused)
        return logits


# ============================================================================
# Training Functions
# ============================================================================

def compute_class_weights(labels: List[int], num_classes: int = 4) -> torch.Tensor:
    """Compute class weights for imbalanced data."""
    class_counts = np.bincount(labels, minlength=num_classes)
    total = len(labels)
    weights = total / (num_classes * class_counts + 1e-6)
    weights = weights / weights.sum() * num_classes
    return torch.tensor(weights, dtype=torch.float32)


def train_epoch(model, dataloader, optimizer, scheduler, device, 
                class_weights=None, line_graph=True, gradient_accumulation_steps=1):
    """Train for one epoch with gradient accumulation support."""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    else:
        criterion = nn.CrossEntropyLoss()
    
    optimizer.zero_grad()
    progress_bar = tqdm(dataloader, desc="Training")
    for step, batch in enumerate(progress_bar):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        graph = batch['graph'].to(device)
        labels = batch['labels'].to(device)
        
        if line_graph:
            line_graph_batch = batch['line_graph'].to(device)
        else:
            line_graph_batch = None
        
        logits = model(input_ids, attention_mask, graph, line_graph_batch)
        loss = criterion(logits, labels)
        
        # Scale loss for gradient accumulation
        loss = loss / gradient_accumulation_steps
        loss.backward()
        
        # Update weights after accumulating gradients
        if (step + 1) % gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * gradient_accumulation_steps
        
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
        
        progress_bar.set_postfix({'loss': f'{loss.item() * gradient_accumulation_steps:.4f}'})
        
        # Clear GPU cache periodically to help with memory
        if step % 10 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Handle remaining gradients
    if len(dataloader) % gradient_accumulation_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    return avg_loss, accuracy, f1


def evaluate(model, dataloader, device, class_weights=None, line_graph=True):
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
            graph = batch['graph'].to(device)
            labels = batch['labels'].to(device)
            
            if line_graph:
                line_graph_batch = batch['line_graph'].to(device)
            else:
                line_graph_batch = None
            
            logits = model(input_ids, attention_mask, graph, line_graph_batch)
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    return avg_loss, accuracy, f1, all_preds, all_labels


def train_epoch_chunked(model, data_dir: str, llm_data: Dict, chunk_files: List[str],
                        optimizer, scheduler, device, batch_size: int,
                        class_weights=None, line_graph=True, 
                        gradient_accumulation_steps=1, shuffle_chunks=True):
    """Train for one epoch, loading and processing chunks one at a time."""
    import random
    
    model.train()
    total_loss = 0
    total_samples = 0
    all_preds = []
    all_labels = []
    
    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Shuffle chunk order for training
    chunk_order = list(range(len(chunk_files)))
    if shuffle_chunks:
        random.shuffle(chunk_order)
    
    collate_fn = partial(collate_multimodal, line_graph=line_graph)
    global_step = 0
    
    for chunk_idx in chunk_order:
        chunk_file = chunk_files[chunk_idx]
        logger.info(f"Loading chunk {chunk_idx + 1}/{len(chunk_files)}: {chunk_file}")
        
        # Load this chunk
        gnn_chunk = load_single_gnn_chunk(data_dir, chunk_file)
        
        # Create dataset for this chunk
        chunk_dataset = ChunkBasedDataset(llm_data, gnn_chunk)
        
        if len(chunk_dataset) == 0:
            logger.warning(f"Chunk {chunk_file} has no matched samples, skipping")
            # Free memory
            del gnn_chunk, chunk_dataset
            continue
        
        # Create dataloader for this chunk
        chunk_loader = DataLoader(
            chunk_dataset, batch_size=batch_size,
            shuffle=True, num_workers=0,
            collate_fn=collate_fn
        )
        
        # Train on this chunk
        chunk_loss = 0
        optimizer.zero_grad()
        
        for step, batch in enumerate(chunk_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            graph = batch['graph'].to(device)
            labels = batch['labels'].to(device)
            
            if line_graph:
                line_graph_batch = batch['line_graph'].to(device)
            else:
                line_graph_batch = None
            
            logits = model(input_ids, attention_mask, graph, line_graph_batch)
            loss = criterion(logits, labels)
            
            # Scale loss for gradient accumulation
            loss = loss / gradient_accumulation_steps
            loss.backward()
            
            # Update weights after accumulating gradients
            if (global_step + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            chunk_loss += loss.item() * gradient_accumulation_steps
            
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            
            global_step += 1
        
        # Handle remaining gradients for this chunk
        if global_step % gradient_accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        total_loss += chunk_loss
        total_samples += len(chunk_loader)
        
        logger.info(f"Chunk {chunk_idx + 1} - Loss: {chunk_loss / len(chunk_loader):.4f}, Samples: {len(chunk_dataset)}")
        
        # Free memory
        del gnn_chunk, chunk_dataset, chunk_loader
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    avg_loss = total_loss / total_samples if total_samples > 0 else 0
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    return avg_loss, accuracy, f1


def evaluate_chunked(model, data_dir: str, llm_data: Dict, chunk_files: List[str],
                     device, batch_size: int, class_weights=None, line_graph=True):
    """Evaluate the model chunk by chunk."""
    model.eval()
    total_loss = 0
    total_samples = 0
    all_preds = []
    all_labels = []
    
    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    else:
        criterion = nn.CrossEntropyLoss()
    
    collate_fn = partial(collate_multimodal, line_graph=line_graph)
    
    with torch.no_grad():
        for chunk_idx, chunk_file in enumerate(chunk_files):
            # Load this chunk
            gnn_chunk = load_single_gnn_chunk(data_dir, chunk_file)
            
            # Create dataset for this chunk
            chunk_dataset = ChunkBasedDataset(llm_data, gnn_chunk)
            
            if len(chunk_dataset) == 0:
                del gnn_chunk, chunk_dataset
                continue
            
            # Create dataloader
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
                
                if line_graph:
                    line_graph_batch = batch['line_graph'].to(device)
                else:
                    line_graph_batch = None
                
                logits = model(input_ids, attention_mask, graph, line_graph_batch)
                loss = criterion(logits, labels)
                
                total_loss += loss.item()
                total_samples += 1
                
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
            
            # Free memory
            del gnn_chunk, chunk_dataset, chunk_loader
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    avg_loss = total_loss / total_samples if total_samples > 0 else 0
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    return avg_loss, accuracy, f1, all_preds, all_labels


def count_total_samples(data_dir: str, llm_data: Dict, chunk_files: List[str]) -> int:
    """Count total matched samples across all chunks without loading graphs."""
    llm_ids = set(llm_data['ids'])
    total = 0
    
    for chunk_file in chunk_files:
        meta_path = os.path.join(data_dir, f'{chunk_file}_meta.pt')
        meta_data = torch.load(meta_path)
        chunk_ids = set(meta_data['ids'])
        total += len(llm_ids & chunk_ids)
    
    return total


def collect_all_labels(data_dir: str, llm_data: Dict, chunk_files: List[str]) -> List[int]:
    """Collect all labels for class weight calculation without loading graphs."""
    llm_id_to_idx = {id_: idx for idx, id_ in enumerate(llm_data['ids'])}
    all_labels = []
    
    for chunk_file in chunk_files:
        meta_path = os.path.join(data_dir, f'{chunk_file}_meta.pt')
        meta_data = torch.load(meta_path)
        
        for gnn_id in meta_data['ids']:
            if gnn_id in llm_id_to_idx:
                llm_idx = llm_id_to_idx[gnn_id]
                all_labels.append(llm_data['labels'][llm_idx].item())
    
    return all_labels


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train combined GPT-2 + ALIGNN model from prepared data'
    )
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='llm_gnn_data',
                        help='Directory containing prepared LLM and GNN data')
    
    # Model arguments
    parser.add_argument('--gpt2_model', type=str, default='gpt2',
                        help='GPT-2 model name (gpt2, gpt2-medium, etc.)')
    parser.add_argument('--hidden_features', type=int, default=256,
                        help='Hidden feature dimension')
    parser.add_argument('--fusion_type', type=str, default='concat',
                        choices=['concat', 'add', 'attention'],
                        help='Fusion type for combining modalities')
    
    # ALIGNN arguments
    parser.add_argument('--alignn_layers', type=int, default=4,
                        help='Number of ALIGNN layers')
    parser.add_argument('--gcn_layers', type=int, default=4,
                        help='Number of GCN layers')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help='Gradient accumulation steps (effective batch = batch_size * steps)')
    parser.add_argument('--freeze_gpt2', action='store_true',
                        help='Freeze GPT-2 encoder to save memory')
    parser.add_argument('--gradient_checkpointing', action='store_true',
                        help='Use gradient checkpointing to save memory')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--warmup_ratio', type=float, default=0.1,
                        help='Warmup ratio for scheduler')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('--use_class_weights', action='store_true',
                        help='Use class weights for imbalanced data')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='llm_gnn_output',
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
    
    # Load LLM data (kept in memory - typically smaller)
    logger.info("Loading LLM data...")
    train_llm = load_llm_data(args.data_dir, 'train')
    val_llm = load_llm_data(args.data_dir, 'val')
    test_llm = load_llm_data(args.data_dir, 'test')
    
    # Load GNN metadata (chunk info only, not the actual graphs)
    gnn_metadata = load_gnn_metadata(args.data_dir)
    
    train_chunk_files = gnn_metadata['splits']['train']['chunk_files']
    val_chunk_files = gnn_metadata['splits']['val']['chunk_files']
    test_chunk_files = gnn_metadata['splits']['test']['chunk_files']
    
    logger.info(f"Train chunks: {len(train_chunk_files)}, Val chunks: {len(val_chunk_files)}, Test chunks: {len(test_chunk_files)}")
    
    # Count total samples without loading graphs
    train_samples = count_total_samples(args.data_dir, train_llm, train_chunk_files)
    val_samples = count_total_samples(args.data_dir, val_llm, val_chunk_files)
    test_samples = count_total_samples(args.data_dir, test_llm, test_chunk_files)
    
    logger.info(f"Train samples: {train_samples}, Val samples: {val_samples}, Test samples: {test_samples}")
    
    # Calculate class weights (without loading full graphs)
    class_weights = None
    if args.use_class_weights:
        all_labels = collect_all_labels(args.data_dir, train_llm, train_chunk_files)
        class_weights = compute_class_weights(all_labels)
        logger.info(f"Class weights: {class_weights.tolist()}")
    
    # Create model
    logger.info("Creating multimodal model...")
    alignn_config = {
        'alignn_layers': args.alignn_layers,
        'gcn_layers': args.gcn_layers,
    }
    
    model = MultimodalClassifier(
        gpt2_model_name=args.gpt2_model,
        num_classes=NUM_LABELS,
        hidden_features=args.hidden_features,
        fusion_type=args.fusion_type,
        alignn_config=alignn_config,
        freeze_gpt2=args.freeze_gpt2,
        gradient_checkpointing=args.gradient_checkpointing,
    )
    model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Estimate steps per epoch based on sample count
    batches_per_epoch = (train_samples + args.batch_size - 1) // args.batch_size
    steps_per_epoch = batches_per_epoch // args.gradient_accumulation_steps
    if batches_per_epoch % args.gradient_accumulation_steps != 0:
        steps_per_epoch += 1
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.learning_rate,
        total_steps=total_steps,
        pct_start=args.warmup_ratio,
    )
    
    logger.info(f"Estimated total training steps: {total_steps}, Warmup steps: {warmup_steps}")
    logger.info(f"Gradient accumulation steps: {args.gradient_accumulation_steps}")
    
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
    
    # Training loop (chunk by chunk)
    logger.info("Starting chunk-by-chunk training...")
    for epoch in range(args.epochs):
        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {epoch + 1}/{args.epochs}")
        logger.info(f"{'='*60}")
        
        # Train on chunks
        train_loss, train_acc, train_f1 = train_epoch_chunked(
            model, args.data_dir, train_llm, train_chunk_files,
            optimizer, scheduler, device, args.batch_size,
            class_weights, line_graph=True,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            shuffle_chunks=True
        )
        
        # Validate on chunks
        val_loss, val_acc, val_f1, _, _ = evaluate_chunked(
            model, args.data_dir, val_llm, val_chunk_files,
            device, args.batch_size, class_weights, line_graph=True
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
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            logger.info(f"New best model saved! Val F1: {val_f1:.4f}")
            
            # Save checkpoint
            torch.save(
                best_model_state,
                os.path.join(args.output_dir, 'best_model.pt')
            )
    
    # Load best model for final evaluation
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Final evaluation on test set
    logger.info("\n" + "="*60)
    logger.info("Final Evaluation on Test Set")
    logger.info("="*60)
    
    test_loss, test_acc, test_f1, test_preds, test_labels = evaluate_chunked(
        model, args.data_dir, test_llm, test_chunk_files,
        device, args.batch_size, class_weights, line_graph=True
    )
    
    logger.info(f"Test - Loss: {test_loss:.4f}, Acc: {test_acc:.4f}, F1: {test_f1:.4f}")
    
    # Detailed classification report
    target_names = [f"{i} ({ORDERING_LABELS[i]})" for i in range(NUM_LABELS)]
    report = classification_report(
        test_labels, test_preds,
        target_names=target_names,
        digits=4,
        zero_division=0
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
            output_dict=True,
            zero_division=0
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
        model_path = os.path.join(args.output_dir, 'final_model.pt')
        torch.save(model.state_dict(), model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Save config
        config = {
            'gpt2_model': args.gpt2_model,
            'hidden_features': args.hidden_features,
            'fusion_type': args.fusion_type,
            'alignn_layers': args.alignn_layers,
            'gcn_layers': args.gcn_layers,
            'num_classes': NUM_LABELS,
        }
        config_path = os.path.join(args.output_dir, 'model_config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
    
    logger.info("\nTraining completed!")
    
    return results


if __name__ == '__main__':
    main()
