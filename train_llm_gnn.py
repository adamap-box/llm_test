"""
Train a combined GPT-2 + ALIGNN model to predict mp_ordering.

This script combines:
1. GPT-2 for text feature extraction from chemical descriptions
2. ALIGNN for crystal structure feature extraction from graph networks

The combined model fuses both modalities to predict magnetic ordering:
- 0: NM (Non-Magnetic)
- 1: FM (Ferromagnetic)
- 2: AFM (Antiferromagnetic)
- 3: FiM (Ferrimagnetic)
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    GPT2Tokenizer,
    GPT2Model,
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
from functools import partial
from typing import Dict, List, Tuple, Optional

# Add alignn paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'alignn'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'alignn_test'))

import dgl
from dgl.dataloading import GraphDataLoader
from jarvis.core.atoms import Atoms
from jarvis.core.specie import chem_data, get_node_attributes

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
# Data Loading Utilities
# ============================================================================

def load_prepared_data(filepath: str) -> List[Dict]:
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
                   structure_data: List[Dict],
                   id_column: str = 'mp_id') -> Tuple[pd.DataFrame, List[Dict]]:
    """Match samples between CSV and structure data by ID."""
    # Create lookup for structure data
    structure_lookup = {item['id']: item for item in structure_data}
    
    # Filter CSV to only include samples with matching structures
    matched_ids = set(csv_df[id_column].values) & set(structure_lookup.keys())
    logger.info(f"Found {len(matched_ids)} matched samples")
    
    # Filter both datasets
    matched_csv = csv_df[csv_df[id_column].isin(matched_ids)].copy()
    matched_csv = matched_csv.reset_index(drop=True)
    
    # Create ordered structure list matching CSV order
    matched_structures = []
    for _, row in matched_csv.iterrows():
        matched_structures.append(structure_lookup[row[id_column]])
    
    return matched_csv, matched_structures


# ============================================================================
# Graph Construction (from graph.py)
# ============================================================================

def compute_bond_cosines(edges):
    """Compute bond angle cosines from bond displacement vectors."""
    r1 = -edges.src["r"]
    r2 = edges.dst["r"]
    bond_cosine = torch.sum(r1 * r2, dim=1) / (
        torch.norm(r1, dim=1) * torch.norm(r2, dim=1) + 1e-8
    )
    bond_cosine = torch.clamp(bond_cosine, -1, 1)
    return {"h": bond_cosine}


def create_line_graph(g):
    """Create line graph with proper edge data transfer (Windows-compatible)."""
    # Create line graph without shared memory to avoid Windows DGL issues
    lg = g.line_graph(shared=False)
    
    # Line graph nodes correspond to original graph edges
    # Copy edge displacement vectors to line graph nodes
    lg.ndata["r"] = g.edata["r"].clone()
    
    # Compute bond cosines for line graph edges
    lg.apply_edges(compute_bond_cosines)
    
    return lg


def nearest_neighbor_edges(atoms, cutoff=8, max_neighbors=12):
    """Construct k-NN edge list."""
    from collections import defaultdict
    
    all_neighbors = atoms.get_all_neighbors(r=cutoff)
    min_nbrs = min(len(nl) for nl in all_neighbors) if len(all_neighbors) > 0 else 0
    
    if min_nbrs < max_neighbors:
        lat = atoms.lattice
        r_cut = max(lat.a, lat.b, lat.c, 2 * cutoff)
        if r_cut > cutoff:
            return nearest_neighbor_edges(atoms, cutoff=r_cut, max_neighbors=max_neighbors)
    
    edges = defaultdict(set)
    for site_idx, neighborlist in enumerate(all_neighbors):
        neighborlist = sorted(neighborlist, key=lambda x: x[2])
        if len(neighborlist) >= max_neighbors:
            max_dist = neighborlist[max_neighbors - 1][2]
        else:
            max_dist = float('inf')
        
        for nbr in neighborlist:
            if nbr[2] <= max_dist:
                dst = nbr[1]
                image = tuple(nbr[3])
                edges[(site_idx, dst)].add(image)
    
    return edges


def build_undirected_edgedata(atoms, edges):
    """Build undirected edge data with displacements."""
    u, v, r = [], [], []
    cart_coords = atoms.cart_coords
    lattice_mat = atoms.lattice_mat
    
    for (src_id, dst_id), images in edges.items():
        for dst_image in images:
            u.append(src_id)
            v.append(dst_id)
            d = cart_coords[dst_id] - cart_coords[src_id] + np.dot(dst_image, lattice_mat)
            r.append(d)
            
            # Add reverse edge
            u.append(dst_id)
            v.append(src_id)
            r.append(-d)
    
    return u, v, r


def atoms_to_graph(atoms, cutoff=8.0, max_neighbors=12, atom_features="cgcnn"):
    """Convert atoms to DGL graph."""
    edges = nearest_neighbor_edges(atoms, cutoff=cutoff, max_neighbors=max_neighbors)
    u, v, r = build_undirected_edgedata(atoms, edges)
    
    # Create graph
    g = dgl.graph((u, v))
    
    # Add edge features
    g.edata["r"] = torch.tensor(np.array(r), dtype=torch.float32)
    bondlength = torch.norm(g.edata["r"], dim=1)
    g.edata["bondlength"] = bondlength
    
    # Add node features (atomic numbers)
    z = torch.tensor([atoms.atomic_numbers[i] for i in range(atoms.num_atoms)])
    g.ndata["atom_features"] = z.unsqueeze(1)
    
    return g


def get_attribute_lookup(atom_features: str = "cgcnn"):
    """Build a lookup array indexed by atomic number."""
    max_z = max(v["Z"] for v in chem_data.values())
    template = get_node_attributes("C", atom_features)
    features = np.zeros((1 + max_z, len(template)))
    
    for element, v in chem_data.items():
        z = v["Z"]
        x = get_node_attributes(element, atom_features)
        if x is not None:
            features[z, :] = x
    
    return features


# ============================================================================
# Combined Multimodal Dataset
# ============================================================================

class MultimodalDataset(Dataset):
    """Dataset combining text and graph data for multimodal learning."""
    
    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        structures: List[Dict],
        tokenizer,
        max_length: int = 512,
        cutoff: float = 8.0,
        max_neighbors: int = 12,
        atom_features: str = "cgcnn",
        line_graph: bool = True,
    ):
        self.texts = texts
        self.labels = labels
        self.structures = structures
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.cutoff = cutoff
        self.max_neighbors = max_neighbors
        self.line_graph = line_graph
        
        # Build atom features lookup
        self.feature_lookup = get_attribute_lookup(atom_features)
        
        # Pre-build graphs
        logger.info("Building crystal graphs...")
        self.graphs = []
        self.line_graphs = []
        self.valid_indices = []  # Track which samples have valid graphs
        
        for idx, item in enumerate(tqdm(structures, desc="Building graphs")):
            try:
                atoms = Atoms.from_dict(item["atoms"])
                g = atoms_to_graph(atoms, cutoff=cutoff, max_neighbors=max_neighbors)
                
                # Skip graphs with no edges (can't create line graph)
                if g.num_edges() == 0:
                    logger.warning(f"Skipping sample {idx}: graph has no edges")
                    continue
                
                # Apply atom features
                z = g.ndata["atom_features"].squeeze().int()
                feat = torch.tensor(self.feature_lookup[z], dtype=torch.float32)
                if g.num_nodes() == 1:
                    feat = feat.unsqueeze(0)
                g.ndata["atom_features"] = feat
                
                if line_graph:
                    # Use helper function for Windows-compatible line graph creation
                    lg = create_line_graph(g)
                    self.line_graphs.append(lg)
                
                self.graphs.append(g)
                self.valid_indices.append(idx)
                
            except Exception as e:
                logger.warning(f"Skipping sample {idx} due to error: {e}")
                continue
        
        # Filter texts and labels to match valid graphs
        self.texts = [texts[i] for i in self.valid_indices]
        self.labels = [labels[i] for i in self.valid_indices]
        logger.info(f"Successfully built {len(self.graphs)} graphs from {len(structures)} structures")
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        # Get graph
        g = self.graphs[idx]
        lg = self.line_graphs[idx] if self.line_graph else None
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'graph': g,
            'line_graph': lg,
            'labels': torch.tensor(label, dtype=torch.long)
        }


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
    
    def __init__(self, model_name: str = "gpt2", hidden_size: int = 768, freeze: bool = False, gradient_checkpointing: bool = False):
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


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train combined GPT-2 + ALIGNN model for mp_ordering prediction'
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
    
    # Model arguments
    parser.add_argument('--gpt2_model', type=str, default='gpt2',
                        help='GPT-2 model name (gpt2, gpt2-medium, etc.)')
    parser.add_argument('--max_length', type=int, default=512,
                        help='Maximum text sequence length')
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
    parser.add_argument('--cutoff', type=float, default=8.0,
                        help='Distance cutoff for graph construction')
    parser.add_argument('--max_neighbors', type=int, default=12,
                        help='Maximum number of neighbors')
    
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
    
    # Split arguments (if not using pre-split structure data)
    parser.add_argument('--use_csv_splits', action='store_true',
                        help='Create splits from CSV instead of using structure files')
    parser.add_argument('--test_size', type=float, default=0.1,
                        help='Test set ratio')
    parser.add_argument('--val_size', type=float, default=0.1,
                        help='Validation set ratio')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random seed')
    
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
    
    # Load CSV data
    csv_df = load_csv_data(
        args.input_csv,
        args.text_column,
        args.label_column,
        args.id_column
    )
    
    # Load structure data
    logger.info("Loading structure data...")
    train_structures = load_prepared_data(args.train_structures)
    val_structures = load_prepared_data(args.val_structures)
    test_structures = load_prepared_data(args.test_structures)
    all_structures = train_structures + val_structures + test_structures
    
    # Match CSV with structures
    matched_csv, matched_structures = match_datasets(csv_df, all_structures, args.id_column)
    
    # Create lookup for structures
    structure_lookup = {item['id']: item for item in matched_structures}
    
    # Get IDs from prepared data splits
    train_ids = set(item['id'] for item in train_structures)
    val_ids = set(item['id'] for item in val_structures)
    test_ids = set(item['id'] for item in test_structures)
    
    # Split CSV data based on structure splits
    train_csv = matched_csv[matched_csv[args.id_column].isin(train_ids)].reset_index(drop=True)
    val_csv = matched_csv[matched_csv[args.id_column].isin(val_ids)].reset_index(drop=True)
    test_csv = matched_csv[matched_csv[args.id_column].isin(test_ids)].reset_index(drop=True)
    
    logger.info(f"Train: {len(train_csv)}, Val: {len(val_csv)}, Test: {len(test_csv)}")
    
    # Get structures in order matching CSV
    train_struct_list = [structure_lookup[id_] for id_ in train_csv[args.id_column]]
    val_struct_list = [structure_lookup[id_] for id_ in val_csv[args.id_column]]
    test_struct_list = [structure_lookup[id_] for id_ in test_csv[args.id_column]]
    
    # Load tokenizer
    logger.info(f"Loading tokenizer: {args.gpt2_model}")
    tokenizer = GPT2Tokenizer.from_pretrained(args.gpt2_model)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Create datasets
    logger.info("Creating datasets...")
    train_dataset = MultimodalDataset(
        texts=train_csv[args.text_column].tolist(),
        labels=train_csv[args.label_column].tolist(),
        structures=train_struct_list,
        tokenizer=tokenizer,
        max_length=args.max_length,
        cutoff=args.cutoff,
        max_neighbors=args.max_neighbors,
    )
    
    val_dataset = MultimodalDataset(
        texts=val_csv[args.text_column].tolist(),
        labels=val_csv[args.label_column].tolist(),
        structures=val_struct_list,
        tokenizer=tokenizer,
        max_length=args.max_length,
        cutoff=args.cutoff,
        max_neighbors=args.max_neighbors,
    )
    
    test_dataset = MultimodalDataset(
        texts=test_csv[args.text_column].tolist(),
        labels=test_csv[args.label_column].tolist(),
        structures=test_struct_list,
        tokenizer=tokenizer,
        max_length=args.max_length,
        cutoff=args.cutoff,
        max_neighbors=args.max_neighbors,
    )
    
    # Create dataloaders
    collate_fn = partial(collate_multimodal, line_graph=True)
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=0,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=0,
        collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=1,
        shuffle=False, num_workers=0,
        collate_fn=collate_fn
    )
    
    # Calculate class weights
    class_weights = None
    if args.use_class_weights:
        class_weights = compute_class_weights(train_csv[args.label_column].tolist())
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
    
    # Adjust steps for gradient accumulation
    steps_per_epoch = len(train_loader) // args.gradient_accumulation_steps
    if len(train_loader) % args.gradient_accumulation_steps != 0:
        steps_per_epoch += 1
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.learning_rate,
        epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=args.warmup_ratio,
    )
    
    logger.info(f"Total training steps: {total_steps}, Warmup steps: {warmup_steps}")
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
    
    # Training loop
    logger.info("Starting training...")
    for epoch in range(args.epochs):
        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {epoch + 1}/{args.epochs}")
        logger.info(f"{'='*60}")
        
        # Train
        train_loss, train_acc, train_f1 = train_epoch(
            model, train_loader, optimizer, scheduler,
            device, class_weights, line_graph=True,
            gradient_accumulation_steps=args.gradient_accumulation_steps
        )
        
        # Validate
        val_loss, val_acc, val_f1, _, _ = evaluate(
            model, val_loader, device, class_weights, line_graph=True
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
    
    test_loss, test_acc, test_f1, test_preds, test_labels = evaluate(
        model, test_loader, device, class_weights, line_graph=True
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
