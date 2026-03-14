"""
Prepare graph data for GNN training.

This script:
1. Loads crystal structure data
2. Builds crystal graphs using ALIGNN graph construction
3. Creates line graphs for angle features
4. Saves graph data to disk in chunks for memory efficiency
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from tqdm import tqdm
import logging
from datetime import datetime
from typing import Dict, List
import math

# Add alignn paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'alignn'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'alignn_test'))

import dgl
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
# Graph Construction Functions
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


def load_prepared_data(filepath: str) -> List[Dict]:
    """Load prepared JSON data file with structure data."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data


def build_single_graph(item: Dict, cutoff: float, max_neighbors: int, 
                       feature_lookup: np.ndarray) -> tuple:
    """Build a single graph from structure data.
    
    Returns:
        Tuple of (graph, line_graph, label, id) or None if failed
    """
    try:
        atoms = Atoms.from_dict(item["atoms"])
        g = atoms_to_graph(atoms, cutoff=cutoff, max_neighbors=max_neighbors)
        
        # Skip graphs with no edges (can't create line graph)
        if g.num_edges() == 0:
            return None
        
        # Apply atom features
        z = g.ndata["atom_features"].squeeze().int()
        feat = torch.tensor(feature_lookup[z], dtype=torch.float32)
        if g.num_nodes() == 1:
            feat = feat.unsqueeze(0)
        g.ndata["atom_features"] = feat
        
        # Create line graph (Windows-compatible)
        lg = create_line_graph(g)
        
        # Get label from structure data
        if 'mp_ordering' in item:
            label = item['mp_ordering']
        elif 'target' in item:
            label = item['target']
        else:
            label = -1  # Unknown label
        
        return (g, lg, label, item['id'])
        
    except Exception as e:
        logger.warning(f"Failed to build graph for {item.get('id', 'unknown')}: {e}")
        return None


def save_chunk(graphs: List, line_graphs: List, labels: List, ids: List,
               output_dir: str, split_name: str, chunk_idx: int):
    """Save a chunk of graphs to disk."""
    chunk_prefix = f"{split_name}_chunk_{chunk_idx:04d}"
    
    graph_path = os.path.join(output_dir, f'{chunk_prefix}_graphs.bin')
    line_graph_path = os.path.join(output_dir, f'{chunk_prefix}_line_graphs.bin')
    meta_path = os.path.join(output_dir, f'{chunk_prefix}_meta.pt')
    
    dgl.save_graphs(graph_path, graphs)
    dgl.save_graphs(line_graph_path, line_graphs)
    
    meta_data = {
        'labels': labels,
        'ids': ids,
    }
    torch.save(meta_data, meta_path)
    
    return chunk_prefix


def build_and_save_graphs_chunked(structures: List[Dict], output_dir: str,
                                   split_name: str, chunk_size: int = 1000,
                                   cutoff: float = 8.0, max_neighbors: int = 12,
                                   atom_features: str = "cgcnn"):
    """Build graphs from structures and save to disk in chunks."""
    feature_lookup = get_attribute_lookup(atom_features)
    
    # Temporary storage for current chunk
    chunk_graphs = []
    chunk_line_graphs = []
    chunk_labels = []
    chunk_ids = []
    
    total_valid = 0
    total_failed = 0
    chunk_idx = 0
    chunk_files = []
    
    logger.info(f"Building {len(structures)} graphs in chunks of {chunk_size}...")
    
    for idx, item in enumerate(tqdm(structures, desc=f"Building {split_name} graphs")):
        result = build_single_graph(item, cutoff, max_neighbors, feature_lookup)
        
        if result is not None:
            g, lg, label, sample_id = result
            chunk_graphs.append(g)
            chunk_line_graphs.append(lg)
            chunk_labels.append(label)
            chunk_ids.append(sample_id)
            total_valid += 1
            
            # Save chunk when full
            if len(chunk_graphs) >= chunk_size:
                chunk_file = save_chunk(
                    chunk_graphs, chunk_line_graphs, chunk_labels, chunk_ids,
                    output_dir, split_name, chunk_idx
                )
                chunk_files.append(chunk_file)
                logger.info(f"Saved chunk {chunk_idx} with {len(chunk_graphs)} graphs")
                
                # Reset chunk storage
                chunk_graphs = []
                chunk_line_graphs = []
                chunk_labels = []
                chunk_ids = []
                chunk_idx += 1
        else:
            total_failed += 1
    
    # Save remaining graphs in final chunk
    if chunk_graphs:
        chunk_file = save_chunk(
            chunk_graphs, chunk_line_graphs, chunk_labels, chunk_ids,
            output_dir, split_name, chunk_idx
        )
        chunk_files.append(chunk_file)
        logger.info(f"Saved final chunk {chunk_idx} with {len(chunk_graphs)} graphs")
    
    logger.info(f"Successfully built {total_valid} graphs, failed: {total_failed}")
    logger.info(f"Saved to {len(chunk_files)} chunks")
    
    return {
        'total_graphs': total_valid,
        'total_failed': total_failed,
        'num_chunks': len(chunk_files),
        'chunk_files': chunk_files,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description='Prepare graph data for GNN training'
    )
    
    # Data arguments
    parser.add_argument('--train_structures', type=str,
                        default='../alignn_test/prepared_data_merged/train_data.json',
                        help='Path to training structure JSON')
    parser.add_argument('--val_structures', type=str,
                        default='../alignn_test/prepared_data_merged/val_data.json',
                        help='Path to validation structure JSON')
    parser.add_argument('--test_structures', type=str,
                        default='../alignn_test/prepared_data_merged/test_data.json',
                        help='Path to test structure JSON')
    
    # Graph construction arguments
    parser.add_argument('--cutoff', type=float, default=8.0,
                        help='Distance cutoff for graph construction')
    parser.add_argument('--max_neighbors', type=int, default=12,
                        help='Maximum number of neighbors')
    parser.add_argument('--atom_features', type=str, default='cgcnn',
                        help='Atom feature type')
    
    # Chunking arguments
    parser.add_argument('--chunk_size', type=int, default=1000,
                        help='Number of graphs per chunk (default: 1000)')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='llm_gnn_data',
                        help='Output directory for prepared data')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load structure data
    logger.info("Loading structure data...")
    train_structures = load_prepared_data(args.train_structures)
    val_structures = load_prepared_data(args.val_structures)
    test_structures = load_prepared_data(args.test_structures)
    
    logger.info(f"Loaded - Train: {len(train_structures)}, Val: {len(val_structures)}, Test: {len(test_structures)}")
    
    # Build and save graphs for each split in chunks
    all_stats = {}
    for split_name, structures in [('train', train_structures), ('val', val_structures), ('test', test_structures)]:
        stats = build_and_save_graphs_chunked(
            structures, args.output_dir, split_name,
            chunk_size=args.chunk_size,
            cutoff=args.cutoff,
            max_neighbors=args.max_neighbors,
            atom_features=args.atom_features
        )
        all_stats[split_name] = stats
        logger.info(f"Built {stats['total_graphs']} {split_name} graphs in {stats['num_chunks']} chunks")
    
    # Save metadata
    metadata = {
        'cutoff': args.cutoff,
        'max_neighbors': args.max_neighbors,
        'atom_features': args.atom_features,
        'chunk_size': args.chunk_size,
        'num_labels': NUM_LABELS,
        'label_mapping': ORDERING_LABELS,
        'splits': {
            split: {
                'total_graphs': stats['total_graphs'],
                'num_chunks': stats['num_chunks'],
                'chunk_files': stats['chunk_files'],
            }
            for split, stats in all_stats.items()
        },
        'timestamp': datetime.now().isoformat(),
    }
    
    metadata_path = os.path.join(args.output_dir, 'gnn_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved metadata to {metadata_path}")
    
    logger.info("\nGNN data preparation completed!")
    logger.info(f"Output directory: {args.output_dir}")
    
    return metadata


if __name__ == '__main__':
    main()
