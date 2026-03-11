"""
Extract hidden features from trained ALIGNN model.

This script processes train/val/test data through a trained model
and extracts hidden features (before final classification layer).
"""

import os
import sys

# Add alignn package path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'alignn'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'alignn_test'))

import json
import argparse
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

from alignn.models.alignn_atomwise import ALIGNNAtomWise, ALIGNNAtomWiseConfig
from data_prepared import get_prepared_train_val_test_loaders


def extract_hidden_features(
    config_path: str,
    model_path: str,
    data_dir: str,
    output_dir: str,
):
    """Extract hidden features from model for all datasets."""
    
    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print("=" * 60)
    print("Extracting hidden features from ALIGNN model")
    print("=" * 60)
    print(f"Model path: {model_path}")
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    
    # Device setup
    device = "cpu"
    if torch.cuda.is_available():
        device = torch.device("cuda")
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Model setup
    line_graph = config["model"]["alignn_layers"] > 0
    model_config = config["model"].copy()
    model_config["name"] = "alignn_atomwise"  # Ensure correct name
    model_config_obj = ALIGNNAtomWiseConfig(**model_config)
    net = ALIGNNAtomWise(model_config_obj)
    
    # Load trained model
    print(f"Loading model from {model_path}")
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.to(device)
    net.eval()
    
    # Process each dataset (skip train and val - already processed)
    datasets = [
        ("train_data.json", "new_train_data.json"),  # Already processed
        ("val_data.json", "new_val_data.json"),  # Already processed
        ("test_data.json", "new_test_data.json"),
    ]
    
    for input_file, output_file in datasets:
        input_path = os.path.join(data_dir, input_file)
        output_path = os.path.join(output_dir, output_file)
        
        if not os.path.exists(input_path):
            print(f"Skipping {input_file} - file not found")
            continue
        
        print(f"\nProcessing {input_file}...")
        
        # Load data loader
        _, _, data_loader, prepare_batch = get_prepared_train_val_test_loaders(
            train_file=input_path,
            val_file=input_path,
            test_file=input_path,
            target=config["target"],
            atom_features=config["atom_features"],
            neighbor_strategy=config["neighbor_strategy"],
            batch_size=1,  # Process one at a time to keep mapping
            line_graph=line_graph,
            id_tag=config["id_tag"],
            use_canonize=config.get("use_canonize", True),
            cutoff=config["cutoff"],
            cutoff_extra=config["cutoff_extra"],
            max_neighbors=config["max_neighbors"],
            classification=True,
            output_dir=output_dir,
            workers=0,
            pin_memory=False,
        )
        
        # Load original data to preserve structure
        with open(input_path, 'r') as f:
            original_data = json.load(f)
        
        # Create ID to index mapping
        id_to_idx = {item["id"]: idx for idx, item in enumerate(original_data)}
        
        # Extract features
        hidden_features_dict = {}
        
        with torch.no_grad():
            for i, (dats, jid) in enumerate(tqdm(zip(data_loader, data_loader.dataset.ids), 
                                                   total=len(data_loader.dataset.ids),
                                                   desc=f"  Extracting")):
                if line_graph:
                    g, lg = dats[0].to(device), dats[1].to(device)
                    lg = lg.local_var()
                    z = net.angle_embedding(lg.edata.pop("h"))
                else:
                    g = dats[0].to(device)
                
                g = g.local_var()
                
                # Initial node features
                x = g.ndata.pop("atom_features")
                x = net.atom_embedding(x)
                
                # Edge embedding
                r = g.edata["r"]
                bondlength = torch.norm(r, dim=1)
                
                if net.config.use_cutoff_function:
                    if net.config.multiply_cutoff:
                        from alignn.models.alignn_atomwise import cutoff_function_based_edges
                        c_off = cutoff_function_based_edges(
                            bondlength,
                            inner_cutoff=net.config.inner_cutoff,
                            exponent=net.config.exponent,
                        ).unsqueeze(dim=1)
                        y = net.edge_embedding(bondlength) * c_off
                    else:
                        from alignn.models.alignn_atomwise import cutoff_function_based_edges
                        bondlength = cutoff_function_based_edges(
                            bondlength,
                            inner_cutoff=net.config.inner_cutoff,
                            exponent=net.config.exponent,
                        )
                        y = net.edge_embedding(bondlength)
                else:
                    y = net.edge_embedding(bondlength)
                
                # ALIGNN updates
                if line_graph:
                    for alignn_layer in net.alignn_layers:
                        x, y, z = alignn_layer(g, lg, x, y, z)
                
                # Gated GCN updates
                for gcn_layer in net.gcn_layers:
                    x, y = gcn_layer(g, x, y)
                
                # Readout to get hidden features (before fc layer)
                h = net.readout(g, x)
                
                # Store hidden features
                hidden_features_dict[jid] = h.cpu().numpy().flatten().tolist()
        
        # Add hidden features to original data
        new_data = []
        for item in original_data:
            item_id = item["id"]
            new_item = item.copy()
            if item_id in hidden_features_dict:
                new_item["hidden_features"] = hidden_features_dict[item_id]
            else:
                print(f"  Warning: ID {item_id} not found in extracted features")
                new_item["hidden_features"] = [0.0] * config["model"]["hidden_features"]
            new_data.append(new_item)
        
        # Save new data
        print(f"  Saving to {output_path}...")
        with open(output_path, 'w') as f:
            json.dump(new_data, f)
        
        print(f"  Done! Processed {len(new_data)} samples with {config['model']['hidden_features']}-dim features")
    
    print(f"\n{'=' * 60}")
    print("Feature extraction completed!")
    print(f"Output files saved to: {output_dir}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Extract hidden features from trained ALIGNN model"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="../alignn_test/prepared_data_merged/config_classification.json",
        help="Path to config JSON file"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="../alignn_test/output_ordering/best_model.pt",
        help="Path to trained model"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="../alignn_test/prepared_data_merged",
        help="Directory containing train/val/test JSON files"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="gnn_dataset",
        help="Output directory for new data files"
    )
    
    args = parser.parse_args()
    
    # Convert to absolute paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, args.config) if not os.path.isabs(args.config) else args.config
    model_path = os.path.join(script_dir, args.model) if not os.path.isabs(args.model) else args.model
    data_dir = os.path.join(script_dir, args.data_dir) if not os.path.isabs(args.data_dir) else args.data_dir
    output_dir = os.path.join(script_dir, args.output_dir) if not os.path.isabs(args.output_dir) else args.output_dir
    
    extract_hidden_features(
        config_path=config_path,
        model_path=model_path,
        data_dir=data_dir,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
