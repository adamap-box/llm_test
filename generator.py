"""
Crystal Structure Text Description Generator

This script generates text descriptions of crystal structures from merged_dataset.json.
These descriptions can be used for NLP-based materials science models (like BERT embeddings).

Based on ALIGNN-BERT-TL/generater.py implementation.
"""

import json
import numpy as np
import argparse
import logging
import os
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import warnings

from jarvis.core.atoms import Atoms
from jarvis.io.vasp.inputs import Poscar
from jarvis.analysis.structure.spacegroup import Spacegroup3D
from jarvis.analysis.diffraction.xrd import XRD
from jarvis.core.specie import Specie

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Configure argument parser
parser = argparse.ArgumentParser(description='Generate text descriptions from merged_dataset.json')
parser.add_argument('--data_path', help='Path to merged_dataset.json', 
                    default='merged_dataset.json', type=str, required=False)
parser.add_argument('--start', default=0, type=int, required=False,
                    help='Start index for processing entries')
parser.add_argument('--end', type=int, required=False,
                    help='End index for processing entries')
parser.add_argument('--output_dir', help='Directory to save output CSV', 
                    default=None, type=str, required=False)
parser.add_argument('--text', help='Text generation method', 
                    choices=['raw', 'chemnlp', 'combo', 'all'], 
                    default='chemnlp', type=str, required=False)
parser.add_argument('--skip_sentence', help='Skip specific topic in description',
                    choices=['chemical', 'structure', 'bond', 'none'],
                    default='none', required=False)
parser.add_argument('--include_spg', help='Include spacegroup analysis (slower but more detailed)',
                    action='store_true', default=False)
args, _ = parser.parse_known_args()


def pymatgen_to_jarvis(entry):
    """
    Convert pymatgen-style structure dict (from merged_dataset.json) to JARVIS Atoms.
    
    Handles two structure formats:
    
    1. Magnetic dataset format:
    {
        "structure": {
            "lattice": {"matrix": [[...], [...], [...]], ...},
            "atoms": [{"species": "Fe", "abc": [...], ...}, ...]
        }
    }
    
    2. Materials Project format:
    {
        "structure": {
            "lattice": {"matrix": [[...], [...], [...]], ...},
            "sites": [{"species": [{"element": "Fe", "occu": 1.0}], "abc": [...], ...}, ...]
        }
    }
    
    JARVIS Atoms.from_dict expects:
    {
        "lattice_mat": [[...], [...], [...]],
        "coords": [[...], [...], ...],  # fractional coordinates
        "elements": ["Fe", "O", ...],
        "cartesian": False
    }
    """
    structure = entry.get('structure')
    
    if structure is None:
        # Try direct 'atoms' field (some entries may have JARVIS format)
        atoms_dict = entry.get('atoms')
        if atoms_dict:
            # Add props if missing
            if 'props' not in atoms_dict:
                n_atoms = len(atoms_dict.get('elements', atoms_dict.get('coords', [])))
                atoms_dict['props'] = [''] * n_atoms
            return Atoms.from_dict(atoms_dict)
        return None
    
    lattice = structure.get('lattice', {})
    lattice_matrix = lattice.get('matrix')
    
    if lattice_matrix is None:
        return None
    
    # Try 'atoms' first (magnetic dataset format), then 'sites' (MP format)
    atoms_list = structure.get('atoms', [])
    sites_list = structure.get('sites', [])
    
    # Extract elements and fractional coordinates
    elements = []
    coords = []
    
    if atoms_list:
        # Magnetic dataset format: atoms with species string and abc coords
        for atom in atoms_list:
            species = atom.get('species', atom.get('label', 'X'))
            abc = atom.get('abc')  # fractional coordinates
            
            if abc is None:
                continue
                
            elements.append(species)
            coords.append(abc)
    
    elif sites_list:
        # Materials Project format: sites with species list and abc coords
        for site in sites_list:
            # Get element from species list
            species_list = site.get('species', [])
            if species_list and isinstance(species_list, list):
                # Take the element with highest occupancy
                element = species_list[0].get('element', 'X')
            else:
                element = site.get('label', 'X')
            
            abc = site.get('abc')  # fractional coordinates
            
            if abc is None:
                continue
            
            elements.append(element)
            coords.append(abc)
    
    if not elements or not coords:
        return None
    
    # Create JARVIS Atoms dict (props field required by JARVIS)
    jarvis_dict = {
        'lattice_mat': lattice_matrix,
        'coords': coords,
        'elements': elements,
        'cartesian': False,
        'props': [''] * len(elements)  # empty props for each atom
    }
    
    return Atoms.from_dict(jarvis_dict)


def atoms_describer(atoms, xrd_peaks=5, xrd_round=1, cutoff=4, take_n_bonds=2, include_spg=True):
    """
    Generate detailed text description of an atomic structure.
    
    Args:
        atoms: JARVIS Atoms object
        xrd_peaks: Number of top XRD peaks to include
        xrd_round: Decimal places for XRD peak rounding
        cutoff: Distance cutoff for neighbor analysis
        take_n_bonds: Number of bond distances to report per bond type
        include_spg: Whether to include spacegroup analysis (slower)
    
    Returns:
        dict: Dictionary with chemical_info and structure_info
    """
    info = {}
    
    # Spacegroup analysis (optional, can be slow)
    spg = None
    if include_spg:
        try:
            spg = Spacegroup3D(atoms)
        except Exception as e:
            logging.debug(f"Spacegroup analysis failed: {e}")
            spg = None
    
    # XRD analysis
    try:
        theta, d_hkls, intens = XRD().simulate(atoms=atoms)
    except Exception:
        theta = []
    
    # Bond distance analysis
    dists = defaultdict(list)
    elements = atoms.elements
    try:
        for i in atoms.get_all_neighbors(r=cutoff):
            for j in i:
                key = "-".join(sorted([elements[j[0]], elements[j[1]]]))
                dists[key].append(j[2])
    except Exception:
        pass
    
    bond_distances = {}
    for i, j in dists.items():
        dist = sorted(set([round(k, 2) for k in j]))
        if len(dist) >= take_n_bonds:
            dist = dist[0:take_n_bonds]
        bond_distances[i] = ", ".join(map(str, dist))
    
    # Atomic fractions
    fracs = {}
    for i, j in (atoms.composition.atomic_fraction).items():
        fracs[i] = round(j, 3)
    
    # Chemical information
    chem_info = {
        "atomic_formula": atoms.composition.reduced_formula,
        "prototype": atoms.composition.prototype,
        "molecular_weight": round(atoms.composition.weight / 2, 2),
        "atomic_fraction": fracs,
        "atomic_X": ", ".join(map(str, [Specie(s).X for s in atoms.uniq_species])),
        "atomic_Z": ", ".join(map(str, [Specie(s).Z for s in atoms.uniq_species])),
    }
    
    # Structure information
    struct_info = {
        "lattice_parameters": ", ".join(map(str, [round(j, 2) for j in atoms.lattice.abc])),
        "lattice_angles": ", ".join(map(str, [round(j, 2) for j in atoms.lattice.angles])),
        "top_k_xrd_peaks": ", ".join(
            map(str, sorted(list(set([round(i, xrd_round) for i in theta])))[0:xrd_peaks])
        ) if theta else "N/A",
        "density": round(atoms.density, 3),
        "bond_distances": bond_distances,
    }
    
    # Add spacegroup info if available
    if spg:
        try:
            struct_info["spg_number"] = spg.space_group_number
            struct_info["spg_symbol"] = spg.space_group_symbol
            struct_info["crystal_system"] = spg.crystal_system
            struct_info["point_group"] = spg.point_group_symbol
            struct_info["wyckoff"] = ", ".join(list(set(spg._dataset["wyckoffs"])))
            struct_info["natoms_primitive"] = spg.primitive_atoms.num_atoms
            struct_info["natoms_conventional"] = spg.conventional_standard_structure.num_atoms
        except Exception as e:
            logging.debug(f"Spacegroup attribute extraction failed: {e}")
    
    info["chemical_info"] = chem_info
    info["structure_info"] = struct_info
    
    return info


def describe_chemical_data(info, skip="none"):
    """
    Convert atoms_describer output to natural language description.
    
    Args:
        info: Dictionary from atoms_describer()
        skip: Topic to skip ('chemical', 'structure', 'bond', or 'none')
    
    Returns:
        str: Natural language description of the structure
    """
    description = ""
    
    if 'chemical_info' in info and skip != 'chemical':
        description += "The chemical information includes: "
        chem_info = info['chemical_info']
        description += f"The chemical has an atomic formula of {chem_info.get('atomic_formula', 'N/A')} "
        description += f"with a prototype of {chem_info.get('prototype', 'N/A')}; "
        description += f"Its molecular weight is {chem_info.get('molecular_weight', 'N/A')} g/mol; "
        
        atomic_fraction = chem_info.get('atomic_fraction', {})
        if isinstance(atomic_fraction, dict):
            frac_str = ", ".join([f"{k}: {v}" for k, v in atomic_fraction.items()])
        else:
            frac_str = str(atomic_fraction)
        description += f"The atomic fractions are {frac_str}, "
        description += f"and the atomic values X and Z are {chem_info.get('atomic_X', 'N/A')} "
        description += f"and {chem_info.get('atomic_Z', 'N/A')}, respectively. "

    if 'structure_info' in info and skip != 'structure':
        description += "The structure information includes: "
        struct_info = info['structure_info']
        description += f"The lattice parameters are {struct_info.get('lattice_parameters', 'N/A')} "
        description += f"with angles {struct_info.get('lattice_angles', 'N/A')} degrees; "
        
        if 'spg_number' in struct_info:
            description += f"The space group number is {struct_info.get('spg_number', 'N/A')} "
            description += f"with the symbol {struct_info.get('spg_symbol', 'N/A')}; "
        
        description += f"The top K XRD peaks are found at {struct_info.get('top_k_xrd_peaks', 'N/A')} degrees; "
        description += f"The material has a density of {struct_info.get('density', 'N/A')} g/cm³"
        
        if 'crystal_system' in struct_info:
            description += f", crystallizes in a {struct_info.get('crystal_system', 'N/A')} system, "
            description += f"and has a point group of {struct_info.get('point_group', 'N/A')}; "
            description += f"The Wyckoff positions are {struct_info.get('wyckoff', 'N/A')}; "
            description += f"The number of atoms in the primitive and conventional cells are "
            description += f"{struct_info.get('natoms_primitive', 'N/A')} and "
            description += f"{struct_info.get('natoms_conventional', 'N/A')}, respectively; "
        else:
            description += "; "
        
        if 'bond_distances' in struct_info and skip != 'bond':
            bond_distances = struct_info['bond_distances']
            if bond_distances:
                bond_descriptions = ", ".join([f"{bond}: {distance}" for bond, distance in bond_distances.items()])
                description += f"The bond distances are as follows: {bond_descriptions}. "
    
    return description.strip()


def info_to_text(info):
    """
    Convert atoms_describer output to formatted text (alternative format).
    
    Args:
        info: Dictionary from atoms_describer()
    
    Returns:
        str: Formatted text description
    """
    line = f"The number of atoms information. "
    
    for category, data in info.items():
        if not isinstance(data, dict):
            line += f"The {category} is {data}. "
        else:
            for key, value in data.items():
                if isinstance(value, dict):
                    tmp = " ".join([f"{k}: {v}" for k, v in value.items()])
                else:
                    tmp = str(value)
                line += f"The {key} is {tmp}. "
    
    return line


def get_crystal_string_t(atoms, include_spg=True):
    """
    Generate combo format: chemnlp description + crystal string.
    
    Args:
        atoms: JARVIS Atoms object
        include_spg: Whether to include spacegroup analysis
    
    Returns:
        str: Combined description with lattice and coordinates
    """
    lengths = atoms.lattice.abc
    angles = atoms.lattice.angles
    atom_ids = atoms.elements
    frac_coords = atoms.frac_coords
    
    crystal_str = (
        " ".join(["{0:.2f}".format(x) for x in lengths])
        + "#\n"
        + " ".join([str(int(x)) for x in angles])
        + "@\n"
        + "\n".join(
            [
                str(t) + " " + " ".join(["{0:.3f}".format(x) for x in c]) + "&"
                for t, c in zip(atom_ids, frac_coords)
            ]
        )
    )
    
    # Get chemnlp description
    info = atoms_describer(atoms, include_spg=include_spg)
    text_desc = info_to_text(info)
    
    crystal_str = text_desc + "\n*\n" + crystal_str
    return crystal_str


def get_raw_poscar(atoms):
    """
    Generate raw POSCAR format string.
    
    Args:
        atoms: JARVIS Atoms object
    
    Returns:
        str: POSCAR format string
    """
    try:
        return Poscar(atoms).to_string()
    except Exception as e:
        logging.warning(f"POSCAR generation failed: {e}")
        return ""


def get_text(atoms, text_type, skip_sentence="none", include_spg=True):
    """
    Generate text description based on specified type.
    
    Args:
        atoms: JARVIS Atoms object
        text_type: Type of text generation ('raw', 'chemnlp', 'combo')
        skip_sentence: Topic to skip in chemnlp mode
        include_spg: Include spacegroup analysis
    
    Returns:
        str: Generated text description
    """
    if text_type == 'raw':
        return get_raw_poscar(atoms)
    elif text_type == 'chemnlp':
        info = atoms_describer(atoms, include_spg=include_spg)
        return describe_chemical_data(info, skip=skip_sentence)
    elif text_type == 'combo':
        return get_crystal_string_t(atoms, include_spg=include_spg)
    else:
        raise ValueError(f"Unknown text type: {text_type}")


def load_merged_dataset(data_path):
    """
    Load merged_dataset.json file.
    
    Args:
        data_path: Path to the JSON file
    
    Returns:
        list: List of entry dictionaries
    """
    logging.info(f"Loading dataset from {data_path}...")
    
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Handle different JSON structures
    if isinstance(data, list):
        entries = data
    elif isinstance(data, dict):
        if 'entries' in data:
            entries = data['entries']
        else:
            entries = list(data.values())
    else:
        raise ValueError(f"Unexpected data type: {type(data)}")
    
    logging.info(f"Loaded {len(entries)} entries")
    return entries


def main(args):
    """Main processing function."""
    
    # Load dataset
    entries = load_merged_dataset(args.data_path)
    
    # Determine range
    start = args.start
    end = len(entries)
    if args.end:
        end = min(args.end, len(entries))
    
    logging.info(f"Processing entries {start} to {end}")
    
    # Process entries
    text_types = ['raw', 'chemnlp', 'combo'] if args.text == 'all' else [args.text]
    
    for text_type in text_types:
        text_dic = defaultdict(list)
        err_ct = 0
        success_ct = 0
        
        for entry in tqdm(entries[start:end], desc=f"Generating {text_type} descriptions"):
            try:
                # Convert structure to JARVIS Atoms
                atoms = pymatgen_to_jarvis(entry)
                
                if atoms is None:
                    err_ct += 1
                    logging.debug(f"Failed to convert structure for entry: {entry.get('mp_id', 'unknown')}")
                    continue
                
                # Generate text description
                text = get_text(atoms, text_type, 
                              skip_sentence=args.skip_sentence, 
                              include_spg=args.include_spg)
                
                # Store results
                text_dic['mp_id'].append(entry.get('mp_id', entry.get('material_id', '')))
                text_dic['formula'].append(entry.get('formula', entry.get('mp_formula', '')))
                text_dic['mp_ordering'].append(entry.get('mp_ordering', entry.get('ordering', '')))
                text_dic['transition_type'].append(entry.get('transition_type', ''))
                text_dic['source'].append(entry.get('source', ''))
                text_dic['text'].append(text)
                
                success_ct += 1
                
            except Exception as e:
                err_ct += 1
                logging.warning(f"Failed to process entry {entry.get('mp_id', 'unknown')}: {e}")
        
        # Create DataFrame and save
        df_text = pd.DataFrame.from_dict(text_dic)
        
        # Generate output filename
        output_file = f"{text_type}_{start}_{end}_skip_{args.skip_sentence}"
        if args.include_spg:
            output_file += "_spg"
        output_file += ".csv"
        
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            output_file = os.path.join(args.output_dir, output_file)
        
        df_text.to_csv(output_file, index=False)
        
        logging.info(f"Saved {success_ct} descriptions to {output_file}")
        logging.info(f"Success: {success_ct}, Errors: {err_ct}")
    
    logging.info("Text generation complete!")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    main(args)
