"""
Validate Output CSV against merged_dataset.json

This script:
1. Checks if the output CSV matches the merged_dataset.json entries
2. Provides functions to lookup and print content by mp_id
"""

import json
import pandas as pd
import argparse
import logging
from typing import Optional, Dict, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_merged_dataset(data_path: str) -> list:
    """Load merged_dataset.json file."""
    logging.info(f"Loading dataset from {data_path}...")
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if isinstance(data, list):
        entries = data
    elif isinstance(data, dict):
        entries = data.get('entries', list(data.values()))
    else:
        raise ValueError(f"Unexpected data type: {type(data)}")
    
    logging.info(f"Loaded {len(entries)} entries from JSON")
    return entries


def load_output_csv(csv_path: str) -> pd.DataFrame:
    """Load the output CSV file."""
    logging.info(f"Loading CSV from {csv_path}...")
    df = pd.read_csv(csv_path)
    logging.info(f"Loaded {len(df)} entries from CSV")
    return df


def validate_output(json_path: str, csv_path: str) -> Dict[str, Any]:
    """
    Validate that the output CSV matches the merged_dataset.json.
    
    Returns a dict with validation results.
    """
    entries = load_merged_dataset(json_path)
    df = load_output_csv(csv_path)
    
    results = {
        'json_count': len(entries),
        'csv_count': len(df),
        'match': False,
        'missing_in_csv': [],
        'extra_in_csv': [],
        'mismatched_formulas': []
    }
    
    # Build lookup from JSON entries
    json_lookup = {}
    for entry in entries:
        mp_id = entry.get('mp_id', entry.get('material_id', ''))
        if mp_id:
            json_lookup[mp_id] = entry
    
    # Build lookup from CSV
    csv_lookup = {}
    for _, row in df.iterrows():
        mp_id = row.get('mp_id', '')
        if pd.notna(mp_id) and mp_id:
            csv_lookup[str(mp_id)] = row
    
    results['json_unique_ids'] = len(json_lookup)
    results['csv_unique_ids'] = len(csv_lookup)
    
    # Check for missing entries in CSV
    for mp_id in json_lookup:
        if mp_id not in csv_lookup:
            results['missing_in_csv'].append(mp_id)
    
    # Check for extra entries in CSV
    for mp_id in csv_lookup:
        if mp_id not in json_lookup:
            results['extra_in_csv'].append(mp_id)
    
    # Check formula consistency
    for mp_id in json_lookup:
        if mp_id in csv_lookup:
            json_formula = json_lookup[mp_id].get('formula', json_lookup[mp_id].get('mp_formula', ''))
            csv_formula = csv_lookup[mp_id].get('formula', '')
            if json_formula and csv_formula and str(json_formula) != str(csv_formula):
                results['mismatched_formulas'].append({
                    'mp_id': mp_id,
                    'json_formula': json_formula,
                    'csv_formula': csv_formula
                })
    
    # Overall match check
    results['match'] = (
        results['json_count'] == results['csv_count'] and
        len(results['missing_in_csv']) == 0 and
        len(results['extra_in_csv']) == 0
    )
    
    return results


def print_validation_report(results: Dict[str, Any]):
    """Print validation report."""
    print("\n" + "="*60)
    print("VALIDATION REPORT")
    print("="*60)
    
    print(f"\nEntry Counts:")
    print(f"  JSON entries:     {results['json_count']}")
    print(f"  CSV entries:      {results['csv_count']}")
    print(f"  JSON unique IDs:  {results['json_unique_ids']}")
    print(f"  CSV unique IDs:   {results['csv_unique_ids']}")
    
    print(f"\nValidation Status: {'✓ PASSED' if results['match'] else '✗ FAILED'}")
    
    if results['missing_in_csv']:
        print(f"\nMissing in CSV ({len(results['missing_in_csv'])} entries):")
        for mp_id in results['missing_in_csv'][:10]:
            print(f"  - {mp_id}")
        if len(results['missing_in_csv']) > 10:
            print(f"  ... and {len(results['missing_in_csv']) - 10} more")
    
    if results['extra_in_csv']:
        print(f"\nExtra in CSV ({len(results['extra_in_csv'])} entries):")
        for mp_id in results['extra_in_csv'][:10]:
            print(f"  - {mp_id}")
        if len(results['extra_in_csv']) > 10:
            print(f"  ... and {len(results['extra_in_csv']) - 10} more")
    
    if results['mismatched_formulas']:
        print(f"\nMismatched formulas ({len(results['mismatched_formulas'])} entries):")
        for item in results['mismatched_formulas'][:5]:
            print(f"  - {item['mp_id']}: JSON='{item['json_formula']}' vs CSV='{item['csv_formula']}'")
    
    print("\n" + "="*60)


def lookup_by_mp_id(mp_id: str, json_path: str = None, csv_path: str = None, 
                    json_data: list = None, csv_df: pd.DataFrame = None) -> Dict[str, Any]:
    """
    Lookup entry by mp_id from both JSON and CSV.
    
    Args:
        mp_id: Material ID to lookup
        json_path: Path to JSON file (optional if json_data provided)
        csv_path: Path to CSV file (optional if csv_df provided)
        json_data: Pre-loaded JSON entries (optional)
        csv_df: Pre-loaded CSV DataFrame (optional)
    
    Returns:
        Dict with 'json' and 'csv' entries
    """
    result = {'mp_id': mp_id, 'json': None, 'csv': None}
    
    # Load JSON if needed
    if json_data is None and json_path:
        json_data = load_merged_dataset(json_path)
    
    # Load CSV if needed
    if csv_df is None and csv_path:
        csv_df = load_output_csv(csv_path)
    
    # Find in JSON
    if json_data:
        for entry in json_data:
            entry_id = entry.get('mp_id', entry.get('material_id', ''))
            if entry_id == mp_id:
                result['json'] = entry
                break
    
    # Find in CSV
    if csv_df is not None:
        matches = csv_df[csv_df['mp_id'] == mp_id]
        if len(matches) > 0:
            result['csv'] = matches.iloc[0].to_dict()
    
    return result


def print_full_json(json_entry: Dict[str, Any], indent: int = 2):
    """
    Print full JSON entry with proper formatting.
    
    Args:
        json_entry: JSON dictionary to print
        indent: Indentation level for JSON formatting
    """
    import json as json_module
    
    def default_serializer(obj):
        """Handle non-serializable objects."""
        if hasattr(obj, '__dict__'):
            return str(obj)
        return str(obj)
    
    print(json_module.dumps(json_entry, indent=indent, default=default_serializer, ensure_ascii=False))


def print_crystal_structure(structure: Dict[str, Any], mp_id: str = None, formula: str = None):
    """
    Display crystal structure information in a formatted way.
    
    Args:
        structure: Structure dictionary from JSON entry
        mp_id: Material ID (optional, for header)
        formula: Chemical formula (optional, for header)
    """
    if not structure:
        print("No structure data available.")
        return
    
    print("\n" + "="*70)
    header = "CRYSTAL STRUCTURE"
    if mp_id:
        header += f" - {mp_id}"
    if formula:
        header += f" ({formula})"
    print(header)
    print("="*70)
    
    # Lattice information
    lattice = structure.get('lattice', {})
    if lattice:
        print("\n┌─ LATTICE PARAMETERS ─────────────────────────────────────────────┐")
        print(f"│  a = {lattice.get('a', 'N/A'):>10.6f} Å    α = {lattice.get('alpha', 'N/A'):>10.4f}°              │")
        print(f"│  b = {lattice.get('b', 'N/A'):>10.6f} Å    β = {lattice.get('beta', 'N/A'):>10.4f}°              │")
        print(f"│  c = {lattice.get('c', 'N/A'):>10.6f} Å    γ = {lattice.get('gamma', 'N/A'):>10.4f}°              │")
        print(f"│  Volume = {lattice.get('volume', 'N/A'):>12.6f} Å³                                    │")
        print("└──────────────────────────────────────────────────────────────────┘")
        
        # Lattice matrix
        matrix = lattice.get('matrix', [])
        if matrix:
            print("\n┌─ LATTICE MATRIX ─────────────────────────────────────────────────┐")
            for i, row in enumerate(matrix):
                vec_name = ['a', 'b', 'c'][i]
                print(f"│  {vec_name}: [{row[0]:>12.8f}, {row[1]:>12.8f}, {row[2]:>12.8f}]    │")
            print("└──────────────────────────────────────────────────────────────────┘")
    
    # Space group (if available at structure level)
    spacegroup = structure.get('spacegroup', structure.get('space_group', {}))
    if spacegroup:
        if isinstance(spacegroup, dict):
            sg_symbol = spacegroup.get('symbol', spacegroup.get('international', 'N/A'))
            sg_number = spacegroup.get('number', 'N/A')
        else:
            sg_symbol = str(spacegroup)
            sg_number = 'N/A'
        print(f"\n  Space Group: {sg_symbol} (#{sg_number})")
    
    # Atomic positions
    atoms = structure.get('atoms', structure.get('sites', []))
    if atoms:
        print(f"\n┌─ ATOMIC POSITIONS ({len(atoms)} atoms) ──────────────────────────────────┐")
        print("│  #   Element      x            y            z          │")
        print("│" + "─"*58 + "│")
        
        for i, atom in enumerate(atoms):
            # Get element - handle different formats
            if 'label' in atom:
                element = atom['label']
            elif 'species' in atom:
                species = atom['species']
                if isinstance(species, list) and len(species) > 0:
                    spec = species[0]
                    if isinstance(spec, dict):
                        element = spec.get('element', spec.get('symbol', 'X'))
                    else:
                        element = str(spec)
                else:
                    element = str(species)
            elif 'element' in atom:
                element = atom['element']
            else:
                element = 'X'
            
            # Get coordinates
            coords = atom.get('abc', atom.get('xyz', atom.get('coords', [0, 0, 0])))
            if len(coords) >= 3:
                x, y, z = coords[0], coords[1], coords[2]
                print(f"│  {i+1:<3} {element:<8}  {x:>10.6f}   {y:>10.6f}   {z:>10.6f}   │")
        
        print("└──────────────────────────────────────────────────────────────────┘")
        
        # Show properties if available (e.g., magnetic moments)
        has_props = any('properties' in atom or 'magmom' in atom for atom in atoms)
        if has_props:
            print("\n┌─ ATOMIC PROPERTIES ───────────────────────────────────────────────┐")
            for i, atom in enumerate(atoms):
                element = atom.get('label', atom.get('species', [{'element': 'X'}])[0].get('element', 'X') if isinstance(atom.get('species', []), list) else 'X')
                props = atom.get('properties', {})
                magmom = atom.get('magmom', props.get('magmom', None))
                if magmom is not None:
                    print(f"│  {i+1:<3} {element:<8}  magmom = {magmom}")
            print("└──────────────────────────────────────────────────────────────────┘")
    
    print("\n" + "="*70)


def print_entry(mp_id: str, json_path: str, csv_path: str, show_text: bool = True, 
                full_json: bool = False, show_structure: bool = False):
    """
    Print entry content for a given mp_id.
    
    Args:
        mp_id: Material ID to print
        json_path: Path to JSON file
        csv_path: Path to CSV file
        show_text: Whether to show full text description
        full_json: Whether to show full JSON entry (all fields)
        show_structure: Whether to display crystal structure
    """
    result = lookup_by_mp_id(mp_id, json_path=json_path, csv_path=csv_path)
    
    print("\n" + "="*60)
    print(f"ENTRY: {mp_id}")
    print("="*60)
    
    if result['json']:
        print("\n--- JSON Entry ---")
        json_entry = result['json']
        
        if full_json:
            # Print full JSON with all fields
            print_full_json(json_entry)
        elif show_structure:
            # Print crystal structure display
            structure = json_entry.get('structure')
            formula = json_entry.get('formula', json_entry.get('mp_formula', ''))
            print_crystal_structure(structure, mp_id, formula)
        else:
            # Print summary
            print(f"  MP ID:           {json_entry.get('mp_id', json_entry.get('material_id', 'N/A'))}")
            print(f"  Formula:         {json_entry.get('formula', json_entry.get('mp_formula', 'N/A'))}")
            print(f"  MP Ordering:     {json_entry.get('mp_ordering', json_entry.get('ordering', 'N/A'))}")
            print(f"  Transition Type: {json_entry.get('transition_type', 'N/A')}")
            print(f"  Source:          {json_entry.get('source', 'N/A')}")
            print(f"  Has Structure:   {'Yes' if json_entry.get('structure') else 'No'}")
            
            # Show all top-level keys
            print(f"\n  All fields: {', '.join(json_entry.keys())}")
            
            # Show structure info if available
            structure = json_entry.get('structure')
            if structure:
                lattice = structure.get('lattice', {})
                print(f"\n  Lattice a,b,c:   {lattice.get('a', 'N/A')}, {lattice.get('b', 'N/A')}, {lattice.get('c', 'N/A')}")
                atoms = structure.get('atoms', structure.get('sites', []))
                print(f"  Num atoms/sites: {len(atoms)}")
            
            # Show other numeric/string fields
            skip_fields = {'structure', 'mp_id', 'material_id', 'formula', 'mp_formula', 
                          'mp_ordering', 'ordering', 'transition_type', 'source'}
            other_fields = {k: v for k, v in json_entry.items() 
                           if k not in skip_fields and not isinstance(v, (dict, list))}
            if other_fields:
                print(f"\n  Other fields:")
                for k, v in other_fields.items():
                    print(f"    {k}: {v}")
    else:
        print("\n--- JSON Entry: NOT FOUND ---")
    
    if result['csv']:
        print("\n--- CSV Entry ---")
        csv_entry = result['csv']
        print(f"  MP ID:           {csv_entry.get('mp_id', 'N/A')}")
        print(f"  Formula:         {csv_entry.get('formula', 'N/A')}")
        print(f"  MP Ordering:     {csv_entry.get('mp_ordering', 'N/A')}")
        print(f"  Transition Type: {csv_entry.get('transition_type', 'N/A')}")
        print(f"  Source:          {csv_entry.get('source', 'N/A')}")
        
        if show_text:
            text = csv_entry.get('text', '')
            print(f"\n  Text Description:")
            print("-" * 50)
            print(text)
            print("-" * 50)
    else:
        print("\n--- CSV Entry: NOT FOUND ---")
    
    print("="*60 + "\n")


def interactive_lookup(json_path: str, csv_path: str):
    """Interactive mode for looking up entries."""
    print("\nLoading data for interactive lookup...")
    json_data = load_merged_dataset(json_path)
    csv_df = load_output_csv(csv_path)
    
    print("\nInteractive Lookup Mode")
    print("Enter mp_id to lookup, or 'quit' to exit")
    print("-" * 40)
    
    while True:
        try:
            mp_id = input("\nEnter mp_id: ").strip()
            
            if mp_id.lower() in ['quit', 'exit', 'q']:
                print("Exiting...")
                break
            
            if not mp_id:
                continue
            
            result = lookup_by_mp_id(mp_id, json_data=json_data, csv_df=csv_df)
            
            if result['json'] is None and result['csv'] is None:
                print(f"No entry found for mp_id: {mp_id}")
                continue
            
            print_entry_from_result(result)
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break


def print_entry_from_result(result: Dict[str, Any], show_text: bool = True):
    """Print entry from lookup result."""
    mp_id = result['mp_id']
    
    print("\n" + "="*60)
    print(f"ENTRY: {mp_id}")
    print("="*60)
    
    if result['json']:
        print("\n--- JSON Entry ---")
        json_entry = result['json']
        print(f"  Formula:         {json_entry.get('formula', json_entry.get('mp_formula', 'N/A'))}")
        print(f"  MP Ordering:     {json_entry.get('mp_ordering', json_entry.get('ordering', 'N/A'))}")
        print(f"  Transition Type: {json_entry.get('transition_type', 'N/A')}")
        print(f"  Source:          {json_entry.get('source', 'N/A')}")
    else:
        print("\n--- JSON Entry: NOT FOUND ---")
    
    if result['csv']:
        print("\n--- CSV Entry ---")
        csv_entry = result['csv']
        print(f"  Formula:         {csv_entry.get('formula', 'N/A')}")
        print(f"  MP Ordering:     {csv_entry.get('mp_ordering', 'N/A')}")
        
        if show_text:
            text = csv_entry.get('text', '')
            print(f"\n  Text Description:")
            print("-" * 50)
            # Truncate long text
            if len(text) > 500:
                print(text[:500] + "...")
            else:
                print(text)
            print("-" * 50)


def main():
    parser = argparse.ArgumentParser(description='Validate output CSV against merged_dataset.json')
    parser.add_argument('--json', default='c:/workspace/alignn_test/merged_dataset.json',
                        help='Path to merged_dataset.json')
    parser.add_argument('--csv', default='output/chemnlp_0_210579_skip_none.csv',
                        help='Path to output CSV')
    parser.add_argument('--validate', action='store_true',
                        help='Run validation check')
    parser.add_argument('--lookup', type=str, default=None,
                        help='Lookup specific mp_id')
    parser.add_argument('--interactive', action='store_true',
                        help='Interactive lookup mode')
    parser.add_argument('--no-text', action='store_true',
                        help='Don\'t show full text description')
    parser.add_argument('--full-json', action='store_true',
                        help='Show full JSON entry with all fields')
    parser.add_argument('--structure', action='store_true',
                        help='Display crystal structure in formatted view')
    
    args = parser.parse_args()
    
    if args.validate:
        results = validate_output(args.json, args.csv)
        print_validation_report(results)
    
    elif args.lookup:
        print_entry(args.lookup, args.json, args.csv, 
                   show_text=not args.no_text, full_json=args.full_json,
                   show_structure=args.structure)
    
    elif args.interactive:
        interactive_lookup(args.json, args.csv)
    
    else:
        # Default: run validation
        results = validate_output(args.json, args.csv)
        print_validation_report(results)


if __name__ == "__main__":
    main()
