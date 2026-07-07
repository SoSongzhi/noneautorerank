"""
Merge Database and Prosit datasets for transfer learning.

This script combines database spectra (real) with Prosit spectra (synthetic)
and prepares them for mixed training.

Usage:
    python scripts/merge_datasets.py \
        --database data/database_spectra.pkl \
        --prosit data/prosit_spectra.pkl \
        --output data/mixed_spectra.pkl \
        --balance_ratio 0.3
"""

import pickle
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter


def load_dataset(path: str) -> Dict:
    """Load a dataset from pickle file."""
    print(f"Loading dataset from {path}...")
    with open(path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"  Loaded {len(data['spectra'])} spectra")
    print(f"  Unique peptides: {len(set(data['peptides']))}")
    return data


def analyze_dataset(data: Dict, name: str):
    """Analyze and print dataset statistics."""
    print(f"\n{name} Dataset Statistics:")
    print(f"  Total spectra: {len(data['spectra'])}")
    print(f"  Unique peptides: {len(set(data['peptides']))}")
    
    # Peptide frequency
    peptide_counts = Counter(data['peptides'])
    print(f"  Peptides with 1 spectrum: {sum(1 for c in peptide_counts.values() if c == 1)}")
    print(f"  Peptides with 2+ spectra: {sum(1 for c in peptide_counts.values() if c >= 2)}")
    print(f"  Max spectra per peptide: {max(peptide_counts.values())}")
    print(f"  Avg spectra per peptide: {np.mean(list(peptide_counts.values())):.2f}")
    
    # Spectrum size
    spectrum_sizes = [len(s) for s in data['spectra']]
    print(f"  Avg peaks per spectrum: {np.mean(spectrum_sizes):.1f}")
    print(f"  Min/Max peaks: {min(spectrum_sizes)}/{max(spectrum_sizes)}")


def balance_datasets(
    db_data: Dict,
    prosit_data: Dict,
    balance_ratio: float = 0.3
) -> Tuple[Dict, Dict]:
    """
    Balance datasets by downsampling the larger one.
    
    Args:
        db_data: Database dataset
        prosit_data: Prosit dataset
        balance_ratio: Target ratio of database/(database+prosit)
                      e.g., 0.3 means 30% database, 70% prosit
    
    Returns:
        (balanced_db_data, balanced_prosit_data)
    """
    db_size = len(db_data['spectra'])
    prosit_size = len(prosit_data['spectra'])
    
    print(f"\nBalancing datasets (target ratio: {balance_ratio:.1%} database)...")
    print(f"  Original - Database: {db_size}, Prosit: {prosit_size}")
    
    # Calculate target sizes
    if balance_ratio >= 1.0:
        # Keep all database, sample prosit
        target_db_size = db_size
        target_prosit_size = int(db_size * (1 - balance_ratio) / balance_ratio)
    else:
        # Calculate based on ratio
        total_target = max(db_size, prosit_size)
        target_db_size = int(total_target * balance_ratio)
        target_prosit_size = int(total_target * (1 - balance_ratio))
    
    # Sample database if needed
    if target_db_size < db_size:
        indices = np.random.choice(db_size, target_db_size, replace=False)
        balanced_db = {
            'spectra': [db_data['spectra'][i] for i in indices],
            'peptides': [db_data['peptides'][i] for i in indices]
        }
    else:
        balanced_db = db_data
    
    # Sample prosit if needed
    if target_prosit_size < prosit_size:
        indices = np.random.choice(prosit_size, target_prosit_size, replace=False)
        balanced_prosit = {
            'spectra': [prosit_data['spectra'][i] for i in indices],
            'peptides': [prosit_data['peptides'][i] for i in indices]
        }
    else:
        balanced_prosit = prosit_data
    
    print(f"  Balanced - Database: {len(balanced_db['spectra'])}, "
          f"Prosit: {len(balanced_prosit['spectra'])}")
    
    actual_ratio = len(balanced_db['spectra']) / (
        len(balanced_db['spectra']) + len(balanced_prosit['spectra'])
    )
    print(f"  Actual ratio: {actual_ratio:.1%} database")
    
    return balanced_db, balanced_prosit


def merge_datasets(
    db_data: Dict,
    prosit_data: Dict,
    shuffle: bool = True
) -> Dict:
    """
    Merge database and Prosit datasets.
    
    Args:
        db_data: Database dataset
        prosit_data: Prosit dataset
        shuffle: Whether to shuffle the merged data
    
    Returns:
        Merged dataset with 'source' labels
    """
    print("\nMerging datasets...")
    
    # Combine data
    merged_data = {
        'spectra': db_data['spectra'] + prosit_data['spectra'],
        'peptides': db_data['peptides'] + prosit_data['peptides'],
        'source': ['database'] * len(db_data['spectra']) + 
                  ['prosit'] * len(prosit_data['spectra'])
    }
    
    # Shuffle if requested
    if shuffle:
        print("  Shuffling merged data...")
        indices = np.random.permutation(len(merged_data['spectra']))
        merged_data['spectra'] = [merged_data['spectra'][i] for i in indices]
        merged_data['peptides'] = [merged_data['peptides'][i] for i in indices]
        merged_data['source'] = [merged_data['source'][i] for i in indices]
    
    print(f"  Total spectra: {len(merged_data['spectra'])}")
    print(f"  Database: {merged_data['source'].count('database')}")
    print(f"  Prosit: {merged_data['source'].count('prosit')}")
    
    return merged_data


def create_train_val_split(
    data: Dict,
    val_ratio: float = 0.1,
    stratify_by_source: bool = True
) -> Tuple[Dict, Dict]:
    """
    Split merged data into train and validation sets.
    
    Args:
        data: Merged dataset
        val_ratio: Ratio of validation data
        stratify_by_source: Whether to stratify by source
    
    Returns:
        (train_data, val_data)
    """
    print(f"\nCreating train/val split (val_ratio={val_ratio})...")
    
    n_total = len(data['spectra'])
    
    if stratify_by_source:
        # Stratified split by source
        db_indices = [i for i, s in enumerate(data['source']) if s == 'database']
        prosit_indices = [i for i, s in enumerate(data['source']) if s == 'prosit']
        
        n_db_val = int(len(db_indices) * val_ratio)
        n_prosit_val = int(len(prosit_indices) * val_ratio)
        
        np.random.shuffle(db_indices)
        np.random.shuffle(prosit_indices)
        
        val_indices = db_indices[:n_db_val] + prosit_indices[:n_prosit_val]
        train_indices = db_indices[n_db_val:] + prosit_indices[n_prosit_val:]
    else:
        # Random split
        indices = np.random.permutation(n_total)
        n_val = int(n_total * val_ratio)
        val_indices = indices[:n_val]
        train_indices = indices[n_val:]
    
    # Create train and val datasets
    train_data = {
        'spectra': [data['spectra'][i] for i in train_indices],
        'peptides': [data['peptides'][i] for i in train_indices],
        'source': [data['source'][i] for i in train_indices]
    }
    
    val_data = {
        'spectra': [data['spectra'][i] for i in val_indices],
        'peptides': [data['peptides'][i] for i in val_indices],
        'source': [data['source'][i] for i in val_indices]
    }
    
    print(f"  Train: {len(train_data['spectra'])} spectra")
    print(f"    Database: {train_data['source'].count('database')}")
    print(f"    Prosit: {train_data['source'].count('prosit')}")
    print(f"  Val: {len(val_data['spectra'])} spectra")
    print(f"    Database: {val_data['source'].count('database')}")
    print(f"    Prosit: {val_data['source'].count('prosit')}")
    
    return train_data, val_data


def main():
    parser = argparse.ArgumentParser(
        description='Merge database and Prosit datasets for transfer learning'
    )
    parser.add_argument(
        '--database',
        type=str,
        required=True,
        help='Path to database spectra pickle file'
    )
    parser.add_argument(
        '--prosit',
        type=str,
        required=True,
        help='Path to Prosit spectra pickle file'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output path for merged dataset'
    )
    parser.add_argument(
        '--balance_ratio',
        type=float,
        default=0.3,
        help='Target ratio of database/(database+prosit) (default: 0.3)'
    )
    parser.add_argument(
        '--val_ratio',
        type=float,
        default=0.1,
        help='Validation set ratio (default: 0.1)'
    )
    parser.add_argument(
        '--no_balance',
        action='store_true',
        help='Do not balance datasets'
    )
    parser.add_argument(
        '--no_shuffle',
        action='store_true',
        help='Do not shuffle merged data'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    print(f"Random seed: {args.seed}")
    
    # Load datasets
    db_data = load_dataset(args.database)
    prosit_data = load_dataset(args.prosit)
    
    # Analyze original datasets
    analyze_dataset(db_data, "Database")
    analyze_dataset(prosit_data, "Prosit")
    
    # Balance if requested
    if not args.no_balance:
        db_data, prosit_data = balance_datasets(
            db_data, prosit_data, args.balance_ratio
        )
    
    # Merge datasets
    merged_data = merge_datasets(
        db_data, prosit_data, shuffle=not args.no_shuffle
    )
    
    # Create train/val split
    train_data, val_data = create_train_val_split(
        merged_data, args.val_ratio, stratify_by_source=True
    )
    
    # Save merged dataset
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving merged dataset to {output_path}...")
    with open(output_path, 'wb') as f:
        pickle.dump({
            'train': train_data,
            'val': val_data,
            'metadata': {
                'database_count': train_data['source'].count('database') + 
                                 val_data['source'].count('database'),
                'prosit_count': train_data['source'].count('prosit') + 
                               val_data['source'].count('prosit'),
                'balance_ratio': args.balance_ratio,
                'val_ratio': args.val_ratio,
                'seed': args.seed
            }
        }, f)
    
    print("✓ Done!")
    print(f"\nTo use this dataset for training:")
    print(f"  python scripts/train_mixed.py \\")
    print(f"    --config config_stage2_mixed.yaml \\")
    print(f"    --data_path {output_path} \\")
    print(f"    --output_dir checkpoints/stage2_mixed")


if __name__ == '__main__':
    main()