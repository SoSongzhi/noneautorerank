"""
Mixed Dataset for Database + Prosit Transfer Learning

This module implements a dataset that combines real database spectra
with synthetic Prosit spectra for transfer learning.

Strategy:
1. Anchor can come from any source
2. Positive prefers same source, then same peptide
3. Negative is randomly selected from different peptides
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from .dataset import SpectrumContrastiveDataset


class MixedSpectrumDataset(SpectrumContrastiveDataset):
    """
    Mixed dataset combining Database and Prosit spectra.
    
    This dataset implements a smart sampling strategy that:
    - Prioritizes same-source positive pairs (70% probability)
    - Falls back to cross-source pairs for the same peptide
    - Uses augmentation when no positive pairs are available
    
    Args:
        spectra_data: List of spectrum arrays [(mz, intensity), ...]
        peptide_labels: List of peptide sequences
        source_labels: List of source labels ('database' or 'prosit')
        same_source_prob: Probability of selecting same-source positive (default: 0.7)
        **kwargs: Additional arguments passed to parent class
    """
    
    def __init__(
        self,
        spectra_data: List[np.ndarray],
        peptide_labels: List[str],
        source_labels: List[str],
        same_source_prob: float = 0.7,
        **kwargs
    ):
        super().__init__(spectra_data, peptide_labels, **kwargs)
        
        self.sources = source_labels
        self.same_source_prob = same_source_prob
        
        # Build peptide-source index: (peptide, source) -> [indices]
        self.peptide_source_indices = {}
        for idx, (peptide, source) in enumerate(zip(peptide_labels, source_labels)):
            key = (peptide, source)
            if key not in self.peptide_source_indices:
                self.peptide_source_indices[key] = []
            self.peptide_source_indices[key].append(idx)
        
        # Count statistics
        self.db_count = sum(1 for s in source_labels if s == 'database')
        self.prosit_count = sum(1 for s in source_labels if s == 'prosit')
        
        print(f"Mixed Dataset Statistics:")
        print(f"  Database spectra: {self.db_count}")
        print(f"  Prosit spectra: {self.prosit_count}")
        print(f"  Total spectra: {len(self.spectra)}")
        print(f"  Unique peptides: {len(self.peptide_to_indices)}")
        print(f"  Same-source probability: {self.same_source_prob}")
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a triplet sample with source-aware positive selection.
        
        Returns:
            dict with keys:
                - anchor: Anchor spectrum tensor
                - positive: Positive spectrum tensor
                - negative: Negative spectrum tensor
                - label: Peptide label (integer)
                - anchor_source: Source of anchor ('database' or 'prosit')
                - positive_source: Source of positive
                - negative_source: Source of negative
        """
        # Get anchor
        anchor_spectrum = self.spectra[idx].copy()
        anchor_peptide = self.labels[idx]
        anchor_source = self.sources[idx]
        anchor_label = self.peptide_to_int[anchor_peptide]
        
        # Select positive with source-aware strategy
        positive_spectrum, pos_source = self._select_positive(
            idx, anchor_peptide, anchor_source, anchor_spectrum
        )
        
        # Select negative (random from different peptide)
        negative_spectrum, neg_source = self._select_negative(
            anchor_peptide, anchor_spectrum
        )
        
        # Apply augmentation
        if self.augment:
            anchor_spectrum = self.augmentation(anchor_spectrum)
            positive_spectrum = self.augmentation(positive_spectrum)
            # Note: negative is not augmented to maintain diversity
        
        return {
            'anchor': torch.FloatTensor(anchor_spectrum),
            'positive': torch.FloatTensor(positive_spectrum),
            'negative': torch.FloatTensor(negative_spectrum),
            'label': anchor_label,
            'anchor_source': anchor_source,
            'positive_source': pos_source,
            'negative_source': neg_source
        }
    
    def _select_positive(
        self,
        anchor_idx: int,
        anchor_peptide: str,
        anchor_source: str,
        anchor_spectrum: np.ndarray
    ) -> Tuple[np.ndarray, str]:
        """
        Select positive sample with source-aware strategy.
        
        Strategy:
        1. With probability same_source_prob, try to select from same source
        2. Otherwise, select from any source with same peptide
        3. If no positive available, use augmentation
        
        Returns:
            (positive_spectrum, positive_source)
        """
        # Get same-source candidates
        same_source_key = (anchor_peptide, anchor_source)
        same_source_indices = []
        if same_source_key in self.peptide_source_indices:
            same_source_indices = [
                i for i in self.peptide_source_indices[same_source_key]
                if i != anchor_idx
            ]
        
        # Get all same-peptide candidates
        all_peptide_indices = [
            i for i in self.peptide_to_indices[anchor_peptide]
            if i != anchor_idx
        ]
        
        # Decide which pool to sample from
        use_same_source = (
            same_source_indices and 
            np.random.rand() < self.same_source_prob
        )
        
        if use_same_source:
            # Sample from same source
            pos_idx = np.random.choice(same_source_indices)
            positive_spectrum = self.spectra[pos_idx].copy()
            positive_source = self.sources[pos_idx]
        elif all_peptide_indices:
            # Sample from any source (cross-source)
            pos_idx = np.random.choice(all_peptide_indices)
            positive_spectrum = self.spectra[pos_idx].copy()
            positive_source = self.sources[pos_idx]
        else:
            # No positive available, use augmentation
            positive_spectrum = anchor_spectrum.copy()
            if self.augment:
                positive_spectrum = self.augmentation(positive_spectrum)
            positive_source = anchor_source
        
        return positive_spectrum, positive_source
    
    def _select_negative(
        self,
        anchor_peptide: str,
        anchor_spectrum: np.ndarray
    ) -> Tuple[np.ndarray, str]:
        """
        Select negative sample (random from different peptide).
        
        Returns:
            (negative_spectrum, negative_source)
        """
        # Get all different peptides
        negative_peptides = [
            p for p in self.peptide_to_indices.keys()
            if p != anchor_peptide
        ]
        
        if negative_peptides:
            # Randomly select a different peptide
            neg_peptide = np.random.choice(negative_peptides)
            neg_idx = np.random.choice(self.peptide_to_indices[neg_peptide])
            negative_spectrum = self.spectra[neg_idx].copy()
            negative_source = self.sources[neg_idx]
        else:
            # Fallback: random noise
            negative_spectrum = np.random.randn(*anchor_spectrum.shape).astype(np.float32)
            negative_source = 'random'
        
        return negative_spectrum, negative_source
    
    def get_source_statistics(self) -> Dict[str, int]:
        """Get statistics about data sources."""
        return {
            'database': self.db_count,
            'prosit': self.prosit_count,
            'total': len(self.spectra),
            'unique_peptides': len(self.peptide_to_indices)
        }
    
    def get_peptide_source_distribution(self, peptide: str) -> Dict[str, int]:
        """
        Get source distribution for a specific peptide.
        
        Args:
            peptide: Peptide sequence
            
        Returns:
            dict with counts: {'database': n, 'prosit': m}
        """
        if peptide not in self.peptide_to_indices:
            return {'database': 0, 'prosit': 0}
        
        indices = self.peptide_to_indices[peptide]
        sources = [self.sources[i] for i in indices]
        
        return {
            'database': sources.count('database'),
            'prosit': sources.count('prosit')
        }


def create_weighted_sampler(
    dataset: MixedSpectrumDataset,
    database_weight: float = 2.0,
    prosit_weight: float = 1.0
) -> torch.utils.data.WeightedRandomSampler:
    """
    Create a weighted sampler to balance database and Prosit data.
    
    This is useful when you have much more Prosit data than database data,
    and want to ensure database data is sampled more frequently.
    
    Args:
        dataset: MixedSpectrumDataset instance
        database_weight: Weight for database samples (default: 2.0)
        prosit_weight: Weight for Prosit samples (default: 1.0)
        
    Returns:
        WeightedRandomSampler instance
        
    Example:
        >>> sampler = create_weighted_sampler(dataset, database_weight=2.0)
        >>> loader = DataLoader(dataset, batch_size=32, sampler=sampler)
    """
    weights = []
    for source in dataset.sources:
        if source == 'database':
            weights.append(database_weight)
        else:
            weights.append(prosit_weight)
    
    weights = torch.DoubleTensor(weights)
    sampler = torch.utils.data.WeightedRandomSampler(
        weights,
        num_samples=len(weights),
        replacement=True
    )
    
    print(f"Created weighted sampler:")
    print(f"  Database weight: {database_weight}")
    print(f"  Prosit weight: {prosit_weight}")
    print(f"  Effective database ratio: {database_weight / (database_weight + prosit_weight):.2%}")
    
    return sampler


if __name__ == '__main__':
    # Test the mixed dataset
    print("Testing MixedSpectrumDataset...")
    
    # Create dummy data
    spectra = [
        np.random.randn(50, 2).astype(np.float32) for _ in range(100)
    ]
    peptides = ['PEPTIDE' + str(i % 10) for i in range(100)]
    sources = ['database'] * 30 + ['prosit'] * 70
    
    # Create dataset
    dataset = MixedSpectrumDataset(
        spectra,
        peptides,
        sources,
        same_source_prob=0.7,
        augment=True
    )
    
    # Test sampling
    print("\nTesting sampling...")
    sample = dataset[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"Anchor source: {sample['anchor_source']}")
    print(f"Positive source: {sample['positive_source']}")
    print(f"Negative source: {sample['negative_source']}")
    
    # Test statistics
    print("\nDataset statistics:")
    stats = dataset.get_source_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Test weighted sampler
    print("\nTesting weighted sampler...")
    sampler = create_weighted_sampler(dataset, database_weight=2.0)
    
    print("\n✓ All tests passed!")