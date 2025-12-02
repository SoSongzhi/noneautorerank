#!/usr/bin/env python3
"""
PiPrime + HighNine Reranker
============================

A hybrid peptide identification system combining:
- PiPrime (Spec2Pep) for de novo sequencing with fastest beam search
- HighNine database for spectrum library matching
- Prosit for theoretical spectrum prediction

This implementation uses an intelligent reranking strategy:
1. Generate candidates using fastest beam search with mass filtering
2. Query database for all candidates (O(1) lookup)
3. Use Prosit only for top-5 candidates not found in database
4. Rerank all candidates by spectral similarity

Author: Research Team
Date: 2024-12
License: MIT
"""

import sys
import os
import torch
import torch.nn.functional as F
import numpy as np
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from pyteomics import mgf
from tqdm import tqdm

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from PrimeNovo.denovo.model import Spec2Pep
from piprime_with_mass_check import PiPrimeWithMassCheck
from piprime_reranker import process_peaks, load_piprime_config
from piprime_efficient_reranker import PiPrimeEfficientReranker
from fastest_beam_search import fastest_beam_search
from piprime_mass_calculator import calculate_precursor_mass_from_mz

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def normalize_peptide(peptide: str) -> str:
    """
    Normalize peptide sequence by replacing leucine with isoleucine.
    
    Parameters:
    -----------
    peptide : str
        Input peptide sequence
        
    Returns:
    --------
    str : Normalized peptide sequence (L->I)
    """
    return peptide.replace('L', 'I')


class PiPrimeHighNineReranker:
    """
    PiPrime + HighNine Hybrid Reranker
    
    This class implements a hybrid peptide identification system that combines:
    - De novo sequencing using PiPrime with fastest beam search
    - Database matching using HighNine spectral library
    - Theoretical spectrum prediction using Prosit (only when needed)
    
    The reranking strategy is optimized for speed:
    - All candidates are first checked against the database (fast O(1) lookup)
    - Only top-5 candidates not found in database use Prosit prediction
    - This reduces Prosit calls from hundreds to at most 5 per spectrum
    """
    
    def __init__(
        self, 
        piprime_model_path: str, 
        index_file: str, 
        output_dir: str, 
        device: str = 'cuda'
    ):
        """
        Initialize the reranker.
        
        Parameters:
        -----------
        piprime_model_path : str
            Path to PiPrime model checkpoint
        index_file : str
            Path to precomputed HighNine index file (.pkl)
        output_dir : str
            Directory to save detailed results for each spectrum
        device : str
            Device to use ('cuda' or 'cpu')
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Load PiPrime model
        logger.info(f"Loading PiPrime model: {piprime_model_path}")
        self.piprime_model = Spec2Pep.load_from_checkpoint(
            piprime_model_path, 
            map_location=self.device
        )
        self.piprime_model.eval()
        self.piprime_model.to(self.device)
        
        # Create PiPrime predictor
        self.piprime_predictor = PiPrimeWithMassCheck(
            self.piprime_model,
            precursor_mass_tol=50,
            isotope_error_range=(0, 1),
            beam_width=100
        )
        
        # Load PiPrime configuration
        self.piprime_config = load_piprime_config()
        
        # Load HighNine reranker with precomputed index
        logger.info(f"Loading HighNine index: {index_file}")
        self.reranker = PiPrimeEfficientReranker(
            piprime_model=self.piprime_model,
            piprime_config=self.piprime_config
        )
        self.reranker.load_precomputed_index(index_file)
        
        logger.info("✅ Initialization complete")
    
    def process_single_spectrum(
        self, 
        mz_array: np.ndarray,
        intensity_array: np.ndarray,
        precursor_mz: float,
        precursor_charge: int,
        ground_truth: str = "",
        spectrum_idx: int = 0
    ) -> Dict:
        """
        Process a single spectrum through the complete pipeline.
        
        Pipeline steps:
        1. Preprocess spectrum using PiPrime configuration
        2. Generate spectrum embedding using PiPrime encoder
        3. Generate candidates using fastest beam search with mass filtering
        4. Rerank candidates using intelligent database + Prosit strategy
        5. Save detailed results to file
        
        Parameters:
        -----------
        mz_array : np.ndarray
            m/z values of spectrum peaks
        intensity_array : np.ndarray
            Intensity values of spectrum peaks
        precursor_mz : float
            Precursor m/z value
        precursor_charge : int
            Precursor charge state
        ground_truth : str, optional
            Ground truth peptide sequence (for evaluation)
        spectrum_idx : int, optional
            Spectrum index (for file naming)
            
        Returns:
        --------
        dict : Processing results including:
            - peptide: Top-1 predicted peptide
            - similarity: Spectral similarity score
            - source: Source of match ('Database', 'Prosit', or 'Failed')
            - is_correct: Whether prediction matches ground truth
            - num_candidates: Number of candidates used for reranking
            - timing_stats: Timing breakdown of beam search
        """
        # Step 1: Preprocess spectrum
        peaks = process_peaks(
            mz_array, intensity_array, 
            precursor_mz, precursor_charge, 
            self.piprime_config
        )
        peaks = peaks.to(self.device)
        
        precursor_mass = calculate_precursor_mass_from_mz(precursor_mz, precursor_charge)
        precursors = torch.tensor(
            [[precursor_mass, precursor_charge, precursor_mz]],
            dtype=torch.float32,
            device=self.device
        )
        
        # Step 2: Generate spectrum embedding and log probability matrix
        with torch.no_grad():
            enc_out, enc_mask = self.piprime_model.encoder(peaks.unsqueeze(0))
            spectrum_embedding = enc_out.mean(dim=1).cpu().numpy()[0]
            
            output_logits, _, _ = self.piprime_model.decoder(
                None, precursors, enc_out, enc_mask
            )
            log_prob_matrix = F.log_softmax(output_logits[0], dim=-1)
        
        # Step 3: Fastest beam search with mass filtering
        candidates_raw, timing_stats = fastest_beam_search(
            self.piprime_predictor,
            log_prob_matrix,
            precursor_mz,
            precursor_charge,
            target_count=1000,
            top_n=10
        )
        
        # Convert to reranker format
        candidates_all = []
        candidates_passed = []
        
        for peptide, score, mass, passes in candidates_raw:
            cand_dict = {
                'peptide': peptide,
                'score': score,
                'mass': mass,
                'passes_mass_check': passes
            }
            candidates_all.append(cand_dict)
            
            if passes:
                candidates_passed.append({
                    'peptide': peptide,
                    'score': score
                })
        
        # Step 4: Intelligent reranking strategy
        if not candidates_passed:
            result = {
                'peptide': '',
                'similarity': -1.0,
                'denovo_score': 0.0,
                'source': 'Failed',
                'is_correct': False
            }
            similarity_dict = {}
            source_dict = {}
        else:
            # Substep 4.1: Query database for all candidates (fast O(1) lookup)
            result_db, similarity_dict_db, source_dict_db = self.reranker.rerank_with_external_embedding(
                query_embedding=spectrum_embedding,
                candidates=candidates_passed,
                precursor_mz=precursor_mz,
                precursor_charge=precursor_charge,
                use_prosit=False,  # Database only in first pass
                top_k=3
            )
            
            # Substep 4.2: Identify candidates not found in database
            not_found_candidates = [
                cand for cand in candidates_passed 
                if source_dict_db.get(cand['peptide'], 'NotFound') == 'NotFound'
            ]
            
            # Substep 4.3: Use Prosit only for top-5 not-found candidates
            if not_found_candidates:
                top5_not_found = sorted(not_found_candidates, key=lambda x: x['score'], reverse=True)[:5]
                
                result_prosit, similarity_dict_prosit, source_dict_prosit = self.reranker.rerank_with_external_embedding(
                    query_embedding=spectrum_embedding,
                    candidates=top5_not_found,
                    precursor_mz=precursor_mz,
                    precursor_charge=precursor_charge,
                    use_prosit=True,  # Use Prosit for these candidates
                    top_k=3
                )
                
                # Merge results
                similarity_dict = {**similarity_dict_db, **similarity_dict_prosit}
                source_dict = {**source_dict_db, **source_dict_prosit}
            else:
                # All candidates found in database
                similarity_dict = similarity_dict_db
                source_dict = source_dict_db
            
            # Substep 4.4: Re-rank all candidates by similarity
            all_results = []
            for cand in candidates_passed:
                peptide = cand['peptide']
                all_results.append({
                    'peptide': peptide,
                    'similarity': similarity_dict.get(peptide, -1.0),
                    'denovo_score': cand['score'],
                    'source': source_dict.get(peptide, 'Unknown')
                })
            
            # Sort by similarity and select top-1
            all_results.sort(key=lambda x: x['similarity'], reverse=True)
            result = all_results[0] if all_results else result_db
        
        # Add similarity and source to all candidates
        for cand in candidates_all:
            if cand['passes_mass_check'] and cand['peptide'] in similarity_dict:
                cand['similarity'] = similarity_dict[cand['peptide']]
                cand['source'] = source_dict.get(cand['peptide'], 'Unknown')
            else:
                cand['similarity'] = -1.0
                cand['source'] = 'NotChecked'
        
        # Step 5: Evaluate accuracy
        if result and ground_truth:
            pred_seq = normalize_peptide(result['peptide'])
            true_seq = normalize_peptide(ground_truth)
            result['is_correct'] = (pred_seq == true_seq and true_seq != '')
        else:
            result['is_correct'] = False
        
        result['num_candidates'] = len(candidates_passed)
        result['num_candidates_total'] = len(candidates_all)
        result['num_candidates_passed'] = len([c for c in candidates_all if c['passes_mass_check']])
        result['timing_stats'] = timing_stats
        
        # Step 6: Save detailed results
        self._save_spectrum_details(
            spectrum_idx, ground_truth, candidates_all,
            result, precursor_mz, precursor_charge, timing_stats
        )
        
        return result
    
    def _save_spectrum_details(
        self, 
        spectrum_idx: int, 
        ground_truth: str, 
        candidates_all: List[Dict],
        result: Dict, 
        precursor_mz: float, 
        precursor_charge: int, 
        timing_stats: Dict
    ):
        """
        Save detailed results for a single spectrum to file.
        
        Saves top-20 candidates sorted by similarity score.
        """
        output_file = self.output_dir / f"spectrum_{spectrum_idx:04d}.txt"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"Spectrum {spectrum_idx}\n")
            f.write("="*80 + "\n\n")
            
            # Basic information
            f.write("Spectrum Information:\n")
            f.write("-"*80 + "\n")
            f.write(f"Precursor m/z: {precursor_mz:.4f}\n")
            f.write(f"Precursor charge: {precursor_charge}\n")
            f.write(f"Ground truth: {ground_truth}\n")
            f.write(f"Ground truth (normalized): {normalize_peptide(ground_truth)}\n")
            f.write("\n")
            
            # Timing statistics
            f.write("Beam Search Timing:\n")
            f.write("-"*80 + "\n")
            f.write(f"Total time: {timing_stats['total_time']*1000:.2f} ms\n")
            f.write(f"  TopK time: {timing_stats['topk_time']*1000:.2f} ms\n")
            f.write(f"  Beam time: {timing_stats['beam_time']*1000:.2f} ms\n")
            f.write(f"  Collapse time: {timing_stats['collapse_time']*1000:.2f} ms\n")
            f.write(f"  Mass time: {timing_stats['mass_time']*1000:.2f} ms\n")
            f.write(f"  Filter time: {timing_stats['filter_time']*1000:.2f} ms\n")
            f.write("\n")
            
            # Statistics
            total_cands = len(candidates_all)
            passed_cands = sum(1 for c in candidates_all if c['passes_mass_check'])
            db_cands = sum(1 for c in candidates_all if c.get('source') == 'Database')
            prosit_cands = sum(1 for c in candidates_all if c.get('source') == 'Prosit')
            
            f.write("Candidate Statistics:\n")
            f.write("-"*80 + "\n")
            f.write(f"Total candidates: {total_cands}\n")
            f.write(f"Passed mass check: {passed_cands} ({passed_cands/total_cands*100:.1f}%)\n")
            f.write(f"Database matches: {db_cands}\n")
            f.write(f"Prosit predictions: {prosit_cands}\n")
            f.write("\n")
            
            # Top-20 candidates sorted by similarity
            candidates_sorted = sorted(
                candidates_all,
                key=lambda x: x.get('similarity', -1.0),
                reverse=True
            )
            
            f.write("Top 20 Candidates (sorted by similarity):\n")
            f.write("-"*100 + "\n")
            f.write(f"{'Rank':<6}{'Peptide':<25}{'Similarity':<12}{'Source':<12}{'Score':<12}{'Pass':<8}{'Correct'}\n")
            f.write("-"*100 + "\n")
            
            ground_truth_norm = normalize_peptide(ground_truth)
            for i, cand in enumerate(candidates_sorted[:20], 1):
                peptide_norm = normalize_peptide(cand['peptide'])
                is_correct = (peptide_norm == ground_truth_norm and ground_truth_norm != '')
                pass_mark = "✓" if cand['passes_mass_check'] else "✗"
                correct_mark = " ✓" if is_correct else ""
                
                f.write(f"{i:<6}{cand['peptide']:<25}{cand.get('similarity', -1.0):<12.4f}"
                       f"{cand.get('source', 'Unknown'):<12}{cand['score']:<12.4f}{pass_mark:<8}{correct_mark}\n")
            
            f.write("\n")
            
            # Final result
            f.write("Final Result:\n")
            f.write("-"*80 + "\n")
            if result.get('peptide'):
                f.write(f"Peptide: {result['peptide']}\n")
                f.write(f"Similarity: {result.get('similarity', -1.0):.4f}\n")
                f.write(f"Source: {result.get('source', 'Unknown')}\n")
                f.write(f"Correct: {'✓' if result.get('is_correct', False) else '✗'}\n")
            else:
                f.write("No result\n")


def main():
    """
    Main function to run PiPrime + HighNine reranker on a test dataset.
    
    Configuration:
    - MGF file: testdata/high_nine_validation_1000_converted.mgf
    - PiPrime model: model_massive.ckpt
    - HighNine index: D:/reference_dataset/reference_dataset.mgf.efficient_index.pkl
    - Number of spectra: 100 (configurable)
    """
    # Configuration
    mgf_file = "testdata/high_nine_validation_1000_converted.mgf"
    piprime_model = "model_massive.ckpt"
    index_file = r"D:\reference_dataset\reference_dataset.mgf.efficient_index.pkl"
    num_spectra = 100
    
    # Create output directory with timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"piprime_highnine_results_{timestamp}")
    output_dir.mkdir(exist_ok=True)
    
    # Validate input files
    if not os.path.exists(mgf_file):
        logger.error(f"MGF file not found: {mgf_file}")
        return
    
    if not os.path.exists(piprime_model):
        logger.error(f"PiPrime model not found: {piprime_model}")
        return
    
    if not os.path.exists(index_file):
        logger.error(f"Index file not found: {index_file}")
        logger.info("Please run build_efficient_index.py first to create the index")
        return
    
    # Print configuration
    logger.info(f"\n{'='*80}")
    logger.info("PiPrime + HighNine Reranker")
    logger.info(f"{'='*80}")
    logger.info(f"MGF file: {mgf_file}")
    logger.info(f"PiPrime model: {piprime_model}")
    logger.info(f"Index file: {index_file}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Number of spectra: {num_spectra}")
    logger.info(f"{'='*80}\n")
    
    # Initialize reranker
    reranker = PiPrimeHighNineReranker(piprime_model, index_file, output_dir)
    
    # Statistics
    total = 0
    correct = 0
    total_candidates = 0
    total_beam_time = 0.0
    source_stats = {'Database': 0, 'Prosit': 0, 'Failed': 0}
    
    # Process spectra
    logger.info("Processing spectra...\n")
    
    with mgf.MGF(mgf_file) as reader:
        spectra_list = list(reader)[:num_spectra]
    
    for idx, spec in enumerate(tqdm(spectra_list, desc="Processing", unit="spectrum")):
        try:
            # Extract spectrum information
            pepmass = spec['params'].get('pepmass', [0])
            precursor_mz = pepmass[0] if isinstance(pepmass, (list, tuple)) else pepmass
            
            charge = spec['params'].get('charge', [2])
            precursor_charge = charge[0] if isinstance(charge, (list, tuple)) else charge
            if isinstance(precursor_charge, str):
                precursor_charge = int(precursor_charge.replace('+', ''))
            
            ground_truth = spec['params'].get('seq', '')
            if not ground_truth:
                continue
            
            # Process spectrum
            result = reranker.process_single_spectrum(
                spec['m/z array'], spec['intensity array'],
                precursor_mz, precursor_charge, ground_truth, idx
            )
            
            # Update statistics
            total += 1
            if result['is_correct']:
                correct += 1
            
            total_candidates += result.get('num_candidates', 0)
            total_beam_time += result.get('timing_stats', {}).get('total_time', 0)
            
            source = result.get('source', 'Unknown')
            if source in source_stats:
                source_stats[source] += 1
            
        except Exception as e:
            tqdm.write(f"Error processing spectrum {idx}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Print final results
    logger.info(f"\n{'='*80}")
    logger.info("Final Results")
    logger.info(f"{'='*80}")
    logger.info(f"Total spectra: {total}")
    logger.info(f"Correct predictions: {correct}")
    logger.info(f"Accuracy: {correct/total*100:.2f}%")
    logger.info(f"")
    logger.info(f"Average candidates: {total_candidates/total:.1f}")
    logger.info(f"Average beam search time: {total_beam_time/total*1000:.2f} ms")
    logger.info(f"")
    logger.info("Source statistics:")
    for source, count in source_stats.items():
        if count > 0:
            logger.info(f"  {source}: {count} ({count/total*100:.1f}%)")
    logger.info(f"")
    logger.info(f"Detailed results saved in: {output_dir}/")
    logger.info(f"{'='*80}")


if __name__ == "__main__":
    main()