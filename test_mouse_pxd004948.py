#!/usr/bin/env python3
"""
测试 high.mouse.PXD004948_mass_filtered.mgf
使用 PiPrime + HighNine Reranker

特点:
1. 测试所有谱图
2. 每500个谱图输出一次准确率
3. 排除数据库中与query相同的谱图（避免自匹配）
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

# 添加路径
sys.path.insert(0, os.path.dirname(__file__))

from PrimeNovo.denovo.model import Spec2Pep
from piprime_with_mass_check import PiPrimeWithMassCheck
from piprime_reranker import process_peaks, load_piprime_config
from piprime_efficient_reranker import PiPrimeEfficientReranker
from fastest_beam_search import fastest_beam_search
from piprime_mass_calculator import calculate_precursor_mass_from_mz
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def normalize_peptide(peptide: str) -> str:
    """标准化peptide序列（L->I）"""
    return peptide.replace('L', 'I')


class PiPrimeHighNineRerankerWithExclusion:
    """
    PiPrime + HighNine Reranker with Self-Exclusion
    
    在数据库查找时排除与query谱图相同的谱图
    """
    
    def __init__(
        self, 
        piprime_model_path: str, 
        index_file: str, 
        output_dir: str,
        mgf_file: str,  # 添加MGF文件路径用于排除
        device: str = 'cuda'
    ):
        """初始化reranker"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # 加载PiPrime模型
        logger.info(f"Loading PiPrime model: {piprime_model_path}")
        self.piprime_model = Spec2Pep.load_from_checkpoint(
            piprime_model_path, 
            map_location=self.device
        )
        self.piprime_model.eval()
        self.piprime_model.to(self.device)
        
        # 创建PiPrime predictor
        self.piprime_predictor = PiPrimeWithMassCheck(
            self.piprime_model,
            precursor_mass_tol=50,
            isotope_error_range=(0, 1),
            beam_width=100
        )
        
        # 加载PiPrime配置
        self.piprime_config = load_piprime_config()
        
        # 加载HighNine reranker
        logger.info(f"Loading HighNine index: {index_file}")
        self.reranker = PiPrimeEfficientReranker(
            piprime_model=self.piprime_model,
            piprime_config=self.piprime_config
        )
        self.reranker.load_precomputed_index(index_file)
        
        # 加载MGF文件，建立spectrum_idx到谱图信息的映射
        logger.info(f"Loading MGF file for exclusion: {mgf_file}")
        self.spectrum_map = {}
        with mgf.MGF(mgf_file) as reader:
            for idx, spec in enumerate(reader):
                pepmass = spec['params'].get('pepmass', [0])
                precursor_mz = pepmass[0] if isinstance(pepmass, (list, tuple)) else pepmass
                
                charge = spec['params'].get('charge', [2])
                precursor_charge = charge[0] if isinstance(charge, (list, tuple)) else charge
                if isinstance(precursor_charge, str):
                    precursor_charge = int(precursor_charge.replace('+', ''))
                
                self.spectrum_map[idx] = {
                    'precursor_mz': precursor_mz,
                    'precursor_charge': precursor_charge,
                    'mz_array': spec['m/z array'],
                    'intensity_array': spec['intensity array']
                }
        
        logger.info(f"Loaded {len(self.spectrum_map)} spectra for exclusion")
        logger.info("✅ Initialization complete")
    
    def _should_exclude_spectrum(self, query_idx: int, ref_spec: Dict) -> bool:
        """
        判断是否应该排除某个参考谱图
        
        排除条件：
        1. 参考谱图的index与query的index相同
        2. 或者precursor_mz和charge都相同（可能是同一个谱图）
        """
        if query_idx not in self.spectrum_map:
            return False
        
        query_info = self.spectrum_map[query_idx]
        
        # 检查index
        if ref_spec.get('index') == query_idx:
            return True
        
        # 检查precursor_mz和charge是否相同（容差0.01 Da）
        if abs(ref_spec.get('precursor_mz', 0) - query_info['precursor_mz']) < 0.01:
            if ref_spec.get('charge', 0) == query_info['precursor_charge']:
                return True
        
        return False
    
    def rerank_with_exclusion(
        self,
        query_embedding,
        query_idx: int,
        candidates,
        precursor_mz,
        precursor_charge,
        use_prosit=True,
        top_k=3
    ):
        """
        使用外部提供的spectrum embedding进行重排序（带排除功能）
        
        与原始rerank_with_external_embedding的区别：
        - 在数据库查找时排除与query相同的谱图
        """
        if self.reranker.precomputed_index is None:
            raise ValueError("Precomputed index not loaded")
        
        results = []
        similarity_cache = {}
        
        for candidate in candidates:
            peptide = candidate['peptide']
            denovo_score = candidate['score']
            
            # 检查缓存
            if peptide in similarity_cache:
                cached_result = similarity_cache[peptide]
                results.append({
                    'peptide': peptide,
                    'denovo_score': denovo_score,
                    'similarity': cached_result['similarity'],
                    'matched_count': cached_result.get('matched_count', 0),
                    'source': cached_result['source']
                })
                continue
            
            # 转换格式
            unimod_peptide = self.reranker.convert_piprime_to_unimod(peptide)
            clean_peptide = self.reranker.normalize_peptide(peptide)
            
            # O(1) 查找数据库
            ref_spectra = None
            if unimod_peptide in self.reranker.precomputed_index:
                ref_spectra = self.reranker.precomputed_index[unimod_peptide]["spectra"]
            elif peptide in self.reranker.precomputed_index:
                ref_spectra = self.reranker.precomputed_index[peptide]["spectra"]
            elif clean_peptide in self.reranker.precomputed_index:
                ref_spectra = self.reranker.precomputed_index[clean_peptide]["spectra"]
            
            if ref_spectra is not None:
                # **关键修改：排除与query相同的谱图**
                filtered_spectra = [
                    spec for spec in ref_spectra 
                    if not self._should_exclude_spectrum(query_idx, spec)
                ]
                
                # 如果排除后没有谱图了，标记为NotFound
                if len(filtered_spectra) == 0:
                    logger.debug(f"All {len(ref_spectra)} reference spectra excluded for {peptide}")
                    ref_spectra = None
                else:
                    ref_spectra = filtered_spectra[:10]  # 限制到前10个
                    logger.debug(f"Excluded {len(ref_spectra) - len(filtered_spectra)} self-matching spectra for {peptide}")
            
            if ref_spectra is not None:
                # 计算与每个参考谱图的相似度
                similarities = []
                for ref_spec in ref_spectra:
                    cache_key = f"ref_{ref_spec['index']}"
                    if cache_key in self.reranker.encoding_cache:
                        ref_embedding = self.reranker.encoding_cache[cache_key]
                    else:
                        ref_embedding = self.reranker.encode_spectrum_from_arrays(
                            ref_spec['mz'], ref_spec['intensity'],
                            ref_spec['precursor_mz'], ref_spec['charge']
                        )
                        self.reranker.encoding_cache[cache_key] = ref_embedding
                    
                    sim = cosine_similarity(
                        query_embedding.reshape(1, -1),
                        ref_embedding.reshape(1, -1)
                    )[0][0]
                    similarities.append(sim)
                
                # 取Top-K相似度的平均值
                top_k_similarities = sorted(similarities, reverse=True)[:top_k]
                final_similarity = np.mean(top_k_similarities)
                
                result_dict = {
                    'peptide': peptide,
                    'denovo_score': denovo_score,
                    'similarity': final_similarity,
                    'matched_count': len(ref_spectra),
                    'source': 'Database'
                }
                results.append(result_dict)
                
                similarity_cache[peptide] = {
                    'similarity': final_similarity,
                    'matched_count': len(ref_spectra),
                    'source': 'Database'
                }
            
            elif use_prosit:
                # 使用Prosit预测
                prosit_peptide = clean_peptide
                prosit_spec = self.reranker.generate_prosit_spectrum(
                    prosit_peptide,
                    precursor_charge,
                    fragmentation_type='HCD'
                )
                
                if prosit_spec:
                    prosit_embedding = self.reranker.encode_spectrum_from_arrays(
                        prosit_spec['mz'], prosit_spec['intensity'],
                        prosit_spec['precursor_mz'], prosit_spec['charge']
                    )
                    
                    similarity = cosine_similarity(
                        query_embedding.reshape(1, -1),
                        prosit_embedding.reshape(1, -1)
                    )[0][0]
                    
                    result_dict = {
                        'peptide': peptide,
                        'denovo_score': denovo_score,
                        'similarity': similarity,
                        'matched_count': 0,
                        'source': 'Prosit'
                    }
                    results.append(result_dict)
                    
                    similarity_cache[peptide] = {
                        'similarity': similarity,
                        'matched_count': 0,
                        'source': 'Prosit'
                    }
                else:
                    result_dict = {
                        'peptide': peptide,
                        'denovo_score': denovo_score,
                        'similarity': -1.0,
                        'matched_count': 0,
                        'source': 'Failed'
                    }
                    results.append(result_dict)
                    
                    similarity_cache[peptide] = {
                        'similarity': -1.0,
                        'matched_count': 0,
                        'source': 'Failed'
                    }
            else:
                result_dict = {
                    'peptide': peptide,
                    'denovo_score': denovo_score,
                    'similarity': -1.0,
                    'matched_count': 0,
                    'source': 'NotFound'
                }
                results.append(result_dict)
                
                similarity_cache[peptide] = {
                    'similarity': -1.0,
                    'matched_count': 0,
                    'source': 'NotFound'
                }
        
        # 按相似度排序
        if not results:
            return {
                'peptide': '',
                'similarity': -1.0,
                'denovo_score': 0.0,
                'source': 'NoResults'
            }, {}, {}
        
        results.sort(key=lambda x: x['similarity'], reverse=True)
        
        # 创建映射
        similarity_dict = {r['peptide']: r['similarity'] for r in results}
        source_dict = {r['peptide']: r['source'] for r in results}
        
        return results[0], similarity_dict, source_dict
    
    def process_single_spectrum(
        self, 
        mz_array: np.ndarray,
        intensity_array: np.ndarray,
        precursor_mz: float,
        precursor_charge: int,
        ground_truth: str = "",
        spectrum_idx: int = 0
    ) -> Dict:
        """处理单个谱图"""
        # 1. 预处理谱图
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
        
        # 2. 生成spectrum embedding和log probability matrix
        with torch.no_grad():
            enc_out, enc_mask = self.piprime_model.encoder(peaks.unsqueeze(0))
            spectrum_embedding = enc_out.mean(dim=1).cpu().numpy()[0]
            
            output_logits, _, _ = self.piprime_model.decoder(
                None, precursors, enc_out, enc_mask
            )
            log_prob_matrix = F.log_softmax(output_logits[0], dim=-1)
        
        # 3. Fastest beam search
        candidates_raw, timing_stats = fastest_beam_search(
            self.piprime_predictor,
            log_prob_matrix,
            precursor_mz,
            precursor_charge,
            target_count=1000,
            top_n=10
        )
        
        # 转换格式
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
        
        # 4. Intelligent Reranking with exclusion (Database first, then Prosit for top-5)
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
            # Step 4.1: Query database for all candidates (no Prosit)
            result_db, similarity_dict_db, source_dict_db = self.rerank_with_exclusion(
                query_embedding=spectrum_embedding,
                query_idx=spectrum_idx,
                candidates=candidates_passed,
                precursor_mz=precursor_mz,
                precursor_charge=precursor_charge,
                use_prosit=False,  # Database only in first pass
                top_k=3
            )
            
            # Step 4.2: Find candidates not in database
            not_found_candidates = [
                cand for cand in candidates_passed
                if source_dict_db.get(cand['peptide'], 'NotFound') == 'NotFound'
            ]
            
            # Step 4.3: Use Prosit only for top-5 not-found candidates
            if not_found_candidates:
                top5_not_found = sorted(not_found_candidates, key=lambda x: x['score'], reverse=True)[:5]
                
                result_prosit, similarity_dict_prosit, source_dict_prosit = self.rerank_with_exclusion(
                    query_embedding=spectrum_embedding,
                    query_idx=spectrum_idx,
                    candidates=top5_not_found,
                    precursor_mz=precursor_mz,
                    precursor_charge=precursor_charge,
                    use_prosit=True,  # Prosit for these only
                    top_k=3
                )
                
                # Merge results
                similarity_dict = {**similarity_dict_db, **similarity_dict_prosit}
                source_dict = {**source_dict_db, **source_dict_prosit}
            else:
                # All found in database
                similarity_dict = similarity_dict_db
                source_dict = source_dict_db
            
            # Step 4.4: Re-rank only candidates with valid similarity scores
            all_results = []
            for cand in candidates_passed:
                peptide = cand['peptide']
                # Only include candidates that were actually checked (similarity != -1.0)
                if peptide in similarity_dict and similarity_dict[peptide] > -1.0:
                    all_results.append({
                        'peptide': peptide,
                        'similarity': similarity_dict[peptide],
                        'denovo_score': cand['score'],
                        'source': source_dict.get(peptide, 'Unknown')
                    })
            
            # Sort by similarity and select top-1
            # If no valid results, fall back to database result
            if all_results:
                all_results.sort(key=lambda x: x['similarity'], reverse=True)
                result = all_results[0]
            else:
                result = result_db
        
        # 添加similarity和source到所有候选
        for cand in candidates_all:
            if cand['passes_mass_check'] and cand['peptide'] in similarity_dict:
                cand['similarity'] = similarity_dict[cand['peptide']]
                cand['source'] = source_dict.get(cand['peptide'], 'Unknown')
            else:
                cand['similarity'] = -1.0
                cand['source'] = 'NotChecked'
        
        # 5. 评估准确率
        if result and ground_truth:
            pred_seq = normalize_peptide(result['peptide'])
            true_seq = normalize_peptide(ground_truth)
            result['is_correct'] = (pred_seq == true_seq and true_seq != '')
        else:
            result['is_correct'] = False
        
        result['num_candidates'] = len(candidates_passed)
        result['num_candidates_total'] = len(candidates_all)
        result['timing_stats'] = timing_stats
        
        return result


def main():
    """主函数"""
    # 配置
    mgf_file = r"C:\Users\research\Desktop\Alldata\high9-massfilter\high.mouse.PXD004948_mass_filtered.mgf"
    piprime_model = "model_massive.ckpt"
    index_file = r"D:\reference_dataset\reference_dataset.mgf.efficient_index.pkl"
    
    # 创建输出目录
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"mouse_pxd004948_results_{timestamp}")
    output_dir.mkdir(exist_ok=True)
    
    # 验证文件
    if not os.path.exists(mgf_file):
        logger.error(f"MGF file not found: {mgf_file}")
        return
    
    if not os.path.exists(piprime_model):
        logger.error(f"PiPrime model not found: {piprime_model}")
        return
    
    if not os.path.exists(index_file):
        logger.error(f"Index file not found: {index_file}")
        return
    
    # 打印配置
    logger.info(f"\n{'='*80}")
    logger.info("PiPrime + HighNine Reranker (With Self-Exclusion)")
    logger.info(f"{'='*80}")
    logger.info(f"MGF file: {mgf_file}")
    logger.info(f"PiPrime model: {piprime_model}")
    logger.info(f"Index file: {index_file}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"{'='*80}\n")
    
    # 初始化reranker
    reranker = PiPrimeHighNineRerankerWithExclusion(
        piprime_model, index_file, output_dir, mgf_file
    )
    
    # 统计变量
    total = 0
    correct = 0
    source_stats = {'Database': 0, 'Prosit': 0, 'Failed': 0}
    predicted_peptides = []  # Batch writing buffer
    
    # 创建结果文件
    results_file = output_dir / "predicted_peptides.txt"
    results_handle = open(results_file, 'w', encoding='utf-8')
    logger.info(f"Results will be saved to: {results_file}\n")
    
    # 处理谱图
    logger.info("Processing spectra...\n")
    
    with mgf.MGF(mgf_file) as reader:
        spectra_list = list(reader)
    
    total_spectra = len(spectra_list)
    logger.info(f"Total spectra to process: {total_spectra}\n")
    
    for idx, spec in enumerate(tqdm(spectra_list, desc="Processing", unit="spectrum")):
        try:
            # 提取信息
            pepmass = spec['params'].get('pepmass', [0])
            precursor_mz = pepmass[0] if isinstance(pepmass, (list, tuple)) else pepmass
            
            charge = spec['params'].get('charge', [2])
            precursor_charge = charge[0] if isinstance(charge, (list, tuple)) else charge
            if isinstance(precursor_charge, str):
                precursor_charge = int(precursor_charge.replace('+', ''))
            
            ground_truth = spec['params'].get('seq', '')
            if not ground_truth:
                continue
            
            # 处理谱图
            result = reranker.process_single_spectrum(
                spec['m/z array'], spec['intensity array'],
                precursor_mz, precursor_charge, ground_truth, idx
            )
            
            # 更新统计
            total += 1
            if result['is_correct']:
                correct += 1
            
            source = result.get('source', 'Unknown')
            if source in source_stats:
                source_stats[source] += 1
            
            # Collect results to buffer (batch writing)
            predicted_peptide = result.get('peptide', '')
            predicted_peptides.append(predicted_peptide)
            
            # Batch write every 100 spectra
            if total % 100 == 0:
                for pep in predicted_peptides:
                    results_handle.write(f"{pep}\n")
                results_handle.flush()
                predicted_peptides = []
            
            # 每500个输出一次
            if total % 500 == 0:
                logger.info(f"\n{'='*80}")
                logger.info(f"Progress: {total}/{total_spectra}")
                logger.info(f"{'='*80}")
                logger.info(f"Correct: {correct}/{total}")
                logger.info(f"Accuracy: {correct/total*100:.2f}%")
                logger.info(f"Source stats:")
                for src, cnt in source_stats.items():
                    if cnt > 0:
                        logger.info(f"  {src}: {cnt} ({cnt/total*100:.1f}%)")
                logger.info(f"{'='*80}\n")
            
        except Exception as e:
            tqdm.write(f"Error processing spectrum {idx}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Write remaining results
    if predicted_peptides:
        for pep in predicted_peptides:
            results_handle.write(f"{pep}\n")
        results_handle.flush()
    
    # 关闭结果文件
    results_handle.close()
    logger.info(f"\nPredicted peptides saved to: {results_file}")
    
    # 最终结果
    logger.info(f"\n{'='*80}")
    logger.info("Final Results")
    logger.info(f"{'='*80}")
    logger.info(f"Total spectra: {total}")
    logger.info(f"Correct predictions: {correct}")
    logger.info(f"Accuracy: {correct/total*100:.2f}%")
    logger.info(f"")
    logger.info("Source statistics:")
    for source, count in source_stats.items():
        if count > 0:
            logger.info(f"  {source}: {count} ({count/total*100:.1f}%)")
    logger.info(f"")
    logger.info(f"Results saved in: {output_dir}/")
    logger.info(f"{'='*80}")


if __name__ == "__main__":
    main()