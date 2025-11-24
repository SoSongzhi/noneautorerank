#!/usr/bin/env python3
"""
PiPrime + HighNine集成测试
在100个谱图上测试准确率

使用PiPrime生成候选peptide和spectrum embedding，
使用HighNine进行reranking

每个谱图的详细结果保存到单独的文件中
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

# 添加路径 - 使用当前目录的PrimeNovo
sys.path.insert(0, os.path.dirname(__file__))

from PrimeNovo.denovo.model import Spec2Pep
from piprime_with_mass_check import PiPrimeWithMassCheck
from piprime_reranker import process_peaks, load_piprime_config
from piprime_efficient_reranker import PiPrimeEfficientReranker
from test_beam_search_accuracy import beam_search_with_mass_pruning, calculate_peptide_mass_from_tokens
from piprime_mass_calculator import (
    calculate_peptide_mass_piprime,
    calculate_precursor_mass_from_mz,
    check_mass_match
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


def normalize_peptide(peptide: str) -> str:
    """标准化peptide序列（L->I）"""
    return peptide.replace('L', 'I')


class PiPrimeHighNineProcessor:
    """PiPrime + HighNine集成处理器"""
    
    def __init__(self, piprime_model_path: str, index_file: str, output_dir: str, device: str = 'cuda'):
        """
        初始化处理器
        
        Parameters:
        -----------
        piprime_model_path : str
            PiPrime模型路径
        index_file : str
            HighNine索引文件路径
        output_dir : str
            输出目录（保存每个spectrum的详细结果）
        device : str
            设备（'cuda' 或 'cpu'）
        """
        from pathlib import Path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        logger.info(f"使用设备: {self.device}")
        
        # 加载PiPrime模型
        logger.info(f"加载PiPrime模型: {piprime_model_path}")
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
        
        # 加载HighNine reranker（使用PiPrime encoder）
        logger.info(f"加载HighNine索引: {index_file}")
        self.reranker = PiPrimeEfficientReranker(
            piprime_model=self.piprime_model,
            piprime_config=self.piprime_config
        )
        self.reranker.load_precomputed_index(index_file)
        
        logger.info("✅ 初始化完成")
    
    def process_single_spectrum(
        self, 
        mz_array: np.ndarray,
        intensity_array: np.ndarray,
        precursor_mz: float,
        precursor_charge: int,
        ground_truth: str = ""
    ) -> Dict:
        """
        处理单个谱图
        
        Parameters:
        -----------
        mz_array : np.ndarray
            m/z数组
        intensity_array : np.ndarray
            强度数组
        precursor_mz : float
            Precursor m/z
        precursor_charge : int
            Precursor电荷
        ground_truth : str
            真实peptide序列（用于统计）
            
        Returns:
        --------
        Dict : 处理结果
        """
        # 1. 预处理谱图
        peaks = process_peaks(
            mz_array, intensity_array, 
            precursor_mz, precursor_charge, 
            self.piprime_config
        )
        peaks = peaks.to(self.device)
        
        # 使用正确的方法计算precursor mass
        precursor_mass = calculate_precursor_mass_from_mz(precursor_mz, precursor_charge)
        precursors = torch.tensor(
            [[precursor_mass, precursor_charge, precursor_mz]],
            dtype=torch.float32,
            device=self.device
        )
        
        # 2. PiPrime前向传播
        with torch.no_grad():
            # Encoder: 生成spectrum embedding
            enc_out, enc_mask = self.piprime_model.encoder(peaks.unsqueeze(0))
            spectrum_embedding = enc_out.mean(dim=1).cpu().numpy()[0]
            
            # Decoder: 生成log probability matrix
            output_logits, _, _ = self.piprime_model.decoder(
                None, precursors, enc_out, enc_mask
            )
            log_prob_matrix = F.log_softmax(output_logits[0], dim=-1)
        
        # 3. Beam search生成候选peptide（带质量过滤）
        candidates_raw = beam_search_with_mass_pruning(
            self.piprime_predictor,
            log_prob_matrix,
            precursor_mz,
            precursor_charge,
            target_count=1000,
            top_n=10
        )
        
        # 转换为reranker需要的格式，并保存所有候选（包括未通过质量检查的）
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
            
            if passes:  # 只保留通过质量检查的用于reranking
                candidates_passed.append({
                    'peptide': peptide,
                    'score': score
                })
        
        # 4. 为所有通过质量检查的候选计算similarity
        if not candidates_passed:
            logger.warning(f"没有候选peptide通过质量检查 (总候选数: {len(candidates_all)}) - 标记为Failed")
            # 不使用fallback，直接返回Failed
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
            # 为所有通过质量检查的候选计算similarity
            result, similarity_dict, source_dict = self.reranker.rerank_with_external_embedding(
                query_embedding=spectrum_embedding,
                candidates=candidates_passed,
                precursor_mz=precursor_mz,
                precursor_charge=precursor_charge,
                use_prosit=True,
                top_k=3  # Top-K用于Database匹配
            )
        
        # 将similarity和source添加到candidates_all中
        for cand in candidates_all:
            if cand['passes_mass_check'] and cand['peptide'] in similarity_dict:
                cand['similarity'] = similarity_dict[cand['peptide']]
                cand['source'] = source_dict.get(cand['peptide'], 'Unknown')
            else:
                cand['similarity'] = -1.0  # 未通过质量检查或未计算
                cand['source'] = 'NotChecked'  # 未通过质量检查
        
        # 5. 检查准确率
        if result and ground_truth:
            pred_seq = normalize_peptide(result['peptide'])
            true_seq = normalize_peptide(ground_truth)
            result['is_correct'] = (pred_seq == true_seq and true_seq != '')
        else:
            result['is_correct'] = False
        
        result['num_candidates'] = len(candidates_passed)
        result['num_candidates_total'] = len(candidates_all)
        result['num_candidates_passed'] = len([c for c in candidates_all if c['passes_mass_check']])
        
        # 6. 保存详细结果到文件（按similarity排序）
        self._save_spectrum_details(
            spectrum_idx=getattr(self, '_current_spectrum_idx', 0),
            ground_truth=ground_truth,
            candidates_all=candidates_all,
            candidates_passed=candidates_passed,
            result=result,
            precursor_mz=precursor_mz,
            precursor_charge=precursor_charge
        )
        
        return result
    
    def _save_spectrum_details(self, spectrum_idx, ground_truth, candidates_all,
                               candidates_passed, result, precursor_mz, precursor_charge):
        """保存单个spectrum的详细结果到文件"""
        output_file = self.output_dir / f"spectrum_{spectrum_idx:04d}.txt"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write(f"Spectrum {spectrum_idx}\n")
            f.write("="*80 + "\n\n")
            
            # Spectrum信息
            f.write("Spectrum Information:\n")
            f.write("-"*80 + "\n")
            f.write(f"Precursor m/z: {precursor_mz:.4f}\n")
            f.write(f"Precursor charge: {precursor_charge}\n")
            f.write(f"Ground truth: {ground_truth}\n")
            f.write(f"Ground truth (normalized): {normalize_peptide(ground_truth)}\n")
            f.write("\n")
            
            # Beam Search结果统计
            f.write("Beam Search Results:\n")
            f.write("-"*80 + "\n")
            f.write(f"Total candidates: {len(candidates_all)}\n")
            passed_count = sum(1 for c in candidates_all if c['passes_mass_check'])
            f.write(f"Passed mass check: {passed_count}\n")
            f.write(f"Used for reranking: {len(candidates_passed)}\n")
            f.write("\n")
            
            # 统计信息
            total_cands = len(candidates_all)
            passed_cands = sum(1 for c in candidates_all if c['passes_mass_check'])
            db_cands = sum(1 for c in candidates_all if c.get('source') == 'Database')
            prosit_cands = sum(1 for c in candidates_all if c.get('source') == 'Prosit')
            failed_cands = sum(1 for c in candidates_all if c.get('source') == 'Failed')
            
            f.write("Candidate Statistics:\n")
            f.write("-"*80 + "\n")
            f.write(f"Total candidates: {total_cands}\n")
            f.write(f"Passed mass check: {passed_cands} ({passed_cands/total_cands*100:.1f}%)\n")
            f.write(f"Database matches: {db_cands} ({db_cands/total_cands*100:.1f}%)\n")
            f.write(f"Prosit predictions: {prosit_cands} ({prosit_cands/total_cands*100:.1f}%)\n")
            f.write(f"Failed: {failed_cands} ({failed_cands/total_cands*100:.1f}%)\n")
            f.write("\n")
            
            # 找到ground truth在候选中的位置
            ground_truth_norm = normalize_peptide(ground_truth)
            gt_found = False
            gt_rank_beam = -1
            gt_rank_similarity = -1
            gt_similarity = -1.0
            gt_source = "NotFound"
            
            for i, cand in enumerate(candidates_all, 1):
                if normalize_peptide(cand['peptide']) == ground_truth_norm:
                    gt_found = True
                    gt_rank_beam = i
                    gt_similarity = cand.get('similarity', -1.0)
                    gt_source = cand.get('source', 'Unknown')
                    break
            
            if gt_found:
                # 按similarity排序后找到ground truth的位置
                candidates_sorted = sorted(
                    candidates_all,
                    key=lambda x: x.get('similarity', -1.0),
                    reverse=True
                )
                for i, cand in enumerate(candidates_sorted, 1):
                    if normalize_peptide(cand['peptide']) == ground_truth_norm:
                        gt_rank_similarity = i
                        break
                
                f.write("Ground Truth Analysis:\n")
                f.write("-"*80 + "\n")
                f.write(f"Ground truth found: ✓\n")
                f.write(f"Rank in beam search: {gt_rank_beam}/{total_cands}\n")
                f.write(f"Rank by similarity: {gt_rank_similarity}/{total_cands}\n")
                f.write(f"Similarity: {gt_similarity:.4f}\n")
                f.write(f"Source: {gt_source}\n")
                f.write(f"Passed mass check: {'✓' if candidates_all[gt_rank_beam-1]['passes_mass_check'] else '✗'}\n")
            else:
                f.write("Ground Truth Analysis:\n")
                f.write("-"*80 + "\n")
                f.write(f"Ground truth found: ✗ (not in top {total_cands} candidates)\n")
            f.write("\n")
            
            # 所有候选peptide（按similarity排序）
            candidates_sorted = sorted(
                candidates_all,
                key=lambda x: x.get('similarity', -1.0),
                reverse=True
            )
            
            f.write(f"All {len(candidates_sorted)} Candidates (sorted by similarity):\n")
            f.write("-"*120 + "\n")
            f.write(f"{'Rank':<6}{'Peptide':<25}{'Similarity':<12}{'Source':<12}{'BeamScore':<12}{'Mass':<12}{'Pass':<8}{'Correct'}\n")
            f.write("-"*120 + "\n")
            
            for i, cand in enumerate(candidates_sorted, 1):
                peptide_norm = normalize_peptide(cand['peptide'])
                is_correct = (peptide_norm == ground_truth_norm and ground_truth_norm != '')
                pass_mark = "✓" if cand['passes_mass_check'] else "✗"
                correct_mark = "✓" if is_correct else ""
                similarity = cand.get('similarity', -1.0)
                source = cand.get('source', 'Unknown')
                
                f.write(f"{i:<6}{cand['peptide']:<25}{similarity:<12.4f}{source:<12}{cand['score']:<12.4f}"
                       f"{cand['mass']:<12.4f}{pass_mark:<8}{correct_mark}\n")
            
            f.write("\n")
            
            # Reranking结果
            f.write("Reranking Result:\n")
            f.write("-"*80 + "\n")
            if result and result.get('peptide'):
                f.write(f"Top-1 Peptide: {result['peptide']}\n")
                f.write(f"Similarity: {result.get('similarity', -1.0):.4f}\n")
                f.write(f"De Novo Score: {result.get('denovo_score', 0.0):.4f}\n")
                f.write(f"Source: {result.get('source', 'Unknown')}\n")
                f.write(f"Correct: {'✓' if result.get('is_correct', False) else '✗'}\n")
            else:
                f.write("No result\n")
            
            f.write("\n")


def main():
    """主函数"""
    # 配置路径
    # 使用当前目录下的文件
    mgf_file = "testdata/high_nine_validation_1000_converted.mgf"
    piprime_model = "model_massive.ckpt"
    index_file = "test_data/high_nine/high_nine_database.mgf.efficient_index.pkl"
    
    # 创建输出目录（带时间戳）
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"piprime_highnine_results_{timestamp}"
    
    # 检查文件是否存在
    if not os.path.exists(mgf_file):
        logger.error(f"MGF文件不存在: {mgf_file}")
        return
    
    if not os.path.exists(piprime_model):
        logger.error(f"PiPrime模型不存在: {piprime_model}")
        return
    
    if not os.path.exists(index_file):
        logger.error(f"索引文件不存在: {index_file}")
        logger.info("请先运行 build_efficient_index.py 构建索引")
        return
    
    logger.info(f"\n{'='*80}")
    logger.info("PiPrime + HighNine 集成测试")
    logger.info(f"{'='*80}")
    logger.info(f"MGF文件: {mgf_file}")
    logger.info(f"PiPrime模型: {piprime_model}")
    logger.info(f"索引文件: {index_file}")
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"测试规模: 1000个谱图")
    logger.info(f"{'='*80}\n")
    
    # 初始化处理器
    processor = PiPrimeHighNineProcessor(piprime_model, index_file, output_dir)
    
    # 统计变量
    total = 0
    correct = 0
    total_candidates = 0
    source_stats = {
        'Database': 0,
        'Prosit': 0,
        'Failed': 0,
        'NoCandidates': 0
    }
    
    # 处理谱图
    logger.info("开始处理谱图...\n")
    
    with mgf.MGF(mgf_file) as reader:
        for idx, spec in enumerate(reader):
            if idx >= 1000:
                break
            
            # 提取信息
            mz_array = spec['m/z array']
            intensity_array = spec['intensity array']
            
            pepmass = spec['params'].get('pepmass', [0])
            precursor_mz = pepmass[0] if isinstance(pepmass, (list, tuple)) else pepmass
            
            charge = spec['params'].get('charge', [2])
            precursor_charge = charge[0] if isinstance(charge, (list, tuple)) else charge
            if isinstance(precursor_charge, str):
                precursor_charge = int(precursor_charge.replace('+', ''))
            
            ground_truth = spec['params'].get('seq', '')
            if not ground_truth:
                logger.warning(f"谱图 {idx} 没有SEQ字段，跳过")
                continue
            
            # 处理谱图
            try:
                # 设置当前spectrum索引（用于保存文件）
                processor._current_spectrum_idx = idx
                
                result = processor.process_single_spectrum(
                    mz_array, intensity_array,
                    precursor_mz, precursor_charge,
                    ground_truth
                )
                
                # 更新统计
                total += 1
                if result['is_correct']:
                    correct += 1
                
                total_candidates += result.get('num_candidates', 0)
                source = result.get('source', 'Unknown')
                if source in source_stats:
                    source_stats[source] += 1
                
                # 每10个输出一次进度
                if total % 10 == 0:
                    logger.info(f"已处理: {total}/1000")
                    logger.info(f"  当前准确率: {correct}/{total} = {correct/total*100:.2f}%")
                    logger.info(f"  平均候选数: {total_candidates/total:.1f}")
                    logger.info("")
                
            except Exception as e:
                logger.error(f"处理谱图 {idx} 时出错: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    # 最终结果
    logger.info(f"\n{'='*80}")
    logger.info("最终结果")
    logger.info(f"{'='*80}")
    logger.info(f"总谱图数: {total}")
    logger.info(f"正确预测: {correct}")
    logger.info(f"准确率: {correct/total*100:.2f}%" if total > 0 else "准确率: N/A")
    logger.info(f"")
    logger.info(f"平均候选数: {total_candidates/total:.1f}" if total > 0 else "平均候选数: N/A")
    logger.info(f"")
    logger.info("来源统计:")
    for source, count in source_stats.items():
        if count > 0:
            logger.info(f"  {source}: {count} ({count/total*100:.1f}%)")
    logger.info(f"")
    logger.info(f"详细结果保存在: {output_dir}/")
    logger.info(f"{'='*80}")


if __name__ == "__main__":
    main()