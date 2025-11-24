#!/usr/bin/env python3
"""
测试Beam Search在1000个谱图上的准确率
每个谱图生成1000条候选peptide，使用质量过滤
每个谱图的结果保存到单独的文件中
"""

import sys
import os
import torch
import numpy as np
import logging
from datetime import datetime
from typing import List, Tuple, Dict, Optional
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))

from piprime_with_mass_check import PiPrimeWithMassCheck
from PrimeNovo.denovo.model import Spec2Pep
from pyteomics import mgf, mass as pyteomics_mass
from piprime_reranker import process_peaks, load_piprime_config
from piprime_mass_calculator import (
    calculate_peptide_mass_piprime,
    calculate_precursor_mass_from_mz,
    check_mass_match
)

# Import PMC module
try:
    from PrimeNovo.denovo import mass_con
    PMC_AVAILABLE = True
except ImportError:
    PMC_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


def calculate_peptide_mass_from_tokens(predictor, tokens):
    """
    从token序列计算peptide质量
    使用与piprime_with_mass_check相同的逻辑，支持修饰
    
    注意：predictor._calculate_peptide_mass已经包含了H2O质量！
    """
    try:
        if not tokens:
            return None
        
        # 使用predictor的_calculate_peptide_mass方法
        # 这个方法使用calculate_peptide_mass_piprime，已经包含H2O
        mass = predictor._calculate_peptide_mass(tokens)
        
        return mass
    except Exception as e:
        logger.warning(f"计算质量失败: {str(e)}, tokens: {tokens[:10]}")
        return None


def get_pmc_result(
    predictor,
    log_prob_matrix: torch.Tensor,
    precursor_mz: float,
    precursor_charge: int
) -> Optional[Tuple[str, float, float, bool]]:
    """
    使用PMC生成peptide
    
    Returns:
    --------
    Optional[Tuple[str, float, float, bool]] : (peptide, score, mass, passes_mass_check) 或 None
    """
    if not PMC_AVAILABLE:
        return None
    
    try:
        # 使用正确的precursor质量计算方法
        precursor_mass = calculate_precursor_mass_from_mz(precursor_mz, precursor_charge)
        mass_tolerance_da = 0.1
        min_mass = precursor_mass - mass_tolerance_da
        max_mass = precursor_mass + mass_tolerance_da
        
        precursor_mass_tensor = torch.tensor([[precursor_mass]], dtype=torch.float32, device=log_prob_matrix.device)
        mass_tolerance = 0.1
        
        # 调用PMC
        pmc_tokens = mass_con.knapDecode(
            log_prob_matrix.unsqueeze(0),
            precursor_mass_tensor,
            mass_tolerance
        )
        
        # CTC collapse
        def ctc_post_processing(tokens):
            result = []
            prev = None
            for t in tokens:
                if t != prev:
                    result.append(t)
                    prev = t
            result = [t for t in result if t != 27]
            return result
        
        pmc_tokens_collapsed = ctc_post_processing(pmc_tokens)
        
        # 转换为peptide
        pmc_aa_seq = [predictor.decoder._idx2aa.get(t, '?') for t in pmc_tokens_collapsed]
        pmc_aa_seq = [aa for aa in pmc_aa_seq if aa and aa != '_' and aa != '?']
        
        if predictor.decoder.reverse:
            pmc_aa_seq = list(reversed(pmc_aa_seq))
        
        pmc_peptide = "".join(pmc_aa_seq)
        
        # 计算质量
        pmc_mass = calculate_peptide_mass_from_tokens(predictor, pmc_tokens_collapsed)
        if pmc_mass is None:
            pmc_mass = 0.0
        
        # 计算score
        pmc_score = 0.0
        for h in range(len(pmc_tokens)):
            token_idx = pmc_tokens[h]
            if h < log_prob_matrix.shape[0]:
                pmc_score += log_prob_matrix[h, token_idx].item()
        
        # 质量检查
        passes = min_mass <= pmc_mass <= max_mass
        
        return (pmc_peptide, pmc_score, pmc_mass, passes)
    
    except Exception as e:
        logger.warning(f"PMC生成失败: {str(e)}")
        return None


def beam_search_with_mass_pruning(
    predictor,
    log_prob_matrix: torch.Tensor,
    precursor_mz: float,
    precursor_charge: int,
    target_count: int = 1000,
    top_n: int = 10
) -> List[Tuple[str, float, float, bool]]:
    """
    使用质量剪枝的Beam Search生成候选peptide
    
    Parameters:
    -----------
    predictor : PiPrimeWithMassCheck
        预测器对象
    log_prob_matrix : torch.Tensor
        Log probability matrix [seq_len, vocab_size]
    precursor_mz : float
        Precursor m/z
    precursor_charge : int
        Precursor电荷
    target_count : int
        目标候选数量（默认1000）
    top_n : int
        每步选择的top-N候选（默认10）
        
    Returns:
    --------
    List[Tuple[str, float, float, bool]] : 候选peptide列表
        每个元素为 (peptide, score, mass, passes_mass_check)
    """
    seq_len, vocab_size = log_prob_matrix.shape
    
    # 计算precursor mass和质量范围
    # 使用正确的precursor质量计算方法
    precursor_mass = calculate_precursor_mass_from_mz(precursor_mz, precursor_charge)
    mass_tolerance_da = 0.1
    min_mass = precursor_mass - mass_tolerance_da
    max_mass = precursor_mass + mass_tolerance_da
    
    # 动态beam width
    def get_beam_width(t):
        if t == 0: return 10
        elif t == 1: return 100
        elif t == 2: return 1000
        else: return target_count
    
    # 初始化
    paths = [(0.0, [])]
    total_pruned = 0
    
    # 逐步扩展
    for t in range(seq_len):
        current_beam_width = get_beam_width(t)
        top_log_probs, top_indices = torch.topk(log_prob_matrix[t], top_n)
        new_paths = []
        pruned_this_step = 0
        
        for current_log_prob, current_path in paths:
            for log_prob, idx in zip(top_log_probs, top_indices):
                idx = idx.item()
                log_prob = log_prob.item()
                
                new_path = current_path + [idx]
                new_log_prob = current_log_prob + log_prob
                
                # 质量剪枝
                collapsed = predictor._ctc_collapse(new_path)
                collapsed_no_blank = [t for t in collapsed if t != 27]
                peptide_mass = calculate_peptide_mass_from_tokens(predictor, collapsed_no_blank)
                
                if peptide_mass is not None and peptide_mass > max_mass:
                    pruned_this_step += 1
                    continue
                
                new_paths.append((new_log_prob, new_path))
        
        new_paths.sort(reverse=True)
        paths = new_paths[:current_beam_width]
        total_pruned += pruned_this_step
        
        if not paths:
            break
    
    # 转换为peptide列表
    results = []
    for log_prob, tokens in paths[:target_count]:
        # CTC collapse
        collapsed = predictor._ctc_collapse(tokens)
        collapsed_no_blank = [t for t in collapsed if t != 27]
        
        # 计算质量
        peptide_mass = calculate_peptide_mass_from_tokens(predictor, collapsed_no_blank)
        if peptide_mass is None:
            peptide_mass = 0.0
        
        # 转换为peptide
        aa_seq = [predictor.decoder._idx2aa.get(t, '?') for t in collapsed_no_blank]
        aa_seq = [aa for aa in aa_seq if aa and aa != '_' and aa != '?']
        if predictor.decoder.reverse:
            aa_seq = list(reversed(aa_seq))
        peptide = "".join(aa_seq)
        
        # 质量检查
        passes = min_mass <= peptide_mass <= max_mass
        
        results.append((peptide, log_prob, peptide_mass, passes))
    
    return results


def save_spectrum_results(
    output_dir: str,
    spec_idx: int,
    ground_truth: str,
    pmc_result: Optional[Tuple[str, float, float, bool]],
    beam_results: List[Tuple[str, float, float, bool]],
    precursor_mz: float,
    precursor_charge: int
):
    """
    保存单个谱图的结果到文件
    
    Parameters:
    -----------
    output_dir : str
        输出目录
    spec_idx : int
        谱图索引
    ground_truth : str
        真实peptide序列
    pmc_result : Optional[Tuple]
        PMC结果 (peptide, score, mass, passes)
    beam_results : List[Tuple]
        Beam search结果列表
    precursor_mz : float
        Precursor m/z
    precursor_charge : int
        Precursor电荷
    """
    filename = os.path.join(output_dir, f"spectrum_{spec_idx:04d}.txt")
    
    # 标准化ground truth
    ground_truth_norm = ground_truth.replace('L', 'I')
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write(f"Spectrum {spec_idx}\n")
        f.write("="*80 + "\n\n")
        
        # 谱图信息
        f.write("Spectrum Information:\n")
        f.write("-"*80 + "\n")
        f.write(f"Precursor m/z: {precursor_mz:.4f}\n")
        f.write(f"Precursor charge: {precursor_charge}\n")
        f.write(f"Ground truth: {ground_truth}\n")
        f.write(f"Ground truth (normalized): {ground_truth_norm}\n")
        f.write("\n")
        
        # PMC结果
        f.write("PMC (Precursor Mass Control) Result:\n")
        f.write("-"*80 + "\n")
        if pmc_result:
            peptide, score, mass, passes = pmc_result
            peptide_norm = peptide.replace('L', 'I')
            is_correct = (peptide_norm == ground_truth_norm)
            status = "✓ CORRECT" if is_correct else ""
            mass_status = "PASS" if passes else "FAIL"
            
            f.write(f"Peptide: {peptide} {status}\n")
            f.write(f"Score: {score:.4f}\n")
            f.write(f"Mass: {mass:.4f} Da ({mass_status})\n")
        else:
            f.write("PMC not available or failed\n")
        f.write("\n")
        
        # Beam search结果
        f.write("Beam Search Results (with Mass Pruning):\n")
        f.write("-"*80 + "\n")
        f.write(f"Total candidates: {len(beam_results)}\n")
        passed_count = sum(1 for _, _, _, passes in beam_results if passes)
        f.write(f"Passed mass check: {passed_count}\n")
        f.write("\n")
        
        f.write("Rank\tPeptide\tScore\tMass\tMass_Check\tCorrect\n")
        f.write("-"*80 + "\n")
        
        for rank, (peptide, score, mass, passes) in enumerate(beam_results, 1):
            peptide_norm = peptide.replace('L', 'I')
            is_correct = (peptide_norm == ground_truth_norm)
            correct_mark = "✓" if is_correct else ""
            mass_status = "PASS" if passes else "FAIL"
            
            f.write(f"{rank}\t{peptide}\t{score:.4f}\t{mass:.4f}\t{mass_status}\t{correct_mark}\n")


def normalize_peptide(peptide: str) -> str:
    """标准化peptide序列（L->I）"""
    return peptide.replace('L', 'I')


def calculate_accuracy(
    pmc_result: Optional[Tuple[str, float, float, bool]],
    candidates: List[Tuple[str, float, float, bool]],
    ground_truth: str
) -> Dict:
    """
    计算准确率
    
    Returns:
    --------
    Dict : {
        'pmc_correct': bool,
        'top1_correct': bool,
        'top10_correct': bool,
        'top100_correct': bool,
        'any_correct': bool,
        'total_candidates': int,
        'passed_mass_check': int
    }
    """
    ground_truth_norm = normalize_peptide(ground_truth)
    
    # PMC准确率
    pmc_correct = False
    if pmc_result:
        pmc_peptide, _, _, _ = pmc_result
        pmc_correct = (normalize_peptide(pmc_peptide) == ground_truth_norm)
    
    # Beam search准确率
    top1_correct = False
    top10_correct = False
    top100_correct = False
    any_correct = False
    
    passed_count = sum(1 for _, _, _, passes in candidates if passes)
    
    for rank, (peptide, score, mass, passes) in enumerate(candidates, 1):
        peptide_norm = normalize_peptide(peptide)
        
        if peptide_norm == ground_truth_norm:
            any_correct = True
            if rank == 1:
                top1_correct = True
            if rank <= 10:
                top10_correct = True
            if rank <= 100:
                top100_correct = True
            break
    
    return {
        'pmc_correct': pmc_correct,
        'top1_correct': top1_correct,
        'top10_correct': top10_correct,
        'top100_correct': top100_correct,
        'any_correct': any_correct,
        'total_candidates': len(candidates),
        'passed_mass_check': passed_count
    }


def main():
    # 设置 - 使用当前目录
    mgf_file = "testdata/high_nine_validation_1000_converted.mgf"
    model_path = "model_massive.ckpt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"beam_search_results_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"{'='*80}")
    logger.info(f"Beam Search准确率测试")
    logger.info(f"{'='*80}")
    logger.info(f"MGF文件: {mgf_file}")
    logger.info(f"模型: {model_path}")
    logger.info(f"设备: {device}")
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"每个谱图生成: 1000条候选")
    logger.info(f"质量容差: ±0.1 Da")
    logger.info(f"PMC可用: {PMC_AVAILABLE}")
    logger.info(f"{'='*80}\n")
    
    # 加载模型
    logger.info("加载模型...")
    model = Spec2Pep.load_from_checkpoint(model_path, map_location=device)
    model.eval()
    model.to(device)
    
    predictor = PiPrimeWithMassCheck(
        model,
        precursor_mass_tol=50,
        isotope_error_range=(0, 1),
        beam_width=100
    )
    
    config = load_piprime_config()
    
    # 统计变量
    total_spectra = 0
    pmc_correct = 0
    top1_correct = 0
    top10_correct = 0
    top100_correct = 0
    any_correct = 0
    total_candidates = 0
    total_passed = 0
    
    # 处理谱图
    logger.info(f"\n开始处理谱图...\n")
    
    with mgf.MGF(mgf_file) as reader:
        for spec_idx, spec in enumerate(reader):
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
                logger.warning(f"谱图 {spec_idx} 没有SEQ字段，跳过")
                continue
            
            # 预处理
            peaks = process_peaks(mz_array, intensity_array, precursor_mz, precursor_charge, config)
            peaks = peaks.to(device)
            
            # 使用正确的precursor质量计算方法
            precursor_mass = calculate_precursor_mass_from_mz(precursor_mz, precursor_charge)
            precursors = torch.tensor(
                [[precursor_mass, precursor_charge, precursor_mz]], 
                dtype=torch.float32, 
                device=device
            )
            
            # 获取log probability matrix
            with torch.no_grad():
                enc_out, enc_mask = model.encoder(peaks.unsqueeze(0))
                output_logits, _, _ = model.decoder(None, precursors, enc_out, enc_mask)
                log_prob_matrix = F.log_softmax(output_logits[0], dim=-1)
            
            # PMC结果
            pmc_result = get_pmc_result(predictor, log_prob_matrix, precursor_mz, precursor_charge)
            
            # Beam search
            beam_results = beam_search_with_mass_pruning(
                predictor,
                log_prob_matrix,
                precursor_mz,
                precursor_charge,
                target_count=1000,
                top_n=10
            )
            
            # 保存结果到文件
            save_spectrum_results(
                output_dir,
                spec_idx,
                ground_truth,
                pmc_result,
                beam_results,
                precursor_mz,
                precursor_charge
            )
            
            # 计算准确率
            accuracy = calculate_accuracy(pmc_result, beam_results, ground_truth)
            
            # 更新统计
            total_spectra += 1
            if accuracy['pmc_correct']:
                pmc_correct += 1
            if accuracy['top1_correct']:
                top1_correct += 1
            if accuracy['top10_correct']:
                top10_correct += 1
            if accuracy['top100_correct']:
                top100_correct += 1
            if accuracy['any_correct']:
                any_correct += 1
            total_candidates += accuracy['total_candidates']
            total_passed += accuracy['passed_mass_check']
            
            # 输出进度
            if (spec_idx + 1) % 10 == 0:
                logger.info(f"已处理: {spec_idx + 1} 个谱图")
                if PMC_AVAILABLE:
                    logger.info(f"  PMC准确率: {pmc_correct}/{total_spectra} = {pmc_correct/total_spectra*100:.2f}%")
                logger.info(f"  Top-1准确率: {top1_correct}/{total_spectra} = {top1_correct/total_spectra*100:.2f}%")
                logger.info(f"  Top-10准确率: {top10_correct}/{total_spectra} = {top10_correct/total_spectra*100:.2f}%")
                logger.info(f"  Top-100准确率: {top100_correct}/{total_spectra} = {top100_correct/total_spectra*100:.2f}%")
                logger.info(f"  Any准确率: {any_correct}/{total_spectra} = {any_correct/total_spectra*100:.2f}%\n")
    
    # 最终统计
    logger.info(f"\n{'='*80}")
    logger.info(f"最终结果")
    logger.info(f"{'='*80}")
    logger.info(f"总谱图数: {total_spectra}")
    logger.info(f"输出目录: {os.path.abspath(output_dir)}")
    logger.info(f"")
    if PMC_AVAILABLE:
        logger.info(f"PMC准确率: {pmc_correct}/{total_spectra} = {pmc_correct/total_spectra*100:.2f}%")
    logger.info(f"Top-1准确率: {top1_correct}/{total_spectra} = {top1_correct/total_spectra*100:.2f}%")
    logger.info(f"Top-10准确率: {top10_correct}/{total_spectra} = {top10_correct/total_spectra*100:.2f}%")
    logger.info(f"Top-100准确率: {top100_correct}/{total_spectra} = {top100_correct/total_spectra*100:.2f}%")
    logger.info(f"Any准确率: {any_correct}/{total_spectra} = {any_correct/total_spectra*100:.2f}%")
    logger.info(f"")
    logger.info(f"平均候选数: {total_candidates/total_spectra:.1f}")
    logger.info(f"平均通过质量检查: {total_passed/total_spectra:.1f}")
    logger.info(f"质量检查通过率: {total_passed/total_candidates*100:.2f}%")
    logger.info(f"{'='*80}")
    
    # 保存总结文件
    summary_file = os.path.join(output_dir, "summary.txt")
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("Beam Search Accuracy Test Summary\n")
        f.write("="*80 + "\n\n")
        f.write(f"Total spectra: {total_spectra}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"MGF file: {mgf_file}\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Device: {device}\n")
        f.write(f"PMC available: {PMC_AVAILABLE}\n")
        f.write("\n")
        f.write("Accuracy Results:\n")
        f.write("-"*80 + "\n")
        if PMC_AVAILABLE:
            f.write(f"PMC accuracy: {pmc_correct}/{total_spectra} = {pmc_correct/total_spectra*100:.2f}%\n")
        f.write(f"Top-1 accuracy: {top1_correct}/{total_spectra} = {top1_correct/total_spectra*100:.2f}%\n")
        f.write(f"Top-10 accuracy: {top10_correct}/{total_spectra} = {top10_correct/total_spectra*100:.2f}%\n")
        f.write(f"Top-100 accuracy: {top100_correct}/{total_spectra} = {top100_correct/total_spectra*100:.2f}%\n")
        f.write(f"Any accuracy: {any_correct}/{total_spectra} = {any_correct/total_spectra*100:.2f}%\n")
        f.write("\n")
        f.write("Statistics:\n")
        f.write("-"*80 + "\n")
        f.write(f"Average candidates per spectrum: {total_candidates/total_spectra:.1f}\n")
        f.write(f"Average passed mass check: {total_passed/total_spectra:.1f}\n")
        f.write(f"Mass check pass rate: {total_passed/total_candidates*100:.2f}%\n")
    
    logger.info(f"\n✅ 总结文件已保存: {summary_file}")


if __name__ == "__main__":
    main()