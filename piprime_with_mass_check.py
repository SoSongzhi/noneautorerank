#!/usr/bin/env python
"""
PiPrime + HighNine质量检查的混合生成器

功能:
1. 使用PiPrime CTC Beam Search生成候选peptide
2. 包括PMC (Precursor Mass Control) 结果
3. 使用HighNine的precursor mass检查方法
4. 输出100条通过质量检查的候选peptide
"""

import torch
import torch.nn.functional as F
import numpy as np
import logging
from typing import List, Dict, Tuple
from pyteomics import mass as pyteomics_mass
from piprime_mass_calculator import (
    calculate_peptide_mass_piprime,
    calculate_precursor_mass_from_mz,
    check_mass_match,
    normalize_sequence_format,
    AA2MAS,
    H2O_MASS
)

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class PiPrimeWithMassCheck:
    """PiPrime生成器 + HighNine质量检查"""
    
    def __init__(self, model, precursor_mass_tol=50, isotope_error_range=(0, 1), beam_width=500):
        """
        初始化
        
        Parameters:
        -----------
        model : Spec2Pep
            PiPrime模型
        precursor_mass_tol : float
            Precursor质量容差 (ppm)
        isotope_error_range : Tuple[int, int]
            同位素误差范围
        beam_width : int
            CTC Beam Search的beam宽度（默认500）
        """
        self.model = model
        self.device = next(model.parameters()).device
        self.precursor_mass_tol = precursor_mass_tol
        self.isotope_error_range = isotope_error_range
        self.beam_width = beam_width
        
        # 获取decoder和token masses
        self.decoder = model.decoder
        self.token_masses = model.decoder._peptide_mass.masses
        
        # ===== 重要：重新初始化CTC decoder，设置更大的beam_width =====
        logger.info(f"🔧 重新初始化CTC Decoder (beam_width={beam_width})")
        
        # 导入CTC decoder - 使用当前目录
        import sys
        import os
        sys.path.insert(0, os.path.dirname(__file__))
        from PrimeNovo.denovo.ctc_beam_search import CTCBeamSearchDecoder
        
        # 创建新的CTC decoder，使用更大的beam_width
        ctc_params = {"beam": beam_width}
        self.ctc_decoder = CTCBeamSearchDecoder(self.decoder, ctc_params)
        
        logger.info(f"✅ PiPrime质量检查生成器初始化完成")
        logger.info(f"   - Beam width: {beam_width}")
        logger.info(f"   - Precursor tolerance: {precursor_mass_tol} ppm")
        logger.info(f"   - Isotope range: {isotope_error_range}")
    
    def generate_candidates_with_mass_check(
        self,
        peaks: torch.Tensor,
        precursor_mz: float,
        precursor_charge: int,
        target_count: int = 100,
        max_candidates: int = 500
    ) -> List[Dict]:
        """
        生成候选peptide并进行质量检查
        
        Parameters:
        -----------
        peaks : torch.Tensor
            谱图峰 (n_peaks, 2)
        precursor_mz : float
            Precursor m/z
        precursor_charge : int
            Precursor电荷
        target_count : int
            目标候选数量 (默认100)
        max_candidates : int
            最大生成候选数 (默认500)
            
        Returns:
        --------
        List[Dict] : 通过质量检查的候选peptide列表
            每个dict包含: {'peptide': str, 'score': float, 'mass_error_ppm': float, 
                          'source': str, 'passes_mass_check': bool}
        """
        peaks = peaks.to(self.device)
        # 使用正确的precursor质量计算方法
        precursor_mass = calculate_precursor_mass_from_mz(precursor_mz, precursor_charge)
        precursors = torch.tensor(
            [[precursor_mass, precursor_charge, precursor_mz]],
            dtype=torch.float32,
            device=self.device
        )
        
        results = []
        
        # ===== 步骤1: 获取PMC结果 (质量控制后的最优结果) =====
        logger.info(f"\n{'='*60}")
        logger.info("步骤1: 获取PMC (Precursor Mass Control) 结果")
        logger.info(f"{'='*60}")
        
        with torch.no_grad():
            # 前向传播
            enc_out, enc_mask = self.model.encoder(peaks.unsqueeze(0))
            output_logits, _, _ = self.model.decoder(None, precursors, enc_out, enc_mask)
            log_probs = F.log_softmax(output_logits, dim=-1)
            
            # 获取PMC结果 (这是PiPrime内部质量控制后的最优结果)
            # 注意: PiPrime的forward方法会调用PMC
            pmc_peptides, pmc_scores = self.model.forward(
                peaks.unsqueeze(0), 
                precursors, 
                [""]  # dummy true_peps
            )
            
            if pmc_peptides and len(pmc_peptides[0]) > 0:
                pmc_peptide = "".join(pmc_peptides[0])
                pmc_score = pmc_scores[0].item() if torch.is_tensor(pmc_scores[0]) else pmc_scores[0]
                
                # 检查PMC结果的质量
                mass_check_result = self._check_precursor_mass(
                    pmc_peptide, precursor_mz, precursor_charge
                )
                
                results.append({
                    'peptide': pmc_peptide,
                    'score': pmc_score,
                    'mass_error_ppm': mass_check_result['mass_error_ppm'],
                    'source': 'PMC',
                    'passes_mass_check': mass_check_result['passes']
                })
                
                logger.info(f"✅ PMC结果: {pmc_peptide}")
                logger.info(f"   - Score: {pmc_score:.6f}")
                logger.info(f"   - Mass error: {mass_check_result['mass_error_ppm']:.2f} ppm")
                logger.info(f"   - Passes check: {mass_check_result['passes']}")
        
        # ===== 步骤2: 使用改进的Beam Search生成多条候选 =====
        logger.info(f"\n{'='*60}")
        logger.info(f"步骤2: Improved Beam Search生成候选 (target={max_candidates})")
        logger.info(f"{'='*60}")
        
        with torch.no_grad():
            # 将log_probs转换为归一化的概率矩阵（最大值缩放到20）
            probs = F.softmax(output_logits[0], dim=-1)  # [seq_len, vocab_size]
            
            # 缩放到最大值20
            max_prob = probs.max()
            if max_prob > 0:
                prob_matrix = probs * (20.0 / max_prob)
            else:
                prob_matrix = probs * 20.0
            
            logger.info(f"  Probability matrix shape: {prob_matrix.shape}")
            logger.info(f"  Max value: {prob_matrix.max().item():.4f}")
            logger.info(f"  Min value: {prob_matrix.min().item():.4f}")
            
            # 使用改进的Beam Search
            beam_candidates = self._simple_beam_search(
                prob_matrix,  # 归一化且缩放到20的概率矩阵
                precursor_mz=precursor_mz,
                precursor_charge=precursor_charge,
                beam_width=100,
                top_n=10,
                max_length=30
            )
            
            logger.info(f"✅ Beam Search返回了 {len(beam_candidates)} 条候选")
            
            # 显示前5条
            for idx, cand in enumerate(beam_candidates[:5]):
                logger.info(f"  Beam #{idx+1}: {cand['peptide']} (score={cand['score']:.4f}, len={len(cand['peptide'])})")
            
            for cand in beam_candidates:
                peptide = cand['peptide']
                score = cand['score']
                
                if peptide and peptide not in [r['peptide'] for r in results]:
                    # 检查质量
                    mass_check_result = self._check_precursor_mass(
                        peptide, precursor_mz, precursor_charge
                    )
                    
                    results.append({
                        'peptide': peptide,
                        'score': score,
                        'mass_error_ppm': mass_check_result['mass_error_ppm'],
                        'source': 'Simple_BeamSearch',
                        'passes_mass_check': mass_check_result['passes']
                    })
        
        logger.info(f"✅ 生成了 {len(results)} 条候选peptide")
        
        # ===== 步骤3: 过滤和排序 =====
        logger.info(f"\n{'='*60}")
        logger.info("步骤3: 质量检查和排序")
        logger.info(f"{'='*60}")
        
        # 统计
        passed = [r for r in results if r['passes_mass_check']]
        failed = [r for r in results if not r['passes_mass_check']]
        
        logger.info(f"通过质量检查: {len(passed)}/{len(results)}")
        logger.info(f"未通过质量检查: {len(failed)}/{len(results)}")
        
        # 优先返回通过质量检查的，然后是未通过的
        # 按score排序
        passed_sorted = sorted(passed, key=lambda x: x['score'], reverse=True)
        failed_sorted = sorted(failed, key=lambda x: x['score'], reverse=True)
        
        # 合并：先通过的，再未通过的
        final_results = passed_sorted + failed_sorted
        
        # 限制到target_count
        final_results = final_results[:target_count]
        
        logger.info(f"\n✅ 最终返回 {len(final_results)} 条候选peptide")
        logger.info(f"   - 通过质量检查: {sum(1 for r in final_results if r['passes_mass_check'])}")
        logger.info(f"   - 未通过质量检查: {sum(1 for r in final_results if not r['passes_mass_check'])}")
        
        return final_results
    
    def _ctc_collapse(self, tokens: List[int]) -> List[int]:
        """
        CTC规约：去除重复的token
        例如: [1, 1, 2, 2, 3] -> [1, 2, 3]
        """
        if not tokens:
            return []
        
        collapsed = [tokens[0]]
        for t in tokens[1:]:
            if t != collapsed[-1]:
                collapsed.append(t)
        return collapsed
    
    def _calculate_peptide_mass(self, tokens: List[int]) -> float:
        """
        使用PiPrime的方式计算peptide质量
        
        Parameters:
        -----------
        tokens : List[int]
            Token序列
            
        Returns:
        --------
        float : 质量（Da），含水
        """
        # 将tokens转换为序列字符串
        sequence = ''.join([self.decoder._idx2aa.get(t, '') for t in tokens
                           if t != self.decoder.get_blank_idx() and t != self.decoder.get_pad_idx()])
        
        # 使用PiPrime的质量计算方法
        peptide_mass = calculate_peptide_mass_piprime(sequence, add_water=True)
        return peptide_mass
    
    def _simple_beam_search(self, prob_matrix: torch.Tensor, precursor_mz: float,
                           precursor_charge: int, beam_width: int = 100,
                           top_n: int = 10, max_length: int = 30) -> List[Dict]:
        """
        改进的Beam Search（借鉴multi_path_dp）
        - 输入：归一化且最大值缩放到20的正数概率矩阵
        - 每步只考虑Top-N个氨基酸
        - 不在搜索过程中进行质量过滤，只在最后过滤
        
        Parameters:
        -----------
        prob_matrix : torch.Tensor
            概率矩阵 [seq_len, vocab_size]，已归一化且缩放到20
        precursor_mz : float
            Precursor m/z
        precursor_charge : int
            Precursor电荷
        beam_width : int
            Beam宽度（生成的peptide数量）
        top_n : int
            每步考虑的Top-N氨基酸
        max_length : int
            最大序列长度
            
        Returns:
        --------
        List[Dict] : 候选列表，每个包含 {'peptide': str, 'score': float}
        """
        seq_len, vocab_size = prob_matrix.shape
        
        # 计算precursor mass和质量范围（用于最终过滤）
        precursor_mass = calculate_precursor_mass_from_mz(precursor_mz, precursor_charge)
        mass_tolerance_da = 0.5  # 放宽到±0.5 Da
        # 注意：precursor_mass是含水的，peptide_mass也是含水的，所以直接比较
        min_mass = precursor_mass - mass_tolerance_da
        max_mass = precursor_mass + mass_tolerance_da
        
        logger.info(f"  Precursor mass: {precursor_mass:.4f} Da")
        logger.info(f"  Mass range: [{min_mass:.4f}, {max_mass:.4f}] Da (±{mass_tolerance_da} Da)")
        logger.info(f"  Beam width: {beam_width}, Top-N: {top_n}")
        
        # 初始化：一个空路径
        paths = [(0.0, [])]  # (累积概率, token序列)
        
        # 逐步扩展
        for t in range(min(seq_len, max_length)):
            # 获取当前位置概率最高的Top-N个氨基酸
            top_probs, top_indices = torch.topk(prob_matrix[t], top_n)
            
            new_paths = []
            
            # 对每个现有路径
            for current_prob, current_path in paths:
                # 尝试添加每个Top-N氨基酸
                for prob, idx in zip(top_probs, top_indices):
                    idx = idx.item()
                    prob = prob.item()
                    
                    # 跳过blank token (0)
                    if idx == 0:
                        continue
                    
                    # 获取氨基酸
                    aa = self.decoder._idx2aa.get(idx, '')
                    if not aa:
                        continue
                    
                    # CTC去重：如果与前一个token相同，跳过
                    if current_path and idx == current_path[-1]:
                        continue
                    
                    # 创建新路径
                    new_path = current_path + [idx]
                    new_prob = current_prob + prob  # 直接累加概率（已经是正数）
                    
                    # 不进行质量过滤，直接添加
                    new_paths.append((new_prob, new_path))
            
            # 保留Top-K路径
            new_paths.sort(reverse=True)  # 按概率降序
            paths = new_paths[:beam_width]
            
            if not paths:
                logger.warning(f"  所有路径在步骤{t}被剪枝")
                break
            
            # 每5步输出状态
            if t % 5 == 0:
                logger.info(f"  Step {t}: created {len(new_paths)} new paths, kept {len(paths)} paths")
                if len(paths) > 0:
                    # 显示第一条路径的信息
                    first_path = paths[0]
                    collapsed = self._ctc_collapse(first_path[1])
                    mass = self._calculate_peptide_mass(collapsed)
                    logger.info(f"    Top path: prob={first_path[0]:.2f}, len={len(collapsed)}, mass={mass:.2f}")
        
        logger.info(f"  Beam search完成: {len(paths)} 条路径")
        
        # 最终质量过滤并转换为peptide
        candidates = []
        filtered_out = 0
        too_short = 0
        
        for prob, tokens in paths:
            # CTC规约
            collapsed = self._ctc_collapse(tokens)
            
            # 计算质量
            peptide_mass = self._calculate_peptide_mass(collapsed)
            
            # 严格的质量过滤
            if not (min_mass <= peptide_mass <= max_mass):
                filtered_out += 1
                continue
            
            # 转换为氨基酸序列
            aa_seq = [self.decoder._idx2aa.get(t, '') for t in collapsed]
            
            # 反转（如果需要）
            if self.decoder.reverse:
                aa_seq = list(reversed(aa_seq))
            
            peptide = "".join(aa_seq)
            
            # 过滤太短的序列
            if len(peptide) >= 5:
                candidates.append({
                    'peptide': peptide,
                    'score': prob,  # 使用累积概率作为分数
                    'mass': peptide_mass
                })
            else:
                too_short += 1
        
        logger.info(f"  质量过滤统计:")
        logger.info(f"    - 总路径: {len(paths)}")
        logger.info(f"    - 质量不匹配: {filtered_out}")
        logger.info(f"    - 序列太短: {too_short}")
        logger.info(f"    - 最终候选: {len(candidates)}")
        
        return candidates
    
    def _greedy_decode_topk_old(self, log_probs: torch.Tensor, precursor_mz: float,
                           precursor_charge: int, topk: int = 100) -> List[Dict]:
        """
        使用改进的Greedy Decoding生成Top-K候选peptide
        
        改进策略：
        1. 使用Beam Search变体，在每步保留多条路径
        2. **每步进行CTC规约并检查precursor mass**
        3. 剪枝超过precursor mass的路径
        4. 确保生成质量匹配的peptide序列
        
        Parameters:
        -----------
        log_probs : torch.Tensor
            Log概率矩阵 [seq_len, vocab_size]
        precursor_mz : float
            Precursor m/z
        precursor_charge : int
            Precursor电荷
        topk : int
            目标候选数量
            
        Returns:
        --------
        List[Dict] : 候选列表，每个包含 {'tokens': List[int], 'score': float}
        """
        seq_len, vocab_size = log_probs.shape
        probs = torch.exp(log_probs)
        
        # 计算precursor mass和质量上限
        precursor_mass = calculate_precursor_mass_from_mz(precursor_mz, precursor_charge)
        
        # 使用0.1 Da的严格质量控制
        mass_tolerance_da = 0.1
        min_mass = precursor_mass - mass_tolerance_da  # 下限
        max_mass = precursor_mass + mass_tolerance_da  # 上限
        
        # 计算预期的peptide长度（基于平均氨基酸质量约110 Da）
        expected_length = int(precursor_mass / 110)
        min_length_for_pruning = max(8, expected_length - 3)  # 至少8个AA，或预期长度-3
        
        logger.info(f"  Precursor mass: {precursor_mass:.4f} Da")
        logger.info(f"  Mass range: [{min_mass:.4f}, {max_mass:.4f}] Da (±{mass_tolerance_da} Da)")
        logger.info(f"  Expected peptide length: ~{expected_length} AA")
        logger.info(f"  Will start pruning after {min_length_for_pruning} AA")
        
        # 使用动态beam width策略
        # 第1步：10个beam，第2步：100个beam，第3步及之后：1000个beam
        beams = [{'tokens': [], 'score': 0.0, 'mass': 0.0}]
        
        pruned_count = 0  # 统计被剪枝的数量
        
        for t in range(seq_len):
            # 动态调整beam width
            if t == 0:
                beam_width = 10
            elif t == 1:
                beam_width = 100
            else:
                beam_width = 1000
            
            new_beams = []
            step_pruned = 0  # 本步被剪枝的数量
            step_processed = 0  # 本步处理的token数
            
            for beam in beams:
                # 获取当前时间步的Top-K概率
                top_k = min(10, vocab_size)  # 每步选择Top-10
                top_probs, top_indices = torch.topk(probs[t], top_k)
                
                for prob, idx in zip(top_probs, top_indices):
                    step_processed += 1
                    idx = idx.item()
                    
                    # 跳过特殊token（blank和EOS）
                    if idx == 0 or idx == 27:
                        continue
                    
                    # 获取氨基酸/修饰
                    aa = self.decoder._idx2aa.get(idx, '')
                    if not aa:
                        continue
                    
                    # ===== 支持修饰：不再跳过修饰token =====
                    # 修饰标记（如+42.011, -17.027）和带修饰的氨基酸（如M+15.995）都保留
                    
                    # CTC去重：如果与前一个token相同，跳过
                    if beam['tokens'] and idx == beam['tokens'][-1]:
                        continue
                    
                    # 创建新token序列
                    new_tokens = beam['tokens'] + [idx]
                    
                    # ===== 关键：CTC规约并计算质量（支持修饰）=====
                    collapsed_tokens = self._ctc_collapse(new_tokens)
                    current_mass = self._calculate_peptide_mass(collapsed_tokens)
                    
                    # 调试：在第一步输出质量计算详情
                    if t == 0 and len(new_beams) < 3:
                        aa_seq = [self.decoder._idx2aa.get(tok, '?') for tok in collapsed_tokens]
                        logger.info(f"    Debug t={t}: tokens={collapsed_tokens[:5]}..., aa={''.join(aa_seq)}, mass={current_mass:.4f}, range=[{min_mass:.4f}, {max_mass:.4f}]")
                    
                    # ===== 智能剪枝策略 =====
                    # 1. 统计氨基酸数量（不包括修饰标记）
                    aa_count = sum(1 for tok in collapsed_tokens
                                  if not (self.decoder._idx2aa.get(tok, '').startswith(('+', '-'))))
                    
                    # 2. 只在序列长度足够时才进行质量剪枝
                    if aa_count >= min_length_for_pruning:
                        # 严格的质量控制：必须在[min_mass, max_mass]范围内
                        if current_mass < min_mass or current_mass > max_mass:
                            step_pruned += 1
                            continue
                    else:
                        # 序列还太短，只剪枝明显过大的（超过2倍precursor mass）
                        if current_mass > precursor_mass * 2:
                            step_pruned += 1
                            continue
                    
                    # 创建新beam
                    new_beam = {
                        'tokens': new_tokens,
                        'score': beam['score'] + torch.log(prob).item(),
                        'mass': current_mass
                    }
                    new_beams.append(new_beam)
            
            pruned_count += step_pruned
            
            # 保留Top-K beams
            new_beams.sort(key=lambda x: x['score'], reverse=True)
            beams = new_beams[:beam_width]
            
            # 每10步输出一次状态
            if t % 10 == 0 and t > 0:
                logger.info(f"  Step {t}/{seq_len}: {len(beams)} beams, processed {step_processed}, pruned {step_pruned}")
            
            # 前几步输出详细信息
            if t <= 2:
                logger.info(f"  Step {t}: beam_width={beam_width}, processed {step_processed} tokens, created {len(new_beams)} beams, kept {len(beams)}, pruned {step_pruned}")
            
            # 如果所有beam都被剪枝了，提前结束
            if not beams:
                logger.warning(f"  All beams pruned at step {t}! Total pruned: {pruned_count}")
                break
        
        logger.info(f"  Beam search completed: {len(beams)} final beams, total pruned: {pruned_count}")
        
        # 过滤和去重（支持修饰）
        candidates = []
        seen_sequences = set()
        
        for beam in beams:
            tokens = beam['tokens']
            
            # 过滤条件：长度至少8个token（可能包含修饰token）
            if len(tokens) < 8:
                continue
            
            # 检查是否包含有效的氨基酸（至少要有一些氨基酸）
            aa_count = 0
            valid = True
            for t in tokens:
                aa = self.decoder._idx2aa.get(t, '')
                if not aa:
                    valid = False
                    break
                # 统计氨基酸数量（不包括纯修饰标记）
                if not (aa.startswith('+') or aa.startswith('-')):
                    aa_count += 1
            
            # 至少要有5个氨基酸
            if not valid or aa_count < 5:
                continue
            
            # 去重
            token_tuple = tuple(tokens)
            if token_tuple not in seen_sequences:
                seen_sequences.add(token_tuple)
                candidates.append({
                    'tokens': tokens,
                    'score': np.exp(beam['score'])
                })
        
        # 如果候选不够，降低长度要求
        if len(candidates) < topk:
            for beam in beams:
                tokens = beam['tokens']
                
                # 降低到至少5个token
                if len(tokens) < 5:
                    continue
                
                # 检查有效性
                aa_count = 0
                valid = True
                for t in tokens:
                    aa = self.decoder._idx2aa.get(t, '')
                    if not aa:
                        valid = False
                        break
                    if not (aa.startswith('+') or aa.startswith('-')):
                        aa_count += 1
                
                # 至少要有3个氨基酸
                if not valid or aa_count < 3:
                    continue
                
                token_tuple = tuple(tokens)
                if token_tuple not in seen_sequences:
                    seen_sequences.add(token_tuple)
                    candidates.append({
                        'tokens': tokens,
                        'score': np.exp(beam['score'])
                    })
                    
                    if len(candidates) >= topk:
                        break
        
        # 按分数排序
        candidates.sort(key=lambda x: x['score'], reverse=True)
        
        return candidates[:topk]
    
    def _check_precursor_mass(
        self,
        peptide: str,
        precursor_mz: float,
        precursor_charge: int
    ) -> Dict:
        """
        检查peptide的precursor mass是否匹配
        使用pyteomics计算质量，但正确处理PiPrime的修饰格式
        
        Returns:
        --------
        Dict : {'passes': bool, 'mass_error_ppm': float, 'best_isotope': int}
        """
        try:
            # 将PiPrime格式转换为pyteomics格式
            # PiPrime: C+57.021 -> pyteomics: C[+57.021]
            pyteomics_peptide = self._convert_to_pyteomics_format(peptide)
            
            # 使用pyteomics计算质量
            peptide_mass = pyteomics_mass.calculate_mass(
                sequence=pyteomics_peptide,
                charge=0  # 中性质量
            )
            
            # 计算理论m/z
            proton_mass = 1.007276
            calc_mz = (peptide_mass + precursor_charge * proton_mass) / precursor_charge
            
            # 检查所有同位素误差
            best_error = float('inf')
            best_isotope = 0
            
            for isotope in range(
                self.isotope_error_range[0],
                self.isotope_error_range[1] + 1
            ):
                # 考虑同位素误差
                corrected_mz = precursor_mz - isotope * 1.00335 / precursor_charge
                
                # 计算PPM误差
                mass_error_ppm = (calc_mz - corrected_mz) / corrected_mz * 1e6
                
                if abs(mass_error_ppm) < abs(best_error):
                    best_error = mass_error_ppm
                    best_isotope = isotope
            
            # 判断是否通过
            passes = abs(best_error) < self.precursor_mass_tol
            
            return {
                'passes': passes,
                'mass_error_ppm': best_error,
                'best_isotope': best_isotope
            }
            
        except Exception as e:
            logger.warning(f"质量检查失败 for {peptide}: {e}")
            return {
                'passes': False,
                'mass_error_ppm': float('inf'),
                'best_isotope': 0
            }
    
    def _convert_to_pyteomics_format(self, peptide: str) -> str:
        """
        将PiPrime的修饰格式转换为pyteomics格式
        
        支持的输入格式:
        - PiPrime格式: C+57.021, M+15.995, N+0.984
        - MGF括号格式: C(+57.02), M(+15.99), N(+.98)
        
        输出格式:
        - Pyteomics格式: C[+57.021], M[+15.995], N[+0.984]
        
        注意：PiPrime中C+57.021是整体token，质量已经包含了C和修饰
        但pyteomics需要分开处理，所以我们需要：
        1. 识别修饰的氨基酸（如C+57.021或C(+57.02)）
        2. 转换为pyteomics格式（C[+57.021]）
        3. pyteomics会自动处理：C的质量 + 修饰质量
        """
        import re
        from piprime_mass_calculator import normalize_sequence_format
        
        # 首先标准化格式：将MGF格式转换为PiPrime格式
        peptide = normalize_sequence_format(peptide)
        
        # 匹配修饰的氨基酸：字母后跟+或-和数字
        # 例如：C+57.021, M+15.995, N+0.984, Q+0.984
        pattern = r'([A-Z])([\+\-][\d\.]+)'
        
        def replace_mod(match):
            aa = match.group(1)
            mod = match.group(2)
            return f"{aa}[{mod}]"
        
        pyteomics_peptide = re.sub(pattern, replace_mod, peptide)
        
        # 替换I为L（pyteomics中它们质量相同）
        pyteomics_peptide = pyteomics_peptide.replace('I', 'L')
        
        return pyteomics_peptide
    
    def _clean_peptide_sequence(self, peptide: str) -> str:
        """清理peptide序列，去除修饰标记"""
        import re
        # 去除修饰标记 (如 M+15.995, N+0.984等)
        clean = re.sub(r'\+[\d\.]+', '', peptide)
        clean = re.sub(r'\-[\d\.]+', '', peptide)
        # 替换L为I (质量相同)
        clean = clean.replace('L', 'I')
        return clean


def test_piprime_with_mass_check():
    """测试函数"""
    import sys
    import os
    
    # 添加路径
    piprime_path = os.path.join(os.path.dirname(__file__), '..', 'pi-PrimeNovo')
    sys.path.insert(0, piprime_path)
    
    from PrimeNovo.denovo.model import Spec2Pep
    from pyteomics import mgf
    
    # 加载模型
    model_path = os.path.join(piprime_path, "model_massive.ckpt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logger.info(f"加载PiPrime模型: {model_path}")
    model = Spec2Pep.load_from_checkpoint(model_path, map_location=device)
    model.eval()
    model.to(device)
    
    # 创建生成器
    generator = PiPrimeWithMassCheck(
        model, 
        precursor_mass_tol=50, 
        isotope_error_range=(0, 1)
    )
    
    # 加载测试谱图
    mgf_file = os.path.join(piprime_path, "testdata", "high_nine_validation_1000_converted.mgf")
    
    with mgf.MGF(mgf_file) as reader:
        for idx, spec in enumerate(reader):
            if idx >= 1:  # 只测试第一个
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
            
            # 预处理谱图
            from piprime_reranker import process_peaks, load_piprime_config
            config = load_piprime_config()
            peaks = process_peaks(
                mz_array, intensity_array, 
                precursor_mz, precursor_charge, 
                config
            )
            
            # 生成候选
            logger.info(f"\n{'='*80}")
            logger.info(f"测试谱图 #{idx}")
            logger.info(f"{'='*80}")
            logger.info(f"Precursor m/z: {precursor_mz:.4f}")
            logger.info(f"Charge: {precursor_charge}")
            logger.info(f"Peaks: {len(mz_array)}")
            
            candidates = generator.generate_candidates_with_mass_check(
                peaks, 
                precursor_mz, 
                precursor_charge,
                target_count=100,
                max_candidates=500
            )
            
            # 显示结果
            logger.info(f"\n{'='*80}")
            logger.info("Top 10 候选peptide:")
            logger.info(f"{'='*80}")
            logger.info(f"{'Rank':<6}{'Peptide':<25}{'Score':<12}{'Mass Error':<15}{'Pass':<8}{'Source'}")
            logger.info(f"{'-'*80}")
            
            for i, cand in enumerate(candidates[:10], 1):
                pass_mark = "✅" if cand['passes_mass_check'] else "❌"
                logger.info(
                    f"{i:<6}{cand['peptide']:<25}{cand['score']:<12.6f}"
                    f"{cand['mass_error_ppm']:<15.2f}{pass_mark:<8}{cand['source']}"
                )
            
            # 统计
            logger.info(f"\n{'='*80}")
            logger.info("统计信息:")
            logger.info(f"{'='*80}")
            logger.info(f"总候选数: {len(candidates)}")
            logger.info(f"通过质量检查: {sum(1 for c in candidates if c['passes_mass_check'])}")
            logger.info(f"PMC结果: {sum(1 for c in candidates if c['source'] == 'PMC')}")
            logger.info(f"CTC Beam Search: {sum(1 for c in candidates if c['source'] == 'CTC_BeamSearch')}")
            
            break


if __name__ == "__main__":
    test_piprime_with_mass_check()