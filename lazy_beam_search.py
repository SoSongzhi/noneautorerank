#!/usr/bin/env python3
"""
延迟计算Beam Search - 只在最后计算质量和CTC collapse

策略：
1. Beam search过程中不进行CTC collapse
2. Beam search过程中不计算质量
3. 只在最后对所有候选进行CTC collapse和质量过滤

预期效果：大幅减少计算次数
"""

import torch
import numpy as np
from typing import List, Tuple
import time


def lazy_beam_search(
    predictor,
    log_prob_matrix: torch.Tensor,
    precursor_mz: float,
    precursor_charge: int,
    target_count: int = 1000,
    top_n: int = 10
) -> Tuple[List[Tuple[str, float, float, bool]], dict]:
    """
    延迟计算的Beam Search
    
    Returns:
    --------
    results : List[Tuple[str, float, float, bool]]
        候选peptide列表
    timing_stats : dict
        时间统计
    """
    from piprime_mass_calculator import calculate_precursor_mass_from_mz
    from test_beam_search_accuracy import calculate_peptide_mass_from_tokens
    
    timing_stats = {
        'beam_search_time': 0,
        'collapse_time': 0,
        'mass_calc_time': 0,
        'filter_time': 0,
        'total_time': 0,
        'num_paths_before_filter': 0,
        'num_paths_after_filter': 0
    }
    
    total_start = time.time()
    
    seq_len, vocab_size = log_prob_matrix.shape
    
    # 计算质量范围
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
    
    # ===== 阶段1: Beam Search（不计算质量，不CTC collapse）=====
    beam_start = time.time()
    
    paths = [(0.0, [])]  # (score, token_sequence)
    
    for t in range(seq_len):
        current_beam_width = get_beam_width(t)
        top_log_probs, top_indices = torch.topk(log_prob_matrix[t], top_n)
        
        new_paths = []
        
        # 简单的路径扩展（不做任何额外计算）
        for current_log_prob, current_path in paths:
            for log_prob, idx in zip(top_log_probs, top_indices):
                idx = idx.item()
                log_prob = log_prob.item()
                
                new_path = current_path + [idx]
                new_log_prob = current_log_prob + log_prob
                
                # 直接添加，不做任何检查
                new_paths.append((new_log_prob, new_path))
        
        # 排序并保留top-k
        new_paths.sort(reverse=True)
        paths = new_paths[:current_beam_width]
        
        if not paths:
            break
    
    beam_time = time.time() - beam_start
    timing_stats['beam_search_time'] = beam_time
    timing_stats['num_paths_before_filter'] = len(paths)
    
    print(f"\n阶段1: Beam Search完成")
    print(f"  时间: {beam_time*1000:.2f} ms")
    print(f"  生成路径数: {len(paths)}")
    
    # ===== 阶段2: 批量CTC Collapse =====
    collapse_start = time.time()
    
    collapsed_paths = []
    for score, tokens in paths:
        collapsed = predictor._ctc_collapse(tokens)
        collapsed_no_blank = [t for t in collapsed if t != 27]
        collapsed_paths.append((score, collapsed_no_blank))
    
    collapse_time = time.time() - collapse_start
    timing_stats['collapse_time'] = collapse_time
    
    print(f"\n阶段2: CTC Collapse完成")
    print(f"  时间: {collapse_time*1000:.2f} ms")
    print(f"  处理路径数: {len(collapsed_paths)}")
    
    # ===== 阶段3: 批量质量计算和过滤 =====
    mass_start = time.time()
    
    paths_with_mass = []
    for score, tokens in collapsed_paths:
        peptide_mass = calculate_peptide_mass_from_tokens(predictor, tokens)
        if peptide_mass is not None:
            paths_with_mass.append((score, tokens, peptide_mass))
    
    mass_time = time.time() - mass_start
    timing_stats['mass_calc_time'] = mass_time
    
    print(f"\n阶段3: 质量计算完成")
    print(f"  时间: {mass_time*1000:.2f} ms")
    print(f"  有效路径数: {len(paths_with_mass)}")
    
    # ===== 阶段4: 质量过滤 =====
    filter_start = time.time()
    
    # 按质量过滤
    passed_paths = [(s, t, m) for s, t, m in paths_with_mass if min_mass <= m <= max_mass]
    failed_paths = [(s, t, m) for s, t, m in paths_with_mass if not (min_mass <= m <= max_mass)]
    
    # 合并：先通过的，再未通过的
    all_paths = passed_paths + failed_paths
    
    filter_time = time.time() - filter_start
    timing_stats['filter_time'] = filter_time
    timing_stats['num_paths_after_filter'] = len(all_paths)
    
    print(f"\n阶段4: 质量过滤完成")
    print(f"  时间: {filter_time*1000:.2f} ms")
    print(f"  通过质量检查: {len(passed_paths)}")
    print(f"  未通过质量检查: {len(failed_paths)}")
    
    # ===== 阶段5: 转换为peptide =====
    results = []
    for score, tokens, peptide_mass in all_paths[:target_count]:
        # 转换为peptide
        aa_seq = [predictor.decoder._idx2aa.get(t, '?') for t in tokens]
        aa_seq = [aa for aa in aa_seq if aa and aa != '_' and aa != '?']
        if predictor.decoder.reverse:
            aa_seq = list(reversed(aa_seq))
        peptide = "".join(aa_seq)
        
        # 质量检查
        passes = min_mass <= peptide_mass <= max_mass
        
        results.append((peptide, score, peptide_mass, passes))
    
    total_time = time.time() - total_start
    timing_stats['total_time'] = total_time
    
    print(f"\n总时间: {total_time*1000:.2f} ms")
    print(f"  Beam Search: {beam_time*1000:.2f} ms ({beam_time/total_time*100:.1f}%)")
    print(f"  CTC Collapse: {collapse_time*1000:.2f} ms ({collapse_time/total_time*100:.1f}%)")
    print(f"  质量计算: {mass_time*1000:.2f} ms ({mass_time/total_time*100:.1f}%)")
    print(f"  质量过滤: {filter_time*1000:.2f} ms ({filter_time/total_time*100:.1f}%)")
    
    return results, timing_stats


def compare_beam_search_methods(
    predictor,
    spectrum_data,
    num_runs: int = 3
):
    """
    对比原始方法和延迟计算方法
    """
    from piprime_reranker import process_peaks, load_piprime_config
    from piprime_mass_calculator import calculate_precursor_mass_from_mz
    from test_beam_search_accuracy import beam_search_with_mass_pruning
    import torch.nn.functional as F
    
    mz_array = spectrum_data['mz_array']
    int_array = spectrum_data['int_array']
    precursor_mz = spectrum_data['precursor_mz']
    precursor_charge = spectrum_data['precursor_charge']
    
    print(f"\n{'='*80}")
    print(f"对比测试 - Precursor m/z: {precursor_mz:.4f}, Charge: {precursor_charge}")
    print(f"{'='*80}")
    
    # 预处理
    config = load_piprime_config()
    peaks = process_peaks(mz_array, int_array, precursor_mz, precursor_charge, config)
    peaks = peaks.to(predictor.device)
    
    precursor_mass = calculate_precursor_mass_from_mz(precursor_mz, precursor_charge)
    precursors = torch.tensor(
        [[precursor_mass, precursor_charge, precursor_mz]],
        dtype=torch.float32,
        device=predictor.device
    )
    
    # 获取log_prob_matrix
    with torch.no_grad():
        enc_out, enc_mask = predictor.model.encoder(peaks.unsqueeze(0))
        output_logits, _, _ = predictor.model.decoder(None, precursors, enc_out, enc_mask)
        log_prob_matrix = F.log_softmax(output_logits[0], dim=-1)
    
    # ===== 测试原始方法 =====
    print(f"\n{'='*80}")
    print("方法1: 原始Beam Search（每步计算质量）")
    print(f"{'='*80}")
    
    original_times = []
    for run in range(num_runs):
        print(f"\n运行 {run+1}/{num_runs}...")
        start = time.time()
        original_results = beam_search_with_mass_pruning(
            predictor,
            log_prob_matrix,
            precursor_mz,
            precursor_charge,
            target_count=1000,
            top_n=10
        )
        elapsed = time.time() - start
        original_times.append(elapsed)
        print(f"  时间: {elapsed*1000:.2f} ms")
        print(f"  候选数: {len(original_results)}")
    
    avg_original = np.mean(original_times)
    std_original = np.std(original_times)
    
    # ===== 测试延迟计算方法 =====
    print(f"\n{'='*80}")
    print("方法2: 延迟计算Beam Search（最后计算质量）")
    print(f"{'='*80}")
    
    lazy_times = []
    lazy_stats_list = []
    for run in range(num_runs):
        print(f"\n运行 {run+1}/{num_runs}...")
        lazy_results, lazy_stats = lazy_beam_search(
            predictor,
            log_prob_matrix,
            precursor_mz,
            precursor_charge,
            target_count=1000,
            top_n=10
        )
        lazy_times.append(lazy_stats['total_time'])
        lazy_stats_list.append(lazy_stats)
        print(f"  候选数: {len(lazy_results)}")
    
    avg_lazy = np.mean(lazy_times)
    std_lazy = np.std(lazy_times)
    
    # ===== 汇总对比 =====
    print(f"\n{'='*80}")
    print("对比结果")
    print(f"{'='*80}\n")
    
    print(f"原始方法:")
    print(f"  平均时间: {avg_original*1000:.2f} ms ± {std_original*1000:.2f} ms")
    print(f"  候选数: {len(original_results)}")
    
    print(f"\n延迟计算方法:")
    print(f"  平均时间: {avg_lazy*1000:.2f} ms ± {std_lazy*1000:.2f} ms")
    print(f"  候选数: {len(lazy_results)}")
    
    # 详细时间分布
    avg_stats = {
        'beam_search_time': np.mean([s['beam_search_time'] for s in lazy_stats_list]),
        'collapse_time': np.mean([s['collapse_time'] for s in lazy_stats_list]),
        'mass_calc_time': np.mean([s['mass_calc_time'] for s in lazy_stats_list]),
        'filter_time': np.mean([s['filter_time'] for s in lazy_stats_list])
    }
    
    print(f"\n延迟计算方法时间分布:")
    print(f"  Beam Search: {avg_stats['beam_search_time']*1000:.2f} ms ({avg_stats['beam_search_time']/avg_lazy*100:.1f}%)")
    print(f"  CTC Collapse: {avg_stats['collapse_time']*1000:.2f} ms ({avg_stats['collapse_time']/avg_lazy*100:.1f}%)")
    print(f"  质量计算: {avg_stats['mass_calc_time']*1000:.2f} ms ({avg_stats['mass_calc_time']/avg_lazy*100:.1f}%)")
    print(f"  质量过滤: {avg_stats['filter_time']*1000:.2f} ms ({avg_stats['filter_time']/avg_lazy*100:.1f}%)")
    
    speedup = avg_original / avg_lazy
    print(f"\n加速比: {speedup:.2f}×")
    
    if speedup > 1:
        print(f"✅ 延迟计算方法快 {(speedup-1)*100:.1f}%")
    else:
        print(f"⚠️ 延迟计算方法慢 {(1-speedup)*100:.1f}%")
    
    # 验证结果一致性
    print(f"\n验证结果一致性:")
    original_peptides = set(p for p, _, _, _ in original_results[:10])
    lazy_peptides = set(p for p, _, _, _ in lazy_results[:10])
    
    if original_peptides == lazy_peptides:
        print(f"✅ Top-10 peptides完全一致")
    else:
        print(f"⚠️ Top-10 peptides有差异")
        print(f"  原始方法独有: {original_peptides - lazy_peptides}")
        print(f"  延迟方法独有: {lazy_peptides - original_peptides}")


def main():
    import argparse
    from PrimeNovo.denovo.model import Spec2Pep
    from piprime_with_mass_check import PiPrimeWithMassCheck
    from pyteomics import mgf
    
    parser = argparse.ArgumentParser(description='Compare beam search methods')
    parser.add_argument('--mgf', type=str, required=True, help='MGF file')
    parser.add_argument('--num-spectra', type=int, default=5, help='Number of spectra to test')
    parser.add_argument('--num-runs', type=int, default=3, help='Number of runs per spectrum')
    args = parser.parse_args()
    
    print("="*80)
    print("Beam Search方法对比测试")
    print("="*80)
    
    # 加载模型
    print("\n加载模型...")
    model_path = "model_massive.ckpt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = Spec2Pep.load_from_checkpoint(model_path, map_location=device)
    model.eval()
    model.to(device)
    
    predictor = PiPrimeWithMassCheck(
        model,
        precursor_mass_tol=50,
        isotope_error_range=(0, 1),
        beam_width=100
    )
    
    print(f"模型已加载到 {device}")
    
    # 加载谱图
    print(f"\n加载谱图: {args.mgf}")
    spectra = list(mgf.read(args.mgf))[:args.num_spectra]
    print(f"将测试 {len(spectra)} 个谱图\n")
    
    # 测试每个谱图
    all_speedups = []
    for idx, spectrum in enumerate(spectra):
        spectrum_data = {
            'mz_array': spectrum['m/z array'],
            'int_array': spectrum['intensity array'],
            'precursor_mz': spectrum['params']['pepmass'][0],
            'precursor_charge': int(spectrum['params'].get('charge', [2])[0])
        }
        
        compare_beam_search_methods(predictor, spectrum_data, args.num_runs)
        
        if idx < len(spectra) - 1:
            print(f"\n{'='*80}\n")


if __name__ == '__main__':
    main()