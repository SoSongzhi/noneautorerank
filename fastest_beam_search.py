#!/usr/bin/env python3
"""
最快的Beam Search实现

核心思想：
1. 预计算所有top-k（一次性完成）
2. 只存储分数，不存储路径（避免列表操作）
3. 用索引记录路径，最后重建（避免遍历）
4. 所有操作都是tensor操作（GPU并行）

预期效果：比原始方法快100-1000×
"""

import torch
import numpy as np
from typing import List, Tuple
import time


def fastest_beam_search(
    predictor,
    log_prob_matrix: torch.Tensor,
    precursor_mz: float,
    precursor_charge: int,
    target_count: int = 1000,
    top_n: int = 10
) -> Tuple[List[Tuple[str, float, float, bool]], dict]:
    """
    最快的Beam Search实现
    
    Returns:
    --------
    results : List[Tuple[str, float, float, bool]]
        候选peptide列表
    timing_stats : dict
        时间统计
    """
    from piprime_mass_calculator import calculate_precursor_mass_from_mz
    from test_beam_search_accuracy import calculate_peptide_mass_from_tokens
    
    timing_stats = {}
    total_start = time.time()
    
    seq_len, vocab_size = log_prob_matrix.shape
    device = log_prob_matrix.device
    
    # 计算质量范围
    precursor_mass = calculate_precursor_mass_from_mz(precursor_mz, precursor_charge)
    mass_tolerance_da = 0.1
    min_mass = precursor_mass - mass_tolerance_da
    max_mass = precursor_mass + mass_tolerance_da
    
    # ===== 阶段1: 预计算所有top-k（一次性完成）=====
    topk_start = time.time()
    
    # 一次性计算所有时间步的top-k
    all_topk_probs, all_topk_indices = torch.topk(
        log_prob_matrix, 
        k=min(top_n, vocab_size), 
        dim=-1
    )
    # all_topk_probs: [seq_len, top_n]
    # all_topk_indices: [seq_len, top_n]
    
    topk_time = time.time() - topk_start
    timing_stats['topk_time'] = topk_time
    
    # ===== 阶段2: Beam Search（只操作分数和索引）=====
    beam_start = time.time()
    
    # 动态beam width
    def get_beam_width(t):
        if t == 0: return 10
        elif t == 1: return 100
        elif t == 2: return 1000
        else: return target_count
    
    # 初始化
    # beam_scores: [beam_width] - 当前beam的分数
    # beam_paths: [beam_width, t] - 当前beam的路径索引
    beam_scores = torch.zeros(1, device=device)
    beam_paths = torch.zeros(1, 0, dtype=torch.long, device=device)
    
    for t in range(seq_len):
        current_beam_width = get_beam_width(t)
        num_beams = beam_scores.shape[0]
        
        # 获取当前时间步的top-k
        topk_probs = all_topk_probs[t]  # [top_n]
        topk_indices = all_topk_indices[t]  # [top_n]
        
        # ===== 关键优化：广播计算所有新分数 =====
        # [num_beams, 1] + [1, top_n] = [num_beams, top_n]
        new_scores = beam_scores.unsqueeze(1) + topk_probs.unsqueeze(0)
        
        # 展平以便选择top-k
        new_scores_flat = new_scores.flatten()  # [num_beams * top_n]
        
        # ===== 关键优化：扩展路径索引（不复制实际路径）=====
        if beam_paths.shape[1] == 0:
            # 第一步：直接创建新路径
            # [top_n, 1]
            new_paths_flat = topk_indices.unsqueeze(1)
        else:
            # 后续步骤：扩展现有路径
            # [num_beams, t] -> [num_beams, 1, t] -> [num_beams, top_n, t]
            expanded_paths = beam_paths.unsqueeze(1).expand(-1, top_n, -1)
            # [num_beams, top_n, t] -> [num_beams * top_n, t]
            expanded_paths_flat = expanded_paths.reshape(-1, beam_paths.shape[1])
            
            # 添加新token索引
            # [top_n] -> [1, top_n] -> [num_beams, top_n] -> [num_beams * top_n, 1]
            new_tokens = topk_indices.unsqueeze(0).expand(num_beams, -1).flatten().unsqueeze(1)
            # [num_beams * top_n, t+1]
            new_paths_flat = torch.cat([expanded_paths_flat, new_tokens], dim=1)
        
        # 选择top-k
        if new_scores_flat.shape[0] > current_beam_width:
            top_scores, top_indices = torch.topk(new_scores_flat, k=current_beam_width)
            beam_scores = top_scores
            beam_paths = new_paths_flat[top_indices]
        else:
            beam_scores = new_scores_flat
            beam_paths = new_paths_flat
        
        if beam_paths.shape[0] == 0:
            break
    
    beam_time = time.time() - beam_start
    timing_stats['beam_time'] = beam_time
    
    # ===== 阶段3: 批量CTC Collapse =====
    collapse_start = time.time()
    
    collapsed_paths = []
    for i in range(min(target_count, beam_paths.shape[0])):
        tokens = beam_paths[i].cpu().tolist()
        score = beam_scores[i].item()
        
        # CTC collapse
        collapsed = predictor._ctc_collapse(tokens)
        collapsed_no_blank = [t for t in collapsed if t != 27]
        
        collapsed_paths.append((score, collapsed_no_blank))
    
    collapse_time = time.time() - collapse_start
    timing_stats['collapse_time'] = collapse_time
    
    # ===== 阶段4: 批量质量计算 =====
    mass_start = time.time()
    
    paths_with_mass = []
    for score, tokens in collapsed_paths:
        peptide_mass = calculate_peptide_mass_from_tokens(predictor, tokens)
        if peptide_mass is not None:
            paths_with_mass.append((score, tokens, peptide_mass))
    
    mass_time = time.time() - mass_start
    timing_stats['mass_time'] = mass_time
    
    # ===== 阶段5: 质量过滤和转换 =====
    filter_start = time.time()
    
    # 按质量过滤
    passed_paths = [(s, t, m) for s, t, m in paths_with_mass if min_mass <= m <= max_mass]
    failed_paths = [(s, t, m) for s, t, m in paths_with_mass if not (min_mass <= m <= max_mass)]
    
    # 合并
    all_paths = passed_paths + failed_paths
    
    # 转换为peptide
    results = []
    for score, tokens, peptide_mass in all_paths[:target_count]:
        aa_seq = [predictor.decoder._idx2aa.get(t, '?') for t in tokens]
        aa_seq = [aa for aa in aa_seq if aa and aa != '_' and aa != '?']
        if predictor.decoder.reverse:
            aa_seq = list(reversed(aa_seq))
        peptide = "".join(aa_seq)
        
        passes = min_mass <= peptide_mass <= max_mass
        results.append((peptide, score, peptide_mass, passes))
    
    filter_time = time.time() - filter_start
    timing_stats['filter_time'] = filter_time
    
    total_time = time.time() - total_start
    timing_stats['total_time'] = total_time
    
    return results, timing_stats


def compare_all_methods(predictor, spectrum_data, num_runs=3):
    """对比所有方法"""
    from piprime_reranker import process_peaks, load_piprime_config
    from piprime_mass_calculator import calculate_precursor_mass_from_mz
    from test_beam_search_accuracy import beam_search_with_mass_pruning
    from lazy_beam_search import lazy_beam_search
    import torch.nn.functional as F
    
    mz_array = spectrum_data['mz_array']
    int_array = spectrum_data['int_array']
    precursor_mz = spectrum_data['precursor_mz']
    precursor_charge = spectrum_data['precursor_charge']
    
    print(f"\n{'='*80}")
    print(f"三种方法对比 - Precursor m/z: {precursor_mz:.4f}, Charge: {precursor_charge}")
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
    
    results_dict = {}
    
    # 方法1: 原始
    print(f"\n方法1: 原始Beam Search")
    print("="*80)
    times = []
    for run in range(num_runs):
        start = time.time()
        results = beam_search_with_mass_pruning(
            predictor, log_prob_matrix, precursor_mz, precursor_charge, 1000, 10
        )
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"运行{run+1}: {elapsed*1000:.2f} ms")
    results_dict['original'] = {'times': times, 'results': results}
    
    # 方法2: 延迟计算
    print(f"\n方法2: 延迟计算Beam Search")
    print("="*80)
    times = []
    for run in range(num_runs):
        results, stats = lazy_beam_search(
            predictor, log_prob_matrix, precursor_mz, precursor_charge, 1000, 10
        )
        times.append(stats['total_time'])
    results_dict['lazy'] = {'times': times, 'results': results}
    
    # 方法3: 最快
    print(f"\n方法3: 最快Beam Search")
    print("="*80)
    times = []
    for run in range(num_runs):
        print(f"\n运行{run+1}:")
        results, stats = fastest_beam_search(
            predictor, log_prob_matrix, precursor_mz, precursor_charge, 1000, 10
        )
        times.append(stats['total_time'])
    results_dict['fastest'] = {'times': times, 'results': results}
    
    # 汇总
    print(f"\n{'='*80}")
    print("最终对比")
    print(f"{'='*80}\n")
    
    for name, data in results_dict.items():
        avg_time = np.mean(data['times'])
        std_time = np.std(data['times'])
        print(f"{name:12s}: {avg_time*1000:8.2f} ms ± {std_time*1000:6.2f} ms")
    
    baseline = np.mean(results_dict['original']['times'])
    print(f"\n加速比:")
    for name, data in results_dict.items():
        if name != 'original':
            speedup = baseline / np.mean(data['times'])
            print(f"  {name:12s}: {speedup:6.2f}×")
    
    # 验证结果一致性
    print(f"\n{'='*80}")
    print("结果一致性验证")
    print(f"{'='*80}\n")
    
    original_results = results_dict['original']['results']
    
    for name, data in results_dict.items():
        if name == 'original':
            continue
        
        test_results = data['results']
        
        print(f"\n{name} vs original:")
        print(f"  候选数量: {len(test_results)} vs {len(original_results)}")
        
        # 对比Top-10 peptides
        original_top10 = set(p for p, _, _, _ in original_results[:10])
        test_top10 = set(p for p, _, _, _ in test_results[:10])
        
        if original_top10 == test_top10:
            print(f"  ✅ Top-10 peptides完全一致")
        else:
            print(f"  ⚠️ Top-10 peptides有差异")
            only_original = original_top10 - test_top10
            only_test = test_top10 - original_top10
            if only_original:
                print(f"    原始方法独有: {only_original}")
            if only_test:
                print(f"    {name}独有: {only_test}")
        
        # 对比通过质量检查的数量
        original_passed = sum(1 for _, _, _, passes in original_results if passes)
        test_passed = sum(1 for _, _, _, passes in test_results if passes)
        
        print(f"  通过质量检查: {test_passed} vs {original_passed}")
        
        if test_passed == original_passed:
            print(f"  ✅ 通过质量检查的数量一致")
        else:
            diff = test_passed - original_passed
            print(f"  ⚠️ 差异: {diff:+d}")
        
        # 对比Top-100中通过质量检查的peptides
        original_top100_passed = set(p for p, _, _, passes in original_results[:100] if passes)
        test_top100_passed = set(p for p, _, _, passes in test_results[:100] if passes)
        
        overlap = len(original_top100_passed & test_top100_passed)
        total = len(original_top100_passed | test_top100_passed)
        
        if total > 0:
            similarity = overlap / total * 100
            print(f"  Top-100通过质量检查的peptides相似度: {similarity:.1f}% ({overlap}/{total})")


def main():
    import argparse
    from PrimeNovo.denovo.model import Spec2Pep
    from piprime_with_mass_check import PiPrimeWithMassCheck
    from pyteomics import mgf
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--mgf', type=str, required=True)
    parser.add_argument('--num-spectra', type=int, default=3)
    parser.add_argument('--num-runs', type=int, default=3)
    args = parser.parse_args()
    
    # 加载模型
    model_path = "model_massive.ckpt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Spec2Pep.load_from_checkpoint(model_path, map_location=device)
    model.eval()
    model.to(device)
    
    predictor = PiPrimeWithMassCheck(model, 50, (0, 1), 100)
    
    # 加载谱图
    spectra = list(mgf.read(args.mgf))[:args.num_spectra]
    
    for idx, spectrum in enumerate(spectra):
        spectrum_data = {
            'mz_array': spectrum['m/z array'],
            'int_array': spectrum['intensity array'],
            'precursor_mz': spectrum['params']['pepmass'][0],
            'precursor_charge': int(spectrum['params'].get('charge', [2])[0])
        }
        
        compare_all_methods(predictor, spectrum_data, args.num_runs)


if __name__ == '__main__':
    main()