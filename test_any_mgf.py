#!/usr/bin/env python3
"""
通用MGF文件测试脚本
基于 piprime_highnine_reranker.py
支持测试任意MGF文件

使用方法:
python test_any_mgf.py <mgf_file_path> [--num_spectra N]

示例:
python test_any_mgf.py "C:\\Users\\research\\Desktop\\Alldata\\high9-massfilter\\high.mouse.PXD004948_mass_filtered.mgf"
python test_any_mgf.py "testdata/high_nine_validation_1000_converted.mgf" --num_spectra 100
"""

import sys
import os
import torch
import torch.nn.functional as F
import numpy as np
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from pyteomics import mgf
from tqdm import tqdm
from datetime import datetime

# 添加路径
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
    """标准化peptide序列（L->I）"""
    return peptide.replace('L', 'I')


class PiPrimeHighNineReranker:
    """
    PiPrime + HighNine 混合重排序系统
    
    结合了:
    - PiPrime的de novo测序（使用fastest beam search）
    - HighNine数据库的谱图库匹配
    - Prosit的理论谱图预测（仅在需要时使用）
    
    重排序策略优化：
    - 所有候选首先在数据库中查找（快速O(1)查找）
    - 只有未在数据库中找到的top-5候选使用Prosit预测
    - 这将Prosit调用从数百次减少到最多5次每谱图
    """
    
    def __init__(
        self, 
        piprime_model_path: str, 
        index_file: str, 
        output_dir: str, 
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
        """处理单个谱图"""
        # Step 1: 预处理谱图
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
        
        # Step 2: 生成spectrum embedding和log probability matrix
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
        
        # 转换为reranker格式
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
        
        # Step 4: 智能重排序策略
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
            # Substep 4.1: 查询数据库中的所有候选（快速O(1)查找）
            result_db, similarity_dict_db, source_dict_db = self.reranker.rerank_with_external_embedding(
                query_embedding=spectrum_embedding,
                candidates=candidates_passed,
                precursor_mz=precursor_mz,
                precursor_charge=precursor_charge,
                use_prosit=False,  # 第一轮只用数据库
                top_k=3
            )
            
            # Substep 4.2: 识别未在数据库中找到的候选
            not_found_candidates = [
                cand for cand in candidates_passed 
                if source_dict_db.get(cand['peptide'], 'NotFound') == 'NotFound'
            ]
            
            # Substep 4.3: 只对未找到的top-5候选使用Prosit
            if not_found_candidates:
                top5_not_found = sorted(not_found_candidates, key=lambda x: x['score'], reverse=True)[:5]
                
                result_prosit, similarity_dict_prosit, source_dict_prosit = self.reranker.rerank_with_external_embedding(
                    query_embedding=spectrum_embedding,
                    candidates=top5_not_found,
                    precursor_mz=precursor_mz,
                    precursor_charge=precursor_charge,
                    use_prosit=True,  # 对这些候选使用Prosit
                    top_k=3
                )
                
                # 合并结果
                similarity_dict = {**similarity_dict_db, **similarity_dict_prosit}
                source_dict = {**source_dict_db, **source_dict_prosit}
            else:
                # 所有候选都在数据库中找到
                similarity_dict = similarity_dict_db
                source_dict = source_dict_db
            
            # Substep 4.4: 按相似度重新排序所有候选
            all_results = []
            for cand in candidates_passed:
                peptide = cand['peptide']
                all_results.append({
                    'peptide': peptide,
                    'similarity': similarity_dict.get(peptide, -1.0),
                    'denovo_score': cand['score'],
                    'source': source_dict.get(peptide, 'Unknown')
                })
            
            # 按相似度排序并选择top-1
            all_results.sort(key=lambda x: x['similarity'], reverse=True)
            result = all_results[0] if all_results else result_db
        
        # 为所有候选添加similarity和source
        for cand in candidates_all:
            if cand['passes_mass_check'] and cand['peptide'] in similarity_dict:
                cand['similarity'] = similarity_dict[cand['peptide']]
                cand['source'] = source_dict.get(cand['peptide'], 'Unknown')
            else:
                cand['similarity'] = -1.0
                cand['source'] = 'NotChecked'
        
        # Step 5: 评估准确率
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
        
        return result


def main():
    """主函数 - 处理命令行参数并运行测试"""
    parser = argparse.ArgumentParser(
        description='测试任意MGF文件使用PiPrime + HighNine Reranker',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python test_any_mgf.py "C:\\path\\to\\file.mgf"
  python test_any_mgf.py "testdata/test.mgf" --num_spectra 100
  python test_any_mgf.py "data.mgf" --model custom_model.ckpt --index custom_index.pkl
        """
    )
    
    parser.add_argument('mgf_file', type=str, help='MGF文件路径')
    parser.add_argument('--num_spectra', type=int, default=None, 
                       help='要处理的谱图数量（默认：全部）')
    parser.add_argument('--model', type=str, default='model_massive.ckpt',
                       help='PiPrime模型路径（默认：model_massive.ckpt）')
    parser.add_argument('--index', type=str, 
                       default=r'D:\reference_dataset\reference_dataset.mgf.efficient_index.pkl',
                       help='HighNine索引文件路径')
    parser.add_argument('--output_interval', type=int, default=500,
                       help='每N个谱图输出一次统计（默认：500）')
    
    args = parser.parse_args()
    
    # 配置
    mgf_file = args.mgf_file
    piprime_model = args.model
    index_file = args.index
    num_spectra = args.num_spectra
    output_interval = args.output_interval
    
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mgf_basename = Path(mgf_file).stem
    output_dir = Path(f"results_{mgf_basename}_{timestamp}")
    output_dir.mkdir(exist_ok=True)
    
    # 验证文件
    if not os.path.exists(mgf_file):
        logger.error(f"❌ MGF file not found: {mgf_file}")
        return
    
    if not os.path.exists(piprime_model):
        logger.error(f"❌ PiPrime model not found: {piprime_model}")
        return
    
    if not os.path.exists(index_file):
        logger.error(f"❌ Index file not found: {index_file}")
        logger.info("Please run build_efficient_index.py first to create the index")
        return
    
    # 打印配置
    logger.info(f"\n{'='*80}")
    logger.info("PiPrime + HighNine Reranker - Universal MGF Tester")
    logger.info(f"{'='*80}")
    logger.info(f"MGF file: {mgf_file}")
    logger.info(f"PiPrime model: {piprime_model}")
    logger.info(f"Index file: {index_file}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Number of spectra: {num_spectra if num_spectra else 'All'}")
    logger.info(f"Output interval: {output_interval}")
    logger.info(f"{'='*80}\n")
    
    # 初始化reranker
    reranker = PiPrimeHighNineReranker(piprime_model, index_file, output_dir)
    
    # 统计变量
    total = 0
    correct = 0
    total_candidates = 0
    total_beam_time = 0.0
    source_stats = {'Database': 0, 'Prosit': 0, 'Failed': 0}
    predicted_peptides = []  # Batch writing buffer
    
    # 创建结果文件
    results_file = output_dir / "predicted_peptides.txt"
    results_handle = open(results_file, 'w', encoding='utf-8')
    logger.info(f"Results will be saved to: {results_file}\n")
    
    # 处理谱图
    logger.info("Processing spectra...\n")
    
    with mgf.MGF(mgf_file) as reader:
        if num_spectra:
            spectra_list = list(reader)[:num_spectra]
        else:
            spectra_list = list(reader)
    
    total_spectra = len(spectra_list)
    logger.info(f"Total spectra to process: {total_spectra}\n")
    
    for idx, spec in enumerate(tqdm(spectra_list, desc="Processing", unit="spectrum")):
        try:
            # 提取谱图信息
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
            
            total_candidates += result.get('num_candidates', 0)
            total_beam_time += result.get('timing_stats', {}).get('total_time', 0)
            
            source = result.get('source', 'Unknown')
            if source in source_stats:
                source_stats[source] += 1
            
            # 收集结果到buffer（批量写入）
            predicted_peptide = result.get('peptide', '')
            predicted_peptides.append(predicted_peptide)
            
            # 每100个谱图批量写入
            if total % 100 == 0:
                for pep in predicted_peptides:
                    results_handle.write(f"{pep}\n")
                results_handle.flush()
                predicted_peptides = []
            
            # 定期输出统计
            if total % output_interval == 0:
                logger.info(f"\n{'='*80}")
                logger.info(f"Progress: {total}/{total_spectra}")
                logger.info(f"{'='*80}")
                logger.info(f"Correct: {correct}/{total}")
                logger.info(f"Accuracy: {correct/total*100:.2f}%")
                logger.info(f"Average candidates: {total_candidates/total:.1f}")
                logger.info(f"Average beam search time: {total_beam_time/total*1000:.2f} ms")
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
    
    # 写入剩余结果
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
    logger.info(f"Average candidates: {total_candidates/total:.1f}")
    logger.info(f"Average beam search time: {total_beam_time/total*1000:.2f} ms")
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