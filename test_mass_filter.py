#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试MGF文件中的谱图是否满足precursor mass条件
使用与diagnose_real_spectra.py相同的正确方法
"""
import os
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from pyteomics import mgf
from tqdm import tqdm
from piprime_mass_calculator import (
    calculate_precursor_mass_from_mz,
    calculate_peptide_mass_piprime,
    check_mass_match,
    mass_cal_piprime,
    H2O_MASS
)

def test_mgf_file(mgf_path, tolerance=0.1, output_dir='mass_filter_results'):
    """
    测试MGF文件中的谱图是否满足precursor mass条件
    将正确和不正确的例子保存到txt文件
    """
    print(f"\n{'='*80}")
    print(f"测试文件: {os.path.basename(mgf_path)}")
    print(f"{'='*80}")
    
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 输出文件
    base_name = os.path.splitext(os.path.basename(mgf_path))[0]
    passed_file = os.path.join(output_dir, f'{base_name}_PASSED.txt')
    failed_file = os.path.join(output_dir, f'{base_name}_FAILED.txt')
    summary_file = os.path.join(output_dir, f'{base_name}_SUMMARY.txt')
    
    try:
        # 读取MGF文件
        with mgf.read(mgf_path) as reader:
            total_spectra = sum(1 for _ in reader)
        
        print(f"总谱图数: {total_spectra}")
        
        # 打开输出文件
        with open(passed_file, 'w', encoding='utf-8') as f_pass, \
             open(failed_file, 'w', encoding='utf-8') as f_fail, \
             open(summary_file, 'w', encoding='utf-8') as f_summary:
            
            # 写入文件头
            f_pass.write("="*100 + "\n")
            f_pass.write("通过质量检查的谱图\n")
            f_pass.write("="*100 + "\n\n")
            
            f_fail.write("="*100 + "\n")
            f_fail.write("未通过质量检查的谱图\n")
            f_fail.write("="*100 + "\n\n")
            
            passed = 0
            failed = 0
            passed_examples = []
            failed_examples = []
            
            # 重新打开文件进行详细分析
            with mgf.read(mgf_path) as reader:
                for idx, spec in enumerate(tqdm(reader, total=total_spectra, desc="处理中")):
                    try:
                        # 获取precursor信息
                        params = spec['params']
                        
                        # 获取precursor m/z和charge
                        pepmass = params.get('pepmass', [0])
                        precursor_mz = float(pepmass[0] if isinstance(pepmass, (list, tuple)) else pepmass)
                        
                        charge = params.get('charge', [2])
                        precursor_charge = charge[0] if isinstance(charge, (list, tuple)) else charge
                        if isinstance(precursor_charge, str):
                            precursor_charge = int(precursor_charge.strip('+'))
                        else:
                            precursor_charge = int(precursor_charge)
                        
                        # 获取序列（如果有）
                        seq = params.get('seq', '')
                        
                        if not seq:
                            continue
                        
                        # 计算precursor质量
                        precursor_mass = calculate_precursor_mass_from_mz(precursor_mz, precursor_charge)
                        
                        # 计算peptide质量
                        try:
                            peptide_mass_no_water, tokens = mass_cal_piprime(seq)
                            peptide_mass = peptide_mass_no_water + H2O_MASS
                            
                            # 检查匹配（使用正确的方法：两边都减去水的质量）
                            is_match = check_mass_match(peptide_mass, precursor_mass, tolerance)
                            
                            # 计算差异
                            precursor_no_water = precursor_mass - H2O_MASS
                            peptide_no_water = peptide_mass - H2O_MASS
                            diff = abs(precursor_no_water - peptide_no_water)
                            
                            # 准备详细信息
                            detail = f"谱图 #{idx}\n"
                            detail += f"{'─'*100}\n"
                            detail += f"序列: {seq}\n"
                            detail += f"Precursor m/z: {precursor_mz:.6f}, Charge: {precursor_charge}+\n"
                            detail += f"\n质量计算:\n"
                            detail += f"  Precursor质量 (含水): {precursor_mass:.6f} Da\n"
                            detail += f"  Precursor质量 (不含水): {precursor_no_water:.6f} Da\n"
                            detail += f"  Peptide质量 (含水): {peptide_mass:.6f} Da\n"
                            detail += f"  Peptide质量 (不含水): {peptide_no_water:.6f} Da\n"
                            detail += f"  差异 (不含水): {diff:.6f} Da\n"
                            detail += f"  容差: ±{tolerance} Da\n"
                            detail += f"\nTokens: {tokens}\n"
                            detail += f"\n"
                            
                            if is_match:
                                passed += 1
                                f_pass.write(detail)
                                f_pass.write("✅ 匹配！\n")
                                f_pass.write("="*100 + "\n\n")
                                
                                # 保存前10个通过的例子
                                if len(passed_examples) < 10:
                                    passed_examples.append({
                                        'idx': idx,
                                        'seq': seq,
                                        'mz': precursor_mz,
                                        'charge': precursor_charge,
                                        'diff': diff
                                    })
                            else:
                                failed += 1
                                f_fail.write(detail)
                                f_fail.write("❌ 不匹配！\n")
                                f_fail.write("="*100 + "\n\n")
                                
                                # 保存所有失败的例子
                                failed_examples.append({
                                    'idx': idx,
                                    'seq': seq,
                                    'mz': precursor_mz,
                                    'charge': precursor_charge,
                                    'diff': diff
                                })
                                
                                # 同时打印到控制台
                                print(f"\n不匹配: seq={seq}, m/z={precursor_mz}, charge={charge}")
                                print(f"  差异: {diff:.6f} Da (容差: ±{tolerance} Da)")
                                
                        except Exception as e:
                            print(f"\n处理序列时出错: {seq}, 错误: {str(e)}")
                            failed += 1
                            f_fail.write(f"谱图 #{idx}\n")
                            f_fail.write(f"序列: {seq}\n")
                            f_fail.write(f"错误: {str(e)}\n")
                            f_fail.write("="*100 + "\n\n")
                        
                    except Exception as e:
                        print(f"\n处理谱图时出错: {str(e)}")
                        failed += 1
            
            # 写入总结
            f_summary.write("="*100 + "\n")
            f_summary.write("质量过滤测试总结\n")
            f_summary.write("="*100 + "\n\n")
            f_summary.write(f"测试文件: {os.path.basename(mgf_path)}\n")
            f_summary.write(f"容差: ±{tolerance} Da\n")
            f_summary.write(f"总谱图数: {total_spectra}\n")
            f_summary.write(f"通过: {passed} ({passed/total_spectra*100:.2f}%)\n")
            f_summary.write(f"失败: {failed} ({failed/total_spectra*100:.2f}%)\n\n")
            
            f_summary.write("="*100 + "\n")
            f_summary.write("通过的例子 (前10个)\n")
            f_summary.write("="*100 + "\n")
            for i, ex in enumerate(passed_examples, 1):
                f_summary.write(f"\n{i}. 谱图 #{ex['idx']}\n")
                f_summary.write(f"   序列: {ex['seq']}\n")
                f_summary.write(f"   m/z: {ex['mz']:.6f}, Charge: {ex['charge']}+\n")
                f_summary.write(f"   差异: {ex['diff']:.6f} Da\n")
            
            f_summary.write("\n" + "="*100 + "\n")
            f_summary.write(f"失败的例子 (共{len(failed_examples)}个)\n")
            f_summary.write("="*100 + "\n")
            for i, ex in enumerate(failed_examples, 1):
                f_summary.write(f"\n{i}. 谱图 #{ex['idx']}\n")
                f_summary.write(f"   序列: {ex['seq']}\n")
                f_summary.write(f"   m/z: {ex['mz']:.6f}, Charge: {ex['charge']}+\n")
                f_summary.write(f"   差异: {ex['diff']:.6f} Da (超出容差)\n")
            
            print(f"\n{'='*40}")
            print(f"测试完成")
            print(f"{'='*40}")
            print(f"总谱图: {total_spectra}")
            print(f"通过: {passed} ({passed/total_spectra*100:.2f}%)")
            print(f"失败: {failed} ({failed/total_spectra*100:.2f}%)")
            print(f"\n结果已保存到:")
            print(f"  通过: {passed_file}")
            print(f"  失败: {failed_file}")
            print(f"  总结: {summary_file}")
            
    except Exception as e:
        print(f"处理文件时出错: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 测试所有testdata目录下的MGF文件
    test_data_dir = os.path.join(os.path.dirname(__file__), 'testdata')
    
    if not os.path.exists(test_data_dir):
        print(f"错误: 测试数据目录不存在: {test_data_dir}")
        sys.exit(1)
    
    # 获取所有MGF文件
    mgf_files = [f for f in os.listdir(test_data_dir) if f.endswith('.mgf')]
    
    if not mgf_files:
        print(f"错误: 在 {test_data_dir} 中找不到MGF文件")
        sys.exit(1)
    
    print(f"在 {test_data_dir} 中找到 {len(mgf_files)} 个MGF文件")
    
    # 测试每个文件
    for mgf_file in sorted(mgf_files):
        mgf_path = os.path.join(test_data_dir, mgf_file)
        test_mgf_file(mgf_path, tolerance=0.1)
    
    print("\n所有测试完成！")
