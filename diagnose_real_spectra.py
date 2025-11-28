#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Diagnose precursor mass calculation from real MGF spectra
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import os
from pyteomics import mgf
from piprime_mass_calculator import (
    calculate_precursor_mass_from_mz,
    calculate_peptide_mass_piprime,
    mass_cal_piprime,
    AA2MAS,
    H2O_MASS
)

def diagnose_spectrum(spec, spec_idx):
    """详细诊断单个谱图"""
    params = spec['params']
    
    # 获取precursor信息
    pepmass = params.get('pepmass', [0])
    precursor_mz = float(pepmass[0] if isinstance(pepmass, (list, tuple)) else pepmass)
    
    charge = params.get('charge', [2])
    precursor_charge = charge[0] if isinstance(charge, (list, tuple)) else charge
    if isinstance(precursor_charge, str):
        precursor_charge = int(precursor_charge.strip('+'))
    else:
        precursor_charge = int(precursor_charge)
    
    seq = params.get('seq', '')
    
    if not seq:
        return False
    
    print(f"\n{'='*100}")
    print(f"谱图 #{spec_idx}")
    print(f"{'='*100}")
    print(f"原始序列: {seq}")
    print(f"Precursor m/z: {precursor_mz:.6f}")
    print(f"Charge: {precursor_charge}+")
    
    # ===== 步骤1: 计算Precursor质量 =====
    print(f"\n{'─'*100}")
    print("步骤1: 从m/z计算Precursor质量")
    print(f"{'─'*100}")
    
    proton_mass = 1.007276
    precursor_mass = calculate_precursor_mass_from_mz(precursor_mz, precursor_charge)
    
    print(f"公式: precursor_mass = m/z × charge - proton_mass × charge")
    print(f"计算: {precursor_mz:.6f} × {precursor_charge} - {proton_mass} × {precursor_charge}")
    print(f"     = {precursor_mz * precursor_charge:.6f} - {proton_mass * precursor_charge:.6f}")
    print(f"     = {precursor_mass:.6f} Da")
    print(f"\n这是peptide的中性质量（含水）")
    print(f"不含水的质量: {precursor_mass - H2O_MASS:.6f} Da")
    
    # ===== 步骤2: 使用PiPrime方法计算Peptide质量 =====
    print(f"\n{'─'*100}")
    print("步骤2: 使用PiPrime方法计算Peptide质量")
    print(f"{'─'*100}")
    
    try:
        # 使用mass_cal_piprime获取详细信息
        peptide_mass_no_water, tokens = mass_cal_piprime(seq)
        peptide_mass_with_water = peptide_mass_no_water + H2O_MASS
        
        print(f"序列分词: {tokens}")
        print(f"\n逐个token的质量:")
        for i, token in enumerate(tokens, 1):
            if token in AA2MAS:
                print(f"  {i}. {token:15s} -> {AA2MAS[token]:10.6f} Da")
            else:
                print(f"  {i}. {token:15s} -> 未知")
        
        print(f"\n总质量 (不含水): {peptide_mass_no_water:.6f} Da")
        print(f"加上水分子: {peptide_mass_no_water:.6f} + {H2O_MASS:.6f} = {peptide_mass_with_water:.6f} Da")
        
        # ===== 步骤3: 比较两个质量 =====
        print(f"\n{'─'*100}")
        print("步骤3: 比较Precursor质量和Peptide质量")
        print(f"{'─'*100}")
        
        print(f"\n含水质量比较:")
        print(f"  Precursor质量 (含水): {precursor_mass:.6f} Da")
        print(f"  Peptide质量 (含水):   {peptide_mass_with_water:.6f} Da")
        print(f"  差异:                  {abs(precursor_mass - peptide_mass_with_water):.6f} Da")
        
        print(f"\n不含水质量比较 (PiPrime的比较方式):")
        precursor_no_water = precursor_mass - H2O_MASS
        peptide_no_water = peptide_mass_with_water - H2O_MASS
        diff_no_water = abs(precursor_no_water - peptide_no_water)
        
        print(f"  Precursor质量 (不含水): {precursor_no_water:.6f} Da")
        print(f"  Peptide质量 (不含水):   {peptide_no_water:.6f} Da")
        print(f"  差异:                    {diff_no_water:.6f} Da")
        
        # ===== 步骤4: 判断是否匹配 =====
        print(f"\n{'─'*100}")
        print("步骤4: 判断是否在容差范围内")
        print(f"{'─'*100}")
        
        tolerance = 0.1  # Da
        is_match = diff_no_water <= tolerance
        
        print(f"容差: ±{tolerance} Da")
        print(f"实际差异: {diff_no_water:.6f} Da")
        
        if is_match:
            print(f"✅ 匹配！在容差范围内")
        else:
            print(f"❌ 不匹配！超出容差范围")
            print(f"\n可能的原因:")
            print(f"  1. 修饰解析错误")
            print(f"  2. MGF文件中的序列标注错误")
            print(f"  3. Precursor m/z测量误差")
            print(f"  4. 质量计算方法不一致")
        
        return not is_match  # 返回True表示不匹配
        
    except Exception as e:
        print(f"\n❌ 计算失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("="*100)
    print("真实谱图Precursor Mass诊断工具")
    print("="*100)
    
    # 测试文件
    mgf_file = os.path.join('testdata', 'high_nine_validation_1000_converted.mgf')
    
    if not os.path.exists(mgf_file):
        print(f"错误: 文件不存在: {mgf_file}")
        exit(1)
    
    print(f"\n读取MGF文件: {mgf_file}")
    
    # 统计
    total_count = 0
    mismatch_count = 0
    match_count = 0
    
    # 只诊断包含修饰的谱图（+0.98, +0.984, +15.99, +15.995等）
    with mgf.read(mgf_file) as reader:
        for idx, spec in enumerate(reader):
            seq = spec['params'].get('seq', '')
            
            # 只看包含修饰的序列
            if seq and ('+0.98' in seq or '+15.99' in seq or '+0.984' in seq or '+15.995' in seq):
                is_mismatch = diagnose_spectrum(spec, idx)
                total_count += 1
                
                if is_mismatch:
                    mismatch_count += 1
                else:
                    match_count += 1
                
                if total_count >= 10:  # 只看前10个包含修饰的
                    break
    
    print(f"\n{'='*100}")
    print("诊断总结")
    print(f"{'='*100}")
    print(f"总谱图数: {total_count}")
    print(f"匹配: {match_count}")
    print(f"不匹配: {mismatch_count}")
    print(f"匹配率: {match_count/total_count*100:.1f}%")