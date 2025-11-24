"""
PiPrime质量计算模块
使用与PiPrime完全相同的质量计算方法
"""
import re
from typing import Tuple, List

# PiPrime的氨基酸质量字典（来自pi-PrimeNovo/PrimeNovo/denovo/model.py第22行）
AA2MAS = {
    'G': 57.021464,
    'A': 71.037114,
    'S': 87.032028,
    'P': 97.052764,
    'V': 99.068414,
    'T': 101.04767,
    'C+57.021': 160.030649,  # Carbamidomethylation
    'L': 113.084064,
    'I': 113.084064,
    'N': 114.042927,
    'D': 115.026943,
    'Q': 128.058578,
    'K': 128.094963,
    'E': 129.042593,
    'M': 131.040485,
    'H': 137.058912,
    'F': 147.068414,
    'R': 156.101111,
    'Y': 163.063329,
    'W': 186.079313,
    'M+15.995': 147.0354,    # Oxidation
    'N+0.984': 115.026943,   # Deamidation
    'Q+0.984': 129.042594,   # Deamidation
    '+42.011': 42.010565,    # Acetylation
    '+43.006': 43.005814,    # Carbamylation
    '-17.027': -17.026549,   # Ammonia loss
    '+43.006-17.027': 25.980265,
    '_': 0  # Blank token
}

# 水分子质量
H2O_MASS = 18.010565


def mass_cal_piprime(sequence: str) -> Tuple[float, List[str]]:
    """
    使用PiPrime的方式计算peptide质量
    这是从pi-PrimeNovo/PrimeNovo/denovo/model.py复制的mass_cal函数
    
    Args:
        sequence: peptide序列字符串，例如 "SISC+57.021TYDDDTYR"
        
    Returns:
        (total_mass, tokens): 总质量和token列表
    """
    # 将I替换为L（PiPrime将它们视为相同）
    sequence = sequence.replace("I", "L")
    
    # 按照大写字母分割序列，保留修饰
    sequence = re.split(r"(?<=.)(?=[A-Z])", sequence)
    
    total = 0.0
    for each in sequence:
        try:
            # 直接从字典获取质量
            total += AA2MAS[each]
        except KeyError:
            # 处理特殊修饰
            h1 = each.count("+42.011")
            h2 = each.count("+43.006")
            h3 = each.count("-17.027")
            total += h1 * 42.010565 + h2 * 43.005814 + h3 * -17.026549
            
            # 移除修饰后获取基础氨基酸
            each = each.replace("+42.011", "")
            each = each.replace("+43.006", "")
            each = each.replace("-17.027", "")
            if each and each in AA2MAS:
                total += AA2MAS[each]
    
    return total, sequence


def calculate_peptide_mass_piprime(sequence: str, add_water: bool = True) -> float:
    """
    计算peptide质量（使用PiPrime方法）
    
    Args:
        sequence: peptide序列
        add_water: 是否加上水分子质量（默认True）
        
    Returns:
        peptide质量
    """
    mass, _ = mass_cal_piprime(sequence)
    if add_water:
        mass += H2O_MASS
    return mass


def normalize_sequence_format(sequence: str) -> str:
    """
    将序列标准化为PiPrime格式
    
    Args:
        sequence: 原始序列，可能包含括号等
        
    Returns:
        标准化后的序列
    """
    # 移除括号：C(+57.021) -> C+57.021
    sequence = sequence.replace("(", "").replace(")", "")
    
    # 移除空格
    sequence = sequence.replace(" ", "")
    
    # 将I替换为L
    sequence = sequence.replace("I", "L")
    
    return sequence


def calculate_precursor_mass_from_mz(precursor_mz: float, precursor_charge: int) -> float:
    """
    从m/z和电荷计算precursor质量
    
    正确公式：precursor_mass = precursor_mz * charge - proton_mass * charge
    
    这个质量是peptide的中性质量（含水），因为：
    - m/z是带电荷的质量
    - 去掉质子后得到中性质量
    - peptide的中性质量包含两端的H和OH（即水）
    
    Args:
        precursor_mz: precursor m/z值
        precursor_charge: precursor电荷
        
    Returns:
        precursor质量（含水的中性质量）
    """
    proton_mass = 1.007276
    # 这个计算得到的就是含水的中性质量
    precursor_mass = precursor_mz * precursor_charge - proton_mass * precursor_charge
    return precursor_mass


def check_mass_match(peptide_mass: float, precursor_mass: float, tolerance: float = 0.1) -> bool:
    """
    检查peptide质量是否与precursor质量匹配
    
    按照PiPrime的方式：
    - peptide_mass: 含水的peptide质量
    - precursor_mass: 含水的precursor质量
    - 比较时都减去水的质量
    
    Args:
        peptide_mass: peptide质量（含水）
        precursor_mass: precursor质量（含水）
        tolerance: 质量容差（Da）
        
    Returns:
        是否匹配
    """
    # 按照PiPrime的方式：mass_true = mass[0].item() - 18.01
    # 两边都减去水的质量后比较
    peptide_mass_no_water = peptide_mass - H2O_MASS
    precursor_mass_no_water = precursor_mass - H2O_MASS
    mass_diff = abs(peptide_mass_no_water - precursor_mass_no_water)
    return mass_diff <= tolerance


if __name__ == "__main__":
    # 测试
    print("=" * 80)
    print("PiPrime质量计算测试")
    print("=" * 80)
    
    test_cases = [
        "SISC+57.021TYDDDTYR",  # 包含C+57.021
        "M+15.995PEPTLDE",       # 包含M+15.995
        "PEPN+0.984TLDE",        # 包含N+0.984
        "PEPTLDE",               # 普通序列
    ]
    
    for seq in test_cases:
        mass, tokens = mass_cal_piprime(seq)
        mass_with_water = mass + H2O_MASS
        print(f"\n序列: {seq}")
        print(f"  Tokens: {tokens}")
        print(f"  质量（不含水）: {mass:.6f} Da")
        print(f"  质量（含水）: {mass_with_water:.6f} Da")
    
    print("\n" + "=" * 80)
    print("Precursor质量计算测试")
    print("=" * 80)
    
    # 测试precursor质量计算
    precursor_mz = 748.3033
    precursor_charge = 2
    precursor_mass = calculate_precursor_mass_from_mz(precursor_mz, precursor_charge)
    print(f"\nPrecursor m/z: {precursor_mz}")
    print(f"Precursor charge: {precursor_charge}")
    print(f"Precursor mass: {precursor_mass:.6f} Da")
    
    # 测试质量匹配
    peptide_seq = "SISC+57.021TYDDDTYR"
    peptide_mass = calculate_peptide_mass_piprime(peptide_seq)
    is_match = check_mass_match(peptide_mass, precursor_mass, tolerance=0.1)
    print(f"\nPeptide: {peptide_seq}")
    print(f"Peptide mass (含水): {peptide_mass:.6f} Da")
    print(f"Peptide mass (不含水): {peptide_mass - H2O_MASS:.6f} Da")
    print(f"Precursor mass (含水): {precursor_mass:.6f} Da")
    print(f"Precursor mass (不含水): {precursor_mass - H2O_MASS:.6f} Da")
    print(f"Mass match (±0.1 Da): {is_match}")
    print(f"Mass difference (不含水): {abs((peptide_mass - H2O_MASS) - (precursor_mass - H2O_MASS)):.6f} Da")