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
    
    支持多种修饰格式：
    - PiPrime格式: C+57.021, M+15.995, N+0.984
    - MGF格式: C(+57.02), M(+15.99), N(+.98)
    
    Args:
        sequence: peptide序列字符串，例如 "SISC+57.021TYDDDTYR" 或 "SISM(+15.99)TYDDDTYR"
        
    Returns:
        (total_mass, tokens): 总质量和token列表
    """
    # 将I替换为L（PiPrime将它们视为相同）
    sequence = sequence.replace("I", "L")
    
    # 标准化修饰格式：将MGF格式转换为PiPrime格式
    # M(+15.99) -> M+15.995
    # N(+.98) -> N+0.984
    # Q(+.98) -> Q+0.984
    # C(+57.02) -> C+57.021
    sequence = re.sub(r'M\(\+15\.99\d*\)', 'M+15.995', sequence)
    sequence = re.sub(r'N\(\+\.98\d*\)', 'N+0.984', sequence)
    sequence = re.sub(r'N\(\+0\.98\d*\)', 'N+0.984', sequence)
    sequence = re.sub(r'Q\(\+\.98\d*\)', 'Q+0.984', sequence)
    sequence = re.sub(r'Q\(\+0\.98\d*\)', 'Q+0.984', sequence)
    sequence = re.sub(r'C\(\+57\.02\d*\)', 'C+57.021', sequence)
    
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


def calculate_precursor_mz_from_mass(peptide_mass: float, charge: int) -> float:
    """
    从peptide质量和电荷计算precursor m/z
    
    公式: m/z = (peptide_mass + proton_mass * charge) / charge
    
    Args:
        peptide_mass: peptide质量（含水）
        charge: 电荷数
        
    Returns:
        precursor m/z值
    """
    proton_mass = 1.007276
    precursor_mz = (peptide_mass + proton_mass * charge) / charge
    return precursor_mz


def peptide_to_precursor_info(peptide: str, charge: int = 2) -> dict:
    """
    计算peptide的完整precursor信息
    
    Args:
        peptide: peptide序列
        charge: 电荷数（默认2）
        
    Returns:
        包含所有计算结果的字典
    """
    # 计算质量
    peptide_mass_with_water = calculate_peptide_mass_piprime(peptide, add_water=True)
    peptide_mass_no_water = peptide_mass_with_water - H2O_MASS
    
    # 计算m/z
    precursor_mz = calculate_precursor_mz_from_mass(peptide_mass_with_water, charge)
    
    # 计算precursor mass（用于验证）
    precursor_mass = calculate_precursor_mass_from_mz(precursor_mz, charge)
    
    return {
        'peptide': peptide,
        'charge': charge,
        'peptide_mass_no_water': peptide_mass_no_water,
        'peptide_mass_with_water': peptide_mass_with_water,
        'precursor_mz': precursor_mz,
        'precursor_mass': precursor_mass,
        'mass_error': abs(peptide_mass_with_water - precursor_mass)
    }


if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Calculate precursor mass and m/z from peptide sequence',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python piprime_mass_calculator.py PEPTIDE
  python piprime_mass_calculator.py PEPTIDE --charge 3
  python piprime_mass_calculator.py "SISC+57.021TYDDDTYR" --charge 2
  python piprime_mass_calculator.py "M+15.995PEPTLDE" --charge 3
  python piprime_mass_calculator.py --test  # Run tests
        """
    )
    
    parser.add_argument('peptide', nargs='?', help='Peptide sequence')
    parser.add_argument('-c', '--charge', type=int, default=2,
                       help='Precursor charge (default: 2)')
    parser.add_argument('--test', action='store_true',
                       help='Run test cases')
    parser.add_argument('--show-mods', action='store_true',
                       help='Show supported modifications')
    
    args = parser.parse_args()
    
    if args.show_mods:
        print("\n" + "="*80)
        print("Supported Amino Acids and Modifications")
        print("="*80)
        print("\nStandard Amino Acids:")
        standard_aa = ['G', 'A', 'S', 'P', 'V', 'T', 'L', 'I', 'N', 'D', 'Q', 'K', 'E', 'M', 'H', 'F', 'R', 'Y', 'W']
        for aa in standard_aa:
            if aa in AA2MAS:
                print(f"  {aa}: {AA2MAS[aa]:.6f} Da")
        
        print("\nCommon Modifications:")
        mods = [
            ('C+57.021', 'Carbamidomethylation'),
            ('M+15.995', 'Oxidation'),
            ('N+0.984', 'Deamidation'),
            ('Q+0.984', 'Deamidation'),
        ]
        for mod, name in mods:
            if mod in AA2MAS:
                print(f"  {mod}: {AA2MAS[mod]:.6f} Da ({name})")
        print(f"\nWater mass (H2O): {H2O_MASS:.6f} Da")
        print("="*80)
        sys.exit(0)
    
    if args.test:
        # 测试
        print("=" * 80)
        print("PiPrime Mass Calculator - Test Cases")
        print("=" * 80)
        
        test_cases = [
            ("PEPTIDE", 2),
            ("SISC+57.021TYDDDTYR", 2),
            ("M+15.995PEPTLDE", 3),
            ("PEPN+0.984TLDE", 2),
        ]
        
        for seq, charge in test_cases:
            info = peptide_to_precursor_info(seq, charge)
            print(f"\nPeptide: {seq}")
            print(f"  Charge: {charge}+")
            print(f"  Peptide mass (no water): {info['peptide_mass_no_water']:.6f} Da")
            print(f"  Peptide mass (with water): {info['peptide_mass_with_water']:.6f} Da")
            print(f"  Precursor m/z: {info['precursor_mz']:.6f}")
            print(f"  Precursor mass: {info['precursor_mass']:.6f} Da")
        
        print("\n" + "=" * 80)
        sys.exit(0)
    
    if args.peptide:
        # 计算单个peptide
        try:
            info = peptide_to_precursor_info(args.peptide, args.charge)
            
            print("\n" + "="*80)
            print(f"Peptide: {info['peptide']}")
            print("="*80)
            print(f"Charge: {info['charge']}+")
            print(f"\nMass Calculations:")
            print(f"  Peptide mass (no water): {info['peptide_mass_no_water']:.6f} Da")
            print(f"  Peptide mass (with water): {info['peptide_mass_with_water']:.6f} Da")
            print(f"\nPrecursor Calculations:")
            print(f"  Precursor m/z: {info['precursor_mz']:.6f}")
            print(f"  Precursor mass: {info['precursor_mass']:.6f} Da")
            print(f"\nVerification:")
            print(f"  Mass error: {info['mass_error']:.9f} Da")
            print("="*80 + "\n")
            
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        parser.print_help()
        print("\nExamples:")
        print("  python piprime_mass_calculator.py PEPTIDE")
        print("  python piprime_mass_calculator.py 'SISC+57.021TYDDDTYR' --charge 2")
        print("  python piprime_mass_calculator.py --test")