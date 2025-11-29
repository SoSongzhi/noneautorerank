#!/usr/bin/env python
"""
构建高效谱图索引 - 零遍历快速查找

用法:
python build_efficient_index.py --reference_mgf <path_to_mgf>
"""

import argparse
import pickle
import numpy as np
import re
from pathlib import Path
from collections import defaultdict
from pyteomics import mgf
import time


class EfficientIndexBuilder:
    """高效索引构建器"""

    def __init__(self):
        pass
    
    def convert_modification_to_unimod(self, peptide: str) -> str:
        """
        将修饰格式转换为Unimod ID标准格式
        用于统一database和预测结果的格式
        
        转换规则:
        - M(+15.99) -> M[UNIMOD:35]
        - C(+57.02) -> C[UNIMOD:4]
        - N(+.98) -> N[UNIMOD:7]
        """
        if not peptide:
            return peptide
        
        # Unimod映射
        mod_name_to_unimod = {
            'Oxidation': 'UNIMOD:35',
            'Deamidated': 'UNIMOD:7',
            'Carbamidomethyl': 'UNIMOD:4',
            'Acetyl': 'UNIMOD:1',
            'Carbamyl': 'UNIMOD:5',
        }
        
        # 质量范围到Unimod ID
        mass_to_unimod = [
            (15.85, 16.15, 'UNIMOD:35'),  # Oxidation
            (0.83, 1.13, 'UNIMOD:7'),     # Deamidation
            (56.87, 57.17, 'UNIMOD:4'),   # Carbamidomethyl
            (41.86, 42.16, 'UNIMOD:1'),   # Acetyl
            (42.86, 43.16, 'UNIMOD:5'),   # Carbamyl
        ]
        
        converted = peptide
        
        # 处理命名修饰
        for mod_name, unimod_id in mod_name_to_unimod.items():
            converted = converted.replace(f'[{mod_name}]', f'[{unimod_id}]')
        
        # 处理数值修饰 (支持圆括号和方括号)
        pattern = r'[\[\(]([+\-]?\d*\.?\d+)[\]\)]'
        
        def replace_numeric_mod(match):
            mass_str = match.group(1)
            try:
                mass_val = float(mass_str)
                
                # 查找匹配的Unimod ID
                for min_mass, max_mass, unimod_id in mass_to_unimod:
                    if min_mass <= mass_val <= max_mass:
                        return f'[{unimod_id}]'
                
                # 未知修饰，保留质量
                if mass_val >= 0:
                    return f'[+{mass_val:.6f}]'
                else:
                    return f'[{mass_val:.6f}]'
                    
            except ValueError:
                return f'[{mass_str}]'
        
        converted = re.sub(pattern, replace_numeric_mod, converted)
        return converted

    def normalize_peptide(self, peptide):
        """标准化肽段序列（去除修饰符号和前缀，并将L转换为I）"""
        clean_seq = re.sub(r'\[.*?\]', '', peptide)
        clean_seq = re.sub(r'^[\[\]A-Za-z0-9\-\+\.]+\-', '', clean_seq)
        # 将L转换为I（亮氨酸和异亮氨酸在质谱中无法区分）
        clean_seq = clean_seq.replace('L', 'I')
        return clean_seq

    def extract_sequence(self, spectrum):
        """从谱图参数中提取序列"""
        seq = spectrum['params'].get('seq', '')
        if not seq and 'title' in spectrum['params']:
            match = re.search(r'[Ss]eq[=:]([A-Z\[\]0-9\-\+\.]+)', spectrum['params']['title'])
            if match:
                seq = match.group(1)
        return seq

    def build_index(self, reference_mgf):
        """
        构建高效索引（带Unimod格式转换）

        索引结构:
        {
            "PEPTIDE": {
                "spectra": [
                    {
                        "mz": np.array([...]),
                        "intensity": np.array([...]),
                        "precursor_mz": 500.25,
                        "charge": 2,
                        "index": 0,
                        "raw_seq": "PEPTIDE",  # 原始序列
                        "unimod_seq": "PEPTIDE[UNIMOD:4]"  # Unimod格式
                    },
                    ...
                ]
            }
        }
        """
        print("="*70)
        print("Building Efficient Spectrum Index with Unimod Conversion")
        print("="*70)
        print(f"Reference MGF: {reference_mgf}")

        index = defaultdict(lambda: {"spectra": []})
        total_spectra = 0
        converted_count = 0

        start_time = time.time()

        with mgf.MGF(reference_mgf) as reader:
            for idx, spectrum in enumerate(reader):
                # 提取序列
                raw_seq = self.extract_sequence(spectrum)
                if not raw_seq:
                    continue

                # 转换为Unimod格式
                unimod_seq = self.convert_modification_to_unimod(raw_seq)
                if unimod_seq != raw_seq:
                    converted_count += 1
                
                # 标准化序列（用于索引key）
                clean_seq = self.normalize_peptide(unimod_seq)
                
                # 同时保存带修饰的序列作为额外的key
                # 这样可以支持带修饰的查询
                modified_seq = unimod_seq.replace('L', 'I')  # 只替换L->I，保留修饰

                # 提取前体信息
                pepmass = spectrum['params'].get('pepmass')
                if isinstance(pepmass, (list, tuple)):
                    precursor_mz = pepmass[0] if len(pepmass) > 0 else 0
                else:
                    precursor_mz = pepmass if pepmass else 0

                charge = spectrum['params'].get('charge')
                if isinstance(charge, (list, tuple)):
                    charge = charge[0] if len(charge) > 0 else 2
                else:
                    charge = charge if charge else 2

                # 清理电荷值
                if isinstance(charge, str):
                    charge = int(charge.replace('+', '').replace('-', ''))
                else:
                    charge = int(charge)

                # 存储谱图信息
                spec_data = {
                    "mz": spectrum['m/z array'],
                    "intensity": spectrum['intensity array'],
                    "precursor_mz": float(precursor_mz),
                    "charge": charge,
                    "index": idx,
                    "raw_seq": raw_seq,
                    "unimod_seq": unimod_seq  # 添加Unimod格式
                }

                # 添加到clean_seq索引（不带修饰）
                index[clean_seq]["spectra"].append(spec_data)
                
                # 如果有修饰，也添加到modified_seq索引（带修饰）
                if modified_seq != clean_seq:
                    index[modified_seq]["spectra"].append(spec_data)
                
                total_spectra += 1

                if (idx + 1) % 10000 == 0:
                    print(f"  Processed {idx + 1} spectra... ({converted_count} with modifications)")

        elapsed = time.time() - start_time

        # 统计信息
        unique_sequences = len(index)
        avg_spectra_per_seq = total_spectra / max(1, unique_sequences)

        print(f"\n{'='*70}")
        print("Index Statistics:")
        print(f"  Total spectra: {total_spectra}")
        print(f"  Spectra with modifications converted: {converted_count} ({converted_count/total_spectra*100:.1f}%)")
        print(f"  Unique sequences: {unique_sequences}")
        print(f"  Average spectra per sequence: {avg_spectra_per_seq:.2f}")
        print(f"  Build time: {elapsed:.1f}s")
        print(f"{'='*70}")

        # 显示一些样例
        print("\nSample sequences (first 5):")
        for i, (seq, data) in enumerate(list(index.items())[:5]):
            print(f"  {seq}: {len(data['spectra'])} spectra")

        # 显示分布
        spectra_counts = [len(data['spectra']) for data in index.values()]
        print(f"\nSpectra count distribution:")
        print(f"  Min: {min(spectra_counts)}")
        print(f"  Max: {max(spectra_counts)}")
        print(f"  Median: {np.median(spectra_counts):.1f}")
        print(f"  Mean: {np.mean(spectra_counts):.2f}")

        sequences_with_multiple = sum(1 for c in spectra_counts if c > 1)
        print(f"  Sequences with multiple spectra: {sequences_with_multiple} ({sequences_with_multiple/unique_sequences*100:.1f}%)")

        return dict(index)

    def save_index(self, index, output_file):
        """保存索引到文件"""
        print(f"\nSaving index to: {output_file}")

        with open(output_file, 'wb') as f:
            pickle.dump(index, f, protocol=pickle.HIGHEST_PROTOCOL)

        # 检查文件大小
        file_size = Path(output_file).stat().st_size / (1024**2)  # MB
        print(f"Index file size: {file_size:.2f} MB")
        print("Done!")


def main():
    parser = argparse.ArgumentParser(
        description='Build efficient spectrum index for fast lookup'
    )
    parser.add_argument(
        '--reference_mgf',
        required=True,
        help='Path to reference MGF file'
    )
    parser.add_argument(
        '--output',
        help='Output index file (default: <reference_mgf>.efficient_index.pkl)'
    )

    args = parser.parse_args()

    # 构建输出文件名
    if args.output is None:
        args.output = f"{args.reference_mgf}.efficient_index.pkl"

    # 构建索引
    builder = EfficientIndexBuilder()
    index = builder.build_index(args.reference_mgf)

    # 保存索引
    builder.save_index(index, args.output)

    print("\n" + "="*70)
    print("Index building complete!")
    print("You can now use this index for fast reranking.")
    print("="*70)


if __name__ == "__main__":
    main()
