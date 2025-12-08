#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
根据peak数量过滤MGF文件中的spectrum

功能：
1. 过滤掉peak数量少于指定阈值的spectrum
2. 将满足条件的spectrum保存到新文件（>=阈值）
3. 将不满足条件的spectrum保存到另一个文件（<阈值）

使用方法:
python filter_peaks.py <mgf_file> [--min_peaks N] [--output_dir DIR]

示例:
python filter_peaks.py "data.mgf" --min_peaks 50
python filter_peaks.py "data.mgf" --min_peaks 50 --output_dir filtered_data/
"""

import sys
import os
import argparse
from pathlib import Path
from pyteomics import mgf
from tqdm import tqdm


def filter_mgf_by_peaks(mgf_file, min_peaks=50, output_dir=None):
    """
    根据peak数量过滤MGF文件
    
    Parameters:
    -----------
    mgf_file : str
        输入MGF文件路径
    min_peaks : int
        最小peak数量阈值（默认50）
    output_dir : str, optional
        输出目录（默认为输入文件所在目录）
    
    Returns:
    --------
    tuple : (passed_file, filtered_file, stats)
        返回两个输出文件路径和统计信息
    """
    mgf_path = Path(mgf_file)
    
    # 设置输出目录
    if output_dir is None:
        output_dir = mgf_path.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # 输出文件名
    base_name = mgf_path.stem
    passed_file = output_dir / f"{base_name}_peak{min_peaks}plus.mgf"
    filtered_file = output_dir / f"{base_name}_peak{min_peaks}minus.mgf"
    
    print(f"\n{'='*80}")
    print(f"过滤MGF文件: {mgf_path.name}")
    print(f"{'='*80}")
    print(f"最小peak数量阈值: {min_peaks}")
    print(f"输出目录: {output_dir}")
    print(f"")
    
    # 统计变量
    total_spectra = 0
    passed_count = 0
    filtered_count = 0
    
    # 先统计总数
    print("统计总spectrum数...")
    with mgf.MGF(mgf_file) as reader:
        total_spectra = sum(1 for _ in reader)
    
    print(f"总spectrum数: {total_spectra:,}\n")
    print("开始过滤...")
    
    # 打开输出文件
    with mgf.MGF(mgf_file) as reader, \
         open(passed_file, 'w') as f_pass, \
         open(filtered_file, 'w') as f_filt:
        
        for spec in tqdm(reader, total=total_spectra, desc="处理中", unit="spectrum"):
            num_peaks = len(spec['m/z array'])
            
            if num_peaks >= min_peaks:
                # 满足条件：保存到 passed 文件
                mgf.write([spec], f_pass)
                passed_count += 1
            else:
                # 不满足条件：保存到 filtered 文件
                mgf.write([spec], f_filt)
                filtered_count += 1
    
    # 统计信息
    stats = {
        'total': total_spectra,
        'passed': passed_count,
        'filtered': filtered_count,
        'passed_pct': passed_count / total_spectra * 100 if total_spectra > 0 else 0,
        'filtered_pct': filtered_count / total_spectra * 100 if total_spectra > 0 else 0
    }
    
    # 输出结果
    print(f"\n{'='*80}")
    print("过滤完成！")
    print(f"{'='*80}")
    print(f"总spectrum数: {total_spectra:,}")
    print(f"")
    print(f"✓ 满足条件 (>= {min_peaks} peaks):")
    print(f"  数量: {passed_count:,} ({stats['passed_pct']:.2f}%)")
    print(f"  文件: {passed_file}")
    print(f"")
    print(f"✗ 不满足条件 (< {min_peaks} peaks):")
    print(f"  数量: {filtered_count:,} ({stats['filtered_pct']:.2f}%)")
    print(f"  文件: {filtered_file}")
    print(f"{'='*80}\n")
    
    # 保存统计信息到文本文件
    stats_file = output_dir / f"{base_name}_peak{min_peaks}_filter_stats.txt"
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write(f"Peak数量过滤统计\n")
        f.write(f"{'='*80}\n")
        f.write(f"输入文件: {mgf_file}\n")
        f.write(f"最小peak阈值: {min_peaks}\n")
        f.write(f"处理时间: {Path(__file__).stat().st_mtime}\n")
        f.write(f"\n")
        f.write(f"统计结果:\n")
        f.write(f"  总spectrum数: {total_spectra:,}\n")
        f.write(f"  满足条件 (>= {min_peaks} peaks): {passed_count:,} ({stats['passed_pct']:.2f}%)\n")
        f.write(f"  不满足条件 (< {min_peaks} peaks): {filtered_count:,} ({stats['filtered_pct']:.2f}%)\n")
        f.write(f"\n")
        f.write(f"输出文件:\n")
        f.write(f"  满足条件: {passed_file.name}\n")
        f.write(f"  不满足条件: {filtered_file.name}\n")
    
    print(f"✓ 统计信息已保存: {stats_file}\n")
    
    return passed_file, filtered_file, stats


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='根据peak数量过滤MGF文件中的spectrum',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 过滤掉peak少于50的spectrum
  python filter_peaks.py "data.mgf" --min_peaks 50
  
  # 指定输出目录
  python filter_peaks.py "data.mgf" --min_peaks 50 --output_dir filtered/
  
  # 使用不同的阈值
  python filter_peaks.py "data.mgf" --min_peaks 100
  
输出文件:
  {basename}_peak{N}plus.mgf  - 满足条件的spectrum (>= N peaks)
  {basename}_peak{N}minus.mgf - 不满足条件的spectrum (< N peaks)
  {basename}_peak{N}_filter_stats.txt - 统计信息
        """
    )
    
    parser.add_argument('mgf_file', type=str, help='输入MGF文件路径')
    parser.add_argument('--min_peaks', type=int, default=50,
                       help='最小peak数量阈值（默认：50）')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='输出目录（默认：输入文件所在目录）')
    
    args = parser.parse_args()
    
    # 验证文件
    if not os.path.exists(args.mgf_file):
        print(f"❌ 错误: 文件不存在: {args.mgf_file}")
        sys.exit(1)
    
    # 验证阈值
    if args.min_peaks < 1:
        print(f"❌ 错误: 最小peak数量必须 >= 1")
        sys.exit(1)
    
    # 执行过滤
    try:
        filter_mgf_by_peaks(args.mgf_file, args.min_peaks, args.output_dir)
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()