"""
Prepare Training Data from MGF Files
从MGF文件准备训练数据
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np
import pickle
from collections import defaultdict
import argparse
from tqdm import tqdm


def parse_mgf_file(mgf_path):
    """
    解析MGF文件
    
    Returns:
        List of dict with keys: 'mz_array', 'intensity_array', 'peptide', 'charge', 'precursor_mz'
    """
    spectra = []
    current_spectrum = {}
    peaks = []
    
    with open(mgf_path, 'r') as f:
        for line in f:
            line = line.strip()
            
            if line == 'BEGIN IONS':
                current_spectrum = {}
                peaks = []
            
            elif line == 'END IONS':
                if peaks and 'peptide' in current_spectrum:
                    # 转换peaks为numpy数组
                    peaks_array = np.array(peaks, dtype=np.float32)
                    current_spectrum['mz_array'] = peaks_array[:, 0]
                    current_spectrum['intensity_array'] = peaks_array[:, 1]
                    
                    # 归一化intensity到[0, 1]
                    max_intensity = current_spectrum['intensity_array'].max()
                    if max_intensity > 0:
                        current_spectrum['intensity_array'] = current_spectrum['intensity_array'] / max_intensity
                    
                    spectra.append(current_spectrum)
                peaks = []
            
            elif line.startswith('TITLE='):
                current_spectrum['title'] = line.split('=', 1)[1]
            
            elif line.startswith('PEPMASS='):
                current_spectrum['precursor_mz'] = float(line.split('=')[1].split()[0])
            
            elif line.startswith('CHARGE='):
                charge_str = line.split('=')[1].replace('+', '').replace('-', '')
                try:
                    current_spectrum['charge'] = int(charge_str)
                except:
                    current_spectrum['charge'] = 2  # 默认值
            
            elif line.startswith('SEQ=') or line.startswith('PEPTIDE='):
                # 提取peptide序列
                peptide = line.split('=', 1)[1]
                # 移除修饰符号，只保留氨基酸序列
                # 这里简化处理，你可能需要根据实际情况调整
                current_spectrum['peptide'] = peptide
            
            elif line and not line.startswith(('TITLE', 'PEPMASS', 'CHARGE', 'RTINSECONDS', 'SCANS')):
                # 这是peak数据
                try:
                    parts = line.split()
                    if len(parts) >= 2:
                        mz = float(parts[0])
                        intensity = float(parts[1])
                        peaks.append([mz, intensity])
                except:
                    pass
    
    return spectra


def extract_peptide_from_title(title):
    """
    从TITLE中提取peptide序列
    MGF文件的TITLE通常包含peptide信息
    """
    # 尝试不同的模式
    import re
    
    # 模式1: 直接包含peptide序列
    match = re.search(r'[A-Z]{6,}', title)
    if match:
        return match.group(0)
    
    # 模式2: 在特定标记后
    for pattern in [r'seq=([A-Z]+)', r'peptide=([A-Z]+)', r'_([A-Z]{6,})_']:
        match = re.search(pattern, title, re.IGNORECASE)
        if match:
            return match.group(1)
    
    return None


def process_mgf_files(mgf_dir, output_path, min_peaks=10, max_peaks=200, min_peptide_count=2):
    """
    处理MGF文件并创建训练数据
    
    Args:
        mgf_dir: MGF文件目录
        output_path: 输出pickle文件路径
        min_peaks: 最小peak数量
        max_peaks: 最大peak数量
        min_peptide_count: 每个peptide最少需要的spectrum数量
    """
    mgf_dir = Path(mgf_dir)
    
    # 查找所有MGF文件
    mgf_files = list(mgf_dir.glob('**/*.mgf'))
    print(f"Found {len(mgf_files)} MGF files")
    
    if not mgf_files:
        print(f"No MGF files found in {mgf_dir}")
        return
    
    # 解析所有MGF文件
    all_spectra = []
    peptide_to_spectra = defaultdict(list)
    
    for mgf_file in tqdm(mgf_files, desc="Parsing MGF files"):
        try:
            spectra = parse_mgf_file(mgf_file)
            print(f"  {mgf_file.name}: {len(spectra)} spectra")
            
            for spectrum in spectra:
                # 提取peptide
                peptide = spectrum.get('peptide')
                if not peptide and 'title' in spectrum:
                    peptide = extract_peptide_from_title(spectrum['title'])
                
                if peptide:
                    # 清理peptide序列
                    peptide = peptide.upper().strip()
                    # 移除常见修饰标记
                    peptide = peptide.replace('(OX)', '').replace('[', '').replace(']', '')
                    
                    # 过滤spectrum
                    n_peaks = len(spectrum['mz_array'])
                    if min_peaks <= n_peaks <= max_peaks:
                        spectrum['peptide'] = peptide
                        spectrum['source_file'] = mgf_file.name
                        peptide_to_spectra[peptide].append(spectrum)
                        all_spectra.append(spectrum)
        
        except Exception as e:
            print(f"  Error parsing {mgf_file.name}: {e}")
    
    print(f"\nTotal spectra parsed: {len(all_spectra)}")
    print(f"Unique peptides: {len(peptide_to_spectra)}")
    
    # 过滤：只保留有足够spectrum的peptide
    filtered_peptides = {
        peptide: spectra 
        for peptide, spectra in peptide_to_spectra.items() 
        if len(spectra) >= min_peptide_count
    }
    
    print(f"Peptides with >= {min_peptide_count} spectra: {len(filtered_peptides)}")
    
    # 统计信息
    peptide_counts = {peptide: len(spectra) for peptide, spectra in filtered_peptides.items()}
    print(f"\nPeptide statistics:")
    print(f"  Min spectra per peptide: {min(peptide_counts.values())}")
    print(f"  Max spectra per peptide: {max(peptide_counts.values())}")
    print(f"  Mean spectra per peptide: {np.mean(list(peptide_counts.values())):.2f}")
    
    # 准备训练数据
    spectra_data = []
    peptide_labels = []
    
    for peptide, spectra_list in filtered_peptides.items():
        for spectrum in spectra_list:
            # 创建spectrum数组 (n_peaks, 2) - [m/z, intensity]
            spectrum_array = np.column_stack([
                spectrum['mz_array'],
                spectrum['intensity_array']
            ]).astype(np.float32)
            
            spectra_data.append(spectrum_array)
            peptide_labels.append(peptide)
    
    print(f"\nFinal dataset:")
    print(f"  Total spectra: {len(spectra_data)}")
    print(f"  Unique peptides: {len(set(peptide_labels))}")
    
    # 保存数据
    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    
    data = {
        'spectra': spectra_data,
        'peptides': peptide_labels,
        'metadata': {
            'n_spectra': len(spectra_data),
            'n_peptides': len(set(peptide_labels)),
            'peptide_counts': peptide_counts,
            'source_files': [mgf_file.name for mgf_file in mgf_files]
        }
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"\n✓ Data saved to: {output_path}")
    
    # 打印一些示例
    print(f"\nExample spectra:")
    for i in range(min(3, len(spectra_data))):
        print(f"  Spectrum {i+1}:")
        print(f"    Peptide: {peptide_labels[i]}")
        print(f"    Shape: {spectra_data[i].shape}")
        print(f"    m/z range: [{spectra_data[i][:, 0].min():.2f}, {spectra_data[i][:, 0].max():.2f}]")
        print(f"    Intensity range: [{spectra_data[i][:, 1].min():.4f}, {spectra_data[i][:, 1].max():.4f}]")
    
    return data


def analyze_dataset(pickle_path):
    """分析已创建的数据集"""
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    
    spectra = data['spectra']
    peptides = data['peptides']
    
    print("="*60)
    print("Dataset Analysis")
    print("="*60)
    
    print(f"\nBasic Statistics:")
    print(f"  Total spectra: {len(spectra)}")
    print(f"  Unique peptides: {len(set(peptides))}")
    
    # Spectrum长度分布
    lengths = [len(s) for s in spectra]
    print(f"\nSpectrum length (number of peaks):")
    print(f"  Min: {min(lengths)}")
    print(f"  Max: {max(lengths)}")
    print(f"  Mean: {np.mean(lengths):.2f}")
    print(f"  Median: {np.median(lengths):.2f}")
    
    # Peptide分布
    from collections import Counter
    peptide_counts = Counter(peptides)
    print(f"\nPeptide distribution:")
    print(f"  Peptides with 2 spectra: {sum(1 for c in peptide_counts.values() if c == 2)}")
    print(f"  Peptides with 3-5 spectra: {sum(1 for c in peptide_counts.values() if 3 <= c <= 5)}")
    print(f"  Peptides with 6-10 spectra: {sum(1 for c in peptide_counts.values() if 6 <= c <= 10)}")
    print(f"  Peptides with >10 spectra: {sum(1 for c in peptide_counts.values() if c > 10)}")
    
    # Top peptides
    print(f"\nTop 10 peptides by spectrum count:")
    for peptide, count in peptide_counts.most_common(10):
        print(f"  {peptide}: {count} spectra")
    
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Prepare training data from MGF files')
    parser.add_argument('--mgf_dir', type=str, default='../testdata',
                       help='Directory containing MGF files')
    parser.add_argument('--output', type=str, default='data/training_data.pkl',
                       help='Output pickle file path')
    parser.add_argument('--min_peaks', type=int, default=10,
                       help='Minimum number of peaks per spectrum')
    parser.add_argument('--max_peaks', type=int, default=200,
                       help='Maximum number of peaks per spectrum')
    parser.add_argument('--min_peptide_count', type=int, default=2,
                       help='Minimum number of spectra per peptide')
    parser.add_argument('--analyze', action='store_true',
                       help='Analyze existing dataset')
    
    args = parser.parse_args()
    
    if args.analyze:
        analyze_dataset(args.output)
    else:
        process_mgf_files(
            mgf_dir=args.mgf_dir,
            output_path=args.output,
            min_peaks=args.min_peaks,
            max_peaks=args.max_peaks,
            min_peptide_count=args.min_peptide_count
        )


if __name__ == '__main__':
    main()