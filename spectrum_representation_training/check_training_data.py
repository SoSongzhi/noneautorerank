"""
检查训练数据的来源和统计信息
"""
import pickle
from pathlib import Path
from collections import Counter

def check_training_data():
    """检查训练数据"""
    data_path = Path(__file__).parent / 'data' / 'training_data.pkl'
    
    if not data_path.exists():
        print(f"❌ 训练数据文件不存在: {data_path}")
        print("\n需要先准备训练数据:")
        print("  python scripts/prepare_data_from_mgf.py --mgf_file <your_mgf> --output data/training_data.pkl")
        return
    
    print("=" * 70)
    print("检查训练数据")
    print("=" * 70)
    
    # 加载数据
    print(f"\n加载数据: {data_path}")
    print(f"文件大小: {data_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"✓ 数据加载成功")
    
    # 检查数据结构
    print(f"\n数据类型: {type(data)}")
    
    if isinstance(data, list):
        print(f"总样本数: {len(data)}")
        
        if len(data) > 0:
            print(f"\n第一个样本的结构:")
            sample = data[0]
            print(f"  类型: {type(sample)}")
            if isinstance(sample, dict):
                print(f"  键: {list(sample.keys())}")
                for key, value in sample.items():
                    if hasattr(value, 'shape'):
                        print(f"    {key}: shape={value.shape}, dtype={value.dtype}")
                    else:
                        print(f"    {key}: {type(value)} = {value}")
        
        # 统计肽段
        print(f"\n统计肽段信息...")
        peptides = []
        spectrum_sources = []
        
        for item in data:
            if isinstance(item, dict):
                if 'peptide' in item:
                    peptides.append(item['peptide'])
                if 'source' in item:
                    spectrum_sources.append(item['source'])
                elif 'file' in item:
                    spectrum_sources.append(item['file'])
        
        if peptides:
            unique_peptides = set(peptides)
            print(f"  总谱图数: {len(peptides)}")
            print(f"  独特肽段数: {len(unique_peptides)}")
            print(f"  平均每个肽段的谱图数: {len(peptides) / len(unique_peptides):.2f}")
            
            # 肽段长度分布
            peptide_lengths = [len(p) for p in peptides]
            print(f"\n  肽段长度统计:")
            print(f"    最短: {min(peptide_lengths)}")
            print(f"    最长: {max(peptide_lengths)}")
            print(f"    平均: {sum(peptide_lengths) / len(peptide_lengths):.2f}")
            
            # 最常见的肽段
            peptide_counts = Counter(peptides)
            print(f"\n  最常见的10个肽段:")
            for pep, count in peptide_counts.most_common(10):
                print(f"    {pep}: {count} 个谱图")
        
        if spectrum_sources:
            source_counts = Counter(spectrum_sources)
            print(f"\n  数据来源:")
            for source, count in source_counts.most_common():
                print(f"    {source}: {count} 个谱图")
    
    elif isinstance(data, tuple) and len(data) == 2:
        spectra_data, peptide_labels = data
        print(f"总样本数: {len(spectra_data)}")
        print(f"独特肽段数: {len(set(peptide_labels))}")
        
        # 统计肽段
        peptide_counts = Counter(peptide_labels)
        print(f"\n最常见的10个肽段:")
        for pep, count in peptide_counts.most_common(10):
            print(f"  {pep}: {count} 个谱图")
    
    print("\n" + "=" * 70)
    print("检查完成")
    print("=" * 70)


def check_nine_species_data():
    """检查九个物种的数据"""
    print("\n" + "=" * 70)
    print("检查九个物种的MGF文件")
    print("=" * 70)
    
    nine_species_dir = Path(__file__).parent.parent / '9spicies'
    
    if not nine_species_dir.exists():
        print(f"❌ 九物种目录不存在: {nine_species_dir}")
        return
    
    mgf_files = list(nine_species_dir.glob("*.mgf"))
    
    if not mgf_files:
        print(f"❌ 没有找到MGF文件")
        return
    
    print(f"\n找到 {len(mgf_files)} 个MGF文件:")
    
    total_size = 0
    for mgf_file in sorted(mgf_files):
        size_mb = mgf_file.stat().st_size / 1024 / 1024
        total_size += size_mb
        print(f"  - {mgf_file.name}: {size_mb:.2f} MB")
    
    print(f"\n总大小: {total_size:.2f} MB")
    
    print("\n如果要使用所有九个物种的数据训练，需要:")
    print("1. 合并所有MGF文件:")
    print("   cat 9spicies/*.mgf > 9spicies/all_species.mgf")
    print("\n2. 准备训练数据:")
    print("   python scripts/prepare_data_from_mgf.py \\")
    print("       --mgf_file 9spicies/all_species.mgf \\")
    print("       --output data/training_data.pkl")
    print("\n3. 开始训练:")
    print("   python scripts/train.py --config config.yaml")


if __name__ == '__main__':
    check_training_data()
    check_nine_species_data()