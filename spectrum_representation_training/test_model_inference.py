"""
测试训练好的模型 - 生成spectrum的embedding表示
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from pyteomics import mgf
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from spectrum_representation_training.models import ContrastiveSpectrumModel


def load_model(checkpoint_path, device='cuda'):
    """加载训练好的模型"""
    print(f"加载模型: {checkpoint_path}")
    
    if not Path(checkpoint_path).exists():
        print(f"❌ 模型文件不存在: {checkpoint_path}")
        print("\n可用的模型文件:")
        checkpoint_dir = Path(checkpoint_path).parent
        if checkpoint_dir.exists():
            for f in checkpoint_dir.glob("*.pth"):
                print(f"  - {f}")
        return None
    
    model = ContrastiveSpectrumModel.load_from_checkpoint(checkpoint_path, device=device)
    model.eval()
    print(f"✓ 模型加载成功")
    print(f"  - Encoder维度: {model.encoder_dim}")
    print(f"  - Embedding维度: {model.embedding_dim}")
    return model


def load_spectrum_from_mgf(mgf_path, max_spectra=None):
    """从MGF文件加载谱图"""
    print(f"\n加载谱图: {mgf_path}")
    
    spectra = []
    peptides = []
    
    with mgf.read(mgf_path) as reader:
        for i, spectrum in enumerate(reader):
            if max_spectra and i >= max_spectra:
                break
            
            # 提取m/z和intensity
            mz = spectrum['m/z array']
            intensity = spectrum['intensity array']
            
            # 归一化intensity
            intensity = intensity / (intensity.max() + 1e-9)
            
            # 组合成(n_peaks, 2)
            spec_array = np.stack([mz, intensity], axis=1)
            spectra.append(spec_array)
            
            # 提取肽段序列
            peptide = spectrum['params'].get('seq', f'unknown_{i}')
            peptides.append(peptide)
    
    print(f"✓ 加载了 {len(spectra)} 个谱图")
    return spectra, peptides


def encode_spectra(model, spectra, batch_size=32, device='cuda'):
    """批量编码谱图"""
    print(f"\n编码谱图...")
    
    all_embeddings = []
    
    with torch.no_grad():
        for i in range(0, len(spectra), batch_size):
            batch = spectra[i:i+batch_size]
            
            # 转换为tensor并padding
            max_peaks = max(s.shape[0] for s in batch)
            padded_batch = []
            
            for spec in batch:
                if spec.shape[0] < max_peaks:
                    padding = np.zeros((max_peaks - spec.shape[0], 2))
                    spec = np.vstack([spec, padding])
                padded_batch.append(spec)
            
            batch_tensor = torch.FloatTensor(np.stack(padded_batch)).to(device)
            
            # 编码
            embeddings = model(batch_tensor)
            all_embeddings.append(embeddings.cpu().numpy())
            
            if (i // batch_size + 1) % 10 == 0:
                print(f"  处理了 {i + len(batch)}/{len(spectra)} 个谱图")
    
    embeddings = np.vstack(all_embeddings)
    print(f"✓ 编码完成，shape: {embeddings.shape}")
    return embeddings


def compute_similarity_matrix(embeddings):
    """计算相似度矩阵"""
    print(f"\n计算相似度矩阵...")
    
    # 余弦相似度 (embeddings已经L2归一化)
    similarity = embeddings @ embeddings.T
    
    print(f"✓ 相似度矩阵 shape: {similarity.shape}")
    print(f"  - 最小值: {similarity.min():.4f}")
    print(f"  - 最大值: {similarity.max():.4f}")
    print(f"  - 平均值: {similarity.mean():.4f}")
    
    return similarity


def analyze_embeddings(embeddings, peptides):
    """分析embedding质量"""
    print(f"\n分析Embedding质量...")
    
    # 按肽段分组
    peptide_to_indices = {}
    for i, pep in enumerate(peptides):
        if pep not in peptide_to_indices:
            peptide_to_indices[pep] = []
        peptide_to_indices[pep].append(i)
    
    # 计算同肽段和不同肽段的相似度
    same_peptide_sims = []
    diff_peptide_sims = []
    
    for pep, indices in peptide_to_indices.items():
        if len(indices) < 2:
            continue
        
        # 同肽段的相似度
        for i in range(len(indices)):
            for j in range(i+1, len(indices)):
                idx1, idx2 = indices[i], indices[j]
                sim = np.dot(embeddings[idx1], embeddings[idx2])
                same_peptide_sims.append(sim)
        
        # 不同肽段的相似度（采样）
        other_indices = [i for i in range(len(peptides)) if i not in indices]
        if len(other_indices) > 0:
            for idx1 in indices[:min(5, len(indices))]:
                for idx2 in np.random.choice(other_indices, min(10, len(other_indices)), replace=False):
                    sim = np.dot(embeddings[idx1], embeddings[idx2])
                    diff_peptide_sims.append(sim)
    
    if same_peptide_sims:
        print(f"\n相似度统计:")
        print(f"  同肽段:")
        print(f"    - 平均: {np.mean(same_peptide_sims):.4f}")
        print(f"    - 标准差: {np.std(same_peptide_sims):.4f}")
        print(f"    - 最小: {np.min(same_peptide_sims):.4f}")
        print(f"    - 最大: {np.max(same_peptide_sims):.4f}")
    
    if diff_peptide_sims:
        print(f"  不同肽段:")
        print(f"    - 平均: {np.mean(diff_peptide_sims):.4f}")
        print(f"    - 标准差: {np.std(diff_peptide_sims):.4f}")
        print(f"    - 最小: {np.min(diff_peptide_sims):.4f}")
        print(f"    - 最大: {np.max(diff_peptide_sims):.4f}")
    
    if same_peptide_sims and diff_peptide_sims:
        separation = np.mean(same_peptide_sims) - np.mean(diff_peptide_sims)
        print(f"\n  分离度: {separation:.4f}")
        
        if separation > 0.5:
            print(f"  ✓ 分离度很好！")
        elif separation > 0.3:
            print(f"  ⚠ 分离度一般")
        else:
            print(f"  ❌ 分离度较差")
    
    return same_peptide_sims, diff_peptide_sims


def visualize_embeddings(embeddings, peptides, output_path='embedding_visualization.png'):
    """可视化embeddings"""
    print(f"\n可视化Embeddings...")
    
    # 使用t-SNE降维到2D
    print("  运行t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # 为每个肽段分配颜色
    unique_peptides = list(set(peptides))
    peptide_to_color = {pep: i for i, pep in enumerate(unique_peptides)}
    colors = [peptide_to_color[pep] for pep in peptides]
    
    # 绘图
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                         c=colors, cmap='tab20', alpha=0.6, s=50)
    
    # 只显示前10个肽段的图例
    if len(unique_peptides) <= 10:
        handles = [plt.Line2D([0], [0], marker='o', color='w', 
                             markerfacecolor=plt.cm.tab20(peptide_to_color[pep]/len(unique_peptides)), 
                             markersize=8, label=pep) 
                  for pep in unique_peptides[:10]]
        plt.legend(handles=handles, loc='best', fontsize=8)
    
    plt.title('Spectrum Embeddings Visualization (t-SNE)', fontsize=14)
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 可视化保存到: {output_path}")
    plt.close()


def test_retrieval(embeddings, peptides, query_idx=0, top_k=10):
    """测试检索功能"""
    print(f"\n测试检索功能...")
    print(f"查询谱图索引: {query_idx}")
    print(f"查询肽段: {peptides[query_idx]}")
    
    # 计算与所有谱图的相似度
    query_emb = embeddings[query_idx]
    similarities = embeddings @ query_emb
    
    # 排序（排除自己）
    sorted_indices = np.argsort(similarities)[::-1]
    sorted_indices = [i for i in sorted_indices if i != query_idx]
    
    print(f"\nTop-{top_k} 最相似的谱图:")
    print(f"{'排名':<6} {'索引':<8} {'相似度':<10} {'肽段':<20} {'匹配'}")
    print("-" * 70)
    
    correct = 0
    for rank, idx in enumerate(sorted_indices[:top_k], 1):
        sim = similarities[idx]
        pep = peptides[idx]
        match = "✓" if pep == peptides[query_idx] else "✗"
        if match == "✓":
            correct += 1
        print(f"{rank:<6} {idx:<8} {sim:<10.4f} {pep:<20} {match}")
    
    print(f"\nTop-{top_k} 准确率: {correct}/{top_k} = {correct/top_k*100:.1f}%")
    
    return correct / top_k


def save_embeddings(embeddings, peptides, output_path='embeddings.npz'):
    """保存embeddings"""
    print(f"\n保存Embeddings到: {output_path}")
    np.savez(output_path, 
             embeddings=embeddings,
             peptides=np.array(peptides))
    print(f"✓ 保存成功")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='测试训练好的模型')
    parser.add_argument('--model', type=str, required=True,
                       help='模型checkpoint路径')
    parser.add_argument('--mgf', type=str, required=True,
                       help='MGF文件路径')
    parser.add_argument('--max_spectra', type=int, default=1000,
                       help='最多加载多少个谱图')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批处理大小')
    parser.add_argument('--device', type=str, default='cuda',
                       help='设备 (cuda/cpu)')
    parser.add_argument('--output_dir', type=str, default='test_results',
                       help='输出目录')
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("=" * 70)
    print("测试Spectrum Embedding模型")
    print("=" * 70)
    
    # 1. 加载模型
    model = load_model(args.model, device=args.device)
    if model is None:
        return
    
    # 2. 加载谱图
    spectra, peptides = load_spectrum_from_mgf(args.mgf, max_spectra=args.max_spectra)
    if len(spectra) == 0:
        print("❌ 没有加载到谱图")
        return
    
    # 3. 编码谱图
    embeddings = encode_spectra(model, spectra, batch_size=args.batch_size, device=args.device)
    
    # 4. 计算相似度矩阵
    similarity_matrix = compute_similarity_matrix(embeddings)
    
    # 5. 分析embedding质量
    same_sims, diff_sims = analyze_embeddings(embeddings, peptides)
    
    # 6. 可视化
    if len(embeddings) >= 10:
        visualize_embeddings(embeddings, peptides, 
                           output_path=output_dir / 'embedding_visualization.png')
    
    # 7. 测试检索
    if len(embeddings) >= 10:
        test_retrieval(embeddings, peptides, query_idx=0, top_k=10)
    
    # 8. 保存结果
    save_embeddings(embeddings, peptides, 
                   output_path=output_dir / 'embeddings.npz')
    
    print("\n" + "=" * 70)
    print("测试完成！")
    print("=" * 70)
    print(f"\n结果保存在: {output_dir}")
    print(f"  - embeddings.npz: Embedding向量")
    print(f"  - embedding_visualization.png: 可视化图")


if __name__ == '__main__':
    main()