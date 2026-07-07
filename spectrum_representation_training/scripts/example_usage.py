"""
Example Usage Script
演示如何使用训练好的模型
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch
import numpy as np
from spectrum_representation_training.models import ContrastiveSpectrumModel


def example_1_encode_single_spectrum():
    """示例1: 编码单个spectrum"""
    print("="*60)
    print("Example 1: Encode a single spectrum")
    print("="*60)
    
    # 加载模型
    model = ContrastiveSpectrumModel.load_from_checkpoint(
        'checkpoints/best_model.pth',
        device='cuda'
    )
    model.eval()
    
    # 创建示例spectrum (n_peaks, 2) - [m/z, intensity]
    spectrum = np.array([
        [100.0, 0.5],
        [200.0, 1.0],
        [300.0, 0.8],
        [400.0, 0.3],
    ], dtype=np.float32)
    
    # 转换为tensor
    spectrum_tensor = torch.FloatTensor(spectrum)
    
    # 编码
    with torch.no_grad():
        embedding = model.encode(spectrum_tensor)
    
    print(f"Spectrum shape: {spectrum.shape}")
    print(f"Embedding shape: {embedding.shape}")
    print(f"Embedding (first 10 dims): {embedding[:10].cpu().numpy()}")
    print()


def example_2_compute_similarity():
    """示例2: 计算两个spectrum的相似度"""
    print("="*60)
    print("Example 2: Compute similarity between two spectra")
    print("="*60)
    
    # 加载模型
    model = ContrastiveSpectrumModel.load_from_checkpoint(
        'checkpoints/best_model.pth',
        device='cuda'
    )
    model.eval()
    
    # 创建两个示例spectrum
    spectrum1 = torch.FloatTensor(np.random.randn(50, 2).astype(np.float32))
    spectrum2 = torch.FloatTensor(np.random.randn(45, 2).astype(np.float32))
    
    # 编码
    with torch.no_grad():
        emb1 = model.encode(spectrum1)
        emb2 = model.encode(spectrum2)
    
    # 计算余弦相似度
    similarity = torch.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0))
    
    print(f"Spectrum 1 shape: {spectrum1.shape}")
    print(f"Spectrum 2 shape: {spectrum2.shape}")
    print(f"Cosine similarity: {similarity.item():.4f}")
    print()


def example_3_batch_encoding():
    """示例3: 批量编码多个spectrum"""
    print("="*60)
    print("Example 3: Batch encode multiple spectra")
    print("="*60)
    
    # 加载模型
    model = ContrastiveSpectrumModel.load_from_checkpoint(
        'checkpoints/best_model.pth',
        device='cuda'
    )
    model.eval()
    
    # 创建多个spectrum
    spectra_list = [
        np.random.randn(50, 2).astype(np.float32),
        np.random.randn(45, 2).astype(np.float32),
        np.random.randn(60, 2).astype(np.float32),
        np.random.randn(40, 2).astype(np.float32),
    ]
    
    # 转换为tensor list
    spectra_tensors = [torch.FloatTensor(s) for s in spectra_list]
    
    # 批量编码
    embeddings = model.encode_batch(spectra_tensors, batch_size=2, device='cuda')
    
    print(f"Number of spectra: {len(spectra_list)}")
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"First embedding (first 10 dims): {embeddings[0, :10].numpy()}")
    print()


def example_4_spectrum_search():
    """示例4: Spectrum检索"""
    print("="*60)
    print("Example 4: Spectrum search in database")
    print("="*60)
    
    # 加载模型
    model = ContrastiveSpectrumModel.load_from_checkpoint(
        'checkpoints/best_model.pth',
        device='cuda'
    )
    model.eval()
    
    # 创建数据库（100个spectrum）
    print("Creating database...")
    database_spectra = [
        np.random.randn(np.random.randint(40, 60), 2).astype(np.float32)
        for _ in range(100)
    ]
    database_tensors = [torch.FloatTensor(s) for s in database_spectra]
    
    # 编码数据库
    print("Encoding database...")
    database_embeddings = model.encode_batch(database_tensors, batch_size=32, device='cuda')
    
    # 创建query spectrum
    query_spectrum = torch.FloatTensor(np.random.randn(50, 2).astype(np.float32))
    
    # 编码query
    with torch.no_grad():
        query_embedding = model.encode(query_spectrum).unsqueeze(0)
    
    # 计算相似度
    similarities = torch.cosine_similarity(
        query_embedding.cpu(), 
        database_embeddings, 
        dim=1
    )
    
    # 获取Top-K
    k = 5
    top_k_values, top_k_indices = torch.topk(similarities, k=k)
    
    print(f"Query spectrum shape: {query_spectrum.shape}")
    print(f"Database size: {len(database_spectra)}")
    print(f"\nTop-{k} most similar spectra:")
    for i, (idx, sim) in enumerate(zip(top_k_indices, top_k_values)):
        print(f"  {i+1}. Index {idx.item()}: similarity = {sim.item():.4f}")
    print()


def example_5_clustering():
    """示例5: Spectrum聚类"""
    print("="*60)
    print("Example 5: Spectrum clustering")
    print("="*60)
    
    try:
        from sklearn.cluster import KMeans
    except ImportError:
        print("scikit-learn not installed. Skipping clustering example.")
        return
    
    # 加载模型
    model = ContrastiveSpectrumModel.load_from_checkpoint(
        'checkpoints/best_model.pth',
        device='cuda'
    )
    model.eval()
    
    # 创建spectrum数据
    print("Creating spectra...")
    spectra = [
        np.random.randn(np.random.randint(40, 60), 2).astype(np.float32)
        for _ in range(200)
    ]
    spectra_tensors = [torch.FloatTensor(s) for s in spectra]
    
    # 编码
    print("Encoding spectra...")
    embeddings = model.encode_batch(spectra_tensors, batch_size=32, device='cuda')
    embeddings_np = embeddings.numpy()
    
    # 聚类
    print("Clustering...")
    n_clusters = 10
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(embeddings_np)
    
    # 统计每个cluster的大小
    from collections import Counter
    cluster_counts = Counter(clusters)
    
    print(f"Number of spectra: {len(spectra)}")
    print(f"Number of clusters: {n_clusters}")
    print(f"Cluster distribution:")
    for cluster_id in sorted(cluster_counts.keys()):
        print(f"  Cluster {cluster_id}: {cluster_counts[cluster_id]} spectra")
    print()


def example_6_quality_control():
    """示例6: 质量控制 - 检查同一peptide的多次测量一致性"""
    print("="*60)
    print("Example 6: Quality control for replicate measurements")
    print("="*60)
    
    # 加载模型
    model = ContrastiveSpectrumModel.load_from_checkpoint(
        'checkpoints/best_model.pth',
        device='cuda'
    )
    model.eval()
    
    # 模拟同一peptide的5次测量
    print("Simulating 5 replicate measurements of the same peptide...")
    base_spectrum = np.random.randn(50, 2).astype(np.float32)
    
    # 添加小的变化模拟实验误差
    replicates = [
        base_spectrum + np.random.randn(50, 2).astype(np.float32) * 0.1
        for _ in range(5)
    ]
    
    replicate_tensors = [torch.FloatTensor(s) for s in replicates]
    
    # 编码
    embeddings = model.encode_batch(replicate_tensors, batch_size=5, device='cuda')
    
    # 计算两两相似度
    similarities = []
    for i in range(len(embeddings)):
        for j in range(i+1, len(embeddings)):
            sim = torch.cosine_similarity(
                embeddings[i].unsqueeze(0),
                embeddings[j].unsqueeze(0)
            ).item()
            similarities.append(sim)
    
    avg_similarity = np.mean(similarities)
    std_similarity = np.std(similarities)
    
    print(f"Number of replicates: {len(replicates)}")
    print(f"Average pairwise similarity: {avg_similarity:.4f} ± {std_similarity:.4f}")
    
    # 质量判断
    threshold = 0.7
    if avg_similarity >= threshold:
        print(f"✓ Quality PASS: Replicates are consistent (>= {threshold})")
    else:
        print(f"✗ Quality FAIL: Replicates are inconsistent (< {threshold})")
    print()


def main():
    """运行所有示例"""
    print("\n" + "="*60)
    print("Spectrum Representation Model - Usage Examples")
    print("="*60 + "\n")
    
    try:
        example_1_encode_single_spectrum()
        example_2_compute_similarity()
        example_3_batch_encoding()
        example_4_spectrum_search()
        example_5_clustering()
        example_6_quality_control()
        
        print("="*60)
        print("All examples completed successfully!")
        print("="*60)
        
    except FileNotFoundError:
        print("\n⚠️  Model checkpoint not found!")
        print("Please train a model first using:")
        print("  python scripts/train.py --config config.yaml")
        print("\nOr update the checkpoint path in this script.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()