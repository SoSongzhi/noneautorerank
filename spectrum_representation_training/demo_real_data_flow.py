# -*- coding: utf-8 -*-
"""
演示真实数据流动过程 - 使用真实MGF文件
展示真实的tensor形状和数值
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple
import sys
import io
from pathlib import Path
import pickle

# 设置UTF-8编码输出
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("=" * 80)
print("Real Spectrum Data Flow Demo - Using Real MGF Data")
print("=" * 80)

# ============================================================================
# Step 0: Load Real Data
# ============================================================================
print("\n" + "=" * 80)
print("Step 0: Loading Real Data")
print("=" * 80)

# 尝试找到真实的数据文件
data_paths = [
    Path("data/training_data.pkl"),
    Path("../data/training_data.pkl"),
    Path("spectrum_representation_training/data/training_data.pkl"),
]

data_file = None
for path in data_paths:
    if path.exists():
        data_file = path
        break

if data_file is None:
    print("\n[WARNING] No training_data.pkl found. Will use MGF file instead.")
    print("\nSearching for MGF files...")
    
    mgf_paths = [
        Path("../9spicies/human.mgf"),
        Path("../testdata/human.mgf"),
        Path("9spicies/human.mgf"),
        Path("testdata/human.mgf"),
    ]
    
    mgf_file = None
    for path in mgf_paths:
        if path.exists():
            mgf_file = path
            break
    
    if mgf_file is None:
        print("\n[ERROR] No MGF file found either!")
        print("\nPlease provide path to your MGF file:")
        print("  python demo_real_data_flow.py --mgf <path_to_mgf>")
        sys.exit(1)
    
    print(f"\n[FOUND] MGF file: {mgf_file}")
    print("\nParsing MGF file (loading first 10 spectra)...")
    
    # 简单的MGF解析器
    from pyteomics import mgf
    
    spectra_list = []
    peptides_list = []
    mz_list = []
    charge_list = []
    
    with mgf.read(str(mgf_file)) as reader:
        for i, spectrum in enumerate(reader):
            if i >= 10:  # 只加载前10个
                break
            
            # 提取谱图数据
            mz_array = spectrum['m/z array']
            intensity_array = spectrum['intensity array']
            
            # 归一化强度
            intensity_array = intensity_array / intensity_array.max()
            
            # 组合成(n_peaks, 2)的数组
            spec_array = np.column_stack([mz_array, intensity_array])
            
            # 提取肽段序列
            peptide = spectrum['params'].get('seq', spectrum['params'].get('title', f'UNKNOWN_{i}'))
            
            # 提取前体离子信息
            precursor_mz = spectrum['params'].get('pepmass', [0])[0] if isinstance(spectrum['params'].get('pepmass'), list) else spectrum['params'].get('pepmass', 0)
            precursor_charge = spectrum['params'].get('charge', [2])[0] if isinstance(spectrum['params'].get('charge'), list) else spectrum['params'].get('charge', 2)
            
            spectra_list.append(spec_array)
            peptides_list.append(peptide)
            mz_list.append(precursor_mz)
            charge_list.append(precursor_charge)
    
    raw_data = {
        'spectra': spectra_list,
        'peptides': peptides_list,
        'precursor_mz': mz_list,
        'precursor_charge': charge_list
    }
    
    print(f"\n[SUCCESS] Loaded {len(spectra_list)} spectra from MGF")

else:
    print(f"\n[FOUND] Training data: {data_file}")
    print("Loading PKL file...")
    
    with open(data_file, 'rb') as f:
        raw_data = pickle.load(f)
    
    print(f"\n[SUCCESS] Loaded {len(raw_data['spectra'])} spectra from PKL")

# ============================================================================
# Step 1: Display Real Data
# ============================================================================
print("\n" + "=" * 80)
print("Step 1: Real Data Overview")
print("=" * 80)

print(f"\nTotal spectra: {len(raw_data['spectra'])}")
print(f"Peptides: {raw_data['peptides'][:6]}")  # 显示前6个

print(f"\n[REAL DATA] Number of peaks per spectrum:")

# 检查PKL文件的键
print(f"\nPKL file keys: {list(raw_data.keys())}")

# 根据实际的键来访问数据
has_mz = 'precursor_mz' in raw_data
has_charge = 'precursor_charge' in raw_data

for i in range(min(6, len(raw_data['spectra']))):
    spec = raw_data['spectra'][i]
    peptide = raw_data['peptides'][i]
    print(f"\n  Spectrum {i} ({peptide}):")
    print(f"    Total peaks: {len(spec)}")
    
    if has_mz:
        print(f"    Precursor m/z: {raw_data['precursor_mz'][i]:.4f}")
    if has_charge:
        print(f"    Charge: {raw_data['precursor_charge'][i]}")
    
    print(f"    First 5 peaks (m/z, intensity):")
    for j, peak in enumerate(spec[:5]):
        print(f"      Peak {j}: m/z={peak[0]:.4f}, intensity={peak[1]:.4f}")

# 如果没有precursor信息，添加默认值
if not has_mz:
    print("\n[INFO] No precursor_mz in PKL, using default values")
    raw_data['precursor_mz'] = [0.0] * len(raw_data['spectra'])
if not has_charge:
    print("[INFO] No precursor_charge in PKL, using default values")
    raw_data['precursor_charge'] = [2] * len(raw_data['spectra'])

# ============================================================================
# Step 2: Triplet Sampling with Real Data
# ============================================================================
print("\n" + "=" * 80)
print("Step 2: Triplet Sampling (using real data)")
print("=" * 80)

def sample_triplet_real(idx: int, data: Dict) -> Tuple:
    """使用真实数据进行triplet采样"""
    # Anchor
    anchor_spectrum = data['spectra'][idx]
    anchor_peptide = data['peptides'][idx]
    anchor_mz = data['precursor_mz'][idx]
    anchor_charge = data['precursor_charge'][idx]
    
    print(f"\n[ANCHOR] (index {idx}):")
    print(f"   Peptide: {anchor_peptide}")
    print(f"   Spectrum shape: {anchor_spectrum.shape}")
    print(f"   Precursor m/z: {anchor_mz:.4f}, charge: {anchor_charge}")
    print(f"   First 5 peaks:")
    for j, peak in enumerate(anchor_spectrum[:5]):
        print(f"     Peak {j}: m/z={peak[0]:.4f}, intensity={peak[1]:.4f}")
    
    # Positive: 找相同肽段的其他谱图
    positive_indices = [i for i, p in enumerate(data['peptides']) 
                       if p == anchor_peptide and i != idx]
    
    if len(positive_indices) == 0:
        print(f"\n   [WARNING] No other spectra with same peptide, using same spectrum as positive")
        pos_idx = idx
    else:
        pos_idx = np.random.choice(positive_indices)
    
    positive_spectrum = data['spectra'][pos_idx]
    positive_mz = data['precursor_mz'][pos_idx]
    positive_charge = data['precursor_charge'][pos_idx]
    
    print(f"\n[POSITIVE] (index {pos_idx}):")
    print(f"   Peptide: {data['peptides'][pos_idx]} {'(same as anchor)' if pos_idx != idx else '(same spectrum)'}")
    print(f"   Spectrum shape: {positive_spectrum.shape}")
    print(f"   Precursor m/z: {positive_mz:.4f}, charge: {positive_charge}")
    print(f"   First 5 peaks:")
    for j, peak in enumerate(positive_spectrum[:5]):
        print(f"     Peak {j}: m/z={peak[0]:.4f}, intensity={peak[1]:.4f}")
    
    # Negative: 找不同肽段的谱图
    negative_indices = [i for i, p in enumerate(data['peptides']) 
                       if p != anchor_peptide]
    
    if len(negative_indices) == 0:
        print(f"\n   [WARNING] All spectra have same peptide, using random spectrum as negative")
        neg_idx = (idx + 1) % len(data['spectra'])
    else:
        neg_idx = np.random.choice(negative_indices)
    
    negative_spectrum = data['spectra'][neg_idx]
    negative_mz = data['precursor_mz'][neg_idx]
    negative_charge = data['precursor_charge'][neg_idx]
    
    print(f"\n[NEGATIVE] (index {neg_idx}):")
    print(f"   Peptide: {data['peptides'][neg_idx]} (different from anchor)")
    print(f"   Spectrum shape: {negative_spectrum.shape}")
    print(f"   Precursor m/z: {negative_mz:.4f}, charge: {negative_charge}")
    print(f"   First 5 peaks:")
    for j, peak in enumerate(negative_spectrum[:5]):
        print(f"     Peak {j}: m/z={peak[0]:.4f}, intensity={peak[1]:.4f}")
    
    return {
        'anchor': (anchor_spectrum, anchor_mz, anchor_charge),
        'positive': (positive_spectrum, positive_mz, positive_charge),
        'negative': (negative_spectrum, negative_mz, negative_charge),
        'label': anchor_peptide
    }

# 采样第一个样本
sample_0 = sample_triplet_real(0, raw_data)

# 如果有足够的数据，采样第二个样本
if len(raw_data['spectra']) > 2:
    sample_1 = sample_triplet_real(min(2, len(raw_data['spectra']) - 1), raw_data)
    batch_samples = [sample_0, sample_1]
else:
    batch_samples = [sample_0]

# ============================================================================
# Step 3: Collate Function - Create Batch with Real Data
# ============================================================================
print("\n" + "=" * 80)
print("Step 3: Collate Function - Create Batch (Real Data)")
print("=" * 80)

def collate_fn_real(batch_samples: List[Dict]) -> Dict:
    """使用真实数据创建batch"""
    print(f"\nReceived {len(batch_samples)} samples, combining into batch")
    
    # 提取所有谱图
    anchor_spectra = [s['anchor'][0] for s in batch_samples]
    positive_spectra = [s['positive'][0] for s in batch_samples]
    negative_spectra = [s['negative'][0] for s in batch_samples]
    
    # 找到最大峰数
    max_peaks = max(
        max(len(s) for s in anchor_spectra),
        max(len(s) for s in positive_spectra),
        max(len(s) for s in negative_spectra)
    )
    print(f"\nMax peaks in batch: {max_peaks}")
    
    # Padding函数
    def pad_spectrum(spectrum: np.ndarray, max_len: int) -> torch.Tensor:
        """将谱图padding到指定长度"""
        padded = np.zeros((max_len, 2))
        padded[:len(spectrum)] = spectrum
        return torch.FloatTensor(padded)
    
    # Padding所有谱图
    print("\n[PADDING PROCESS]:")
    anchor_tensors = []
    for i, spec in enumerate(anchor_spectra):
        padded = pad_spectrum(spec, max_peaks)
        anchor_tensors.append(padded)
        print(f"  Anchor {i}: {len(spec)} peaks -> padded to {max_peaks} peaks")
    
    positive_tensors = []
    for i, spec in enumerate(positive_spectra):
        padded = pad_spectrum(spec, max_peaks)
        positive_tensors.append(padded)
        print(f"  Positive {i}: {len(spec)} peaks -> padded to {max_peaks} peaks")
    
    negative_tensors = []
    for i, spec in enumerate(negative_spectra):
        padded = pad_spectrum(spec, max_peaks)
        negative_tensors.append(padded)
        print(f"  Negative {i}: {len(spec)} peaks -> padded to {max_peaks} peaks")
    
    # Stack成batch
    batch = {
        'anchor': torch.stack(anchor_tensors),
        'positive': torch.stack(positive_tensors),
        'negative': torch.stack(negative_tensors),
        'anchor_mz': torch.FloatTensor([s['anchor'][1] for s in batch_samples]),
        'anchor_charge': torch.LongTensor([s['anchor'][2] for s in batch_samples]),
        'labels': [s['label'] for s in batch_samples]
    }
    
    print(f"\n[SUCCESS] Batch created:")
    print(f"   anchor shape: {batch['anchor'].shape}")
    print(f"   positive shape: {batch['positive'].shape}")
    print(f"   negative shape: {batch['negative'].shape}")
    
    return batch

# 创建batch
batch = collate_fn_real(batch_samples)

print("\n[BATCH DETAILS - REAL DATA]:")
print(f"\nAnchor batch (first sample, first 5 peaks):")
for i in range(min(5, batch['anchor'].shape[1])):
    print(f"  Peak {i}: m/z={batch['anchor'][0, i, 0]:.4f}, intensity={batch['anchor'][0, i, 1]:.4f}")

print(f"\nPositive batch (first sample, first 5 peaks):")
for i in range(min(5, batch['positive'].shape[1])):
    print(f"  Peak {i}: m/z={batch['positive'][0, i, 0]:.4f}, intensity={batch['positive'][0, i, 1]:.4f}")

print(f"\nNegative batch (first sample, first 5 peaks):")
for i in range(min(5, batch['negative'].shape[1])):
    print(f"  Peak {i}: m/z={batch['negative'][0, i, 0]:.4f}, intensity={batch['negative'][0, i, 1]:.4f}")

# ============================================================================
# Step 4: Model Forward Pass (simplified, using real data)
# ============================================================================
print("\n" + "=" * 80)
print("Step 4: Model Forward Pass (Real Data)")
print("=" * 80)

class SimpleEncoder(nn.Module):
    """简化的编码器"""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 512)
        
    def forward(self, x):
        encoded = self.linear(x)
        cls_token = encoded.mean(dim=1, keepdim=True)
        return torch.cat([cls_token, encoded], dim=1)

class ProjectionHead(nn.Module):
    """投影头"""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(512, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 256)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return F.normalize(x, p=2, dim=1)

encoder = SimpleEncoder()
projection = ProjectionHead()

print("\n[MODEL ARCHITECTURE]:")
print(f"   Encoder: input(batch, peaks, 2) -> output(batch, peaks+1, 512)")
print(f"   Projection: input(batch, 512) -> output(batch, 256)")

print("\n[FORWARD PASS WITH REAL DATA]:")

with torch.no_grad():
    # 编码anchor
    print("\n1. Encoding Anchor (Real Spectrum):")
    anchor_encoded = encoder(batch['anchor'])
    print(f"   Input shape: {batch['anchor'].shape}")
    print(f"   Encoded shape: {anchor_encoded.shape}")
    
    anchor_cls = anchor_encoded[:, 0, :]
    print(f"   CLS token shape: {anchor_cls.shape}")
    print(f"   CLS token first 5 dims: {anchor_cls[0, :5]}")
    
    anchor_emb = projection(anchor_cls)
    print(f"   Projected shape: {anchor_emb.shape}")
    print(f"   Embedding first 5 dims: {anchor_emb[0, :5]}")
    print(f"   Embedding L2 norm: {torch.norm(anchor_emb[0]):.4f}")
    
    # 编码positive
    print("\n2. Encoding Positive (Real Spectrum):")
    positive_encoded = encoder(batch['positive'])
    positive_cls = positive_encoded[:, 0, :]
    positive_emb = projection(positive_cls)
    print(f"   Embedding first 5 dims: {positive_emb[0, :5]}")
    
    # 编码negative
    print("\n3. Encoding Negative (Real Spectrum):")
    negative_encoded = encoder(batch['negative'])
    negative_cls = negative_encoded[:, 0, :]
    negative_emb = projection(negative_cls)
    print(f"   Embedding first 5 dims: {negative_emb[0, :5]}")

# ============================================================================
# Step 5: Calculate Loss (with real data)
# ============================================================================
print("\n" + "=" * 80)
print("Step 5: Calculate Loss (Real Data)")
print("=" * 80)

margin = 0.2
temperature = 0.05

with torch.no_grad():
    # Triplet Loss
    print("\n[5.1 TRIPLET LOSS - REAL DATA]:")
    print("-" * 60)
    
    d_pos = torch.sum((anchor_emb - positive_emb) ** 2, dim=1)
    d_neg = torch.sum((anchor_emb - negative_emb) ** 2, dim=1)
    
    for i in range(len(batch_samples)):
        print(f"\nSample {i}:")
        print(f"   d(anchor, positive) = {d_pos[i].item():.4f}")
        print(f"   d(anchor, negative) = {d_neg[i].item():.4f}")
        loss_i = max(0, d_pos[i].item() - d_neg[i].item() + margin)
        print(f"   loss = max(0, {d_pos[i].item():.4f} - {d_neg[i].item():.4f} + {margin}) = {loss_i:.4f}")
    
    triplet_loss = torch.mean(torch.clamp(d_pos - d_neg + margin, min=0))
    print(f"\n[RESULT] Triplet Loss = {triplet_loss:.4f}")
    
    # Contrastive Loss
    print("\n[5.2 CONTRASTIVE LOSS - REAL DATA]:")
    print("-" * 60)
    
    all_embs = torch.cat([anchor_emb, positive_emb], dim=0)
    sim_matrix = torch.matmul(all_embs, all_embs.T) / temperature
    
    print(f"\nSimilarity matrix shape: {sim_matrix.shape}")
    print(f"Similarity matrix (temperature={temperature}):")
    print(sim_matrix)
    
    # 计算contrastive loss
    batch_size = len(batch_samples)
    labels = torch.arange(batch_size).repeat(2)
    mask = torch.eq(labels.unsqueeze(0), labels.unsqueeze(1))
    
    exp_sim = torch.exp(sim_matrix)
    
    contrastive_losses = []
    for i in range(batch_size):
        positive_mask = mask[i].clone()
        positive_mask[i] = False
        
        pos_sim = exp_sim[i][positive_mask].sum()
        all_mask = torch.ones_like(mask[i])
        all_mask[i] = False
        all_sim = exp_sim[i][all_mask].sum()
        
        loss_i = -torch.log(pos_sim / all_sim)
        contrastive_losses.append(loss_i)
        print(f"\nAnchor {i}: loss = {loss_i:.4f}")
    
    contrastive_loss = torch.stack(contrastive_losses).mean()
    print(f"\n[RESULT] Contrastive Loss = {contrastive_loss:.4f}")
    
    # Total Loss
    print("\n[5.3 TOTAL LOSS - REAL DATA]:")
    print("-" * 60)
    
    triplet_weight = 0.3
    contrastive_weight = 2.0
    
    total_loss = triplet_weight * triplet_loss + contrastive_weight * contrastive_loss
    print(f"\ntotal_loss = {triplet_weight} x {triplet_loss:.4f} + {contrastive_weight} x {contrastive_loss:.4f}")
    print(f"           = {total_loss:.4f}")

print("\n" + "=" * 80)
print("Real Data Demo Complete!")
print("=" * 80)
print("\nThis demo used REAL spectra from your data files!")
print("All tensor shapes and values are from actual mass spectrometry data.")