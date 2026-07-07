# -*- coding: utf-8 -*-
"""
演示数据流动过程 - 从原始数据到Loss计算
展示真实的tensor形状和数值
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple
import sys
import io

# 设置UTF-8编码输出
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("=" * 80)
print("Spectrum Representation Learning - Data Flow Demo")
print("=" * 80)

# ============================================================================
# Step 1: Raw Data (simulating PKL file content)
# ============================================================================
print("\n" + "=" * 80)
print("Step 1: Raw Data (simulated from PKL file)")
print("=" * 80)

# Simulate 6 spectra
raw_data = {
    'spectra': [
        # PEPTIDEK - 2 spectra
        np.array([[100.5, 0.8], [150.2, 0.6], [200.3, 0.9]]),  # 3 peaks
        np.array([[102.1, 0.82], [152.5, 0.58], [198.7, 0.88], [250.1, 0.45]]),  # 4 peaks
        
        # SEQUENCER - 2 spectra
        np.array([[120.3, 0.9], [180.1, 0.7], [240.5, 0.85]]),  # 3 peaks
        np.array([[118.9, 0.88], [178.3, 0.72], [238.1, 0.83]]),  # 3 peaks
        
        # PROTEINK - 2 spectra
        np.array([[110.8, 0.85], [160.5, 0.65], [210.2, 0.75], [260.8, 0.55]]),  # 4 peaks
        np.array([[112.3, 0.87], [162.1, 0.63], [212.5, 0.73]]),  # 3 peaks
    ],
    'peptides': ['PEPTIDEK', 'PEPTIDEK', 'SEQUENCER', 'SEQUENCER', 'PROTEINK', 'PROTEINK'],
    'precursor_mz': [450.2, 450.3, 520.1, 520.2, 480.5, 480.6],
    'precursor_charge': [2, 2, 2, 2, 2, 2]
}

print(f"\nTotal spectra: {len(raw_data['spectra'])}")
print(f"Peptides: {raw_data['peptides']}")
print(f"\nNumber of peaks per spectrum:")
for i, spec in enumerate(raw_data['spectra']):
    print(f"  Spectrum {i} ({raw_data['peptides'][i]}): {len(spec)} peaks")
    print(f"    First 3 peaks: {spec[:3].tolist()}")

# ============================================================================
# Step 2: Triplet Sampling (Dataset.__getitem__)
# ============================================================================
print("\n" + "=" * 80)
print("Step 2: Triplet Sampling (creating triplet for index 0)")
print("=" * 80)

def sample_triplet(idx: int, data: Dict) -> Tuple:
    """Simulate Dataset's triplet sampling"""
    # Anchor
    anchor_spectrum = data['spectra'][idx]
    anchor_peptide = data['peptides'][idx]
    anchor_mz = data['precursor_mz'][idx]
    anchor_charge = data['precursor_charge'][idx]
    
    print(f"\n[ANCHOR] (index {idx}):")
    print(f"   Peptide: {anchor_peptide}")
    print(f"   Spectrum shape: {anchor_spectrum.shape}")
    print(f"   First 3 peaks: {anchor_spectrum[:3].tolist()}")
    
    # Positive: find another spectrum with same peptide
    positive_indices = [i for i, p in enumerate(data['peptides']) 
                       if p == anchor_peptide and i != idx]
    pos_idx = np.random.choice(positive_indices)
    positive_spectrum = data['spectra'][pos_idx]
    positive_mz = data['precursor_mz'][pos_idx]
    positive_charge = data['precursor_charge'][pos_idx]
    
    print(f"\n[POSITIVE] (index {pos_idx}):")
    print(f"   Peptide: {data['peptides'][pos_idx]} (same as anchor)")
    print(f"   Spectrum shape: {positive_spectrum.shape}")
    print(f"   First 3 peaks: {positive_spectrum[:3].tolist()}")
    
    # Negative: find spectrum with different peptide
    negative_indices = [i for i, p in enumerate(data['peptides']) 
                       if p != anchor_peptide]
    neg_idx = np.random.choice(negative_indices)
    negative_spectrum = data['spectra'][neg_idx]
    negative_mz = data['precursor_mz'][neg_idx]
    negative_charge = data['precursor_charge'][neg_idx]
    
    print(f"\n[NEGATIVE] (index {neg_idx}):")
    print(f"   Peptide: {data['peptides'][neg_idx]} (different from anchor)")
    print(f"   Spectrum shape: {negative_spectrum.shape}")
    print(f"   First 3 peaks: {negative_spectrum[:3].tolist()}")
    
    return {
        'anchor': (anchor_spectrum, anchor_mz, anchor_charge),
        'positive': (positive_spectrum, positive_mz, positive_charge),
        'negative': (negative_spectrum, negative_mz, negative_charge),
        'label': anchor_peptide
    }

# Sample index 0 and index 2
sample_0 = sample_triplet(0, raw_data)
sample_2 = sample_triplet(2, raw_data)

# ============================================================================
# Step 3: Collate Function - Create Batch
# ============================================================================
print("\n" + "=" * 80)
print("Step 3: Collate Function - Combine samples into batch")
print("=" * 80)

def collate_fn(batch_samples: List[Dict]) -> Dict:
    """Simulate DataLoader's collate_fn"""
    print(f"\nReceived {len(batch_samples)} samples, combining into batch")
    
    # Extract all spectra
    anchor_spectra = [s['anchor'][0] for s in batch_samples]
    positive_spectra = [s['positive'][0] for s in batch_samples]
    negative_spectra = [s['negative'][0] for s in batch_samples]
    
    # Find max number of peaks
    max_peaks = max(
        max(len(s) for s in anchor_spectra),
        max(len(s) for s in positive_spectra),
        max(len(s) for s in negative_spectra)
    )
    print(f"\nMax peaks in batch: {max_peaks}")
    
    # Padding function
    def pad_spectrum(spectrum: np.ndarray, max_len: int) -> torch.Tensor:
        """Pad spectrum to specified length"""
        padded = np.zeros((max_len, 2))
        padded[:len(spectrum)] = spectrum
        return torch.FloatTensor(padded)
    
    # Pad all spectra
    print("\nPadding process:")
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
    
    # Stack into batch
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

# Create batch
batch = collate_fn([sample_0, sample_2])

print("\n[BATCH DETAILS]:")
print(f"\nAnchor batch (first 2 samples, first 3 peaks):")
print(batch['anchor'][:, :3, :])
print(f"\nPositive batch (first 2 samples, first 3 peaks):")
print(batch['positive'][:, :3, :])
print(f"\nNegative batch (first 2 samples, first 3 peaks):")
print(batch['negative'][:, :3, :])

# ============================================================================
# Step 4: Model Forward Pass (simplified)
# ============================================================================
print("\n" + "=" * 80)
print("Step 4: Model Forward Pass")
print("=" * 80)

class SimpleEncoder(nn.Module):
    """Simplified encoder (simulating PiPrime's SpectrumEncoder)"""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 512)  # Simplified: direct mapping to 512-dim
        
    def forward(self, x):
        # x: (batch, peaks, 2)
        # Simplified: encode each peak, then average
        encoded = self.linear(x)  # (batch, peaks, 512)
        # Add CLS token (simplified: use average)
        cls_token = encoded.mean(dim=1, keepdim=True)  # (batch, 1, 512)
        return torch.cat([cls_token, encoded], dim=1)  # (batch, peaks+1, 512)

class ProjectionHead(nn.Module):
    """Projection head"""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(512, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 256)
        
    def forward(self, x):
        # x: (batch, 512)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return F.normalize(x, p=2, dim=1)  # L2 normalization

# Create models
encoder = SimpleEncoder()
projection = ProjectionHead()

print("\n[MODEL ARCHITECTURE]:")
print(f"   Encoder: input(batch, peaks, 2) -> output(batch, peaks+1, 512)")
print(f"   Projection: input(batch, 512) -> output(batch, 256)")

# Forward pass
print("\n[FORWARD PASS]:")

with torch.no_grad():
    # Encode anchor
    print("\n1. Encoding Anchor:")
    anchor_encoded = encoder(batch['anchor'])
    print(f"   Input shape: {batch['anchor'].shape}")
    print(f"   Encoded shape: {anchor_encoded.shape}")
    
    anchor_cls = anchor_encoded[:, 0, :]  # Extract CLS token
    print(f"   CLS token shape: {anchor_cls.shape}")
    print(f"   CLS token first 5 dims: {anchor_cls[0, :5]}")
    
    anchor_emb = projection(anchor_cls)
    print(f"   Projected shape: {anchor_emb.shape}")
    print(f"   Embedding first 5 dims: {anchor_emb[0, :5]}")
    print(f"   Embedding L2 norm: {torch.norm(anchor_emb[0]):.4f} (should be ~1.0)")
    
    # Encode positive
    print("\n2. Encoding Positive:")
    positive_encoded = encoder(batch['positive'])
    positive_cls = positive_encoded[:, 0, :]
    positive_emb = projection(positive_cls)
    print(f"   Projected shape: {positive_emb.shape}")
    print(f"   Embedding first 5 dims: {positive_emb[0, :5]}")
    
    # Encode negative
    print("\n3. Encoding Negative:")
    negative_encoded = encoder(batch['negative'])
    negative_cls = negative_encoded[:, 0, :]
    negative_emb = projection(negative_cls)
    print(f"   Projected shape: {negative_emb.shape}")
    print(f"   Embedding first 5 dims: {negative_emb[0, :5]}")

# ============================================================================
# Step 5: Calculate Loss
# ============================================================================
print("\n" + "=" * 80)
print("Step 5: Calculate Loss")
print("=" * 80)

# 5.1 Triplet Loss
print("\n[5.1 TRIPLET LOSS]:")
print("-" * 60)

margin = 0.2

with torch.no_grad():
    # Calculate distances
    d_pos = torch.sum((anchor_emb - positive_emb) ** 2, dim=1)
    d_neg = torch.sum((anchor_emb - negative_emb) ** 2, dim=1)
    
    print("\nSample 0:")
    print(f"   d(anchor, positive) = {d_pos[0].item():.4f}")
    print(f"   d(anchor, negative) = {d_neg[0].item():.4f}")
    print(f"   margin = {margin}")
    print(f"   loss = max(0, {d_pos[0].item():.4f} - {d_neg[0].item():.4f} + {margin})")
    loss_0 = max(0, d_pos[0].item() - d_neg[0].item() + margin)
    print(f"        = {loss_0:.4f}")
    
    print("\nSample 1:")
    print(f"   d(anchor, positive) = {d_pos[1].item():.4f}")
    print(f"   d(anchor, negative) = {d_neg[1].item():.4f}")
    print(f"   margin = {margin}")
    print(f"   loss = max(0, {d_pos[1].item():.4f} - {d_neg[1].item():.4f} + {margin})")
    loss_1 = max(0, d_pos[1].item() - d_neg[1].item() + margin)
    print(f"        = {loss_1:.4f}")
    
    triplet_loss = (loss_0 + loss_1) / 2
    print(f"\n[RESULT] Triplet Loss = ({loss_0:.4f} + {loss_1:.4f}) / 2 = {triplet_loss:.4f}")

# 5.2 Contrastive Loss (InfoNCE)
print("\n[5.2 CONTRASTIVE LOSS (InfoNCE)]:")
print("-" * 60)

temperature = 0.05

with torch.no_grad():
    # Combine embeddings
    all_embs = torch.cat([anchor_emb, positive_emb], dim=0)  # (4, 256)
    print(f"\nCombined embeddings shape: {all_embs.shape}")
    
    # Create labels (to identify same peptides)
    # Sample 0: PEPTIDEK, Sample 1: SEQUENCER
    # all_embs[0] = anchor_0 (PEPTIDEK)
    # all_embs[1] = anchor_1 (SEQUENCER)
    # all_embs[2] = positive_0 (PEPTIDEK)
    # all_embs[3] = positive_1 (SEQUENCER)
    
    # Calculate similarity matrix
    sim_matrix = torch.matmul(all_embs, all_embs.T) / temperature
    print(f"\nSimilarity matrix shape: {sim_matrix.shape}")
    print(f"Similarity matrix (divided by temperature={temperature}):")
    print(sim_matrix)
    
    # Calculate loss for anchor_0
    print("\nCalculating loss for anchor_0:")
    print(f"   anchor_0 similarity with all samples: {sim_matrix[0]}")
    
    # positive_mask: mark which are positive samples
    # For anchor_0 (PEPTIDEK), positive_0 (index 2) is positive
    positive_mask = torch.zeros(4, dtype=torch.bool)
    positive_mask[2] = True  # positive_0 is positive sample
    print(f"   Positive mask: {positive_mask}")
    
    # Calculate exp(similarity)
    exp_sim = torch.exp(sim_matrix[0])
    print(f"   exp(similarity): {exp_sim}")
    
    # Positive sample similarity
    pos_sim = exp_sim[positive_mask].sum()
    print(f"   Sum of positive similarities: {pos_sim:.4f}")
    
    # All sample similarities (excluding self)
    mask = torch.ones(4, dtype=torch.bool)
    mask[0] = False  # Exclude self
    all_sim = exp_sim[mask].sum()
    print(f"   Sum of all similarities (excluding self): {all_sim:.4f}")
    
    # Loss
    loss_anchor_0 = -torch.log(pos_sim / all_sim)
    print(f"   loss = -log({pos_sim:.4f} / {all_sim:.4f}) = {loss_anchor_0:.4f}")
    
    # Calculate loss for anchor_1 (similar)
    print("\nCalculating loss for anchor_1:")
    positive_mask_1 = torch.zeros(4, dtype=torch.bool)
    positive_mask_1[3] = True  # positive_1 is positive sample
    exp_sim_1 = torch.exp(sim_matrix[1])
    pos_sim_1 = exp_sim_1[positive_mask_1].sum()
    mask_1 = torch.ones(4, dtype=torch.bool)
    mask_1[1] = False
    all_sim_1 = exp_sim_1[mask_1].sum()
    loss_anchor_1 = -torch.log(pos_sim_1 / all_sim_1)
    print(f"   loss = -log({pos_sim_1:.4f} / {all_sim_1:.4f}) = {loss_anchor_1:.4f}")
    
    contrastive_loss = (loss_anchor_0 + loss_anchor_1) / 2
    print(f"\n[RESULT] Contrastive Loss = ({loss_anchor_0:.4f} + {loss_anchor_1:.4f}) / 2 = {contrastive_loss:.4f}")

# 5.3 Total Loss
print("\n[5.3 TOTAL LOSS]:")
print("-" * 60)

triplet_weight = 0.3
contrastive_weight = 2.0

total_loss = triplet_weight * triplet_loss + contrastive_weight * contrastive_loss
print(f"\ntotal_loss = {triplet_weight} x {triplet_loss:.4f} + {contrastive_weight} x {contrastive_loss:.4f}")
print(f"           = {triplet_weight * triplet_loss:.4f} + {contrastive_weight * contrastive_loss:.4f}")
print(f"           = {total_loss:.4f}")

# ============================================================================
# Step 6: Backpropagation and Parameter Update (conceptual)
# ============================================================================
print("\n" + "=" * 80)
print("Step 6: Backpropagation and Parameter Update (conceptual)")
print("=" * 80)

print("""
In actual training, the following steps would be executed:

1. Backpropagation:
   total_loss.backward()
   
   This calculates gradients for all parameters:
   - encoder parameter gradients
   - projection parameter gradients

2. Parameter update:
   optimizer.step()
   
   Update parameters based on gradients:
   - theta_new = theta_old - learning_rate x gradient
   
3. Zero gradients:
   optimizer.zero_grad()
   
   Prepare for next batch

4. Repeat until model converges
""")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 80)
print("[DATA FLOW SUMMARY]")
print("=" * 80)

print("""
Complete Flow:

1. Raw Data (PKL file)
   |- spectra: List[np.ndarray]  # Each spectrum is (n_peaks, 2) array
   |- peptides: List[str]         # Peptide sequences
   +- precursor_mz/charge: List   # Precursor ion info

2. Dataset Sampling (Triplet)
   |- __getitem__(idx) -> returns a triplet
   |   |- anchor: query spectrum
   |   |- positive: another spectrum of same peptide
   |   +- negative: spectrum of different peptide

3. DataLoader + Collate
   |- Collect batch_size triplets
   |- Pad to same length
   +- Convert to tensor: (batch_size, max_peaks, 2)

4. Model Forward Pass
   |- Encoder: (batch, peaks, 2) -> (batch, peaks+1, 512)
   |- Extract CLS: (batch, peaks+1, 512) -> (batch, 512)
   +- Projection: (batch, 512) -> (batch, 256)

5. Loss Calculation
   |- Triplet Loss: push away negative samples
   |   +- max(0, d_pos - d_neg + margin)
   |
   +- Contrastive Loss: pull together positive samples
       +- -log(exp(sim_pos) / sum(exp(sim_all)))

6. Backpropagation + Parameter Update
   +- Gradient descent to optimize parameters

Key Points:
[OK] Triplet sampling ensures each batch has positive/negative pairs
[OK] Padding handles different length spectra
[OK] L2 normalization ensures embeddings on unit sphere
[OK] Two losses work together: pulling + pushing
[OK] Two-stage training: freeze encoder first, then fine-tune
""")

print("\n" + "=" * 80)
print("Demo Complete!")
print("=" * 80)