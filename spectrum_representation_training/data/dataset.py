"""
Spectrum Contrastive Dataset
对比学习数据集
"""
import torch
from torch.utils.data import Dataset
import numpy as np
from typing import List, Dict
from .augmentation import SpectrumAugmentation


class SpectrumContrastiveDataset(Dataset):
    """
    对比学习数据集
    
    Args:
        spectra_data: List of spectrum arrays [(n_peaks, 2), ...]
        peptide_labels: List of peptide sequences
        augment: 是否使用数据增强
        mode: 'triplet' or 'contrastive'
        augmentation_params: 数据增强参数字典
    """
    
    def __init__(self, 
                 spectra_data: List[np.ndarray], 
                 peptide_labels: List[str],
                 augment: bool = True,
                 mode: str = 'triplet',
                 augmentation_params: Dict = None):
        self.spectra = spectra_data
        self.labels = peptide_labels
        self.augment = augment
        self.mode = mode
        
        # 数据增强
        if augment:
            if augmentation_params is None:
                self.augmentation = SpectrumAugmentation()
            else:
                self.augmentation = SpectrumAugmentation(**augmentation_params)
        
        # 为每个peptide建立索引
        self.peptide_to_indices = {}
        for idx, peptide in enumerate(peptide_labels):
            if peptide not in self.peptide_to_indices:
                self.peptide_to_indices[peptide] = []
            self.peptide_to_indices[peptide].append(idx)
        
        # 创建label映射（peptide string -> integer）
        unique_peptides = list(self.peptide_to_indices.keys())
        self.peptide_to_int = {p: i for i, p in enumerate(unique_peptides)}
        self.int_to_peptide = {i: p for p, i in self.peptide_to_int.items()}
        
        print(f"Dataset initialized: {len(self.spectra)} spectra, {len(unique_peptides)} unique peptides")
    
    def __len__(self):
        return len(self.spectra)
    
    def __getitem__(self, idx):
        anchor_spectrum = self.spectra[idx].copy()
        anchor_peptide = self.labels[idx]
        anchor_label = self.peptide_to_int[anchor_peptide]
        
        if self.mode == 'triplet':
            # Triplet模式：返回anchor, positive, negative
            
            # 选择positive（相同peptide的其他spectrum）
            positive_indices = [i for i in self.peptide_to_indices[anchor_peptide] if i != idx]
            if positive_indices:
                pos_idx = np.random.choice(positive_indices)
                positive_spectrum = self.spectra[pos_idx].copy()
            else:
                # 如果没有其他样本，使用增强后的自己
                positive_spectrum = anchor_spectrum.copy()
                if self.augment:
                    positive_spectrum = self.augmentation(positive_spectrum)
            
            # 选择negative（不同peptide的spectrum）
            negative_peptides = [p for p in self.peptide_to_indices.keys() if p != anchor_peptide]
            if negative_peptides:
                neg_peptide = np.random.choice(negative_peptides)
                neg_idx = np.random.choice(self.peptide_to_indices[neg_peptide])
                negative_spectrum = self.spectra[neg_idx].copy()
            else:
                # 如果只有一个peptide，使用随机噪声
                negative_spectrum = np.random.randn(*anchor_spectrum.shape).astype(np.float32)
            
            # 数据增强
            if self.augment:
                anchor_spectrum = self.augmentation(anchor_spectrum)
                positive_spectrum = self.augmentation(positive_spectrum)
                # negative通常不增强，保持原样
            
            return {
                'anchor': torch.FloatTensor(anchor_spectrum),
                'positive': torch.FloatTensor(positive_spectrum),
                'negative': torch.FloatTensor(negative_spectrum),
                'label': anchor_label
            }
        
        else:  # contrastive模式
            # 只返回spectrum和label，在batch中构建正负样本对
            if self.augment:
                anchor_spectrum = self.augmentation(anchor_spectrum)
            
            return {
                'spectrum': torch.FloatTensor(anchor_spectrum),
                'label': anchor_label,
                'peptide': anchor_peptide  # 保留peptide string用于调试
            }
    
    def get_peptide_statistics(self):
        """获取数据集统计信息"""
        stats = {}
        for peptide, indices in self.peptide_to_indices.items():
            stats[peptide] = len(indices)
        return stats
    
    def get_spectrum_lengths(self):
        """获取所有spectrum的长度分布"""
        lengths = [len(s) for s in self.spectra]
        return {
            'min': min(lengths),
            'max': max(lengths),
            'mean': np.mean(lengths),
            'median': np.median(lengths),
            'std': np.std(lengths)
        }


def load_data_from_pickle(pickle_path):
    """
    从pickle文件加载数据
    
    Args:
        pickle_path: pickle文件路径
    
    Returns:
        spectra_data: List of spectrum arrays
        peptide_labels: List of peptide sequences
    """
    import pickle
    
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    
    # 假设pickle文件包含 {'spectra': [...], 'peptides': [...]}
    if isinstance(data, dict):
        spectra_data = data['spectra']
        peptide_labels = data['peptides']
    elif isinstance(data, tuple):
        spectra_data, peptide_labels = data
    else:
        raise ValueError("Unsupported pickle format")
    
    return spectra_data, peptide_labels


def create_train_val_split(spectra_data, peptide_labels, val_split=0.2, random_seed=42):
    """
    划分训练集和验证集
    
    Args:
        spectra_data: List of spectrum arrays
        peptide_labels: List of peptide sequences
        val_split: 验证集比例
        random_seed: 随机种子
    
    Returns:
        train_spectra, train_labels, val_spectra, val_labels
    """
    np.random.seed(random_seed)
    
    n_samples = len(spectra_data)
    indices = np.random.permutation(n_samples)
    
    n_train = int(n_samples * (1 - val_split))
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]
    
    train_spectra = [spectra_data[i] for i in train_indices]
    train_labels = [peptide_labels[i] for i in train_indices]
    val_spectra = [spectra_data[i] for i in val_indices]
    val_labels = [peptide_labels[i] for i in val_indices]
    
    return train_spectra, train_labels, val_spectra, val_labels