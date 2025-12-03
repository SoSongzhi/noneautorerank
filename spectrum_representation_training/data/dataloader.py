"""
DataLoader utilities
创建训练和验证DataLoader
"""
import torch
from torch.utils.data import DataLoader
import numpy as np


def collate_fn_triplet(batch):
    """
    Triplet模式的collate function
    将变长的spectrum pad到相同长度
    
    Args:
        batch: List of dict with keys 'anchor', 'positive', 'negative', 'label'
    
    Returns:
        dict with padded tensors
    """
    # 找到最大peak数
    max_peaks_anchor = max(item['anchor'].shape[0] for item in batch)
    max_peaks_positive = max(item['positive'].shape[0] for item in batch)
    max_peaks_negative = max(item['negative'].shape[0] for item in batch)
    max_peaks = max(max_peaks_anchor, max_peaks_positive, max_peaks_negative)
    
    # Pad spectra
    anchors, positives, negatives, labels = [], [], [], []
    
    for item in batch:
        # Pad anchor
        anchor = item['anchor']
        if anchor.shape[0] < max_peaks:
            padding = torch.zeros(max_peaks - anchor.shape[0], 2)
            anchor = torch.cat([anchor, padding], dim=0)
        anchors.append(anchor)
        
        # Pad positive
        positive = item['positive']
        if positive.shape[0] < max_peaks:
            padding = torch.zeros(max_peaks - positive.shape[0], 2)
            positive = torch.cat([positive, padding], dim=0)
        positives.append(positive)
        
        # Pad negative
        negative = item['negative']
        if negative.shape[0] < max_peaks:
            padding = torch.zeros(max_peaks - negative.shape[0], 2)
            negative = torch.cat([negative, padding], dim=0)
        negatives.append(negative)
        
        labels.append(item['label'])
    
    return {
        'anchor': torch.stack(anchors),
        'positive': torch.stack(positives),
        'negative': torch.stack(negatives),
        'labels': torch.tensor(labels)
    }


def collate_fn_contrastive(batch):
    """
    Contrastive模式的collate function
    
    Args:
        batch: List of dict with keys 'spectrum', 'label'
    
    Returns:
        dict with padded tensors
    """
    max_peaks = max(item['spectrum'].shape[0] for item in batch)
    
    spectra, labels = [], []
    
    for item in batch:
        spectrum = item['spectrum']
        if spectrum.shape[0] < max_peaks:
            padding = torch.zeros(max_peaks - spectrum.shape[0], 2)
            spectrum = torch.cat([spectrum, padding], dim=0)
        spectra.append(spectrum)
        labels.append(item['label'])
    
    return {
        'spectra': torch.stack(spectra),
        'labels': torch.tensor(labels)
    }


def create_dataloaders(train_dataset, val_dataset, 
                      batch_size=128, num_workers=4, mode='triplet',
                      pin_memory=True):
    """
    创建训练和验证DataLoader
    
    Args:
        train_dataset: 训练数据集
        val_dataset: 验证数据集
        batch_size: 批大小
        num_workers: 数据加载线程数
        mode: 'triplet' or 'contrastive'
        pin_memory: 是否使用pin_memory加速
    
    Returns:
        train_loader, val_loader
    """
    collate_fn = collate_fn_triplet if mode == 'triplet' else collate_fn_contrastive
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=True  # 丢弃最后不完整的batch
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    return train_loader, val_loader


class InfiniteDataLoader:
    """
    无限循环的DataLoader，用于训练
    """
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.iterator = iter(dataloader)
    
    def __iter__(self):
        return self
    
    def __next__(self):
        try:
            batch = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.dataloader)
            batch = next(self.iterator)
        return batch
    
    def __len__(self):
        return len(self.dataloader)