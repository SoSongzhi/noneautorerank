"""
Spectrum Encoder Wrapper
包装PiPrime的SpectrumEncoder，添加pooling功能
"""
import torch
import torch.nn as nn
import sys
from pathlib import Path

# 添加PrimeNovo到路径
sys.path.append(str(Path(__file__).parent.parent.parent))
from PrimeNovo.components.transformers import SpectrumEncoder


class SpectrumEncoderWrapper(nn.Module):
    """
    包装PiPrime的SpectrumEncoder，添加pooling层
    
    Args:
        pretrained_encoder: PiPrime预训练的SpectrumEncoder
        pooling: 'cls', 'mean', 'max', 'attention'
        freeze: 是否冻结encoder参数
        dim_model: encoder的维度
        n_head: attention heads数量
        dim_feedforward: feedforward维度
        n_layers: transformer层数
    """
    
    def __init__(self, pretrained_encoder=None, pooling='cls', freeze=True, 
                 dim_model=512, n_head=8, dim_feedforward=1024, n_layers=9):
        super().__init__()
        
        if pretrained_encoder is not None:
            self.encoder = pretrained_encoder
        else:
            # 创建新的encoder（如果没有预训练模型）
            self.encoder = SpectrumEncoder(
                dim_model=dim_model,
                n_head=n_head,
                dim_feedforward=dim_feedforward,
                n_layers=n_layers,
                dropout=0.0
            )
        
        self.pooling = pooling
        self.dim_model = dim_model
        
        # 冻结encoder参数
        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        # Attention pooling
        if pooling == 'attention':
            self.attention_weights = nn.Linear(dim_model, 1)
    
    def forward(self, spectra):
        """
        Args:
            spectra: (batch_size, n_peaks, 2) - [m/z, intensity]
        
        Returns:
            pooled: (batch_size, dim_model) - pooled representation
        """
        # 通过encoder
        encoded, mask = self.encoder(spectra)  # (batch, n_peaks+1, dim_model)
        
        # Pooling
        if self.pooling == 'cls':
            # 使用第一个token (latent spectrum)
            pooled = encoded[:, 0, :]
        
        elif self.pooling == 'mean':
            # Mean pooling (排除padding)
            mask_expanded = (~mask).unsqueeze(-1).float()
            sum_embeddings = (encoded * mask_expanded).sum(dim=1)
            sum_mask = mask_expanded.sum(dim=1)
            pooled = sum_embeddings / sum_mask.clamp(min=1e-9)
        
        elif self.pooling == 'max':
            # Max pooling (排除padding)
            mask_expanded = (~mask).unsqueeze(-1).float()
            masked_encoded = encoded * mask_expanded + (mask_expanded - 1) * 1e9
            pooled = masked_encoded.max(dim=1)[0]
        
        elif self.pooling == 'attention':
            # Attention pooling
            attention_scores = self.attention_weights(encoded)  # (batch, n_peaks+1, 1)
            mask_expanded = (~mask).unsqueeze(-1).float()
            attention_scores = attention_scores.masked_fill(mask.unsqueeze(-1), float('-inf'))
            attention_weights = torch.softmax(attention_scores, dim=1)
            pooled = (encoded * attention_weights).sum(dim=1)
        
        return pooled
    
    def unfreeze(self):
        """解冻encoder参数用于fine-tuning"""
        for param in self.encoder.parameters():
            param.requires_grad = True
    
    @property
    def device(self):
        """获取当前设备"""
        return next(self.parameters()).device