"""
Projection Head
MLP投影头，将encoder输出映射到对比学习空间
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    """
    投影头：将encoder输出映射到对比学习空间
    
    Args:
        input_dim: 输入维度 (encoder的dim_model)
        hidden_dim: 隐藏层维度
        output_dim: 输出embedding维度
        num_layers: MLP层数
        dropout: Dropout概率
    """
    
    def __init__(self, input_dim=512, hidden_dim=512, output_dim=256, 
                 num_layers=3, dropout=0.1):
        super().__init__()
        
        layers = []
        
        # 第一层
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        
        # 中间层
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        
        # 输出层
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, input_dim)
        
        Returns:
            embeddings: (batch_size, output_dim) - L2 normalized
        """
        embeddings = self.mlp(x)
        # L2 normalization for cosine similarity
        return F.normalize(embeddings, p=2, dim=-1)