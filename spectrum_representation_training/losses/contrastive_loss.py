"""
Contrastive Loss Functions
对比学习损失函数：InfoNCE和Supervised Contrastive Loss
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    """
    InfoNCE Loss (NT-Xent) - Normalized Temperature-scaled Cross Entropy Loss
    用于对比学习，最大化正样本对的相似度，最小化负样本对的相似度
    
    Args:
        temperature: Temperature parameter for scaling
    """
    
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, embeddings, labels):
        """
        Args:
            embeddings: (batch_size, embedding_dim) - L2 normalized embeddings
            labels: (batch_size,) - peptide labels (integer)
        
        Returns:
            loss: scalar
        """
        batch_size = embeddings.size(0)
        device = embeddings.device
        
        # 计算相似度矩阵 (cosine similarity因为embeddings已经L2 normalized)
        similarity_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature
        
        # 创建正样本mask
        labels = labels.unsqueeze(1)
        positive_mask = (labels == labels.T).float()
        
        # 移除对角线（自己和自己）
        positive_mask = positive_mask - torch.eye(batch_size, device=device)
        
        # 计算loss
        # exp(sim(i,j))
        exp_sim = torch.exp(similarity_matrix)
        
        # 分母：所有样本（除了自己）
        # sum_j exp(sim(i,j)) for j != i
        denominator = exp_sim.sum(dim=1) - torch.diag(exp_sim)
        
        # 分子：正样本
        # sum_j exp(sim(i,j)) for j in positives(i)
        numerator = (exp_sim * positive_mask).sum(dim=1)
        
        # 避免除零和log(0)
        numerator = numerator.clamp(min=1e-9)
        denominator = denominator.clamp(min=1e-9)
        
        # InfoNCE loss: -log(numerator / denominator)
        loss = -torch.log(numerator / denominator)
        
        # 只计算有正样本的样本
        valid_mask = positive_mask.sum(dim=1) > 0
        if valid_mask.sum() > 0:
            loss = loss[valid_mask].mean()
        else:
            loss = torch.tensor(0.0, device=device)
        
        return loss


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss
    改进的对比损失，对所有正样本对都进行优化
    
    Reference:
        Khosla et al. "Supervised Contrastive Learning" NeurIPS 2020
    
    Args:
        temperature: Temperature parameter
        base_temperature: Base temperature for normalization
    """
    
    def __init__(self, temperature=0.07, base_temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
    
    def forward(self, embeddings, labels):
        """
        Args:
            embeddings: (batch_size, embedding_dim) - L2 normalized
            labels: (batch_size,) - peptide labels
        
        Returns:
            loss: scalar
        """
        batch_size = embeddings.size(0)
        device = embeddings.device
        
        # 计算相似度矩阵
        similarity_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature
        
        # 创建mask
        labels = labels.unsqueeze(1)
        mask = (labels == labels.T).float()
        
        # 移除对角线
        mask = mask - torch.eye(batch_size, device=device)
        
        # 对于数值稳定性，减去最大值
        logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
        logits = similarity_matrix - logits_max.detach()
        
        # 计算exp
        exp_logits = torch.exp(logits)
        
        # 计算log_prob
        # log(exp(sim(i,j)) / sum_k exp(sim(i,k))) for k != i
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) - torch.diag(exp_logits).unsqueeze(1) + 1e-9)
        
        # 计算每个样本的正样本数量
        positive_pairs = mask.sum(dim=1)
        positive_pairs = positive_pairs.clamp(min=1.0)  # 避免除零
        
        # 计算mean log-likelihood over positive pairs
        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / positive_pairs
        
        # Loss
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        
        # 只计算有正样本的样本
        valid_mask = mask.sum(dim=1) > 0
        if valid_mask.sum() > 0:
            loss = loss[valid_mask].mean()
        else:
            loss = torch.tensor(0.0, device=device)
        
        return loss


class NTXentLoss(nn.Module):
    """
    NT-Xent Loss (Normalized Temperature-scaled Cross Entropy Loss)
    SimCLR使用的损失函数
    
    Args:
        temperature: Temperature parameter
    """
    
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, z_i, z_j):
        """
        Args:
            z_i: (batch_size, embedding_dim) - augmented view 1
            z_j: (batch_size, embedding_dim) - augmented view 2
        
        Returns:
            loss: scalar
        """
        batch_size = z_i.size(0)
        device = z_i.device
        
        # 合并两个view
        z = torch.cat([z_i, z_j], dim=0)  # (2*batch_size, embedding_dim)
        
        # 计算相似度矩阵
        similarity_matrix = torch.matmul(z, z.T) / self.temperature
        
        # 创建正样本mask
        # 对于z_i[k]，正样本是z_j[k]（在位置batch_size+k）
        # 对于z_j[k]，正样本是z_i[k]（在位置k）
        positive_mask = torch.zeros(2 * batch_size, 2 * batch_size, device=device)
        for i in range(batch_size):
            positive_mask[i, batch_size + i] = 1
            positive_mask[batch_size + i, i] = 1
        
        # 移除对角线
        mask = torch.eye(2 * batch_size, device=device)
        similarity_matrix = similarity_matrix - mask * 1e9
        
        # 计算loss
        exp_sim = torch.exp(similarity_matrix)
        
        # 分母：所有样本（除了自己）
        denominator = exp_sim.sum(dim=1)
        
        # 分子：正样本
        numerator = (exp_sim * positive_mask).sum(dim=1)
        
        # Loss
        loss = -torch.log(numerator / (denominator + 1e-9))
        
        return loss.mean()