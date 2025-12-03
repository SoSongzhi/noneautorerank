"""
Triplet Loss
三元组损失函数，支持hard negative mining
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletLoss(nn.Module):
    """
    Triplet Loss with hard negative mining
    
    Args:
        margin: Margin for triplet loss
        mining: 'hard', 'semi-hard', 'all', None
    """
    
    def __init__(self, margin=0.5, mining='hard'):
        super().__init__()
        self.margin = margin
        self.mining = mining
    
    def forward(self, anchor, positive, negative):
        """
        标准triplet loss
        
        Args:
            anchor: (batch_size, embedding_dim)
            positive: (batch_size, embedding_dim)
            negative: (batch_size, embedding_dim)
        
        Returns:
            loss: scalar
        """
        pos_dist = F.pairwise_distance(anchor, positive, p=2)
        neg_dist = F.pairwise_distance(anchor, negative, p=2)
        
        loss = F.relu(pos_dist - neg_dist + self.margin)
        
        return loss.mean()
    
    def forward_with_mining(self, embeddings, labels):
        """
        使用hard negative mining的triplet loss
        
        Args:
            embeddings: (batch_size, embedding_dim)
            labels: (batch_size,) - peptide labels (integer)
        
        Returns:
            loss: scalar
        """
        # 计算所有pair的距离矩阵
        dist_matrix = torch.cdist(embeddings, embeddings, p=2)
        
        # 创建mask
        labels = labels.unsqueeze(1)
        positive_mask = (labels == labels.T).float()
        negative_mask = (labels != labels.T).float()
        
        # 移除对角线（自己和自己）
        positive_mask = positive_mask - torch.eye(len(labels), device=labels.device)
        
        if self.mining == 'hard':
            # Hard positive: 最远的正样本
            # 将非正样本的距离设为0，这样max会选择正样本中最大的
            positive_dist = dist_matrix * positive_mask
            hardest_positive_dist = positive_dist.max(dim=1)[0]
            
            # Hard negative: 最近的负样本
            # 将正样本的距离设为很大的值，这样min会选择负样本中最小的
            masked_dist = dist_matrix + (1 - negative_mask) * 1e9
            hardest_negative_dist = masked_dist.min(dim=1)[0]
            
            # Triplet loss
            loss = F.relu(hardest_positive_dist - hardest_negative_dist + self.margin)
            
            # 只计算有正样本和负样本的样本
            valid_mask = (positive_mask.sum(dim=1) > 0) & (negative_mask.sum(dim=1) > 0)
            if valid_mask.sum() > 0:
                loss = loss[valid_mask].mean()
            else:
                loss = torch.tensor(0.0, device=embeddings.device)
        
        elif self.mining == 'semi-hard':
            # Semi-hard negative: 比positive远但在margin内的负样本
            # 这里简化实现，选择距离在[pos_dist, pos_dist + margin]范围内的负样本
            
            # 对每个anchor，找到所有正样本的平均距离
            positive_dist = dist_matrix * positive_mask
            avg_positive_dist = positive_dist.sum(dim=1) / positive_mask.sum(dim=1).clamp(min=1)
            
            # 找semi-hard negatives
            avg_positive_dist = avg_positive_dist.unsqueeze(1)
            semi_hard_mask = negative_mask * (dist_matrix > avg_positive_dist) * (dist_matrix < avg_positive_dist + self.margin)
            
            if semi_hard_mask.sum() > 0:
                # 使用semi-hard negatives
                masked_dist = dist_matrix + (1 - semi_hard_mask) * 1e9
                semi_hard_negative_dist = masked_dist.min(dim=1)[0]
            else:
                # 如果没有semi-hard negatives，使用hard negatives
                masked_dist = dist_matrix + (1 - negative_mask) * 1e9
                semi_hard_negative_dist = masked_dist.min(dim=1)[0]
            
            loss = F.relu(avg_positive_dist.squeeze() - semi_hard_negative_dist + self.margin)
            
            valid_mask = (positive_mask.sum(dim=1) > 0) & (negative_mask.sum(dim=1) > 0)
            if valid_mask.sum() > 0:
                loss = loss[valid_mask].mean()
            else:
                loss = torch.tensor(0.0, device=embeddings.device)
        
        else:  # 'all' - 使用所有valid triplets
            # 对每个anchor，计算所有valid triplets的loss
            positive_dist = dist_matrix.unsqueeze(2)  # (batch, batch, 1)
            negative_dist = dist_matrix.unsqueeze(1)  # (batch, 1, batch)
            
            # 创建triplet mask
            positive_mask_3d = positive_mask.unsqueeze(2)  # (batch, batch, 1)
            negative_mask_3d = negative_mask.unsqueeze(1)  # (batch, 1, batch)
            triplet_mask = positive_mask_3d * negative_mask_3d
            
            # 计算所有triplets的loss
            triplet_loss = F.relu(positive_dist - negative_dist + self.margin)
            triplet_loss = triplet_loss * triplet_mask
            
            # 平均
            num_valid_triplets = triplet_mask.sum()
            if num_valid_triplets > 0:
                loss = triplet_loss.sum() / num_valid_triplets
            else:
                loss = torch.tensor(0.0, device=embeddings.device)
        
        return loss