"""
Training Script
训练spectrum表征学习模型
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
import numpy as np
import yaml
import argparse
from tqdm import tqdm
import os
from datetime import datetime

# 导入自定义模块
from spectrum_representation_training.models import ContrastiveSpectrumModel
from spectrum_representation_training.losses import TripletLoss, InfoNCELoss, SupConLoss
from spectrum_representation_training.data import (
    SpectrumContrastiveDataset, 
    create_dataloaders,
    load_data_from_pickle,
    create_train_val_split
)

# 尝试导入PiPrime
try:
    from PrimeNovo.denovo.model import Spec2Pep
    PIPRIME_AVAILABLE = True
except ImportError:
    print("Warning: PrimeNovo not found. Will create encoder from scratch.")
    PIPRIME_AVAILABLE = False


def set_seed(seed):
    """设置随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def load_pretrained_encoder(checkpoint_path, device):
    """加载PiPrime预训练encoder"""
    if not PIPRIME_AVAILABLE:
        print("PrimeNovo not available, creating encoder from scratch")
        return None
    
    try:
        # 加载PiPrime模型
        model = Spec2Pep.load_from_checkpoint(checkpoint_path)
        encoder = model.encoder
        encoder.to(device)
        print(f"✓ Loaded pretrained encoder from {checkpoint_path}")
        return encoder
    except Exception as e:
        print(f"Failed to load pretrained encoder: {e}")
        return None


def create_model(config, device):
    """创建模型"""
    # 加载预训练encoder
    pretrained_encoder = None
    if config['pretrained']['use_pretrained'] and config['pretrained']['checkpoint_path']:
        pretrained_encoder = load_pretrained_encoder(
            config['pretrained']['checkpoint_path'], 
            device
        )
    
    # 创建模型
    model = ContrastiveSpectrumModel(
        pretrained_encoder=pretrained_encoder,
        encoder_dim=config['model']['encoder_dim'],
        embedding_dim=config['model']['embedding_dim'],
        pooling=config['model']['pooling'],
        freeze_encoder=config['model']['freeze_encoder'],
        projection_layers=config['model']['projection_layers']
    )
    
    model.to(device)
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel created:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    return model


def create_loss_functions(config):
    """创建损失函数"""
    # Triplet Loss
    triplet_loss = TripletLoss(
        margin=config['training']['triplet_margin'],
        mining=config['training']['triplet_mining']
    )
    
    # Contrastive Loss
    if config['training']['contrastive_type'] == 'infonce':
        contrastive_loss = InfoNCELoss(
            temperature=config['training']['temperature']
        )
    else:  # supcon
        contrastive_loss = SupConLoss(
            temperature=config['training']['temperature']
        )
    
    return triplet_loss, contrastive_loss


def train_epoch(model, train_loader, optimizer, triplet_loss, contrastive_loss, 
                config, device, epoch):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    total_triplet_loss = 0.0
    total_contrastive_loss = 0.0
    
    triplet_weight = config['training']['triplet_weight']
    contrastive_weight = config['training']['contrastive_weight']
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for batch_idx, batch in enumerate(pbar):
        # 移动到device
        anchor = batch['anchor'].to(device)
        positive = batch['positive'].to(device)
        negative = batch['negative'].to(device)
        labels = batch['labels'].to(device)
        
        # 前向传播
        anchor_emb = model(anchor)
        positive_emb = model(positive)
        negative_emb = model(negative)
        
        # 计算损失
        loss_triplet = triplet_loss(anchor_emb, positive_emb, negative_emb)
        
        # Contrastive loss (使用anchor和positive)
        all_embs = torch.cat([anchor_emb, positive_emb], dim=0)
        all_labels = torch.cat([labels, labels], dim=0)
        loss_contrastive = contrastive_loss(all_embs, all_labels)
        
        # 总损失
        loss = triplet_weight * loss_triplet + contrastive_weight * loss_contrastive
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        if config['training']['grad_clip'] > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 
                max_norm=config['training']['grad_clip']
            )
        
        optimizer.step()
        
        # 记录
        total_loss += loss.item()
        total_triplet_loss += loss_triplet.item()
        total_contrastive_loss += loss_contrastive.item()
        
        # 更新进度条
        if batch_idx % config['output']['log_interval'] == 0:
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'triplet': f'{loss_triplet.item():.4f}',
                'contrast': f'{loss_contrastive.item():.4f}'
            })
    
    n_batches = len(train_loader)
    return {
        'loss': total_loss / n_batches,
        'triplet_loss': total_triplet_loss / n_batches,
        'contrastive_loss': total_contrastive_loss / n_batches
    }


def validate(model, val_loader, triplet_loss, contrastive_loss, config, device):
    """验证"""
    model.eval()
    total_loss = 0.0
    total_triplet_loss = 0.0
    total_contrastive_loss = 0.0
    
    triplet_weight = config['training']['triplet_weight']
    contrastive_weight = config['training']['contrastive_weight']
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            anchor = batch['anchor'].to(device)
            positive = batch['positive'].to(device)
            negative = batch['negative'].to(device)
            labels = batch['labels'].to(device)
            
            anchor_emb = model(anchor)
            positive_emb = model(positive)
            negative_emb = model(negative)
            
            loss_triplet = triplet_loss(anchor_emb, positive_emb, negative_emb)
            
            all_embs = torch.cat([anchor_emb, positive_emb], dim=0)
            all_labels = torch.cat([labels, labels], dim=0)
            loss_contrastive = contrastive_loss(all_embs, all_labels)
            
            loss = triplet_weight * loss_triplet + contrastive_weight * loss_contrastive
            
            total_loss += loss.item()
            total_triplet_loss += loss_triplet.item()
            total_contrastive_loss += loss_contrastive.item()
    
    n_batches = len(val_loader)
    return {
        'loss': total_loss / n_batches,
        'triplet_loss': total_triplet_loss / n_batches,
        'contrastive_loss': total_contrastive_loss / n_batches
    }


def evaluate_similarity(model, val_loader, device, sample_size=1000):
    """评估相似度指标"""
    model.eval()
    
    all_embeddings = []
    all_labels = []
    
    with torch.no_grad():
        for batch in val_loader:
            anchor = batch['anchor'].to(device)
            labels = batch['labels']
            
            embeddings = model(anchor)
            all_embeddings.append(embeddings.cpu())
            all_labels.append(labels)
            
            if len(all_embeddings) * embeddings.size(0) >= sample_size:
                break
    
    embeddings = torch.cat(all_embeddings, dim=0)[:sample_size]
    labels = torch.cat(all_labels, dim=0)[:sample_size]
    
    # 计算相似度
    same_peptide_sims = []
    diff_peptide_sims = []
    
    for i in range(len(labels)):
        for j in range(i+1, len(labels)):
            sim = torch.cosine_similarity(
                embeddings[i].unsqueeze(0), 
                embeddings[j].unsqueeze(0)
            ).item()
            
            if labels[i] == labels[j]:
                same_peptide_sims.append(sim)
            else:
                diff_peptide_sims.append(sim)
                if len(diff_peptide_sims) >= len(same_peptide_sims) * 2:
                    break
        if len(diff_peptide_sims) >= sample_size:
            break
    
    metrics = {
        'same_peptide_mean': np.mean(same_peptide_sims) if same_peptide_sims else 0.0,
        'same_peptide_std': np.std(same_peptide_sims) if same_peptide_sims else 0.0,
        'diff_peptide_mean': np.mean(diff_peptide_sims) if diff_peptide_sims else 0.0,
        'diff_peptide_std': np.std(diff_peptide_sims) if diff_peptide_sims else 0.0,
    }
    metrics['separation'] = metrics['same_peptide_mean'] - metrics['diff_peptide_mean']
    
    return metrics


def train(config, args):
    """主训练函数"""
    # 设置随机种子
    set_seed(config['seed'])
    
    # 设置设备
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 创建输出目录
    save_dir = Path(config['output']['save_dir'])
    save_dir.mkdir(exist_ok=True, parents=True)
    log_dir = Path(config['output']['log_dir'])
    log_dir.mkdir(exist_ok=True, parents=True)
    
    # 加载数据
    print("\nLoading data...")
    if args.data_path:
        data_path = args.data_path
    else:
        data_path = config['data']['data_path']
    
    spectra_data, peptide_labels = load_data_from_pickle(data_path)
    
    # 划分训练集和验证集
    train_spectra, train_labels, val_spectra, val_labels = create_train_val_split(
        spectra_data, peptide_labels,
        val_split=config['data']['val_split'],
        random_seed=config['seed']
    )
    
    # 创建数据集
    train_dataset = SpectrumContrastiveDataset(
        train_spectra, train_labels,
        augment=config['data']['augmentation'],
        mode=config['data']['mode'],
        augmentation_params=config['data']['augmentation_params']
    )
    
    val_dataset = SpectrumContrastiveDataset(
        val_spectra, val_labels,
        augment=False,  # 验证时不增强
        mode=config['data']['mode']
    )
    
    # 创建DataLoader
    train_loader, val_loader = create_dataloaders(
        train_dataset, val_dataset,
        batch_size=config['training']['batch_size'],
        num_workers=config['data']['num_workers'],
        mode=config['data']['mode'],
        pin_memory=config['data']['pin_memory']
    )
    
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # 创建模型
    print("\nCreating model...")
    model = create_model(config, device)
    
    # 创建损失函数
    triplet_loss, contrastive_loss = create_loss_functions(config)
    
    # 创建优化器
    optimizer = AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # 创建学习率调度器
    if config['training']['scheduler'] == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=config['training']['epochs'])
    else:
        scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    
    # 训练循环
    best_val_loss = float('inf')
    best_separation = -float('inf')
    
    print("\n" + "="*50)
    print("Starting training...")
    print("="*50)
    
    for epoch in range(1, config['training']['epochs'] + 1):
        print(f"\nEpoch {epoch}/{config['training']['epochs']}")
        print("-" * 50)
        
        # 解冻encoder
        if epoch == config['training']['freeze_epochs'] + 1:
            print("🔓 Unfreezing encoder for fine-tuning...")
            model.unfreeze_encoder()
            
            # 使用不同学习率
            optimizer = AdamW([
                {'params': model.encoder_wrapper.parameters(), 
                 'lr': config['training']['learning_rate'] * 0.1},
                {'params': model.projection_head.parameters(), 
                 'lr': config['training']['learning_rate']}
            ], weight_decay=config['training']['weight_decay'])
            
            scheduler = CosineAnnealingLR(
                optimizer, 
                T_max=config['training']['epochs'] - epoch + 1
            )
        
        # 训练
        train_metrics = train_epoch(
            model, train_loader, optimizer, 
            triplet_loss, contrastive_loss,
            config, device, epoch
        )
        
        # 验证
        if epoch % config['output']['eval_interval'] == 0:
            val_metrics = validate(
                model, val_loader,
                triplet_loss, contrastive_loss,
                config, device
            )
            
            print(f"\nTrain Loss: {train_metrics['loss']:.4f} "
                  f"(Triplet: {train_metrics['triplet_loss']:.4f}, "
                  f"Contrastive: {train_metrics['contrastive_loss']:.4f})")
            print(f"Val Loss: {val_metrics['loss']:.4f} "
                  f"(Triplet: {val_metrics['triplet_loss']:.4f}, "
                  f"Contrastive: {val_metrics['contrastive_loss']:.4f})")
            
            # 评估相似度
            if 'similarity' in config['evaluation']['metrics']:
                sim_metrics = evaluate_similarity(
                    model, val_loader, device,
                    sample_size=config['evaluation']['sample_size']
                )
                print(f"\nSimilarity Metrics:")
                print(f"  Same Peptide: {sim_metrics['same_peptide_mean']:.4f} ± {sim_metrics['same_peptide_std']:.4f}")
                print(f"  Diff Peptide: {sim_metrics['diff_peptide_mean']:.4f} ± {sim_metrics['diff_peptide_std']:.4f}")
                print(f"  Separation: {sim_metrics['separation']:.4f}")
            
            # 保存最佳模型
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                model.save(save_dir / 'best_model.pth')
                print(f"✓ Saved best model (val_loss: {val_metrics['loss']:.4f})")
            
            if 'similarity' in config['evaluation']['metrics']:
                if sim_metrics['separation'] > best_separation:
                    best_separation = sim_metrics['separation']
                    model.save(save_dir / 'best_separation_model.pth')
                    print(f"✓ Saved best separation model (separation: {sim_metrics['separation']:.4f})")
        
        # 定期保存
        if epoch % config['output']['save_interval'] == 0:
            model.save(save_dir / f'model_epoch_{epoch}.pth')
        
        # 更新学习率
        scheduler.step()
    
    print("\n" + "="*50)
    print("Training completed!")
    print("="*50)
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Best separation: {best_separation:.4f}")
    print(f"Models saved to: {save_dir}")


def main():
    parser = argparse.ArgumentParser(description='Train Spectrum Representation Model')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file')
    parser.add_argument('--data_path', type=str, default=None,
                       help='Path to data file (overrides config)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory (overrides config)')
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 覆盖配置
    if args.output_dir:
        config['output']['save_dir'] = args.output_dir
    
    # 训练
    train(config, args)


if __name__ == '__main__':
    main()