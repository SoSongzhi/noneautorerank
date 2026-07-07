"""
测试训练脚本 - 验证整个训练流程
"""
import os
import sys
import torch
import yaml
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root.parent))

def test_imports():
    """测试所有必要的导入"""
    print("=" * 60)
    print("测试 1: 检查导入")
    print("=" * 60)
    
    try:
        from data import load_data_from_pickle, create_train_val_split
        print("✓ 数据加载函数导入成功")
    except Exception as e:
        print(f"✗ 数据加载函数导入失败: {e}")
        return False
    
    try:
        from models import SpectrumEncoder, ProjectionHead, ContrastiveModel
        print("✓ 模型导入成功")
    except Exception as e:
        print(f"✗ 模型导入失败: {e}")
        return False
    
    try:
        from losses import TripletLoss, InfoNCELoss
        print("✓ 损失函数导入成功")
    except Exception as e:
        print(f"✗ 损失函数导入失败: {e}")
        return False
    
    print()
    return True

def test_config():
    """测试配置文件加载"""
    print("=" * 60)
    print("测试 2: 检查配置文件")
    print("=" * 60)
    
    config_path = project_root / 'config.yaml'
    if not config_path.exists():
        print(f"✗ 配置文件不存在: {config_path}")
        return False
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print(f"✓ 配置文件加载成功")
        print(f"  - 数据路径: {config['data']['data_path']}")
        print(f"  - 训练模式: {config['data']['mode']}")
        print(f"  - Batch size: {config['training']['batch_size']}")
        print()
        return True, config
    except Exception as e:
        print(f"✗ 配置文件加载失败: {e}")
        return False, None

def test_data_file(config):
    """测试数据文件是否存在"""
    print("=" * 60)
    print("测试 3: 检查数据文件")
    print("=" * 60)
    
    data_path = project_root / config['data']['data_path']
    print(f"查找数据文件: {data_path}")
    print(f"绝对路径: {data_path.absolute()}")
    
    if not data_path.exists():
        print(f"✗ 数据文件不存在")
        print(f"  请确保文件存在于: {data_path.absolute()}")
        return False
    
    print(f"✓ 数据文件存在")
    print(f"  文件大小: {data_path.stat().st_size / 1024 / 1024:.2f} MB")
    print()
    return True

def test_data_loading(config):
    """测试数据加载"""
    print("=" * 60)
    print("测试 4: 加载数据")
    print("=" * 60)
    
    try:
        from data import load_data_from_pickle, create_train_val_split
        
        data_path = project_root / config['data']['data_path']
        print(f"从 {data_path} 加载数据...")
        
        spectra_data = load_data_from_pickle(str(data_path))
        print(f"✓ 数据加载成功")
        print(f"  - 总样本数: {len(spectra_data)}")
        
        if len(spectra_data) > 0:
            sample = spectra_data[0]
            print(f"  - 样本示例:")
            print(f"    - Peptide: {sample.get('peptide', 'N/A')}")
            print(f"    - Spectrum shape: {sample.get('spectrum', []).shape if hasattr(sample.get('spectrum', []), 'shape') else 'N/A'}")
        
        # 测试数据分割
        train_data, val_data = create_train_val_split(
            spectra_data,
            train_ratio=config['data']['train_split']
        )
        print(f"✓ 数据分割成功")
        print(f"  - 训练集: {len(train_data)} 样本")
        print(f"  - 验证集: {len(val_data)} 样本")
        print()
        return True, train_data, val_data
        
    except Exception as e:
        print(f"✗ 数据加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None

def test_model_creation(config):
    """测试模型创建"""
    print("=" * 60)
    print("测试 5: 创建模型")
    print("=" * 60)
    
    try:
        from models import ContrastiveModel
        
        model = ContrastiveModel(
            d_model=config['model']['d_model'],
            n_head=config['model']['n_head'],
            dim_feedforward=config['model']['dim_feedforward'],
            n_layers=config['model']['n_layers'],
            dropout=config['model']['dropout'],
            max_charge=config['model']['max_charge'],
            embedding_dim=config['model']['embedding_dim'],
            pooling_method=config['model']['pooling_method']
        )
        
        print(f"✓ 模型创建成功")
        
        # 计算参数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"  - 总参数量: {total_params:,}")
        print(f"  - 可训练参数: {trainable_params:,}")
        print()
        return True, model
        
    except Exception as e:
        print(f"✗ 模型创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_forward_pass(model, train_data):
    """测试前向传播"""
    print("=" * 60)
    print("测试 6: 前向传播")
    print("=" * 60)
    
    try:
        from data import SpectrumContrastiveDataset
        from data.dataloader import triplet_collate_fn
        from torch.utils.data import DataLoader
        
        # 创建小批量数据集
        dataset = SpectrumContrastiveDataset(
            train_data[:10],  # 只用10个样本测试
            mode='triplet',
            augment=False
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=2,
            collate_fn=triplet_collate_fn,
            shuffle=False
        )
        
        # 获取一个batch
        batch = next(iter(dataloader))
        
        print(f"✓ 数据批次创建成功")
        print(f"  - Batch size: {len(batch['anchor_mz_array'])}")
        
        # 前向传播
        model.eval()
        with torch.no_grad():
            anchor_emb = model(
                batch['anchor_mz_array'],
                batch['anchor_intensity_array'],
                batch['anchor_precursor_mz'],
                batch['anchor_precursor_charge']
            )
            
            positive_emb = model(
                batch['positive_mz_array'],
                batch['positive_intensity_array'],
                batch['positive_precursor_mz'],
                batch['positive_precursor_charge']
            )
            
            negative_emb = model(
                batch['negative_mz_array'],
                batch['negative_intensity_array'],
                batch['negative_precursor_mz'],
                batch['negative_precursor_charge']
            )
        
        print(f"✓ 前向传播成功")
        print(f"  - Anchor embedding shape: {anchor_emb.shape}")
        print(f"  - Positive embedding shape: {positive_emb.shape}")
        print(f"  - Negative embedding shape: {negative_emb.shape}")
        print()
        return True
        
    except Exception as e:
        print(f"✗ 前向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_loss_computation(model, train_data):
    """测试损失计算"""
    print("=" * 60)
    print("测试 7: 损失计算")
    print("=" * 60)
    
    try:
        from data import SpectrumContrastiveDataset
        from data.dataloader import triplet_collate_fn
        from torch.utils.data import DataLoader
        from losses import TripletLoss
        
        # 创建数据集和加载器
        dataset = SpectrumContrastiveDataset(
            train_data[:10],
            mode='triplet',
            augment=False
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=2,
            collate_fn=triplet_collate_fn,
            shuffle=False
        )
        
        batch = next(iter(dataloader))
        
        # 前向传播
        model.eval()
        with torch.no_grad():
            anchor_emb = model(
                batch['anchor_mz_array'],
                batch['anchor_intensity_array'],
                batch['anchor_precursor_mz'],
                batch['anchor_precursor_charge']
            )
            
            positive_emb = model(
                batch['positive_mz_array'],
                batch['positive_intensity_array'],
                batch['positive_precursor_mz'],
                batch['positive_precursor_charge']
            )
            
            negative_emb = model(
                batch['negative_mz_array'],
                batch['negative_intensity_array'],
                batch['negative_precursor_mz'],
                batch['negative_precursor_charge']
            )
        
        # 计算损失
        criterion = TripletLoss(margin=1.0)
        loss = criterion(anchor_emb, positive_emb, negative_emb)
        
        print(f"✓ 损失计算成功")
        print(f"  - Triplet Loss: {loss.item():.4f}")
        print()
        return True
        
    except Exception as e:
        print(f"✗ 损失计算失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("开始测试训练流程")
    print("=" * 60 + "\n")
    
    # 测试1: 导入
    if not test_imports():
        print("\n❌ 导入测试失败，停止测试")
        return
    
    # 测试2: 配置
    result = test_config()
    if isinstance(result, tuple):
        success, config = result
        if not success:
            print("\n❌ 配置测试失败，停止测试")
            return
    else:
        print("\n❌ 配置测试失败，停止测试")
        return
    
    # 测试3: 数据文件
    if not test_data_file(config):
        print("\n❌ 数据文件测试失败，停止测试")
        return
    
    # 测试4: 数据加载
    result = test_data_loading(config)
    if isinstance(result, tuple):
        success, train_data, val_data = result
        if not success:
            print("\n❌ 数据加载测试失败，停止测试")
            return
    else:
        print("\n❌ 数据加载测试失败，停止测试")
        return
    
    # 测试5: 模型创建
    result = test_model_creation(config)
    if isinstance(result, tuple):
        success, model = result
        if not success:
            print("\n❌ 模型创建测试失败，停止测试")
            return
    else:
        print("\n❌ 模型创建测试失败，停止测试")
        return
    
    # 测试6: 前向传播
    if not test_forward_pass(model, train_data):
        print("\n❌ 前向传播测试失败，停止测试")
        return
    
    # 测试7: 损失计算
    if not test_loss_computation(model, train_data):
        print("\n❌ 损失计算测试失败，停止测试")
        return
    
    # 所有测试通过
    print("=" * 60)
    print("✅ 所有测试通过！训练流程验证成功")
    print("=" * 60)
    print("\n可以运行以下命令开始训练:")
    print(f"  cd {project_root}")
    print("  python scripts/train.py")
    print()

if __name__ == '__main__':
    main()