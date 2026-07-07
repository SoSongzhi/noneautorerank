"""
Contrastive Spectrum Model
完整的对比学习模型
"""
import torch
import torch.nn as nn
from .spectrum_encoder import SpectrumEncoderWrapper
from .projection_head import ProjectionHead


class ContrastiveSpectrumModel(nn.Module):
    """
    完整的对比学习模型
    
    Args:
        pretrained_encoder: PiPrime预训练encoder
        encoder_dim: Encoder输出维度
        embedding_dim: 最终embedding维度
        pooling: Pooling策略 ('cls', 'mean', 'max', 'attention')
        freeze_encoder: 是否冻结encoder
        projection_layers: Projection head的层数
    """
    
    def __init__(self, pretrained_encoder=None, encoder_dim=512, 
                 embedding_dim=256, pooling='cls', freeze_encoder=True,
                 projection_layers=3):
        super().__init__()
        
        self.encoder_wrapper = SpectrumEncoderWrapper(
            pretrained_encoder=pretrained_encoder,
            pooling=pooling,
            freeze=freeze_encoder,
            dim_model=encoder_dim
        )
        
        self.projection_head = ProjectionHead(
            input_dim=encoder_dim,
            hidden_dim=encoder_dim,
            output_dim=embedding_dim,
            num_layers=projection_layers
        )
        
        self.encoder_dim = encoder_dim
        self.embedding_dim = embedding_dim
    
    def forward(self, spectra):
        """
        Args:
            spectra: (batch_size, n_peaks, 2) - [m/z, intensity]
        
        Returns:
            embeddings: (batch_size, embedding_dim) - L2 normalized
        """
        pooled = self.encoder_wrapper(spectra)
        embeddings = self.projection_head(pooled)
        return embeddings
    
    def encode(self, spectra):
        """
        便捷方法：编码spectrum
        
        Args:
            spectra: (batch_size, n_peaks, 2) or (n_peaks, 2)
        
        Returns:
            embeddings: (batch_size, embedding_dim) or (embedding_dim,)
        """
        # 处理单个spectrum的情况
        if spectra.dim() == 2:
            spectra = spectra.unsqueeze(0)
            embeddings = self.forward(spectra)
            return embeddings.squeeze(0)
        return self.forward(spectra)
    
    def encode_batch(self, spectra_list, batch_size=32, device='cuda'):
        """
        批量编码多个spectrum
        
        Args:
            spectra_list: List of spectrum tensors
            batch_size: 批处理大小
            device: 设备
        
        Returns:
            embeddings: (n_spectra, embedding_dim)
        """
        self.eval()
        all_embeddings = []
        
        with torch.no_grad():
            for i in range(0, len(spectra_list), batch_size):
                batch = spectra_list[i:i+batch_size]
                
                # Pad到相同长度
                max_peaks = max(s.shape[0] for s in batch)
                padded_batch = []
                for spectrum in batch:
                    if spectrum.shape[0] < max_peaks:
                        padding = torch.zeros(max_peaks - spectrum.shape[0], 2)
                        spectrum = torch.cat([spectrum, padding], dim=0)
                    padded_batch.append(spectrum)
                
                batch_tensor = torch.stack(padded_batch).to(device)
                embeddings = self.forward(batch_tensor)
                all_embeddings.append(embeddings.cpu())
        
        return torch.cat(all_embeddings, dim=0)
    
    def unfreeze_encoder(self):
        """解冻encoder用于fine-tuning"""
        self.encoder_wrapper.unfreeze()
    
    def save(self, path):
        """保存模型"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'encoder_dim': self.encoder_dim,
            'embedding_dim': self.embedding_dim,
        }, path)
    
    @classmethod
    def load_from_checkpoint(cls, path, device='cuda'):
        """从checkpoint加载模型"""
        checkpoint = torch.load(path, map_location=device)
        
        model = cls(
            encoder_dim=checkpoint['encoder_dim'],
            embedding_dim=checkpoint['embedding_dim'],
            freeze_encoder=False  # 加载时不冻结
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        
        return model
    
    @property
    def device(self):
        """获取当前设备"""
        return next(self.parameters()).device