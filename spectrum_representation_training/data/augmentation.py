"""
Spectrum Data Augmentation
Spectrum数据增强策略
"""
import numpy as np


class SpectrumAugmentation:
    """
    Spectrum数据增强
    
    Args:
        noise_std: 高斯噪声标准差
        intensity_scale_range: 强度缩放范围 (min, max)
        peak_dropout_rate: Peak dropout概率
        mass_shift_range: m/z偏移范围 (min, max)
    """
    
    def __init__(self, 
                 noise_std=0.01,
                 intensity_scale_range=(0.9, 1.1),
                 peak_dropout_rate=0.05,
                 mass_shift_range=(-0.01, 0.01)):
        self.noise_std = noise_std
        self.intensity_scale_range = intensity_scale_range
        self.peak_dropout_rate = peak_dropout_rate
        self.mass_shift_range = mass_shift_range
    
    def __call__(self, spectrum):
        """
        应用数据增强
        
        Args:
            spectrum: (n_peaks, 2) - [m/z, intensity]
        
        Returns:
            augmented_spectrum: (n_peaks, 2)
        """
        spectrum = spectrum.copy()
        
        # 1. 添加高斯噪声
        if self.noise_std > 0:
            noise = np.random.normal(0, self.noise_std, spectrum.shape)
            spectrum = spectrum + noise
        
        # 2. 强度缩放
        if self.intensity_scale_range is not None:
            scale = np.random.uniform(*self.intensity_scale_range)
            spectrum[:, 1] = spectrum[:, 1] * scale
        
        # 3. Peak dropout
        if self.peak_dropout_rate > 0:
            dropout_mask = np.random.random(len(spectrum)) > self.peak_dropout_rate
            spectrum = spectrum * dropout_mask[:, np.newaxis]
        
        # 4. Mass shift (只对m/z)
        if self.mass_shift_range is not None:
            mass_shift = np.random.uniform(*self.mass_shift_range)
            spectrum[:, 0] = spectrum[:, 0] + mass_shift
        
        # 确保intensity非负
        spectrum[:, 1] = np.maximum(spectrum[:, 1], 0)
        
        # 确保m/z非负
        spectrum[:, 0] = np.maximum(spectrum[:, 0], 0)
        
        return spectrum.astype(np.float32)
    
    def add_noise(self, spectrum, noise_std=None):
        """只添加高斯噪声"""
        if noise_std is None:
            noise_std = self.noise_std
        noise = np.random.normal(0, noise_std, spectrum.shape)
        return (spectrum + noise).astype(np.float32)
    
    def scale_intensity(self, spectrum, scale_range=None):
        """只进行强度缩放"""
        if scale_range is None:
            scale_range = self.intensity_scale_range
        scale = np.random.uniform(*scale_range)
        spectrum = spectrum.copy()
        spectrum[:, 1] = spectrum[:, 1] * scale
        return spectrum.astype(np.float32)
    
    def dropout_peaks(self, spectrum, dropout_rate=None):
        """只进行peak dropout"""
        if dropout_rate is None:
            dropout_rate = self.peak_dropout_rate
        dropout_mask = np.random.random(len(spectrum)) > dropout_rate
        return (spectrum * dropout_mask[:, np.newaxis]).astype(np.float32)
    
    def shift_mass(self, spectrum, shift_range=None):
        """只进行m/z偏移"""
        if shift_range is None:
            shift_range = self.mass_shift_range
        mass_shift = np.random.uniform(*shift_range)
        spectrum = spectrum.copy()
        spectrum[:, 0] = spectrum[:, 0] + mass_shift
        spectrum[:, 0] = np.maximum(spectrum[:, 0], 0)
        return spectrum.astype(np.float32)