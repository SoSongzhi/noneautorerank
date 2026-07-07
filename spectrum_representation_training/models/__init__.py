"""Models for spectrum representation learning"""
from .spectrum_encoder import SpectrumEncoderWrapper
from .projection_head import ProjectionHead
from .contrastive_model import ContrastiveSpectrumModel

__all__ = ['SpectrumEncoderWrapper', 'ProjectionHead', 'ContrastiveSpectrumModel']