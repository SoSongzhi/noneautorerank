"""Loss functions for contrastive learning"""
from .triplet_loss import TripletLoss
from .contrastive_loss import InfoNCELoss, SupConLoss

__all__ = ['TripletLoss', 'InfoNCELoss', 'SupConLoss']