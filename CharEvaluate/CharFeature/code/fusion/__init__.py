"""
fusion/__init__.py - 特征融合模块初始化文件
"""

from .feature_fusion import fuse_features
from .regression import AestheticRegressor

__all__ = [
    'fuse_features',
    'AestheticRegressor'
]
