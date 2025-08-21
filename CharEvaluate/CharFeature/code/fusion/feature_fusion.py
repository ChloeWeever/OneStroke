"""
feature_fusion.py - 特征融合逻辑
"""

import numpy as np
from typing import Dict, Any

def fuse_features(features: Dict[str, Any]) -> np.ndarray:
    """
    融合手工特征和深度特征
    
    Args:
        features: 包含各种特征的字典
        
    Returns:
        fused_features: 融合后的特征向量 (23维)
    """
    # 获取手工特征 (12维)
    handcrafted_features = features['handcrafted_features']
    
    # 获取深度特征
    deep_features_template = features['deep_features_template']  # 10维
    deep_features_copy = features['deep_features_copy']          # 10维
    
    # 如果没有深度特征，使用零向量填充
    if deep_features_template is None or deep_features_copy is None:
        deep_features_template = np.zeros(10)
        deep_features_copy = np.zeros(10)
    
    # 计算深度特征差值 (10维)
    feature_deep = deep_features_template - deep_features_copy
    
    # 计算欧氏距离 (1维)
    euclidean = np.linalg.norm(deep_features_template - deep_features_copy)
    
    # 融合所有特征: [handcrafted(12) + feature_deep(10) + euclidean(1)] = 23维
    fused_features = np.concatenate([
        handcrafted_features,
        feature_deep,
        [euclidean]
    ])
    
    return fused_features

'''
# 用于测试的示例代码
if __name__ == "__main__":
    # 示例特征字典
    example_features = {
        'handcrafted_features': np.random.rand(12),
        'deep_features_template': np.random.rand(10),
        'deep_features_copy': np.random.rand(10),
        'feature_deep': None,
        'euclidean': None
    }
    
    # 融合特征
    fused = fuse_features(example_features)
    print(f"融合特征维度: {fused.shape}")
    print(f"融合特征: {fused}")
'''