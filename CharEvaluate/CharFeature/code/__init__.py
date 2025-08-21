"""
code/__init__.py - 书法美学评估系统主模块初始化文件
"""

# 导入预处理模块
from .preprocessing import preprocess_image, register_images

# 导入特征计算模块
from .features import (
    calculate_f1_overlap_ratio,
    calculate_f2_shape_aesthetic_score,
    calculate_f3_convex_hull_overlap,
    calculate_projection_similarity,
    calculate_f4_to_f11,
    calculate_f12_keypoint_distance,
    detect_keypoints,
    extract_all_features,
    extract_features_for_srafe,
    get_feature_names
)

# 导入SRN模块
from .srn import SiameseRegressionNetwork, SRNLoss, get_backbone

# 导入特征融合模块
from .fusion import fuse_features, AestheticRegressor

# 导入评估模块
from .evaluation import calculate_mae, calculate_pcc

# 导入工具模块
from .utils import CalligraphyDataset, Config, DEFAULT_TRAINING_CONFIG, DEFAULT_INFERENCE_CONFIG

# 导入主程序接口
from .main import evaluate_calligraphy_aesthetic, batch_evaluate

__all__ = [
    # 预处理模块
    'preprocess_image',
    'register_images',
    
    # 特征计算模块
    'calculate_f1_overlap_ratio',
    'calculate_f2_shape_aesthetic_score',
    'calculate_f3_convex_hull_overlap',
    'calculate_projection_similarity',
    'calculate_f4_to_f11',
    'calculate_f12_keypoint_distance',
    'detect_keypoints',
    'extract_all_features',
    'extract_features_for_srafe',
    'get_feature_names',
    
    # SRN模块
    'SiameseRegressionNetwork',
    'SRNLoss',
    'get_backbone',
    
    # 特征融合模块
    'fuse_features',
    'AestheticRegressor',
    
    # 评估模块
    'calculate_mae',
    'calculate_pcc',
    
    # 工具模块
    'CalligraphyDataset',
    'Config',
    'DEFAULT_TRAINING_CONFIG',
    'DEFAULT_INFERENCE_CONFIG',
    
    # 主程序接口
    'evaluate_calligraphy_aesthetic',
    'batch_evaluate'
]