"""
regression.py - LightGBM回归预测

实现SRAFE方法的美学评分回归器，使用LightGBM模型进行预测
"""

import numpy as np
import lightgbm as lgb
import os
from typing import Union

class AestheticRegressor:
    """
    美学评分回归器类，用于SRAFE方法的最终评分预测
    """
    
    def __init__(self, model_path: str = None):
        """
        初始化回归器
        
        Args:
            model_path: LightGBM模型文件路径
        """
        self.model = None
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            if model_path:
                print(f"Warning: Model file not found: {model_path}")
            # 可以在这里初始化一个默认模型或保持为None
    
    def load_model(self, model_path: str):
        """
        加载预训练的LightGBM模型
        
        Args:
            model_path: 模型文件路径 (.txt格式)
        """
        try:
            self.model = lgb.Booster(model_file=model_path)
            print(f"Model loaded successfully from {model_path}")
        except Exception as e:
            print(f"Error loading model from {model_path}: {e}")
            self.model = None
    
    def predict(self, features: Union[np.ndarray, list]) -> float:
        """
        预测美学评分，输入为23维融合特征
        
        Args:
            features: 特征向量 (23维融合特征)
                - 12维手工特征
                - 10维深度特征差值
                - 1维欧氏距离
            
        Returns:
            score: 美学评分 (0-10)
        """
        if self.model is None:
            print("Warning: No model loaded, returning default score (5.0)")
            return 5.0
        
        # 确保特征是numpy数组
        if not isinstance(features, np.ndarray):
            features = np.array(features)
        
        # 确保特征是2D数组 (n_samples, n_features)
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        # 验证特征维度
        if features.shape[1] != 23:
            raise ValueError(f"Expected 23 features, got {features.shape[1]}")
        
        # 使用训练好的模型进行预测
        try:
            # 使用最佳迭代次数进行预测（如果有早停）
            if hasattr(self.model, 'best_iteration') and self.model.best_iteration > 0:
                score = self.model.predict(features, num_iteration=self.model.best_iteration)[0]
            else:
                score = self.model.predict(features)[0]
            
            # 确保评分在0-10范围内
            score = np.clip(score, 0, 10)
            return float(score)
        except Exception as e:
            print(f"Error in prediction: {e}")
            return 5.0  # 返回默认评分

def create_dummy_regressor():
    """
    创建示例回归器用于测试
    
    Returns:
        regressor: 示例回归器实例
    """
    regressor = AestheticRegressor()
    # 这里可以加载示例模型或保持为None状态用于测试
    return regressor

'''
# 用于测试的示例代码
if __name__ == "__main__":
    # 创建回归器
    regressor = AestheticRegressor('models/aesthetic_regressor.txt')
    
    # 生成示例特征进行预测 (23维SRAFE特征)
    dummy_features = np.random.rand(23)  # 23维特征
    score = regressor.predict(dummy_features)
    print(f"Predicted aesthetic score: {score:.2f}")
    
    # 测试特征维度验证
    try:
        wrong_features = np.random.rand(10)  # 错误的维度
        score = regressor.predict(wrong_features)
    except ValueError as e:
        print(f"Feature dimension validation works: {e}")
'''