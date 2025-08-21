"""
metrics.py - 评估指标（MAE, PCC）计算

根据需求表实现评估指标计算，包括：
- MAE（平均绝对误差）：衡量预测值与真实值的平均绝对差异
- PCC（皮尔逊相关系数）：衡量预测值与真实值的线性相关性
"""

import numpy as np
from scipy.stats import pearsonr
from typing import Tuple, Dict, Union, Any
import warnings

def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    计算平均绝对误差(MAE)
    
    Args:
        y_true: 真实值数组，应在0-10范围内
        y_pred: 预测值数组，应在0-10范围内
        
    Returns:
        mae: 平均绝对误差，值越小表示预测越准确
        
    Raises:
        TypeError: 输入类型错误
        ValueError: 输入数组维度不匹配或值域超出范围
    """
    # 类型检查
    if not isinstance(y_true, (np.ndarray, list)) or not isinstance(y_pred, (np.ndarray, list)):
        raise TypeError("输入必须是numpy数组或列表")
    
    # 转换为numpy数组
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    
    # 维度检查
    if y_true.shape != y_pred.shape:
        raise ValueError(f"输入数组维度不匹配: y_true {y_true.shape}, y_pred {y_pred.shape}")
    
    # 范围检查
    if (y_true < 0).any() or (y_true > 10).any() or (y_pred < 0).any() or (y_pred > 10).any():
        warnings.warn("存在超出0-10范围的评分值")
        y_true = np.clip(y_true, 0, 10)
        y_pred = np.clip(y_pred, 0, 10)
    
    # 计算MAE
    mae = np.mean(np.abs(y_true - y_pred))
    return float(mae)

def calculate_pcc(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    """
    计算皮尔逊相关系数(PCC)
    
    Args:
        y_true: 真实值数组，应在0-10范围内
        y_pred: 预测值数组，应在0-10范围内
        
    Returns:
        pcc: 皮尔逊相关系数，范围[-1, 1]，越接近1表示正相关性越强
        p_value: p值，用于检验相关系数的显著性
        
    Raises:
        TypeError: 输入类型错误
        ValueError: 输入数组维度不匹配、数据不足或存在常数序列
    """
    # 类型检查
    if not isinstance(y_true, (np.ndarray, list)) or not isinstance(y_pred, (np.ndarray, list)):
        raise TypeError("输入必须是numpy数组或列表")
    
    # 转换为numpy数组
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    
    # 维度检查
    if y_true.shape != y_pred.shape:
        raise ValueError(f"输入数组维度不匹配: y_true {y_true.shape}, y_pred {y_pred.shape}")
    
    # 数据量检查
    if len(y_true) < 2:
        raise ValueError("计算相关系数需要至少2个数据点")
    
    # 常数序列检查
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        raise ValueError("输入序列不能是常数序列")
    
    try:
        pcc, p_value = pearsonr(y_true, y_pred)
        return float(pcc), float(p_value)
    except Exception as e:
        raise RuntimeError(f"计算皮尔逊相关系数失败: {str(e)}")

def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, 
                        verbose: bool = False) -> Dict[str, Union[float, Dict[str, Any]]]:
    """
    评估预测结果，计算综合评估指标
    
    Args:
        y_true: 真实值数组，应在0-10范围内
        y_pred: 预测值数组，应在0-10范围内
        verbose: 是否返回详细信息，默认False
        
    Returns:
        metrics: 评估指标字典，包含：
            - mae: 平均绝对误差
            - pcc: 皮尔逊相关系数
            - stats: (verbose=True时) 详细统计信息
            
    Raises:
        ValueError: 当输入数据无效时
    """
    try:
        # 基础指标计算
        mae = calculate_mae(y_true, y_pred)
        pcc, p_value = calculate_pcc(y_true, y_pred)
        
        # 构建基础指标字典
        metrics = {
            'mae': float(mae),
            'pcc': float(pcc)
        }
        
        # 添加详细统计信息（如果需要）
        if verbose:
            metrics['stats'] = {
                'p_value': float(p_value),
                'sample_size': len(y_true),
                'score_range': {
                    'true': [float(np.min(y_true)), float(np.max(y_true))],
                    'pred': [float(np.min(y_pred)), float(np.max(y_pred))]
                },
                'mean_scores': {
                    'true': float(np.mean(y_true)),
                    'pred': float(np.mean(y_pred))
                },
                'std_scores': {
                    'true': float(np.std(y_true)),
                    'pred': float(np.std(y_pred))
                }
            }
        
        return metrics
        
    except Exception as e:
        raise ValueError(f"评估计算失败: {str(e)}")