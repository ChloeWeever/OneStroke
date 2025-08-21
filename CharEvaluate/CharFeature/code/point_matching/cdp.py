"""
cdp.py - Coherent Point Drift (CDP)算法实现
"""

import numpy as np
import yaml
from typing import Tuple, Dict
from ..utils.point_matching_utils import (
    validate_keypoints,
    normalize_points,
    denormalize_points,
    remove_duplicate_points,
    calculate_pairwise_distances
)


class CDPMatcher:
    """CDP点集匹配器"""
    
    def __init__(self, config_path: str):
        """
        初始化CDP匹配器
        
        Args:
            config_path: 配置文件路径
        """
        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # 提取算法参数
        self.max_iterations = self.config['algorithm']['max_iterations']
        self.tolerance = self.config['algorithm']['tolerance']
        self.beta = self.config['algorithm']['beta']
        self.lambda_param = self.config['algorithm']['lambda_param']
        self.outliers_ratio = self.config['algorithm']['outliers_ratio']
        
    def _initialize_sigma2(self, 
                          template_points: np.ndarray, 
                          copy_points: np.ndarray) -> float:
        """
        初始化高斯核宽度参数
        """
        # 计算点集直径
        template_diam = np.max(calculate_pairwise_distances(
            template_points, template_points))
        copy_diam = np.max(calculate_pairwise_distances(
            copy_points, copy_points))
        
        # 使用点集直径的平均值作为初始sigma2
        sigma2 = (template_diam + copy_diam) / 4
        return sigma2
    
    def _e_step(self, 
                template_points: np.ndarray,
                copy_points: np.ndarray,
                sigma2: float) -> Tuple[np.ndarray, float]:
        """
        E步：计算概率矩阵P
        """
        N, D = template_points.shape
        M, _ = copy_points.shape
        
        # 计算距离矩阵
        dist = calculate_pairwise_distances(template_points, copy_points)
        
        # 计算高斯核
        kernel = np.exp(-dist / (2 * sigma2))
        
        # 计算先验概率
        c = (2 * np.pi * sigma2) ** (D/2) * self.outliers_ratio / (1 - self.outliers_ratio)
        c = c / M
        
        # 计算后验概率矩阵P
        P = kernel / (np.sum(kernel, axis=1)[:, None] + c)
        
        # 计算负对数似然
        Pt1 = np.sum(P, axis=0)
        P1 = np.sum(P, axis=1)
        Np = np.sum(P1)
        
        E = (dist * P).sum() / (2 * sigma2) + Np * D * np.log(sigma2) / 2
        
        return P, E
    
    def _m_step(self,
                template_points: np.ndarray,
                copy_points: np.ndarray,
                P: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        M步：更新变换参数
        """
        Np = np.sum(P)
        D = template_points.shape[1]  # 维度（2D情况下为2）
        
        mu_x = np.sum(P @ copy_points, axis=0) / Np
        mu_y = np.sum(template_points * P.sum(axis=1)[:, None], axis=0) / Np
        
        # 中心化
        x_centered = copy_points - mu_x
        y_centered = template_points - mu_y
        
        # 计算协方差矩阵
        A = x_centered.T @ (P.T @ y_centered)
        
        # 添加正则化项
        I = np.eye(D)
        A += self.lambda_param * I
        
        # SVD分解
        U, _, Vt = np.linalg.svd(A)
        
        # 计算旋转矩阵
        R = Vt.T @ U.T
        
        # 计算平移向量
        t = mu_x - R @ mu_y
        
        # 计算距离矩阵（用于更新sigma2）
        dist = calculate_pairwise_distances(template_points, copy_points)
        
        # 更新sigma2
        sigma2 = np.sum(dist * P) / (Np * D)
        
        # 返回变换参数和更新的sigma2
        transform = np.column_stack([R, t.reshape(-1, 1)])
        return transform, sigma2 
    
    def match_points(self,
                    template_points: np.ndarray,
                    copy_points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        执行点集匹配
        
        Args:
            template_points: 模板关键点坐标，shape=(N, 2)
            copy_points: 摹写图关键点坐标，shape=(M, 2)
            
        Returns:
            matched_points: 配准后的copy_points坐标
            correspondence_matrix: 对应关系矩阵
            mean_error: 平均配准误差
        """
        # 验证输入
        if not validate_keypoints(template_points) or not validate_keypoints(copy_points):
            raise ValueError("Invalid keypoint format")
        
        # 预处理
        if self.config['preprocessing']['remove_duplicates']:
            template_points = remove_duplicate_points(template_points)
            copy_points = remove_duplicate_points(copy_points)
            
        if self.config['preprocessing']['normalize']:
            template_points, template_params = normalize_points(template_points)
            copy_points, copy_params = normalize_points(copy_points)
        
        # 初始化参数
        sigma2 = self._initialize_sigma2(template_points, copy_points)
        prev_energy = float('inf')
        
        # 迭代优化
        for iteration in range(self.max_iterations):
            # E步
            P, energy = self._e_step(template_points, copy_points, sigma2)
            
            # 检查收敛
            if abs(energy - prev_energy) < self.tolerance:
                break
            prev_energy = energy
            
            # M步
            transform, sigma2 = self._m_step(template_points, copy_points, P)
            
            # 应用变换
            homogeneous_points = np.column_stack([copy_points, np.ones(len(copy_points))])
            copy_points = (transform @ homogeneous_points.T).T
        
        # 恢复原始坐标系
        if self.config['preprocessing']['normalize']:
            copy_points = denormalize_points(copy_points, copy_params)
            template_points = denormalize_points(template_points, template_params)
        
        # 计算最终的对应关系和误差
        final_distances = calculate_pairwise_distances(template_points, copy_points)
        correspondence_matrix = final_distances < self.config['matching']['max_distance']
        mean_error = np.mean(final_distances[correspondence_matrix])
        
        return copy_points, correspondence_matrix, mean_error


def match_keypoints(template_points: np.ndarray,
                   copy_points: np.ndarray,
                   config_path: str) -> Dict:
    """
    公共接口：执行关键点匹配
    
    Args:
        template_points: 模板关键点坐标
        copy_points: 摹写图关键点坐标
        config_path: 配置文件路径
        
    Returns:
        dict: 包含匹配结果的字典
    """
    # 创建匹配器实例
    matcher = CDPMatcher(config_path)
    
    # 执行匹配
    matched_points, correspondence_matrix, mean_error = matcher.match_points(
        template_points, copy_points)
    
    # 返回结果
    return {
        'matched_points': matched_points,
        'correspondence_matrix': correspondence_matrix,
        'mean_error': mean_error
    }
