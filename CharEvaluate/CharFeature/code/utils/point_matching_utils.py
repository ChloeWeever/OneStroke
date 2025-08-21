"""
point_matching_utils.py - CDP算法的工具函数集
"""

import numpy as np
import torch
from typing import Tuple, List, Dict, Optional
import cv2
from scipy.spatial.distance import cdist


def validate_keypoints(points: np.ndarray) -> bool:
    """
    验证关键点数据格式是否有效
    
    Args:
        points: shape为(N, 2)的numpy数组，表示N个关键点的(x,y)坐标
        
    Returns:
        bool: 数据格式是否有效
    """
    if not isinstance(points, np.ndarray):
        return False
    if points.ndim != 2 or points.shape[1] != 2:
        return False
    if not np.isfinite(points).all():
        return False
    return True


def convert_rcnn_to_points(detections: Dict) -> np.ndarray:
    """
    将Faster R-CNN的检测结果转换为标准关键点格式
    
    Args:
        detections: Faster R-CNN输出的检测结果字典
        
    Returns:
        np.ndarray: shape为(N, 2)的关键点坐标数组
    """
    # 提取边界框中心点作为关键点
    boxes = detections.get("pred_boxes", [])
    if isinstance(boxes, torch.Tensor):
        boxes = boxes.cpu().numpy()
    
    # 计算边界框中心点
    centers = []
    for box in boxes:
        x1, y1, x2, y2 = box
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        centers.append([center_x, center_y])
    
    return np.array(centers)


def normalize_points(points: np.ndarray) -> Tuple[np.ndarray, dict]:
    """
    对点集进行归一化处理
    
    Args:
        points: 原始点集坐标
        
    Returns:
        normalized_points: 归一化后的点集
        normalize_params: 归一化参数，用于恢复原始坐标
    """
    # 计算中心点
    centroid = np.mean(points, axis=0)
    
    # 计算缩放因子
    scale = np.sqrt(np.sum((points - centroid) ** 2) / points.shape[0])
    
    # 归一化
    normalized_points = (points - centroid) / scale
    
    params = {
        'centroid': centroid,
        'scale': scale
    }
    
    return normalized_points, params


def denormalize_points(points: np.ndarray, params: dict) -> np.ndarray:
    """
    将归一化的点集恢复到原始坐标系
    
    Args:
        points: 归一化后的点集
        params: normalize_points返回的参数字典
        
    Returns:
        np.ndarray: 原始坐标系下的点集
    """
    return points * params['scale'] + params['params']


def remove_duplicate_points(points: np.ndarray, threshold: float = 1e-5) -> np.ndarray:
    """
    移除重复的关键点
    
    Args:
        points: 原始点集
        threshold: 判定为重复点的距离阈值
        
    Returns:
        np.ndarray: 去重后的点集
    """
    if len(points) == 0:
        return points
        
    # 计算点之间的欧氏距离
    distances = cdist(points, points)
    
    # 找出重复点的索引
    duplicate_indices = []
    for i in range(len(points)):
        if i in duplicate_indices:
            continue
        for j in range(i + 1, len(points)):
            if distances[i, j] < threshold:
                duplicate_indices.append(j)
    
    # 保留非重复点
    mask = np.ones(len(points), dtype=bool)
    mask[duplicate_indices] = False
    
    return points[mask]


def calculate_pairwise_distances(points1: np.ndarray, 
                               points2: np.ndarray) -> np.ndarray:
    """
    计算两个点集之间的欧氏距离矩阵
    
    Args:
        points1: 第一个点集，shape为(N, 2)
        points2: 第二个点集，shape为(M, 2)
        
    Returns:
        np.ndarray: shape为(N, M)的距离矩阵
    """
    return cdist(points1, points2, metric='euclidean')


def visualize_point_matches(template_img: np.ndarray,
                          copy_img: np.ndarray,
                          template_points: np.ndarray,
                          copy_points: np.ndarray,
                          matches: List[Tuple[int, int]],
                          save_path: Optional[str] = None) -> np.ndarray:
    """
    可视化关键点匹配结果
    
    Args:
        template_img: 模板图像
        copy_img: 摹写图像
        template_points: 模板关键点
        copy_points: 摹写关键点
        matches: 匹配对列表，每个元素为(template_idx, copy_idx)
        save_path: 可选的保存路径
        
    Returns:
        np.ndarray: 可视化结果图像
    """
    # 创建拼接图像
    h1, w1 = template_img.shape[:2]
    h2, w2 = copy_img.shape[:2]
    vis_img = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    
    # 复制原图到可视化图像
    vis_img[:h1, :w1] = template_img
    vis_img[:h2, w1:w1+w2] = copy_img
    
    # 绘制匹配线
    for t_idx, c_idx in matches:
        pt1 = tuple(map(int, template_points[t_idx]))
        pt2 = tuple(map(int, copy_points[c_idx]))
        pt2 = (pt2[0] + w1, pt2[1])  # 调整第二幅图中点的x坐标
        
        # 绘制连线
        cv2.line(vis_img, pt1, pt2, (0, 255, 0), 1)
        # 绘制关键点
        cv2.circle(vis_img, pt1, 3, (0, 0, 255), -1)
        cv2.circle(vis_img, pt2, 3, (0, 0, 255), -1)
    
    # 保存结果
    if save_path:
        cv2.imwrite(save_path, vis_img)
    
    return vis_img


def calculate_registration_error(source_points: np.ndarray,
                               target_points: np.ndarray,
                               transformation: np.ndarray) -> float:
    """
    计算点集配准误差
    
    Args:
        source_points: 原始点集
        target_points: 目标点集
        transformation: 变换矩阵
        
    Returns:
        float: 平均配准误差
    """
    # 应用变换
    if transformation.shape[0] == 2:  # 仿射变换
        transformed_points = np.dot(source_points, transformation)
    else:  # 透视变换
        homogeneous = np.column_stack([source_points, np.ones(len(source_points))])
        transformed = np.dot(homogeneous, transformation.T)
        transformed_points = transformed[:, :2] / transformed[:, 2:]
    
    # 计算误差
    errors = np.sqrt(np.sum((transformed_points - target_points) ** 2, axis=1))
    mean_error = np.mean(errors)
    
    return mean_error


def remove_outliers(template_points: np.ndarray, 
                   copy_points: np.ndarray, 
                   correspondence_matrix: np.ndarray,
                   threshold: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    移除离群点
    
    Args:
        template_points: 模板点集
        copy_points: 摹写点集
        correspondence_matrix: 对应关系矩阵
        threshold: 离群点阈值
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: 去除离群点后的点集
    """
    # 计算对应点对的距离
    distances = []
    valid_template_points = []
    valid_copy_points = []
    
    for i in range(len(template_points)):
        for j in range(len(copy_points)):
            if correspondence_matrix[i, j]:
                dist = np.linalg.norm(template_points[i] - copy_points[j])
                distances.append(dist)
    
    # 计算距离的中位数和标准差
    if len(distances) > 0:
        median_dist = np.median(distances)
        std_dist = np.std(distances)
        
        # 确定离群点阈值
        outlier_threshold = median_dist + threshold * std_dist
        
        # 移除离群点
        idx = 0
        for i in range(len(template_points)):
            for j in range(len(copy_points)):
                if correspondence_matrix[i, j]:
                    if distances[idx] <= outlier_threshold:
                        valid_template_points.append(template_points[i])
                        valid_copy_points.append(copy_points[j])
                    idx += 1
    
    return np.array(valid_template_points), np.array(valid_copy_points)
