"""
features.py - 12个书法美学特征计算函数

每个函数接收预处理和配准后的图像作为输入（500x500的二值图像），
返回0-10之间的浮点数评分。
"""

import numpy as np
import cv2
from scipy.stats import pearsonr
from typing import Tuple, List, Dict, Any
import os
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from .point_matching import match_keypoints, convert_rcnn_to_points
from .faster_RCNN_Detectron.predict_faster_rcnn import setup_predictor

def calculate_f1_overlap_ratio(template: np.ndarray, copy: np.ndarray) -> float:
    """
    计算f1: 重叠率 (Overlap ratio)
    
    f1 = |SC ∩ ST| / |SC ∪ ST|
    
    Args:
        template: 模板图像 (二值图像，0为背景，255为前景)
        copy: 摹本图像 (二值图像，0为背景，255为前景)
        
    Returns:
        f1: 重叠率 [0, 1]
        
    Raises:
        TypeError: 如果输入不是numpy数组
        ValueError: 如果输入图像尺寸不一致
    """
    # 输入类型检查
    if not isinstance(template, np.ndarray) or not isinstance(copy, np.ndarray):
        raise TypeError("输入必须是numpy数组")
    
    # 尺寸一致性检查
    if template.shape != copy.shape:
        raise ValueError(f"模板图像 {template.shape} 和摹本图像 {copy.shape} 尺寸不一致")
    
    # 确保图像是二值图像
    _, template_bin = cv2.threshold(template, 127, 255, cv2.THRESH_BINARY)
    _, copy_bin = cv2.threshold(copy, 127, 255, cv2.THRESH_BINARY)
    
    # 计算交集和并集
    intersection = cv2.bitwise_and(template_bin, copy_bin)
    union = cv2.bitwise_or(template_bin, copy_bin)
    
    # 计算像素数量
    intersection_count = np.count_nonzero(intersection)
    union_count = np.count_nonzero(union)
    
    # 计算f1
    f1 = intersection_count / union_count if union_count > 0 else 0.0
    
    return f1

def calculate_f2_shape_aesthetic_score(f1: float) -> float:
    """
    计算f2: 形状美学评分 (Shape aesthetic score)
    
    f2 = 10 * f1 * (2 - f1)
    
    Args:
        f1: 重叠率，应在[0, 1]范围内
        
    Returns:
        f2: 形状美学评分 [0, 10]
        
    Raises:
        TypeError: 如果f1不是浮点数
        ValueError: 如果f1不在[0, 1]范围内
    """
    # 参数类型检查
    if not isinstance(f1, (int, float)):
        raise TypeError("f1必须是数值类型")
    
    # 参数范围检查
    if not 0 <= f1 <= 1:
        raise ValueError("f1必须在0到1之间")
        
    # 根据论文公式(7): f2 = 10 * f1 * (2 - f1)
    f2 = 10 * f1 * (2 - f1)
    
    # 确保f2在0-10范围内
    f2 = max(0.0, min(10.0, f2))
    
    return f2

def calculate_convex_hull(image: np.ndarray) -> np.ndarray:
    """
    计算图像的凸包
    
    Args:
        image: 二值图像
        
    Returns:
        convex_hull: 凸包掩码图像
    """
    # 获取前景像素的坐标
    _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return np.zeros_like(image)
    
    # 取最大轮廓
    largest_contour = max(contours, key=cv2.contourArea)
    
    # 计算凸包
    hull = cv2.convexHull(largest_contour)
    
    # 创建凸包掩码
    h, w = image.shape[:2]
    hull_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(hull_mask, [hull], 0, 255, -1)
    
    return hull_mask

def calculate_f3_convex_hull_overlap(template: np.ndarray, copy: np.ndarray) -> float:
    """
    计算f3: 凸包重叠率
    
    f3 = |SC ∩ ST| / |SC ∪ ST|，应用于凸包
    
    Args:
        template: 模板图像 (二值图像)
        copy: 摹本图像 (二值图像)
        
    Returns:
        f3: 凸包重叠率 [0, 1]
    """
    # 计算模板和摹本的凸包
    template_hull = calculate_convex_hull(template)
    copy_hull = calculate_convex_hull(copy)
    
    # 计算凸包的重叠率
    intersection = cv2.bitwise_and(template_hull, copy_hull)
    union = cv2.bitwise_or(template_hull, copy_hull)
    
    intersection_count = np.count_nonzero(intersection)
    union_count = np.count_nonzero(union)
    
    f3 = intersection_count / union_count if union_count > 0 else 0.0
    
    return f3

def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    """
    旋转图像（保持完整内容）
    Args:
        image: 输入图像
        angle: 旋转角度（度）
    Returns:
        rotated: 旋转后的图像
    """
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    
    # 计算旋转后的边界尺寸
    radians = np.deg2rad(angle)
    cos = np.abs(np.cos(radians))
    sin = np.abs(np.sin(radians))
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    
    # 调整旋转矩阵
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    M[0, 2] += (nW - w) / 2
    M[1, 2] += (nH - h) / 2
    
    # 旋转图像
    rotated = cv2.warpAffine(image, M, (nW, nH),
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT,
                             borderValue=0)
    return rotated

def calculate_projection(image: np.ndarray, angle: float) -> np.ndarray:
    """
    计算指定角度的投影（沿基线方向的线积分）
    Args:
        image: 二值图像（0为背景，255为前景）
        angle: 投影角度（度），应在[-180, 180]范围内
    Returns:
        projection: 投影向量
    Raises:
        TypeError: 如果输入类型不正确
        ValueError: 如果角度超出范围
    """
    # 参数类型检查
    if not isinstance(image, np.ndarray):
        raise TypeError("image必须是numpy数组")
    if not isinstance(angle, (int, float)):
        raise TypeError("angle必须是数值类型")
        
    # 角度范围检查
    if not -180 <= angle <= 180:
        raise ValueError("angle必须在-180到180度之间")
    
    # 旋转图像使基线方向水平
    rotated = rotate_image(image, -angle)
    
    # 沿旋转后的x轴（列方向）投影（线积分）
    projection = np.sum(rotated, axis=0) / 255.0  # 归一化
    
    return projection

def calculate_projection_similarity(template_proj: np.ndarray, 
                                   copy_proj: np.ndarray) -> Tuple[float, float]:
    """
    计算两个投影向量的相似度
    Args:
        template_proj: 模板图像的投影向量
        copy_proj: 摹本图像的投影向量
    Returns:
        overlap_ratio: 投影重叠率
        correlation: 投影相关系数
    Raises:
        ValueError: 如果投影向量维度不一致
    """
    # 维度检查
    if template_proj.shape != copy_proj.shape:
        raise ValueError(f"投影向量维度不一致: 模板 {template_proj.shape}, 摹本 {copy_proj.shape}")
        
    # 确保为一维数组
    if len(template_proj.shape) != 1 or len(copy_proj.shape) != 1:
        raise ValueError("投影向量必须是一维数组")
    """
    计算两个投影向量的IOU和相关性
    Args:
        template_proj: 模板投影向量
        copy_proj: 摹本投影向量
    Returns:
        (iou, correlation): IOU和Pearson相关系数
    """
    # 对齐向量长度（填充0）
    max_len = max(len(template_proj), len(copy_proj))
    template_padded = np.pad(template_proj, (0, max_len - len(template_proj)))
    copy_padded = np.pad(copy_proj, (0, max_len - len(copy_proj)))
    
    # 计算IOU (Intersection over Union)
    intersection = np.minimum(template_padded, copy_padded)
    union = np.maximum(template_padded, copy_padded)
    iou = np.sum(intersection) / np.sum(union) if np.sum(union) > 0 else 0.0
    
    # 计算Pearson相关系数
    if len(template_proj) > 1 and len(copy_proj) > 1:
        min_len = min(len(template_proj), len(copy_proj))
        corr, _ = pearsonr(template_proj[:min_len], copy_proj[:min_len])
    else:
        corr = 0.0
    
    return iou, corr

def calculate_f4_to_f11(template: np.ndarray, copy: np.ndarray) -> List[float]:
    """
    计算f4-f11：四个方向(0°, -45°, 45°, 90°)的投影IOU和相关性
    Returns:
        [f4, f5, f6, f7, f8, f9, f10, f11]
    """
    angles = [0, -45, 45, 90]
    features = []
    
    for angle in angles:
        # 计算模板和摹本的投影
        template_proj = calculate_projection(template, angle)
        copy_proj = calculate_projection(copy, angle)
        
        # 计算IOU(f4/f6/f8/f10)和相关性(f5/f7/f9/f11)
        iou, corr = calculate_projection_similarity(template_proj, copy_proj)
        features.extend([iou, corr])
    
    return features

def calculate_f12_keypoint_distance(template: np.ndarray, 
                                   copy: np.ndarray) -> float:
    """
    计算f12: 关键点平均欧氏距离特征
    
    使用Faster R-CNN检测关键点，然后用CDP算法进行匹配
    
    Args:
        template: 模板图像 (二值图像)
        copy: 摹本图像 (二值图像)
        
    Returns:
        f12: 归一化到[0,10]的关键点距离分数，距离越小分数越高
    """
    try:
        # 1. 检测关键点
        template_points = detect_keypoints(template)
        copy_points = detect_keypoints(copy)
        
        if len(template_points) == 0 or len(copy_points) == 0:
            print("Warning: No keypoints detected")
            return 5.0
        
        # 2. 使用CDP算法进行点集匹配
        cdp_config_path = "configs/point_matching_config.yaml"
        matching_result = match_keypoints(template_points, copy_points, cdp_config_path)
        
        # 获取平均配准误差
        mean_error = matching_result['mean_error']
        
        # 3. 将误差转换为0-10分数（误差越小，分数越高）
        # 假设最大可接受误差为图像对角线长度的20%
        max_error = np.sqrt(template.shape[0]**2 + template.shape[1]**2) * 0.2
        
        # 计算分数：error=0时得10分，error=max_error时得0分
        score = max(0.0, min(10.0, 10 * (1 - mean_error / max_error)))
        
        return score
        
    except Exception as e:
        print(f"Warning: Error in keypoint distance calculation: {e}")
        return 5.0  # 出错时返回中等分数

def detect_keypoints(image: np.ndarray, 
                       config_path: str = "code/configs/faster_rcnn_training_config.yaml",
                       model_weights: str = "models/faster_rcnn_model_final.pth") -> np.ndarray:
    """
    使用Faster R-CNN检测书法图像中的关键点
    
    Args:
        image: 输入图像
        config_path: Faster R-CNN配置文件路径
        model_weights: 模型权重文件路径
        
    Returns:
        keypoints: Nx2的关键点坐标数组
        
    Raises:
        FileNotFoundError: 如果配置文件或模型文件不存在
        RuntimeError: 如果模型初始化或预测失败
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    if not os.path.exists(model_weights):
        raise FileNotFoundError(f"模型文件不存在: {model_weights}")
        
    try:
        # 初始化预测器
        predictor = setup_predictor(config_path, model_weights)
    except Exception as e:
        raise RuntimeError(f"Faster R-CNN预测器初始化失败: {str(e)}")
        
    try:
        # 预测关键点
        outputs = predictor(image)
        return convert_rcnn_to_points(outputs)
    except Exception as e:
        raise RuntimeError(f"关键点检测失败: {str(e)}")
    """
    使用Faster R-CNN检测图像关键点
    
    Args:
        image: 输入图像
        config_path: Faster R-CNN配置文件路径
        model_weights: 模型权重文件路径
        
    Returns:
        keypoints: 关键点坐标数组 (N x 2)
    """
    try:
        # 初始化Faster R-CNN预测器
        predictor, _ = setup_predictor(config_path, model_weights)
        
        # 执行关键点检测
        detections = predictor(image)
        
        # 转换检测结果为标准点集格式
        keypoints = convert_rcnn_to_points(detections)
        
        # 如果Faster R-CNN成功检测到关键点，直接返回
        if len(keypoints) > 0:
            return keypoints
            
    except Exception as e:
        print(f"Warning: Error in keypoint detection: {e}")
    
    # 如果Faster R-CNN失败或未检测到关键点，使用轮廓检测作为备用方法
    # 查找轮廓

    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    keypoints = []
    
    for contour in contours:
        # 获取轮廓的近似点
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # 添加近似点作为关键点
        for point in approx:
            keypoints.append([point[0][0], point[0][1]])
    
    # 如果没有检测到关键点，使用轮廓的端点
    if len(keypoints) == 0:
        if len(contours) > 0:
            largest_contour = max(contours, key=cv2.contourArea)
            # 获取轮廓的起点和终点
            if len(largest_contour) > 0:
                keypoints.append([largest_contour[0][0][0], largest_contour[0][0][1]])
                if len(largest_contour) > 1:
                    keypoints.append([largest_contour[-1][0][0], largest_contour[-1][0][1]])
    
    return np.array(keypoints)


def extract_all_features(template: np.ndarray, copy: np.ndarray) -> np.ndarray:
    """
    提取所有12个美学特征
    
    Args:
        template: 预处理和配准后的模板图像 (500x500 二值图像)
        copy: 预处理和配准后的摹本图像 (500x500 二值图像)
        
    Returns:
        features: 12维特征向量 [f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12]
    """
    # 1. 计算f1和f2 (形状特征)
    f1 = calculate_f1_overlap_ratio(template, copy)
    f2 = calculate_f2_shape_aesthetic_score(f1)  # 使用f1计算f2
    
    # 2. 计算f3 (凸包重叠率)
    f3 = calculate_f3_convex_hull_overlap(template, copy)
    
    # 3. 计算f4-f11 (投影特征)
    proj_features = calculate_f4_to_f11(template, copy)
    f4, f5, f6, f7, f8, f9, f10, f11 = proj_features
    
    # 4. 计算f12 (关键点平均欧氏距离)
    try:
        # 检测关键点 (使用简化实现)
        template_keypoints = detect_keypoints(template)
        copy_keypoints = detect_keypoints(copy)
        
        # 使用CDP进行关键点配准
        cdp_config_path = "configs/point_matching_config.yaml"
        matching_result = match_keypoints(template_keypoints, copy_keypoints, cdp_config_path)
        registered_copy_keypoints = matching_result['matched_points']
        
        # 计算平均欧氏距离
        f12 = calculate_f12_keypoint_distance(template_keypoints, registered_copy_keypoints)
    except Exception as e:
        print(f"Warning: Error in keypoint processing: {e}")
        f12 = 0.0
    
    # 组合所有特征
    features = np.array([f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12])
    
    return features

def extract_features_for_srafe(template: np.ndarray, copy: np.ndarray) -> Dict[str, Any]:
    """
    为SRAFE方法提取特征，包括手工特征和用于后续深度特征融合的信息
    
    Args:
        template: 预处理和配准后的模板图像 (500x500 二值图像)
        copy: 预处理和配准后的摹本图像 (500x500 二值图像)
        
    Returns:
        result: 包含各种特征的字典
    """
    # 提取12个手工美学特征
    handcrafted_features = extract_all_features(template, copy)
    
    # 为后续的特征融合准备信息
    result = {
        'handcrafted_features': handcrafted_features,  # 12维手工特征
        # 以下字段将在SRN处理后填充
        'deep_features_template': None,  # 模板的深度特征
        'deep_features_copy': None,      # 摹本的深度特征
        'feature_deep': None,            # 两个深度特征的差值(10维)
        'euclidean': None                # 两个深度特征的欧氏距离(1维)
    }
    
    return result

def get_feature_names() -> List[str]:
    """
    获取12个特征的名称
    
    Returns:
        feature_names: 特征名称列表
    """
    return [
        "f1_overlap_ratio",          # 重叠率
        "f2_shape_aesthetic_score",  # 形状美学评分
        "f3_convex_hull_overlap",    # 凸包重叠率
        "f4_proj_0_iou",             # 0°方向投影重叠率
        "f5_proj_0_corr",            # 0°方向投影相关性
        "f6_proj_neg45_iou",         # -45°方向投影重叠率
        "f7_proj_neg45_corr",        # -45°方向投影相关性
        "f8_proj_45_iou",            # 45°方向投影重叠率
        "f9_proj_45_corr",           # 45°方向投影相关性
        "f10_proj_90_iou",           # 90°方向投影重叠率
        "f11_proj_90_corr",          # 90°方向投影相关性
        "f12_keypoint_distance"      # 关键点平均欧氏距离
    ]
