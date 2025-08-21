"""
image_preprocessing.py - 图像预处理模块

包含灰度变换、二值化、去噪等预处理功能
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Union

def grayscale_transform(image: np.ndarray) -> np.ndarray:
    """
    灰度变换
    
    Args:
        image: 输入图像 (可以是彩色或灰度)
        
    Returns:
        gray_image: 灰度图像
    """
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image.copy()
    
    return gray_image

def binarize_image(image: np.ndarray, threshold: int = 127) -> np.ndarray:
    """
    图像二值化
    
    Args:
        image: 灰度图像
        threshold: 二值化阈值
        
    Returns:
        binary_image: 二值图像 (0为背景, 255为前景)
    """
    _, binary_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    return binary_image

def remove_noise(image: np.ndarray, min_area: int = 50) -> np.ndarray:
    """
    去噪 - 移除小的连通域
    
    Args:
        image: 二值图像
        min_area: 最小连通域面积阈值
        
    Returns:
        cleaned_image: 去噪后的图像
    """
    # 确保图像是二值图像
    _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    
    # 查找连通域
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 创建输出图像
    cleaned_image = np.zeros_like(image)
    
    # 保留面积大于阈值的连通域
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= min_area:
            cv2.drawContours(cleaned_image, [contour], -1, 255, -1)
    
    return cleaned_image

def preprocess_image(image: Union[np.ndarray, str], 
                    threshold: int = 127, 
                    min_area: int = 50) -> np.ndarray:
    """
    完整的图像预处理流程
    
    Args:
        image: 输入图像或图像路径
        threshold: 二值化阈值，范围[0, 255]
        min_area: 去噪的最小连通域面积，应为正整数
        
    Returns:
        processed_image: 预处理后的图像 (500x500 二值图像，0为背景，255为前景)
        
    Raises:
        TypeError: 如果输入类型错误
        ValueError: 如果参数值不合法或图像无效
        RuntimeError: 如果预处理过程失败
    """
    processed_image = None
    try:
        # 参数验证
        if isinstance(image, str):
            image = cv2.imread(image)
            if image is None:
                raise ValueError("无法读取图像文件")
        
        if not isinstance(image, np.ndarray):
            raise TypeError("image必须是numpy数组或有效的图像路径")
            
        if image.size == 0:
            raise ValueError("输入图像为空")
            
        if not 0 <= threshold <= 255:
            raise ValueError("threshold必须在0到255之间")
            
        if not isinstance(min_area, int) or min_area <= 0:
            raise ValueError("min_area必须是正整数")
            
        # 1. 灰度变换
        try:
            gray = grayscale_transform(image)
        except Exception as e:
            raise RuntimeError(f"灰度变换失败: {str(e)}")
        
        # 2. 二值化
        try:
            binary = binarize_image(gray, threshold)
        except Exception as e:
            raise RuntimeError(f"二值化失败: {str(e)}")
        
        # 3. 去噪
        try:
            cleaned = remove_noise(binary, min_area)
        except Exception as e:
            raise RuntimeError(f"去噪失败: {str(e)}")
        
        # 4. 调整尺寸为500x500
        try:
            processed_image = cv2.resize(cleaned, (500, 500), interpolation=cv2.INTER_AREA)
            
            # 验证输出
            if processed_image.shape != (500, 500):
                raise RuntimeError("调整后的图像尺寸不正确")
                
            # 确保输出是二值图像
            unique_values = np.unique(processed_image)
            if not np.array_equal(unique_values, np.array([0])) and \
               not np.array_equal(unique_values, np.array([255])) and \
               not np.array_equal(unique_values, np.array([0, 255])):
                raise RuntimeError("输出不是有效的二值图像")
                
        except Exception as e:
            raise RuntimeError(f"尺寸调整失败: {str(e)}")
            
        return processed_image
        
    except Exception as e:
        if isinstance(e, (TypeError, ValueError, RuntimeError)):
            raise
        raise RuntimeError(f"图像预处理失败: {str(e)}")
        
    finally:
        if processed_image is None:
            # 如果处理失败，返回空白图像
            processed_image = np.zeros((500, 500), dtype=np.uint8)


'''
# 用于测试的示例代码
if __name__ == "__main__":
    # 创建一个示例图像进行测试
    test_image = np.zeros((400, 400), dtype=np.uint8)
    cv2.rectangle(test_image, (100, 100), (300, 300), 255, -1)
    
    # 添加一些噪声
    cv2.circle(test_image, (50, 50), 5, 255, -1)
    
    # 预处理
    processed = preprocess_image(test_image)
    
    print(f"原始图像尺寸: {test_image.shape}")
    print(f"处理后图像尺寸: {processed.shape}")
    print("预处理完成")
'''
