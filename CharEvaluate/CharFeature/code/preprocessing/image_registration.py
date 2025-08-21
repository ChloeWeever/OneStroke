"""
image_registration.py - 图像配准模块

包含尺寸配准和位置配准功能
"""

import cv2
import numpy as np
from typing import Tuple

def resize_with_aspect_ratio(image: np.ndarray, target_width: int) -> np.ndarray:
    """
    保持宽高比调整图像尺寸
    
    Args:
        image: 输入图像
        target_width: 目标宽度
        
    Returns:
        resized_image: 调整尺寸后的图像
    """
    h, w = image.shape[:2]
    ratio = target_width / w
    target_height = int(h * ratio)
    resized_image = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)
    return resized_image

def get_character_roi(image: np.ndarray) -> Tuple[int, int, int, int]:
    """
    获取字符的最小外接矩形ROI
    
    Args:
        image: 二值图像
        
    Returns:
        (x, y, w, h): ROI的坐标和尺寸
    """
    # 查找轮廓
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return 0, 0, image.shape[1], image.shape[0]
    
    # 获取最大轮廓的边界框
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    return x, y, w, h

def size_registration(template: np.ndarray, copy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    尺寸配准 - 使模板和摹本的字符具有相同像素数
    
    Args:
        template: 模板图像 (二值图像)
        copy: 摹本图像 (二值图像)
        
    Returns:
        (registered_template, registered_copy): 配准后的图像对
        
    Raises:
        TypeError: 如果输入类型错误
        ValueError: 如果输入无效或字符区域为空
    """
    # 输入验证
    if not isinstance(template, np.ndarray) or not isinstance(copy, np.ndarray):
        raise TypeError("输入必须是numpy数组")
        
    if template.size == 0 or copy.size == 0:
        raise ValueError("输入图像为空")
        
    # 获取字符ROI
    tx, ty, tw, th = get_character_roi(template)
    cx, cy, cw, ch = get_character_roi(copy)
    
    # 计算像素数
    template_roi = template[ty:ty+th, tx:tx+tw]
    copy_roi = copy[cy:cy+ch, cx:cx+cw]
    
    ST = np.count_nonzero(template_roi)
    SC = np.count_nonzero(copy_roi)
    
    if ST == 0 or SC == 0:
        return template, copy
    
    # 计算缩放因子
    scale_factor = np.sqrt(ST / SC)
    
    # 调整摹本尺寸
    new_cw = max(1, int(cw * scale_factor))
    new_ch = max(1, int(ch * scale_factor))
    
    # 保持高宽比例调整ROI尺寸
    try:
        resized_copy = cv2.resize(copy_roi, (new_cw, new_ch), 
                                interpolation=cv2.INTER_AREA if scale_factor < 1 else cv2.INTER_LINEAR)
    except Exception as e:
        raise RuntimeError(f"尺寸调整失败: {str(e)}")
    
    # 创建输出图像
    registered_template = np.zeros((500, 500), dtype=template.dtype)
    registered_copy = np.zeros((500, 500), dtype=copy.dtype)
    
    try:
        # 将ROI放置在图像中心
        center_y, center_x = 250, 250
        
        # 放置模板
        ty_offset = max(0, min(500 - th, center_y - th // 2))
        tx_offset = max(0, min(500 - tw, center_x - tw // 2))
        registered_template[ty_offset:ty_offset+th, tx_offset:tx_offset+tw] = template_roi
        
        # 放置摹本
        cy_offset = max(0, min(500 - new_ch, center_y - new_ch // 2))
        cx_offset = max(0, min(500 - new_cw, center_x - new_cw // 2))
        registered_copy[cy_offset:cy_offset+new_ch, cx_offset:cx_offset+new_cw] = resized_copy
    except Exception as e:
        raise RuntimeError(f"ROI放置失败: {str(e)}")
    
    return registered_template, registered_copy

def calculate_center_of_gravity(image: np.ndarray) -> Tuple[float, float]:
    """
    计算图像的重心
    
    Args:
        image: 二值图像
        
    Returns:
        (x, y): 重心坐标
    """
    # 获取前景像素坐标
    coords = np.column_stack(np.where(image > 0))
    
    if coords.size == 0:
        return image.shape[1] / 2, image.shape[0] / 2
    
    # 计算重心
    center_y = np.mean(coords[:, 0])
    center_x = np.mean(coords[:, 1])
    
    return center_x, center_y

def location_registration(template: np.ndarray, copy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    位置配准 - 使模板和摹本的重心重合
    
    Args:
        template: 模板图像 (二值图像)
        copy: 摹本图像 (二值图像)
        
    Returns:
        (registered_template, registered_copy): 配准后的图像对
        
    Raises:
        TypeError: 如果输入类型错误
        ValueError: 如果输入无效
    """
    # 输入验证
    if not isinstance(template, np.ndarray) or not isinstance(copy, np.ndarray):
        raise TypeError("输入必须是numpy数组")
        
    if template.size == 0 or copy.size == 0:
        raise ValueError("输入图像为空")
        
    # 计算重心
    try:
        template_cx, template_cy = calculate_center_of_gravity(template)
        copy_cx, copy_cy = calculate_center_of_gravity(copy)
    except Exception as e:
        raise RuntimeError(f"重心计算失败: {str(e)}")
    
    # 计算偏移量
    center_x, center_y = 250, 250  # 目标中心点
    offset_x_template = int(center_x - template_cx)
    offset_y_template = int(center_y - template_cy)
    offset_x_copy = int(center_x - copy_cx)
    offset_y_copy = int(center_y - copy_cy)
    
    # 创建输出图像
    registered_template = np.zeros((500, 500), dtype=template.dtype)
    registered_copy = np.zeros((500, 500), dtype=copy.dtype)
    
    try:
        # 平移图像
        h, w = template.shape[:2]
        registered_template[max(0, offset_y_template):min(500, offset_y_template + h),
                          max(0, offset_x_template):min(500, offset_x_template + w)] = \
            template[max(0, -offset_y_template):min(h, 500 - offset_y_template),
                    max(0, -offset_x_template):min(w, 500 - offset_x_template)]
        
        h, w = copy.shape[:2]
        registered_copy[max(0, offset_y_copy):min(500, offset_y_copy + h),
                       max(0, offset_x_copy):min(500, offset_x_copy + w)] = \
            copy[max(0, -offset_y_copy):min(h, 500 - offset_y_copy),
                 max(0, -offset_x_copy):min(w, 500 - offset_x_copy)]
    except Exception as e:
        raise RuntimeError(f"图像平移失败: {str(e)}")
    
    return registered_template, registered_copy

def register_images(template: np.ndarray, copy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    完整的图像配准流程
    
    Args:
        template: 模板图像 (500x500 二值图像)
        copy: 摹本图像 (500x500 二值图像)
        
    Returns:
        (registered_template, registered_copy): 配准后的图像对
        
    Raises:
        TypeError: 如果输入类型错误
        ValueError: 如果输入无效或维度不正确
        RuntimeError: 如果配准过程失败
    """
    # 输入验证
    if not isinstance(template, np.ndarray) or not isinstance(copy, np.ndarray):
        raise TypeError("输入必须是numpy数组")
        
    if template.size == 0 or copy.size == 0:
        raise ValueError("输入图像为空")
        
    if template.shape != (500, 500) or copy.shape != (500, 500):
        raise ValueError("输入图像必须是500x500的二值图像")
        
    try:
        # 1. 尺寸配准
        size_registered_template, size_registered_copy = size_registration(template, copy)
        
        # 2. 位置配准
        final_template, final_copy = location_registration(size_registered_template, size_registered_copy)
        
        # 验证输出
        if final_template.shape != (500, 500) or final_copy.shape != (500, 500):
            raise RuntimeError("配准后的图像尺寸不正确")
            
        return final_template, final_copy
        
    except Exception as e:
        raise RuntimeError(f"图像配准失败: {str(e)}")


'''
# 用于测试的示例代码
if __name__ == "__main__":
    # 创建示例图像进行测试
    template = np.zeros((500, 500), dtype=np.uint8)
    copy = np.zeros((500, 500), dtype=np.uint8)
    
    # 在图像中绘制字符
    cv2.rectangle(template, (100, 100), (300, 300), 255, -1)
    cv2.rectangle(copy, (150, 150), (350, 350), 255, -1)
    
    # 配准
    registered_template, registered_copy = register_images(template, copy)
    
    print("图像配准完成")
    print(f"模板图像尺寸: {registered_template.shape}")
    print(f"摹本图像尺寸: {registered_copy.shape}")
'''