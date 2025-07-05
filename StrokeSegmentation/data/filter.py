import os
import shutil

import cv2
import numpy as np
from tools.tool import get_directory_count


def filter_colors_and_mark_red_centers(image_path, output_path=None):
    """
    将图片中黑色和红色之外的颜色都设置成白色，找出红色区域中心点并标记为绿色

    参数:
        image_path (str): 输入图片路径
        output_path (str, 可选): 输出图片路径。如果为None，则不保存图片

    返回:
        tuple: (处理后的图像数组, 红色区域中心点列表[(x1,y1), (x2,y2), ...])
    """
    # 读取图片
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("无法读取图片" + image_path + "，请检查路径是否正确")

    # 将BGR转换为RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 图像去噪
    img_rgb = cv2.medianBlur(img_rgb, 5)

    # 定义颜色范围
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([100, 100, 100])
    lower_red = np.array([150, 0, 0])
    upper_red = np.array([255, 150, 150])

    # 创建颜色掩码
    black_mask = cv2.inRange(img_rgb, lower_black, upper_black)
    red_mask = cv2.inRange(img_rgb, lower_red, upper_red)
    combined_mask = cv2.bitwise_or(black_mask, red_mask)

    # 形态学操作
    kernel = np.ones((3, 3), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)

    # 创建处理后的图像
    white_img = np.ones_like(img_rgb) * 255
    result = np.where(combined_mask[:, :, np.newaxis].astype(bool), img_rgb, white_img)
    result[red_mask > 0] = [255, 0, 0]  # 纯红色
    result[black_mask > 0] = [0, 0, 0]  # 纯黑色

    # 图像裁剪和缩放
    height, width, _ = result.shape
    crop_size = min(height, width, 580)
    x = (width - crop_size) // 2
    y = (height - crop_size) // 2
    result = result[y:y + crop_size, x:x + crop_size]
    result = cv2.resize(result, (500, 500))

    # 检测红色区域中心点
    red_mask_processed = cv2.inRange(result, np.array([250, 0, 0]), np.array([255, 5, 5]))
    contours, _ = cv2.findContours(red_mask_processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    red_centers = []
    for cnt in contours:
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            red_centers.append((cX, cY))
            # 将中心点标记为绿色
            cv2.circle(result, (cX, cY), 2, (0, 255, 0), -1)  # 绿色点

    # 保存结果
    if output_path is not None:
        result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, result_bgr)

    return result, red_centers


def process_image_to_binary(image_path, output_path=None, white_threshold=200):
    """
    将图像中除白色以外的颜色都设成黑色，然后进行二值化处理

    参数:
        image_path (str): 输入图片路径
        output_path (str, 可选): 输出图片路径。如果为None，则不保存图片
        white_threshold (int): 白色阈值，RGB值都大于此值被认为是白色(0-255)

    返回:
        numpy.ndarray: 二值化后的图像数组(0-255)
    """
    # 读取图片
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("无法读取图片，请检查路径是否正确")

    # 将BGR转换为RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 创建白色掩码（RGB三个通道都大于阈值）
    white_mask = np.all(img_rgb > white_threshold, axis=-1)

    # 创建全黑的图像
    black_img = np.zeros_like(img_rgb)

    # 将白色区域保留，其他设为黑色
    processed_img = np.where(white_mask[..., np.newaxis], img_rgb, black_img)

    # 转换为灰度图
    gray_img = cv2.cvtColor(processed_img, cv2.COLOR_RGB2GRAY)

    # 二值化处理（白色区域为255，其他为0）
    _, binary_img = cv2.threshold(gray_img, white_threshold, 255, cv2.THRESH_BINARY)

    # 如果需要保存结果
    if output_path is not None:
        cv2.imwrite(output_path, binary_img)

    return binary_img


def process_all_image(prefix, start_index,
                      stroke_num=[5, 4, 4, 7, 3, 6, 5, 7, 9, 7, 6, 8, 8, 15, 9, 7, 8, 6, 13, 6, ]):
    now = start_index
    for i in range(0, len(stroke_num)):
        if now < 1000:
            filler = ''
            for j in range(0, 4 - len(str(now))):
                filler += '0'
            filter_colors_and_mark_red_centers(f"data/input_img/{prefix}{filler}{now}.JPG", "tmp.jpg")
        else:
            filter_colors_and_mark_red_centers(f"data/input_img/{prefix}{now}.JPG", "tmp.jpg")
        directory = os.path.dirname(f"data/output_img/{i+20}/")
        if not os.path.exists(directory):
            os.makedirs(directory)
        n = get_directory_count(f"data/output_img/{i+20}/")
        directory = os.path.dirname(f"data/output_img/{i+20}/{n}/")
        os.makedirs(directory)
        process_image_to_binary("tmp.jpg", f"data/output_img/{i+20}/{n}/0.jpg")
        now += 1
        if now == 4000:
            now += 1
        for k in range(0, stroke_num[i]):
            if now % 1000 == 0:
                now += 1
            if now == 6647:
                now += 1
            if now < 1000:
                filler = ''
                for j in range(0, 4 - len(str(now))):
                    filler += '0'
                filter_colors_and_mark_red_centers(f"data/input_img/{prefix}{filler}{now}.JPG",
                                                   f"data/output_img/{i+20}/{n}/{k + 1}.jpg")
            else:
                filter_colors_and_mark_red_centers(f"data/input_img/{prefix}{now}.JPG",
                                                   f"data/output_img/{i+20}/{n}/{k + 1}.jpg")
            now += 1
            if now == 4000:
                now += 1


# 使用示例
if __name__ == "__main__":
    name_strokes = []
    strokes = [5, 4, 4, 7, 3, 6, 5, 7, 9, 7, 6, 8, 8, 15, 9, 7, 8, 6, 13, 6] + name_strokes    # 163
    process_all_image("IMG_", 692, strokes)
    # i = 4524
    # while (i >= 4451):
    #     dir = f"data/input_img/IMG_{i}.JPG"
    #     if os.path.exists(dir):
    #         os.rename(dir, f"data/input_img/IMG_{i + 1}.JPG")
    #     i -= 1
    #for i in range(0, 22):
    #    if os.path.exists(f"data/output_img/{i}/18"):
    #        shutil.rmtree(f"data/output_img/{i}/18")
