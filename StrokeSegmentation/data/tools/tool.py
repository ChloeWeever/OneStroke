import os
import cv2
import numpy as np
from PIL import Image
from data.asset.stroke_vector_mapping import STROKE_VECTOR_MAP
import matplotlib.pyplot as plt  # 新增导入


def get_directory_count(path):
    count = 0
    for item in os.listdir(path):
        if os.path.isdir(os.path.join(path, item)):
            count += 1
    return count


def create_and_visualize_mask(image_path, white_threshold=240):
    """
    从图像创建1维二值mask并可视化/保存

    参数:
        image_path (str): 输入图像路径
        white_threshold (int): 白色判定阈值(0-255)
        output_npy_path (str): 保存.npy文件的路径
        output_img_path (str): 保存可视化图片的路径
    """
    # 1. 创建二值mask
    img = Image.open(image_path).convert('RGB')
    img_array = np.array(img)

    if img_array.shape[:2] != (500, 500):
        img = img.resize((500, 500))
        img_array = np.array(img)

    white_pixels = np.all(img_array >= white_threshold, axis=-1)
    mask = np.where(white_pixels, 0, 1).astype(np.uint8)

    return mask


def extract_green_mask_hsv(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (500, 500))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 定义绿色在HSV空间中的上下限
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([80, 255, 255])

    mask = cv2.inRange(hsv, lower_green, upper_green)
    binary_mask = np.where(mask > 0, 1, 0).astype(np.uint8)

    return binary_mask


if __name__ == '__main__':
    for i in range(0, 20):
        for j in range(0, 19):
            mask_key = np.zeros((500, 500))
            masks = [np.zeros((500, 500), dtype=bool) for _ in range(5)]
            for k in range(1, STROKE_VECTOR_MAP[i][0] + 1):
                img_path = f'../output_img/{i}/{j}/{k}.jpg'
                mask_key = np.logical_or(mask_key, extract_green_mask_hsv(img_path))
                masks[STROKE_VECTOR_MAP[i][k] - 1] = np.logical_or(masks[STROKE_VECTOR_MAP[i][k] - 1],
                                                                   create_and_visualize_mask(img_path))
            save_path = f'../output_img/{i}/{j}/mask_key_point.npy'
            np.save(save_path, mask_key)

            masks.append(mask_key)
            mask_500x500x6 = np.stack(masks, axis=-1)
            save_path = f'../output_img/{i}/{j}/0.npy'
            np.save(save_path, mask_500x500x6)
            masks.pop()

            for idx, mask in enumerate(masks, start=1):
                save_path = f'../output_img/{i}/{j}/mask_{idx}.npy'
                np.save(save_path, mask)

            mask_key_color = np.zeros((500, 500, 3), dtype=np.uint8)
            mask_key_color[mask_key == 1] = [0, 255, 0]

            base_img_path = f'../output_img/{i}/{j}/0.jpg'
            base_img = Image.open(base_img_path).convert('RGB').resize((500, 500))
            base_img_array = np.array(base_img)

            overlay_img = cv2.addWeighted(base_img_array, 1.0, mask_key_color, 0.6, 0)

            save_path = f'../output_img/{i}/{j}/key_point.png'
            Image.fromarray(overlay_img).save(save_path)

            fig, axes = plt.subplots(1, len(masks), figsize=(20, 5))
            for idx, mask in enumerate(masks):
                axes[idx].imshow(mask, cmap='gray')
                axes[idx].axis('off')

            save_path = f'../output_img/{i}/{j}/mask_img.jpg'
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
            plt.close(fig)

