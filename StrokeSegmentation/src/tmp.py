import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import List, Dict, Tuple, Callable, Optional, Union
from pathlib import Path
from predictor.unet_predictor import UNetPredictor
from evaluate_.evaluate import *
from PIL import Image

def predict(img_path,threshold):
    img = Image.open(img_path).convert("RGB")
    img_array = np.array(img)
    white_pixels = np.all(img_array >= 240, axis=-1)
    mask_o = np.where(white_pixels, 0, 1).astype(np.uint8)
    mask_o = np.expand_dims(mask_o, axis=-1)
    mask_o = np.repeat(mask_o, 6, axis=-1)
    model = UNetPredictor("models/unet_model_4.pth")
    result = model.predict(img_path, threshold=threshold)
    result = np.logical_and(result, mask_o)
    return result


if __name__ == '__main__':
    ture_mask_path = Path("data/output_img/30/19")  # 真实图像的二值化掩码
    weight_path = Path("models/unet_model_4.pth")  # 更改模型
    result_path = Path("data/output_img/30/19/0.jpg")  # 需要预测图像
    evaluator = Evaluator(validity_threshold=0.6)

    # 兼容本项目unet模型代码
    # accuracy_result = evaluator.main(ture_mask_path, weight_path, result_path, threshold=0.5)

    # 其余模型调用请仿照此处，添加模型实例化代码,其中模型预测返回结果result形状为 (H, W, 1)
    model = UNetPredictor(weight_path)
    accuracy_result = evaluator.main(ture_mask_path, None, result_path, threshold=0.5, predict_fn=predict)

    print("\n========== 评估结果 ==========")
    for i, acc in enumerate(accuracy_result["class_accuracies"], 1):
        print(f"笔画类 v{i} 准确率: {acc:.2f}%")

    print(f"\n整字总准确率: {accuracy_result['total_accuracy']:.2f}%")

    print("\n笔画类权重分布:")
    for i, weight in enumerate(accuracy_result["class_weights"], 1):
        print(f"v{i}: {weight:.2%}")
