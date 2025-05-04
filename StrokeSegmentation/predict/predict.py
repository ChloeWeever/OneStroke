import os

import torch
from torchvision import transforms
from PIL import Image
from src.unet_model import UNet
import numpy as np


class UNetPredictor:
    def __init__(self, model_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = UNet(n_channels=3, n_classes=6).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def predict(self, image_path, threshold=0.5):
        # 加载并预处理图像
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # 预测
        with torch.no_grad():
            output = self.model(image_tensor)
            probabilities = torch.sigmoid(output)
            binary_output = (probabilities > threshold).float()

        result = binary_output.squeeze(0).permute(1, 2, 0).cpu().numpy()
        return result


if __name__ == '__main__':
    print(f"Working directory: {os.getcwd()}")
    predictor = UNetPredictor('../models/unet_model_1.pth')
    result = predictor.predict('test.jpg')
    print(f"Prediction shape: {result.shape}")
    # 假设 result 是模型返回的 (500, 500, 6) 的 numpy 数组
    print(len(result))
    for i in range(6):
        mask = result[:, :, i]  # 取第i个类别 (500, 500)
        # 将 0/1 转换为 0~255 的像素值（0 -> 黑色，1 -> 白色）
        mask_image = (mask * 255).astype(np.uint8)
        # 转换为图像并保存
        img = Image.fromarray(mask_image, mode='L')  # 'L' 表示灰度图
        img.save(f'prediction_class_{i}.png')
        if (i == 5):
            for x in range(500):
                for y in range(500):
                    if mask[x, y] == 1:
                        print(x, y)