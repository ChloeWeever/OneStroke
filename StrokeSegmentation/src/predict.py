import os
import torch
from torchvision import transforms
from PIL import Image
from model.unet_model import UNet
from model.other.fcn.fcn_model import FCN
import numpy as np
from core.config import settings



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


class FCNPredictor:
    def __init__(self, model_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = FCN(n_channels=3, n_classes=6).to(self.device)
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
    print(settings)
    
    # 加载输入图像
    img = Image.open(settings.PREDICT_INPUT).convert('RGB')
    img_array = np.array(img)
    
    if settings.MODEL == 'unet':
        # 使用UNet模型进行预测（原始代码）
        print("\n使用UNet模型进行预测:")
        unet_predictor = UNetPredictor(settings.PREDICT_MODEL, settings.DEVICE)
        unet_result = unet_predictor.predict(settings.PREDICT_INPUT)
        print(f"UNet预测结果形状: {unet_result.shape}")
        # 以下为原始UNet预测结果的处理和保存（保持不变）
        if img_array.shape[:2] != (500, 500):
            img = img.resize((500, 500))
            img_array = np.array(img)

        white_pixels = np.all(img_array >= 240, axis=-1)
        mask_o = np.where(white_pixels, 0, 1).astype(np.uint8)

        # 原始图像
        img_array = np.array(img)  # (500, 500, 3)

        # 每个预测通道保存为灰度图
        result_images = []
        for i in range(6):
            mask = unet_result[:, :, i]  # (500, 500)
            mask = np.logical_and(mask, mask_o)
            mask_image = (mask * 255).astype(np.uint8)
            result_images.append(mask_image)

            # 保存
            img = Image.fromarray(mask_image, mode='L')
            img.save(f'prediction_class_{i}.png')

        # 将所有预测图堆叠成 (500, 500, 6) -> (500, 500, 1) x6 -> 拼接为 (500, 600, 3)
        # 方法一：横向拼接（6张预测图 + 原图）—— 每张预测图宽度为 500
        stacked = np.zeros((500, 500 * 7, 3), dtype=np.uint8)  # (500, 4000, 3)

        # 放入原图
        stacked[:, :500, :] = img_array

        # 放入6张预测图（每张扩展为3通道）
        for i, mask_image in enumerate(result_images):
            # 扩展为3通道
            mask_3channel = np.repeat(mask_image[:, :, np.newaxis], 3, axis=2)  # (500, 500, 3)
            stacked[:, 500 + i * 500: 500 + (i + 1) * 500, :] = mask_3channel

        # 保存拼接图
        Image.fromarray(stacked).save('output_combined.png')
        print("UNet预测结果已保存为 'output_combined.png' 和 'prediction_class_0-5.png'")
   
    elif settings.MODEL == 'fcn':
        # 使用FCN模型进行预测
        print("\n使用FCN模型进行预测:")
        try:
            fcn_predictor = FCNPredictor(settings.PREDICT_MODEL, settings.DEVICE)
            fcn_result = fcn_predictor.predict(settings.PREDICT_INPUT)
            print(f"FCN预测结果形状: {fcn_result.shape}")
            
            # 保存FCN预测结果
            # 调整图像尺寸以匹配预测结果
            if img_array.shape[:2] != (500, 500):
                img = img.resize((500, 500))
                img_array = np.array(img)
            
            # 创建白色像素掩码
            white_pixels = np.all(img_array >= 240, axis=-1)
            mask_o = np.where(white_pixels, 0, 1).astype(np.uint8)
            
            # 保存FCN预测结果
            fcn_result_images = []
            for i in range(6):
                mask = fcn_result[:, :, i]  # (500, 500)
                mask = np.logical_and(mask, mask_o)
                mask_image = (mask * 255).astype(np.uint8)
                fcn_result_images.append(mask_image)
                
                # 保存FCN预测结果
                img = Image.fromarray(mask_image, mode='L')
                img.save(f'fcn_prediction_class_{i}.png')
                
            print("FCN预测结果已保存为 'fcn_prediction_class_0-5.png'")
            
        except Exception as e:
            print(f"FCN模型预测过程中出现错误: {e}")
            print("请确保FCN模型文件 'models/fcn_model_new.pth' 存在")
    
    