import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class SegmentationDataset(Dataset):
    def __init__(self, val=False, transform=None, image_size=(500, 500)):
        self.transform = transform
        self.image_size = image_size
        self.images = []
        for i in range(0, 40):
            if not val:
                for j in range(0, 15):
                    self.images.append(f"data/output_img/{i}/{j}/0.jpg")
            else:
                for j in range(15, 19):
                    self.images.append(f"data/output_img/{i}/{j}/0.jpg")


        # 基本转换
        self.base_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = '0.jpg'
        img_path = self.images[idx]
        mask_path = img_path.replace('.jpg', '.npy')

        # 加载图像
        image = Image.open(img_path).convert('RGB')
        image = self.base_transform(image)

        # 加载mask (500x500x6的numpy数组)
        mask = np.load(mask_path)
        assert mask.shape == (500, 500, 6), f"Mask形状应为(500,500,6)，但得到{mask.shape}"
        mask = torch.from_numpy(mask).permute(2, 0, 1).float()  # 转为CHW格式

        if self.transform:
            image = self.transform(image)

        return image, mask
