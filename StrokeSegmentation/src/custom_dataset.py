import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class SegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None, image_size=(500, 500)):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.image_size = image_size
        self.images = [f for f in os.listdir(images_dir) if f.endswith('.png') or f.endswith('.jpg')]

        # 基本转换
        self.base_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.images_dir, img_name)
        mask_path = os.path.join(self.masks_dir, img_name.replace('.jpg', '.npy').replace('.png', '.npy'))

        # 加载图像
        image = Image.open(img_path).convert('RGB')
        image = self.base_transform(image)

        # 加载mask (500x500x11的numpy数组)
        mask = np.load(mask_path)
        assert mask.shape == (500, 500, 6), f"Mask形状应为(500,500,6)，但得到{mask.shape}"
        mask = torch.from_numpy(mask).permute(2, 0, 1).float()  # 转为CHW格式

        if self.transform:
            image = self.transform(image)

        return image, mask


