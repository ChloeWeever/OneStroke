import os

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms

from src.loss_func import WeightedBCEWithLogitsLoss
from unet_model import UNet
from custom_dataset import SegmentationDataset
from model_trainer import UNetTrainer


def main():
    # 配置参数
    config = {
        'image_size': (500, 500),
        'batch_size': 4,
        'num_epochs': 25,
        'learning_rate': 0.001,
        'num_classes': 6,
        'bilinear': True
    }

    # 设备配置
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("mps")

    # 数据转换
    data_transforms = {
        'train': transforms.Compose([
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
    }

    # 创建数据集
    train_dataset = SegmentationDataset(
        val=False,
        transform=data_transforms['train'],
        image_size=config['image_size']
    )

    val_dataset = SegmentationDataset(
        val=True,
        transform=data_transforms['val'],
        image_size=config['image_size']
    )

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4
    )

    # 初始化模型
    model = UNet(
        n_channels=3,
        n_classes=config['num_classes'],
        bilinear=config['bilinear']
    ).to(device)

    # 定义损失函数和优化器
    #criterion = nn.BCEWithLogitsLoss()
    criterion = WeightedBCEWithLogitsLoss(base_weight=1.0)
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)

    # 创建训练器
    trainer = UNetTrainer(
        model=model,
        device=device,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler
    )

    # 训练模型
    trained_model = trainer.train(train_loader, val_loader, num_epochs=config['num_epochs'])

    # 保存模型
    torch.save(trained_model.state_dict(), '../models/unet_model.pth')
    print("Model saved as 'unet_model.pth'")


if __name__ == '__main__':
    main()
