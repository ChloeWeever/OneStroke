import torch
from torch.utils.tensorboard import SummaryWriter
from torchviz import make_dot

# 假设你的模型是 unet_model.UNet
from src.unet_model import UNet

# 初始化模型
model = UNet(n_channels=3, n_classes=6)

# 创建虚拟输入
x = torch.randn(1, 3, 500, 500)  # batch_size=1, channel=3, size=500x500

# 前向传播
y = model(x)

# 生成可视化图
dot = make_dot(y, params=dict(model.named_parameters()))

# 保存为 PDF 或 PNG 文件
dot.render("unet_model", format="png")  # 会生成 unet_model.png
