import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ConvBlock(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, use_se=True):
        super(ConvBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.use_se = use_se
        if use_se:
            self.se = SEBlock(out_channels)

    def forward(self, x):
        x = self.conv_block(x)
        if self.use_se:
            x = self.se(x)
        return x


class DownSample(nn.Module):
    """Downscaling with maxpool then convolution block"""

    def __init__(self, in_channels, out_channels, use_se=True):
        super(DownSample, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_channels, out_channels, use_se)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class UpSample(nn.Module):
    """Upscaling then convolution block"""

    def __init__(self, in_channels, out_channels, use_deconv=True):
        super(UpSample, self).__init__()
        
        if use_deconv:
            # 使用转置卷积进行上采样
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        else:
            # 使用双线性插值进行上采样
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(in_channels, out_channels, kernel_size=1)
            )

    def forward(self, x):
        return self.up(x)


class FCNHead(nn.Module):
    """FCN输出头，将特征映射转换为分割掩码"""
    
    def __init__(self, in_channels, out_channels):
        super(FCNHead, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
    def forward(self, x):
        return self.conv(x)