# model_components.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    """通道注意力模块，可选在 Decoder 或 ASPP 后加"""
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


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling"""
    def __init__(self, in_channels, out_channels=256, rates=(12, 24, 36)):
        super(ASPP, self).__init__()
        self.branches = nn.ModuleList()
        # 1x1卷积
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ))
        # 空洞卷积
        for r in rates:
            self.branches.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=r, dilation=r, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))
        # 图像池化分支
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        # 最终卷积融合
        self.conv = nn.Sequential(
            nn.Conv2d((len(self.branches)+1)*out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        res = [branch(x) for branch in self.branches]
        img_pool = self.image_pool(x)
        img_pool = F.interpolate(img_pool, size=x.shape[2:], mode='bilinear', align_corners=False)
        res.append(img_pool)
        x = torch.cat(res, dim=1)
        return self.conv(x)


class DeepLabDecoder(nn.Module):
    def __init__(self, low_level_inplanes, low_level_channels=48, aspp_out_channels=256, n_classes=6, use_se=False):
        super(DeepLabDecoder, self).__init__()
        self.use_se = use_se
        self.reduce_low = nn.Sequential(
            nn.Conv2d(low_level_inplanes, low_level_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(low_level_channels),
            nn.ReLU(inplace=True)
        )
        # 先输出 256 通道特征，不直接变成 n_classes
        self.conv_feat = nn.Sequential(
            nn.Conv2d(low_level_channels + aspp_out_channels, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
        if self.use_se:
            self.se = SEBlock(256)   # 256通道
        self.classifier = nn.Conv2d(256, n_classes, kernel_size=1)  # 分类头

    def forward(self, aspp_feat, low_level_feat):
        low = self.reduce_low(low_level_feat)
        aspp_up = F.interpolate(aspp_feat, size=low.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([aspp_up, low], dim=1)
        x = self.conv_feat(x)
        if self.use_se:
            x = self.se(x)  # 在256通道特征上做注意力
        x = self.classifier(x)  # 最后映射到 n_classes
        return x



class DeepLabHead(nn.Module):
    """可单独调用的输出头，将特征映射转换为分割掩码"""
    def __init__(self, in_channels, n_classes=6):
        super(DeepLabHead, self).__init__()
        self.conv = nn.Conv2d(in_channels, n_classes, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
