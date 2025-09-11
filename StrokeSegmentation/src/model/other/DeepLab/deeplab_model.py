# deeplab_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from .model_components import ASPP, DeepLabDecoder


class DeepLabV3Plus(nn.Module):
    """DeepLabV3+主类"""
    def __init__(self, n_channels=3, n_classes=6, pretrained_backbone=True, use_se=False):
        super(DeepLabV3Plus, self).__init__()
        # ---- Backbone: ResNet50 ----
        resnet = models.resnet50(pretrained=pretrained_backbone, replace_stride_with_dilation=[False, True, True])
        self.conv1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1  # low-level特征
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4  # high-level特征

        # ---- ASPP ----
        self.aspp = ASPP(in_channels=2048, out_channels=256)

        # ---- Decoder + 分类头 ----
        self.decoder = DeepLabDecoder(low_level_inplanes=256, low_level_channels=48,
                                      aspp_out_channels=256, n_classes=n_classes, use_se=use_se)

        self._initialize_weights()

    def _initialize_weights(self):
        """初始化新增层权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if getattr(m, 'bias', None) is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """前向传播"""
        input_size = x.shape[2:]

        # ---- Backbone ----
        x = self.conv1(x)      # /2
        x = self.maxpool(x)    # /4
        low_level_feat = self.layer1(x)
        x = self.layer2(low_level_feat)
        x = self.layer3(x)
        x = self.layer4(x)

        # ---- ASPP ----
        x = self.aspp(x)

        # ---- Decoder + 分类头 ----
        x = self.decoder(x, low_level_feat)

        # ---- 上采样到输入尺寸 ----
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=False)
        return x
