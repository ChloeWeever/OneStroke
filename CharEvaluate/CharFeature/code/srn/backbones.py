"""
backbones.py - 预训练模型（ResNet101等）
"""

import torch.nn as nn
import torchvision.models as models

def get_backbone(name: str, pretrained: bool = True) -> nn.Module:
    """
    获取预训练骨干网络
    
    Args:
        name: 骨干网络名称
        pretrained: 是否使用预训练权重
        
    Returns:
        backbone: 骨干网络模型
    """
    if name == 'resnet101':
        backbone = models.resnet101(pretrained=pretrained)
    elif name == 'resnet50':
        backbone = models.resnet50(pretrained=pretrained)
    elif name == 'resnet34':
        backbone = models.resnet34(pretrained=pretrained)
    elif name == 'resnet18':
        backbone = models.resnet18(pretrained=pretrained)
    else:
        raise ValueError(f"Unsupported backbone: {name}")
    
    return backbone