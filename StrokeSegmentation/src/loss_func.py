import torch.nn as nn
import torch.nn.functional as F


class WeightedBCEWithLogitsLoss(nn.Module):
    def __init__(self, base_weight=1.0):
        super().__init__()
        self.base_weight = base_weight  # 基础权重（标签全0时的权重）

    def forward(self, predictions, targets):
        """
        predictions: [B, 11, H, W] (未归一化的原始输出)
        targets: [B, 11, H, W] (二值标签)
        """
        # 计算每个像素的权重：[B, H, W]
        num_ones = targets.sum(dim=1)  # 统计每个像素的1的数量
        weights = (num_ones + self.base_weight).unsqueeze(1)  # 增加通道维 [B,1,H,W]

        # 计算基础BCE损失（不降维）
        loss = F.binary_cross_entropy_with_logits(
            predictions, targets, reduction='none'
        )  # 输出形状 [B,11,H,W]

        # 应用权重
        weighted_loss = loss * weights
        return weighted_loss.mean()  # 全局平均