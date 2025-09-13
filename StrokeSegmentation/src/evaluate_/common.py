import torch.nn.functional as F
import torch
from core.config import settings

def BCE_Dice(model):
    pass

def get_bce(predictions, targets):
    """
    predictions: [6, H, W] (logits)
    targets: [6, H, W] (二值标签)
    """
    bce = F.binary_cross_entropy_with_logits(predictions, targets, reduction='mean')
    return bce

def get_dice(predictions, targets):
    """
    predictions: [6, H, W] (logits)
    targets: [6, H, W] (二值标签)
    """
def get_dice(predictions, targets):
    """
    predictions: [6, H, W] (logits)
    targets: [6, H, W] (二值标签)
    
    返回：所有类别的平均Dice系数
    """
    # 将logits转换为概率
    preds = torch.sigmoid(predictions)
    
    # 计算每个类别的Dice系数
    smooth = 1e-6  # 防止除零错误
    dice_scores = []
    
    # 遍历每个类别
    for i in range(predictions.size(0)):
        # 计算交集
        intersection = (preds[i] * targets[i]).sum()
        
        # 计算并集（用于Dice系数计算）
        union = preds[i].sum() + targets[i].sum()
        
        # 计算Dice系数
        dice = (2. * intersection + smooth) / (union + smooth)
        dice_scores.append(dice)
    
    # 返回所有类别的平均Dice系数
    return torch.mean(torch.tensor(dice_scores))