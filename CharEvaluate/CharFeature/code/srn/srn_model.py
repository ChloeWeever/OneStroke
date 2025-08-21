"""
srn_model.py - 孪生回归网络(SRN)实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbones import get_backbone

class SiameseRegressionNetwork(nn.Module):
    """
    孪生回归网络(SRN)实现
    
    Args:
        backbone_name: 骨干网络名称，支持['resnet101', 'resnet50', 'resnet34', 'resnet18']
        pretrained: 是否使用预训练权重
        feature_dims: 特征维度配置，格式为[骨干网络输出维度, 中间层维度, 最终特征维度]
        dropout_rate: Dropout比率，用于防止过拟合
        
    Raises:
        ValueError: 当backbone_name不支持或feature_dims配置无效时
        RuntimeError: 当模型构建失败时
    """
    def __init__(self, backbone_name='resnet101', pretrained=True, 
                 feature_dims=[2048, 512, 128], dropout_rate=0.5):
        super(SiameseRegressionNetwork, self).__init__()
        
        # 参数验证
        if not isinstance(backbone_name, str):
            raise TypeError("backbone_name必须是字符串类型")
            
        if not isinstance(feature_dims, (list, tuple)) or len(feature_dims) != 3:
            raise ValueError("feature_dims必须是包含3个元素的列表或元组")
            
        if not 0 <= dropout_rate <= 1:
            raise ValueError("dropout_rate必须在0到1之间")
            
        self.backbone_name = backbone_name
        
        try:
            # 获取骨干网络
            self.backbone = get_backbone(backbone_name, pretrained)
            
            # 验证骨干网络输出维度
            if not hasattr(self.backbone, 'fc'):
                raise ValueError(f"不支持的骨干网络类型: {backbone_name}")
                
            backbone_out_dim = self.backbone.fc.in_features
            if backbone_out_dim != feature_dims[0]:
                raise ValueError(f"骨干网络输出维度({backbone_out_dim})与配置维度({feature_dims[0]})不匹配")
                
            # 特征提取层
            self.feature_extractor = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),  # 全局平均池化
                nn.Flatten(),
                nn.Linear(feature_dims[0], feature_dims[1]),  # 第一次降维
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate),
                nn.Linear(feature_dims[1], feature_dims[2]),  # 第二次降维
                nn.ReLU(inplace=True)
            )
            
        except Exception as e:
            raise RuntimeError(f"SRN模型构建失败: {str(e)}")
        
        # 回归层
        self.regression_layer = nn.Linear(128, 10)
        
    def forward_once(self, x):
        """
        单次前向传播
        
        Args:
            x: 输入图像张量
            
        Returns:
            features: 特征向量
        """
        # 骨干网络特征提取
        x = self.backbone(x)
        
        # 特征提取
        features = self.feature_extractor(x)
        
        return features
    
    def forward(self, template, copy):
        """
        前向传播
        
        Args:
            template: 模板图像
            copy: 摹本图像
            
        Returns:
            (template_features, copy_features): 模板和摹本的特征向量
        """
        # 分别提取模板和摹本的特征
        template_features = self.forward_once(template)
        copy_features = self.forward_once(copy)
        
        # 回归层
        template_output = self.regression_layer(template_features)
        copy_output = self.regression_layer(copy_features)
        
        return template_output, copy_output

class SRNLoss(nn.Module):
    """
    SRN损失函数
    """
    def __init__(self):
        super(SRNLoss, self).__init__()
        
    def forward(self, template_output, copy_output, labels):
        """
        计算损失
        
        Args:
            template_output: 模板输出
            copy_output: 摹本输出
            labels: 真实标签
            
        Returns:
            loss: 损失值
        """
        # 计算欧氏距离
        euclidean_distance = F.pairwise_distance(template_output, copy_output, keepdim=True)
        
        # 计算损失
        # 根据论文公式(12): loss = 1/2N * Σ(||y_n - (10 - y_n)||_2)^2
        # 其中 10 - y_n 是预测的美学评分，y_n 是真实标签
        loss = torch.mean(torch.pow(euclidean_distance - (10 - labels), 2))
        
        return loss


'''
# 用于测试的示例代码
if __name__ == "__main__":
    # 创建SRN模型实例
    model = SiameseRegressionNetwork()
    
    # 创建示例输入
    template = torch.randn(1, 1, 500, 500)  # 模板图像
    copy = torch.randn(1, 1, 500, 500)     # 摹本图像
    
    # 前向传播
    template_features, copy_features = model(template, copy)
    
    print("SRN模型测试:")
    print(f"模板特征维度: {template_features.shape}")
    print(f"摹本特征维度: {copy_features.shape}")
    
    # 创建损失函数实例
    criterion = SRNLoss()
    
    # 创建示例标签
    labels = torch.tensor([[8.5]])  # 真实美学评分
    
    # 计算损失
    loss = criterion(template_features, copy_features, labels)
    
    print(f"损失值: {loss.item()}")
'''