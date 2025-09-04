import torch
import torch.nn as nn
import torch.nn.functional as F
from .model_components import ConvBlock, DownSample, UpSample, FCNHead


class FCN(nn.Module):
    """全卷积网络(FCN)模型，用于图像分割任务"""
    
    def __init__(self, n_channels, n_classes=6, use_se=True, use_deconv=True):
        """
        初始化FCN模型
        
        参数:
            n_channels: 输入图像的通道数
            n_classes: 输出分割类别数
            use_se: 是否使用SE注意力模块
            use_deconv: 是否使用转置卷积进行上采样
        """
        super(FCN, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.use_se = use_se
        
        # 编码器（下采样）部分
        self.encoder = nn.ModuleList([
            # 输入层
            ConvBlock(n_channels, 64, use_se),
            # 下采样层
            DownSample(64, 128, use_se),
            DownSample(128, 256, use_se),
            DownSample(256, 512, use_se),
            DownSample(512, 1024, use_se)
        ])
        
        # 解码器（上采样）部分
        self.decoder = nn.ModuleList([
            # 上采样层
            UpSample(1024, 512, use_deconv),
            UpSample(512, 256, use_deconv),
            UpSample(256, 128, use_deconv),
            UpSample(128, 64, use_deconv)
        ])
        
        # 分类器头部
        self.head = FCNHead(64, n_classes)
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """前向传播过程"""
        # 保存编码器各层的输出，用于跳跃连接
        skip_connections = []
        
        # 编码器前向传播
        for i, layer in enumerate(self.encoder):
            x = layer(x)
            # 除了最后一层，其他层的输出都保存用于跳跃连接
            if i < len(self.encoder) - 1:
                skip_connections.append(x)
        
        # 解码器前向传播，结合跳跃连接
        for i, layer in enumerate(self.decoder):
            x = layer(x)
            # 处理尺寸不匹配
            if i < len(self.decoder):
                skip_x = skip_connections[-(i+1)]
                diffY = skip_x.size()[2] - x.size()[2]
                diffX = skip_x.size()[3] - x.size()[3]
                
                x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                              diffY // 2, diffY - diffY // 2])
                
                # 跳跃连接：元素级相加
                x = x + skip_x
        
        # 应用分类器头
        logits = self.head(x)
        
        return logits