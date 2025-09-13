import torch
import torch.nn as nn
import torch.nn.functional as F
from .model_components import (
    ConvBlock, DownSample, UpSample, 
    SkipConnection, TransUNetHead, SEBlock
)


class TransUNet(nn.Module):
    """TransUNet模型：结合CNN和Transformer的医学图像分割网络"""
    
    def __init__(self, n_channels, n_classes=6, img_size=224, 
                 embed_dim=768, num_heads=12, 
                 depth=12, mlp_ratio=4., qkv_bias=True, 
                 use_se=True, use_deconv=True, dropout_rate=0.1):
        """
        初始化TransUNet模型
        
        参数:
            n_channels: 输入图像的通道数
            n_classes: 输出分割类别数
            img_size: 输入图像大小
            embed_dim: Transformer嵌入维度
            num_heads: Transformer多头注意力头数
            depth: Transformer编码器深度
            mlp_ratio: MLP隐藏层维度比例
            qkv_bias: 是否在QKV线性层使用偏置
            use_se: 是否使用SE注意力模块
            use_deconv: 是否使用转置卷积进行上采样
            dropout_rate: Dropout率
        """
        super(TransUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.img_size = img_size
        self.embed_dim = embed_dim
        
        # 移除图像尺寸必须被32整除的限制
        
        # CNN编码器部分
        # 第一层：输入通道到64通道
        self.cnn_encoder1 = ConvBlock(n_channels, 64, use_se)
        
        # 第二层：64通道到128通道
        self.cnn_encoder2 = DownSample(64, 128, use_se)
        
        # 第三层：128通道到256通道
        self.cnn_encoder3 = DownSample(128, 256, use_se)
        
        # 第四层：256通道到512通道
        self.cnn_encoder4 = DownSample(256, 512, use_se)
        
        # 第五层：512通道到embed_dim通道（为Transformer准备）
        self.cnn_encoder5 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(512, embed_dim, kernel_size=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True)
        )
        
        # 自定义Transformer编码器（替换VisionTransformer，移除尺寸限制）
        self.transformer_blocks = nn.ModuleList()
        dpr = [x.item() for x in torch.linspace(0, dropout_rate, depth)]
        for i in range(depth):
            # 添加自定义的Transformer块
            self.transformer_blocks.append(
                self._create_transformer_block(embed_dim, num_heads, mlp_ratio, 
                                              qkv_bias, dropout_rate, dpr[i])
            )
        
        # 层归一化
        self.norm = nn.LayerNorm(embed_dim)
        
        # 特征转换层：将Transformer输出转换回CNN特征图
        self.trans_conv = nn.Conv2d(embed_dim, 512, kernel_size=1)
        
        # 跳跃连接处理层
        self.skip1 = SkipConnection(64, 64)
        self.skip2 = SkipConnection(128, 64)
        self.skip3 = SkipConnection(256, 128)
        self.skip4 = SkipConnection(512, 256)
        
        # CNN解码器部分
        self.cnn_decoder1 = UpSample(512, 256, use_deconv)
        self.cnn_decoder2 = UpSample(256, 128, use_deconv)
        self.cnn_decoder3 = UpSample(128, 64, use_deconv)
        self.cnn_decoder4 = UpSample(64, 64, use_deconv)
        
        # 输出头
        self.head = TransUNetHead(64, n_classes)
        
        # 初始化权重
        self._initialize_weights()
    
    def _create_transformer_block(self, dim, num_heads, mlp_ratio, qkv_bias, dropout_rate, drop_path_rate):
        """创建Transformer块"""
        return nn.Sequential(
            # 层归一化
            nn.LayerNorm(dim),
            # 多头自注意力
            self._create_attention(dim, num_heads, qkv_bias, dropout_rate),
            # DropPath
            nn.Identity() if drop_path_rate == 0 else self._create_drop_path(drop_path_rate),
            # 层归一化
            nn.LayerNorm(dim),
            # MLP
            self._create_mlp(dim, mlp_ratio, dropout_rate),
            # DropPath
            nn.Identity() if drop_path_rate == 0 else self._create_drop_path(drop_path_rate)
        )
    
    def _create_attention(self, dim, num_heads, qkv_bias, dropout_rate):
        """创建注意力机制"""
        return nn.Sequential(
            nn.Linear(dim, dim * 3, bias=qkv_bias),
            nn.Dropout(dropout_rate),
            # 注意：这里我们不直接实现完整的注意力机制，而是在forward中处理
            # 因为我们需要更灵活地处理不同尺寸的输入
        )
    
    def _create_mlp(self, dim, mlp_ratio, dropout_rate):
        """创建MLP"""
        hidden_dim = int(dim * mlp_ratio)
        return nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout_rate)
        )
    
    def _create_drop_path(self, drop_prob):
        """创建DropPath层"""
        class DropPath(nn.Module):
            def __init__(self, drop_prob):
                super().__init__()
                self.drop_prob = drop_prob
            def forward(self, x):
                if self.drop_prob == 0. or not self.training:
                    return x
                keep_prob = 1 - self.drop_prob
                shape = (x.shape[0],) + (1,) * (x.ndim - 1)
                random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
                random_tensor.floor_()
                return x.div(keep_prob) * random_tensor
        return DropPath(drop_prob)
    
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
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """前向传播过程"""
        # 保存原始输入大小，用于最终输出调整
        orig_size = x.size()[2:]
        
        # CNN编码器前向传播
        x1 = self.cnn_encoder1(x)  # 64通道，原始大小
        x2 = self.cnn_encoder2(x1)  # 128通道，1/2大小
        x3 = self.cnn_encoder3(x2)  # 256通道，1/4大小
        x4 = self.cnn_encoder4(x3)  # 512通道，1/8大小
        x5 = self.cnn_encoder5(x4)  # embed_dim通道，1/16大小
        
        # Transformer处理 - 自定义实现以支持任意尺寸
        batch_size, channels, height, width = x5.shape
        
        # 将特征图转换为序列格式
        x_flat = x5.flatten(2).transpose(1, 2)  # B, num_patches, embed_dim
        
        # 应用Transformer块
        for block in self.transformer_blocks:
            # 自定义处理Transformer块的逻辑
            # 层归一化
            x_norm = block[0](x_flat)
            # 多头自注意力
            qkv = block[1][0](x_norm)
            B, N, C = qkv.shape
            num_heads = self.transformer_blocks[0][1][0].out_features // (C // 3)
            head_dim = C // (3 * num_heads)
            qkv = qkv.reshape(B, N, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            
            # 计算注意力
            scale = head_dim ** -0.5
            attn = (q @ k.transpose(-2, -1)) * scale
            attn = attn.softmax(dim=-1)
            attn = block[1][1](attn)  # dropout
            
            # 应用注意力
            x_attn = (attn @ v).transpose(1, 2).reshape(B, N, C // 3)
            
            # DropPath
            x_attn = x_flat + block[2](x_attn)
            
            # 层归一化
            x_norm2 = block[3](x_attn)
            
            # MLP
            x_mlp = block[4](x_norm2)
            
            # DropPath
            x_flat = x_attn + block[5](x_mlp)
        
        # 层归一化
        x_flat = self.norm(x_flat)
        
        # 将Transformer输出转换回特征图
        x_transformer = x_flat.transpose(1, 2).reshape(batch_size, channels, height, width)
        x_transformer = self.trans_conv(x_transformer)
        
        # CNN解码器前向传播，结合跳跃连接
        # 处理尺寸不匹配
        def handle_size_mismatch(x, skip_x):
            diffY = skip_x.size()[2] - x.size()[2]
            diffX = skip_x.size()[3] - x.size()[3]
            x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                          diffY // 2, diffY - diffY // 2])
            return x
        
        # 解码器第一层
        x = self.cnn_decoder1(x_transformer)
        x = handle_size_mismatch(x, x4)
        x = x + self.skip4(x4)
        
        # 解码器第二层
        x = self.cnn_decoder2(x)
        x = handle_size_mismatch(x, x3)
        x = x + self.skip3(x3)
        
        # 解码器第三层
        x = self.cnn_decoder3(x)
        x = handle_size_mismatch(x, x2)
        x = x + self.skip2(x2)
        
        # 解码器第四层
        x = self.cnn_decoder4(x)
        x = handle_size_mismatch(x, x1)
        x = x + self.skip1(x1)
        
        # 应用输出头
        logits = self.head(x)
        
        # 确保输出尺寸与输入尺寸一致
        if logits.size()[2:] != orig_size:
            logits = F.interpolate(logits, size=orig_size, mode='bilinear', align_corners=True)
        
        return logits


class TransUNetSmall(TransUNet):
    """轻量级TransUNet模型，适用于资源受限环境"""
    
    def __init__(self, n_channels, n_classes=6, img_size=224, use_se=True, use_deconv=True):
        super(TransUNetSmall, self).__init__(
            n_channels=n_channels,
            n_classes=n_classes,
            img_size=img_size,
            embed_dim=384,  # 减小嵌入维度
            num_heads=6,    # 减少注意力头数
            depth=6,        # 减少编码器深度
            mlp_ratio=4.,
            qkv_bias=True,
            use_se=use_se,
            use_deconv=use_deconv,
            dropout_rate=0.1
        )


class TransUNetLarge(TransUNet):
    """大型TransUNet模型，适用于需要高精度的场景"""
    
    def __init__(self, n_channels, n_classes=6, img_size=224, use_se=True, use_deconv=True):
        super(TransUNetLarge, self).__init__(
            n_channels=n_channels,
            n_classes=n_classes,
            img_size=img_size,
            embed_dim=1024,  # 增加嵌入维度
            num_heads=16,    # 增加注意力头数
            depth=16,        # 增加编码器深度
            mlp_ratio=4.,
            qkv_bias=True,
            use_se=use_se,
            use_deconv=use_deconv,
            dropout_rate=0.1
        )