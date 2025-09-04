import torch
import torch.nn as nn
import torch.nn.functional as F
import math


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


class PatchEmbedding(nn.Module):
    """将输入图像分割为补丁并映射到嵌入向量"""
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super(PatchEmbedding, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        
        # 使用卷积层进行分块和嵌入
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        # 位置编码
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        nn.init.trunc_normal_(self.pos_embed, std=.02)
        nn.init.trunc_normal_(self.cls_token, std=.02)
        nn.init.kaiming_normal_(self.proj.weight, mode='fan_out')
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)
    
    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size and W == self.img_size, \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size}*{self.img_size})."
        
        # 分块和嵌入
        x = self.proj(x).flatten(2).transpose(1, 2)  # B, num_patches, embed_dim
        
        # 添加cls_token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # B, 1, embed_dim
        x = torch.cat((cls_tokens, x), dim=1)  # B, num_patches + 1, embed_dim
        
        # 添加位置编码
        x = x + self.pos_embed
        
        return x


class Attention(nn.Module):
    """多头自注意力机制"""
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, dropout_rate=0.0):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        
        self.scale = qk_scale or head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(dropout_rate)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        B, N, C = x.shape
        
        # 生成查询、键、值
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # B, num_heads, N, head_dim
        
        # 计算注意力权重
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # 应用注意力
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        
        # 投影和dropout
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class MLP(nn.Module):
    """多层感知器"""
    def __init__(self, in_features, hidden_features=None, out_features=None, dropout_rate=0.0):
        super(MLP, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer编码器块"""
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, qk_scale=None, 
                 dropout_rate=0.0, attn_dropout_rate=0.0, drop_path_rate=0.0):
        super(TransformerBlock, self).__init__()
        
        # 层归一化
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        
        # 多头自注意力
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            dropout_rate=attn_dropout_rate
        )
        
        # DropPath
        self.drop_path = nn.Identity()
        if drop_path_rate > 0.:
            self.drop_path = DropPath(drop_path_rate)
        
        # 层归一化
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        
        # MLP
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(
            in_features=dim, hidden_features=mlp_hidden_dim, 
            dropout_rate=dropout_rate
        )
        
    def forward(self, x):
        # 残差连接 + 注意力
        x = x + self.drop_path(self.attn(self.norm1(x)))
        
        # 残差连接 + MLP
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        return x


class DropPath(nn.Module):
    """随机丢弃路径"""
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        
    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # B, 1, 1, ...
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # 0 or 1
        output = x.div(keep_prob) * random_tensor
        
        return output


class VisionTransformer(nn.Module):
    """Vision Transformer模型"""
    def __init__(self, img_size=224, patch_size=16, in_channels=3, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True,
                 qk_scale=None, dropout_rate=0.1, attn_dropout_rate=0.1, 
                 drop_path_rate=0.1, norm_layer=nn.LayerNorm):
        super(VisionTransformer, self).__init__()
        
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        
        # 补丁嵌入
        self.patch_embed = PatchEmbedding(
            img_size=img_size, patch_size=patch_size, in_channels=in_channels,
            embed_dim=embed_dim
        )
        num_patches = self.patch_embed.num_patches
        
        # 分类头
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        
        # 随机丢弃路径率调度
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        
        # Transformer编码器
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                qk_scale=qk_scale, dropout_rate=dropout_rate, attn_dropout_rate=attn_dropout_rate,
                drop_path_rate=dpr[i]
            )
            for i in range(depth)
        ])
        
        # 层归一化
        self.norm = norm_layer(embed_dim)
        
        # 初始化权重
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)
    
    def forward(self, x):
        x = self.patch_embed(x)  # B, num_patches + 1, embed_dim
        
        for block in self.blocks:
            x = block(x)  # B, num_patches + 1, embed_dim
        
        x = self.norm(x)  # B, num_patches + 1, embed_dim
        
        # 使用cls_token进行分类
        x = self.head(x[:, 0])  # B, num_classes
        
        return x


class SkipConnection(nn.Module):
    """跳跃连接模块"""
    def __init__(self, in_channels, out_channels):
        super(SkipConnection, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class TransUNetHead(nn.Module):
    """TransUNet输出头，将特征映射转换为分割掩码"""
    def __init__(self, in_channels, out_channels):
        super(TransUNetHead, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
    def forward(self, x):
        return self.conv(x)