import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt


class WeightedBCEWithLogitsLoss(nn.Module):
    def __init__(self, base_weight=1.0, radius=5):
        super().__init__()
        self.base_weight = base_weight
        self.radius = radius

        # 预计算用于轮廓权重计算的距离核
        self._precompute_distance_kernel(radius)

    def _precompute_distance_kernel(self, r):
        """预计算轮廓权重使用的距离核"""
        kernel_size = 2 * r - 1
        center = r - 1
        y, x = torch.meshgrid(torch.arange(kernel_size), torch.arange(kernel_size))
        distances = torch.sqrt((x - center).float() ** 2 + (y - center).float() ** 2)
        distances[center, center] = 0  # 中心像素距离设为0
        self.register_buffer('distance_kernel', distances)

    def forward(self, predictions, targets):
        """
        predictions: [B, 6, H, W] (logits)
        targets: [B, 6, H, W] (二值标签)
        """
        B, C, H, W = targets.shape

        # 计算特征权重（通道求和）
        feature_weights = targets.sum(dim=1).float()  # [B, H, W]

        # 计算轮廓权重（优化后的方法）
        contour_weights = torch.stack([
            self._calculate_contour_weights(targets[b])
            for b in range(B)
        ])  # [B, H, W]

        # 合并权重
        weights = (feature_weights + contour_weights) ** 2

        # 归一化权重
        weights = 2 * (weights - weights.min()) / (weights.max() - weights.min() + 1e-10)

        weights = weights.unsqueeze(1)  # [B, 1, H, W]

        # 计算BCE损失并应用权重
        loss = F.binary_cross_entropy_with_logits(
            predictions, targets, reduction='none'
        )

        return (loss * weights.to(predictions.device)).mean()

    def _calculate_contour_weights(self, mask):
        """计算单样本的轮廓权重（6个通道）"""
        # 合并所有通道（逻辑或操作）
        combined_mask = mask.sum(dim=0) > 0  # [H, W]
        combined_mask = combined_mask.float()

        # 使用向量化操作计算边界
        kernel = self.distance_kernel
        kernel_size = kernel.shape[0]
        padding = kernel_size // 2

        # 填充mask以便边界处理
        padded_mask = F.pad(combined_mask.unsqueeze(0).unsqueeze(0),
                            (padding, padding, padding, padding),
                            mode='replicate').squeeze()

        # 初始化最小距离图
        min_distances = torch.full_like(combined_mask, float('inf'))

        # 遍历核中的每个位置（中心点除外）
        for i in range(kernel_size):
            for j in range(kernel_size):
                if i == padding and j == padding:
                    continue  # 跳过中心像素

                # 获取偏移后的mask
                shifted_mask = padded_mask[i:i + combined_mask.shape[0],
                               j:j + combined_mask.shape[1]]

                # 计算差异
                differences = (shifted_mask != combined_mask).float()

                # 计算加权距离（避免除以零）
                distances = torch.where(
                    differences > 0,
                    kernel[i, j] / (differences + 1e-10),
                    torch.tensor(float('inf'), device=mask.device)
                )

                # 更新最小距离
                min_distances = torch.minimum(min_distances, distances)

        # 计算轮廓权重（最小距离的倒数）
        contour_weights = torch.where(
            min_distances < float('inf'),
            1.0 / (min_distances + 1e-10),
            torch.tensor(0.0, device=mask.device)
        )

        return contour_weights


'''
from math import sqrt
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class WeightedBCEWithLogitsLoss(nn.Module):
    def __init__(self, base_weight=1.0):
        super().__init__()
        self.base_weight = base_weight  # 基础权重（标签全0时的权重）

    def forward(self, predictions, targets):
        # print("shape of predictions and targets:" + str(predictions.shape) + str(targets.shape))
        """
        predictions: [B, 6, 500, 500] (未归一化的原始输出)
        targets: [B, 6, 500, 500] (二值标签)
        """
        # 计算每个像素的权重：[B, 500, 500]
        num_ones = targets.sum(dim=1)  # 统计每个像素的1的数量
        w_c = np.zeros((4, 500, 500), dtype=np.float32)
        for i in range(len(predictions)):
            w_c[i, :, :] = cal_weight_contour_matrix(targets[i], 3)

        weights = (num_ones.cpu().float() + w_c) ** 2

        min_val = torch.min(weights)
        max_val = torch.max(weights)

        if max_val > min_val:
            weights = (weights - min_val) / (max_val - min_val)
        else:
            weights = np.zeros_like(weights)

        weights = weights.unsqueeze(1)

        # 计算基础BCE损失（不降维）
        loss = F.binary_cross_entropy_with_logits(
            predictions, targets, reduction='none'
        )  # 输出形状 [B,6,500,500]

        # 应用权重
        device = loss.device
        weights = weights.to(device)
        weighted_loss = loss * weights

        return weighted_loss.mean()  # 全局平均


def cal_weight_feature(predictions, targets):
    return targets.sum(dim=1)


def cal_weight_feature_matrix(mask_path):
    feature_matrix = np.zeros((500, 500), dtype=np.float32)
    mask = np.load(mask_path)
    for i in range(500):
        for j in range(500):
            feature_matrix[i, j] = 0
            for k in range(6):
                feature_matrix[i, j] += mask[:, :, k][i, j]
    return feature_matrix


def cal_weight_contour_matrix(full_mask, r):
    mask = np.zeros((500, 500), dtype=np.float32)
    for i in range(6):
        mask = np.logical_or(mask, full_mask[i].cpu().detach().numpy())
    contour_matrix = np.zeros((500, 500), dtype=np.float32)
    for i in range(500):
        for j in range(500):
            window = np.zeros((2 * r - 1, 2 * r - 1), dtype=np.float32)
            for x in range(i - r + 1, i + r):
                for y in range(j - r + 1, j + r):
                    g_x_y = 0
                    g_i_j = mask[i, j]
                    if 0 <= x < 500 and 0 <= y < 500:
                        g_x_y = mask[x, y]
                    if x == i and y == j:
                        window[x - i + r - 1, y - j + r - 1] = 0
                        continue
                    if g_x_y == g_i_j:
                        window[x - i + r - 1, y - j + r - 1] = 0
                    else:
                        diff = np.logical_xor(g_x_y, g_i_j)
                        # Add a small epsilon to prevent division by zero
                        window[x - i + r - 1, y - j + r - 1] = (1 / (diff ** 2 + 1e-10)) * sqrt(
                            (x - i) ** 2 + (y - j) ** 2)
            non_zero = window[window != 0]
            if non_zero.size == 0:
                min_value = 0
                contour_matrix[i, j] = 0
            else:
                min_value = np.min(non_zero)
                contour_matrix[i, j] = 1 / min_value
    return contour_matrix
'''

'''
def cal_weight_contour_matrix(self, image_path, r):
    img = Image.open(image_path).convert('RGB')
    img_array = np.array(img)

    if img_array.shape[:2] != (500, 500):
        img = img.resize((500, 500))
        img_array = np.array(img)

    white_pixels = np.all(img_array >= 240, axis=-1)
    mask = np.where(white_pixels, 0, 1).astype(np.float32)
    contour_matrix = np.zeros((500, 500), dtype=np.float32)
    for i in range(500):
        for j in range(500):
            window = np.zeros((2 * r - 1, 2 * r - 1), dtype=np.float32)
            for x in range(i - r + 1, i + r):
                for y in range(j - r + 1, j + r):
                    g_x_y = 0
                    g_i_j = mask[i, j]
                    if 0 <= x < 500 and 0 <= y < 500:
                        g_x_y = mask[x, y]
                    if x == i and y == j:
                        window[x - i + r - 1, y - j + r - 1] = 0
                        continue
                    if g_x_y == g_i_j:
                        window[x - i + r - 1, y - j + r - 1] = 0
                    else:
                        diff = g_x_y - g_i_j
                        # Add a small epsilon to prevent division by zero
                        window[x - i + r - 1, y - j + r - 1] = (1 / (diff ** 2 + 1e-10)) * sqrt(
                            (x - i) ** 2 + (y - j) ** 2)
            non_zero = window[window != 0]
            if non_zero.size == 0:
                min_value = 0
                contour_matrix[i, j] = 0
            else:
                min_value = np.min(non_zero)
                contour_matrix[i, j] = 1 / min_value
    return contour_matrix
'''
'''
if __name__ == '__main__':
    w_c = cal_weight_contour_matrix('../data/output_img/0/0/0.jpg', 5)
    w_f = cal_weight_feature_matrix('../data/output_img/0/0/0.npy')
    weight_matrix = ((w_c + w_f) ** 2)
    weight_matrix /= np.max(weight_matrix)

    # 加载并处理图像
    img = Image.open("../data/output_img/0/0/0.jpg").convert('RGB')
    img_array = np.array(img)
    if img_array.shape[:2] != (500, 500):
        img = img.resize((500, 500))
        img_array = np.array(img)

    # 创建mask
    white_pixels = np.all(img_array >= 240, axis=-1)
    mask = np.where(white_pixels, 0, 1).astype(np.uint8)
    weight_matrix = np.where(mask, weight_matrix, 0)

    # 创建坐标网格
    x = np.arange(0, 500)
    y = np.arange(0, 500)
    x, y = np.meshgrid(x, y)

    # 创建3D图形
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')

    # 高精细度曲面绘制
    surf = ax.plot_surface(
        x, y, weight_matrix,
        cmap='viridis',
        rstride=1,  # 行步长设为1（最小间隔）
        cstride=1,  # 列步长设为1（最小间隔）
        linewidth=0,  # 线宽设为0（去除网格线）
        antialiased=True,  # 抗锯齿
        shade=True  # 启用阴影效果
    )

    # 添加颜色条
    cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
    cbar.set_label('Weight Value', rotation=270, labelpad=15)

    # 设置标签和标题
    ax.set_xlabel('X Coordinate', labelpad=10)
    ax.set_ylabel('Y Coordinate', labelpad=10)
    ax.set_zlabel('Weight Value', labelpad=10)
    ax.set_title("High-Resolution 3D Surface Plot of Weight Matrix", pad=20)

    # 优化视角
    ax.view_init(elev=40, azim=35)  # 调整视角

    # 设置z轴范围（可选）
    if np.max(weight_matrix) > 0:
        ax.set_zlim(0, np.max(weight_matrix) * 1.1)

    # 提高图形质量
    plt.tight_layout()
    plt.show()
'''
