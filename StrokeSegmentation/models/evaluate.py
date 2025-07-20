import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path

from StrokeSegmentation.src.predict import UNetPredictor


class Evaluator(nn.Module):
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

    def forward(self, target):
        """
        predictions: [5, H, W] (logits)
        targets: [5, H, W] (二值标签)
        """

        # 计算特征权重（通道求和）
        feature_weight = target.sum(dim=0).float()

        # 计算轮廓权重（优化后的方法）
        contour_weights =self._calculate_contour_weights(target)

        # 合并权重
        weights = (feature_weight + contour_weights) ** 2

        # 归一化权重
        weights = (weights - weights.min()) / (weights.max() - weights.min() + 1e-10)

        return weights * (target.sum(dim=0) > 0).float()

    def _calculate_contour_weights(self, mask):
        """计算单样本的轮廓权重（5个通道）"""
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

    def _calculate_iou(self, pred, target):
        """计算IoU"""
        pred_bool = pred > 0.5  # 假设阈值为 0.5
        target_bool = target > 0.5
        intersection = (pred_bool & target_bool).float().sum()
        print(f"intersection:{intersection}\n")
        union = target_bool.float().sum()
        print(f"union:{union}\n")
        return intersection / (union + 1e-10)

    def _calculate_accuracy_per_class(self, prediction, target, weights, class_idx):
        """计算单个笔画类的准确率"""
        # 提取当前类别的预测和目标
        pred_k = prediction[class_idx]
        target_k = target[class_idx]

        # 计算IoU
        iou = self._calculate_iou(pred_k, target_k)

        # 判断是否为有效笔画
        valid = 1 if iou >= 0.6 else 0
        print(f"第{class_idx + 1}笔画有效性：{valid},IOU:{iou}")
        if valid == 0:
            return 0.0

        # 计算像素级准确率
        correct_pixels = (pred_k == target_k).float()
        weighted_correct = (correct_pixels * weights).sum()
        weighted_total = weights.sum()

        accuracy = 100.0 * (weighted_correct / (weighted_total + 1e-10))
        return accuracy

    def _calculate_accuracy(self, prediction, target_path):
        """计算整字预测准确率"""
        # 计算权重
        targets = self.load_masks_from_dir(target_path)
        weights = self.forward(targets)
        print(f"{weights}\n")

        # 计算每个笔画类的准确率
        class_accuracies = []
        class_areas = []

        for k in range(5):
            # 计算当前笔画类的准确率
            acc = self._calculate_accuracy_per_class(prediction, targets, weights, k)
            class_accuracies.append(acc)

            # 计算当前笔画类的面积
            area = targets[k].sum().item()
            class_areas.append(area)

        # 计算每个笔画类的权重
        total_area = sum(class_areas)
        class_weights = [area / (total_area + 1e-10) for area in class_areas]

        # 计算总准确率
        total_accuracy = sum(w * a for w, a in zip(class_weights, class_accuracies))

        return {
            'class_accuracies': class_accuracies,  # 每个笔画类的准确率(0~100)
            'total_accuracy': total_accuracy,  # 整字总准确率(0~100)
            'class_weights': class_weights  # 每个笔画类的权重
        }

    @staticmethod
    def load_masks_from_dir(directory, prefix='mask_', verify_binary=True):
        """
        从目录加载指定命名的5个mask文件并转换为target张量

        返回:
            torch.Tensor [5, H, W] 符合要求的target张量
        """
        dir_path = Path(directory)
        if not dir_path.is_dir():
            raise ValueError(f"路径 {directory} 不是有效目录")

        # 1. 收集需要的mask文件路径
        mask_paths = []
        missing_masks = []
        for i in range(1, 6):  # mask_1到mask_5
            path = dir_path / f"{prefix}{i}.npy"
            if path.exists():
                mask_paths.append(path)
            else:
                missing_masks.append(path.name)

        if missing_masks:
            raise FileNotFoundError(f"缺少以下mask文件: {', '.join(missing_masks)}")

        # 2. 加载并验证数据
        masks = []
        shapes = set()
        for path in mask_paths:
            arr = np.load(path)

            # # 验证二值数据（如果启用）
            # if verify_binary:
            #     unique_vals = np.unique(arr)
            #     if not set(unique_vals).issubset({0, 1}):
            #         raise ValueError(f"文件 {path.name} 包含非二值数据: {unique_vals}")

            masks.append(arr)
            shapes.add(arr.shape)

        # 3. 检查形状一致性
        if len(shapes) != 1:
            sample_shapes = [f"{path.name}: {arr.shape}" for path, arr in zip(mask_paths, masks)]
            raise ValueError(f"所有mask形状必须相同，检测到:\n" + "\n".join(sample_shapes))

        # 4. 转换为Tensor
        target = np.stack(masks)  # [5, H, W]
        target_tensor = torch.from_numpy(target).float()

        # 强制二值化（即使输入有255等值）
        return (target_tensor > 0.5).float()

if __name__ == '__main__':
    target_path = 'D:\\OneStroke-main\\OneStroke-main\\StrokeSegmentation\\data\\output_img\\0\\0'   #真实图像的二值化掩码
    predictor = UNetPredictor('../models/unet_model_2.pth')  #更改模型
    result = predictor.predict('test.jpg')  # 需要预测图像
    pred_tensor = torch.from_numpy(result).permute(2, 0, 1)
    evaluator = Evaluator()

    # 2. 计算准确率
    accuracy_result = evaluator._calculate_accuracy(pred_tensor, target_path)

    print("\n========== 评估结果 ==========")
    for i, acc in enumerate(accuracy_result["class_accuracies"], 1):
        print(f"笔画类 v{i} 准确率: {acc:.2f}%")

    print(f"\n整字总准确率: {accuracy_result['total_accuracy']:.2f}%")

    print("\n笔画类权重分布:")
    for i, weight in enumerate(accuracy_result["class_weights"], 1):
        print(f"v{i}: {weight:.2%}")