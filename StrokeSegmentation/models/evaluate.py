import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import List, Dict, Tuple
from pathlib import Path
from StrokeSegmentation.predict.predict import UNetPredictor


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
        contour_weights = self._calculate_contour_weights(target)

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
        pred_bool = pred > 0.5
        target_bool = target > 0.5
        #print(f"pred_bool:{pred_bool.float().sum()}\n")
        #print(f"target_bool:{target_bool.float().sum()}\n")
        intersection = (pred_bool & target_bool).float().sum()
        # print(f"intersection:{intersection}\n")
        union = target_bool.float().sum()
        # print(f"union:{union}\n")
        if union == 0:
            return -1.0
        return intersection / (union + 1e-10)

    def _calculate_accuracy_per_class(self, prediction, target, weights, class_idx):
        """
        计算单个笔画类的准确率

        返回：
            accuracy 单个笔画类准确率
        """
        # 提取当前类别的预测和目标
        pred_k = prediction[class_idx]
        target_k = target[class_idx]

        # 计算IoU

        iou = self._calculate_iou(pred_k, target_k)
        # print(f"第{class_idx + 1}笔画IOU:{iou}")

        # print(f"第{class_idx + 1}笔画有效性：{valid},IOU:{iou}")
        if iou >= 0.6:
            # 计算像素级准确率
            correct_pixels = (pred_k == target_k).float()
            weighted_correct = (correct_pixels * weights).sum()
            weighted_total = weights.sum()
            accuracy = 100.0 * (weighted_correct / (weighted_total + 1e-10))
            return accuracy
        elif 0.0 <= iou < 0.6:
            # 获取所有连通域
            true_components = self.get_connected_components(target_k)
            pred_components = self.get_connected_components(pred_k)

            # 计算wrong_section
            wrong_section = 0
            for _, true_component in true_components:
                # 检查该真实连通域是否与其它预测笔画类别有交集
                for other_class in range(5):
                    if other_class == class_idx:  # 跳过当前类
                        continue

                    # 获取其他类别的预测掩码
                    other_pred = prediction[other_class]
                    other_pred_components = self.get_connected_components(other_pred)

                    # 检查是否有交集
                    has_intersection = False
                    for _, pred_component in pred_components:
                        for _, other_pred_component in other_pred_components:
                            if np.logical_and(true_component, other_pred_component).sum() > 0:
                                has_intersection = True
                                break
                        if has_intersection:
                            break

                    if has_intersection:
                        wrong_section += 1
            print(f"第{class_idx + 1}笔画wrong_section:{wrong_section}")
            # 计算优化后的分数 (限制在0-60范围内)

            penalty = 1.0 / (1 + wrong_section) ** 0.5
            score = 60 * (iou * penalty)
            print(f"第{class_idx + 1}笔画score:{score}")
            return score  # 确保分数在0-60之间
        elif iou == -1.0:
            return 100.0

    def _calculate_accuracy(self, target_path, prediction):
        """
        计算整字预测准确率

        返回：
            'class_accuracies': class_accuracies,  # 每个笔画类的准确率(0~100)
            'total_accuracy': total_accuracy,  # 整字总准确率(0~100)
            'class_weights': class_weights  # 每个笔画类的权重
        """
        # 计算权重
        targets = self.load_masks_from_dir(target_path)
        weights = self.forward(targets)
        # print(f"{weights}\n")

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

    def load_masks_from_dir(self, directory, prefix='mask_', verify_binary=True):
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
            masks.append(arr)
            shapes.add(arr.shape)

        # 3. 检查形状一致性
        if len(shapes) != 1:
            sample_shapes = [f"{path.name}: {arr.shape}" for path, arr in zip(mask_paths, masks)]
            raise ValueError(f"所有mask形状必须相同，检测到:\n" + "\n".join(sample_shapes))

        # 4. 转换为Tensor
        target = np.stack(masks)
        target_tensor = torch.from_numpy(target).float()

        # 强制二值化（即使输入有255等值）
        return (target_tensor > 0.5).float()

    def get_connected_components(self, mask) -> List[Tuple[int, np.ndarray]]:
        """
        获取单个笔画mask中的所有连通域

        返回:
            连通域列表，每个元素是(连通域编号, 该连通域的mask)
        """
        if torch.is_tensor(mask):
            mask_np = mask.cpu().numpy()
        else:
            mask_np = np.array(mask)

        # 使用OpenCV的连通域分析
        mask_np = ((mask_np > 0.5) * 255).astype(np.uint8)

        # 使用OpenCV的连通域分析
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_np, connectivity=8)

        # 提取各个连通域
        components = []
        for label in range(1, num_labels):  # 跳过背景(0)
            component_mask = (labels == label).astype(np.uint8)
            components.append((label, component_mask))

        return components

    def get_all_connected_components(self, target) -> Dict[int, List[Tuple[int, np.ndarray]]]:
        """
        获取所有五类笔画的连通域

        返回:
            字典，键是笔画类别(0-4)，值是该类笔画的所有连通域列表
        """
        all_components = {}

        if torch.is_tensor(target):
            target_np = target.cpu().numpy()
        else:
            target_np = np.array(target)

        for k in range(5):  # 五类笔画
            mask = target_np[k]  # 获取第k类笔画
            components = self.get_connected_components(mask)
            all_components[k] = components

        return all_components

    def main(self, target_path, predictor, result_path):
        predictor = UNetPredictor(predictor)  # 更改模型
        result = predictor.predict(result_path)  # 需要预测图像
        pred_tensor = torch.from_numpy(result).permute(2, 0, 1)

        # 2. 计算准确率
        accuracy_result = self._calculate_accuracy(target_path,pred_tensor)

        return accuracy_result


if __name__ == '__main__':
    target_path = Path("..\data\output_img\\12\\0")  # 真实图像的二值化掩码
    predictor = Path("../models/unet_model_3.pth")  # 更改模型
    result = Path("..\data\output_img\\12\\0\\0.jpg")  # 需要预测图像
    evaluator = Evaluator()

    # 2. 计算准确率
    accuracy_result = evaluator.main(target_path, predictor, result)

    print("\n========== 评估结果 ==========")
    for i, acc in enumerate(accuracy_result["class_accuracies"], 1):
        print(f"笔画类 v{i} 准确率: {acc:.2f}%")

    print(f"\n整字总准确率: {accuracy_result['total_accuracy']:.2f}%")

    print("\n笔画类权重分布:")
    for i, weight in enumerate(accuracy_result["class_weights"], 1):
        print(f"v{i}: {weight:.2%}")
