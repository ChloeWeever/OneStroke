import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import List, Dict, Tuple, Callable, Optional, Union
from pathlib import Path
from StrokeSegmentation.predict.predict import UNetPredictor


class Evaluator(nn.Module):
    def __init__(self, base_weight=1.0, radius=5, validity_threshold: float = 0.6):
        super().__init__()
        self.base_weight = base_weight
        self.radius = radius
        # 连通域有效性阈值：预测与真实连通域的重合占真实连通域面积比例
        self.validity_threshold = validity_threshold

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

    def _calculate_accuracy_per_class(self, prediction, target, weights, class_idx):
        """
        基于“整字预测准确率评估算法”的连通域级评估：
        - 对第 class_idx 类真实笔画的每个连通域进行有效性判定（与预测的重合比例 >= 阈值）
        - 有效：按像素权重在该连通域内计算加权准确率（满分100）
        - 无效：计算 wrong_section 惩罚，得分 0~60 范围内（60 * overlap_ratio * penalty）
        - 类别准确率为各连通域分数按连通域面积加权的平均
        """
        # 提取当前类别的预测和目标（转为 numpy 方便与连通域掩码进行逻辑计算）
        if torch.is_tensor(prediction):
            pred_np = prediction.detach().cpu().numpy()
        else:
            pred_np = np.array(prediction)

        if torch.is_tensor(target):
            target_np = target.detach().cpu().numpy()
        else:
            target_np = np.array(target)

        pred_k = pred_np[class_idx] > 0.5
        target_k = target_np[class_idx] > 0.5

        # 获取真实连通域
        true_components = self.get_connected_components(target_k)

        # 若该类没有真实笔画，视为满分
        if len(true_components) == 0:
            return 100.0

        # 计算每个连通域的面积，用于连通域加权
        component_areas = []
        component_scores = []

        # 预取像素权重（torch -> numpy）
        if torch.is_tensor(weights):
            weights_np = weights.detach().cpu().numpy()
        else:
            weights_np = np.array(weights)

        # 其他类别的预测掩码（用于 wrong_section 计算）
        other_pred_np_list = []
        for j in range(5):
            if j == class_idx:
                continue
            other_pred_np_list.append(pred_np[j] > 0.5)

        for _, comp_mask_uint8 in true_components:
            comp_mask = comp_mask_uint8.astype(bool)
            area_true = comp_mask.sum()
            if area_true == 0:
                continue

            # 与当前类别预测的重合比例（相当于 IoU 的分子/真实面积）
            overlap_pixels = np.logical_and(comp_mask, pred_k).sum()
            overlap_ratio = overlap_pixels / (area_true + 1e-10)

            # 计算该连通域内的加权准确率（基础值）
            correct_pixels = np.logical_and(comp_mask, pred_k)
            weights_comp = weights_np * comp_mask
            weighted_correct = (weights_comp * correct_pixels).sum()
            weighted_total = weights_comp.sum()
            base_accuracy = weighted_correct / (weighted_total + 1e-10)

            # 有效连通域：重合比例达到阈值 -> 在该连通域内计算像素加权准确率
            if overlap_ratio >= self.validity_threshold:
                comp_score = 100.0 * base_accuracy
            else:
                # 无效连通域：统计丢失落入其它类别的种类数（wrong_section）
                wrong_section = 0
                for other_pred in other_pred_np_list:
                    if np.logical_and(comp_mask, other_pred).sum() > 0:
                        wrong_section += 1

                # 2. 计算 P_i_k
                P_i_k = wrong_section ** 2

                # 3. 按照算法公式计算准确率
                comp_score = 100.0 * (
                        P_i_k * base_accuracy +
                        (1 - self.validity_threshold * P_i_k) * base_accuracy
                )

                # 确保分数在合理范围内（0-100）
                comp_score = max(0, min(100, comp_score))

            component_areas.append(float(area_true))
            component_scores.append(float(comp_score))

        if len(component_areas) == 0:
            return 100.0

        # 连通域面积加权的类别准确率
        total_area = sum(component_areas)
        class_accuracy = 0.0
        for area, score in zip(component_areas, component_scores):
            class_accuracy += (area / (total_area + 1e-10)) * score

        return class_accuracy

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
            # 计算当前笔画类的准确率（连通域级评估）
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

    def main(
        self,
        target_path: Union[str, Path],
        predictor: Optional[Union[str, Path, object]],
        result_path: Union[str, Path],
        threshold: float = 0.5,
        predict_fn: Optional[Callable[[Union[str, Path], float], np.ndarray]] = None,
    ):
        """
        评估入口
        参数：
            target_path: 真实掩码所在目录，包含 mask_1.npy ... mask_5.npy
            predictor: 兼容三种形式之一：
                - 模型权重文件路径（将自动实例化 UNetPredictor）
                - 具有 predict(image_path, threshold) 方法的对象
                - None（当直接提供 predict_fn 时可为 None）
            result_path: 待评估图像路径
            threshold: 预测二值化阈值（传给 predict）
            predict_fn: 可调用的预测函数，签名与 predict.py 的 predict 保持一致：
                predict_fn(image_path, threshold=0.5) -> np.ndarray[H, W, C]
        优先级：若提供 predict_fn，则优先使用；否则回退到 predictor。
        """

        # 选择预测方式：优先使用外部传入的 predict 函数
        if predict_fn is not None:
            result = predict_fn(result_path, threshold=threshold)
        else:
            # 若传入的是具有 predict 方法的对象，则直接使用；否则视为权重路径
            if predictor is not None and hasattr(predictor, 'predict'):
                result = predictor.predict(result_path, threshold=threshold)
            else:
                # 兼容旧用法：predictor 作为模型权重路径
                predictor_obj = UNetPredictor(predictor)
                result = predictor_obj.predict(result_path, threshold=threshold)
        pred_tensor = torch.from_numpy(result).permute(2, 0, 1)

        # 2. 计算准确率
        accuracy_result = self._calculate_accuracy(target_path,pred_tensor)

        return accuracy_result


if __name__ == '__main__':
    ture_mask_path = Path("../data/output_img/39/0")  # 真实图像的二值化掩码
    weight_path = Path("../models/unet_model_3.pth")  # 更改模型
    result_path = Path("../data/output_img/39/0/0.jpg")  # 需要预测图像
    evaluator = Evaluator(validity_threshold=0.6)

    # 兼容本项目unet模型代码
    # accuracy_result = evaluator.main(ture_mask_path, weight_path, result_path, threshold=0.5)

    # 其余模型调用请仿照此处，添加模型实例化代码,其中模型预测返回结果result形状为 (H, W, 1)
    model = UNetPredictor(weight_path)
    accuracy_result = evaluator.main(ture_mask_path, None, result_path, threshold=0.5, predict_fn=model.predict)

    print("\n========== 评估结果 ==========")
    for i, acc in enumerate(accuracy_result["class_accuracies"], 1):
        print(f"笔画类 v{i} 准确率: {acc:.2f}%")

    print(f"\n整字总准确率: {accuracy_result['total_accuracy']:.2f}%")

    print("\n笔画类权重分布:")
    for i, weight in enumerate(accuracy_result["class_weights"], 1):
        print(f"v{i}: {weight:.2%}")
