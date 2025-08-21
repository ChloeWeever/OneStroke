"""
dataset_srn.py - 数据集加载与处理工具

适配新的数据目录结构:
- copies/: 存储临摹作品（按汉字分目录）
- templates/: 汉字模板图像（.png，直接以汉字命名）
- metadata/: 元信息文件（含标签、划分信息等）
- splits/: 物理划分数据集（含 train/、val/、test / 子目录，非符号链接）
"""

import os
import cv2
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any, Optional
from torch.utils.data import Dataset
from pathlib import Path

class CalligraphyDataset(Dataset):
    """
    书法数据集类，适配新的数据目录结构
    """
    
    def __init__(self, data_dir: str, split: str = 'train', transform: Optional[Any] = None):
        """
        初始化数据集
        
        Args:
            data_dir: 数据目录根路径
            split: 数据集划分 ('train', 'val', 'test')
            transform: 数据变换函数
            
        Raises:
            ValueError: 如果split参数无效
            RuntimeError: 如果数据集初始化失败
        """
        if not os.path.isdir(data_dir):
            raise ValueError(f"数据目录不存在: {data_dir}")
            
        if split not in ['train', 'val', 'test']:
            raise ValueError(f"无效的数据集划分: {split}, 必须是'train', 'val'或'test'之一")
            
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        
        # 定义子目录路径
        self.metadata_dir = os.path.join(data_dir, 'metadata')
        self.templates_dir = os.path.join(data_dir, 'templates')
        self.copies_dir = os.path.join(data_dir, 'copies')
        
        # 加载标签文件
        try:
            labels_file = os.path.join(self.metadata_dir, f'{split}_list.csv')
            self.labels_df = pd.read_csv(labels_file)
            
            # 验证数据框架构
            required_columns = ['template_path', 'copy_path', 'score']
            missing_cols = [col for col in required_columns if col not in self.labels_df.columns]
            if missing_cols:
                raise ValueError(f"标签文件缺少必需列: {', '.join(missing_cols)}")
                
        except FileNotFoundError:
            raise RuntimeError(f"找不到标签文件: {labels_file}")
        except pd.errors.EmptyDataError:
            raise RuntimeError(f"标签文件为空: {labels_file}")
        except Exception as e:
            raise RuntimeError(f"加载标签文件失败: {str(e)}")
            
        # 验证数据有效性
        self.validate_paths()
    
    def __len__(self) -> int:
        """
        返回数据集大小
        """
        return len(self.labels_df)
    
    def validate_paths(self) -> None:
        """
        验证数据集路径的有效性
        
        Raises:
            RuntimeError: 如果发现无效的文件路径
        """
        invalid_templates = []
        invalid_copies = []
        
        for _, row in self.labels_df.iterrows():
            template_path = Path(self.templates_dir) / row['template_path']
            copy_path = Path(self.copies_dir) / row['copy_path']
            
            if not template_path.is_file():
                invalid_templates.append(str(template_path))
            if not copy_path.is_file():
                invalid_copies.append(str(copy_path))
                
        if invalid_templates:
            raise RuntimeError(
                f"无效的模板文件路径 ({len(invalid_templates)} 个):\n" +
                "\n".join(invalid_templates[:5]) +
                ("..." if len(invalid_templates) > 5 else "")
            )
            
        if invalid_copies:
            raise RuntimeError(
                f"无效的临摹文件路径 ({len(invalid_copies)} 个):\n" +
                "\n".join(invalid_copies[:5]) +
                ("..." if len(invalid_copies) > 5 else "")
            )
    
    def load_image(self, image_path: str, default_size: Tuple[int, int] = (500, 500)) -> np.ndarray:
        """
        加载并处理单个图像
        
        Args:
            image_path: 图像文件路径
            default_size: 默认图像尺寸
            
        Returns:
            image: 处理后的图像
            
        Raises:
            RuntimeError: 如果图像加载或处理失败
        """
        try:
            # 使用Path对象处理路径
            path = Path(image_path)
            if not path.is_file():
                raise FileNotFoundError(f"图像文件不存在: {path}")
                
            # 尝试加载图像
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise RuntimeError(f"图像无法读取: {path}")
                
            # 验证图像完整性
            if image.size == 0:
                raise RuntimeError(f"图像为空: {path}")
                
            # 确保图像尺寸正确
            if image.shape != default_size:
                try:
                    image = cv2.resize(image, default_size, interpolation=cv2.INTER_AREA)
                except Exception as e:
                    raise RuntimeError(f"图像缩放失败: {str(e)}")
                    
            # 验证处理后的图像
            if not isinstance(image, np.ndarray) or image.shape != default_size:
                raise RuntimeError(f"图像处理后格式无效: shape={image.shape}")
                
            return image
            
        except Exception as e:
            raise RuntimeError(f"图像加载失败 {image_path}: {str(e)}")
            
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        获取单个样本
        
        Args:
            idx: 数据索引
            
        Returns:
            template: 模板图像 (500x500灰度图)
            copy: 摹本图像 (500x500灰度图)
            label: 美学评分 (0-10的浮点数)
            
        Raises:
            IndexError: 如果索引无效
            RuntimeError: 如果样本加载失败
        """
        if not 0 <= idx < len(self):
            raise IndexError(f"索引{idx}超出范围[0, {len(self)-1}]")
            
        try:
            # 获取样本信息
            row = self.labels_df.iloc[idx]
            
            # 构建完整文件路径
            template_path = Path(self.templates_dir) / row['template_path']
            copy_path = Path(self.copies_dir) / row['copy_path']
            
            # 加载并验证图像
            template = self.load_image(str(template_path))
            copy = self.load_image(str(copy_path))
            
            # 获取并规范化标签
            try:
                label = float(row['score'])
                if not 0 <= label <= 10:
                    label = np.clip(label, 0, 10)
                    print(f"Warning: Sample {idx} - Score {label} clipped to range [0, 10]")
            except (ValueError, TypeError) as e:
                raise RuntimeError(f"样本 {idx} - 分数解析失败: {str(e)}")
            
            # 应用变换
            if self.transform is not None:
                try:
                    template = self.transform(template)
                    copy = self.transform(copy)
                except Exception as e:
                    raise RuntimeError(f"样本 {idx} - 图像变换失败: {str(e)}")
            
            return template, copy, label
            
        except Exception as e:
            raise RuntimeError(f"样本 {idx} 加载失败: {str(e)}")

def load_calligraphy_data(data_dir: str) -> Dict[str, CalligraphyDataset]:
    """
    加载所有数据集分割
    
    Args:
        data_dir: 数据根目录
        
    Returns:
        datasets: 包含训练集、验证集和测试集的字典
        
    Raises:
        RuntimeError: 如果数据加载失败
    """
    try:
        datasets = {}
        for split in ['train', 'val', 'test']:
            try:
                dataset = CalligraphyDataset(data_dir=data_dir, split=split)
                datasets[split] = dataset
                print(f"成功加载{split}集 ({len(dataset)}个样本)")
            except Exception as e:
                print(f"警告: 加载{split}集失败: {str(e)}")
                datasets[split] = None
        
        # 验证是否至少加载了一个数据集
        if all(ds is None for ds in datasets.values()):
            raise RuntimeError("所有数据集加载均失败")
            
        return datasets
        
    except Exception as e:
        raise RuntimeError(f"加载书法数据失败: {str(e)}")
        
def get_dataset_stats(dataset: CalligraphyDataset) -> Dict[str, Any]:
    """
    获取数据集统计信息
    
    Args:
        dataset: 书法数据集实例
        
    Returns:
        stats: 统计信息字典，包含:
            - sample_count: 样本数量
            - mean_score: 平均分数
            - score_std: 分数标准差
            - score_range: 分数范围元组(最小值,最大值)
    """
    if dataset is None or len(dataset) == 0:
        return {
            'sample_count': 0,
            'mean_score': 0.0,
            'score_std': 0.0,
            'score_range': (0.0, 0.0)
        }
        
    try:
        scores = dataset.labels_df['score'].values
        if len(scores) == 0:
            raise ValueError("数据集标签为空")
            
        return {
            'sample_count': len(dataset),
            'mean_score': float(np.mean(scores)),
            'score_std': float(np.std(scores)),
            'score_range': (float(np.min(scores)), float(np.max(scores)))
        }
        
    except Exception as e:
        print(f"警告: 计算数据集统计信息失败: {str(e)}")
        return {
            'sample_count': len(dataset),
            'mean_score': 0.0,
            'score_std': 0.0,
            'score_range': (0.0, 0.0)
        }
        
def create_dummy_metadata_files(root_dir: str = 'data') -> None:
    """
    创建示例元数据文件用于测试
    
    Args:
        root_dir: 数据根目录路径
    """
    # 使用Path对象处理路径
    root = Path(root_dir)
    
    try:
        # 创建必要的目录
        metadata_dir = root / 'metadata'
        templates_dir = root / 'templates'
        copies_dir = root / 'copies'
        
        for dir_path in [metadata_dir, templates_dir, copies_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # 生成示例数据
        splits_data = {
            'train': {
                'template_path': ['一.png', '二.png', '三.png'],
                'copy_path': ['一/student1.png', '二/student2.png', '三/student3.png'],
                'score': [8.5, 7.2, 9.1]
            },
            'val': {
                'template_path': ['四.png', '五.png'],
                'copy_path': ['四/student1.png', '五/student2.png'],
                'score': [8.8, 7.9]
            },
            'test': {
                'template_path': ['明.png', '好.png'],
                'copy_path': ['明/student1.png', '好/student2.png'],
                'score': [9.2, 8.1]
            }
        }
        
        # 创建CSV文件
        for split, data in splits_data.items():
            df = pd.DataFrame(data)
            csv_path = metadata_dir / f'{split}_list.csv'
            df.to_csv(csv_path, index=False)
            print(f"创建{split}集元数据文件: {csv_path}")
            
        # 创建示例图像文件
        def create_dummy_image(path: Path) -> None:
            """创建空白图像文件"""
            path.parent.mkdir(parents=True, exist_ok=True)
            img = np.zeros((500, 500), dtype=np.uint8)
            cv2.imwrite(str(path), img)
            
        # 为每个数据集创建示例图像
        for split_data in splits_data.values():
            for template, copy in zip(split_data['template_path'], split_data['copy_path']):
                create_dummy_image(templates_dir / template)
                create_dummy_image(copies_dir / copy)
                
        print(f"示例数据集创建完成！\n目录结构:\n{root_dir}/\n  ├─ metadata/\n  ├─ templates/\n  └─ copies/")
        
    except Exception as e:
        raise RuntimeError(f"创建示例数据集失败: {str(e)}")

def test_dataset_functionality(data_dir: str = 'data') -> None:  #这个函数的意义在于存在
    """
    测试数据集功能
    
    Args:
        data_dir: 数据根目录
    """
    try:
        print("测试数据集功能...")
        
        # 创建示例数据
        create_dummy_metadata_files(data_dir)
        
        # 加载所有数据集
        datasets = load_calligraphy_data(data_dir)
        print("\n数据集加载结果:")
        for split, dataset in datasets.items():
            if dataset is not None:
                stats = get_dataset_stats(dataset)
                print(f"\n{split}集统计信息:")
                print(f"  样本数量: {stats['sample_count']}")
                print(f"  平均分数: {stats['mean_score']:.2f}")
                print(f"  分数标准差: {stats['score_std']:.2f}")
                print(f"  分数范围: {stats['score_range']}")
                
                # 测试数据加载
                sample_idx = 0
                template, copy, score = dataset[sample_idx]
                print(f"\n测试样本 {sample_idx} 加载:")
                print(f"  模板图像尺寸: {template.shape}")
                print(f"  临摹图像尺寸: {copy.shape}")
                print(f"  评分: {score:.2f}")
                
        print("\n数据集功能测试完成！")
        
    except Exception as e:
        print(f"测试失败: {str(e)}")

# 用于测试的示例代码
if __name__ == "__main__":
    test_dataset_functionality()
