"""
基于 Detectron2 框架的 Faster R-CNN 模型训练脚本
遵循项目结构要求，通过配置文件和环境变量管理路径
"""

import os
import torch
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.model_zoo import model_zoo
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer
from detectron2.evaluation import COCOEvaluator
import yaml

# 设置日志
setup_logger()


def setup_cfg(args, config_path):
    """
    创建配置
    """
    cfg = get_cfg()
    
    # 加载配置文件
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 合并基础模型配置
    model_config = f"COCO-Detection/faster_rcnn_{config['model']['backbone']}_FPN_3x.yaml"
    cfg.merge_from_file(model_zoo.get_config_file(model_config))
    
    # 设置模型权重
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_config)
    
    # 设置模型参数
    cfg.MODEL.RETINANET.NUM_CLASSES = config['model']['num_classes']
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = config['model']['num_classes']
    
    # 获取数据集根目录（从环境变量）
    dataset_root = os.environ.get(config['dataset']['data_dir_env'], 'data')
    
    # 注册训练集和验证集
    train_dataset_name = config['dataset']['train_dataset_name']
    val_dataset_name = config['dataset']['val_dataset_name']
    
    # 为VOC数据集创建数据集字典
    def create_voc_dataset_dict(dataset_root, voc_subdirs, imageset_file, annotations_subdir, images_subdir):
        """创建VOC数据集字典"""
        dataset_dicts = []
        
        # 遍历所有VOC子目录
        for voc_subdir in voc_subdirs:
            voc_dir = os.path.join(dataset_root, 'VOCdevkit', voc_subdir)
            imageset_path = os.path.join(voc_dir, imageset_file)
            annotations_dir = os.path.join(voc_dir, annotations_subdir)
            images_dir = os.path.join(voc_dir, images_subdir)
            
            # 读取ImageSets文件中的图像ID
            with open(imageset_path, 'r') as f:
                image_ids = [line.strip() for line in f.readlines()]
            
            # 为每个图像创建数据字典
            for image_id in image_ids:
                record = {}
                
                # 图像文件路径
                image_file = os.path.join(images_dir, f"{image_id}.jpg")
                if not os.path.exists(image_file):
                    # 尝试png格式
                    image_file = os.path.join(images_dir, f"{image_id}.png")
                
                # 获取图像尺寸
                import cv2
                height, width = cv2.imread(image_file).shape[:2]
                
                record["file_name"] = image_file
                record["image_id"] = image_id
                record["height"] = height
                record["width"] = width
                
                # 标注文件路径
                annotation_file = os.path.join(annotations_dir, f"{image_id}.xml")
                
                # 解析XML标注文件
                from xml.etree import ElementTree as ET
                tree = ET.parse(annotation_file)
                root = tree.getroot()
                
                objs = []
                for obj in root.findall('object'):
                    # 获取边界框坐标
                    bbox = obj.find('bndbox')
                    xmin = int(bbox.find('xmin').text)
                    ymin = int(bbox.find('ymin').text)
                    xmax = int(bbox.find('xmax').text)
                    ymax = int(bbox.find('ymax').text)
                    
                    obj_dict = {
                        "bbox": [xmin, ymin, xmax, ymax],
                        "bbox_mode": 0,  # XYXY_ABS格式
                        "category_id": 0,  # 根据实际类别数调整
                        "iscrowd": 0
                    }
                    objs.append(obj_dict)
                
                record["annotations"] = objs
                dataset_dicts.append(record)
        
        return dataset_dicts
    
    # 注册训练集和验证集
    def register_voc_dataset(dataset_name, dataset_root, voc_subdirs, imageset_file, annotations_subdir, images_subdir):
        """注册VOC数据集"""
        from detectron2.data import DatasetCatalog, MetadataCatalog
        
        DatasetCatalog.register(dataset_name, lambda: create_voc_dataset_dict(
            dataset_root, voc_subdirs, imageset_file, annotations_subdir, images_subdir))
        MetadataCatalog.get(dataset_name).set(thing_classes=["object"])  # 根据实际类别修改
    
    # 注册训练集和验证集
    register_voc_dataset(
        train_dataset_name, dataset_root, config['dataset']['voc_subdirs'],
        config['dataset']['train_imagesets'], config['dataset']['annotations_subdir'],
        config['dataset']['images_subdir'])
    
    register_voc_dataset(
        val_dataset_name, dataset_root, config['dataset']['voc_subdirs'],
        config['dataset']['val_imagesets'], config['dataset']['annotations_subdir'],
        config['dataset']['images_subdir'])
    
    # 设置类别
    # 注意：这里需要根据实际数据集类别进行修改
    MetadataCatalog.get(train_dataset_name).thing_classes = ["object"]  # 根据实际类别修改
    MetadataCatalog.get(val_dataset_name).thing_classes = ["object"]    # 根据实际类别修改
    
    # 设置数据集
    cfg.DATASETS.TRAIN = (train_dataset_name,)
    cfg.DATASETS.TEST = (val_dataset_name,)
    
    # 设置数据加载器
    cfg.DATALOADER.NUM_WORKERS = config['dataloader']['num_workers']
    
    # 设置求解器参数
    cfg.SOLVER.IMS_PER_BATCH = config['solver']['ims_per_batch']
    cfg.SOLVER.BASE_LR = config['solver']['base_lr']
    cfg.SOLVER.MAX_ITER = config['train']['max_iterations']
    cfg.SOLVER.STEPS = [config['solver']['step_size']]  # 在此迭代次数时降低学习率
    cfg.SOLVER.GAMMA = config['solver']['gamma']  # 学习率衰减因子
    
    # 设置检查点和评估周期
    cfg.SOLVER.CHECKPOINT_PERIOD = config['train']['checkpoint_period']
    cfg.TEST.EVAL_PERIOD = config['train']['eval_period']
    
    # 设置输出目录
    cfg.OUTPUT_DIR = config['output']['output_dir']
    
    # 创建输出目录（如果不存在）
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


class Trainer(DefaultTrainer):
    """
    自定义训练器，添加评估器
    """
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, output_dir=output_folder)


def main(args):
    """
    主函数
    """
    # 配置文件路径
    config_path = "configs/faster_rcnn_training_config.yaml"
    
    # 设置配置
    cfg = setup_cfg(args, config_path)
    
    # 创建训练器
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=False)
    
    # 开始训练
    trainer.train()
    
    # 保存最终模型到指定目录
    model_save_dir = cfg.MODEL_SAVE_DIR if hasattr(cfg, 'MODEL_SAVE_DIR') else 'models'
    os.makedirs(model_save_dir, exist_ok=True)
    
    # 复制最终模型
    final_model_path = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    if os.path.exists(final_model_path):
        import shutil
        shutil.copy2(final_model_path, os.path.join(model_save_dir, "faster_rcnn_model_final.pth"))
        print(f"模型已保存到 {os.path.join(model_save_dir, 'faster_rcnn_model_final.pth')}")


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("命令行参数:", args)
    main(args)