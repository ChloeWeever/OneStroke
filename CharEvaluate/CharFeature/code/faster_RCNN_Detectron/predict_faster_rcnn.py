"""
基于 Detectron2 框架的 Faster R-CNN 模型推理脚本
"""

import os
import torch
import cv2
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.model_zoo import model_zoo
import yaml


def setup_predictor(config_path, model_weights_path):
    """
    设置预测器
    """
    # 加载配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    cfg = get_cfg()
    
    # 合并模型配置
    model_config = f"COCO-Detection/faster_rcnn_{config['model']['backbone']}_FPN_3x.yaml"
    cfg.merge_from_file(model_zoo.get_config_file(model_config))
    
    # 设置模型权重
    cfg.MODEL.WEIGHTS = model_weights_path
    
    # 设置模型参数
    cfg.MODEL.RETINANET.NUM_CLASSES = config['model']['num_classes']
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = config['model']['num_classes']
    
    # 设置置信度阈值
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # 设置阈值以过滤低置信度预测
    
    # 创建预测器
    predictor = DefaultPredictor(cfg)
    
    return predictor, config


def predict_and_visualize(predictor, config, image_path, output_dir="output"):
    """
    对单张图像进行预测并可视化结果
    """
    # 读取图像
    im = cv2.imread(image_path)
    
    # 进行预测
    outputs = predictor(im)
    
    # 可视化结果
    # 注意：这里需要根据实际数据集类别进行修改
    MetadataCatalog.get(config['dataset']['train_dataset_name']).thing_classes = ["object"]
    
    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(config['dataset']['train_dataset_name']), scale=1.2)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    
    # 保存结果
    os.makedirs(output_dir, exist_ok=True)
    image_name = os.path.basename(image_path)
    output_path = os.path.join(output_dir, f"result_{image_name}")
    cv2.imwrite(output_path, out.get_image()[:, :, ::-1])
    
    print(f"预测结果已保存到 {output_path}")
    return output_path


def main():
    """
    主函数
    """
    # 配置文件路径
    config_path = "code/configs/faster_rcnn_training_config.yaml"
    
    # 模型权重路径
    model_weights_path = "models/faster_rcnn_model_final.pth"
    
    # 设置预测器
    predictor, config = setup_predictor(config_path, model_weights_path)
    
    # 测试图像路径（需要根据实际情况修改）
    test_image_path = os.environ.get('TEST_IMAGE_PATH', 'test_image.jpg')
    
    # 进行预测和可视化
    if os.path.exists(test_image_path):
        predict_and_visualize(predictor, config, test_image_path, "code/faster_RCNN_Detectron/output")
    else:
        print(f"测试图像 {test_image_path} 不存在，请设置正确的路径。")


if __name__ == "__main__":
    main()