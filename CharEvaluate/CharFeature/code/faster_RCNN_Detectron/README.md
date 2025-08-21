# Faster R-CNN 训练和推理指南

本目录包含使用 Detectron2 框架训练和推理 Faster R-CNN 模型的代码。

## 项目结构

```
code/faster_RCNN_Detectron/
├── train_faster_rcnn.py        # 训练脚本
├── predict_faster_rcnn.py      # 推理脚本
├── README.md                   # 本说明文件
└── output/                     # 训练输出目录
```

## 配置文件

训练配置保存在 `configs/faster_rcnn_training_config.yaml` 文件中。

## 环境变量设置

在训练和推理之前，需要设置以下环境变量：

```bash
# 数据集根目录（应指向包含VOCdevkit的目录）
export DATASET_ROOT=/path/to/your/dataset

# 测试图像路径（仅推理时需要）
export TEST_IMAGE_PATH=/path/to/test/image.jpg
```

在 Windows 系统中，使用 `set` 命令代替 `export`：

```cmd
set DATASET_ROOT=D:\path\to\your\dataset
set TEST_IMAGE_PATH=D:\path\to\test\image.jpg
```

### 关于测试图片存放位置的说明

根据数据集结构，测试图片可以存放在以下任意位置：
1. `DATASET_ROOT/VOCdevkit/VOC2007/JPEGImages/` 目录下
2. `DATASET_ROOT/VOCdevkit/VOC2012/JPEGImages/` 目录下
3. 项目根目录或其他任何位置（需要提供完整路径）

推荐将测试图片放在项目根目录下，并设置 `TEST_IMAGE_PATH` 为图片的完整路径。

## 训练模型

1. 确保已安装 Detectron2 和相关依赖
2. 设置环境变量 `DATASET_ROOT`
3. 运行训练脚本：

```bash
cd /path/to/project/root
python code/faster_RCNN_Detectron/train_faster_rcnn.py
```

训练结果将保存在 `code/faster_RCNN_Detectron/output` 目录中，最终模型将保存在 `models/faster_rcnn_model_final.pth`。

## 推理测试

1. 确保已训练好模型或下载了预训练模型
2. 设置环境变量 `DATASET_ROOT` 和 `TEST_IMAGE_PATH`
3. 运行推理脚本：

```bash
cd /path/to/project/root
python code/faster_RCNN_Detectron/predict_faster_rcnn.py
```

预测结果将保存在 `code/faster_RCNN_Detectron/output` 目录中。

## 配置说明

配置文件 `configs/faster_rcnn_training_config.yaml` 包含以下主要配置项：

- `model`: 模型配置，包括骨干网络、类别数等
- `train`: 训练配置，包括批次大小、学习率、最大迭代次数等
- `dataset`: 数据集配置，包括数据集名称、路径环境变量等
- `dataloader`: 数据加载配置，包括工作线程数等
- `output`: 输出配置，包括输出目录等
- `solver`: 优化器配置，包括每批次图像数、基础学习率等

根据实际需求修改配置文件中的参数。