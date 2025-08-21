# SRAFE: Siamese Regression Aesthetic Fusion Evaluation for Chinese Calligraphic Copy

主要基于论文"SRAFE: Siamese Regression Aesthetic Fusion Evaluation for Chinese Calligraphic Copy"、"Non-rigid point set registration: Coherent Point Drift"、detectron2官方文档，实现SRAFE系统。

## 项目概述

SRAFE系统是一个用于中文书法临摹作品美学质量评估的完整解决方案，结合了手工美学特征和深度学习方法。该系统基于以下核心思想：

1. 手工美学特征提取：设计了12个反映书法形状、结构和笔画特征的手工特征
2. 孪生回归网络(SRN)：使用深度学习提取书法的深层美学表示
3. 特征融合：将手工特征与深度特征融合，生成23维综合特征
4. 回归预测：使用LightGBM回归器输出0-10分制的美学评分

## 项目结构

```
calligraphy_aesthetic/
├── README.md
├── requirements.txt
├── code/
│   ├── main.py
│   ├── features.py
│   ├── experiments/
│   │   ├── train.py
│   │   ├── predict.py
│   │   └── evaluate.py
│   ├── preprocessing/
│   │   ├── image_preprocessing.py
│   │   └── image_registration.py
│   ├── srn/
│   │   ├── srn_model.py
│   │   ├── train_srn.py
│   │   └── backbones.py
│   ├── point_matching/
│   │   └── cdp.py
│   ├── faster_RCNN_Detectron/
│   │   ├── predict_faster_rcnn.py
│   │   └── train_faster_rcnn.py
│   ├── fusion/
│   │   ├── feature_fusion.py
│   │   ├── regression.py
│   │   └── train_regressor.py
│   └── utils/
│       ├── point_matching_utils.py
│       ├── dataset_srn.py
│       └── global_config.py
├── configs/
│   ├── training_config.yaml
│   ├── inference_config.yaml
│   ├── point_matching_config.yaml
│   └── faster_rcnn_training_config.yaml
├── data/
│   ├── templates/
│   ├── copies/
│   ├── metadata/
│   └── splits/
├── models/  ##模型存储
│   ├── srn_resnet101.pth
│   └── aesthetic_regressor.txt
└── docs/
    ├── dataset_description.md
    ├── feature_description.md
    └── api_reference.md
```

## 注意事项

- 图像应为500x500的二值图像
- 模板图像应为标准楷体字
- 需要足够的训练数据来获得良好的模型性能
- 评估结果仅供参考，实际美学评价还需结合专家意见

## 系统整体流程

1. 图像预处理：包括灰度变换、二值化、去噪等操作，将输入图像转换为标准的500x500二值图像。
2. 图像配准：通过尺寸配准和位置配准，使模板和摹本图像的字符具有相同的像素数和重心位置。
3. 特征提取：包括手工特征提取（12个美学特征）和深度特征提取（使用孪生回归网络SRN）。
4. 关键点检测：使用Faster R-CNN模型检测书法图像的关键点，当Faster R-CNN检测失败时会降级使用轮廓检测。
5. 特征融合：将手工特征与深度特征进行融合，形成23维特征向量。
6. 美学评分预测：使用LightGBM回归模型对融合特征进行预测，输出0-10分的美学评分。

## 部分模块详细说明

### 1. 图像预处理模块
- 文件：`code/preprocessing/image_preprocessing.py`
- 功能：实现完整的图像预处理流程，包括灰度变换、二值化、去噪和尺寸调整，能够将输入图像处理为标准的500x500二值图像。

### 2. 图像配准模块
- 文件：`code/preprocessing/image_registration.py`
- 功能：实现尺寸配准和位置配准功能，能够使模板和摹本图像的字符具有相同的像素数和重心位置。

### 3. 手工特征提取
- 文件：`code/features.py`
- 功能：实现12个书法美学特征的计算，包括重叠率、形状美学评分、凸包重叠率、投影相似度等，能够全面评估书法的美学特征。

### 4. 深度特征提取
- 文件：`code/srn/srn_model.py`
- 功能：定义SiameseRegressionNetwork类，使用ResNet系列作为骨干网络，能够提取模板与摹本图像的深度特征。

### 5. 关键点检测
- 文件：`code/faster_RCNN_Detectron/predict_faster_rcnn.py`
- 功能：实现基于Detectron2的Faster R-CNN模型推理，能够检测书法图像的关键点。

### 6. 点集配准
- 文件：`code/point_matching/cdp.py`
- 功能：实现Coherent Point Drift (CDP)算法，能够对关键点进行配准。

### 7. 特征融合
- 文件：`code/fusion/feature_fusion.py`
- 功能：实现手工特征与深度特征的融合逻辑，形成23维特征向量。

### 8. 美学评分预测
- 文件：`code/fusion/regression.py`
- 功能：实现基于LightGBM的回归模型，能够对融合特征进行预测，输出美学评分。

每个模块都能独立运行，完成各自任务，最终实现：输入照片，经过预处理后，得到500x500二值图像，最后输出0-10的美学评分。

## 配置文件说明

### 训练配置 (`configs/training_config.yaml`)

- `srn`: SRN模型训练配置
  - `backbone`: 骨干网络类型 (resnet101/resnet50/resnet34/resnet18)
  - `pretrained`: 是否使用预训练权重
  - `learning_rate`: 学习率
  - `batch_size`: 批次大小
  - `epochs`: 训练轮数
  - `device`: 计算设备 (cuda/cpu)

- `regressor`: 回归器训练配置
  - `model_type`: 模型类型 (lightgbm)
  - `params`: 模型参数

- `data`: 数据配置
  - `data_dir`: 数据目录
  - `train_list_file`: 训练集列表文件
  - `val_list_file`: 验证集列表文件
  - `test_list_file`: 测试集列表文件

### 推理配置 (`configs/inference_config.yaml`)

- `models`: 模型路径配置
  - `srn_model_path`: SRN模型路径
  - `regressor_model_path`: 回归器模型路径

- `device`: 计算设备 (cuda/cpu)  ####配置中关于计算设备选择的均是cuda

### 点匹配配置 (`configs/point_matching_config.yaml`)

- `cdp`: CDP算法参数
  - `max_iterations`: 最大迭代次数
  - `tolerance`: 收敛阈值

- `matching`: 特征点匹配参数
  - `min_confidence`: 最小置信度
  - `max_distance`: 最大匹配距离

- `compute`: 计算设备配置
  - `use_gpu`: 是否使用GPU加速
  - `device`: 计算设备 (cuda/cpu)

### Faster R-CNN训练配置 (`configs/faster_rcnn_training_config.yaml`)

- `MODEL`: 模型配置
  - `WEIGHTS`: 预训练权重路径
  - `RESNETS`: ResNet配置
  - `ROI_HEADS`: ROI头部配置

- `SOLVER`: 求解器配置
  - `IMS_PER_BATCH`: 批次大小
  - `BASE_LR`: 基础学习率
  - `MAX_ITER`: 最大迭代次数

- `DATASETS`: 数据集配置
  - `TRAIN`: 训练集
  - `TEST`: 测试集



## 重要说明！！！

#### 1. SRN 模型训练限制
由于训练 SRN 模型的数据尚未收集完毕，暂时无法开展训练，也无法得知模型效果。

#### 2. Detectron2 环境问题
基于 Detectron2 的 Faster RCNN 模型，虽已正确安装所有依赖（如 CUDA）、完成环境配置且 PyTorch 版本与 CUDA 版本匹配，但在安装 Detectron2 时，持续报错 `no module named 'torch'`。尽管已安装 PyTorch，该问题仍存在，经五六天尝试仍未解决。因此暂未使用图片测试整个系统


### 参考安装依赖列表
| Package                          | Version          |
|----------------------------------|------------------|
| absl-py                          | 2.3.1            |
| annotated-types                  | 0.7.0            |
| anyio                            | 4.10.0           |
| attrs                            | 25.3.0           |
| beautifulsoup4                   | 4.13.4           |
| certifi                          | 2025.8.3         |
| charset-normalizer               | 3.4.3            |
| click                            | 8.2.1            |
| cloudpickle                      | 3.1.1            |
| colorama                         | 0.4.6            |
| contourpy                        | 1.3.3            |
| cycler                           | 0.12.1           |
| Cython                           | 3.1.3            |
| filelock                         | 3.18.0           |
| fonttools                        | 4.59.1           |
| fsspec                           | 2025.7.0         |
| future                           | 1.0.0            |
| fvcore                           | 0.1.1.post20200716 |
| grpcio                           | 1.74.0           |
| h11                              | 0.16.0           |
| html5lib                         | 1.1              |
| httpcore                         | 1.0.9            |
| httpx                            | 0.27.2           |
| httpx-sse                        | 0.4.1            |
| idna                             | 3.10             |
| iopath                           | 0.1.10           |
| Jinja2                           | 3.1.6            |
| joblib                           | 1.5.1            |
| jsonschema                       | 4.25.0           |
| jsonschema-specifications        | 2025.4.1         |
| kiwisolver                       | 1.4.9            |
| lightgbm                         | 4.6.0            |
| lxml                             | 6.0.0            |
| Markdown                         | 3.8.2            |
| markdownify                      | 1.2.0            |
| MarkupSafe                       | 3.0.2            |
| matplotlib                       | 3.10.5           |
| mcp                              | 1.12.4           |
| mcp-server-fetch                 | 2025.4.7         |
| mock                             | 5.2.0            |
| mpmath                           | 1.3.0            |
| networkx                         | 3.5              |
| ninja                            | 1.13.0           |
| numpy                            | 2.3.2            |
| opencv-python                    | 4.12.0.88        |
| packaging                        | 25.0             |
| pandas                           | 2.3.1            |
| pillow                           | 11.3.0           |
| pip                              | 25.2             |
| portalocker                      | 3.2.0            |
| Protego                          | 0.5.0            |
| protobuf                         | 6.32.0           |
| pycocotools                      | 2.0.10           |
| pydantic                         | 2.11.7           |
| pydantic-core                    | 2.33.2           |
| pydantic-settings                | 2.10.1           |
| pyparsing                        | 3.2.3            |
| pytest-runner                    | 6.0.1            |
| python-dateutil                  | 2.9.0.post0      |
| python-dotenv                    | 1.11.1           |
| python-multipart                 | 0.0.20           |
| pytz                             | 2025.2           |
| pywin32                          | 311              |
| PyYAML                           | 6.0.2            |
| readability                      | 0.3.0            |
| referencing                      | 0.36.2           |
| regex                            | 2025.7.34        |
| requests                         | 2.32.4           |
| rpds-py                          | 0.27.0           |
| scikit-learn                     | 1.7.1            |
| scipy                            | 1.16.1           |
| setuptools                       | 80.9.0           |
| six                              | 1.17.0           |
| sniffio                          | 1.3.1            |
| soupsieve                        | 2.7              |
| sse-starlette                    | 3.0.2            |
| starlette                        | 0.47.2           |
| sympy                            | 1.14.0           |
| tabulate                         | 0.9.0            |
| tensorboard                      | 2.20.0           |
| tensorboard-data-server          | 0.7.2            |
| termcolor                        | 3.1.0            |
| threadpoolctl                    | 3.6.0            |
| torch                            | 2.8.0            |
| torchaudio                       | 2.8.0            |
| torchvision                      | 0.23.0           |
| tqdm                             | 4.67.1           |
| typing_extensions                | 4.14.1           |
| typing-inspection                | 0.4.1            |
| tzdata                           | 2025.2           |
| urllib3                          | 2.5.0            |
| uvicorn                          | 0.35.0           |
| webencodings                     | 0.5.1            |
| Werkzeug                         | 3.1.3            |
| yacs                             | 0.1.8            |

