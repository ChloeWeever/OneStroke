# API参考

## 主要模块

### 1. features.py - 美学特征计算

#### `calculate_f1_overlap_ratio(template: np.ndarray, copy: np.ndarray) -> float`
计算重叠率特征(f1)

**参数:**
- `template`: 模板图像 (500x500二值图像)
- `copy`: 摹本图像 (500x500二值图像)

**返回:**
- `f1`: 重叠率 [0, 1]

#### `calculate_f2_shape_aesthetic_score(f1: float) -> float`
计算形状美学评分(f2)

**参数:**
- `f1`: 重叠率

**返回:**
- `f2`: 形状美学评分 [0, 10]

#### `calculate_f3_convex_hull_overlap(template: np.ndarray, copy: np.ndarray) -> float`
计算凸包重叠率(f3)

**参数:**
- `template`: 模板图像
- `copy`: 摹本图像

**返回:**
- `f3`: 凸包重叠率 [0, 1]

#### `calculate_f4_to_f11(template: np.ndarray, copy: np.ndarray) -> List[float]`
计算投影特征(f4-f11)

**参数:**
- `template`: 模板图像
- `copy`: 摹本图像

**返回:**
- `[f4, f5, f6, f7, f8, f9, f10, f11]`: 投影特征列表

#### `calculate_f12_keypoint_distance(template_keypoints: np.ndarray, copy_keypoints: np.ndarray) -> float`
计算关键点平均欧氏距离(f12)

**参数:**
- `template_keypoints`: 模板关键点 (N x 2数组)
- `copy_keypoints`: 摹本关键点 (N x 2数组)

**返回:**
- `f12`: 关键点平均欧氏距离

#### `extract_all_features(template: np.ndarray, copy: np.ndarray) -> np.ndarray`
提取所有12个美学特征

**参数:**
- `template`: 模板图像
- `copy`: 摹本图像

**返回:**
- `features`: 12维特征向量

### 2. preprocessing/ - 图像预处理模块

#### `image_preprocessing.py`

##### `preprocess_image(image: np.ndarray) -> np.ndarray`
图像预处理(灰度变换、二值化、去噪)

**参数:**
- `image`: 输入图像

**返回:**
- `processed_image`: 预处理后的图像

##### `adaptive_threshold_binarization(image: np.ndarray) -> np.ndarray`
自适应阈值二值化

**参数:**
- `image`: 灰度图像

**返回:**
- `binary_image`: 二值图像

##### `remove_noise(image: np.ndarray, min_area: int = 50) -> np.ndarray`
连通域去噪

**参数:**
- `image`: 二值图像
- `min_area`: 最小连通域面积

**返回:**
- `clean_image`: 去噪后的图像

#### `image_registration.py`

##### `register_images(copy: np.ndarray, template: np.ndarray) -> Tuple[np.ndarray, np.ndarray]`
图像配准(尺寸配准 + 位置配准)

**参数:**
- `copy`: 摹本图像
- `template`: 模板图像

**返回:**
- `(registered_copy, registered_template)`: 配准后的图像对

##### `size_registration(copy: np.ndarray, template: np.ndarray) -> Tuple[np.ndarray, np.ndarray]`
尺寸配准

**参数:**
- `copy`: 摹本图像
- `template`: 模板图像

**返回:**
- `(resized_copy, resized_template)`: 尺寸配准后的图像对

##### `location_registration(copy: np.ndarray, template: np.ndarray) -> Tuple[np.ndarray, np.ndarray]`
位置配准(重心对齐)

**参数:**
- `copy`: 摹本图像
- `template`: 模板图像

**返回:**
- `(aligned_copy, aligned_template)`: 位置配准后的图像对

### 3. srn/ - 孪生回归网络模块

#### `srn_model.py`

##### `SiameseRegressionNetwork`
孪生回归网络类

**方法:**
- `__init__(backbone: str = 'resnet101')`: 初始化网络
- `forward(template: torch.Tensor, copy: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]`: 前向传播

##### `SRNLoss`
SRN损失函数类

**方法:**
- `__init__()`: 初始化损失函数
- `forward(template_features: torch.Tensor, copy_features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor`: 计算损失

#### `train_srn.py`

##### `train_srn(config_file: str)`
训练SRN模型

**参数:**
- `config_file`: 配置文件路径

### 4. fusion/ - 特征融合模块

#### `feature_fusion.py`

##### `fuse_features(features: Dict[str, Any]) -> np.ndarray`
特征融合

**参数:**
- `features`: 包含各种特征的字典

**返回:**
- `fused_features`: 23维融合特征

#### `regression.py`

##### `AestheticRegressor`
美学回归器类

**方法:**
- `__init__(model_path: str)`: 初始化回归器
- `predict(features: np.ndarray) -> float`: 预测美学评分

#### `train_regressor.py`

##### `train_regressor(config_file: str)`
训练回归器

**参数:**
- `config_file`: 配置文件路径

### 5. evaluation/ - 评估模块

#### `metrics.py`

##### `calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float`
计算平均绝对误差

**参数:**
- `y_true`: 真实值
- `y_pred`: 预测值

**返回:**
- `mae`: 平均绝对误差

##### `calculate_pcc(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]`
计算皮尔逊相关系数

**参数:**
- `y_true`: 真实值
- `y_pred`: 预测值

**返回:**
- `(pcc, p_value)`: 皮尔逊相关系数和p值

### 6. utils/ - 工具模块

#### `dataset.py`

##### `CalligraphyDataset`
书法数据集类

**方法:**
- `__init__(data_dir: str, labels_file: str, transform=None)`: 初始化数据集
- `__len__() -> int`: 返回数据集大小
- `__getitem__(idx: int) -> Tuple[np.ndarray, np.ndarray, float]`: 获取单个样本

##### `load_calligraphy_data(data_dir: str, train_list_file: str, val_list_file: str, test_list_file: str) -> Dict[str, List[Tuple[str, str, float]]]`
加载书法数据

**参数:**
- `data_dir`: 数据目录
- `train_list_file`: 训练集列表文件
- `val_list_file`: 验证集列表文件
- `test_list_file`: 测试集列表文件

**返回:**
- `data_splits`: 数据划分字典

#### `config.py`

##### `Config`
配置管理类

**方法:**
- `__init__(config_file: str = None)`: 初始化配置
- `load_config(config_file: str)`: 加载配置文件
- `get(key: str, default: Any = None) -> Any`: 获取配置项
- `set(key: str, value: Any)`: 设置配置项
- `save_config(config_file: str)`: 保存配置到文件

### 6. 主程序接口

#### `main.py`

##### `evaluate_calligraphy_aesthetic(template_path: str, copy_path: str, srn_model_path: str = None, regressor_model_path: str = None) -> float`
评估书法美学质量

**参数:**
- `template_path`: 模板图像路径
- `copy_path`: 摹本图像路径
- `srn_model_path`: SRN模型路径
- `regressor_model_path`: 回归器模型路径

**返回:**
- `score`: 美学评分 (0-10)

### batch_evaluate

**描述**: 批量评估书法作品的美学评分

**函数签名**: `batch_evaluate(input_csv, output_csv, config_path=None)`

**参数**:
- `input_csv` (str): 输入CSV文件路径
- `output_csv` (str): 输出CSV文件路径
- `config_path` (str, 可选): 配置文件路径

**返回值**: None

## 7. 实验脚本

### `experiments/train.py`
完整训练流程

#### main

**描述**: SRAFE系统的完整训练流程主函数，支持训练SRN模型和回归器

**函数签名**: `main()`

**命令行参数**:
- `--config` (str): 配置文件路径，默认为'configs/training_config.yaml'
- `--train-srn` (flag): 训练SRN模型
- `--train-regressor` (flag): 训练回归器
- `--all` (flag): 训练所有模型
- `--resume` (flag): 从检查点恢复训练

**返回值**: None

#### validate_config

**描述**: 验证配置文件的有效性

**函数签名**: `validate_config(config_path)`

**参数**:
- `config_path` (str): 配置文件路径

**返回值**: Config对象或None（验证失败时）

#### setup_logging

**描述**: 设置训练过程的日志记录

**函数签名**: `setup_logging(log_dir='logs')`

**参数**:
- `log_dir` (str): 日志文件保存目录

**返回值**: None

### `experiments/evaluate.py`
模型评估脚本

#### main

**描述**: SRAFE系统的评估主函数，支持模型性能评估和结果分析

**函数签名**: `main()`

**命令行参数**:
- `--test-list` (str): 测试集列表文件路径，默认为'data/metadata/test_list.csv'
- `--srn-model` (str): SRN模型路径，默认为'models/srn_resnet101.pth'
- `--regressor-model` (str): 回归器模型路径，默认为'models/aesthetic_regressor.txt'
- `--save-details` (flag): 保存详细评估结果
- `--log-dir` (str): 日志保存目录，默认为'logs/evaluate'

**返回值**: None

#### evaluate_model

**描述**: 评估模型性能并计算评估指标

**函数签名**: `evaluate_model(test_list_file, srn_model_path=None, regressor_model_path=None, save_details=True)`

**参数**:
- `test_list_file` (str): 测试集列表文件路径
- `srn_model_path` (str, 可选): SRN模型路径
- `regressor_model_path` (str, 可选): 回归器模型路径
- `save_details` (bool): 是否保存详细评估结果

**返回值**: dict - 评估指标字典，包含MAE、PCC、样本数、失败样本数、评估耗时、误差分析等

#### validate_test_file

**描述**: 验证测试集文件的有效性

**函数签名**: `validate_test_file(test_list_file)`

**参数**:
- `test_list_file` (str): 测试集文件路径

**返回值**: pandas.DataFrame或None（验证失败时）

#### validate_paths

**描述**: 验证文件路径的有效性

**函数签名**: `validate_paths(test_list_file, srn_model_path, regressor_model_path)`

**参数**:
- `test_list_file` (str): 测试集文件路径
- `srn_model_path` (str): SRN模型路径
- `regressor_model_path` (str): 回归器模型路径

**返回值**: tuple - 验证后的路径元组

#### setup_logging

**描述**: 设置评估过程的日志记录

**函数签名**: `setup_logging(log_dir='logs/evaluate')`

**参数**:
- `log_dir` (str): 日志文件保存目录

**返回值**: None

### `experiments/predict.py`
美学评分预测脚本

#### main

**描述**: SRAFE系统的预测主函数，支持单幅图像和批量预测模式

**函数签名**: `main()`

**命令行参数**:
- `--single` (flag): 单个样本预测模式
- `--batch` (flag): 批量预测模式
- `--template` (str): 模板图像路径
- `--copy` (str): 摹本图像路径
- `--input-file` (str): 输入文件路径 (CSV格式)
- `--output-file` (str): 输出文件路径，默认为'results/predictions.csv'
- `--srn-model` (str): SRN模型路径，默认为'models/srn_resnet101.pth'
- `--regressor-model` (str): 回归器模型路径，默认为'models/aesthetic_regressor.txt'
- `--log-dir` (str): 日志保存目录，默认为'logs/predict'

**返回值**: None

#### predict_single

**描述**: 预测单个书法作品的美学评分

**函数签名**: `predict_single(template_path, copy_path, srn_model_path=None, regressor_model_path=None)`

**参数**:
- `template_path` (str): 模板图像路径
- `copy_path` (str): 摹本图像路径
- `srn_model_path` (str, 可选): SRN模型路径
- `regressor_model_path` (str, 可选): 回归器模型路径

**返回值**: float或None（预测失败时）

#### predict_batch

**描述**: 批量预测书法作品的美学评分

**函数签名**: `predict_batch(input_file, output_file, srn_model_path=None, regressor_model_path=None)`

**参数**:
- `input_file` (str): 输入文件路径 (CSV格式，包含template_path和copy_path列)
- `output_file` (str): 输出文件路径
- `srn_model_path` (str, 可选): SRN模型路径
- `regressor_model_path` (str, 可选): 回归器模型路径

**返回值**: tuple - (预测结果数据框, 预测统计信息字典)

#### validate_input_file

**描述**: 验证输入文件的有效性

**函数签名**: `validate_input_file(input_file)`

**参数**:
- `input_file` (str): CSV文件路径

**返回值**: pandas.DataFrame或None（验证失败时）

#### validate_image_path

**描述**: 验证图像路径的有效性

**函数签名**: `validate_image_path(image_path, image_type)`

**参数**:
- `image_path` (str): 图像文件路径
- `image_type` (str): 图像类型描述("模板"或"摹本")

**返回值**: bool - 路径是否有效

#### setup_logging

**描述**: 设置预测过程的日志记录

**函数签名**: `setup_logging(log_dir='logs/predict')`

**参数**:
- `log_dir` (str): 日志文件保存目录

**返回值**: None

## 配置文件

### `configs/training_config.yaml`
训练配置文件

### `configs/inference_config.yaml`
推理配置文件
