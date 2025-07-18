# OneStroke

## 项目结构

```
OneStroke/
├── StrokeSegmentation/       # 笔画分割模型训练与预测
│   ├── data/                  # 数据集与工具
│   ├── essay_pic/             # 文章中使用的图片
│   ├── predict/               # 模型预测代码
│   └── src/                   # 模型训练代码
├── stroke_collector/          # 笔画采集应用
│   ├── build/                 # 构建输出目录
│   ├── ios/                   # iOS 原生代码
│   ├── lib/                   # Flutter 主要代码
│   └── ...                    # 其他 Flutter 标准文件
├── app/                       # 主应用
│   ├── build/                 # 构建输出目录
│   ├── ios/                   # iOS 原生代码
│   ├── lib/                   # Flutter 主要代码
│   └── ...                    # 其他 Flutter 标准文件
└── README.md
```

---

## 子项目说明

### 1. StrokeSegmentation

用于训练笔画分割模型，使用 U-Net 架构，输入图像后输出每个笔画的分割图。

#### 主要文件

- [src/main.py](file:///Users/luyukai/OneStroke/StrokeSegmentation/src/main.py): 模型训练入口。
- [src/unet_model.py](file:///Users/luyukai/OneStroke/StrokeSegmentation/src/unet_model.py): U-Net 模型定义。
- [src/loss_func.py](file:///Users/luyukai/OneStroke/StrokeSegmentation/src/loss_func.py): 自定义损失函数。
- `data/tools/`: 图像处理工具。
- [predict/predict.py](file:///Users/luyukai/OneStroke/StrokeSegmentation/predict/predict.py): 模型推理脚本。

#### 模型训练迭代记录

- **第一次训练**：使用 BCE 损失函数，交叉点预测不足。
- **第二次训练**：引入加权 BCE，改善交叉点效果，但 v2/v4 类笔画效果下降。
- **第三次训练**：引入注意力机制 + 指数权重，效果提升，但存在轻微过拟合。

#### 训练建议

- 使用更大数据集避免过拟合。
- 尝试轻量模型（如 LeNet）进行笔画提取。

---

### 2. stroke_collector

Flutter 应用，用于采集用户书写笔画数据，支持保存图像与路径信息。

#### 功能模块

- [lib/main.dart](file:///Users/luyukai/OneStroke/app/lib/main.dart): 应用入口。
- [lib/toolbar.dart](file:///Users/luyukai/OneStroke/stroke_collector/lib/toolbar.dart): 工具栏 UI。
- `lib/drawPad.dart`: 手写笔画绘制组件。
- iOS 插件支持：`image_picker`, `permission_handler`, `path_provider` 等。

#### 依赖插件

- `image_picker`: 用于图像选择。
- `permission_handler`: 用于权限管理。
- `path_provider`: 用于路径管理。

#### 构建说明

```bash
cd stroke_collector
flutter pub get
flutter build ios
```

---

### 3. app

主应用，整合笔画分割模型与采集功能，提供完整的用户体验。

#### 功能模块

- `lib/main.dart`: 应用入口。
- `lib/notebook/`: 笔记本与绘图功能。
- `lib/copybook.dart`: 字帖练习模块。
- `lib/forum.dart`: 论坛模块。
- `lib/person.dart`: 个人中心模块。

#### 依赖插件

- `image_picker`
- `permission_handler`
- `path_provider`
- `flutter_svg`

#### 构建说明

```bash
cd app
flutter pub get
flutter build ios
```

---

## 依赖管理

项目依赖的 Python 包（用于模型训练）：

```text
torch
torchvision
numpy
pandas
matplotlib
opencv-python
```

Flutter 插件（用于应用开发）：

```yaml
dependencies:
  flutter:
    sdk: flutter
  image_picker: ^0.8.7+3
  permission_handler: ^10.2.0
  path_provider: ^2.1.1
  flutter_svg: ^2.0.6
```

---

## 构建与部署

### 构建流程

#### 1. 模型训练

```bash
cd StrokeSegmentation
python src/main.py
```

模型将保存在 `models/unet_model.pth` 中。

#### 2. Flutter 应用构建

```bash
cd stroke_collector
flutter build ios
```

或构建 release：

```bash
flutter build ios --release
```

#### 3. 主应用构建

```bash
cd app
flutter build ios
```

---

如需进一步说明或文档支持，请联系项目维护者。
