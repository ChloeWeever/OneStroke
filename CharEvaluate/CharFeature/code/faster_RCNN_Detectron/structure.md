# data_pascalvoc 文件夹结构（VOC训练数据集存放结构）

```
📁 pascalvoc
├── 📄 label_list
├── 📄 train.txt
└── 📁 VOCdevkit
    ├── 📁 VOC2007
    │   ├── 📁 Annotations
    │   │   ├── 000001.xml
    │   │   ├── 000002.xml
    │   │   ├── 000003.xml
    │   │   ├── 000004.xml
    │   │   ├── 000005.xml
    │   │   ├── 000006.xml
    │   │   ├── 000007.xml
    │   │   ├── 000008.xml
    │   │   ├── 000009.xml
    │   │   ├── 000010.xml
    │   │   └── ... (更多 xml 文件)
    │   ├── 📁 ImageSets
    │   ├── 📁 JPEGImages
    │   ├── 📁 SegmentationClass
    │   └── 📁 SegmentationObject
    └── 📁 VOC2012
        ├── 📁 Annotations
        ├── 📁 ImageSets
        ├── 📁 JPEGImages
        ├── 📁 SegmentationClass
        └── 📁 SegmentationObject
```

## 文件夹说明

- 📁 表示文件夹
- 📄 表示文件
- VOC2007 和 VOC2012 具有相同的子文件夹结构
- Annotations 文件夹中包含大量的 xml 标注文件（已省略部分显示）
