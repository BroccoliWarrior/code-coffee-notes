# 血细胞检测系统

基于HOG特征提取 + PCA降维 + SVM分类器的血细胞目标检测系统

## 项目简介

本项目实现了一个完整的血细胞检测系统，能够识别和定位血液涂片中的红细胞、白细胞和血小板。系统采用传统机器学习方法，具有以下特点：

- **特征提取**: 使用HOG (Histogram of Oriented Gradients) 提取图像特征
- **降维处理**: 通过PCA降维减少特征维度，提高训练效率
- **分类器**: 采用支持向量机(SVM)进行多类别分类
- **目标检测**: 结合选择性搜索生成候选区域
- **后处理**: 使用非极大值抑制(NMS)消除重复检测
- **数据增强**: 支持多种数据增强技术平衡样本分布

## 系统架构

```
血细胞检测系统
├── 数据加载模块 (data_loader.py)
│   ├── BCCD数据集解析
│   ├── XML标注文件处理
│   └── 正负样本提取
├── 特征提取模块 (feature_extractor.py)
│   ├── HOG特征提取
│   ├── 图像预处理
│   └── PCA降维
├── 检测模块 (detector.py)
│   ├── 选择性搜索
│   ├── SVM分类器
│   └── NMS后处理
├── 可视化模块 (visualizer.py)
│   ├── 检测结果可视化
│   ├── 训练样本展示
│   └── 评估指标图表
├── 数据增强模块 (data_augmentation.py)
│   ├── 图像变换
│   ├── 样本均衡
│   └── 批量增强
└── 主程序 (main.py)
    ├── 训练模式
    ├── 预测模式
    ├── 评估模式
    └── 演示模式
```

## 安装依赖

### 使用pip安装
```bash
pip install -r requirements.txt
```

### 主要依赖包
- Python >= 3.7
- OpenCV >= 4.5.0
- scikit-learn >= 1.0.0
- scikit-image >= 0.18.0
- matplotlib >= 3.3.0
- numpy >= 1.19.0
- tqdm >= 4.60.0
- Pillow >= 8.0.0

## 数据集准备

### BCCD数据集结构
```
BCCD/
├── JPEGImages/          # 图像文件
│   ├── BloodImage_001.jpg
│   ├── BloodImage_002.jpg
│   └── ...
├── Annotations/         # XML标注文件
│   ├── BloodImage_001.xml
│   ├── BloodImage_002.xml
│   └── ...
├── ImageSets/Main/      # 数据分割文件
│   ├── train.txt        # 训练集图像列表
│   ├── val.txt          # 验证集图像列表
│   └── test.txt         # 测试集图像列表
└── classes.txt          # 类别名称文件
```

### 数据集下载
1. 从 [BCCD官方仓库](https://github.com/Shenggan/BCCD_Dataset) 下载数据集
2. 解压到项目根目录下的 `BCCD` 文件夹

## 使用方法

### 1. 训练模型
```bash
python main.py --mode train --dataset_path BCCD --output_dir outputs --target_size 64 64 --hog_orientations 9 --pca_components 128 --variance_ratio 0.95 --svm_c 1.0 --svm_kernel rbf --negative_samples 2000 --enable_augmentation --balance_samples --visualize --save_images
```

**训练参数说明：**
- `--target_size`: HOG特征提取的目标图像尺寸
- `--hog_orientations`: HOG方向梯度数量
- `--pca_components`: PCA主成分数量
- `--variance_ratio`: PCA保留的方差比例
- `--svm_c`: SVM正则化参数
- `--negative_samples`: 负样本数量
- `--enable_augmentation`: 启用数据增强
- `--balance_samples`: 平衡各类别样本数量

### 2. 预测检测
```bash
python main.py --mode predict --dataset_path BCCD --output_dir outputs --model_path outputs/models/detector.pkl --feature_model_path outputs/models/feature_extractor.pkl --confidence_threshold 0.95 --nms_threshold 0.3 --num_predict_images 20 --num_vis_images 6 --visualize --save_images
```

**预测参数说明：**
- `--confidence_threshold`: 置信度阈值
- `--nms_threshold`: 非极大值抑制阈值
- `--num_predict_images`: 预测的图像数量
- `--num_vis_images`: 可视化的图像数量

### 3. 模型评估
```bash
python main.py --mode evaluate --dataset_path BCCD --output_dir outputs --model_path outputs/models/detector.pkl --feature_model_path outputs/models/feature_extractor.pkl --confidence_threshold 0.95 --nms_threshold 0.3 --iou_threshold 0.15 --num_vis_images 6 --visualize
```

**评估参数说明：**
- `--iou_threshold`: IoU匹配阈值
- 生成详细的评估指标图表
- 输出准确率、精确率、召回率、F1分数

### 4. 演示模式
```bash
python main.py --mode demo --dataset_path BCCD --output_dir outputs --target_size 64 64 --hog_orientations 9 --pca_components 128 --variance_ratio 0.95 --svm_c 1.0 --svm_kernel rbf --negative_samples 1500 --enable_augmentation --balance_samples --confidence_threshold 0.95 --nms_threshold 0.3 --iou_threshold 0.15 --num_predict_images 15 --num_vis_images 6 --visualize --save_images
```

演示模式会依次运行训练→预测→评估三个阶段。

## 输出结果

### 文件结构
```
outputs/
├── models/                          # 训练好的模型
│   ├── feature_extractor.pkl       # HOG特征提取器+PCA
│   └── detector.pkl                 # SVM检测器
├── results/                         # 结果文件
│   ├── training_result.json        # 训练结果
│   ├── prediction_summary.json     # 预测摘要
│   ├── evaluation_result.json      # 评估结果
│   └── [image_name]_result.jpg     # 检测结果图像
└── visualizations/                  # 可视化图表
    ├── training_samples.png         # 训练样本展示
    ├── predictions.png              # 预测结果可视化
    ├── evaluation_comparison.png    # 评估对比图
    └── evaluation_metrics.png       # 评估指标图表
```

### 评估指标
- **准确率 (Accuracy)**: 正确匹配的检测 / 总匹配数
- **精确率 (Precision)**: 正确检测 / 总检测数
- **召回率 (Recall)**: 匹配的目标 / 真实目标总数
- **F1分数**: 精确率和召回率的调和平均

## 技术特点

### HOG特征提取
- 9个方向梯度直方图
- 8x8像素单元格
- 2x2单元格块归一化
- 支持图像预处理(灰度化、直方图均衡化)

### PCA降维
- 自动选择主成分数量
- 保留95%方差信息
- 显著减少特征维度

### SVM分类器
- 一对多分类策略
- RBF核函数
- 概率输出支持

### 选择性搜索
- Fast/Quality两种策略
- 可配置候选区域数量
- 最小区域大小过滤

### 数据增强
- 旋转、翻转、亮度调节
- 对比度变换、噪声添加
- 弹性变形
- 自动样本均衡

## 参数调优建议

### 检测效果不佳时
1. **降低置信度阈值**: `--confidence_threshold 0.3`
2. **降低IoU阈值**: `--iou_threshold 0.1`
3. **增加负样本**: `--negative_samples 3000`
4. **启用数据增强**: `--enable_augmentation --balance_samples`

### 训练速度优化
1. **减少PCA组件**: `--pca_components 64`
2. **降低方差保留**: `--variance_ratio 0.9`
3. **减少候选区域**: `--max_proposals 1000`

### 内存优化
1. **减小目标尺寸**: `--target_size 32 32`
2. **限制处理图像**: `--num_predict_images 10`

## 常见问题

### Q: 训练时间过长？
A: 可以：
- 减少负样本数量
- 降低PCA组件数
- 减小图像尺寸
- 使用线性SVM核

### Q: 检测精度低？
A: 建议：
- 增加训练数据
- 调整SVM参数C值
- 尝试不同的HOG参数
- 启用数据增强和样本均衡
