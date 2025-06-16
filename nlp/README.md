# NLP自然语言处理项目

本项目包含9个自然语言处理相关的任务，涵盖了从基础文本处理到深度学习模型应用的多个方面。

## 环境要求

- Python 3.8+
- 依赖包：
  - nltk >= 3.8
  - jieba >= 0.42
  - gensim >= 4.3.0
  - numpy >= 1.21.0
  - wordcloud >= 1.9.0
  - matplotlib >= 3.5.0
  - torch >= 2.0.0

## 任务说明

### 任务1：NLTK Gutenberg语料库分析
- 文件：`task1_nltk_gutenberg.py`
- 功能：使用NLTK库分析Gutenberg语料库中的文本数据
- 主要实现：
  - 下载并加载Gutenberg语料库
  - 文本预处理：去除标点符号、转换为小写
  - 词频统计和排序
  - 文本长度分布分析
  - 使用matplotlib绘制词频分布图
  - 生成文本统计报告

### 任务2：唐诗分析
- 文件：`task2_tang_poetry.py`
- 功能：对唐诗进行文本分析和处理
- 主要实现：
  - 加载唐诗数据集
  - 使用jieba进行分词处理
  - 统计诗人作品数量
  - 分析诗歌长度分布
  - 提取高频词汇
  - 分析诗歌主题分布
  - 生成诗歌特征统计报告

### 任务3：分词处理
- 文件：`task3_tokenization.py`
- 功能：实现中英文文本的分词处理
- 主要实现：
  - 中文分词：
    - 使用jieba进行精确模式分词
    - 使用jieba进行全模式分词
    - 使用jieba进行搜索引擎模式分词
  - 英文分词：
    - 使用NLTK的word_tokenize进行分词
    - 使用NLTK的sent_tokenize进行分句
  - 分词结果评估和对比
  - 自定义词典支持

### 任务4：词性标注
- 文件：`task4_pos_tagging.py`
- 功能：对文本进行词性标注
- 主要实现：
  - 中文词性标注：
    - 使用jieba进行词性标注
    - 支持自定义词性标注规则
  - 英文词性标注：
    - 使用NLTK的pos_tag进行标注
    - 支持Penn Treebank词性标注集
  - 词性标注结果可视化
  - 词性分布统计

### 任务5：词语相似度计算
- 文件：`task5_word_similarity.py`
- 功能：计算词语之间的相似度
- 主要实现：
  - 加载预训练词向量模型
  - 实现多种相似度计算方法：
    - 余弦相似度
    - 欧氏距离
    - 曼哈顿距离
  - 词语类比关系计算
  - 相似词查找
  - 可视化相似度结果

### 任务6：词云生成
- 文件：`task6_wordcloud.py`
- 功能：生成文本的词云可视化
- 主要实现：
  - 文本预处理和分词
  - 词频统计
  - 自定义词云参数：
    - 字体设置
    - 颜色方案
    - 形状设置
  - 生成词云图
  - 保存和展示结果

### 任务7：命名实体识别
- 文件：`task7_ner.py`
- 功能：识别文本中的命名实体
- 主要实现：
  - 使用NLTK的命名实体识别器
  - 支持识别的实体类型：
    - 人名（PERSON）
    - 组织（ORGANIZATION）
    - 地点（LOCATION）
    - 时间（TIME）
    - 日期（DATE）
  - 实体识别结果可视化
  - 实体统计和分析

### 任务8：LSTM正弦余弦预测
- 文件：`task8_lstm_sincos.py`
- 功能：使用LSTM模型预测正弦和余弦函数
- 主要实现：
  - 数据生成和预处理
  - LSTM模型构建：
    - 输入层
    - LSTM层
    - 全连接层
  - 模型训练：
    - 损失函数定义
    - 优化器设置
    - 训练循环实现
  - 预测结果可视化
  - 模型评估

### 任务9：LSTM诗歌生成
- 文件：`task9_lstm_poetry.py`
- 功能：使用LSTM模型生成诗歌
- 主要实现：
  - 诗歌数据预处理：
    - 文本清洗
    - 字符级编码
    - 序列生成
  - LSTM模型构建：
    - 嵌入层
    - 多层LSTM
    - Dropout层
    - 全连接层
  - 模型训练：
    - 批次生成
    - 训练过程监控
    - 模型保存
  - 诗歌生成：
    - 温度采样
    - 多样性控制
    - 格式约束
  - 生成结果评估

## 项目结构

```
.
├── data/               # 数据文件目录
├── models/            # 模型文件目录
├── result/            # 结果输出目录
├── task1_nltk_gutenberg.py
├── task2_tang_poetry.py
├── task3_tokenization.py
├── task4_pos_tagging.py
├── task5_word_similarity.py
├── task6_wordcloud.py
├── task7_ner.py
├── task8_lstm_sincos.py
├── task9_lstm_poetry.py
└── requirements.txt
```

## 运行说明

1. 安装依赖：
```bash
pip install -r requirements.txt
```

2. 运行具体任务：
```bash
python taskX_*.py
```

## 注意事项

- 运行任务前请确保已安装所有依赖包
- 部分任务可能需要下载额外的NLTK数据
- 确保data目录中包含所需的数据文件
- 对于深度学习相关任务（任务8和任务9），建议使用GPU进行训练
- 部分任务可能需要较大的内存空间，请确保系统资源充足 