# 汉语词法分析器 - 增强版

这是一个专门处理未登录词（OOV）问题的汉语词性标注器，基于条件随机场（CRF）模型，具有高级特征工程和完整的评估体系。

## 🚀 快速开始

### 方法一：一键运行（推荐）

**交互模式**（可选择使用现有模型或重新训练）：
```bash
python run_complete_analysis.py
```

**自动模式**（自动使用现有模型，适合脚本环境）：
```bash
python run_complete_analysis.py --auto
```

这个脚本会自动：
- 检查所有必要文件
- 智能检测现有的训练模型
- 让您选择使用现有模型还是重新训练
- 创建时间戳命名的输出目录
- 生成所有评估指标并保存到metrics文件夹
- 创建可视化图表
- 生成详细的分析报告

### 方法二：分步运行
```bash
# 1. 训练和基础评估
python enhanced_pos_tagger.py

# 2. 详细评估（可选）
python evaluation_metrics.py
```

## 📁 输出文件结构

运行完成后，会生成以下目录结构：

```
analysis_results_YYYYMMDD_HHMMSS/
├── models/                          # 训练好的模型文件
│   ├── enhanced_crf_model.pkl       # CRF模型文件
│   └── enhanced_vocab.pkl           # 词汇表和统计信息
├── metrics/                         # 评估指标文件
│   └── detailed_evaluation_results.json  # 详细JSON格式指标
├── visualizations/                  # 可视化图表
│   ├── evaluation_results.png       # 性能指标对比图
│   └── oov_distribution.png         # OOV词分布饼图
└── reports/                         # 详细分析报告
    ├── complete_evaluation_report.txt    # 完整评估报告
    ├── error_analysis_report.txt         # 错误模式分析
    └── oov_processing_demo.txt           # OOV词处理演示
```

## 📊 评估指标说明

### 主要性能指标
- **标注准确率 (Word-level Accuracy)**: 词和词性都正确的比例
- **精确率 (Precision)**: 预测为正例中实际为正例的比例
- **召回率 (Recall)**: 实际正例中被正确预测的比例
- **F1分数**: 精确率和召回率的调和平均数
- **OOV召回率**: 未登录词的正确标注比例

### 特殊功能
- **OOV词检测**: 自动识别和处理未登录词
- **字符级特征**: 利用字符级信息提高OOV词处理能力
- **形态学特征**: 分析词缀、长度、字符类型等特征
- **可视化分析**: 自动生成性能图表和分布图

## 🔧 依赖环境

```python
# 必须的Python包
jieba>=0.42.1
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
sklearn-crfsuite>=0.3.6
matplotlib>=3.4.0
seaborn>=0.11.0
```

安装命令：
```bash
pip install jieba pandas numpy scikit-learn sklearn-crfsuite matplotlib seaborn
```

## 📝 输入数据格式

### 训练数据 (train_pd.txt)
```
我/r 是/v 学生/n
今天/t 天气/n 很/d 好/a
```

### 测试数据 (test_pd_src.txt)
```
他是老师
今天下雨了
```

## 🎯 模型特点

### 1. 增强的特征工程
- **词级别特征**: 词汇、词长、词性历史
- **字符级特征**: 字符频率、字符组合、字符位置
- **形态学特征**: 词缀、数字模式、特殊符号
- **上下文特征**: 前后词汇、词性转移概率

### 2. OOV词处理策略
- **字符级回退**: 利用字符级统计信息
- **形态学分析**: 分析词缀和构词模式
- **模式匹配**: 识别时间、数字、专有名词等模式
- **统计推断**: 基于训练数据的统计信息推断词性

### 3. 完整的评估体系
- **多维度指标**: 整体性能 + OOV词专项性能
- **错误分析**: 详细的错误模式统计和案例分析
- **可视化展示**: 直观的图表展示各项指标
- **实时演示**: OOV词处理效果展示

## 📈 使用建议

1. **首次使用**: 建议使用`run_complete_analysis.py`获得完整的分析结果
2. **模型调优**: 查看错误分析报告，针对性改进特征工程
3. **OOV优化**: 关注OOV词处理演示，了解模型对未登录词的处理效果
4. **性能监控**: 定期运行评估，监控模型在新数据上的表现

## 🔍 故障排除

### 常见问题
1. **编码错误**: 确保训练和测试数据使用正确的中文编码（GBK/UTF-8）
2. **内存不足**: 大数据集可能需要调整batch size或使用更多内存
3. **可视化失败**: 确保安装了中文字体，或者matplotlib配置正确
4. **模型加载失败**: 检查pickle文件是否完整，Python版本是否兼容

### 性能优化建议
- 增加训练数据量提高模型泛化能力
- 调整特征工程策略针对特定领域优化
- 使用更复杂的模型（如BERT）进一步提升性能

## 📧 联系方式

如有问题或建议，请提交Issue或联系开发者。 