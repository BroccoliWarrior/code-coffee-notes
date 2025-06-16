#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强版汉语词法分析器 - 专门解决未登录词问题
包含更高级的特征工程和OOV处理策略
"""

import re
import jieba
import jieba.posseg as pseg
import pandas as pd
import numpy as np
import os
from collections import defaultdict, Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_recall_fscore_support, confusion_matrix
import sklearn_crfsuite
from sklearn_crfsuite import metrics
import pickle
import warnings
warnings.filterwarnings('ignore')

# 导入评估模块
from evaluation_metrics import POSTaggingEvaluator

class EnhancedChinesePOSTagger:
    def __init__(self):
        self.crf_model = None
        self.word_vocab = set()
        self.pos_vocab = set()
        self.word_pos_dict = defaultdict(Counter)
        self.char_vocab = set()
        self.char_pos_dict = defaultdict(Counter)
        self.oov_threshold = 2  # 提高OOV阈值
        self.char_bigram_dict = defaultdict(Counter)
        self.pos_transition_dict = defaultdict(Counter)
        
    def load_training_data(self, file_path):
        """加载训练数据并提取更多统计信息"""
        print("正在加载训练数据...")
        sentences = []
        
        # 尝试多种编码格式
        encodings_to_try = ['gbk', 'gb2312', 'utf-8', 'gb18030', 'big5']
        
        file_content = None
        used_encoding = None
        
        for encoding in encodings_to_try:
            try:
                with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
                    file_content = f.readlines()
                    used_encoding = encoding
                    print(f"成功使用编码: {encoding}")
                    break
            except UnicodeDecodeError:
                continue
        
        if file_content is None:
            raise ValueError("无法识别文件编码，请检查文件格式")
        
        # 处理文件内容
        for line in file_content:
            line = line.strip()
            if not line:
                continue
                    
            words_pos = []
            tokens = line.split()
            prev_pos = None
            
            for token in tokens:
                if '/' in token:
                    parts = token.rsplit('/', 1)
                    if len(parts) == 2:
                        word, pos = parts
                        if word and pos:
                            words_pos.append((word, pos))
                            self.word_vocab.add(word)
                            self.pos_vocab.add(pos)
                            self.word_pos_dict[word][pos] += 1
                            
                            # 收集字符信息
                            for char in word:
                                self.char_vocab.add(char)
                                self.char_pos_dict[char][pos] += 1
                            
                            # 收集字符二元组信息
                            for i in range(len(word)-1):
                                bigram = word[i:i+2]
                                self.char_bigram_dict[bigram][pos] += 1
                            
                            # 收集词性转移信息
                            if prev_pos:
                                self.pos_transition_dict[prev_pos][pos] += 1
                            prev_pos = pos
            
            if words_pos:
                sentences.append(words_pos)
        
        print(f"加载完成: {len(sentences)} 个句子")
        print(f"词汇量: {len(self.word_vocab)}")
        print(f"字符数: {len(self.char_vocab)}")
        print(f"词性标签数: {len(self.pos_vocab)}")
        return sentences
    
    def is_oov_word(self, word):
        """判断是否为未登录词"""
        return word not in self.word_vocab or sum(self.word_pos_dict[word].values()) <= self.oov_threshold
    
    def get_char_features(self, word):
        """提取字符级特征"""
        features = {}
        
        # 字符频率特征
        char_in_vocab = sum(1 for char in word if char in self.char_vocab)
        features['char_in_vocab_ratio'] = char_in_vocab / len(word) if word else 0
        
        # 首尾字符特征
        if word:
            first_char = word[0]
            last_char = word[-1]
            features['first_char_common'] = sum(self.char_pos_dict[first_char].values()) > 10
            features['last_char_common'] = sum(self.char_pos_dict[last_char].values()) > 10
            
            # 最可能的词性（基于字符）
            first_char_pos = self.char_pos_dict[first_char].most_common(1)
            last_char_pos = self.char_pos_dict[last_char].most_common(1)
            
            if first_char_pos:
                features['first_char_likely_pos'] = first_char_pos[0][0]
            if last_char_pos:
                features['last_char_likely_pos'] = last_char_pos[0][0]
        
        # 字符二元组特征
        bigram_features = []
        for i in range(len(word)-1):
            bigram = word[i:i+2]
            if bigram in self.char_bigram_dict:
                most_common_pos = self.char_bigram_dict[bigram].most_common(1)
                if most_common_pos:
                    bigram_features.append(most_common_pos[0][0])
        
        if bigram_features:
            features['bigram_likely_pos'] = Counter(bigram_features).most_common(1)[0][0]
        
        return features
    
    def get_morphological_features(self, word):
        """提取形态学特征"""
        features = {}
        
        # 词长特征
        features['word_length'] = len(word)
        features['is_single_char'] = len(word) == 1
        features['is_two_char'] = len(word) == 2
        features['is_long_word'] = len(word) >= 4
        
        # 字符类型特征
        features['has_digits'] = any(c.isdigit() for c in word)
        features['has_letters'] = any(c.isalpha() for c in word)
        features['has_punct'] = any(c in '，。！？：；""''（）【】《》' for c in word)
        features['all_digits'] = word.isdigit()
        features['all_letters'] = word.isalpha()
        
        # 特殊模式
        features['has_year_pattern'] = bool(re.search(r'\d{4}年', word))
        features['has_month_pattern'] = bool(re.search(r'\d{1,2}月', word))
        features['has_day_pattern'] = bool(re.search(r'\d{1,2}日', word))
        features['has_number_pattern'] = bool(re.search(r'\d+', word))
        features['has_percentage'] = '％' in word or '%' in word
        
        # 词缀特征
        common_suffixes = ['性', '者', '员', '家', '师', '长', '部', '会', '社', '厂', '店']
        common_prefixes = ['副', '总', '主', '老', '小', '大', '新', '旧']
        
        for suffix in common_suffixes:
            if word.endswith(suffix):
                features[f'suffix_{suffix}'] = True
                
        for prefix in common_prefixes:
            if word.startswith(prefix):
                features[f'prefix_{prefix}'] = True
        
        return features
    
    def extract_enhanced_features(self, sentence, i):
        """提取增强特征"""
        word = sentence[i]
        features = {
            'word': word,
            'word.lower': word.lower(),
            'is_oov': self.is_oov_word(word),
        }
        
        # 基础特征
        features.update(self.get_morphological_features(word))
        
        # 字符级特征（特别用于OOV词）
        if self.is_oov_word(word):
            features.update(self.get_char_features(word))
        
        # 上下文特征
        context_window = 2
        for j in range(max(0, i-context_window), min(len(sentence), i+context_window+1)):
            if j == i:
                continue
            position = 'prev' if j < i else 'next'
            distance = abs(j - i)
            context_word = sentence[j]
            
            features[f'{position}_{distance}_word'] = context_word
            features[f'{position}_{distance}_is_oov'] = self.is_oov_word(context_word)
            features[f'{position}_{distance}_length'] = len(context_word)
        
        # 句子位置特征
        features['sentence_begin'] = (i == 0)
        features['sentence_end'] = (i == len(sentence) - 1)
        features['position_ratio'] = i / len(sentence) if len(sentence) > 1 else 0
        
        return features
    
    def prepare_crf_data(self, sentences):
        """准备CRF训练数据"""
        print("正在准备CRF训练数据...")
        X, y = [], []
        
        for sentence in sentences:
            words = [word for word, pos in sentence]
            pos_tags = [pos for word, pos in sentence]
            
            sentence_features = []
            for i in range(len(words)):
                features = self.extract_enhanced_features(words, i)
                sentence_features.append(features)
            
            X.append(sentence_features)
            y.append(pos_tags)
        
        return X, y
    
    def train(self, training_file):
        """训练增强CRF模型"""
        sentences = self.load_training_data(training_file)
        X, y = self.prepare_crf_data(sentences)
        
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)
        
        print("正在训练增强CRF模型...")
        # 调整CRF参数以提高OOV处理能力
        self.crf_model = sklearn_crfsuite.CRF(
            algorithm='lbfgs',
            c1=0.05,  # 降低L1正则化
            c2=0.05,  # 降低L2正则化
            max_iterations=200,  # 增加迭代次数
            all_possible_transitions=True
        )
        
        self.crf_model.fit(X_train, y_train)
        
        print("验证集上的性能:")
        y_pred = self.crf_model.predict(X_val)
        self.evaluate_model(y_val, y_pred)
        
        print("模型训练完成!")
    
    def segment_and_tag(self, text):
        """分词并标注词性"""
        words = list(jieba.cut(text))
        
        features = []
        for i in range(len(words)):
            feature = self.extract_enhanced_features(words, i)
            features.append(feature)
        
        if self.crf_model:
            pos_tags = self.crf_model.predict([features])[0]
        else:
            pos_tags = [pos for word, pos in pseg.cut(text)]
        
        return list(zip(words, pos_tags))
    
    def evaluate_model(self, y_true, y_pred):
        """评估模型性能"""
        true_labels = [label for sentence in y_true for label in sentence]
        pred_labels = [label for sentence in y_pred for label in sentence]
        
        accuracy = sum(t == p for t, p in zip(true_labels, pred_labels)) / len(true_labels)
        precision, recall, f1, support = precision_recall_fscore_support(
            true_labels, pred_labels, average='weighted', zero_division=0
        )
        
        print(f"标注准确率: {accuracy:.4f}")
        print(f"加权平均 - 精确率: {precision:.4f}, 召回率: {recall:.4f}, F1分数: {f1:.4f}")
        
        # 详细的分类报告
        print("\n各词性标签的详细评估:")
        print(classification_report(true_labels, pred_labels, zero_division=0))
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def evaluate_oov_performance(self, test_sentences, predictions):
        """专门评估OOV词的性能"""
        iv_correct = 0  # In-vocabulary correct
        iv_total = 0    # In-vocabulary total
        oov_correct = 0 # Out-of-vocabulary correct
        oov_total = 0   # Out-of-vocabulary total
        
        for test_sentence, pred_sentence in zip(test_sentences, predictions):
            # 使用jieba作为参考标准
            gold_words_pos = list(pseg.cut(test_sentence))
            
            # 对齐预测和标准答案
            min_len = min(len(gold_words_pos), len(pred_sentence))
            
            for i in range(min_len):
                gold_word, gold_pos = gold_words_pos[i]
                pred_word, pred_pos = pred_sentence[i]
                
                if self.is_oov_word(gold_word):
                    oov_total += 1
                    if gold_word == pred_word and gold_pos == pred_pos:
                        oov_correct += 1
                else:
                    iv_total += 1
                    if gold_word == pred_word and gold_pos == pred_pos:
                        iv_correct += 1
        
        iv_accuracy = iv_correct / iv_total if iv_total > 0 else 0
        oov_recall = oov_correct / oov_total if oov_total > 0 else 0
        
        print(f"\n=== OOV词性能分析 ===")
        print(f"登录词(IV)数量: {iv_total}, 准确率: {iv_accuracy:.4f}")
        print(f"未登录词(OOV)数量: {oov_total}, 召回率: {oov_recall:.4f}")
        print(f"OOV词占比: {oov_total/(iv_total+oov_total)*100:.2f}%")
        
        return {
            'iv_accuracy': iv_accuracy,
            'oov_recall': oov_recall,
            'oov_ratio': oov_total/(iv_total+oov_total) if (iv_total+oov_total) > 0 else 0
        }
    
    def analyze_feature_importance(self):
        """分析特征重要性"""
        if self.crf_model:
            print("\n=== 特征重要性分析 ===")
            try:
                # sklearn-crfsuite使用不同的方法获取特征权重
                state_features = self.crf_model.state_features_
                transition_features = self.crf_model.transition_features_
                
                print("状态特征数量:", len(state_features))
                print("转移特征数量:", len(transition_features))
                
                # 分析状态特征
                feature_importance = {}
                for i, item in enumerate(state_features):
                    if i >= 1000:  # 限制分析数量避免过慢
                        break
                    if len(item) == 3:
                        feature_name, label, weight = item
                        if feature_name not in feature_importance:
                            feature_importance[feature_name] = []
                        feature_importance[feature_name].append(abs(weight))
                
                # 计算平均权重
                avg_importance = {}
                for feature_name, weights in feature_importance.items():
                    avg_importance[feature_name] = np.mean(weights)
                
                # 显示最重要的特征
                sorted_features = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)
                print("最重要的20个状态特征:")
                for i, (feature, weight) in enumerate(sorted_features[:20]):
                    print(f"{i+1:2d}. {feature}: {weight:.4f}")
                    
            except AttributeError:
                print("当前模型不支持特征重要性分析")
                print("可能是因为sklearn-crfsuite版本问题或模型未正确训练")
            except Exception as e:
                print(f"特征重要性分析出错: {e}")
                print("跳过特征重要性分析，继续其他评估...")
    
    def load_test_data(self, file_path):
        """加载测试数据"""
        test_sentences = []
        
        # 尝试多种编码格式
        encodings_to_try = ['gbk', 'gb2312', 'utf-8', 'gb18030', 'big5']
        
        for encoding in encodings_to_try:
            try:
                with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            test_sentences.append(line)
                print(f"测试数据成功使用编码: {encoding}")
                break
            except UnicodeDecodeError:
                continue
        
        return test_sentences
    
    def comprehensive_evaluation(self, test_file):
        """综合评估"""
        print("\n" + "="*60)
        print("开始增强版模型综合评估")
        print("="*60)
        
        test_sentences = self.load_test_data(test_file)
        print(f"测试句子数: {len(test_sentences)}")
        
        # 预测
        print("正在进行预测...")
        predictions = []
        for sentence in test_sentences:
            pred_result = self.segment_and_tag(sentence)
            predictions.append(pred_result)
        
        # OOV性能评估
        oov_results = self.evaluate_oov_performance(test_sentences, predictions)
        
        # 特征重要性分析
        self.analyze_feature_importance()
        
        return oov_results
    
    def save_model(self, model_path, vocab_path):
        """保存模型和词汇表"""
        # 创建输出目录
        model_dir = os.path.dirname(model_path)
        if model_dir and not os.path.exists(model_dir):
            os.makedirs(model_dir)
            print(f"创建模型保存目录: {model_dir}")
        
        vocab_dir = os.path.dirname(vocab_path)
        if vocab_dir and not os.path.exists(vocab_dir):
            os.makedirs(vocab_dir)
            print(f"创建词汇表保存目录: {vocab_dir}")
        
        with open(model_path, 'wb') as f:
            pickle.dump(self.crf_model, f)
        
        vocab_data = {
            'word_vocab': self.word_vocab,
            'pos_vocab': self.pos_vocab,
            'word_pos_dict': dict(self.word_pos_dict),
            'char_vocab': self.char_vocab,
            'char_pos_dict': dict(self.char_pos_dict),
            'char_bigram_dict': dict(self.char_bigram_dict),
            'pos_transition_dict': dict(self.pos_transition_dict)
        }
        with open(vocab_path, 'wb') as f:
            pickle.dump(vocab_data, f)
        
        print(f"增强模型已保存到: {model_path}")
        print(f"词汇表已保存到: {vocab_path}")
    
    def load_model(self, model_path, vocab_path):
        """加载模型和词汇表"""
        with open(model_path, 'rb') as f:
            self.crf_model = pickle.load(f)
        
        with open(vocab_path, 'rb') as f:
            vocab_data = pickle.load(f)
            self.word_vocab = vocab_data['word_vocab']
            self.pos_vocab = vocab_data['pos_vocab']
            self.word_pos_dict = defaultdict(Counter, vocab_data['word_pos_dict'])
            self.char_vocab = vocab_data['char_vocab']
            self.char_pos_dict = defaultdict(Counter, vocab_data['char_pos_dict'])
            self.char_bigram_dict = defaultdict(Counter, vocab_data['char_bigram_dict'])
            self.pos_transition_dict = defaultdict(Counter, vocab_data['pos_transition_dict'])
        
        print("增强模型加载完成!")


def main():
    """主函数"""
    print("增强版汉语词法分析器 - 专门处理未登录词问题")
    print("="*50)
    
    # 创建输出目录
    output_dir = "output"
    models_dir = os.path.join(output_dir, "models")
    metrics_dir = os.path.join(output_dir, "metrics")
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    
    print(f"创建输出目录: {output_dir}")
    print(f"模型保存目录: {models_dir}")
    print(f"指标保存目录: {metrics_dir}")
    
    # 初始化增强标注器
    enhanced_tagger = EnhancedChinesePOSTagger()
    
    # 训练模型
    print("\n1. 训练增强模型")
    enhanced_tagger.train('test_pd_src/train_pd.txt')
    
    # 保存模型到指定目录
    model_path = os.path.join(models_dir, 'enhanced_crf_model.pkl')
    vocab_path = os.path.join(models_dir, 'enhanced_vocab.pkl')
    enhanced_tagger.save_model(model_path, vocab_path)
    
    # 使用完整评估系统
    print("\n2. 使用完整评估系统进行综合评估")
    evaluator = POSTaggingEvaluator(enhanced_tagger, enhanced_tagger.word_vocab)
    
    # 进行全面评估
    evaluation_results = evaluator.comprehensive_evaluation('test_pd_src/test_pd_src.txt')
    
    # 生成错误分析报告到指标目录
    print("\n3. 生成错误分析报告")
    error_analysis_path = os.path.join(metrics_dir, 'enhanced_error_analysis.txt')
    evaluator.analyze_error_patterns('test_pd_src/test_pd_src.txt', error_analysis_path)
    
    # 生成可视化图表到指标目录
    print("\n4. 生成可视化图表")
    try:
        # 修改当前工作目录到指标目录
        original_dir = os.getcwd()
        os.chdir(metrics_dir)
        evaluator.visualize_results(evaluation_results)
        os.chdir(original_dir)
        print(f"可视化图表已保存到: {metrics_dir}")
    except Exception as e:
        print(f"可视化生成失败: {e}")
        os.chdir(original_dir)  # 确保返回原目录
    
    # 保存评估结果摘要到文件
    print("\n5. 保存评估结果摘要")
    results_summary_path = os.path.join(metrics_dir, 'evaluation_summary.txt')
    with open(results_summary_path, 'w', encoding='utf-8') as f:
        f.write("增强版汉语词法分析器评估结果摘要\n")
        f.write("="*60 + "\n\n")
        f.write(f"标注准确率 (Word-level Accuracy): {evaluation_results['word_accuracy']:.4f}\n")
        f.write(f"精确率 (Precision): {evaluation_results['precision']:.4f}\n")
        f.write(f"召回率 (Recall): {evaluation_results['recall']:.4f}\n")
        f.write(f"F1分数 (F1-score): {evaluation_results['f1_score']:.4f}\n")
        f.write(f"OOV召回率 (OOV Recall): {evaluation_results['oov_recall']:.4f}\n")
        f.write(f"\nOOV词统计:\n")
        f.write(f"登录词(IV)数量: {evaluation_results['oov_metrics']['iv_total']}\n")
        f.write(f"未登录词(OOV)数量: {evaluation_results['oov_metrics']['oov_total']}\n")
        f.write(f"OOV词占比: {evaluation_results['oov_metrics']['oov_ratio']*100:.2f}%\n")
    
    print(f"评估结果摘要已保存到: {results_summary_path}")
    
    # OOV词处理演示
    print("\n6. OOV词处理演示")
    test_texts = [
        "阿尔法狗击败了世界围棋冠军。",  # 包含英文音译词
        "他在微博上发布了最新动态。",    # 包含网络词汇
        "这款智能手机的性价比很高。",    # 包含复合词
        "COVID-19疫苗接种工作正在进行。" # 包含外来词
    ]
    
    # 保存OOV演示结果
    oov_demo_path = os.path.join(metrics_dir, 'oov_demo_results.txt')
    with open(oov_demo_path, 'w', encoding='utf-8') as f:
        f.write("OOV词处理演示结果\n")
        f.write("="*50 + "\n\n")
        
        for text in test_texts:
            result = enhanced_tagger.segment_and_tag(text)
            print(f"原文: {text}")
            f.write(f"原文: {text}\n")
            
            # 标识OOV词
            tagged_result = []
            for word, pos in result:
                if enhanced_tagger.is_oov_word(word):
                    tagged_result.append(f"{word}/{pos}[OOV]")
                else:
                    tagged_result.append(f"{word}/{pos}")
            
            tagged_str = ' '.join(tagged_result)
            print(f"标注: {tagged_str}")
            f.write(f"标注: {tagged_str}\n\n")
    
    print(f"OOV演示结果已保存到: {oov_demo_path}")
    
    # 最终总结
    print("\n" + "="*60)
    print("训练和评估完成!")
    print("="*60)
    print("生成的文件:")
    print(f"模型文件目录: {models_dir}")
    print(f"  - enhanced_crf_model.pkl: 训练好的CRF模型")
    print(f"  - enhanced_vocab.pkl: 词汇表和统计信息")
    print(f"\n指标文件目录: {metrics_dir}")
    print(f"  - evaluation_summary.txt: 评估结果摘要")
    print(f"  - enhanced_error_analysis.txt: 详细错误分析报告")
    print(f"  - evaluation_results.png: 性能指标可视化图")
    print(f"  - oov_distribution.png: OOV词分布图")
    print(f"  - oov_demo_results.txt: OOV词处理演示结果")
    print("="*60)


if __name__ == "__main__":
    main()