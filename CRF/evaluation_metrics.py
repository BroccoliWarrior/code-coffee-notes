#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
汉语词法分析器评估指标计算
包含：Precision、Recall、F1-score、标注准确率、OOV召回率
"""

import jieba.posseg as pseg
import numpy as np
import json
import os
from collections import defaultdict, Counter
from sklearn.metrics import precision_recall_fscore_support, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['font.sans-serif'] = ['SimHei']  # 支持中文显示
plt.rcParams['axes.unicode_minus'] = False

class POSTaggingEvaluator:
    def __init__(self, tagger, word_vocab=None):
        """
        初始化评估器
        
        Args:
            tagger: 训练好的词性标注器
            word_vocab: 训练词汇表，用于判断OOV词
        """
        self.tagger = tagger
        self.word_vocab = word_vocab or set()
        
    def load_test_data(self, test_file):
        """加载测试数据"""
        test_sentences = []
        
        # 尝试多种编码格式
        encodings_to_try = ['gbk', 'gb2312', 'utf-8', 'gb18030', 'big5']
        
        for encoding in encodings_to_try:
            try:
                with open(test_file, 'r', encoding=encoding, errors='ignore') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            test_sentences.append(line)
                break
            except UnicodeDecodeError:
                continue
        
        return test_sentences
    
    def create_reference_standard(self, test_sentences):
        """创建参考标准（使用jieba作为参考）"""
        reference_data = []
        for sentence in test_sentences:
            words_pos = list(pseg.cut(sentence))
            reference_data.append(words_pos)
        return reference_data
    
    def align_predictions(self, reference, predictions):
        """对齐参考标准和预测结果"""
        aligned_ref = []
        aligned_pred = []
        
        for ref_sentence, pred_sentence in zip(reference, predictions):
            # 简化对齐：截断到较短长度
            min_len = min(len(ref_sentence), len(pred_sentence))
            
            ref_words = [word for word, pos in ref_sentence[:min_len]]
            ref_pos = [pos for word, pos in ref_sentence[:min_len]]
            pred_words = [word for word, pos in pred_sentence[:min_len]]
            pred_pos = [pos for word, pos in pred_sentence[:min_len]]
            
            aligned_ref.extend(ref_pos)
            aligned_pred.extend(pred_pos)
        
        return aligned_ref, aligned_pred
    
    def calculate_word_level_accuracy(self, reference, predictions):
        """计算词级别的准确率"""
        correct_words = 0
        total_words = 0
        
        for ref_sentence, pred_sentence in zip(reference, predictions):
            min_len = min(len(ref_sentence), len(pred_sentence))
            
            for i in range(min_len):
                ref_word, ref_pos = ref_sentence[i]
                pred_word, pred_pos = pred_sentence[i]
                
                total_words += 1
                if ref_word == pred_word and ref_pos == pred_pos:
                    correct_words += 1
        
        return correct_words / total_words if total_words > 0 else 0
    
    def calculate_pos_level_metrics(self, y_true, y_pred):
        """计算词性级别的精确率、召回率、F1分数"""
        # 整体加权平均
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )
        
        # 每个词性的详细指标
        precision_per_pos, recall_per_pos, f1_per_pos, support_per_pos = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        
        # 获取所有词性标签
        pos_labels = list(set(y_true + y_pred))
        
        pos_metrics = {}
        for i, pos in enumerate(sorted(pos_labels)):
            if i < len(precision_per_pos):
                pos_metrics[pos] = {
                    'precision': precision_per_pos[i],
                    'recall': recall_per_pos[i],
                    'f1': f1_per_pos[i],
                    'support': support_per_pos[i]
                }
        
        return {
            'overall': {
                'precision': precision,
                'recall': recall,
                'f1': f1
            },
            'per_pos': pos_metrics
        }
    
    def calculate_oov_recall(self, test_sentences, predictions):
        """计算OOV词召回率"""
        iv_correct = 0      # 登录词正确数
        iv_total = 0        # 登录词总数
        oov_correct = 0     # OOV词正确数
        oov_total = 0       # OOV词总数
        
        oov_detailed = {
            'correct_examples': [],
            'incorrect_examples': []
        }
        
        for sentence, pred_sentence in zip(test_sentences, predictions):
            # 使用jieba作为参考标准
            ref_words_pos = list(pseg.cut(sentence))
            
            min_len = min(len(ref_words_pos), len(pred_sentence))
            
            for i in range(min_len):
                ref_word, ref_pos = ref_words_pos[i]
                pred_word, pred_pos = pred_sentence[i]
                
                # 判断是否为OOV词
                is_oov = (hasattr(self.tagger, 'is_oov_word') and 
                         self.tagger.is_oov_word(ref_word)) or \
                        (ref_word not in self.word_vocab)
                
                if is_oov:
                    oov_total += 1
                    if ref_word == pred_word and ref_pos == pred_pos:
                        oov_correct += 1
                        oov_detailed['correct_examples'].append({
                            'word': ref_word,
                            'pos': ref_pos,
                            'context': sentence
                        })
                    else:
                        oov_detailed['incorrect_examples'].append({
                            'word': ref_word,
                            'ref_pos': ref_pos,
                            'pred_pos': pred_pos,
                            'context': sentence
                        })
                else:
                    iv_total += 1
                    if ref_word == pred_word and ref_pos == pred_pos:
                        iv_correct += 1
        
        iv_accuracy = iv_correct / iv_total if iv_total > 0 else 0
        oov_recall = oov_correct / oov_total if oov_total > 0 else 0
        
        return {
            'iv_accuracy': iv_accuracy,
            'iv_total': iv_total,
            'oov_recall': oov_recall,
            'oov_total': oov_total,
            'oov_ratio': oov_total / (iv_total + oov_total) if (iv_total + oov_total) > 0 else 0,
            'detailed': oov_detailed
        }
    
    def load_annotated_test_data(self, test_file):
        """加载已标注的测试数据（与训练数据格式相同）"""
        test_sentences = []
        
        # 尝试多种编码格式
        encodings_to_try = ['gbk', 'gb2312', 'utf-8', 'gb18030', 'big5']
        
        for encoding in encodings_to_try:
            try:
                with open(test_file, 'r', encoding=encoding, errors='ignore') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        
                        words_pos = []
                        tokens = line.split()
                        
                        for token in tokens:
                            if '/' in token:
                                parts = token.rsplit('/', 1)
                                if len(parts) == 2:
                                    word, pos = parts
                                    if word and pos:
                                        words_pos.append((word, pos))
                        
                        if words_pos:
                            test_sentences.append(words_pos)
                break
            except UnicodeDecodeError:
                continue
        
        return test_sentences
    
    def comprehensive_evaluation_with_gold_standard(self, test_file):
        """使用人工标注的黄金标准进行综合评估"""
        print("="*60)
        print("开始综合评估（使用人工标注黄金标准）")
        print("="*60)
        
        # 检查测试文件是否包含标注信息
        try:
            with open(test_file, 'r', encoding='gbk', errors='ignore') as f:
                first_line = f.readline().strip()
                if '/' not in first_line:
                    print("检测到测试文件没有标注信息，将使用jieba作为参考标准")
                    return self.comprehensive_evaluation(test_file)
        except:
            pass
        
        # 1. 加载已标注的测试数据
        print("正在加载已标注的测试数据...")
        gold_standard_data = self.load_annotated_test_data(test_file)
        print(f"测试句子数: {len(gold_standard_data)}")
        
        if not gold_standard_data:
            print("无法加载标注数据，回退到jieba参考标准评估")
            return self.comprehensive_evaluation(test_file)
        
        # 2. 生成预测结果
        print("正在生成预测结果...")
        predictions = []
        test_sentences_raw = []
        
        for gold_sentence in gold_standard_data:
            # 重构原始句子
            raw_sentence = ''.join([word for word, pos in gold_sentence])
            test_sentences_raw.append(raw_sentence)
            
            # 获取模型预测
            pred_result = self.tagger.segment_and_tag(raw_sentence)
            predictions.append(pred_result)
        
        # 3. 计算词级别准确率
        print("\n=== 1. 词级别准确率（基于黄金标准）===")
        word_accuracy = self.calculate_word_level_accuracy(gold_standard_data, predictions)
        print(f"词级别准确率: {word_accuracy:.4f}")
        
        # 4. 计算词性级别指标
        print("\n=== 2. 词性级别指标（基于黄金标准）===")
        y_true, y_pred = self.align_predictions(gold_standard_data, predictions)
        pos_metrics = self.calculate_pos_level_metrics(y_true, y_pred)
        
        print(f"整体精确率 (Precision): {pos_metrics['overall']['precision']:.4f}")
        print(f"整体召回率 (Recall): {pos_metrics['overall']['recall']:.4f}")
        print(f"整体F1分数 (F1-score): {pos_metrics['overall']['f1']:.4f}")
        
        # 5. 计算OOV召回率（基于训练词汇表）
        print("\n=== 3. OOV词性能分析（基于训练词汇表）===")
        oov_metrics = self.calculate_oov_recall_with_gold_standard(gold_standard_data, predictions)
        print(f"登录词(IV)数量: {oov_metrics['iv_total']}, 准确率: {oov_metrics['iv_accuracy']:.4f}")
        print(f"未登录词(OOV)数量: {oov_metrics['oov_total']}, 召回率: {oov_metrics['oov_recall']:.4f}")
        print(f"OOV词占比: {oov_metrics['oov_ratio']*100:.2f}%")
        
        # 6. 详细的词性分类报告
        print("\n=== 4. 详细词性分类报告 ===")
        print(classification_report(y_true, y_pred, zero_division=0))
        
        # 7. 总结所有指标
        print("\n" + "="*60)
        print("评估结果总结（基于黄金标准）")
        print("="*60)
        print(f"1. 标注准确率 (Word-level Accuracy): {word_accuracy:.4f}")
        print(f"2. 精确率 (Precision): {pos_metrics['overall']['precision']:.4f}")
        print(f"3. 召回率 (Recall): {pos_metrics['overall']['recall']:.4f}")
        print(f"4. F1分数 (F1-score): {pos_metrics['overall']['f1']:.4f}")
        print(f"5. OOV召回率 (OOV Recall): {oov_metrics['oov_recall']:.4f}")
        
        evaluation_results = {
            'word_accuracy': word_accuracy,
            'precision': pos_metrics['overall']['precision'],
            'recall': pos_metrics['overall']['recall'],
            'f1_score': pos_metrics['overall']['f1'],
            'oov_recall': oov_metrics['oov_recall'],
            'oov_metrics': oov_metrics,
            'pos_metrics': pos_metrics,
            'evaluation_method': 'gold_standard'
        }
        
        # 8. 保存详细的JSON格式评估结果
        try:
            json_results = {
                'evaluation_method': 'gold_standard',
                'summary': {
                    'word_accuracy': float(word_accuracy),
                    'precision': float(pos_metrics['overall']['precision']),
                    'recall': float(pos_metrics['overall']['recall']),
                    'f1_score': float(pos_metrics['overall']['f1']),
                    'oov_recall': float(oov_metrics['oov_recall'])
                },
                'oov_analysis': {
                    'iv_total': int(oov_metrics['iv_total']),
                    'iv_accuracy': float(oov_metrics['iv_accuracy']),
                    'oov_total': int(oov_metrics['oov_total']),
                    'oov_recall': float(oov_metrics['oov_recall']),
                    'oov_ratio': float(oov_metrics['oov_ratio'])
                },
                'pos_wise_metrics': {}
            }
            
            # 转换每个词性的指标
            for pos, metrics in pos_metrics['per_pos'].items():
                json_results['pos_wise_metrics'][pos] = {
                    'precision': float(metrics['precision']),
                    'recall': float(metrics['recall']),
                    'f1': float(metrics['f1']),
                    'support': int(metrics['support'])
                }
            
            # 保存JSON文件
            json_path = 'detailed_evaluation_results_gold_standard.json'
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_results, f, ensure_ascii=False, indent=2)
            print(f"\n详细评估结果（黄金标准）已保存到: {json_path}")
            
        except Exception as e:
            print(f"保存JSON结果时出错: {e}")
        
        return evaluation_results
    
    def calculate_oov_recall_with_gold_standard(self, gold_standard_data, predictions):
        """基于黄金标准计算OOV词召回率"""
        iv_correct = 0      # 登录词正确数
        iv_total = 0        # 登录词总数
        oov_correct = 0     # OOV词正确数
        oov_total = 0       # OOV词总数
        
        oov_detailed = {
            'correct_examples': [],
            'incorrect_examples': []
        }
        
        for gold_sentence, pred_sentence in zip(gold_standard_data, predictions):
            min_len = min(len(gold_sentence), len(pred_sentence))
            
            for i in range(min_len):
                gold_word, gold_pos = gold_sentence[i]
                pred_word, pred_pos = pred_sentence[i]
                
                # 基于训练词汇表判断是否为OOV词
                is_oov = (hasattr(self.tagger, 'is_oov_word') and 
                         self.tagger.is_oov_word(gold_word)) or \
                        (gold_word not in self.word_vocab)
                
                if is_oov:
                    oov_total += 1
                    if gold_word == pred_word and gold_pos == pred_pos:
                        oov_correct += 1
                        oov_detailed['correct_examples'].append({
                            'word': gold_word,
                            'pos': gold_pos,
                            'context': ''.join([w for w, p in gold_sentence])
                        })
                    else:
                        oov_detailed['incorrect_examples'].append({
                            'word': gold_word,
                            'gold_pos': gold_pos,
                            'pred_word': pred_word,
                            'pred_pos': pred_pos,
                            'context': ''.join([w for w, p in gold_sentence])
                        })
                else:
                    iv_total += 1
                    if gold_word == pred_word and gold_pos == pred_pos:
                        iv_correct += 1
        
        iv_accuracy = iv_correct / iv_total if iv_total > 0 else 0
        oov_recall = oov_correct / oov_total if oov_total > 0 else 0
        
        return {
            'iv_accuracy': iv_accuracy,
            'iv_total': iv_total,
            'oov_recall': oov_recall,
            'oov_total': oov_total,
            'oov_ratio': oov_total / (iv_total + oov_total) if (iv_total + oov_total) > 0 else 0,
            'detailed': oov_detailed
        }
    
    def comprehensive_evaluation(self, test_file):
        """综合评估"""
        print("="*60)
        print("开始综合评估")
        print("="*60)
        
        # 1. 加载测试数据
        test_sentences = self.load_test_data(test_file)
        print(f"测试句子数: {len(test_sentences)}")
        
        # 2. 生成预测结果
        print("正在生成预测结果...")
        predictions = []
        for sentence in test_sentences:
            pred_result = self.tagger.segment_and_tag(sentence)
            predictions.append(pred_result)
        
        # 3. 创建参考标准
        print("正在创建参考标准...")
        reference = self.create_reference_standard(test_sentences)
        
        # 4. 计算词级别准确率
        print("\n=== 1. 词级别准确率 ===")
        word_accuracy = self.calculate_word_level_accuracy(reference, predictions)
        print(f"词级别准确率: {word_accuracy:.4f}")
        
        # 5. 计算词性级别指标
        print("\n=== 2. 词性级别指标 ===")
        y_true, y_pred = self.align_predictions(reference, predictions)
        pos_metrics = self.calculate_pos_level_metrics(y_true, y_pred)
        
        print(f"整体精确率 (Precision): {pos_metrics['overall']['precision']:.4f}")
        print(f"整体召回率 (Recall): {pos_metrics['overall']['recall']:.4f}")
        print(f"整体F1分数 (F1-score): {pos_metrics['overall']['f1']:.4f}")
        
        # 6. 计算OOV召回率
        print("\n=== 3. OOV词性能分析 ===")
        oov_metrics = self.calculate_oov_recall(test_sentences, predictions)
        print(f"登录词(IV)数量: {oov_metrics['iv_total']}, 准确率: {oov_metrics['iv_accuracy']:.4f}")
        print(f"未登录词(OOV)数量: {oov_metrics['oov_total']}, 召回率: {oov_metrics['oov_recall']:.4f}")
        print(f"OOV词占比: {oov_metrics['oov_ratio']*100:.2f}%")
        
        # 7. 详细的词性分类报告
        print("\n=== 4. 详细词性分类报告 ===")
        print(classification_report(y_true, y_pred, zero_division=0))
        
        # 8. 总结所有指标
        print("\n" + "="*60)
        print("评估结果总结")
        print("="*60)
        print(f"1. 标注准确率 (Word-level Accuracy): {word_accuracy:.4f}")
        print(f"2. 精确率 (Precision): {pos_metrics['overall']['precision']:.4f}")
        print(f"3. 召回率 (Recall): {pos_metrics['overall']['recall']:.4f}")
        print(f"4. F1分数 (F1-score): {pos_metrics['overall']['f1']:.4f}")
        print(f"5. OOV召回率 (OOV Recall): {oov_metrics['oov_recall']:.4f}")
        
        evaluation_results = {
            'word_accuracy': word_accuracy,
            'precision': pos_metrics['overall']['precision'],
            'recall': pos_metrics['overall']['recall'],
            'f1_score': pos_metrics['overall']['f1'],
            'oov_recall': oov_metrics['oov_recall'],
            'oov_metrics': oov_metrics,
            'pos_metrics': pos_metrics
        }
        
        # 9. 保存详细的JSON格式评估结果
        try:
            json_results = {
                'summary': {
                    'word_accuracy': float(word_accuracy),
                    'precision': float(pos_metrics['overall']['precision']),
                    'recall': float(pos_metrics['overall']['recall']),
                    'f1_score': float(pos_metrics['overall']['f1']),
                    'oov_recall': float(oov_metrics['oov_recall'])
                },
                'oov_analysis': {
                    'iv_total': int(oov_metrics['iv_total']),
                    'iv_accuracy': float(oov_metrics['iv_accuracy']),
                    'oov_total': int(oov_metrics['oov_total']),
                    'oov_recall': float(oov_metrics['oov_recall']),
                    'oov_ratio': float(oov_metrics['oov_ratio'])
                },
                'pos_wise_metrics': {}
            }
            
            # 转换每个词性的指标
            for pos, metrics in pos_metrics['per_pos'].items():
                json_results['pos_wise_metrics'][pos] = {
                    'precision': float(metrics['precision']),
                    'recall': float(metrics['recall']),
                    'f1': float(metrics['f1']),
                    'support': int(metrics['support'])
                }
            
            # 保存JSON文件
            json_path = 'detailed_evaluation_results.json'
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_results, f, ensure_ascii=False, indent=2)
            print(f"\n详细评估结果已保存到: {json_path}")
            
        except Exception as e:
            print(f"保存JSON结果时出错: {e}")
        
        return evaluation_results
    
    def analyze_error_patterns(self, test_file, output_file='error_analysis.txt'):
        """分析错误模式"""
        test_sentences = self.load_test_data(test_file)
        predictions = []
        for sentence in test_sentences:
            pred_result = self.tagger.segment_and_tag(sentence)
            predictions.append(pred_result)
        
        reference = self.create_reference_standard(test_sentences)
        
        # 收集错误案例
        errors = []
        for sentence, ref_sentence, pred_sentence in zip(test_sentences, reference, predictions):
            min_len = min(len(ref_sentence), len(pred_sentence))
            
            for i in range(min_len):
                ref_word, ref_pos = ref_sentence[i]
                pred_word, pred_pos = pred_sentence[i]
                
                if ref_pos != pred_pos:
                    errors.append({
                        'sentence': sentence,
                        'word': ref_word,
                        'ref_pos': ref_pos,
                        'pred_pos': pred_pos,
                        'is_oov': hasattr(self.tagger, 'is_oov_word') and self.tagger.is_oov_word(ref_word)
                    })
        
        # 分析错误模式
        error_patterns = defaultdict(int)
        oov_errors = defaultdict(int)
        
        for error in errors:
            pattern = f"{error['ref_pos']} -> {error['pred_pos']}"
            error_patterns[pattern] += 1
            
            if error['is_oov']:
                oov_errors[pattern] += 1
        
        # 保存错误分析
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("词性标注错误分析报告\n")
            f.write("="*50 + "\n\n")
            
            f.write("最常见的错误模式 (前20个):\n")
            f.write("-"*30 + "\n")
            for pattern, count in sorted(error_patterns.items(), key=lambda x: x[1], reverse=True)[:20]:
                f.write(f"{pattern}: {count}次\n")
            
            f.write("\nOOV词错误模式 (前10个):\n")
            f.write("-"*30 + "\n")
            for pattern, count in sorted(oov_errors.items(), key=lambda x: x[1], reverse=True)[:10]:
                f.write(f"{pattern}: {count}次\n")
            
            f.write("\n错误示例 (前20个):\n")
            f.write("-"*30 + "\n")
            for i, error in enumerate(errors[:20]):
                f.write(f"{i+1}. 句子: {error['sentence']}\n")
                f.write(f"   词: {error['word']} | 参考: {error['ref_pos']} | 预测: {error['pred_pos']} | OOV: {error['is_oov']}\n\n")
        
        print(f"错误分析已保存到: {output_file}")
    
    def visualize_results(self, evaluation_results):
        """可视化评估结果"""
        try:
            # 创建指标对比图
            metrics = ['准确率', '精确率', '召回率', 'F1分数', 'OOV召回率']
            values = [
                evaluation_results['word_accuracy'],
                evaluation_results['precision'],
                evaluation_results['recall'],
                evaluation_results['f1_score'],
                evaluation_results['oov_recall']
            ]
            
            plt.figure(figsize=(10, 6))
            bars = plt.bar(metrics, values, color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E'])
            plt.title('汉语词法分析器性能指标', fontsize=16, fontweight='bold')
            plt.ylabel('分数', fontsize=12)
            plt.ylim(0, 1)
            
            # 在柱状图上添加数值
            for bar, value in zip(bars, values):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                        f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
            
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig('evaluation_results.png', dpi=300, bbox_inches='tight')
            print("性能指标图已保存为: evaluation_results.png")
            plt.show()
            
            # OOV词分析饼图
            oov_metrics = evaluation_results['oov_metrics']
            iv_count = oov_metrics['iv_total']
            oov_count = oov_metrics['oov_total']
            
            if iv_count > 0 or oov_count > 0:
                plt.figure(figsize=(8, 6))
                labels = ['登录词 (IV)', 'OOV词']
                sizes = [iv_count, oov_count]
                colors = ['#87CEEB', '#FFA07A']
                
                plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
                plt.title('词汇分布: 登录词 vs OOV词', fontsize=14, fontweight='bold')
                plt.axis('equal')
                plt.tight_layout()
                plt.savefig('oov_distribution.png', dpi=300, bbox_inches='tight')
                print("词汇分布图已保存为: oov_distribution.png")
                plt.show()
            else:
                print("没有足够的数据生成OOV分布图")
                
        except Exception as e:
            print(f"可视化出错: {e}")
            print("跳过图表生成，可能是matplotlib配置问题")


def main():
    """主函数：演示如何使用评估器"""
    # 这里假设您已经有了训练好的模型
    # 实际使用时，请先运行 chinese_pos_tagger.py 或 enhanced_pos_tagger.py
    
    print("汉语词法分析器评估系统")
    print("="*40)
    print("请先运行训练脚本生成模型，然后使用此评估系统")
    print("\n使用步骤:")
    print("1. 运行 python chinese_pos_tagger.py")
    print("2. 运行 python enhanced_pos_tagger.py")
    print("3. 使用生成的模型进行评估")


if __name__ == "__main__":
    main() 