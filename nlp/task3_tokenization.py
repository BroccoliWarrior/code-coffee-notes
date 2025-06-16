#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
任务3：分词（中英文）
目的：掌握中文和英文的分词方法，体现语言处理差异
"""

import json
import os
import re

def install_and_import_packages():
    """检查并导入所需的包"""
    try:
        import jieba
        print("✓ jieba 中文分词库已可用")
    except ImportError:
        print("正在安装 jieba...")
        import subprocess
        subprocess.check_call(["pip", "install", "jieba"])
        import jieba
    
    try:
        import nltk
        from nltk.corpus import webtext
        from nltk.tokenize import word_tokenize
        print("✓ NLTK 英文处理库已可用")
        
        # 设置NLTK数据路径到data文件夹
        data_path = os.path.join(os.getcwd(), "data", "nltk_data")
        os.makedirs(data_path, exist_ok=True)
        nltk.data.path.insert(0, data_path)
        
        # 下载必要的数据
        try:
            webtext.fileids()
            print("✓ NLTK webtext 语料库已可用")
        except:
            print(f"正在下载 NLTK 数据到 {data_path}...")
            nltk.download('webtext', download_dir=data_path)
            nltk.download('punkt', download_dir=data_path)
        
        return jieba, nltk, webtext, word_tokenize
    except ImportError:
        print("错误：请先安装 nltk 库")
        return None, None, None, None

def load_sample_tang_poems(data_dir, num_poems=3):
    """从唐诗数据集加载样本诗歌"""
    sample_file = os.path.join(data_dir, "唐诗三百首.json")
    
    if not os.path.exists(sample_file):
        print(f"错误：找不到文件 {sample_file}")
        return []
    
    try:
        with open(sample_file, 'r', encoding='utf-8') as f:
            poems = json.load(f)
        return poems[:num_poems]
    except Exception as e:
        print(f"加载唐诗文件失败：{e}")
        return []

def chinese_tokenization_demo(jieba, poems):
    """中文分词演示"""
    print("\n" + "="*80)
    print("中文分词演示 - 使用 jieba 分词")
    print("="*80)
    
    for i, poem in enumerate(poems, 1):
        print(f"\n【示例 {i}】")
        print(f"题目：{poem.get('title', '无题')}")
        print(f"作者：{poem.get('author', '佚名')}")
        
        # 获取诗歌内容
        paragraphs = poem.get('paragraphs', [])
        full_text = ''.join(paragraphs)
        
        print(f"\n原文：")
        for line in paragraphs:
            print(f"      {line}")
        
        print(f"\n完整文本：{full_text}")
        
        # jieba 分词
        # 精确模式
        seg_precise = jieba.cut(full_text, cut_all=False)
        precise_result = list(seg_precise)
        
        # 全模式
        seg_full = jieba.cut(full_text, cut_all=True)
        full_result = list(seg_full)
        
        # 搜索引擎模式
        seg_search = jieba.cut_for_search(full_text)
        search_result = list(seg_search)
        
        print(f"\n精确模式分词：")
        print(f"      {' / '.join(precise_result)}")
        print(f"      词数：{len(precise_result)} 个")
        
        print(f"\n全模式分词：")
        print(f"      {' / '.join(full_result)}")
        print(f"      词数：{len(full_result)} 个")
        
        print(f"\n搜索引擎模式分词：")
        print(f"      {' / '.join(search_result)}")
        print(f"      词数：{len(search_result)} 个")
        
        print("-" * 60)

def english_tokenization_demo(nltk, webtext, word_tokenize):
    """英文分词演示"""
    print("\n" + "="*80)
    print("英文分词演示 - 使用 NLTK word_tokenize")
    print("="*80)
    
    # 获取webtext语料库中的文件
    file_ids = webtext.fileids()
    print(f"webtext 语料库包含的文件：{file_ids}")
    
    # 选择包含对话的文件
    demo_files = []
    for file_id in file_ids:
        if any(keyword in file_id.lower() for keyword in ['chat', 'movie', 'dialog', 'conversation']):
            demo_files.append(file_id)
    
    # 如果没找到相关文件，就使用前几个文件
    if not demo_files:
        demo_files = file_ids[:2]
    
    print(f"选择分析的文件：{demo_files}")
    
    for i, file_id in enumerate(demo_files, 1):
        print(f"\n【示例 {i}】文件：{file_id}")
        
        # 读取文本内容
        raw_text = webtext.raw(file_id)
        
        # 取前500个字符作为示例
        sample_text = raw_text[:500]
        
        # 清理文本（移除多余的空白字符）
        cleaned_text = re.sub(r'\s+', ' ', sample_text).strip()
        
        print(f"\n原文（前500字符）：")
        print(f"      {cleaned_text}")
        
        # NLTK word_tokenize 分词
        tokens = word_tokenize(cleaned_text)
        
        print(f"\n分词结果：")
        print(f"      {' | '.join(tokens)}")
        print(f"      词数：{len(tokens)} 个")
        
        # 统计词性分布
        from collections import Counter
        
        # 简单的词性分析
        words = [token for token in tokens if token.isalpha()]
        punctuation = [token for token in tokens if not token.isalnum()]
        numbers = [token for token in tokens if token.isdigit()]
        
        print(f"\n词汇分析：")
        print(f"      单词：{len(words)} 个")
        print(f"      标点：{len(punctuation)} 个")
        print(f"      数字：{len(numbers)} 个")
        
        # 显示一些示例词汇
        if words:
            print(f"      示例单词：{' | '.join(words[:10])}")
        
        print("-" * 60)

def comparison_analysis(jieba, word_tokenize):
    """中英文分词对比分析"""
    print("\n" + "="*80)
    print("中英文分词对比分析")
    print("="*80)
    
    # 中文示例
    chinese_text = "春眠不觉晓，处处闻啼鸟。夜来风雨声，花落知多少。"
    english_text = "Spring sleep not notice dawn, everywhere hear crying birds. Night come wind rain sound, flowers fall know how much."
    
    print("中文分词示例：")
    print(f"原文：{chinese_text}")
    chinese_tokens = list(jieba.cut(chinese_text))
    print(f"分词：{' / '.join(chinese_tokens)}")
    print(f"词数：{len(chinese_tokens)} 个")
    
    print(f"\n英文分词示例：")
    print(f"原文：{english_text}")
    english_tokens = word_tokenize(english_text)
    print(f"分词：{' | '.join(english_tokens)}")
    print(f"词数：{len(english_tokens)} 个")
    
    print(f"\n对比分析：")
    print(f"• 中文需要分词器（jieba）来识别词界")
    print(f"• 英文主要按空格和标点分词")
    print(f"• 中文词汇密度更高（{len(chinese_text)//len(chinese_tokens):.1f}字/词）")
    print(f"• 英文词汇较长（{len(english_text)//len(english_tokens):.1f}字符/词）")

def main():
    """主函数"""
    print("=" * 80)
    print("任务3：分词（中英文）")
    print("=" * 80)
    
    # 安装和导入所需包
    jieba, nltk, webtext, word_tokenize = install_and_import_packages()
    
    if not all([jieba, nltk, webtext, word_tokenize]):
        print("错误：无法加载必要的库")
        return
    
    # 1. 中文分词 - 使用《全唐诗》
    data_dir = "data/Complete_Tang_Poems"
    poems = load_sample_tang_poems(data_dir, 3)
    
    if poems:
        chinese_tokenization_demo(jieba, poems)
    else:
        print("无法加载唐诗数据，跳过中文分词演示")
    
    # 2. 英文分词 - 使用 NLTK webtext
    english_tokenization_demo(nltk, webtext, word_tokenize)
    
    # 3. 对比分析
    comparison_analysis(jieba, word_tokenize)
    
    print("\n" + "="*80)
    print("任务3完成！")
    print("成功演示了中英文分词的差异和特点")
    print("="*80)

if __name__ == "__main__":
    main() 