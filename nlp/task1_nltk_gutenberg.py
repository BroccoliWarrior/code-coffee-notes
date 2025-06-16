#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
任务1：读取NLTK数据集 - 古腾堡英文小说语料库
"""

import nltk
from nltk.corpus import gutenberg

def download_nltk_data():
    """下载所需的NLTK数据到data文件夹"""
    import os
    
    # 设置NLTK数据路径到data文件夹
    data_path = os.path.join(os.getcwd(), "data", "nltk_data")
    os.makedirs(data_path, exist_ok=True)
    nltk.data.path.insert(0, data_path)
    
    try:
        # 尝试访问gutenberg语料库，如果失败则下载
        gutenberg.fileids()
        print("NLTK古腾堡语料库已可用")
    except:
        print(f"正在下载NLTK古腾堡语料库到 {data_path}...")
        nltk.download('gutenberg', download_dir=data_path)
        nltk.download('punkt', download_dir=data_path)

def main():
    """主函数"""
    print("=" * 60)
    print("任务1：读取NLTK古腾堡语料库")
    print("=" * 60)
    
    # 下载必要的数据
    download_nltk_data()
    
    # 1. 查看文件名列表
    print("\n1. 古腾堡语料库文件列表：")
    print("-" * 40)
    file_list = gutenberg.fileids()
    for i, filename in enumerate(file_list, 1):
        print(f"{i:2d}. {filename}")
    
    # 2. 选择Emma小说进行分析
    target_file = 'austen-emma.txt'
    print(f"\n2. 选择分析文件：{target_file}")
    print("-" * 40)
    
    # 3. 读取文本内容
    print("\n3. 文本内容分析：")
    print("-" * 40)
    
    # 获取原始文本
    raw_text = gutenberg.raw(target_file)
    
    # 输出文本前1000个字符
    print("文本前1000个字符：")
    print("'" + "=" * 50)
    print(raw_text[:1000])
    print("=" * 50 + "'")
    
    # 4. 获取词汇和句子
    words = gutenberg.words(target_file)
    sents = gutenberg.sents(target_file)
    
    # 5. 统计信息
    print(f"\n4. 统计信息：")
    print("-" * 40)
    print(f"总词数：{len(words):,} 词")
    print(f"总句子数：{len(sents):,} 句")
    print(f"文本总字符数：{len(raw_text):,} 字符")
    
    # 6. 额外统计信息
    print(f"\n5. 额外分析：")
    print("-" * 40)
    
    # 计算平均句长
    avg_sent_length = len(words) / len(sents)
    print(f"平均句长：{avg_sent_length:.2f} 词/句")
    
    # 计算词汇多样性（类型-标记比）
    unique_words = set(word.lower() for word in words if word.isalpha())
    vocab_diversity = len(unique_words) / len(words)
    print(f"词汇多样性：{vocab_diversity:.4f}")
    print(f"独特词汇数：{len(unique_words):,} 词")
    
    # 显示前几句话
    print(f"\n6. 文本前3句话：")
    print("-" * 40)
    for i, sent in enumerate(sents[:3], 1):
        sentence_text = ' '.join(sent)
        print(f"句子{i}: {sentence_text}")
    
    print("\n" + "=" * 60)
    print("任务1完成！")
    print("=" * 60)

if __name__ == "__main__":
    main() 