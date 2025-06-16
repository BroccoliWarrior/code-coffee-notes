#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
任务5：词语相似度（中英文）
目的：理解词向量的意义，通过相似度评估词语关系
"""

import json
import os
import re
from collections import Counter
import numpy as np

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
        import gensim
        from gensim.models import Word2Vec
        print("✓ gensim 词向量库已可用")
    except ImportError:
        print("正在安装 gensim...")
        import subprocess
        subprocess.check_call(["pip", "install", "gensim"])
        import gensim
        from gensim.models import Word2Vec
    
    try:
        import nltk
        from nltk.tokenize import word_tokenize
        print("✓ NLTK 已可用")
        
        # 设置NLTK数据路径
        data_path = os.path.join(os.getcwd(), "data", "nltk_data")
        os.makedirs(data_path, exist_ok=True)
        nltk.data.path.insert(0, data_path)
        
        # 检查punkt是否可用
        try:
            word_tokenize("test")
        except:
            print(f"下载punkt到 {data_path}...")
            nltk.download('punkt', download_dir=data_path)
            
    except ImportError:
        print("⚠ NLTK 不可用，将使用简单分词")
        nltk = None
        word_tokenize = None
    
    return jieba, gensim, nltk, word_tokenize

def load_chinese_corpus(data_dir):
    """加载中文语料库"""
    print("\n" + "="*60)
    print("加载中文语料库")
    print("="*60)
    
    # 尝试加载唐诗数据
    tang_poetry_file = os.path.join(data_dir, "Complete_Tang_Poems", "唐诗三百首.json")
    
    corpus = []
    
    if os.path.exists(tang_poetry_file):
        try:
            with open(tang_poetry_file, 'r', encoding='utf-8') as f:
                poems = json.load(f)
            
            print(f"成功加载 {len(poems)} 首唐诗")
            
            for poem in poems:
                # 合并诗句
                text = ''.join(poem.get('paragraphs', []))
                if text.strip():
                    corpus.append(text)
                    
        except Exception as e:
            print(f"加载唐诗失败：{e}")
    
    # 添加更多中文文本示例
    additional_texts = [
        "春花秋月何时了，往事知多少。小楼昨夜又东风，故国不堪回首月明中。",
        "明月几时有，把酒问青天。不知天上宫阙，今夕是何年。",
        "山重水复疑无路，柳暗花明又一村。箫鼓追随春社近，衣冠简朴古风存。",
        "海内存知己，天涯若比邻。无为在歧路，儿女共沾巾。",
        "白日依山尽，黄河入海流。欲穷千里目，更上一层楼。",
        "床前明月光，疑是地上霜。举头望明月，低头思故乡。",
        "春眠不觉晓，处处闻啼鸟。夜来风雨声，花落知多少。",
        "山水如画，风月无边。诗酒年华，琴瑟和鸣。",
        "江山如此多娇，引无数英雄竞折腰。惜秦皇汉武，略输文采。",
        "大江东去，浪淘尽，千古风流人物。故垒西边，人道是，三国周郎赤壁。"
    ]
    
    corpus.extend(additional_texts)
    print(f"总计中文文本：{len(corpus)} 篇")
    
    return corpus

def load_english_corpus():
    """加载英文语料库"""
    print("\n" + "="*60)
    print("准备英文语料库")
    print("="*60)
    
    # 英文文本示例
    english_texts = [
        "The king and queen ruled the kingdom with wisdom and grace.",
        "A man walked down the street while a woman crossed the road.",
        "The sun rises in the east and sets in the west every day.",
        "Beautiful flowers bloom in spring when nature awakens from winter.",
        "Students study hard to learn new knowledge and skills.",
        "The cat sat on the mat while the dog played in the garden.",
        "Music and art bring joy and beauty to people's lives.",
        "Technology advances rapidly in the modern digital world.",
        "Friends and family gather together during holidays and celebrations.",
        "Books contain wisdom and stories that inspire and educate readers.",
        "Mountains and rivers create magnificent natural landscapes.",
        "Love and happiness are the most precious things in life.",
        "Scientists discover new facts about the universe and nature.",
        "Teachers educate young minds to prepare them for the future.",
        "The ocean is vast and deep, full of mysterious creatures."
    ]
    
    print(f"英文文本：{len(english_texts)} 句")
    return english_texts

def preprocess_chinese_texts(corpus, jieba):
    """预处理中文文本"""
    print("\n处理中文文本...")
    
    processed_sentences = []
    total_words = 0
    
    for text in corpus:
        # 使用jieba分词
        words = [word.strip() for word in jieba.cut(text) if word.strip() and len(word.strip()) > 1]
        
        # 过滤掉标点符号和单字符
        words = [word for word in words if word.isalnum() and len(word) > 1]
        
        if len(words) > 2:  # 至少需要3个词
            processed_sentences.append(words)
            total_words += len(words)
    
    print(f"处理完成：{len(processed_sentences)} 个句子，{total_words} 个词")
    return processed_sentences

def preprocess_english_texts(corpus, word_tokenize):
    """预处理英文文本"""
    print("\n处理英文文本...")
    
    processed_sentences = []
    total_words = 0
    
    for text in corpus:
        # 分词
        if word_tokenize:
            words = word_tokenize(text.lower())
        else:
            # 简单分词
            words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        
        # 过滤短词和停用词
        words = [word for word in words if len(word) > 2 and word.isalpha()]
        
        if len(words) > 2:
            processed_sentences.append(words)
            total_words += len(words)
    
    print(f"处理完成：{len(processed_sentences)} 个句子，{total_words} 个词")
    return processed_sentences

def train_chinese_word2vec(sentences):
    """训练中文Word2Vec模型"""
    print("\n" + "="*60)
    print("训练中文Word2Vec模型")
    print("="*60)
    
    # 导入Word2Vec
    from gensim.models import Word2Vec
    
    # 训练参数
    vector_size = 100
    window = 5
    min_count = 2
    epochs = 10
    
    print(f"训练参数：")
    print(f"  向量维度: {vector_size}")
    print(f"  窗口大小: {window}")
    print(f"  最小词频: {min_count}")
    print(f"  训练轮数: {epochs}")
    
    # 训练模型
    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        epochs=epochs,
        sg=0  # 使用CBOW算法
    )
    
    print(f"\n模型训练完成！")
    print(f"词汇表大小：{len(model.wv)} 个词")
    
    # 保存模型到models/task5文件夹
    model_dir = os.path.join("models", "task5")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "chinese_word2vec.model")
    model.save(model_path)
    print(f"模型已保存到：{model_path}")
    
    return model

def train_english_word2vec(sentences):
    """训练英文Word2Vec模型"""
    print("\n" + "="*60)
    print("训练英文Word2Vec模型")
    print("="*60)
    
    # 导入Word2Vec
    from gensim.models import Word2Vec
    
    # 训练参数
    vector_size = 100
    window = 5
    min_count = 1  # 英文语料较少，降低最小词频
    epochs = 20    # 增加训练轮数
    
    print(f"训练参数：")
    print(f"  向量维度: {vector_size}")
    print(f"  窗口大小: {window}")
    print(f"  最小词频: {min_count}")
    print(f"  训练轮数: {epochs}")
    
    # 训练模型
    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        epochs=epochs,
        sg=0  # 使用CBOW算法
    )
    
    print(f"\n模型训练完成！")
    print(f"词汇表大小：{len(model.wv)} 个词")
    
    # 保存模型到models/task5文件夹
    model_dir = os.path.join("models", "task5")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "english_word2vec.model")
    model.save(model_path)
    print(f"模型已保存到：{model_path}")
    
    return model

def demonstrate_chinese_similarity(model):
    """演示中文词语相似度"""
    print("\n" + "="*80)
    print("中文词语相似度演示")
    print("="*80)
    
    # 检查模型词汇
    vocab = list(model.wv.key_to_index.keys())
    print(f"模型词汇表（前20个）：{vocab[:20]}")
    
    # 预定义的词对
    word_pairs = [
        ("山水", "风月"),
        ("春花", "秋月"),
        ("明月", "星辰"),
        ("江山", "天下"),
        ("诗酒", "琴瑟"),
        ("故乡", "家园"),
        ("青春", "年华"),
        ("美人", "佳人"),
        ("朋友", "知己"),
        ("学习", "知识")
    ]
    
    print(f"\n中文词语相似度计算：")
    print("-" * 60)
    
    valid_pairs = 0
    
    for word1, word2 in word_pairs:
        try:
            if word1 in model.wv and word2 in model.wv:
                similarity = model.wv.similarity(word1, word2)
                print(f"'{word1}' 与 '{word2}' 的相似度：{similarity:.4f}")
                valid_pairs += 1
            else:
                missing = []
                if word1 not in model.wv:
                    missing.append(word1)
                if word2 not in model.wv:
                    missing.append(word2)
                print(f"'{word1}' 与 '{word2}' - 词汇不在模型中：{missing}")
        except Exception as e:
            print(f"'{word1}' 与 '{word2}' - 计算错误：{e}")
    
    print(f"\n成功计算 {valid_pairs} 对词语的相似度")
    
    # 寻找相似词
    if vocab:
        print(f"\n寻找与特定词最相似的词：")
        test_words = ["明月", "春花", "山水", "诗人", "故乡"]
        
        for test_word in test_words:
            if test_word in model.wv:
                try:
                    similar_words = model.wv.most_similar(test_word, topn=3)
                    print(f"与 '{test_word}' 最相似的词：")
                    for word, score in similar_words:
                        print(f"  {word}: {score:.4f}")
                except Exception as e:
                    print(f"  {test_word}: 无法找到相似词 - {e}")
            else:
                print(f"  '{test_word}' 不在词汇表中")

def demonstrate_english_similarity(model):
    """演示英文词语相似度"""
    print("\n" + "="*80)
    print("英文词语相似度演示")
    print("="*80)
    
    # 检查模型词汇
    vocab = list(model.wv.key_to_index.keys())
    print(f"模型词汇表（前20个）：{vocab[:20]}")
    
    # 预定义的词对
    word_pairs = [
        ("king", "queen"),
        ("man", "woman"),
        ("sun", "moon"),
        ("beautiful", "lovely"),
        ("study", "learn"),
        ("friend", "companion"),
        ("music", "song"),
        ("mountain", "hill"),
        ("love", "happiness"),
        ("book", "knowledge")
    ]
    
    print(f"\n英文词语相似度计算：")
    print("-" * 60)
    
    valid_pairs = 0
    
    for word1, word2 in word_pairs:
        try:
            if word1 in model.wv and word2 in model.wv:
                similarity = model.wv.similarity(word1, word2)
                print(f"'{word1}' 与 '{word2}' 的相似度：{similarity:.4f}")
                valid_pairs += 1
            else:
                missing = []
                if word1 not in model.wv:
                    missing.append(word1)
                if word2 not in model.wv:
                    missing.append(word2)
                print(f"'{word1}' 与 '{word2}' - 词汇不在模型中：{missing}")
        except Exception as e:
            print(f"'{word1}' 与 '{word2}' - 计算错误：{e}")
    
    print(f"\n成功计算 {valid_pairs} 对词语的相似度")
    
    # 寻找相似词
    if vocab:
        print(f"\n寻找与特定词最相似的词：")
        test_words = ["king", "beautiful", "study", "friend", "love"]
        
        for test_word in test_words:
            if test_word in model.wv:
                try:
                    similar_words = model.wv.most_similar(test_word, topn=3)
                    print(f"与 '{test_word}' 最相似的词：")
                    for word, score in similar_words:
                        print(f"  {word}: {score:.4f}")
                except Exception as e:
                    print(f"  {test_word}: 无法找到相似词 - {e}")
            else:
                print(f"  '{test_word}' 不在词汇表中")

def load_existing_models():
    """加载已存在的模型"""
    model_dir = os.path.join("models", "task5")
    chinese_model_path = os.path.join(model_dir, "chinese_word2vec.model")
    english_model_path = os.path.join(model_dir, "english_word2vec.model")
    
    chinese_model = None
    english_model = None
    
    # 尝试加载中文模型
    if os.path.exists(chinese_model_path):
        try:
            from gensim.models import Word2Vec
            chinese_model = Word2Vec.load(chinese_model_path)
            print(f"✓ 加载中文Word2Vec模型：{chinese_model_path}")
            print(f"  词汇表大小：{len(chinese_model.wv)} 个词")
        except Exception as e:
            print(f"❌ 加载中文模型失败：{e}")
    
    # 尝试加载英文模型
    if os.path.exists(english_model_path):
        try:
            from gensim.models import Word2Vec
            english_model = Word2Vec.load(english_model_path)
            print(f"✓ 加载英文Word2Vec模型：{english_model_path}")
            print(f"  词汇表大小：{len(english_model.wv)} 个词")
        except Exception as e:
            print(f"❌ 加载英文模型失败：{e}")
    
    return chinese_model, english_model

def word_analogy_demo(chinese_model, english_model):
    """词语类比演示"""
    print("\n" + "="*80)
    print("词语类比演示（词向量运算）")
    print("="*80)
    
    print("中文词语类比：")
    if chinese_model:
        chinese_analogies = [
            ("明月", "夜晚", "太阳"),  # 明月-夜晚+太阳=?
            ("春天", "花朵", "秋天"),  # 春天-花朵+秋天=?
        ]
        
        for word1, word2, word3 in chinese_analogies:
            try:
                if all(w in chinese_model.wv for w in [word1, word2, word3]):
                    result = chinese_model.wv.most_similar(
                        positive=[word3, word1], 
                        negative=[word2], 
                        topn=1
                    )
                    if result:
                        answer, score = result[0]
                        print(f"  {word1} - {word2} + {word3} = {answer} (相似度: {score:.4f})")
                else:
                    print(f"  {word1} - {word2} + {word3} = 词汇不完整")
            except Exception as e:
                print(f"  {word1} - {word2} + {word3} = 计算失败: {e}")
    
    print("\n英文词语类比：")
    if english_model:
        english_analogies = [
            ("king", "man", "queen"),    # king-man+queen=?
            ("study", "student", "teach"), # study-student+teach=?
        ]
        
        for word1, word2, word3 in english_analogies:
            try:
                if all(w in english_model.wv for w in [word1, word2, word3]):
                    result = english_model.wv.most_similar(
                        positive=[word3, word1], 
                        negative=[word2], 
                        topn=1
                    )
                    if result:
                        answer, score = result[0]
                        print(f"  {word1} - {word2} + {word3} = {answer} (相似度: {score:.4f})")
                else:
                    print(f"  {word1} - {word2} + {word3} = 词汇不完整")
            except Exception as e:
                print(f"  {word1} - {word2} + {word3} = 计算失败: {e}")

def main():
    """主函数"""
    print("=" * 80)
    print("任务5：词语相似度（中英文）")
    print("=" * 80)
    
    # 安装和导入所需包
    packages = install_and_import_packages()
    jieba, gensim, nltk, word_tokenize = packages
    
    # 1. 尝试加载已存在的模型
    chinese_model, english_model = load_existing_models()
    
    need_training = False
    
    if chinese_model is None or english_model is None:
        print("\n部分或全部模型不存在，开始训练...")
        need_training = True
        
        # 2. 加载语料库
        data_dir = "data"
        chinese_corpus = load_chinese_corpus(data_dir)
        english_corpus = load_english_corpus()
        
        # 3. 预处理文本
        chinese_sentences = preprocess_chinese_texts(chinese_corpus, jieba)
        english_sentences = preprocess_english_texts(english_corpus, word_tokenize)
        
        # 4. 训练词向量模型
        if chinese_model is None and chinese_sentences:
            print("\n训练中文Word2Vec模型...")
            chinese_model = train_chinese_word2vec(chinese_sentences)
        
        if english_model is None and english_sentences:
            print("\n训练英文Word2Vec模型...")
            english_model = train_english_word2vec(english_sentences)
    else:
        print("\n✓ 使用已训练的模型，跳过训练阶段")
    
    # 5. 演示词语相似度
    if chinese_model:
        demonstrate_chinese_similarity(chinese_model)
    
    if english_model:
        demonstrate_english_similarity(english_model)
    
    # 6. 词语类比演示
    word_analogy_demo(chinese_model, english_model)
    
    print("\n" + "="*80)
    print("任务5完成！")
    print("词向量技术帮助我们：")
    print("• 量化词语之间的语义关系")
    print("• 发现词语的潜在语义模式")
    print("• 进行词语类比和推理")
    print("• 为后续NLP任务提供特征表示")
    print("="*80)

if __name__ == "__main__":
    main() 