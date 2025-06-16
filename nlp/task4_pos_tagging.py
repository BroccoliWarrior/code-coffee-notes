#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
任务4：词性标注（POS Tagging）
目的：学习如何为词语打上词性标签，理解词性对语言理解的重要性
"""

import json
import os
import re
from collections import Counter

def install_and_import_packages():
    """检查并导入所需的包"""
    try:
        import jieba
        import jieba.posseg as pseg
        print("✓ jieba 中文分词和词性标注库已可用")
    except ImportError:
        print("正在安装 jieba...")
        import subprocess
        subprocess.check_call(["pip", "install", "jieba"])
        import jieba
        import jieba.posseg as pseg
    
    try:
        import nltk
        from nltk.corpus import webtext
        from nltk.tokenize import word_tokenize
        from nltk import pos_tag
        print("✓ NLTK 英文处理库已可用")
        
        # 设置NLTK数据路径到data文件夹
        data_path = os.path.join(os.getcwd(), "data", "nltk_data")
        os.makedirs(data_path, exist_ok=True)
        nltk.data.path.insert(0, data_path)
        
        # 下载必要的数据
        required_data = ['webtext', 'punkt', 'averaged_perceptron_tagger']
        for data_name in required_data:
            try:
                if data_name == 'webtext':
                    webtext.fileids()
                elif data_name == 'punkt':
                    word_tokenize("test")
                elif data_name == 'averaged_perceptron_tagger':
                    pos_tag(['test'])
                print(f"✓ NLTK {data_name} 数据已可用")
            except:
                print(f"正在下载 NLTK {data_name} 到 {data_path}...")
                nltk.download(data_name, download_dir=data_path)
        
        return jieba, pseg, nltk, webtext, word_tokenize, pos_tag
    except ImportError:
        print("错误：请先安装 nltk 库")
        return None, None, None, None, None, None

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

def explain_chinese_pos_tags():
    """解释中文词性标签"""
    print("\n" + "="*60)
    print("中文词性标签说明（jieba词性标注体系）")
    print("="*60)
    
    pos_explanations = {
        'n': '名词 (noun)',
        'nr': '人名 (person name)',
        'ns': '地名 (place name)', 
        'nt': '机构名 (organization name)',
        'nz': '其他专名 (other proper noun)',
        'v': '动词 (verb)',
        'vd': '副动词 (adverb verb)',
        'vn': '名动词 (nominal verb)',
        'a': '形容词 (adjective)',
        'ad': '副形词 (adverb adjective)',
        'an': '名形词 (nominal adjective)',
        'd': '副词 (adverb)',
        'm': '数量词 (numeral)',
        'q': '量词 (quantifier)',
        'r': '代词 (pronoun)',
        'p': '介词 (preposition)',
        'c': '连词 (conjunction)',
        'u': '助词 (auxiliary)',
        'xc': '其他虚词 (other function word)',
        'w': '标点符号 (punctuation)',
        'PER': '人名 (person)',
        'LOC': '地名 (location)',
        'ORG': '机构名 (organization)',
    }
    
    for tag, explanation in pos_explanations.items():
        print(f"  {tag:3s} - {explanation}")

def explain_english_pos_tags():
    """解释英文词性标签"""
    print("\n" + "="*60)
    print("英文词性标签说明（Penn Treebank标注体系）")
    print("="*60)
    
    pos_explanations = {
        'CC': '并列连词 (coordinating conjunction)',
        'CD': '基数词 (cardinal digit)',
        'DT': '限定词 (determiner)',
        'EX': 'there存在句 (existential there)',
        'FW': '外来词 (foreign word)',
        'IN': '介词/从属连词 (preposition/subordinating conjunction)',
        'JJ': '形容词 (adjective)',
        'JJR': '形容词比较级 (adjective, comparative)',
        'JJS': '形容词最高级 (adjective, superlative)',
        'LS': '列表标记 (list marker)',
        'MD': '情态动词 (modal)',
        'NN': '名词，单数 (noun, singular)',
        'NNS': '名词，复数 (noun plural)',
        'NNP': '专有名词，单数 (proper noun, singular)',
        'NNPS': '专有名词，复数 (proper noun, plural)',
        'PDT': '前限定词 (predeterminer)',
        'POS': '所有格结尾 (possessive ending)',
        'PRP': '人称代词 (personal pronoun)',
        'PRP$': '所有格代词 (possessive pronoun)',
        'RB': '副词 (adverb)',
        'RBR': '副词比较级 (adverb, comparative)',
        'RBS': '副词最高级 (adverb, superlative)',
        'RP': '小品词 (particle)',
        'TO': 'to不定式标记 (to)',
        'UH': '感叹词 (interjection)',
        'VB': '动词原形 (verb, base form)',
        'VBD': '动词过去式 (verb, past tense)',
        'VBG': '动词现在分词 (verb, gerund/present participle)',
        'VBN': '动词过去分词 (verb, past participle)',
        'VBP': '动词现在时非第三人称 (verb, present tense)',
        'VBZ': '动词现在时第三人称单数 (verb, 3rd person singular present)',
        'WDT': '疑问限定词 (wh-determiner)',
        'WP': '疑问代词 (wh-pronoun)',
        'WP$': '疑问所有格代词 (possessive wh-pronoun)',
        'WRB': '疑问副词 (wh-adverb)',
    }
    
    common_tags = ['NN', 'VB', 'JJ', 'RB', 'DT', 'IN', 'PRP', 'CC', 'VBD', 'VBG', 'NNS']
    print("常见标签：")
    for tag in common_tags:
        if tag in pos_explanations:
            print(f"  {tag:4s} - {pos_explanations[tag]}")

def chinese_pos_tagging_demo(jieba, pseg, poems):
    """中文词性标注演示"""
    print("\n" + "="*80)
    print("中文词性标注演示 - 使用 jieba.posseg")
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
        
        # jieba 词性标注
        words_with_pos = pseg.cut(full_text)
        pos_result = [(word, flag) for word, flag in words_with_pos]
        
        print(f"\n词性标注结果：")
        for word, pos in pos_result:
            if word.strip():  # 跳过空白字符
                print(f"      {word:8s} / {pos}")
        
        # 统计词性分布
        pos_counter = Counter([pos for word, pos in pos_result if word.strip()])
        print(f"\n词性分布统计：")
        for pos, count in pos_counter.most_common():
            print(f"      {pos:3s}: {count:2d} 个")
        
        print("-" * 60)

def english_pos_tagging_demo(nltk, webtext, word_tokenize, pos_tag):
    """英文词性标注演示"""
    print("\n" + "="*80)
    print("英文词性标注演示 - 使用 NLTK pos_tag")
    print("="*80)
    
    # 备用英文文本样本
    english_samples = [
        {
            "title": "莎士比亚《哈姆雷特》经典对白",
            "text": "To be, or not to be, that is the question. Whether 'tis nobler in the mind to suffer the slings and arrows of outrageous fortune."
        },
        {
            "title": "电影对白示例", 
            "text": "I can't believe you're leaving tomorrow. We've been planning this trip for months, and now everything feels so surreal."
        },
        {
            "title": "新闻报道示例",
            "text": "The international conference concluded yesterday with representatives from 195 countries signing a landmark climate agreement."
        }
    ]
    
    # 尝试使用webtext，如果失败则使用备用文本
    demo_texts = []
    try:
        if webtext:
            file_ids = webtext.fileids()[:2]
            for file_id in file_ids:
                raw_text = webtext.raw(file_id)[:300]
                cleaned_text = re.sub(r'\s+', ' ', raw_text).strip()
                demo_texts.append({
                    "title": f"webtext: {file_id}",
                    "text": cleaned_text
                })
    except:
        print("使用备用英文文本进行演示")
    
    if not demo_texts:
        demo_texts = english_samples
    
    for i, sample in enumerate(demo_texts[:3], 1):
        print(f"\n【示例 {i}】{sample['title']}")
        
        text = sample['text']
        print(f"\n原文：")
        print(f"      {text}")
        
        # 分词
        tokens = word_tokenize(text)
        
        # 词性标注
        pos_result = pos_tag(tokens)
        
        print(f"\n词性标注结果：")
        for word, pos in pos_result:
            print(f"      {word:12s} / {pos}")
        
        # 统计词性分布
        pos_counter = Counter([pos for word, pos in pos_result])
        print(f"\n词性分布统计：")
        for pos, count in pos_counter.most_common():
            print(f"      {pos:4s}: {count:2d} 个")
        
        print("-" * 60)

def pos_tagging_comparison(jieba, pseg, word_tokenize, pos_tag):
    """中英文词性标注对比分析"""
    print("\n" + "="*80)
    print("中英文词性标注对比分析")
    print("="*80)
    
    # 对应的中英文句子
    chinese_text = "李白是唐朝著名的浪漫主义诗人，他写了许多优美的诗歌。"
    english_text = "Li Bai was a famous romantic poet of the Tang Dynasty, who wrote many beautiful poems."
    
    print("中文词性标注：")
    print(f"原文：{chinese_text}")
    chinese_pos = [(word, flag) for word, flag in pseg.cut(chinese_text)]
    print("结果：")
    for word, pos in chinese_pos:
        if word.strip():
            print(f"      {word:6s} / {pos}")
    
    print(f"\n英文词性标注：")
    print(f"原文：{english_text}")
    english_tokens = word_tokenize(english_text)
    english_pos = pos_tag(english_tokens)
    print("结果：")
    for word, pos in english_pos:
        print(f"      {word:10s} / {pos}")
    
    print(f"\n对比分析：")
    print(f"• 中文标注体系更注重语义角色（如人名nr、地名ns）")
    print(f"• 英文标注体系更注重语法功能（如时态VBD、复数NNS）")
    print(f"• 中文词性相对简化，英文词性更加细分")
    print(f"• 两种语言的标注结果都能帮助理解句子结构")

def analyze_pos_patterns(jieba, pseg):
    """分析词性模式"""
    print("\n" + "="*80)
    print("词性模式分析")
    print("="*80)
    
    # 不同类型的句子
    sentences = [
        "春花秋月何时了，往事知多少。",
        "明月几时有，把酒问青天。", 
        "海内存知己，天涯若比邻。",
        "山重水复疑无路，柳暗花明又一村。"
    ]
    
    for i, sentence in enumerate(sentences, 1):
        print(f"\n句子 {i}：{sentence}")
        pos_sequence = [flag for word, flag in pseg.cut(sentence) if word.strip()]
        print(f"词性序列：{' - '.join(pos_sequence)}")
        
        # 分析句式特点
        if 'v' in pos_sequence and 'n' in pos_sequence:
            print("      包含主谓结构")
        if pos_sequence.count('n') >= 2:
            print("      名词丰富，描述性强")
        if 'a' in pos_sequence:
            print("      包含形容词，有修饰成分")

def main():
    """主函数"""
    print("=" * 80)
    print("任务4：词性标注（POS Tagging）")
    print("=" * 80)
    
    # 安装和导入所需包
    packages = install_and_import_packages()
    if not all(packages):
        print("错误：无法加载必要的库")
        return
    
    jieba, pseg, nltk, webtext, word_tokenize, pos_tag = packages
    
    # 显示词性标签说明
    explain_chinese_pos_tags()
    explain_english_pos_tags()
    
    # 1. 中文词性标注 - 使用《全唐诗》
    data_dir = "data/Complete_Tang_Poems"
    poems = load_sample_tang_poems(data_dir, 3)
    
    if poems:
        chinese_pos_tagging_demo(jieba, pseg, poems)
    else:
        print("无法加载唐诗数据，跳过中文词性标注演示")
    
    # 2. 英文词性标注
    english_pos_tagging_demo(nltk, webtext, word_tokenize, pos_tag)
    
    # 3. 对比分析
    pos_tagging_comparison(jieba, pseg, word_tokenize, pos_tag)
    
    # 4. 词性模式分析
    analyze_pos_patterns(jieba, pseg)
    
    print("\n" + "="*80)
    print("任务4完成！")
    print("成功演示了中英文词性标注的方法和特点")
    print("词性标注是自然语言处理中的重要基础任务")
    print("="*80)

if __name__ == "__main__":
    main() 