#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
任务6：词云生成（中英文）
目的：可视化语料中词频情况，直观展示语言特色
"""

import json
import os
import re
from collections import Counter
import matplotlib.pyplot as plt

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
        from wordcloud import WordCloud
        print("✓ wordcloud 词云库已可用")
    except ImportError:
        print("正在安装 wordcloud...")
        import subprocess
        subprocess.check_call(["pip", "install", "wordcloud"])
        from wordcloud import WordCloud
    
    try:
        import matplotlib.pyplot as plt
        print("✓ matplotlib 绘图库已可用")
    except ImportError:
        print("正在安装 matplotlib...")
        import subprocess
        subprocess.check_call(["pip", "install", "matplotlib"])
        import matplotlib.pyplot as plt
    
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
    
    return jieba, WordCloud, plt, nltk, word_tokenize

def load_chinese_corpus(data_dir):
    """加载中文语料库"""
    print("\n" + "="*60)
    print("加载中文语料库（《全唐诗》）")
    print("="*60)
    
    corpus_text = ""
    
    # 加载唐诗三百首
    tang_poetry_file = os.path.join(data_dir, "Complete_Tang_Poems", "唐诗三百首.json")
    
    if os.path.exists(tang_poetry_file):
        try:
            with open(tang_poetry_file, 'r', encoding='utf-8') as f:
                poems = json.load(f)
            
            print(f"成功加载 {len(poems)} 首唐诗")
            
            for poem in poems:
                # 合并诗句
                text = ''.join(poem.get('paragraphs', []))
                if text.strip():
                    corpus_text += text + " "
                    
        except Exception as e:
            print(f"加载唐诗失败：{e}")
    
    # 尝试加载更多唐诗文件
    tang_poems_dir = os.path.join(data_dir, "Complete_Tang_Poems")
    if os.path.exists(tang_poems_dir):
        try:
            # 加载部分poet.tang文件
            import glob
            tang_files = glob.glob(os.path.join(tang_poems_dir, "poet.tang.*.json"))[:5]  # 只加载前5个文件
            
            for file_path in tang_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        poems = json.load(f)
                    
                    for poem in poems[:50]:  # 每个文件只取前50首
                        text = ''.join(poem.get('paragraphs', []))
                        if text.strip():
                            corpus_text += text + " "
                    
                    print(f"✓ 加载文件: {os.path.basename(file_path)}")
                except Exception as e:
                    print(f"✗ 跳过文件: {os.path.basename(file_path)} - {e}")
                    
        except Exception as e:
            print(f"加载额外唐诗文件失败：{e}")
    
    print(f"总计中文文本字符数：{len(corpus_text):,}")
    return corpus_text

def load_english_corpus():
    """加载英文语料库"""
    print("\n" + "="*60)
    print("准备英文语料库")
    print("="*60)
    
    # 英文文本示例（扩展版本）
    english_texts = [
        "The king and queen ruled the kingdom with wisdom and grace for many years.",
        "A man walked down the street while a woman crossed the road in the evening.",
        "The sun rises in the east and sets in the west every single day without fail.",
        "Beautiful flowers bloom in spring when nature awakens from the cold winter sleep.",
        "Students study hard to learn new knowledge and develop important skills for life.",
        "The cat sat on the comfortable mat while the dog played happily in the garden.",
        "Music and art bring joy and beauty to people's lives in wonderful ways.",
        "Technology advances rapidly in the modern digital world of computers and internet.",
        "Friends and family gather together during holidays and special celebrations.",
        "Books contain wisdom and stories that inspire and educate young readers everywhere.",
        "Mountains and rivers create magnificent natural landscapes across the country.",
        "Love and happiness are the most precious things in human life and relationships.",
        "Scientists discover new facts about the universe and nature through research.",
        "Teachers educate young minds to prepare them for the challenging future ahead.",
        "The ocean is vast and deep, full of mysterious creatures and hidden treasures.",
        "Children play games in the playground while parents watch with smiling faces.",
        "Writers create stories that transport readers to different worlds and times.",
        "Artists paint colorful pictures that capture the beauty of life and nature.",
        "Doctors help sick people recover their health and live better lives.",
        "Farmers grow food that feeds hungry people in cities and towns everywhere."
    ]
    
    corpus_text = " ".join(english_texts)
    print(f"英文文本字符数：{len(corpus_text):,}")
    return corpus_text

def get_chinese_stopwords():
    """获取中文停用词列表"""
    # 常用中文停用词
    stopwords = {
        '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你',
        '会', '着', '没有', '看', '好', '自己', '这', '那', '来', '下', '把', '他', '她', '它', '与', '为', '而', '之', '于', '以',
        '及', '或', '但', '因', '所', '从', '当', '被', '将', '已', '又', '还', '只', '更', '最', '很', '非常', '十分', '比较',
        '什么', '怎么', '哪里', '为什么', '如何', '这样', '那样', '这里', '那里', '现在', '过去', '将来', '今天', '昨天', '明天',
        '年', '月', '日', '时', '分', '秒', '个', '些', '每', '各', '另', '其', '某', '任', '全', '整', '半', '多', '少',
        '大', '小', '长', '短', '高', '低', '新', '旧', '好', '坏', '美', '丑', '真', '假', '对', '错', '正', '负', '阴', '阳',
        '之', '乎', '者', '也', '矣', '焉', '哉', '兮', '耳', '而已', '罢了', '而', '则', '且', '又', '亦', '即', '乃', '若', '如',
        '，', '。', '？', '！', '；', '：', '"', '"', ''', ''', '（', '）', '【', '】', '《', '》', '、', '…', '—', '·'
    }
    return stopwords

def get_english_stopwords():
    """获取英文停用词列表"""
    # 常用英文停用词
    stopwords = {
        'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he', 'in', 'is', 'it',
        'its', 'of', 'on', 'that', 'the', 'to', 'was', 'will', 'with', 'the', 'this', 'but', 'they',
        'have', 'had', 'what', 'said', 'each', 'which', 'she', 'do', 'how', 'their', 'if', 'up', 'out',
        'many', 'then', 'them', 'these', 'so', 'some', 'her', 'would', 'make', 'like', 'into', 'him',
        'time', 'two', 'more', 'go', 'no', 'way', 'could', 'my', 'than', 'first', 'been', 'call', 'who',
        'oil', 'sit', 'now', 'find', 'down', 'day', 'did', 'get', 'come', 'made', 'may', 'part', 'over',
        'new', 'sound', 'take', 'only', 'little', 'work', 'know', 'place', 'year', 'live', 'me', 'back',
        'give', 'most', 'very', 'after', 'thing', 'our', 'just', 'name', 'good', 'sentence', 'man', 'think',
        'say', 'great', 'where', 'help', 'through', 'much', 'before', 'line', 'right', 'too', 'mean', 'old',
        'any', 'same', 'tell', 'boy', 'follow', 'came', 'want', 'show', 'also', 'around', 'form', 'three',
        'small', 'set', 'put', 'end', 'why', 'again', 'turn', 'here', 'off', 'went', 'old', 'number', 'word',
        'but', 'not', 'what', 'all', 'were', 'when', 'we', 'there', 'can', 'said', 'each', 'which', 'do',
        'how', 'their', 'if', 'will', 'up', 'other', 'about', 'out', 'many', 'then', 'them', 'these', 'so'
    }
    return stopwords

def process_chinese_text(text, jieba):
    """处理中文文本，分词并过滤停用词"""
    print("\n处理中文文本...")
    
    # 清理文本
    text = re.sub(r'[^\u4e00-\u9fff]', ' ', text)  # 只保留中文字符
    
    # jieba分词
    words = jieba.cut(text)
    
    # 获取停用词
    stopwords = get_chinese_stopwords()
    
    # 过滤停用词和短词
    filtered_words = []
    for word in words:
        word = word.strip()
        if len(word) >= 2 and word not in stopwords and word.isalpha():
            filtered_words.append(word)
    
    # 统计词频
    word_freq = Counter(filtered_words)
    
    print(f"分词完成，有效词汇：{len(word_freq)} 个")
    print(f"最高频词汇：{word_freq.most_common(10)}")
    
    return word_freq

def process_english_text(text, word_tokenize):
    """处理英文文本，分词并过滤停用词"""
    print("\n处理英文文本...")
    
    # 清理文本
    text = re.sub(r'[^a-zA-Z\s]', ' ', text.lower())  # 只保留字母和空格
    
    # 分词
    if word_tokenize:
        words = word_tokenize(text)
    else:
        words = text.split()
    
    # 获取停用词
    stopwords = get_english_stopwords()
    
    # 过滤停用词和短词
    filtered_words = []
    for word in words:
        word = word.strip().lower()
        if len(word) >= 3 and word not in stopwords and word.isalpha():
            filtered_words.append(word)
    
    # 统计词频
    word_freq = Counter(filtered_words)
    
    print(f"分词完成，有效词汇：{len(word_freq)} 个")
    print(f"最高频词汇：{word_freq.most_common(10)}")
    
    return word_freq

def create_chinese_wordcloud(word_freq, WordCloud, plt):
    """生成中文词云图"""
    print("\n" + "="*60)
    print("生成中文词云图")
    print("="*60)
    
    # 准备词频字符串
    if not word_freq:
        print("⚠ 没有有效的中文词汇，使用示例数据")
        word_freq = Counter({
            '明月': 50, '春风': 45, '江山': 40, '美人': 35, '诗酒': 30,
            '琴瑟': 25, '山水': 20, '风月': 18, '花鸟': 15, '云雾': 12
        })
    
    # 设置中文字体路径（Windows系统）
    font_paths = [
        'C:/Windows/Fonts/simhei.ttf',      # 黑体
        'C:/Windows/Fonts/simsun.ttc',      # 宋体
        'C:/Windows/Fonts/msyh.ttc',        # 微软雅黑
        'C:/Windows/Fonts/simkai.ttf',      # 楷体
    ]
    
    font_path = None
    for path in font_paths:
        if os.path.exists(path):
            font_path = path
            break
    
    if not font_path:
        print("⚠ 未找到中文字体，将使用默认字体")
    else:
        print(f"使用字体：{font_path}")
    
    # 设置matplotlib中文字体
    if font_path:
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
        plt.rcParams['axes.unicode_minus'] = False
    
    # 创建词云对象
    wordcloud = WordCloud(
        font_path=font_path,
        width=800,
        height=600,
        background_color='white',
        max_words=100,
        colormap='viridis',
        random_state=42
    ).generate_from_frequencies(word_freq)
    
    # 绘制词云
    plt.figure(figsize=(12, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    
    # 设置中文标题
    if font_path:
        plt.title('中文词云图 - 《全唐诗》语料', fontsize=16, pad=20, fontproperties='SimHei')
    else:
        plt.title('Chinese Word Cloud - Tang Poetry Corpus', fontsize=16, pad=20)
    
    # 保存图像
    output_dir = os.path.join("result", "task6")
    os.makedirs(output_dir, exist_ok=True)
    
    chinese_wordcloud_path = os.path.join(output_dir, "chinese_wordcloud.png")
    plt.savefig(chinese_wordcloud_path, dpi=300, bbox_inches='tight')
    print(f"中文词云图已保存到：{chinese_wordcloud_path}")
    
    # 显示图像
    plt.show()
    plt.close()
    
    return chinese_wordcloud_path

def create_english_wordcloud(word_freq, WordCloud, plt):
    """生成英文词云图"""
    print("\n" + "="*60)
    print("生成英文词云图")
    print("="*60)
    
    # 准备词频字符串
    if not word_freq:
        print("⚠ 没有有效的英文词汇，使用示例数据")
        word_freq = Counter({
            'king': 20, 'queen': 18, 'beautiful': 15, 'nature': 12, 'wisdom': 10,
            'knowledge': 9, 'students': 8, 'flowers': 7, 'mountains': 6, 'happiness': 5
        })
    
    # 创建词云对象
    wordcloud = WordCloud(
        width=800,
        height=600,
        background_color='white',
        max_words=100,
        colormap='plasma',
        random_state=42
    ).generate_from_frequencies(word_freq)
    
    # 绘制词云
    plt.figure(figsize=(12, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('English Word Cloud - Text Corpus', fontsize=16, pad=20)
    
    # 保存图像
    output_dir = os.path.join("result", "task6")
    os.makedirs(output_dir, exist_ok=True)
    
    english_wordcloud_path = os.path.join(output_dir, "english_wordcloud.png")
    plt.savefig(english_wordcloud_path, dpi=300, bbox_inches='tight')
    print(f"英文词云图已保存到：{english_wordcloud_path}")
    
    # 显示图像
    plt.show()
    plt.close()
    
    return english_wordcloud_path

def create_comparison_plot(chinese_freq, english_freq, plt):
    """创建中英文词频对比图"""
    print("\n生成词频对比图...")
    
    # 设置matplotlib中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 中文词频柱状图
    if chinese_freq:
        chinese_top = chinese_freq.most_common(10)
        words, freqs = zip(*chinese_top)
        ax1.bar(range(len(words)), freqs, color='skyblue')
        ax1.set_xticks(range(len(words)))
        ax1.set_xticklabels(words, rotation=45, ha='right', fontsize=10)
        ax1.set_title('中文高频词汇 Top 10', fontsize=14)
        ax1.set_ylabel('词频', fontsize=12)
    
    # 英文词频柱状图
    if english_freq:
        english_top = english_freq.most_common(10)
        words, freqs = zip(*english_top)
        ax2.bar(range(len(words)), freqs, color='lightcoral')
        ax2.set_xticks(range(len(words)))
        ax2.set_xticklabels(words, rotation=45, ha='right', fontsize=10)
        ax2.set_title('English High Frequency Words Top 10', fontsize=14)
        ax2.set_ylabel('Frequency', fontsize=12)
    
    plt.tight_layout()
    
    # 保存对比图
    output_dir = os.path.join("result", "task6")
    comparison_path = os.path.join(output_dir, "word_frequency_comparison.png")
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    print(f"词频对比图已保存到：{comparison_path}")
    
    plt.show()
    plt.close()
    
    return comparison_path

def main():
    """主函数"""
    print("=" * 80)
    print("任务6：词云生成（中英文）")
    print("=" * 80)
    
    # 安装和导入所需包
    packages = install_and_import_packages()
    jieba, WordCloud, plt, nltk, word_tokenize = packages
    
    # 1. 加载语料库
    data_dir = "data"
    chinese_corpus = load_chinese_corpus(data_dir)
    english_corpus = load_english_corpus()
    
    # 2. 处理文本并统计词频
    chinese_word_freq = process_chinese_text(chinese_corpus, jieba)
    english_word_freq = process_english_text(english_corpus, word_tokenize)
    
    # 3. 生成词云图
    chinese_wordcloud_path = create_chinese_wordcloud(chinese_word_freq, WordCloud, plt)
    english_wordcloud_path = create_english_wordcloud(english_word_freq, WordCloud, plt)
    
    # 4. 生成对比图
    comparison_path = create_comparison_plot(chinese_word_freq, english_word_freq, plt)
    
    # 5. 总结报告
    print("\n" + "="*80)
    print("任务6完成！")
    print("="*80)
    print(f"✓ 中文词云图：{chinese_wordcloud_path}")
    print(f"✓ 英文词云图：{english_wordcloud_path}")
    print(f"✓ 词频对比图：{comparison_path}")
    print(f"\n中文语料统计：")
    print(f"  - 总词汇数：{len(chinese_word_freq)} 个")
    print(f"  - 最高频词：{chinese_word_freq.most_common(1)[0] if chinese_word_freq else '无'}")
    print(f"\n英文语料统计：")
    print(f"  - 总词汇数：{len(english_word_freq)} 个")
    print(f"  - 最高频词：{english_word_freq.most_common(1)[0] if english_word_freq else '无'}")
    print(f"\n词云生成帮助我们：")
    print("• 直观展示语料中的关键词汇")
    print("• 发现文本的主题和特色")
    print("• 比较不同语言的词汇特点")
    print("• 为进一步的文本分析提供参考")
    print("="*80)

if __name__ == "__main__":
    main() 