#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
任务7：命名实体识别（NER）
目的：提取文本中出现的人名、地名、组织名等实体
"""

import json
import os
import re
from collections import Counter, defaultdict
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
        import nltk
        from nltk.tokenize import word_tokenize
        from nltk import pos_tag, ne_chunk
        from nltk.tree import Tree
        print("✓ NLTK 已可用")
        
        # 设置NLTK数据路径
        data_path = os.path.join(os.getcwd(), "data", "nltk_data")
        os.makedirs(data_path, exist_ok=True)
        nltk.data.path.insert(0, data_path)
        
        # 检查必要的数据
        required_data = ['punkt', 'averaged_perceptron_tagger', 'maxent_ne_chunker', 'words']
        for data_name in required_data:
            try:
                if data_name == 'punkt':
                    word_tokenize("test")
                elif data_name == 'averaged_perceptron_tagger':
                    pos_tag(['test'])
                elif data_name == 'maxent_ne_chunker':
                    ne_chunk([('test', 'NN')])
                elif data_name == 'words':
                    from nltk.corpus import words
                    words.words()[:10]
                print(f"✓ NLTK {data_name} 数据已可用")
            except:
                print(f"下载 NLTK {data_name} 到 {data_path}...")
                nltk.download(data_name, download_dir=data_path)
            
    except ImportError:
        print("⚠ NLTK 不可用，将使用简单方法")
        nltk = None
        word_tokenize = None
        pos_tag = None
        ne_chunk = None
        Tree = None
    
    return jieba, nltk, word_tokenize, pos_tag, ne_chunk, Tree

def get_chinese_entity_database():
    """获取中文实体数据库"""
    # 唐代著名人物
    persons = {
        # 唐代皇帝
        '李世民', '唐太宗', '李隆基', '唐玄宗', '李治', '唐高宗', '武则天', '李显', '李旦',
        # 著名诗人
        '李白', '杜甫', '白居易', '王维', '李商隐', '杜牧', '韩愈', '柳宗元', '刘禹锡', '元稹',
        '王昌龄', '王之涣', '孟浩然', '岑参', '高适', '王勃', '杨炯', '卢照邻', '骆宾王',
        '陈子昂', '宋之问', '沈佺期', '张九龄', '王翰', '李颀', '綦毋潜', '常建', '祖咏',
        '储光羲', '王湾', '李华', '刘长卿', '钱起', '郎士元', '韦应物', '戴叔伦', '卢纶',
        '李益', '司空曙', '刘方平', '顾况', '李端', '皇甫冉', '张籍', '王建', '韩翃',
        '李德裕', '温庭筠', '李珣', '韦庄', '皮日休', '陆龟蒙', '聂夷中', '杜荀鹤',
        # 历史人物
        '诸葛亮', '孔子', '庄子', '屈原', '司马迁', '曹操', '刘备', '关羽', '张飞',
        '项羽', '刘邦', '韩信', '萧何', '张良', '范蠡', '西施', '貂蝉', '王昭君', '杨贵妃'
    }
    
    # 地名
    places = {
        # 古代地名
        '长安', '洛阳', '金陵', '建康', '成都', '扬州', '苏州', '杭州', '开封', '大梁',
        '咸阳', '邯郸', '临安', '广陵', '江陵', '襄阳', '荆州', '益州', '幽州', '并州',
        '凉州', '安西', '河西', '陇右', '剑南', '淮南', '江南', '岭南', '关中', '关东',
        # 山川河流
        '泰山', '华山', '嵩山', '恒山', '衡山', '终南山', '峨眉山', '庐山', '黄山',
        '黄河', '长江', '淮河', '汉水', '湘江', '漓江', '洞庭湖', '鄱阳湖', '太湖',
        '渭水', '洛水', '汾水', '易水', '燕山', '阴山', '昆仑', '天山', '祁连山',
        # 现代地名对应
        '西安', '北京', '南京', '上海', '广州', '深圳', '武汉', '重庆', '天津',
        '中国', '华夏', '神州', '九州', '中华', '东土', '西域', '塞外', '江东', '江西'
    }
    
    # 朝代国名
    dynasties = {
        '夏', '商', '周', '春秋', '战国', '秦', '汉', '三国', '晋', '南北朝',
        '隋', '唐', '五代', '宋', '元', '明', '清', '魏', '蜀', '吴',
        '前汉', '后汉', '东汉', '西汉', '前秦', '后秦', '北魏', '南朝', '北朝'
    }
    
    # 官职称谓
    titles = {
        '皇帝', '天子', '圣上', '陛下', '殿下', '王爷', '公主', '太子', '皇后', '贵妃',
        '丞相', '宰相', '尚书', '侍郎', '刺史', '太守', '县令', '将军', '都督', '节度使',
        '翰林', '学士', '进士', '举人', '秀才', '博士', '祭酒', '司马', '司空', '司徒'
    }
    
    return {
        'PERSON': persons,
        'LOCATION': places,
        'ORGANIZATION': dynasties,
        'TITLE': titles
    }

def load_chinese_corpus(data_dir):
    """加载中文语料库"""
    print("\n" + "="*60)
    print("加载中文语料库（《全唐诗》）")
    print("="*60)
    
    poems = []
    
    # 加载唐诗三百首
    tang_poetry_file = os.path.join(data_dir, "Complete_Tang_Poems", "唐诗三百首.json")
    
    if os.path.exists(tang_poetry_file):
        try:
            with open(tang_poetry_file, 'r', encoding='utf-8') as f:
                poems_data = json.load(f)
            
            print(f"成功加载 {len(poems_data)} 首唐诗")
            poems.extend(poems_data[:50])  # 只取前50首进行演示
                    
        except Exception as e:
            print(f"加载唐诗失败：{e}")
    
    # 如果没有诗歌数据，使用内置示例
    if not poems:
        poems = [
            {
                "title": "静夜思",
                "author": "李白",
                "paragraphs": ["床前明月光，疑是地上霜。", "举头望明月，低头思故乡。"]
            },
            {
                "title": "春晓",
                "author": "孟浩然", 
                "paragraphs": ["春眠不觉晓，处处闻啼鸟。", "夜来风雨声，花落知多少。"]
            },
            {
                "title": "登鹳雀楼",
                "author": "王之涣",
                "paragraphs": ["白日依山尽，黄河入海流。", "欲穷千里目，更上一层楼。"]
            }
        ]
        print("使用内置诗歌示例")
    
    return poems

def chinese_ner_rule_based(text, entity_db):
    """基于规则的中文命名实体识别"""
    entities = []
    
    for entity_type, entity_set in entity_db.items():
        for entity in entity_set:
            # 查找实体在文本中的位置
            start = 0
            while True:
                pos = text.find(entity, start)
                if pos == -1:
                    break
                entities.append({
                    'text': entity,
                    'label': entity_type,
                    'start': pos,
                    'end': pos + len(entity)
                })
                start = pos + 1
    
    # 按位置排序并去重
    entities = sorted(entities, key=lambda x: x['start'])
    
    # 去除重叠的实体（保留较长的）
    filtered_entities = []
    for entity in entities:
        overlap = False
        for existing in filtered_entities:
            if (entity['start'] < existing['end'] and entity['end'] > existing['start']):
                if len(entity['text']) <= len(existing['text']):
                    overlap = True
                    break
        if not overlap:
            filtered_entities.append(entity)
    
    return filtered_entities

def analyze_chinese_poetry_ner(poems, entity_db):
    """分析中文诗歌中的命名实体"""
    print("\n" + "="*60)
    print("中文命名实体识别 - 《全唐诗》")
    print("="*60)
    
    all_entities = defaultdict(list)
    entity_stats = defaultdict(Counter)
    
    for i, poem in enumerate(poems[:10], 1):  # 只分析前10首
        title = poem.get('title', '无题')
        author = poem.get('author', '佚名')
        paragraphs = poem.get('paragraphs', [])
        
        print(f"\n【诗歌 {i}】")
        print(f"题目：{title}")
        print(f"作者：{author}")
        
        # 合并诗句
        full_text = ''.join(paragraphs)
        print(f"原文：{full_text}")
        
        # 识别实体
        entities = chinese_ner_rule_based(full_text, entity_db)
        
        if entities:
            print(f"识别到的实体：")
            for entity in entities:
                print(f"  {entity['text']} - {entity['label']}")
                all_entities[entity['label']].append(entity['text'])
                entity_stats[entity['label']][entity['text']] += 1
        else:
            print("  未识别到实体")
        
        # 作者也是一个实体
        if author in entity_db['PERSON']:
            print(f"  {author} - PERSON (作者)")
            all_entities['PERSON'].append(author)
            entity_stats['PERSON'][author] += 1
        
        print("-" * 40)
    
    # 统计总结
    print(f"\n实体识别统计总结：")
    for entity_type, entities in entity_stats.items():
        print(f"\n{entity_type} ({len(entities)} 种)：")
        for entity, count in entities.most_common(10):
            print(f"  {entity}: {count} 次")
    
    return all_entities, entity_stats

def load_english_corpus():
    """加载英文语料库"""
    print("\n" + "="*60)
    print("准备英文语料库")
    print("="*60)
    
    # 包含命名实体的英文文本示例
    english_texts = [
        "Barack Obama was born in Hawaii and served as President of the United States.",
        "Apple Inc. is headquartered in Cupertino, California, near San Francisco.",
        "William Shakespeare was born in Stratford-upon-Avon, England in 1564.",
        "The Amazon River flows through South America, primarily through Brazil.",
        "Microsoft Corporation was founded by Bill Gates and Paul Allen in 1975.",
        "The Great Wall of China extends across northern China for thousands of miles.",
        "Harvard University is located in Cambridge, Massachusetts, near Boston.",
        "The Beatles were a famous rock band from Liverpool, England.",
        "Mount Everest is located on the border between Nepal and Tibet.",
        "The United Nations headquarters is located in New York City."
    ]
    
    print(f"英文文本：{len(english_texts)} 句")
    return english_texts

def english_ner_nltk(texts, word_tokenize, pos_tag, ne_chunk, Tree):
    """使用NLTK进行英文命名实体识别"""
    print("\n" + "="*60)
    print("英文命名实体识别 - 使用NLTK")
    print("="*60)
    
    all_entities = defaultdict(list)
    entity_stats = defaultdict(Counter)
    
    for i, text in enumerate(texts, 1):
        print(f"\n【句子 {i}】")
        print(f"原文：{text}")
        
        try:
            # 分词
            tokens = word_tokenize(text)
            
            # 词性标注
            pos_tags = pos_tag(tokens)
            
            # 命名实体识别
            ner_tree = ne_chunk(pos_tags)
            
            # 提取实体
            entities = []
            for subtree in ner_tree:
                if isinstance(subtree, Tree):
                    entity_name = ' '.join([token for token, pos in subtree.leaves()])
                    entity_label = subtree.label()
                    entities.append((entity_name, entity_label))
                    all_entities[entity_label].append(entity_name)
                    entity_stats[entity_label][entity_name] += 1
            
            if entities:
                print(f"识别到的实体：")
                for entity_name, entity_label in entities:
                    print(f"  {entity_name} - {entity_label}")
            else:
                print("  未识别到实体")
            
            # 显示NER树结构（简化版）
            print(f"NER树结构：")
            tree_str = str(ner_tree)
            if len(tree_str) > 200:
                tree_str = tree_str[:200] + "..."
            print(f"  {tree_str}")
            
        except Exception as e:
            print(f"  处理失败：{e}")
        
        print("-" * 40)
    
    # 统计总结
    print(f"\n实体识别统计总结：")
    for entity_type, entities in entity_stats.items():
        print(f"\n{entity_type} ({len(entities)} 种)：")
        for entity, count in entities.most_common():
            print(f"  {entity}: {count} 次")
    
    return all_entities, entity_stats

def create_ner_visualization(chinese_stats, english_stats, plt):
    """创建NER结果可视化"""
    print("\n生成NER结果可视化图表...")
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 中文实体类型分布
    if chinese_stats:
        chinese_types = list(chinese_stats.keys())
        chinese_counts = [len(entities) for entities in chinese_stats.values()]
        
        ax1.pie(chinese_counts, labels=chinese_types, autopct='%1.1f%%', startangle=90)
        ax1.set_title('中文实体类型分布')
    
    # 英文实体类型分布
    if english_stats:
        english_types = list(english_stats.keys())
        english_counts = [len(entities) for entities in english_stats.values()]
        
        ax2.pie(english_counts, labels=english_types, autopct='%1.1f%%', startangle=90)
        ax2.set_title('English Entity Type Distribution')
    
    # 中文高频实体
    if chinese_stats:
        all_chinese_entities = []
        for entities in chinese_stats.values():
            all_chinese_entities.extend(entities.most_common(5))
        
        if all_chinese_entities:
            entities, counts = zip(*all_chinese_entities[:10])
            ax3.barh(range(len(entities)), counts, color='skyblue')
            ax3.set_yticks(range(len(entities)))
            ax3.set_yticklabels(entities)
            ax3.set_title('中文高频实体 Top 10')
            ax3.set_xlabel('频次')
    
    # 英文高频实体
    if english_stats:
        all_english_entities = []
        for entities in english_stats.values():
            all_english_entities.extend(entities.most_common(5))
        
        if all_english_entities:
            entities, counts = zip(*all_english_entities[:10])
            ax4.barh(range(len(entities)), counts, color='lightcoral')
            ax4.set_yticks(range(len(entities)))
            ax4.set_yticklabels(entities)
            ax4.set_title('English High Frequency Entities Top 10')
            ax4.set_xlabel('Frequency')
    
    plt.tight_layout()
    
    # 保存图表
    output_dir = os.path.join("result", "task7")
    os.makedirs(output_dir, exist_ok=True)
    
    chart_path = os.path.join(output_dir, "ner_analysis_chart.png")
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    print(f"NER分析图表已保存到：{chart_path}")
    
    plt.show()
    plt.close()
    
    return chart_path


def main():
    """主函数"""
    print("=" * 80)
    print("任务7：命名实体识别（NER）")
    print("=" * 80)
    
    # 安装和导入所需包
    packages = install_and_import_packages()
    jieba, nltk, word_tokenize, pos_tag, ne_chunk, Tree = packages
    
    # 1. 准备中文实体数据库
    entity_db = get_chinese_entity_database()
    print(f"中文实体数据库包含：")
    for entity_type, entities in entity_db.items():
        print(f"  {entity_type}: {len(entities)} 个")
    
    # 2. 加载语料库
    data_dir = "data"
    chinese_poems = load_chinese_corpus(data_dir)
    english_texts = load_english_corpus()
    
    # 3. 中文命名实体识别
    chinese_entities, chinese_stats = analyze_chinese_poetry_ner(chinese_poems, entity_db)
    
    # 4. 英文命名实体识别
    english_entities = {}
    english_stats = {}
    
    if all([word_tokenize, pos_tag, ne_chunk, Tree]):
        english_entities, english_stats = english_ner_nltk(
            english_texts, word_tokenize, pos_tag, ne_chunk, Tree
        )
    else:
        print("\n⚠ NLTK NER功能不可用，跳过英文实体识别")
    
    # 5. 生成可视化结果
    # 设置matplotlib中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
    plt.rcParams['axes.unicode_minus'] = False
    
    chart_path = create_ner_visualization(chinese_stats, english_stats, plt)
    

    # 6. 总结报告
    print("\n" + "="*80)
    print("任务7完成！")
    print("="*80)
    print(f"✓ NER分析图表：{chart_path}")
    
    print(f"\n中文NER统计：")
    total_chinese = sum(len(entities) for entities in chinese_entities.values())
    print(f"  - 总实体数：{total_chinese} 个")
    print(f"  - 实体类型：{len(chinese_entities)} 种")
    
    print(f"\n英文NER统计：")
    total_english = sum(len(entities) for entities in english_entities.values())
    print(f"  - 总实体数：{total_english} 个")
    print(f"  - 实体类型：{len(english_entities)} 种")
    
    print(f"\n命名实体识别帮助我们：")
    print("• 提取文本中的关键信息")
    print("• 识别人物、地点、组织等实体")
    print("• 为信息检索和知识图谱构建提供基础")
    print("• 支持文本结构化和语义分析")
    print("="*80)

if __name__ == "__main__":
    main() 