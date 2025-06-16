#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
任务2：读取《全唐诗》数据集
"""

import json
import os
import glob
from collections import Counter

def load_tang_poetry_files(data_dir):
    """加载所有唐诗JSON文件"""
    # 查找所有唐诗文件
    tang_files = glob.glob(os.path.join(data_dir, "poet.tang.*.json"))
    tang_files.extend(glob.glob(os.path.join(data_dir, "唐诗*.json")))
    
    all_poems = []
    file_count = 0
    
    print(f"发现 {len(tang_files)} 个唐诗数据文件")
    print("正在加载文件...")
    
    for file_path in sorted(tang_files):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                poems = json.load(f)
                all_poems.extend(poems)
                file_count += 1
                print(f"✓ 加载文件: {os.path.basename(file_path)} - {len(poems)} 首诗")
        except Exception as e:
            print(f"✗ 加载失败: {os.path.basename(file_path)} - 错误: {e}")
    
    print(f"\n总计加载了 {file_count} 个文件，共 {len(all_poems)} 首诗")
    return all_poems

def display_poems(poems, num_poems=10):
    """展示若干首诗的格式化内容"""
    print(f"\n{'='*80}")
    print(f"诗歌内容展示（前{num_poems}首）")
    print(f"{'='*80}")
    
    for i, poem in enumerate(poems[:num_poems], 1):
        print(f"\n【诗歌 {i}】")
        print(f"题目：{poem.get('title', '无题')}")
        print(f"作者：{poem.get('author', '佚名')}")
        
        # 格式化打印诗句
        paragraphs = poem.get('paragraphs', [])
        if paragraphs:
            print("内容：")
            for line in paragraphs:
                print(f"      {line}")
        
        # 显示标签（如果有）
        tags = poem.get('tags', [])
        if tags:
            print(f"标签：{' | '.join(tags[:5])}")  # 只显示前5个标签
        
        print("-" * 60)

def analyze_poetry_statistics(poems):
    """分析诗歌统计信息"""
    print(f"\n{'='*80}")
    print("《全唐诗》数据统计分析")
    print(f"{'='*80}")
    
    # 基本统计
    total_poems = len(poems)
    print(f"总诗数：{total_poems:,} 首")
    
    # 统计总字符数和诗句数
    total_chars = 0
    total_lines = 0
    authors = []
    all_tags = []
    
    for poem in poems:
        # 统计字符和行数
        paragraphs = poem.get('paragraphs', [])
        for line in paragraphs:
            total_chars += len(line)
            total_lines += 1
        
        # 收集作者信息
        author = poem.get('author', '佚名')
        if author:
            authors.append(author)
        
        # 收集标签信息
        tags = poem.get('tags', [])
        all_tags.extend(tags)
    
    print(f"总诗句数：{total_lines:,} 句")
    print(f"总字符数：{total_chars:,} 字")
    print(f"平均每首诗句数：{total_lines/total_poems:.1f} 句")
    print(f"平均每句字数：{total_chars/total_lines:.1f} 字")
    
    # 作者统计
    author_counter = Counter(authors)
    print(f"\n作者统计：")
    print(f"不同作者数量：{len(author_counter)} 位")
    print("作品数量最多的前10位作者：")
    for author, count in author_counter.most_common(10):
        print(f"  {author}：{count} 首")
    
    # 标签统计
    if all_tags:
        tag_counter = Counter(all_tags)
        print(f"\n标签统计：")
        print(f"不同标签数量：{len(tag_counter)} 个")
        print("最常见的前10个标签：")
        for tag, count in tag_counter.most_common(10):
            print(f"  {tag}：{count} 次")
    
    return {
        'total_poems': total_poems,
        'total_lines': total_lines,
        'total_chars': total_chars,
        'authors': len(author_counter),
        'tags': len(tag_counter) if all_tags else 0
    }

def search_poems_by_author(poems, author_name, limit=5):
    """按作者搜索诗歌"""
    author_poems = [poem for poem in poems if poem.get('author') == author_name]
    
    if author_poems:
        print(f"\n{'='*80}")
        print(f"{author_name} 的诗作（前{min(limit, len(author_poems))}首）")
        print(f"{'='*80}")
        display_poems(author_poems, limit)
        print(f"\n{author_name} 共有 {len(author_poems)} 首诗作")
    else:
        print(f"\n未找到作者 '{author_name}' 的诗作")

def main():
    """主函数"""
    print("=" * 80)
    print("任务2：读取《全唐诗》数据集")
    print("=" * 80)
    
    # 设置数据目录
    data_dir = "data/Complete_Tang_Poems"
    
    if not os.path.exists(data_dir):
        print(f"错误：数据目录 '{data_dir}' 不存在")
        return
    
    # 加载所有唐诗数据
    poems = load_tang_poetry_files(data_dir)
    
    if not poems:
        print("错误：未能加载任何诗歌数据")
        return
    
    # 展示诗歌内容
    display_poems(poems, 8)
    
    # 统计分析
    stats = analyze_poetry_statistics(poems)
    
    # 按作者搜索示例
    famous_poets = ["李白", "杜甫", "白居易", "王维"]
    for poet in famous_poets:
        search_poems_by_author(poems, poet, 3)
    
    print(f"\n{'='*80}")
    print("任务2完成！")
    print(f"成功读取并分析了 {stats['total_poems']:,} 首唐诗")
    print(f"共 {stats['total_chars']:,} 字，{stats['total_lines']:,} 句")
    print(f"{'='*80}")

if __name__ == "__main__":
    main() 