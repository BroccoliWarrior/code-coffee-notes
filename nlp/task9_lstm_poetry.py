#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
任务9：LSTM写诗
目的：训练语言模型进行古诗自动生成，体现文本生成能力
特点：支持模型加载，如果已有训练好的模型则直接使用
"""

import json
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import pickle
import random

def install_and_import_packages():
    """检查并导入所需的包"""
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import Dataset, DataLoader
        print("✓ PyTorch 已可用")
        
        # 检查GPU
        print(f"PyTorch版本: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"✓ 检测到GPU加速: {torch.cuda.get_device_name(0)}")
            device = torch.device('cuda')
        else:
            print("ℹ 使用CPU训练")
            device = torch.device('cpu')
            
    except ImportError:
        print("正在安装 PyTorch...")
        import subprocess
        subprocess.check_call(["pip", "install", "torch"])
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import Dataset, DataLoader
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        import jieba
        print("✓ jieba 中文分词库已可用")
    except ImportError:
        print("正在安装 jieba...")
        import subprocess
        subprocess.check_call(["pip", "install", "jieba"])
        import jieba
    
    return torch, nn, optim, Dataset, DataLoader, device, jieba

def load_poetry_corpus(data_dir):
    """加载诗歌语料库"""
    print("\n" + "="*60)
    print("加载《全唐诗》语料库")
    print("="*60)
    
    all_poems = []
    
    # 尝试加载多个唐诗文件
    tang_poetry_files = [
        "唐诗三百首.json",
        "poet.tang.43030.json",
        "poet.tang.0.json"
    ]
    
    for filename in tang_poetry_files:
        file_path = os.path.join(data_dir, "Complete_Tang_Poems", filename)
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    poems_data = json.load(f)
                
                if isinstance(poems_data, list):
                    all_poems.extend(poems_data)
                    print(f"✓ 加载 {filename}: {len(poems_data)} 首诗")
                else:
                    print(f"⚠ {filename} 格式异常")
                    
            except Exception as e:
                print(f"✗ 加载 {filename} 失败: {e}")
    
    # 如果没有找到诗歌文件，使用内置示例
    if not all_poems:
        all_poems = [
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
            },
            {
                "title": "相思",
                "author": "王维",
                "paragraphs": ["红豆生南国，春来发几枝。", "愿君多采撷，此物最相思。"]
            },
            {
                "title": "登高",
                "author": "杜甫",
                "paragraphs": ["风急天高猿啸哀，渚清沙白鸟飞回。", "无边落木萧萧下，不尽长江滚滚来。"]
            },
            {
                "title": "望岳",
                "author": "杜甫",
                "paragraphs": ["岱宗夫如何，齐鲁青未了。", "造化钟神秀，阴阳割昏晓。"]
            },
            {
                "title": "春夜喜雨",
                "author": "杜甫",
                "paragraphs": ["好雨知时节，当春乃发生。", "随风潜入夜，润物细无声。"]
            },
            {
                "title": "黄鹤楼",
                "author": "崔颢",
                "paragraphs": ["昔人已乘黄鹤去，此地空余黄鹤楼。", "黄鹤一去不复返，白云千载空悠悠。"]
            }
        ]
        print("使用内置诗歌示例")
    
    print(f"总共加载诗歌：{len(all_poems)} 首")
    return all_poems

def clean_and_preprocess_poems(poems):
    """清洗和预处理诗歌数据"""
    print("\n" + "="*60)
    print("清洗和预处理诗歌数据")
    print("="*60)
    
    clean_poems = []
    
    for poem in poems:
        try:
            # 获取诗歌内容
            paragraphs = poem.get('paragraphs', [])
            if not paragraphs:
                continue
                
            # 合并所有段落
            full_text = ''.join(paragraphs)
            
            # 清洗文本
            # 移除标点符号，保留中文字符
            clean_text = re.sub(r'[^\u4e00-\u9fff]', '', full_text)
            
            # 过滤太短的诗（少于10个字）
            if len(clean_text) < 10:
                continue
                
            # 过滤太长的诗（超过200个字）
            if len(clean_text) > 200:
                clean_text = clean_text[:200]
            
            clean_poems.append(clean_text)
            
        except Exception as e:
            continue
    
    print(f"清洗后诗歌数量：{len(clean_poems)}")
    
    # 合并所有诗歌文本
    corpus = ''.join(clean_poems)
    print(f"语料库总字符数：{len(corpus)}")
    
    # 统计字符频率
    char_counts = Counter(corpus)
    print(f"不重复字符数：{len(char_counts)}")
    
    # 显示最常见的字符
    print("\n最常见的20个字符：")
    for char, count in char_counts.most_common(20):
        print(f"  '{char}': {count} 次")
    
    return clean_poems, corpus, char_counts

class PoetryDataset:
    """诗歌数据集类"""
    def __init__(self, corpus, seq_length=10):
        self.corpus = corpus
        self.seq_length = seq_length
        
        # 创建字符到索引的映射
        self.chars = sorted(list(set(corpus)))
        self.char_to_idx = {char: i for i, char in enumerate(self.chars)}
        self.idx_to_char = {i: char for i, char in enumerate(self.chars)}
        self.vocab_size = len(self.chars)
        
        # 创建训练序列
        self.sequences = []
        self.targets = []
        
        for i in range(len(corpus) - seq_length):
            seq = corpus[i:i + seq_length]
            target = corpus[i + seq_length]
            self.sequences.append([self.char_to_idx[char] for char in seq])
            self.targets.append(self.char_to_idx[target])
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        # 直接返回列表，让DataLoader处理张量转换
        return self.sequences[idx], self.targets[idx]

class LSTMPoetryModel:
    """LSTM诗歌生成模型类"""
    def __init__(self, vocab_size, embedding_dim=64, hidden_dim=128, num_layers=2):
        import torch.nn as nn
        
        class PoetryLSTM(nn.Module):
            def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers):
                super(PoetryLSTM, self).__init__()
                self.hidden_dim = hidden_dim
                self.num_layers = num_layers
                
                self.embedding = nn.Embedding(vocab_size, embedding_dim)
                self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, 
                                  batch_first=True, dropout=0.2)
                self.dropout = nn.Dropout(0.2)
                self.fc = nn.Linear(hidden_dim, vocab_size)
                
            def forward(self, x, hidden=None):
                embedded = self.embedding(x)
                lstm_out, hidden = self.lstm(embedded, hidden)
                lstm_out = self.dropout(lstm_out)
                # 只使用最后一个时间步的输出
                output = self.fc(lstm_out[:, -1, :])
                return output, hidden
                
            
            def init_hidden(self, batch_size, device):
                h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
                c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
                return (h0, c0)
        
        self.model = PoetryLSTM(vocab_size, embedding_dim, hidden_dim, num_layers)
        
    def get_model(self):
        return self.model

def train_model(model, dataset, device, epochs=30, batch_size=64, learning_rate=0.001):
    """训练模型"""
    print(f"\n开始训练模型...")
    print(f"  训练轮数: {epochs}")
    print(f"  批次大小: {batch_size}")
    print(f"  学习率: {learning_rate}")
    print(f"  设备: {device}")
    
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    
    # 创建数据加载器，使用自定义collate函数
    def collate_fn(batch):
        sequences, targets = zip(*batch)
        sequences = torch.tensor(sequences, dtype=torch.long)
        targets = torch.tensor(targets, dtype=torch.long)
        return sequences, targets
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    
    # 设置优化器和损失函数
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    model.to(device)
    model.train()
    
    # 记录训练历史
    train_losses = []
    train_accuracies = []
    
    print("\n开始训练...")
    for epoch in range(epochs):
        epoch_loss = 0
        epoch_accuracy = 0
        num_batches = 0
        
        for batch_idx, (sequences, targets) in enumerate(dataloader):
            sequences = sequences.to(device)
            targets = targets.to(device)
            
            # 前向传播
            optimizer.zero_grad()
            outputs, _ = model(sequences)
            loss = criterion(outputs, targets)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            # 计算准确率
            _, predicted = torch.max(outputs.data, 1)
            accuracy = (predicted == targets).float().mean()
            
            epoch_loss += loss.item()
            epoch_accuracy += accuracy.item()
            num_batches += 1
            
            # 每100个批次打印一次进度
            if batch_idx % 100 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Batch [{batch_idx}/{len(dataloader)}], '
                      f'Loss: {loss.item():.4f}, Accuracy: {accuracy.item():.4f}')
        
        # 记录每个epoch的平均损失和准确率
        avg_loss = epoch_loss / num_batches
        avg_accuracy = epoch_accuracy / num_batches
        train_losses.append(avg_loss)
        train_accuracies.append(avg_accuracy)
        
        print(f'Epoch [{epoch+1}/{epochs}] 完成, 平均损失: {avg_loss:.4f}, 平均准确率: {avg_accuracy:.4f}')
    
    # 保存模型
    model_dir = os.path.join("models", "task9")
    os.makedirs(model_dir, exist_ok=True)
    
    model_path = os.path.join(model_dir, "lstm_poetry_pytorch.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab_size': dataset.vocab_size,
        'seq_length': dataset.seq_length,
        'embedding_dim': 64,
        'hidden_dim': 128,
        'num_layers': 2
    }, model_path)
    
    print(f"模型已保存到: {model_path}")
    
    return {
        'loss': train_losses,
        'accuracy': train_accuracies
    }

def plot_training_history(history):
    """绘制训练历史"""
    print("\n生成训练历史图表...")
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # 损失曲线
    ax1.plot(history['loss'], label='训练损失', linewidth=2, color='blue')
    ax1.set_title('模型训练损失')
    ax1.set_xlabel('训练轮数')
    ax1.set_ylabel('损失值')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 准确率曲线
    ax2.plot(history['accuracy'], label='训练准确率', linewidth=2, color='green')
    ax2.set_title('模型训练准确率')
    ax2.set_xlabel('训练轮数')
    ax2.set_ylabel('准确率')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图表
    output_dir = os.path.join("result", "task9")
    os.makedirs(output_dir, exist_ok=True)
    
    chart_path = os.path.join(output_dir, "training_history.png")
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    print(f"训练历史图表已保存到：{chart_path}")
    
    plt.show()
    plt.close()
    
    return chart_path

def generate_poetry(model, seed_text, dataset, device, length=50, temperature=1.0):
    """生成诗歌"""
    import torch
    import torch.nn.functional as F
    
    model.eval()
    
    if len(seed_text) < dataset.seq_length:
        # 如果种子文本太短，重复最后一个字符
        seed_text = seed_text + seed_text[-1] * (dataset.seq_length - len(seed_text))
    
    # 取最后seq_length个字符作为输入
    input_text = seed_text[-dataset.seq_length:]
    generated = input_text
    
    with torch.no_grad():
        for _ in range(length):
            # 将输入文本转换为索引序列
            sequence = [dataset.char_to_idx.get(char, 0) for char in input_text]
            sequence = torch.tensor([sequence], dtype=torch.long).to(device)
            
            # 预测下一个字符
            output, _ = model(sequence)
            predictions = F.softmax(output / temperature, dim=1)
            
            # 采样下一个字符
            next_idx = torch.multinomial(predictions, 1).item()
            next_char = dataset.idx_to_char[next_idx]
            
            generated += next_char
            input_text = input_text[1:] + next_char
    
    return generated

def generate_multiple_poems(model, dataset, device):
    """生成多首诗歌"""
    print("\n" + "="*60)
    print("生成诗歌作品")
    print("="*60)
    
    # 不同的种子文本
    seed_texts = [
        "春风",
        "明月",
        "江南",
        "山水",
        "梅花",
        "秋雨",
        "夕阳",
        "青山"
    ]
    
    # 不同的温度参数
    temperatures = [0.5, 0.8, 1.0, 1.2]
    
    generated_poems = []
    
    print("生成诗歌：")
    print("-" * 40)
    
    for i, seed in enumerate(seed_texts, 1):
        temperature = temperatures[i % len(temperatures)]
        
        poem = generate_poetry(
            model, seed, dataset, device,
            length=40, temperature=temperature
        )
        
        # 格式化诗歌（每8个字换行）
        formatted_poem = ""
        for j in range(0, len(poem), 8):
            line = poem[j:j+8]
            if len(line) >= 4:  # 至少4个字才显示
                formatted_poem += line + "\n"
        
        generated_poems.append({
            'seed': seed,
            'temperature': temperature,
            'poem': poem,
            'formatted': formatted_poem.strip()
        })
        
        print(f"【诗歌 {i}】种子：{seed}，温度：{temperature}")
        print(formatted_poem.strip())
        print("-" * 40)
    
    return generated_poems

def save_generated_poems(generated_poems):
    """保存生成的诗歌"""
    output_dir = os.path.join("result", "task9")
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存为文本文件
    txt_path = os.path.join(output_dir, "generated_poems.txt")
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("LSTM古诗生成结果（PyTorch版本）\n")
        f.write("=" * 40 + "\n\n")
        
        for i, poem_data in enumerate(generated_poems, 1):
            f.write(f"【诗歌 {i}】\n")
            f.write(f"种子文本：{poem_data['seed']}\n")
            f.write(f"温度参数：{poem_data['temperature']}\n")
            f.write(f"生成诗句：\n{poem_data['formatted']}\n")
            f.write("-" * 40 + "\n\n")
    
    # 保存为JSON文件
    json_path = os.path.join(output_dir, "generated_poems.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(generated_poems, f, ensure_ascii=False, indent=2)
    
    print(f"生成的诗歌已保存到：")
    print(f"  文本文件：{txt_path}")
    print(f"  JSON文件：{json_path}")
    
    return txt_path, json_path

def save_model_artifacts(dataset):
    """保存模型相关文件"""
    model_dir = os.path.join("models", "task9")
    os.makedirs(model_dir, exist_ok=True)
    
    # 保存字符映射
    artifacts = {
        'char_to_idx': dataset.char_to_idx,
        'idx_to_char': dataset.idx_to_char,
        'chars': dataset.chars,
        'seq_length': dataset.seq_length,
        'vocab_size': dataset.vocab_size
    }
    
    artifacts_path = os.path.join(model_dir, "lstm_poetry_artifacts_pytorch.pkl")
    with open(artifacts_path, 'wb') as f:
        pickle.dump(artifacts, f)
    
    print(f"模型配置文件已保存到：{artifacts_path}")
    return artifacts_path

def load_existing_model(torch, device):
    """加载已存在的模型"""
    model_dir = os.path.join("models", "task9")
    model_path = os.path.join(model_dir, "lstm_poetry_pytorch.pth")
    artifacts_path = os.path.join(model_dir, "lstm_poetry_artifacts_pytorch.pkl")
    
    # 检查文件是否存在
    if not os.path.exists(model_path) or not os.path.exists(artifacts_path):
        return None, None
    
    try:
        # 加载模型配置
        with open(artifacts_path, 'rb') as f:
            artifacts = pickle.load(f)
        
        print(f"✓ 找到已训练的模型")
        print(f"  模型文件：{model_path}")
        print(f"  配置文件：{artifacts_path}")
        print(f"  词汇表大小：{artifacts['vocab_size']}")
        print(f"  序列长度：{artifacts['seq_length']}")
        
        # 重建模型
        model_wrapper = LSTMPoetryModel(
            vocab_size=artifacts['vocab_size'],
            embedding_dim=64,
            hidden_dim=128,
            num_layers=2
        )
        model = model_wrapper.get_model()
        
        # 加载模型权重
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        # 创建伪数据集对象用于生成
        class LoadedDataset:
            def __init__(self, artifacts):
                self.char_to_idx = artifacts['char_to_idx']
                self.idx_to_char = artifacts['idx_to_char']
                self.chars = artifacts['chars']
                self.seq_length = artifacts['seq_length']
                self.vocab_size = artifacts['vocab_size']
        
        dataset = LoadedDataset(artifacts)
        
        print("✓ 模型加载成功，跳过训练阶段")
        return model, dataset
        
    except Exception as e:
        print(f"❌ 加载模型失败：{e}")
        return None, None

def main():
    """主函数"""
    print("=" * 80)
    print("任务9：LSTM写诗（PyTorch版本）")
    print("=" * 80)
    
    # 安装和导入所需包
    packages = install_and_import_packages()
    torch, nn, optim, Dataset, DataLoader, device, jieba = packages
    
    # 1. 尝试加载已存在的模型
    model, dataset = load_existing_model(torch, device)
    
    if model is not None and dataset is not None:
        # 如果成功加载模型，跳过训练
        print("使用已训练的模型，直接进行诗歌生成...")
        chart_path = None  # 没有新的训练历史
        
    else:
        # 如果没有模型，进行完整的训练流程
        print("没有找到已训练的模型，开始从头训练...")
        
        # 2. 加载诗歌语料库
        data_dir = "data"
        poems = load_poetry_corpus(data_dir)
        
        # 3. 清洗和预处理数据
        clean_poems, corpus, char_counts = clean_and_preprocess_poems(poems)
        
        # 4. 创建数据集
        seq_length = 10
        dataset = PoetryDataset(corpus, seq_length)
        
        print(f"\n训练数据统计：")
        print(f"  序列数量: {len(dataset)}")
        print(f"  序列长度: {dataset.seq_length}")
        print(f"  词汇表大小: {dataset.vocab_size}")
        
        # 5. 构建LSTM模型
        model_wrapper = LSTMPoetryModel(
            vocab_size=dataset.vocab_size,
            embedding_dim=64,
            hidden_dim=128,
            num_layers=2
        )
        model = model_wrapper.get_model()
        
        print(f"\n模型参数：")
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  总参数数量: {total_params:,}")
        print(f"  可训练参数: {trainable_params:,}")
        
        # 6. 训练模型
        print(f"\n准备开始训练...")
        epochs = 25 if len(clean_poems) > 100 else 15  # 根据数据量调整训练轮数
        
        history = train_model(
            model, dataset, device,
            epochs=epochs,
            batch_size=32,
            learning_rate=0.001
        )
        
        # 7. 绘制训练历史
        chart_path = plot_training_history(history)
        
        # 8. 保存模型配置文件
        artifacts_path = save_model_artifacts(dataset)
    
    # 9. 生成诗歌
    generated_poems = generate_multiple_poems(model, dataset, device)
    
    # 10. 保存结果
    txt_path, json_path = save_generated_poems(generated_poems)
    
    # 11. 最终报告
    print("\n" + "="*80)
    print("任务9完成！")
    print("="*80)
    if chart_path:
        print(f"✓ 训练历史图表：{chart_path}")
    print(f"✓ 生成诗歌文本：{txt_path}")
    print(f"✓ 生成诗歌JSON：{json_path}")
    
    print(f"\n模型统计：")
    print(f"  - 字符词汇表：{dataset.vocab_size} 个字符")
    print(f"  - 序列长度：{dataset.seq_length}")
    
    if chart_path:  # 只有新训练的模型才显示训练统计
        print(f"  - 最终训练损失：{history['loss'][-1]:.4f}")
        print(f"  - 最终训练准确率：{history['accuracy'][-1]:.4f}")
    else:
        print("  - 使用已训练模型")
    
    print(f"\n示例生成诗句：")
    for i, poem_data in enumerate(generated_poems[:3], 1):
        lines = poem_data['formatted'].split('\n')
        first_line = lines[0] if lines else ""
        print(f"  {i}. {first_line}...")
    
    print(f"\nLSTM古诗生成帮助我们：")
    print("• 学习古典诗歌的语言模式和韵律")
    print("• 理解循环神经网络在序列建模中的应用")
    print("• 体验深度学习在创意写作中的潜力")
    print("• 探索文本生成的温度调节技巧")
    print("• 掌握PyTorch深度学习框架的使用")
    print("="*80)

if __name__ == "__main__":
    main() 