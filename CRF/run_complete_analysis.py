#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完整的汉语词法分析器训练和评估流程
自动创建文件夹结构并保存所有结果
"""

import os
import sys
from datetime import datetime

def create_output_structure():
    """创建输出目录结构"""
    # 使用时间戳创建唯一的输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = f"analysis_results_{timestamp}"
    
    # 创建目录结构
    dirs = {
        'base': base_dir,
        'models': os.path.join(base_dir, 'models'),
        'metrics': os.path.join(base_dir, 'metrics'),
        'visualizations': os.path.join(base_dir, 'visualizations'),
        'reports': os.path.join(base_dir, 'reports')
    }
    
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
        print(f"创建目录: {dir_path}")
    
    return dirs

def find_existing_models():
    """查找现有的模型文件"""
    # 查找所有可能的模型目录
    model_dirs = []
    
    # 检查根目录下的模型文件
    if os.path.exists('enhanced_crf_model.pkl') and os.path.exists('enhanced_vocab.pkl'):
        model_dirs.append({
            'path': '.',
            'model_file': 'enhanced_crf_model.pkl',
            'vocab_file': 'enhanced_vocab.pkl',
            'description': '根目录'
        })
    
    # 查找所有 analysis_results_* 目录下的模型
    for item in os.listdir('.'):
        if os.path.isdir(item) and item.startswith('analysis_results_'):
            models_dir = os.path.join(item, 'models')
            if os.path.exists(models_dir):
                model_file = os.path.join(models_dir, 'enhanced_crf_model.pkl')
                vocab_file = os.path.join(models_dir, 'enhanced_vocab.pkl')
                if os.path.exists(model_file) and os.path.exists(vocab_file):
                    model_dirs.append({
                        'path': models_dir,
                        'model_file': model_file,
                        'vocab_file': vocab_file,
                        'description': f'历史结果目录 ({item})'
                    })
    
    return model_dirs

def run_training_and_evaluation(auto_mode=False):
    """运行训练和评估"""
    print("="*70)
    print("汉语词法分析器完整训练和评估流程")
    print("="*70)
    
    # 检查必要文件是否存在
    required_files = [
        'test_pd_src/test_pd_src.txt',
        'enhanced_pos_tagger.py',
        'evaluation_metrics.py'
    ]
    
    print("检查必要文件...")
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"错误: 缺少必要文件 {file_path}")
            return False
        else:
            print(f"✓ 找到文件: {file_path}")
    
    # 创建输出目录结构
    print("\n创建输出目录结构...")
    dirs = create_output_structure()
    
    # 修改主程序以使用新的目录结构
    from enhanced_pos_tagger import EnhancedChinesePOSTagger
    from evaluation_metrics import POSTaggingEvaluator
    
    # 初始化标注器
    print("\n初始化增强版标注器...")
    enhanced_tagger = EnhancedChinesePOSTagger()
    
    # 查找现有模型
    existing_models = find_existing_models()
    use_existing = False
    
    if existing_models:
        print(f"\n发现 {len(existing_models)} 个现有模型:")
        for i, model_info in enumerate(existing_models, 1):
            model_size = os.path.getsize(model_info['model_file']) / (1024*1024)
            vocab_size = os.path.getsize(model_info['vocab_file']) / (1024*1024)
            print(f"{i}. {model_info['description']}")
            print(f"   模型文件: {model_info['model_file']} ({model_size:.1f}MB)")
            print(f"   词汇文件: {model_info['vocab_file']} ({vocab_size:.1f}MB)")
        
        if auto_mode:
            print("自动模式：使用第一个现有模型")
            selected_model = existing_models[0]
            use_existing = True
        else:
            print("\n选择操作:")
            print("1. 使用第一个现有模型进行评估")
            print("2. 重新训练新模型")
            print("3. 手动选择现有模型")
            
            try:
                choice = input("请输入选择 (1-3) [默认1]: ").strip()
                if not choice:
                    choice = "1"
                
                if choice == "1":
                    selected_model = existing_models[0]
                    use_existing = True
                elif choice == "3" and len(existing_models) > 1:
                    print("可用模型:")
                    for i, model_info in enumerate(existing_models, 1):
                        print(f"{i}. {model_info['description']}")
                    model_choice = input(f"请选择模型 (1-{len(existing_models)}): ").strip()
                    try:
                        model_idx = int(model_choice) - 1
                        if 0 <= model_idx < len(existing_models):
                            selected_model = existing_models[model_idx]
                            use_existing = True
                        else:
                            print("选择无效，将重新训练模型")
                    except ValueError:
                        print("输入无效，将重新训练模型")
                else:
                    print("将重新训练模型")
            except (EOFError, KeyboardInterrupt):
                print("\n使用默认选择：加载第一个现有模型")
                selected_model = existing_models[0]
                use_existing = True
    
    if use_existing and 'selected_model' in locals():
        print(f"\n加载现有模型: {selected_model['description']}")
        try:
            enhanced_tagger.load_model(selected_model['model_file'], selected_model['vocab_file'])
            print("模型加载成功！")
            # 将现有模型复制到新的输出目录
            import shutil
            new_model_path = os.path.join(dirs['models'], 'enhanced_crf_model.pkl')
            new_vocab_path = os.path.join(dirs['models'], 'enhanced_vocab.pkl')
            shutil.copy2(selected_model['model_file'], new_model_path)
            shutil.copy2(selected_model['vocab_file'], new_vocab_path)
            print(f"模型已复制到: {dirs['models']}")
        except Exception as e:
            print(f"模型加载失败: {e}")
            print("将重新训练模型...")
            use_existing = False
    
    if not use_existing:
        # 检查训练数据
        if not os.path.exists('test_pd_src/train_pd.txt'):
            print("错误: 缺少训练数据文件 test_pd_src/train_pd.txt")
            return False
        
        # 训练模型
        print("\n开始训练模型...")
        enhanced_tagger.train('test_pd_src/train_pd.txt')
        
        # 保存模型
        model_path = os.path.join(dirs['models'], 'enhanced_crf_model.pkl')
        vocab_path = os.path.join(dirs['models'], 'enhanced_vocab.pkl')
        enhanced_tagger.save_model(model_path, vocab_path)
    
    # 创建评估器
    print("\n初始化评估器...")
    evaluator = POSTaggingEvaluator(enhanced_tagger, enhanced_tagger.word_vocab)
    
    # 进行综合评估
    print("\n开始综合评估...")
    # 切换到指标目录保存JSON结果
    original_dir = os.getcwd()
    os.chdir(dirs['metrics'])
    
    # 尝试使用黄金标准评估，如果失败则回退到jieba评估
    try:
        evaluation_results = evaluator.comprehensive_evaluation_with_gold_standard('../../test_pd_src/test_pd_src.txt')
        print("✓ 使用黄金标准评估")
    except Exception as e:
        print(f"黄金标准评估失败: {e}")
        print("回退到jieba参考标准评估...")
        evaluation_results = evaluator.comprehensive_evaluation('../../test_pd_src/test_pd_src.txt')
        print("✓ 使用jieba参考标准评估")
    
    os.chdir(original_dir)
    
    # 生成错误分析报告
    print("\n生成错误分析报告...")
    error_analysis_path = os.path.join(dirs['reports'], 'error_analysis_report.txt')
    evaluator.analyze_error_patterns('test_pd_src/test_pd_src.txt', error_analysis_path)
    
    # 生成可视化图表
    print("\n生成可视化图表...")
    try:
        os.chdir(dirs['visualizations'])
        evaluator.visualize_results(evaluation_results)
        os.chdir(original_dir)
        print(f"可视化图表已保存到: {dirs['visualizations']}")
    except Exception as e:
        print(f"可视化生成失败: {e}")
        os.chdir(original_dir)
    
    # 生成完整的评估报告
    print("\n生成完整评估报告...")
    generate_complete_report(dirs, evaluation_results, enhanced_tagger)
    
    # 生成OOV演示
    print("\n生成OOV词处理演示...")
    generate_oov_demo(dirs, enhanced_tagger)
    
    print(f"\n" + "="*70)
    print("分析完成！所有结果已保存到:")
    print(f"主目录: {dirs['base']}")
    print(f"├── models/           (训练好的模型文件)")
    print(f"├── metrics/          (评估指标JSON文件)")
    print(f"├── visualizations/   (性能图表)")
    print(f"└── reports/          (详细分析报告)")
    print("="*70)
    
    return True

def generate_complete_report(dirs, evaluation_results, tagger):
    """生成完整的评估报告"""
    report_path = os.path.join(dirs['reports'], 'complete_evaluation_report.txt')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("汉语词法分析器完整评估报告\n")
        f.write("="*60 + "\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("1. 模型概述\n")
        f.write("-"*30 + "\n")
        f.write("模型类型: 增强版CRF词性标注器\n")
        f.write("特征工程: 形态学特征、字符级特征、OOV处理\n")
        f.write(f"词汇表大小: {len(tagger.word_vocab)}\n")
        f.write(f"字符表大小: {len(tagger.char_vocab)}\n")
        f.write(f"词性标签数: {len(tagger.pos_vocab)}\n\n")
        
        f.write("2. 整体性能指标\n")
        f.write("-"*30 + "\n")
        f.write(f"标注准确率: {evaluation_results['word_accuracy']:.4f}\n")
        f.write(f"精确率: {evaluation_results['precision']:.4f}\n")
        f.write(f"召回率: {evaluation_results['recall']:.4f}\n")
        f.write(f"F1分数: {evaluation_results['f1_score']:.4f}\n\n")
        
        f.write("3. OOV词处理性能\n")
        f.write("-"*30 + "\n")
        oov_metrics = evaluation_results['oov_metrics']
        f.write(f"登录词(IV)数量: {oov_metrics['iv_total']}\n")
        f.write(f"登录词准确率: {oov_metrics['iv_accuracy']:.4f}\n")
        f.write(f"未登录词(OOV)数量: {oov_metrics['oov_total']}\n")
        f.write(f"OOV词召回率: {oov_metrics['oov_recall']:.4f}\n")
        f.write(f"OOV词占比: {oov_metrics['oov_ratio']*100:.2f}%\n\n")
        
        f.write("4. 词性标签性能 (Top 10)\n")
        f.write("-"*30 + "\n")
        pos_metrics = evaluation_results['pos_metrics']['per_pos']
        sorted_pos = sorted(pos_metrics.items(), key=lambda x: x[1]['f1'], reverse=True)
        for i, (pos, metrics) in enumerate(sorted_pos[:10]):
            f.write(f"{i+1:2d}. {pos:8s} P:{metrics['precision']:.3f} R:{metrics['recall']:.3f} F1:{metrics['f1']:.3f}\n")
    
    print(f"完整评估报告已保存到: {report_path}")

def generate_oov_demo(dirs, tagger):
    """生成OOV词处理演示"""
    demo_path = os.path.join(dirs['reports'], 'oov_processing_demo.txt')
    
    test_texts = [
        "阿尔法狗击败了世界围棋冠军。",
        "他在微博上发布了最新动态。",
        "这款智能手机的性价比很高。",
        "COVID-19疫苗接种工作正在进行。",
        "人工智能技术日新月异。",
        "区块链技术备受关注。"
    ]
    
    with open(demo_path, 'w', encoding='utf-8') as f:
        f.write("OOV词处理演示\n")
        f.write("="*50 + "\n\n")
        f.write("说明: [OOV]标记表示该词为未登录词\n\n")
        
        for i, text in enumerate(test_texts, 1):
            result = tagger.segment_and_tag(text)
            f.write(f"示例 {i}:\n")
            f.write(f"原文: {text}\n")
            
            # 标识OOV词
            tagged_result = []
            oov_count = 0
            for word, pos in result:
                if tagger.is_oov_word(word):
                    tagged_result.append(f"{word}/{pos}[OOV]")
                    oov_count += 1
                else:
                    tagged_result.append(f"{word}/{pos}")
            
            f.write(f"标注: {' '.join(tagged_result)}\n")
            f.write(f"OOV词数量: {oov_count}/{len(result)}\n\n")
    
    print(f"OOV演示结果已保存到: {demo_path}")

if __name__ == "__main__":
    # 支持命令行参数
    auto_mode = len(sys.argv) > 1 and sys.argv[1] == "--auto"
    
    if auto_mode:
        print("运行自动模式（非交互式）")
    
    success = run_training_and_evaluation(auto_mode)
    if success:
        print("\n所有分析已完成！")
    else:
        print("\n分析过程中出现错误，请检查文件和依赖项。")
        sys.exit(1) 