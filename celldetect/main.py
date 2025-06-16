#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
血细胞检测系统主程序
基于HOG特征提取 + PCA降维 + SVM分类器的目标检测系统

支持的模式:
- train: 训练模式
- predict: 预测模式  
- evaluate: 评估模式
- demo: 演示模式
"""

import argparse
import os
import sys
import time
import json
import numpy as np
import random
from tqdm import tqdm

# 导入自定义模块
from data_loader import BCCDDataLoader
from feature_extractor import HOGFeatureExtractor
from detector import CellDetector
from visualizer import DetectionVisualizer
from data_augmentation import DataAugmenter


def create_parser():
    """创建命令行参数解析器"""
    parser = argparse.ArgumentParser(description='血细胞检测系统')
    
    # 基本参数
    parser.add_argument('--mode', type=str, required=True, 
                       choices=['train', 'predict', 'evaluate', 'demo'],
                       help='运行模式')
    parser.add_argument('--dataset_path', type=str, default='BCCD',
                       help='数据集路径')
    parser.add_argument('--output_dir', type=str, default='outputs',
                       help='输出目录')
    
    # 模型参数
    parser.add_argument('--model_path', type=str, default=None,
                       help='模型保存/加载路径')
    parser.add_argument('--feature_model_path', type=str, default=None,
                       help='特征提取器模型路径')
    
    # HOG特征参数
    parser.add_argument('--hog_orientations', type=int, default=9,
                       help='HOG方向数量')
    parser.add_argument('--hog_pixels_per_cell', type=int, nargs=2, default=[8, 8],
                       help='每个cell的像素数')
    parser.add_argument('--hog_cells_per_block', type=int, nargs=2, default=[2, 2],
                       help='每个block的cell数')
    parser.add_argument('--target_size', type=int, nargs=2, default=[64, 64],
                       help='目标图像大小')
    
    # PCA参数
    parser.add_argument('--pca_components', type=int, default=None,
                       help='PCA组件数量')
    parser.add_argument('--variance_ratio', type=float, default=0.95,
                       help='保留的方差比例')
    
    # SVM参数
    parser.add_argument('--svm_c', type=float, default=1.0,
                       help='SVM正则化参数')
    parser.add_argument('--svm_kernel', type=str, default='rbf',
                       choices=['linear', 'poly', 'rbf', 'sigmoid'],
                       help='SVM核函数')
    parser.add_argument('--svm_gamma', type=str, default='scale',
                       help='SVM gamma参数')
    
    # 训练参数
    parser.add_argument('--negative_samples', type=int, default=1000,
                       help='负样本数量')
    parser.add_argument('--train_split', type=str, default='train',
                       help='训练数据分割')
    
    # 数据增强参数
    parser.add_argument('--enable_augmentation', action='store_true',
                       help='启用数据增强')
    parser.add_argument('--balance_samples', action='store_true',
                       help='平衡样本数量')
    parser.add_argument('--target_samples_per_class', type=int, default=None,
                       help='每个类别的目标样本数量')
    parser.add_argument('--show_augmentation_preview', action='store_true',
                       help='显示数据增强预览')
    
    # 检测参数
    parser.add_argument('--confidence_threshold', type=float, default=0.5,
                       help='置信度阈值')
    parser.add_argument('--nms_threshold', type=float, default=0.3,
                       help='NMS阈值')
    parser.add_argument('--iou_threshold', type=float, default=0.5,
                       help='IoU阈值，用于评估模式')
    
    # 选择性搜索参数
    parser.add_argument('--search_strategy', type=str, default='fast',
                       choices=['fast', 'quality'],
                       help='选择性搜索策略')
    parser.add_argument('--max_proposals', type=int, default=2000,
                       help='最大候选区域数量')
    parser.add_argument('--min_region_size', type=int, default=20,
                       help='最小区域大小')
    
    # 可视化参数
    parser.add_argument('--visualize', action='store_true',
                       help='是否可视化结果')
    parser.add_argument('--save_images', action='store_true',
                       help='是否保存结果图像')
    parser.add_argument('--num_vis_images', type=int, default=4,
                       help='可视化图像数量')
    parser.add_argument('--num_predict_images', type=int, default=None,
                       help='预测模式中处理的图像数量（None表示处理全部）')
    
    return parser


def setup_directories(output_dir):
    """创建输出目录"""
    dirs = ['models', 'results', 'visualizations']
    for dir_name in dirs:
        os.makedirs(os.path.join(output_dir, dir_name), exist_ok=True)


def create_feature_extractor(args):
    """创建特征提取器"""
    return HOGFeatureExtractor(
        orientations=args.hog_orientations,
        pixels_per_cell=tuple(args.hog_pixels_per_cell),
        cells_per_block=tuple(args.hog_cells_per_block),
        target_size=tuple(args.target_size)
    )


def create_detector(feature_extractor, args):
    """创建检测器"""
    svm_params = {
        'C': args.svm_c,
        'kernel': args.svm_kernel,
        'gamma': args.svm_gamma,
        'probability': True
    }
    
    detector = CellDetector(feature_extractor, svm_params)
    
    # 配置选择性搜索
    detector.selective_search.strategy = args.search_strategy
    detector.selective_search.max_proposals = args.max_proposals
    detector.selective_search.min_size = args.min_region_size
    
    return detector


def train_mode(args):
    """训练模式"""
    print("=" * 50)
    print("开始训练模式")
    print("=" * 50)
    
    # 创建输出目录
    setup_directories(args.output_dir)
    
    # 初始化数据加载器
    print("初始化数据加载器...")
    data_loader = BCCDDataLoader(args.dataset_path)
    
    # 加载训练数据 - 使用全部train.txt
    print("加载训练数据...")
    train_images = data_loader.load_image_list('train')
    print(f"找到 {len(train_images)} 张训练图像")
    
    # 提取正样本
    print("提取正样本...")
    positive_regions, positive_labels = data_loader.extract_positive_samples(train_images)
    print(f"提取到 {len(positive_regions)} 个正样本")
    
    # 统计各类别样本数量
    class_info = data_loader.get_class_info()
    for class_id, class_name in class_info['id_to_class'].items():
        count = positive_labels.count(class_id)
        print(f"  {class_name}: {count} 个样本")
    
    # 数据增强和样本均衡
    if args.enable_augmentation or args.balance_samples:
        print("\n" + "=" * 50)
        print("数据增强和样本均衡")
        print("=" * 50)
        
        # 创建数据增强器
        augmenter = DataAugmenter(target_size=tuple(args.target_size))
        
        # 显示增强预览
        if args.show_augmentation_preview and positive_regions:
            print("显示数据增强预览...")
            preview_path = os.path.join(args.output_dir, 'visualizations', 'augmentation_preview.png')
            augmenter.create_augmentation_preview(positive_regions[0], save_path=preview_path)
        
        # 样本均衡
        if args.balance_samples:
            print("执行样本均衡...")
            positive_regions, positive_labels = augmenter.balance_samples(
                positive_regions, 
                positive_labels,
                target_samples_per_class=args.target_samples_per_class
            )
            
            # 更新统计
            print(f"均衡后总正样本数量: {len(positive_regions)}")
    
    # 生成负样本
    print(f"\n生成 {args.negative_samples} 个负样本...")
    negative_regions = data_loader.generate_negative_samples(
        train_images, 
        num_samples=args.negative_samples
    )
    print(f"生成了 {len(negative_regions)} 个负样本")
    
    # 创建特征提取器和检测器
    print("创建特征提取器和检测器...")
    feature_extractor = create_feature_extractor(args)
    detector = create_detector(feature_extractor, args)
    
    # 可视化训练样本
    if args.visualize:
        print("可视化训练样本...")
        visualizer = DetectionVisualizer()
        visualizer.plot_training_samples(
            positive_regions[:50],  # 只显示前50个正样本
            negative_regions[:10],  # 只显示前10个负样本
            positive_labels[:50],
            save_path=os.path.join(args.output_dir, 'visualizations', 'training_samples.png')
        )
    
    # 训练检测器
    print("开始训练检测器...")
    start_time = time.time()
    
    train_result = detector.train(
        positive_regions=positive_regions,
        positive_labels=positive_labels,
        negative_regions=negative_regions,
        pca_components=args.pca_components,
        variance_ratio=args.variance_ratio
    )
    
    training_time = time.time() - start_time
    print(f"训练完成！耗时: {training_time:.2f} 秒")
    
    # 保存模型
    model_dir = os.path.join(args.output_dir, 'models')
    feature_model_path = os.path.join(model_dir, 'feature_extractor.pkl')
    detector_model_path = os.path.join(model_dir, 'detector.pkl')
    
    feature_extractor.save_model(feature_model_path)
    detector.save_model(detector_model_path)
    
    # 保存训练结果
    train_result['training_time'] = training_time
    train_result['data_augmentation'] = {
        'enabled': args.enable_augmentation,
        'balance_samples': args.balance_samples,
        'target_samples_per_class': args.target_samples_per_class
    }
    train_result['model_paths'] = {
        'feature_extractor': feature_model_path,
        'detector': detector_model_path
    }
    
    result_path = os.path.join(args.output_dir, 'results', 'training_result.json')
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(train_result, f, indent=2, ensure_ascii=False)
    
    print(f"训练结果已保存到: {result_path}")
    print("训练模式完成！")


def predict_mode(args):
    """预测模式"""
    print("=" * 50)
    print("开始预测模式")
    print("=" * 50)
    
    # 检查模型文件
    if args.feature_model_path is None:
        args.feature_model_path = os.path.join(args.output_dir, 'models', 'feature_extractor.pkl')
    if args.model_path is None:
        args.model_path = os.path.join(args.output_dir, 'models', 'detector.pkl')
    
    if not os.path.exists(args.feature_model_path):
        print(f"错误: 特征提取器模型文件不存在: {args.feature_model_path}")
        return
    if not os.path.exists(args.model_path):
        print(f"错误: 检测器模型文件不存在: {args.model_path}")
        return
    
    # 创建输出目录
    setup_directories(args.output_dir)
    
    # 加载模型
    print("加载模型...")
    feature_extractor = create_feature_extractor(args)
    feature_extractor.load_model(args.feature_model_path)
    
    detector = create_detector(feature_extractor, args)
    detector.load_model(args.model_path)
    
    # 初始化数据加载器
    data_loader = BCCDDataLoader(args.dataset_path)
    
    # 加载测试图像 - 使用全部test.txt
    test_images = data_loader.load_image_list('test')
    
    # 根据参数限制处理的图像数量（随机抽样）
    if args.num_predict_images is not None:
        # 确保不超过原始列表长度
        k = min(args.num_predict_images, len(test_images))
        # 从 test_images 中随机抽取 k 张图像（不放回采样）
        test_images = random.sample(test_images, k)
        print(f"限制处理图像数量为: {k}（随机抽样）")

    print(f"对 {len(test_images)} 张图像进行预测...")
    
    # 进行预测
    predictions_list = []
    images_list = []
    
    for image_name in tqdm(test_images, desc="预测中"):
        image, annotations = data_loader.parse_annotation(image_name)
        images_list.append(image)
        
        # 检测
        detections = detector.detect(
            image,
            confidence_threshold=args.confidence_threshold,
            nms_threshold=args.nms_threshold
        )
        predictions_list.append(detections)
        
        print(f"图像 {image_name}: 检测到 {len(detections)} 个目标")
        for det in detections:
            class_name = data_loader.id_to_class[det['class_id']]
            print(f"  {class_name}: 置信度 {det['confidence']:.3f}")
    
    # 可视化结果 - 只显示前几张图像
    if args.visualize:
        print("可视化预测结果...")
        visualizer = DetectionVisualizer()
        vis_count = min(args.num_vis_images, len(images_list))
        visualizer.plot_detection_results(
            images_list[:vis_count],
            predictions_list[:vis_count],
            num_images=vis_count,
            save_path=os.path.join(args.output_dir, 'visualizations', 'predictions.png')
        )
    
    # 保存结果图像 - 保存所有图像
    if args.save_images:
        print("保存结果图像...")
        visualizer = DetectionVisualizer()
        for i, (image, detections, image_name) in enumerate(zip(images_list, predictions_list, test_images)):
            save_path = os.path.join(args.output_dir, 'results', f'{image_name}_result.jpg')
            visualizer.save_detection_image(image, detections, save_path)
    
    # 创建检测摘要
    visualizer = DetectionVisualizer()
    summary = visualizer.create_detection_summary(predictions_list)
    
    print("\n检测摘要:")
    print(f"总检测数量: {summary['total_detections']}")
    print(f"平均每张图检测数量: {summary['average_detections_per_image']:.2f}")
    print(f"平均置信度: {summary['average_confidence']:.3f}")
    print("各类别检测数量:")
    for class_name, count in summary['class_counts'].items():
        print(f"  {class_name}: {count}")
    
    # 保存摘要
    summary_path = os.path.join(args.output_dir, 'results', 'prediction_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"预测摘要已保存到: {summary_path}")
    print("预测模式完成！")


def evaluate_mode(args):
    """评估模式"""
    print("=" * 50)
    print("开始评估模式")
    print("=" * 50)
    
    # 检查模型文件
    if args.feature_model_path is None:
        args.feature_model_path = os.path.join(args.output_dir, 'models', 'feature_extractor.pkl')
    if args.model_path is None:
        args.model_path = os.path.join(args.output_dir, 'models', 'detector.pkl')
    
    if not os.path.exists(args.feature_model_path) or not os.path.exists(args.model_path):
        print("错误: 模型文件不存在，请先运行训练模式")
        return
    
    # 加载模型
    print("加载模型...")
    feature_extractor = create_feature_extractor(args)
    feature_extractor.load_model(args.feature_model_path)
    
    detector = create_detector(feature_extractor, args)
    detector.load_model(args.model_path)
    
    # 初始化数据加载器
    data_loader = BCCDDataLoader(args.dataset_path)
    
    # 加载测试数据 - 使用全部val.txt
    test_images = data_loader.load_image_list('val')
    
    print(f"在 {len(test_images)} 张图像上进行评估...")
    
    # 加载测试数据和标注
    test_images_data = []
    test_annotations_data = []
    
    for image_name in tqdm(test_images, desc="加载数据"):
        image, annotations = data_loader.parse_annotation(image_name)
        test_images_data.append(image)
        test_annotations_data.append(annotations)
    
    # 进行评估
    print("进行检测和评估...")
    start_time = time.time()
    
    eval_result = detector.evaluate(
        test_images_data,
        test_annotations_data,
        iou_threshold=args.iou_threshold
    )
    
    eval_time = time.time() - start_time
    
    print("评估结果:")
    print(f"准确率: {eval_result['accuracy']:.4f}")
    print(f"精确率: {eval_result['precision']:.4f}")
    print(f"召回率: {eval_result.get('recall', 0):.4f}")
    print(f"F1分数: {eval_result.get('f1_score', 0):.4f}")
    print(f"预测数量: {eval_result['num_predictions']}")
    print(f"匹配数量: {eval_result['num_matches']}")
    print(f"真实目标数量: {eval_result.get('num_ground_truth', 0)}")
    print(f"评估时间: {eval_time:.2f} 秒")
    
    # 对前几张图像进行检测用于可视化
    print("生成检测结果用于可视化...")
    vis_count = min(args.num_vis_images, len(test_images_data))
    vis_images = test_images_data[:vis_count]
    vis_annotations = test_annotations_data[:vis_count]
    vis_predictions = []
    
    for image in vis_images:
        detections = detector.detect(
            image,
            confidence_threshold=args.confidence_threshold,
            nms_threshold=args.nms_threshold
        )
        vis_predictions.append(detections)
    
    # 可视化评估结果
    if args.visualize:
        print("可视化评估结果...")
        
        visualizer = DetectionVisualizer()
        
        # 绘制检测结果对比
        visualizer.plot_detection_results(
            vis_images,
            vis_predictions,
            vis_annotations,
            num_images=vis_count,
            save_path=os.path.join(args.output_dir, 'visualizations', 'evaluation_comparison.png')
        )
        
        # 绘制评估指标图表
        print("生成评估指标图表...")
        
        # 准备指标数据
        metrics_data = {
            'accuracy': eval_result['accuracy'],
            'precision': eval_result['precision'],
            'recall': eval_result.get('recall', 0),
            'f1_score': eval_result.get('f1_score', 0)
        }
        
        # 绘制指标柱状图
        import matplotlib.pyplot as plt
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 指标对比图
        metrics = list(metrics_data.keys())
        values = list(metrics_data.values())
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        
        bars = ax1.bar(metrics, values, color=colors, alpha=0.8)
        ax1.set_title('评估指标对比', fontsize=14, fontweight='bold')
        ax1.set_ylabel('得分', fontsize=12)
        ax1.set_ylim(0, 1)
        
        # 在柱状图上添加数值标签
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=11)
        
        ax1.grid(True, alpha=0.3)
        
        # 2. 检测数量统计
        detection_stats = {
            '预测数量': eval_result['num_predictions'],
            '匹配数量': eval_result['num_matches'],
            '真实目标': eval_result.get('num_ground_truth', 0)
        }
        
        stats_names = list(detection_stats.keys())
        stats_values = list(detection_stats.values())
        
        bars2 = ax2.bar(stats_names, stats_values, color=['#FF9F43', '#10AC84', '#5F27CD'], alpha=0.8)
        ax2.set_title('检测数量统计', fontsize=14, fontweight='bold')
        ax2.set_ylabel('数量', fontsize=12)
        
        for bar, value in zip(bars2, stats_values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + max(stats_values)*0.01,
                    f'{value}', ha='center', va='bottom', fontsize=11)
        
        ax2.grid(True, alpha=0.3)
        
        # 3. 各类别检测结果（如果有详细信息）
        class_info = data_loader.get_class_info()
        class_names = list(class_info['id_to_class'].values())
        
        # 统计各类别的检测结果
        class_predictions = {name: 0 for name in class_names}
        class_ground_truth = {name: 0 for name in class_names}
        
        # 统计预测结果
        for predictions in vis_predictions:
            for pred in predictions:
                class_name = data_loader.id_to_class[pred['class_id']]
                class_predictions[class_name] += 1
        
        # 统计真实标注
        for annotations in vis_annotations:
            for ann in annotations:
                class_name = data_loader.id_to_class[ann['class_id']]
                class_ground_truth[class_name] += 1
        
        x = np.arange(len(class_names))
        width = 0.35
        
        pred_counts = [class_predictions[name] for name in class_names]
        gt_counts = [class_ground_truth[name] for name in class_names]
        
        bars3 = ax3.bar(x - width/2, pred_counts, width, label='预测数量', color='#FF6B6B', alpha=0.8)
        bars4 = ax3.bar(x + width/2, gt_counts, width, label='真实数量', color='#4ECDC4', alpha=0.8)
        
        ax3.set_title('各类别检测对比（样例图像）', fontsize=14, fontweight='bold')
        ax3.set_ylabel('数量', fontsize=12)
        ax3.set_xlabel('类别', fontsize=12)
        ax3.set_xticks(x)
        ax3.set_xticklabels(class_names)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 保存图表
        metrics_path = os.path.join(args.output_dir, 'visualizations', 'evaluation_metrics.png')
        plt.savefig(metrics_path, dpi=300, bbox_inches='tight')
        print(f"评估指标图表已保存到: {metrics_path}")
        
        plt.show()
    
    # 保存详细的评估结果
    eval_result['evaluation_time'] = eval_time
    eval_result['test_images_count'] = len(test_images)
    eval_result['iou_threshold'] = args.iou_threshold
    eval_result['confidence_threshold'] = args.confidence_threshold
    eval_result['nms_threshold'] = args.nms_threshold
    eval_result['class_statistics'] = {
        'predictions': class_predictions,
        'ground_truth': class_ground_truth
    }
    
    eval_path = os.path.join(args.output_dir, 'results', 'evaluation_result.json')
    with open(eval_path, 'w', encoding='utf-8') as f:
        json.dump(eval_result, f, indent=2, ensure_ascii=False)
    
    print(f"评估结果已保存到: {eval_path}")
    print("评估模式完成！")


def demo_mode(args):
    """演示模式"""
    print("=" * 50)
    print("开始演示模式")
    print("=" * 50)
    
    print("演示模式将依次运行训练、预测和评估...")
    
    # 训练
    print("\n1. 开始训练...")
    train_mode(args)
    
    # 预测
    print("\n2. 开始预测...")
    predict_mode(args)
    
    # 评估
    print("\n3. 开始评估...")
    evaluate_mode(args)
    
    print("\n演示模式完成！")


def main():
    """主函数"""
    parser = create_parser()
    args = parser.parse_args()
    
    # 检查数据集路径
    if not os.path.exists(args.dataset_path):
        print(f"错误: 数据集路径不存在: {args.dataset_path}")
        sys.exit(1)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 打印配置信息
    print("配置信息:")
    print(f"  模式: {args.mode}")
    print(f"  数据集路径: {args.dataset_path}")
    print(f"  输出目录: {args.output_dir}")
    print(f"  HOG参数: orientations={args.hog_orientations}, target_size={args.target_size}")
    print(f"  SVM参数: C={args.svm_c}, kernel={args.svm_kernel}")
    print(f"  检测参数: confidence={args.confidence_threshold}, nms={args.nms_threshold}")
    print(f"  IoU阈值: {args.iou_threshold}")
    if args.mode == 'predict' and args.num_predict_images is not None:
        print(f"  预测图像数量: {args.num_predict_images}")
    if args.visualize:
        print(f"  可视化图像数量: {args.num_vis_images}")
    
    # 根据模式执行相应功能
    try:
        if args.mode == 'train':
            train_mode(args)
        elif args.mode == 'predict':
            predict_mode(args)
        elif args.mode == 'evaluate':
            evaluate_mode(args)
        elif args.mode == 'demo':
            demo_mode(args)
    except KeyboardInterrupt:
        print("\n用户中断程序")
    except Exception as e:
        print(f"\n程序执行出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main() 