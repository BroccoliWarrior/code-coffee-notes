#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据增强模块
用于血细胞检测的样本均衡和数据增强
"""

import cv2
import numpy as np
import random
from typing import List, Tuple
from collections import Counter


class DataAugmenter:
    """数据增强器"""
    
    def __init__(self, target_size: Tuple[int, int] = (64, 64)):
        """
        初始化数据增强器
        
        Args:
            target_size: 目标图像大小
        """
        self.target_size = target_size
    
    def rotate_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        """
        旋转图像
        
        Args:
            image: 输入图像
            angle: 旋转角度
            
        Returns:
            旋转后的图像
        """
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        # 获取旋转矩阵
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # 应用旋转
        rotated = cv2.warpAffine(image, rotation_matrix, (w, h), 
                                borderMode=cv2.BORDER_REFLECT_101)
        
        return rotated
    
    def flip_image(self, image: np.ndarray, flip_code: int) -> np.ndarray:
        """
        翻转图像
        
        Args:
            image: 输入图像
            flip_code: 翻转代码 (0:垂直, 1:水平, -1:水平+垂直)
            
        Returns:
            翻转后的图像
        """
        return cv2.flip(image, flip_code)
    
    def adjust_brightness(self, image: np.ndarray, factor: float) -> np.ndarray:
        """
        调整亮度
        
        Args:
            image: 输入图像
            factor: 亮度因子 (>1提高亮度, <1降低亮度)
            
        Returns:
            调整亮度后的图像
        """
        # 转换为HSV色彩空间
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        hsv = hsv.astype(np.float32)
        
        # 调整V通道（亮度）
        hsv[:, :, 2] = hsv[:, :, 2] * factor
        hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
        
        # 转换回RGB
        hsv = hsv.astype(np.uint8)
        result = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        return result
    
    def adjust_contrast(self, image: np.ndarray, factor: float) -> np.ndarray:
        """
        调整对比度
        
        Args:
            image: 输入图像
            factor: 对比度因子
            
        Returns:
            调整对比度后的图像
        """
        # 转换为浮点数
        img_float = image.astype(np.float32)
        
        # 调整对比度：new_img = factor * (old_img - 128) + 128
        result = factor * (img_float - 128) + 128
        result = np.clip(result, 0, 255)
        
        return result.astype(np.uint8)
    
    def add_noise(self, image: np.ndarray, noise_factor: float = 0.1) -> np.ndarray:
        """
        添加高斯噪声
        
        Args:
            image: 输入图像
            noise_factor: 噪声强度
            
        Returns:
            添加噪声后的图像
        """
        # 生成高斯噪声
        noise = np.random.normal(0, noise_factor * 255, image.shape)
        
        # 添加噪声
        noisy_image = image.astype(np.float32) + noise
        noisy_image = np.clip(noisy_image, 0, 255)
        
        return noisy_image.astype(np.uint8)
    
    def elastic_transform(self, image: np.ndarray, alpha: float = 1, sigma: float = 50) -> np.ndarray:
        """
        弹性变形
        
        Args:
            image: 输入图像
            alpha: 变形强度
            sigma: 平滑参数
            
        Returns:
            变形后的图像
        """
        h, w = image.shape[:2]
        
        # 生成随机位移场
        dx = cv2.GaussianBlur((np.random.rand(h, w) * 2 - 1), (0, 0), sigma) * alpha
        dy = cv2.GaussianBlur((np.random.rand(h, w) * 2 - 1), (0, 0), sigma) * alpha
        
        # 创建网格
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        x_new = (x + dx).astype(np.float32)
        y_new = (y + dy).astype(np.float32)
        
        # 应用变形
        result = cv2.remap(image, x_new, y_new, cv2.INTER_LINEAR, 
                          borderMode=cv2.BORDER_REFLECT_101)
        
        return result
    
    def augment_image(self, image: np.ndarray, augmentation_type: str = 'random') -> np.ndarray:
        """
        对单张图像进行增强
        
        Args:
            image: 输入图像
            augmentation_type: 增强类型
            
        Returns:
            增强后的图像
        """
        if augmentation_type == 'random':
            # 随机选择增强方法
            methods = ['rotate', 'flip', 'brightness', 'contrast', 'noise', 'elastic']
            method = random.choice(methods)
        else:
            method = augmentation_type
        
        if method == 'rotate':
            angle = random.uniform(-30, 30)
            return self.rotate_image(image, angle)
        
        elif method == 'flip':
            flip_code = random.choice([0, 1])
            return self.flip_image(image, flip_code)
        
        elif method == 'brightness':
            factor = random.uniform(0.7, 1.3)
            return self.adjust_brightness(image, factor)
        
        elif method == 'contrast':
            factor = random.uniform(0.8, 1.2)
            return self.adjust_contrast(image, factor)
        
        elif method == 'noise':
            noise_factor = random.uniform(0.05, 0.15)
            return self.add_noise(image, noise_factor)
        
        elif method == 'elastic':
            alpha = random.uniform(0.5, 2.0)
            sigma = random.uniform(30, 70)
            return self.elastic_transform(image, alpha, sigma)
        
        else:
            return image
    
    def balance_samples(self, positive_regions: List[np.ndarray], 
                       positive_labels: List[int],
                       target_samples_per_class: int = None) -> Tuple[List[np.ndarray], List[int]]:
        """
        平衡样本数量
        
        Args:
            positive_regions: 正样本区域列表
            positive_labels: 正样本标签列表
            target_samples_per_class: 每个类别的目标样本数量
            
        Returns:
            平衡后的样本和标签
        """
        # 统计各类别样本数量
        label_counts = Counter(positive_labels)
        print("Original sample distribution:")
        for label, count in sorted(label_counts.items()):
            print(f"  Class {label}: {count} samples")
        
        # 确定目标样本数量
        if target_samples_per_class is None:
            target_samples_per_class = max(label_counts.values())
        
        print(f"\nTarget samples per class: {target_samples_per_class}")
        
        # 按类别组织样本
        samples_by_class = {}
        for region, label in zip(positive_regions, positive_labels):
            if label not in samples_by_class:
                samples_by_class[label] = []
            samples_by_class[label].append(region)
        
        # 平衡样本
        balanced_regions = []
        balanced_labels = []
        
        for class_id, samples in samples_by_class.items():
            current_count = len(samples)
            needed_count = target_samples_per_class - current_count
            
            # 添加原始样本
            balanced_regions.extend(samples)
            balanced_labels.extend([class_id] * current_count)
            
            # 如果需要更多样本，进行数据增强
            if needed_count > 0:
                print(f"Generating {needed_count} augmented samples for class {class_id}")
                
                augmented_samples = []
                for i in range(needed_count):
                    # 随机选择一个原始样本
                    source_sample = random.choice(samples)
                    
                    # 应用数据增强
                    augmented_sample = self.augment_image(source_sample, 'random')
                    augmented_samples.append(augmented_sample)
                
                balanced_regions.extend(augmented_samples)
                balanced_labels.extend([class_id] * needed_count)
        
        # 打印平衡后的分布
        final_counts = Counter(balanced_labels)
        print("\nBalanced sample distribution:")
        for label, count in sorted(final_counts.items()):
            print(f"  Class {label}: {count} samples")
        
        return balanced_regions, balanced_labels
    
    def augment_batch(self, images: List[np.ndarray], 
                     num_augmentations: int = 1) -> List[np.ndarray]:
        """
        批量增强图像
        
        Args:
            images: 图像列表
            num_augmentations: 每张图像的增强数量
            
        Returns:
            增强后的图像列表
        """
        augmented_images = []
        
        for image in images:
            # 添加原始图像
            augmented_images.append(image)
            
            # 添加增强图像
            for _ in range(num_augmentations):
                augmented_image = self.augment_image(image, 'random')
                augmented_images.append(augmented_image)
        
        return augmented_images
    
    def create_augmentation_preview(self, image: np.ndarray, 
                                   save_path: str = None) -> np.ndarray:
        """
        创建数据增强预览图
        
        Args:
            image: 输入图像
            save_path: 保存路径
            
        Returns:
            预览图像
        """
        import matplotlib.pyplot as plt
        
        # 应用不同的增强方法
        augmentations = {
            'Original': image,
            'Rotated': self.rotate_image(image, 20),
            'Flipped': self.flip_image(image, 1),
            'Brightness+': self.adjust_brightness(image, 1.3),
            'Brightness-': self.adjust_brightness(image, 0.7),
            'Contrast+': self.adjust_contrast(image, 1.2),
            'Noise': self.add_noise(image, 0.1),
            'Elastic': self.elastic_transform(image, 1.0, 50)
        }
        
        # 创建预览图
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
        
        for i, (name, aug_image) in enumerate(augmentations.items()):
            axes[i].imshow(aug_image)
            axes[i].set_title(name)
            axes[i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Augmentation preview saved to: {save_path}")
        
        plt.show()
        
        return fig 