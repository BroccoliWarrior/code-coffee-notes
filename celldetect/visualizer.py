import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Dict, Tuple
import os

# 设置matplotlib字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Liberation Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


class DetectionVisualizer:
    """检测结果可视化器"""
    
    def __init__(self):
        """初始化可视化器"""
        # 类别颜色映射（BGR格式，用于OpenCV）
        self.class_colors = {
            0: (0, 255, 0),    # RBC - 绿色
            1: (0, 0, 255),    # WBC - 红色
            2: (255, 0, 0),    # Platelets - 蓝色
        }
        
        # 类别名称映射（英文，简洁）
        self.class_names = {
            0: 'RBC',
            1: 'WBC', 
            2: 'Platelets'
        }
    
    def draw_detections(self, 
                       image: np.ndarray, 
                       detections: List[Dict],
                       show_confidence: bool = False,
                       line_thickness: int = 2) -> np.ndarray:
        """
        在图像上绘制检测结果
        
        Args:
            image: 输入图像（RGB格式）
            detections: 检测结果列表
            show_confidence: 是否显示置信度
            line_thickness: 边框线条粗细
            
        Returns:
            绘制了检测框的图像
        """
        # 复制图像避免修改原图
        result_image = image.copy()
        
        # 如果输入是RGB格式，转换为BGR用于OpenCV绘制
        if len(result_image.shape) == 3:
            is_rgb = True
            draw_image = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
        else:
            is_rgb = False
            draw_image = result_image.copy()
        
        for detection in detections:
            bbox = detection['bbox']
            class_id = detection['class_id']
            confidence = detection['confidence']
            
            # 获取颜色和类别名称
            color = self.class_colors.get(class_id, (128, 128, 128))
            class_name = self.class_names.get(class_id, f'Class_{class_id}')
            
            # 绘制边界框
            x1, y1, x2, y2 = bbox
            cv2.rectangle(draw_image, (x1, y1), (x2, y2), color, line_thickness)
            
            # 准备标签文本
            if show_confidence:
                label = f'{class_name}:{confidence:.2f}'
            else:
                label = class_name
            
            # 文本参数
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            text_thickness = 2
            
            # 获取文本尺寸
            (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, text_thickness)
            
            # 计算文本位置（在边界框左上角附近）
            text_x = x1
            text_y = y1 - 10  # 在边界框上方
            
            # 如果文本会超出图像顶部，放在边界框内部
            if text_y - text_height < 0:
                text_y = y1 + text_height + 10
            
            # 绘制文本（白色文字，清晰可见）
            cv2.putText(draw_image, label, 
                       (text_x, text_y),
                       font, font_scale, (255, 255, 255), text_thickness)
        
        # 如果原图是RGB格式，转换回RGB
        if is_rgb:
            result_image = cv2.cvtColor(draw_image, cv2.COLOR_BGR2RGB)
        else:
            result_image = draw_image
            
        return result_image
    
    def draw_ground_truth(self,
                         image: np.ndarray,
                         annotations: List[Dict],
                         line_thickness: int = 2) -> np.ndarray:
        """
        在图像上绘制真实标注（使用虚线边框）
        
        Args:
            image: 输入图像
            annotations: 真实标注列表
            line_thickness: 边框线条粗细
            
        Returns:
            绘制了真实标注框的图像
        """
        result_image = image.copy()
        
        # 如果输入是RGB格式，转换为BGR用于OpenCV绘制
        if len(result_image.shape) == 3:
            is_rgb = True
            draw_image = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
        else:
            is_rgb = False
            draw_image = result_image.copy()
        
        for annotation in annotations:
            bbox = annotation['bbox']
            class_id = annotation['class_id']
            
            # 获取颜色和类别名称
            color = self.class_colors.get(class_id, (128, 128, 128))
            class_name = self.class_names.get(class_id, f'Class_{class_id}')
            
            # 绘制边界框（虚线效果）
            x1, y1, x2, y2 = bbox
            self._draw_dashed_rectangle(draw_image, (x1, y1), (x2, y2), color, line_thickness)
            
            # 绘制标签（在边界框下方）
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            text_thickness = 2
            
            # 获取文本尺寸
            (text_width, text_height), baseline = cv2.getTextSize(class_name, font, font_scale, text_thickness)
            
            # 计算文本位置（在边界框下方）
            text_x = x1
            text_y = y2 + text_height + 10
            
            # 如果文本会超出图像底部，放在边界框内部
            if text_y > draw_image.shape[0]:
                text_y = y2 - 10
            
            # 绘制文本
            cv2.putText(draw_image, f'GT:{class_name}',
                       (text_x, text_y),
                       font, font_scale, (255, 255, 255), text_thickness)
        
        # 如果原图是RGB格式，转换回RGB
        if is_rgb:
            result_image = cv2.cvtColor(draw_image, cv2.COLOR_BGR2RGB)
        else:
            result_image = draw_image
            
        return result_image
    
    def _draw_dashed_rectangle(self, image, pt1, pt2, color, thickness):
        """绘制虚线矩形"""
        x1, y1 = pt1
        x2, y2 = pt2
        
        dash_length = 10
        gap_length = 5
        
        # 上边
        self._draw_dashed_line(image, (x1, y1), (x2, y1), color, thickness, dash_length, gap_length)
        # 右边
        self._draw_dashed_line(image, (x2, y1), (x2, y2), color, thickness, dash_length, gap_length)
        # 下边
        self._draw_dashed_line(image, (x2, y2), (x1, y2), color, thickness, dash_length, gap_length)
        # 左边
        self._draw_dashed_line(image, (x1, y2), (x1, y1), color, thickness, dash_length, gap_length)
    
    def _draw_dashed_line(self, image, pt1, pt2, color, thickness, dash_length, gap_length):
        """绘制虚线"""
        x1, y1 = pt1
        x2, y2 = pt2
        
        line_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        if line_length == 0:
            return
        
        dx = (x2 - x1) / line_length
        dy = (y2 - y1) / line_length
        
        current_length = 0
        is_dash = True
        
        while current_length < line_length:
            if is_dash:
                next_length = min(current_length + dash_length, line_length)
            else:
                next_length = min(current_length + gap_length, line_length)
            
            start_x = int(x1 + dx * current_length)
            start_y = int(y1 + dy * current_length)
            end_x = int(x1 + dx * next_length)
            end_y = int(y1 + dy * next_length)
            
            if is_dash:
                cv2.line(image, (start_x, start_y), (end_x, end_y), color, thickness)
            
            current_length = next_length
            is_dash = not is_dash
    
    def compare_detection_and_gt(self,
                                image: np.ndarray,
                                detections: List[Dict],
                                ground_truths: List[Dict]) -> np.ndarray:
        """
        比较检测结果和真实标注
        
        Args:
            image: 输入图像
            detections: 检测结果
            ground_truths: 真实标注
            
        Returns:
            比较结果图像
        """
        # 先绘制真实标注（虚线）
        result_image = self.draw_ground_truth(image, ground_truths)
        
        # 再绘制检测结果（实线）
        result_image = self.draw_detections(result_image, detections)
        
        return result_image
    
    def plot_training_samples(self,
                             positive_samples: List[np.ndarray],
                             negative_samples: List[np.ndarray],
                             labels: List[int],
                             num_samples_per_class: int = 5,
                             save_path: str = None) -> None:
        """
        可视化训练样本
        
        Args:
            positive_samples: 正样本列表
            negative_samples: 负样本列表
            labels: 正样本标签
            num_samples_per_class: 每类显示的样本数量
            save_path: 保存路径
        """
        # 获取唯一类别
        unique_classes = list(set(labels))
        num_classes = len(unique_classes)
        
        # 创建子图
        fig, axes = plt.subplots(num_classes + 1, num_samples_per_class, 
                               figsize=(15, 3 * (num_classes + 1)))
        
        if num_classes + 1 == 1:
            axes = axes.reshape(1, -1)
        elif num_samples_per_class == 1:
            axes = axes.reshape(-1, 1)
        
        # 显示正样本
        for class_idx, class_id in enumerate(unique_classes):
            class_samples = [positive_samples[i] for i, label in enumerate(labels) if label == class_id]
            class_name = self.class_names.get(class_id, f'Class_{class_id}')
            
            for sample_idx in range(min(num_samples_per_class, len(class_samples))):
                ax = axes[class_idx, sample_idx]
                ax.imshow(class_samples[sample_idx])
                ax.set_title(f'{class_name}')
                ax.axis('off')
            
            # 填充空白子图
            for sample_idx in range(len(class_samples), num_samples_per_class):
                ax = axes[class_idx, sample_idx]
                ax.axis('off')
        
        # 显示负样本
        for sample_idx in range(min(num_samples_per_class, len(negative_samples))):
            ax = axes[num_classes, sample_idx]
            ax.imshow(negative_samples[sample_idx])
            ax.set_title('Background')
            ax.axis('off')
        
        # 填充负样本空白子图
        for sample_idx in range(len(negative_samples), num_samples_per_class):
            ax = axes[num_classes, sample_idx]
            ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Training samples visualization saved to: {save_path}")
        
        plt.show()
    
    def plot_detection_results(self,
                              images: List[np.ndarray],
                              detections_list: List[List[Dict]],
                              ground_truths_list: List[List[Dict]] = None,
                              num_images: int = 4,
                              save_path: str = None) -> None:
        """
        可视化检测结果
        
        Args:
            images: 图像列表
            detections_list: 检测结果列表
            ground_truths_list: 真实标注列表（可选）
            num_images: 显示的图像数量
            save_path: 保存路径
        """
        num_images = min(num_images, len(images))
        
        if ground_truths_list is not None:
            fig, axes = plt.subplots(2, num_images, figsize=(4 * num_images, 8))
            if num_images == 1:
                axes = axes.reshape(-1, 1)
        else:
            fig, axes = plt.subplots(1, num_images, figsize=(4 * num_images, 4))
            if num_images == 1:
                axes = [axes]
        
        for i in range(num_images):
            image = images[i]
            detections = detections_list[i]
            
            # 生成图片编号（5位数字，前面补0）
            image_id = f"{i:05d}"
            
            if ground_truths_list is not None:
                ground_truths = ground_truths_list[i]
                
                # 显示检测结果
                result_image = self.draw_detections(image, detections)
                if num_images == 1:
                    axes[0].imshow(result_image)
                    axes[0].set_title(f'Detection: {image_id} ({len(detections)} objects)')
                    axes[0].axis('off')
                else:
                    axes[0, i].imshow(result_image)
                    axes[0, i].set_title(f'Detection: {image_id} ({len(detections)} objects)')
                    axes[0, i].axis('off')
                
                # 显示比较结果
                comparison_image = self.compare_detection_and_gt(image, detections, ground_truths)
                if num_images == 1:
                    axes[1].imshow(comparison_image)
                    axes[1].set_title(f'Comparison: {image_id}')
                    axes[1].axis('off')
                else:
                    axes[1, i].imshow(comparison_image)
                    axes[1, i].set_title(f'Comparison: {image_id}')
                    axes[1, i].axis('off')
            else:
                # 只显示检测结果
                result_image = self.draw_detections(image, detections)
                if num_images == 1:
                    axes.imshow(result_image)
                    axes.set_title(f'Detection: {image_id} ({len(detections)} objects)')
                    axes.axis('off')
                else:
                    axes[i].imshow(result_image)
                    axes[i].set_title(f'Detection: {image_id} ({len(detections)} objects)')
                    axes[i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Detection visualization saved to: {save_path}")
        
        plt.show()
    
    def save_detection_image(self,
                            image: np.ndarray,
                            detections: List[Dict],
                            save_path: str) -> None:
        """
        保存检测结果图像
        
        Args:
            image: 输入图像
            detections: 检测结果
            save_path: 保存路径
        """
        result_image = self.draw_detections(image, detections)
        
        # 转换为BGR格式用于保存
        if len(result_image.shape) == 3:
            bgr_image = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
        else:
            bgr_image = result_image
        
        cv2.imwrite(save_path, bgr_image)
        print(f"Detection result image saved to: {save_path}")
    
    def create_detection_summary(self,
                                detections_list: List[List[Dict]],
                                class_names: Dict[int, str] = None) -> Dict:
        """
        创建检测结果摘要
        
        Args:
            detections_list: 检测结果列表
            class_names: 类别名称映射
            
        Returns:
            检测摘要信息
        """
        if class_names is None:
            class_names = self.class_names
        
        total_detections = 0
        class_counts = {}
        confidence_scores = []
        
        for detections in detections_list:
            total_detections += len(detections)
            
            for detection in detections:
                class_id = detection['class_id']
                confidence = detection['confidence']
                
                class_name = class_names.get(class_id, f'Class_{class_id}')
                
                if class_name not in class_counts:
                    class_counts[class_name] = 0
                class_counts[class_name] += 1
                
                confidence_scores.append(confidence)
        
        summary = {
            'total_detections': total_detections,
            'total_images': len(detections_list),
            'average_detections_per_image': total_detections / len(detections_list) if detections_list else 0,
            'class_counts': class_counts,
            'average_confidence': np.mean(confidence_scores) if confidence_scores else 0,
            'min_confidence': np.min(confidence_scores) if confidence_scores else 0,
            'max_confidence': np.max(confidence_scores) if confidence_scores else 0
        }
        
        return summary 