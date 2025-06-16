import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from typing import List, Tuple, Dict, Optional
import pickle
from feature_extractor import HOGFeatureExtractor


class SelectiveSearch:
    """选择性搜索候选区域生成器"""
    
    def __init__(self, 
                 strategy: str = 'fast',
                 min_size: int = 20,
                 max_proposals: int = 2000):
        """
        初始化选择性搜索
        
        Args:
            strategy: 搜索策略 ('fast', 'quality')
            min_size: 最小区域大小
            max_proposals: 最大候选区域数量
        """
        self.strategy = strategy
        self.min_size = min_size
        self.max_proposals = max_proposals
        
        # 创建选择性搜索对象
        self.ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    
    def generate_proposals(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        生成候选区域
        
        Args:
            image: 输入图像 (RGB格式)
            
        Returns:
            候选区域列表 [(x, y, w, h), ...]
        """
        # 转换为BGR格式 (OpenCV要求)
        if len(image.shape) == 3:
            bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            bgr_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        # 设置输入图像
        self.ss.setBaseImage(bgr_image)
        
        # 选择搜索策略
        if self.strategy == 'fast':
            self.ss.switchToSelectiveSearchFast()
        else:
            self.ss.switchToSelectiveSearchQuality()
        
        # 生成候选区域
        proposals = self.ss.process()
        
        # 过滤小区域
        filtered_proposals = []
        for x, y, w, h in proposals:
            if w >= self.min_size and h >= self.min_size:
                filtered_proposals.append((x, y, w, h))
                
                if len(filtered_proposals) >= self.max_proposals:
                    break
        
        return filtered_proposals


class CellDetector:
    """血细胞检测器"""
    
    def __init__(self, 
                 feature_extractor: HOGFeatureExtractor,
                 svm_params: Dict = None):
        """
        初始化检测器
        
        Args:
            feature_extractor: HOG特征提取器
            svm_params: SVM参数
        """
        self.feature_extractor = feature_extractor
        self.svm_params = svm_params or {
            'C': 1.0,
            'kernel': 'rbf',
            'gamma': 'scale',
            'probability': True
        }
        
        # 分类器
        self.svm_classifier = None
        self.class_info = None
        
        # 选择性搜索
        self.selective_search = SelectiveSearch()
        
    def train(self, 
              positive_regions: List[np.ndarray], 
              positive_labels: List[int],
              negative_regions: List[np.ndarray],
              pca_components: Optional[int] = None,
              variance_ratio: float = 0.95) -> Dict:
        """
        训练SVM分类器
        
        Args:
            positive_regions: 正样本区域列表
            positive_labels: 正样本标签列表
            negative_regions: 负样本区域列表
            pca_components: PCA组件数量
            variance_ratio: 保留的方差比例
            
        Returns:
            训练结果信息
        """
        print("开始训练检测器...")
        
        # 提取正样本特征
        print("提取正样本HOG特征...")
        positive_features = self.feature_extractor.extract_features_batch(positive_regions)
        
        if positive_features.size == 0:
            raise ValueError("无法提取正样本特征")
        
        # 提取负样本特征
        print("提取负样本HOG特征...")
        negative_features = self.feature_extractor.extract_features_batch(negative_regions)
        
        if negative_features.size == 0:
            raise ValueError("无法提取负样本特征")
        
        # 合并特征和标签
        all_features = np.vstack([positive_features, negative_features])
        all_labels = positive_labels + [len(set(positive_labels))] * len(negative_regions)  # 背景类别
        
        print(f"总特征维度: {all_features.shape}")
        print(f"正样本: {len(positive_regions)}, 负样本: {len(negative_regions)}")
        
        # 训练PCA
        print("训练PCA降维器...")
        self.feature_extractor.fit_pca(
            all_features, 
            n_components=pca_components,
            variance_ratio=variance_ratio
        )
        
        # 应用PCA降维
        print("应用PCA降维...")
        reduced_features = self.feature_extractor.transform_features(all_features)
        
        print(f"降维后特征维度: {reduced_features.shape}")
        
        # 训练SVM分类器
        print("训练SVM分类器...")
        self.svm_classifier = OneVsRestClassifier(SVC(**self.svm_params))
        self.svm_classifier.fit(reduced_features, all_labels)
        
        # 存储类别信息
        self.class_info = {
            'positive_classes': list(set(positive_labels)),
            'background_class': len(set(positive_labels)),
            'num_classes': len(set(positive_labels)) + 1
        }
        
        # 计算训练准确率
        train_predictions = self.svm_classifier.predict(reduced_features)
        train_accuracy = np.mean(train_predictions == all_labels)
        
        print(f"训练完成！训练准确率: {train_accuracy:.4f}")
        
        return {
            'train_accuracy': train_accuracy,
            'num_positive': len(positive_regions),
            'num_negative': len(negative_regions),
            'feature_dim_original': all_features.shape[1],
            'feature_dim_reduced': reduced_features.shape[1],
            'class_info': self.class_info
        }
    
    def detect(self, 
               image: np.ndarray,
               confidence_threshold: float = 0.5,
               nms_threshold: float = 0.3) -> List[Dict]:
        """
        在图像中检测目标
        
        Args:
            image: 输入图像
            confidence_threshold: 置信度阈值
            nms_threshold: NMS阈值
            
        Returns:
            检测结果列表
        """
        if self.svm_classifier is None:
            raise ValueError("分类器尚未训练")
        
        # 生成候选区域
        proposals = self.selective_search.generate_proposals(image)
        
        if not proposals:
            return []
        
        # 提取候选区域
        candidate_regions = []
        valid_proposals = []
        
        h, w = image.shape[:2]
        for x, y, region_w, region_h in proposals:
            # 确保区域在图像范围内
            x = max(0, min(x, w-1))
            y = max(0, min(y, h-1))
            x2 = min(w, x + region_w)
            y2 = min(h, y + region_h)
            
            if x2 > x and y2 > y:
                region = image[y:y2, x:x2]
                if region.size > 0:
                    candidate_regions.append(region)
                    valid_proposals.append((x, y, x2-x, y2-y))
        
        if not candidate_regions:
            return []
        
        # 提取特征并分类
        features = self.feature_extractor.extract_and_reduce_features(candidate_regions)
        
        if features.size == 0:
            return []
        
        # 预测类别和置信度
        predictions = self.svm_classifier.predict(features)
        probabilities = self.svm_classifier.predict_proba(features)
        
        # 收集检测结果
        detections = []
        background_class = self.class_info['background_class']
        
        for i, (pred, prob, (x, y, w, h)) in enumerate(zip(predictions, probabilities, valid_proposals)):
            if pred != background_class:  # 非背景类别
                max_prob = np.max(prob)
                if max_prob >= confidence_threshold:
                    detections.append({
                        'bbox': [x, y, x+w, y+h],
                        'class_id': int(pred),
                        'confidence': float(max_prob),
                        'area': w * h
                    })
        
        # 应用非极大值抑制
        if detections:
            detections = self.apply_nms(detections, nms_threshold)
        
        return detections
    
    def apply_nms(self, detections: List[Dict], threshold: float = 0.3) -> List[Dict]:
        """
        应用非极大值抑制
        
        Args:
            detections: 检测结果列表
            threshold: IoU阈值
            
        Returns:
            NMS后的检测结果
        """
        if not detections:
            return []
        
        # 按置信度排序
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        
        # 按类别分别应用NMS
        final_detections = []
        
        # 获取所有类别
        classes = list(set([det['class_id'] for det in detections]))
        
        for class_id in classes:
            class_detections = [det for det in detections if det['class_id'] == class_id]
            
            keep = []
            while class_detections:
                # 选择置信度最高的检测
                current = class_detections.pop(0)
                keep.append(current)
                
                # 计算与其他检测的IoU
                remaining = []
                for det in class_detections:
                    iou = self.calculate_iou(current['bbox'], det['bbox'])
                    if iou < threshold:
                        remaining.append(det)
                
                class_detections = remaining
            
            final_detections.extend(keep)
        
        return final_detections
    
    def calculate_iou(self, box1: List[int], box2: List[int]) -> float:
        """
        计算两个边界框的IoU
        
        Args:
            box1: 边界框1 [x1, y1, x2, y2]
            box2: 边界框2 [x1, y1, x2, y2]
            
        Returns:
            IoU值
        """
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # 计算交集
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # 计算并集
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def evaluate(self, 
                 test_images: List[np.ndarray],
                 test_annotations: List[List[Dict]],
                 iou_threshold: float = 0.5) -> Dict:
        """
        评估检测器性能
        
        Args:
            test_images: 测试图像列表
            test_annotations: 测试标注列表
            iou_threshold: IoU阈值
            
        Returns:
            评估结果
        """
        if self.svm_classifier is None:
            raise ValueError("分类器尚未训练")
        
        total_ground_truth = 0  # 总的真实目标数量
        total_predictions = 0   # 总的预测数量
        total_matches = 0       # 总的匹配数量
        correct_matches = 0     # 类别正确的匹配数量
        
        for image, annotations in zip(test_images, test_annotations):
            # 统计真实目标数量
            total_ground_truth += len(annotations)
            
            # 预测
            predictions = self.detect(image)
            total_predictions += len(predictions)
            
            # 匹配预测和真实标注
            matched_gt = set()
            
            for pred in predictions:
                best_iou = 0
                best_match = None
                
                for i, gt in enumerate(annotations):
                    if i in matched_gt:
                        continue
                    
                    iou = self.calculate_iou(pred['bbox'], gt['bbox'])
                    if iou > best_iou and iou >= iou_threshold:
                        best_iou = iou
                        best_match = i
                
                if best_match is not None:
                    matched_gt.add(best_match)
                    total_matches += 1
                    
                    # 检查类别是否正确
                    if pred['class_id'] == annotations[best_match]['class_id']:
                        correct_matches += 1
        
        # 计算性能指标
        precision = correct_matches / total_predictions if total_predictions > 0 else 0
        recall = total_matches / total_ground_truth if total_ground_truth > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = correct_matches / total_matches if total_matches > 0 else 0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'num_predictions': total_predictions,
            'num_matches': total_matches,
            'num_ground_truth': total_ground_truth,
            'correct_matches': correct_matches
        }
    
    def save_model(self, filepath: str) -> None:
        """
        保存检测器模型
        
        Args:
            filepath: 保存路径
        """
        model_data = {
            'svm_classifier': self.svm_classifier,
            'svm_params': self.svm_params,
            'class_info': self.class_info
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"检测器模型已保存到: {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """
        加载检测器模型
        
        Args:
            filepath: 模型文件路径
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.svm_classifier = model_data['svm_classifier']
        self.svm_params = model_data['svm_params']
        self.class_info = model_data['class_info']
        
        print(f"检测器模型已从{filepath}加载") 