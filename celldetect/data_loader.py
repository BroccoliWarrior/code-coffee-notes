import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from typing import List, Tuple, Dict


class BCCDDataLoader:
    """BCCD数据集加载器"""
    
    def __init__(self, dataset_path: str):
        """
        初始化数据加载器
        
        Args:
            dataset_path: BCCD数据集根目录路径
        """
        self.dataset_path = dataset_path
        self.images_path = os.path.join(dataset_path, 'JPEGImages')
        self.annotations_path = os.path.join(dataset_path, 'Annotations')
        self.imagesets_path = os.path.join(dataset_path, 'ImageSets', 'Main')
        
        # 类别映射
        self.class_to_id = {'RBC': 0, 'WBC': 1, 'Platelets': 2}
        self.id_to_class = {v: k for k, v in self.class_to_id.items()}
    
    def load_image_list(self, split: str = 'train') -> List[str]:
        """
        加载指定数据集分割的图像列表
        
        Args:
            split: 数据集分割 ('train', 'val', 'test')
            
        Returns:
            图像文件名列表
        """
        split_file = os.path.join(self.imagesets_path, f'{split}.txt')
        if not os.path.exists(split_file):
            print(f"警告: {split_file} 不存在，使用所有可用图像")
            # 如果split文件不存在，返回所有图像
            images = []
            for f in os.listdir(self.images_path):
                if f.endswith('.jpg'):
                    images.append(f[:-4])  # 去掉.jpg扩展名
            return images
        
        with open(split_file, 'r') as f:
            image_names = [line.strip() for line in f.readlines()]
        return image_names
    
    def parse_annotation(self, image_name: str) -> Tuple[np.ndarray, List[Dict]]:
        """
        解析单个图像的XML标注文件
        
        Args:
            image_name: 图像文件名（不含扩展名）
            
        Returns:
            图像数组和标注信息列表
        """
        # 加载图像
        image_path = os.path.join(self.images_path, f'{image_name}.jpg')
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 解析XML文件
        xml_path = os.path.join(self.annotations_path, f'{image_name}.xml')
        
        annotations = []
        if os.path.exists(xml_path):
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            for obj in root.findall('object'):
                # 获取类别名称
                class_name = obj.find('n').text if obj.find('n') is not None else obj.find('name').text
                
                # 获取边界框坐标
                bbox = obj.find('bndbox')
                xmin = int(bbox.find('xmin').text)
                ymin = int(bbox.find('ymin').text)
                xmax = int(bbox.find('xmax').text)
                ymax = int(bbox.find('ymax').text)
                
                # 获取类别ID
                class_id = self.class_to_id.get(class_name, -1)
                
                if class_id != -1:  # 只保留已知类别
                    annotations.append({
                        'class_name': class_name,
                        'class_id': class_id,
                        'bbox': [xmin, ymin, xmax, ymax],
                        'area': (xmax - xmin) * (ymax - ymin)
                    })
        
        return image, annotations
    
    def extract_positive_samples(self, image_names: List[str]) -> Tuple[List[np.ndarray], List[int]]:
        """
        从标注中提取正样本区域
        
        Args:
            image_names: 图像文件名列表
            
        Returns:
            正样本区域列表和对应的类别标签
        """
        positive_regions = []
        labels = []
        
        for image_name in image_names:
            image, annotations = self.parse_annotation(image_name)
            
            for ann in annotations:
                xmin, ymin, xmax, ymax = ann['bbox']
                # 确保边界框在图像范围内
                h, w = image.shape[:2]
                xmin = max(0, min(xmin, w-1))
                ymin = max(0, min(ymin, h-1))
                xmax = max(xmin+1, min(xmax, w))
                ymax = max(ymin+1, min(ymax, h))
                
                # 提取区域
                region = image[ymin:ymax, xmin:xmax]
                if region.size > 0:  # 确保区域不为空
                    positive_regions.append(region)
                    labels.append(ann['class_id'])
        
        return positive_regions, labels
    
    def generate_negative_samples(self, image_names: List[str], num_samples: int = 1000, 
                                min_size: int = 20, max_size: int = 200) -> List[np.ndarray]:
        """
        生成负样本（随机裁剪的背景区域）
        
        Args:
            image_names: 图像文件名列表
            num_samples: 生成的负样本数量
            min_size: 最小区域大小
            max_size: 最大区域大小
            
        Returns:
            负样本区域列表
        """
        negative_regions = []
        samples_per_image = max(1, num_samples // len(image_names))
        
        for image_name in image_names:
            image, annotations = self.parse_annotation(image_name)
            h, w = image.shape[:2]
            
            # 创建正样本区域的掩码
            positive_mask = np.zeros((h, w), dtype=bool)
            for ann in annotations:
                xmin, ymin, xmax, ymax = ann['bbox']
                xmin = max(0, min(xmin, w-1))
                ymin = max(0, min(ymin, h-1))
                xmax = max(xmin+1, min(xmax, w))
                ymax = max(ymin+1, min(ymax, h))
                positive_mask[ymin:ymax, xmin:xmax] = True
            
            # 随机采样负样本
            for _ in range(samples_per_image):
                attempts = 0
                while attempts < 50:  # 最多尝试50次
                    # 随机生成区域大小
                    region_w = np.random.randint(min_size, min(max_size, w))
                    region_h = np.random.randint(min_size, min(max_size, h))
                    
                    # 随机生成位置
                    x = np.random.randint(0, max(1, w - region_w))
                    y = np.random.randint(0, max(1, h - region_h))
                    
                    # 检查是否与正样本重叠
                    overlap = np.sum(positive_mask[y:y+region_h, x:x+region_w])
                    overlap_ratio = overlap / (region_w * region_h)
                    
                    if overlap_ratio < 0.3:  # 重叠率小于30%才认为是负样本
                        region = image[y:y+region_h, x:x+region_w]
                        negative_regions.append(region)
                        break
                    
                    attempts += 1
        
        return negative_regions[:num_samples]
    
    def get_class_info(self) -> Dict:
        """获取类别信息"""
        return {
            'class_to_id': self.class_to_id,
            'id_to_class': self.id_to_class,
            'num_classes': len(self.class_to_id)
        } 