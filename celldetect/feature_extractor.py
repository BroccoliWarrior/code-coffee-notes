import cv2
import numpy as np
from skimage.feature import hog
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple, Optional
import pickle


class HOGFeatureExtractor:
    """HOG特征提取器"""
    
    def __init__(self, 
                 orientations: int = 9,
                 pixels_per_cell: Tuple[int, int] = (8, 8),
                 cells_per_block: Tuple[int, int] = (2, 2),
                 block_norm: str = 'L2-Hys',
                 feature_vector: bool = True,
                 transform_sqrt: bool = False,
                 target_size: Tuple[int, int] = (64, 64)):
        """
        初始化HOG特征提取器
        
        Args:
            orientations: 梯度方向的数量
            pixels_per_cell: 每个cell的像素数量
            cells_per_block: 每个block的cell数量
            block_norm: 块归一化方法
            feature_vector: 是否返回特征向量
            transform_sqrt: 是否应用gamma校正
            target_size: 目标图像大小
        """
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.block_norm = block_norm
        self.feature_vector = feature_vector
        self.transform_sqrt = transform_sqrt
        self.target_size = target_size
        
        # PCA和标准化器
        self.pca = None
        self.scaler = None
        self.pca_components = None
        
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        预处理图像
        
        Args:
            image: 输入图像
            
        Returns:
            预处理后的图像
        """
        # 转换为灰度图像
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # 调整大小
        resized = cv2.resize(gray, self.target_size, interpolation=cv2.INTER_AREA)
        
        # 直方图均衡化
        equalized = cv2.equalizeHist(resized)
        
        return equalized
    
    def extract_hog_features(self, image: np.ndarray) -> np.ndarray:
        """
        提取单个图像的HOG特征
        
        Args:
            image: 输入图像
            
        Returns:
            HOG特征向量
        """
        # 预处理图像
        processed_image = self.preprocess_image(image)
        
        # 提取HOG特征
        features = hog(
            processed_image,
            orientations=self.orientations,
            pixels_per_cell=self.pixels_per_cell,
            cells_per_block=self.cells_per_block,
            block_norm=self.block_norm,
            feature_vector=self.feature_vector,
            transform_sqrt=self.transform_sqrt
        )
        
        return features
    
    def extract_features_batch(self, images: List[np.ndarray]) -> np.ndarray:
        """
        批量提取HOG特征
        
        Args:
            images: 图像列表
            
        Returns:
            特征矩阵 (n_samples, n_features)
        """
        features_list = []
        
        for image in images:
            try:
                features = self.extract_hog_features(image)
                features_list.append(features)
            except Exception as e:
                print(f"提取特征时出错: {e}")
                continue
        
        if not features_list:
            return np.array([])
        
        return np.array(features_list)
    
    def fit_pca(self, features: np.ndarray, n_components: Optional[int] = None, 
                variance_ratio: float = 0.95) -> None:
        """
        训练PCA降维器
        
        Args:
            features: 特征矩阵
            n_components: PCA组件数量，如果为None则根据方差比例确定
            variance_ratio: 保留的方差比例
        """
        # 标准化特征
        self.scaler = StandardScaler()
        scaled_features = self.scaler.fit_transform(features)
        
        # 如果未指定组件数，根据方差比例确定
        if n_components is None:
            # 先用所有组件训练PCA来确定合适的组件数
            temp_pca = PCA()
            temp_pca.fit(scaled_features)
            
            # 计算累积方差比例
            cumsum_ratio = np.cumsum(temp_pca.explained_variance_ratio_)
            n_components = np.argmax(cumsum_ratio >= variance_ratio) + 1
            
            print(f"根据{variance_ratio*100}%方差比例，选择{n_components}个PCA组件")
        
        # 训练最终的PCA
        self.pca = PCA(n_components=n_components)
        self.pca.fit(scaled_features)
        self.pca_components = n_components
        
        print(f"PCA训练完成，特征维度从{features.shape[1]}降至{n_components}")
        print(f"解释方差比例: {np.sum(self.pca.explained_variance_ratio_):.4f}")
    
    def transform_features(self, features: np.ndarray) -> np.ndarray:
        """
        应用PCA降维
        
        Args:
            features: 输入特征矩阵
            
        Returns:
            降维后的特征矩阵
        """
        if self.scaler is None or self.pca is None:
            raise ValueError("PCA尚未训练，请先调用fit_pca方法")
        
        # 标准化
        scaled_features = self.scaler.transform(features)
        
        # PCA降维
        reduced_features = self.pca.transform(scaled_features)
        
        return reduced_features
    
    def extract_and_reduce_features(self, images: List[np.ndarray]) -> np.ndarray:
        """
        提取特征并进行降维
        
        Args:
            images: 图像列表
            
        Returns:
            降维后的特征矩阵
        """
        # 提取HOG特征
        features = self.extract_features_batch(images)
        
        if features.size == 0:
            return np.array([])
        
        # 如果PCA已训练，则进行降维
        if self.pca is not None:
            features = self.transform_features(features)
        
        return features
    
    def save_model(self, filepath: str) -> None:
        """
        保存特征提取器模型
        
        Args:
            filepath: 保存路径
        """
        model_data = {
            'orientations': self.orientations,
            'pixels_per_cell': self.pixels_per_cell,
            'cells_per_block': self.cells_per_block,
            'block_norm': self.block_norm,
            'feature_vector': self.feature_vector,
            'transform_sqrt': self.transform_sqrt,
            'target_size': self.target_size,
            'pca': self.pca,
            'scaler': self.scaler,
            'pca_components': self.pca_components
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"特征提取器模型已保存到: {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """
        加载特征提取器模型
        
        Args:
            filepath: 模型文件路径
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # 恢复参数
        self.orientations = model_data['orientations']
        self.pixels_per_cell = model_data['pixels_per_cell']
        self.cells_per_block = model_data['cells_per_block']
        self.block_norm = model_data['block_norm']
        self.feature_vector = model_data['feature_vector']
        self.transform_sqrt = model_data['transform_sqrt']
        self.target_size = model_data['target_size']
        self.pca = model_data['pca']
        self.scaler = model_data['scaler']
        self.pca_components = model_data['pca_components']
        
        print(f"特征提取器模型已从{filepath}加载")
    
    def get_feature_info(self) -> dict:
        """获取特征信息"""
        info = {
            'hog_params': {
                'orientations': self.orientations,
                'pixels_per_cell': self.pixels_per_cell,
                'cells_per_block': self.cells_per_block,
                'target_size': self.target_size
            }
        }
        
        if self.pca is not None:
            info['pca_info'] = {
                'n_components': self.pca_components,
                'explained_variance_ratio': float(np.sum(self.pca.explained_variance_ratio_))
            }
        
        return info 