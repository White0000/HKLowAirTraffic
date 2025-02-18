import os
import cv2
import numpy as np
import open3d as o3d
from typing import List, Tuple

# 数据预处理器类，负责对原始图像与点云数据进行清洗、转换和处理
class DataPreprocessor:
    # 初始化数据预处理器，指定原始数据与处理后数据的存储路径
    def __init__(self, raw_data_path: str, processed_data_path: str):
        self.raw_data_path = raw_data_path
        self.processed_data_path = processed_data_path

    # 处理图像数据，将图像转换为灰度并进行滤波去噪
    def process_image_data(self, image_name: str) -> np.ndarray:
        image_path = os.path.join(self.raw_data_path, image_name)
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image {image_name} not found in {self.raw_data_path}")
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        filtered_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
        processed_image_path = os.path.join(self.processed_data_path, image_name)
        cv2.imwrite(processed_image_path, filtered_image)
        return filtered_image

    # 处理点云数据，对点云进行降采样和去噪
    def process_point_cloud_data(self, point_cloud_name: str) -> o3d.geometry.PointCloud:
        point_cloud_path = os.path.join(self.raw_data_path, point_cloud_name)
        pcd = o3d.io.read_point_cloud(point_cloud_path)
        if pcd.is_empty():
            raise FileNotFoundError(f"Point cloud {point_cloud_name} not found in {self.raw_data_path}")
        downsampled_pcd = pcd.voxel_down_sample(voxel_size=0.1)
        cl, ind = downsampled_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        cleaned_pcd = downsampled_pcd.select_by_index(ind)
        processed_pcd_path = os.path.join(self.processed_data_path, point_cloud_name)
        o3d.io.write_point_cloud(processed_pcd_path, cleaned_pcd)
        return cleaned_pcd

    # 批量处理指定目录下的图像与点云文件
    def process_data(self) -> None:
        image_files = [f for f in os.listdir(self.raw_data_path) if f.endswith(".jpg") or f.endswith(".png")]
        for image_name in image_files:
            self.process_image_data(image_name)
        point_cloud_files = [f for f in os.listdir(self.raw_data_path) if f.endswith(".pcd") or f.endswith(".ply")]
        for point_cloud_name in point_cloud_files:
            self.process_point_cloud_data(point_cloud_name)

# 主程序入口 (可根据需要移除或改写)
if __name__ == "__main__":
    raw_data_path = "data/raw"
    processed_data_path = "data/processed"
    preprocessor = DataPreprocessor(raw_data_path, processed_data_path)
    preprocessor.process_data()
