import time
import random
import numpy as np
import open3d as o3d
import psutil
import os
from typing import List, Tuple

from src.core.path_planning import AStarPlanner
from src.perception.obstacle_detector import ObstacleDetector
from src.perception.pointcloud_processor import PointCloudProcessor
from src.control.drone_controller import DroneController
from src.control.airspace_manager import AirspaceManager

# 基准测试类, 用于评估整个系统在处理大规模数据时的性能(响应时间、内存/CPU等)
class BenchmarkTest:
    # 初始化基准测试工具, 创建各模块实例
    def __init__(self):
        # AStarPlanner, ObstacleDetector, PointCloudProcessor, DroneController, AirspaceManager
        self.path_planner = AStarPlanner()
        self.obstacle_detector = ObstacleDetector(mask_model_path="models/mask2former.pth")
        self.pointcloud_processor = PointCloudProcessor(voxel_size=0.05)
        self.drone_controller = DroneController(connection_string="udpin:localhost:14550")
        self.airspace_manager = AirspaceManager(airspace_zones=[(0, 10), (10, 20), (20, 30)])

    # 获取系统CPU/内存使用率
    def get_system_metrics(self) -> dict:
        cpu_usage = psutil.cpu_percent(interval=1)
        memory_info = psutil.virtual_memory()
        memory_usage = memory_info.percent
        return {"CPU Usage (%)": cpu_usage, "Memory Usage (%)": memory_usage}

    # 测试路径规划性能(A*), 记录起止时间差
    def test_path_planning_performance(
            self,
            start: Tuple[float, float, float],
            goal: Tuple[float, float, float],
            obstacles: List[Tuple[float, float, float]]
    ) -> float:
        start_time = time.time()
        path = self.path_planner.plan(start=start, goal=goal, obstacles=obstacles)
        end_time = time.time()
        path_planning_time = end_time - start_time
        return path_planning_time

    # 测试障碍物检测性能, 对图像进行推理, 记录时间
    def test_obstacle_detection_performance(self, image: np.ndarray) -> float:
        start_time = time.time()
        obstacles = self.obstacle_detector.detect_obstacles_from_image(image)
        end_time = time.time()
        obstacle_detection_time = end_time - start_time
        return obstacle_detection_time

    # 测试点云处理性能(降采样+去噪), 记录时间
    def test_pointcloud_processing_performance(self, point_cloud_data: np.ndarray) -> float:
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(point_cloud_data)
        start_time = time.time()
        processed_point_cloud = self.pointcloud_processor.downsample_pointcloud(point_cloud)
        processed_point_cloud = self.pointcloud_processor.remove_noise(processed_point_cloud)
        end_time = time.time()
        pointcloud_processing_time = end_time - start_time
        return pointcloud_processing_time

    # 测试系统综合性能(路径规划, 障碍物检测, 点云处理等), 返回各用时
    def test_system_performance(self) -> dict:
        start = (0, 0, 0)
        goal = (10, 10, 10)
        obstacles = [(5, 5, 5), (7, 8, 10)]
        test_image = np.random.rand(256, 256, 3)
        point_cloud_data = np.random.rand(10000, 3)
        path_planning_time = self.test_path_planning_performance(start, goal, obstacles)
        obstacle_detection_time = self.test_obstacle_detection_performance(test_image)
        pointcloud_processing_time = self.test_pointcloud_processing_performance(point_cloud_data)
        system_metrics = self.get_system_metrics()
        return {
            "path_planning_time": path_planning_time,
            "obstacle_detection_time": obstacle_detection_time,
            "pointcloud_processing_time": pointcloud_processing_time,
            "system_metrics": system_metrics
        }

    # 运行性能基准测试, 返回测试结果
    def run_performance_benchmark(self) -> dict:
        return self.test_system_performance()
