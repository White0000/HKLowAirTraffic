import time
import random
import numpy as np
import open3d as o3d
from typing import List, Tuple
from unittest import TestCase

# 假设引入其他相关模块
from src.core.path_planning import AStarPlanner
from src.perception.obstacle_detector import ObstacleDetector
from src.utils.config_loader import ConfigLoader
from src.control.drone_controller import DroneController
from src.control.airspace_manager import AirspaceManager


class TestPathPlanningSystem(TestCase):
    """
    集成测试类，负责对路径规划系统的各个模块进行联合测试。
    测试内容包括路径规划、障碍物检测、飞行控制等模块的协同工作。
    """

    def setUp(self) -> None:
        """
        测试前的准备工作：初始化模块实例、加载配置等。
        """
        print("Setting up the test environment...")
        self.config_loader = ConfigLoader()
        self.config = self.config_loader.load_config('configs/drone_config.yaml')

        # 初始化各个模块
        self.path_planner = AStarPlanner()
        self.obstacle_detector = ObstacleDetector(mask_model_path="models/mask2former.pth")
        self.drone_controller = DroneController(connection_string="udpin:localhost:14550")
        self.airspace_manager = AirspaceManager(airspace_zones=[(0, 10), (10, 20), (20, 30)])

    def test_path_planning(self):
        """
        测试路径规划功能，验证 A* 算法的路径生成是否正确。
        """
        start = (0, 0, 0)  # 起点坐标
        goal = (10, 10, 10)  # 终点坐标
        obstacles = [(5, 5, 5), (7, 8, 10)]  # 模拟障碍物

        print("Running path planning test...")
        path = self.path_planner.plan(start=start, goal=goal, obstacles=obstacles)

        # 验证路径是否有效
        self.assertTrue(len(path) > 0, "No path found")
        print(f"Path found: {path}")

    def test_obstacle_detection(self):
        """
        测试障碍物检测功能，验证障碍物的正确识别与定位。
        """
        print("Running obstacle detection test...")

        # 模拟图像或点云数据
        test_image = np.random.rand(256, 256, 3)  # 模拟图像数据（RGB）
        obstacles = self.obstacle_detector.detect_obstacles_from_image(test_image)

        # 验证检测到的障碍物
        self.assertGreater(len(obstacles), 0, "No obstacles detected")
        print(f"Detected obstacles: {obstacles}")

    def test_airspace_management(self):
        """
        测试空域管理功能，验证空域预约和冲突检测。
        """
        print("Running airspace management test...")

        start_time = 0
        end_time = 10
        altitude = 10  # 中空层（10-20m）

        # 尝试预约空域
        success = self.airspace_manager.reserve_airspace_slot(start_time, end_time, altitude)
        self.assertTrue(success, "Failed to reserve airspace slot")

        # 尝试预约冲突的空域
        success = self.airspace_manager.reserve_airspace_slot(start_time, end_time, altitude)
        self.assertFalse(success, "Airspace conflict detected, but no error")

        print("Airspace management test passed")

    def test_drone_control(self):
        """
        测试无人机控制功能，验证飞行控制命令是否正确传递。
        """
        print("Running drone control test...")

        # 模拟路径和飞行器位置
        path = [(0, 0, 10), (2, 2, 10), (5, 5, 10)]
        drone_position = (0, 0, 10)

        # 飞行控制指令
        self.drone_controller.execute_path(path)

        # 模拟飞行器当前位置
        self.drone_controller.visualize_trajectory(path, drone_position)

        print("Drone control test passed")

    def test_system_integration(self):
        """
        测试系统集成，验证路径规划、障碍物检测、飞行控制和空域管理的联合工作。
        """
        print("Running system integration test...")

        # 模拟路径、障碍物和飞行器位置
        path = [(0, 0, 10), (2, 2, 10), (5, 5, 10)]
        obstacles = [(1, 1, 10), (3, 3, 10)]
        point_cloud_data = np.random.rand(100, 3)
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(point_cloud_data)
        drone_position = (0, 0, 10)

        # 运行完整系统流程
        self.drone_controller.execute_path(path)
        self.airspace_manager.reserve_airspace_slot(0, 10, 10)

        # 集成测试：可视化所有模块
        self.drone_controller.visualize_trajectory(path, drone_position)
        self.obstacle_detector.visualize_obstacles(obstacles)
        self.drone_controller.visualize_path(path)

        print("System integration test passed")

    def test_performance(self):
        """
        性能测试：验证系统在高负载下的响应时间。
        """
        print("Running performance test...")

        start_time = time.time()

        # 模拟大规模点云数据
        point_cloud_data = np.random.rand(10000, 3)  # 大规模点云数据
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(point_cloud_data)

        # 测试点云的处理和路径规划
        path = [(0, 0, 10), (2, 2, 10), (5, 5, 10)]
        self.obstacle_detector.visualize_obstacles([(1, 1, 10), (3, 3, 10)])

        self.assertLess(time.time() - start_time, 1, "Performance test failed: Took too long")
        print("Performance test passed")


if __name__ == "__main__":
    # 运行所有测试
    import unittest

    unittest.main()
