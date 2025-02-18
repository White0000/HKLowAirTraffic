import open3d as o3d
import numpy as np
from typing import List, Tuple

# 3D 可视化工具类，使用 Open3D 库进行点云、路径、障碍物和飞行状态的可视化
class Visualizer:
    # 初始化可视化工具，创建 Open3D 可视化窗口
    def __init__(self):
        # 创建Open3D可视化窗口
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name="3D Path Planning Visualization")
        self.path_lines = None
        self.drone_marker = None
        self.obstacle_cloud = None
        self.path_points = None

    # 可视化点云数据
    def visualize_point_cloud(self, point_cloud: o3d.geometry.PointCloud) -> None:
        # 可视化输入的点云
        if self.obstacle_cloud is None:
            self.obstacle_cloud = point_cloud
            self.vis.add_geometry(self.obstacle_cloud)
        else:
            self.obstacle_cloud.points = point_cloud.points
        self.vis.update_geometry(self.obstacle_cloud)

    # 可视化规划好的路径，将路径连接成线段并显示在3D空间中
    def visualize_path(self, path: List[Tuple[float, float, float]]) -> None:
        if len(path) < 2:
            return
        if self.path_points is None:
            self.path_points = np.array(path)
            path_line_set = o3d.geometry.LineSet()
            path_line_set.points = o3d.utility.Vector3dVector(self.path_points)
            lines = [[i, i + 1] for i in range(len(self.path_points) - 1)]
            path_line_set.lines = o3d.utility.Vector2iVector(lines)
            self.path_lines = path_line_set
            self.vis.add_geometry(self.path_lines)
        else:
            self.path_points = np.array(path)
            self.path_lines.points = o3d.utility.Vector3dVector(self.path_points)
            lines = [[i, i + 1] for i in range(len(self.path_points) - 1)]
            self.path_lines.lines = o3d.utility.Vector2iVector(lines)
        self.vis.update_geometry(self.path_lines)

    # 可视化障碍物
    def visualize_obstacles(self, obstacles: List[Tuple[float, float, float]]) -> None:
        obstacle_points = np.array(obstacles)
        obstacle_cloud = o3d.geometry.PointCloud()
        obstacle_cloud.points = o3d.utility.Vector3dVector(obstacle_points)
        obstacle_cloud.paint_uniform_color([1.0, 0.0, 0.0])
        if self.obstacle_cloud is None:
            self.obstacle_cloud = obstacle_cloud
            self.vis.add_geometry(self.obstacle_cloud)
        else:
            self.obstacle_cloud.points = obstacle_cloud.points
        self.vis.update_geometry(self.obstacle_cloud)

    # 可视化飞行器轨迹与飞行器当前位置
    def visualize_trajectory(self, trajectory: List[Tuple[float, float, float]],
                             drone_position: Tuple[float, float, float]) -> None:
        if len(trajectory) > 1:
            trajectory_points = np.array(trajectory)
            trajectory_line_set = o3d.geometry.LineSet()
            trajectory_line_set.points = o3d.utility.Vector3dVector(trajectory_points)
            lines = [[i, i + 1] for i in range(len(trajectory_points) - 1)]
            trajectory_line_set.lines = o3d.utility.Vector2iVector(lines)
            self.vis.add_geometry(trajectory_line_set)
            self.vis.update_geometry(trajectory_line_set)

        drone_marker = o3d.geometry.TriangleMesh.create_sphere(radius=0.5)
        drone_marker.translate(np.array(drone_position))
        drone_marker.paint_uniform_color([0.0, 0.0, 1.0])
        if self.drone_marker is None:
            self.drone_marker = drone_marker
            self.vis.add_geometry(self.drone_marker)
        else:
            translation = np.array(drone_position) - self.drone_marker.get_center()
            self.drone_marker.translate(translation)
        self.vis.update_geometry(self.drone_marker)

    # 实时更新可视化内容
    def update_visualization(self, point_cloud: o3d.geometry.PointCloud,
                             path: List[Tuple[float, float, float]],
                             obstacles: List[Tuple[float, float, float]],
                             drone_position: Tuple[float, float, float]) -> None:
        self.vis.clear_geometries()
        if point_cloud:
            self.visualize_point_cloud(point_cloud)
        if path:
            self.visualize_path(path)
            self.visualize_trajectory(path, drone_position)
        if obstacles:
            self.visualize_obstacles(obstacles)
        self.vis.poll_events()
        self.vis.update_renderer()

    # 可视化最终结果
    def visualize_final_result(self, path: List[Tuple[float, float, float]],
                               obstacles: List[Tuple[float, float, float]],
                               drone_position: Tuple[float, float, float]) -> None:
        self.visualize_path(path)
        self.visualize_obstacles(obstacles)
        self.visualize_trajectory(path, drone_position)
        self.vis.run()
