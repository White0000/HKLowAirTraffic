import numpy as np
import open3d as o3d
from typing import List, Tuple

# 点云数据处理类
class PointCloudProcessor:
    # 初始化点云处理器，设置体素大小
    def __init__(self, voxel_size: float = 0.1):
        # 体素网格滤波体素大小
        self.voxel_size = voxel_size

    # 使用体素网格滤波进行下采样
    def downsample_pointcloud(self, point_cloud: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        # 进行点云下采样
        return point_cloud.voxel_down_sample(self.voxel_size)

    # 使用统计离群点移除算法去除点云中的噪声
    def remove_noise(self, point_cloud: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        # 移除噪声
        cl, ind = point_cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        return point_cloud.select_by_index(ind)

    # 使用RANSAC算法对点云进行平面分割
    def segment_planes(self, point_cloud: o3d.geometry.PointCloud) -> Tuple[o3d.geometry.PointCloud, o3d.geometry.PointCloud]:
        # 分割出平面与剩余部分
        plane_model, inliers = point_cloud.segment_plane(distance_threshold=0.02, ransac_n=3, num_iterations=1000)
        inlier_cloud = point_cloud.select_by_index(inliers)
        outlier_cloud = point_cloud.select_by_index(inliers, invert=True)
        return inlier_cloud, outlier_cloud

    # 提取点云中的障碍物，通过DBSCAN聚类识别
    def extract_obstacles(self, point_cloud: o3d.geometry.PointCloud) -> List[o3d.geometry.PointCloud]:
        # 进行障碍物聚类
        labels = np.array(point_cloud.cluster_dbscan(eps=0.02, min_points=10))
        obstacle_clusters = []
        max_label = labels.max()
        for i in range(max_label + 1):
            cluster = point_cloud.select_by_index(np.where(labels == i)[0])
            obstacle_clusters.append(cluster)
        return obstacle_clusters

    # 使用ICP对点云进行配准（对齐）
    def register_point_clouds(self, source: o3d.geometry.PointCloud, target: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        # 执行ICP配准
        icp_result = o3d.pipelines.registration.registration_icp(
            source, target, max_correspondence_distance=0.02,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint()
        )
        source.transform(icp_result.transformation)
        return source

    # 可视化处理后的单个点云
    def visualize_point_cloud(self, point_cloud: o3d.geometry.PointCloud) -> None:
        # 显示点云
        o3d.visualization.draw_geometries([point_cloud])

    # 可视化多个点云
    def visualize_multiple_point_clouds(self, point_clouds: List[o3d.geometry.PointCloud]) -> None:
        # 显示多个点云
        o3d.visualization.draw_geometries(point_clouds)
