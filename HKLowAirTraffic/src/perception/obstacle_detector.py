import os
import numpy as np
import cv2
import torch
import traceback
from typing import List, Tuple
from open3d import geometry, utility

class ObstacleDetector:
    # 基于深度学习和点云数据的障碍物检测类
    def __init__(self, mask_model_path: str, threshold: float = 0.5):
        # threshold为置信度阈值, depth_map为深度图(可后续赋值)
        self.threshold = threshold
        self.depth_map = None
        self.mask_model = None

        # 首先检查并尝试将 .pkl 转换为 .pth
        converted_path = self._convert_pkl_to_pth_if_needed(mask_model_path)

        try:
            # PyTorch 2.6+ 默认 weights_only=True, 可能导致部分pickle对象无法反序列化
            # 若文件不可信, 切勿强制weights_only=False
            self.mask_model = torch.load(
                converted_path,
                map_location=torch.device('cpu'),
                weights_only=False
            )
            self.mask_model.eval()

        except RuntimeError as e:
            traceback.print_exc()
            if "Invalid magic number" in str(e):
                msg = (
                    "Model file possibly corrupt or not a valid PyTorch model.\n"
                    "Please verify or re-download from a trusted source.\n"
                    f"Detail: {e}"
                )
                raise RuntimeError(msg) from e
            else:
                raise RuntimeError(f"Failed to load model. Detail: {e}") from e
        except Exception as e:
            traceback.print_exc()
            raise RuntimeError(f"Failed to load model. Detail: {e}") from e

    def _convert_pkl_to_pth_if_needed(self, file_path: str) -> str:
        # 若后缀为 .pkl, 尝试用 Detectron2 读取并提取 state_dict, 存为 .pth
        # 若失败, 继续用原路径加载
        base, ext = os.path.splitext(file_path)
        if ext.lower() != ".pkl":
            return file_path

        pth_path = base + ".pth"
        if os.path.exists(pth_path):
            return pth_path  # 已转换过

        try:
            import detectron2.config
            from detectron2.checkpoint import DetectionCheckpointer
            from detectron2.modeling import build_model

            print(f"[Info] Attempting to convert '{file_path}' -> '{pth_path}' using Detectron2...")

            cfg = detectron2.config.get_cfg()
            # 不一定要指定具体model config, 只要能解析checkpoint即可
            model = build_model(cfg)
            checkpointer = DetectionCheckpointer(model)
            checkpoint = checkpointer.load(file_path)

            if "model" in checkpoint:
                state_dict = checkpoint["model"]
            else:
                state_dict = checkpoint  # 有些文件直接是普通dict

            torch.save(state_dict, pth_path)
            print(f"[Info] Conversion succeeded: '{pth_path}' created.")
            return pth_path

        except ImportError:
            print("[Warning] Detectron2 not installed. Can't auto-convert .pkl -> .pth.")
            # 不做转换, 尝试直接加载原file_path
            return file_path
        except Exception as e:
            traceback.print_exc()
            print("[Warning] Failed to convert .pkl -> .pth, will try original file.")
            return file_path

    def set_depth_map(self, depth_map: np.ndarray) -> None:
        # 设置深度图, 用于后续障碍物Z信息获取
        self.depth_map = depth_map

    def detect_obstacles_from_image(self, image: np.ndarray) -> List[Tuple[int, int, float]]:
        # 对输入图像进行检测, 返回(x,y,z)坐标列表
        if not self.mask_model:
            return []
        with torch.no_grad():
            input_tensor = torch.from_numpy(image).unsqueeze(0).float()
            output = self.mask_model(input_tensor)
        mask = output > self.threshold
        obstacle_positions = self.extract_positions_from_mask(mask)
        return obstacle_positions

    def extract_positions_from_mask(self, mask: torch.Tensor) -> List[Tuple[int, int, float]]:
        # 从二值掩码中提取(x,y,z)
        positions = []
        indices = torch.nonzero(mask, as_tuple=False)
        for idx in indices:
            y_coord = idx[0].item()
            x_coord = idx[1].item()
            z_coord = self.get_depth_at_position(x_coord, y_coord)
            positions.append((x_coord, y_coord, z_coord))
        return positions

    def get_depth_at_position(self, x: int, y: int) -> float:
        # 从深度图获取对应(x,y)的深度值, 若未设置depth_map则返回0.0
        if self.depth_map is None:
            return 0.0
        return float(self.depth_map[y, x])

    def detect_obstacles_from_pointcloud(
        self,
        point_cloud: np.ndarray,
        max_distance: float = 50.0
    ) -> List[Tuple[float, float, float]]:
        # 从点云中检测障碍物并过滤过远的点
        obstacles = []
        for point in point_cloud:
            if np.linalg.norm(point) < max_distance:
                obstacles.append(tuple(point))
        return obstacles

    def track_dynamic_obstacles(
        self,
        previous_positions: List[Tuple[float, float, float]],
        current_positions: List[Tuple[float, float, float]]
    ) -> List[Tuple[float, float, float]]:
        # 对比前后帧障碍物位置, 估计运动轨迹
        dynamic_obstacles = []
        if not current_positions:
            return previous_positions
        for prev in previous_positions:
            closest_obstacle = min(
                current_positions,
                key=lambda x: np.linalg.norm(np.array(x) - np.array(prev))
            )
            dynamic_obstacles.append(closest_obstacle)
        return dynamic_obstacles

    def generate_obstacle_map(self, obstacles: List[Tuple[float, float, float]]) -> geometry.PointCloud:
        # 将障碍物坐标生成Open3D点云对象
        points = np.array(obstacles, dtype=np.float32)
        pcd = geometry.PointCloud()
        pcd.points = utility.Vector3dVector(points)
        return pcd

    def visualize_obstacles(self, obstacles: List[Tuple[float, float, float]]) -> None:
        # 使用Open3D可视化障碍物
        import open3d as o3d
        pcd = self.generate_obstacle_map(obstacles)
        o3d.visualization.draw_geometries([pcd])
