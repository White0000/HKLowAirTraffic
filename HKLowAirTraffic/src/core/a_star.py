import heapq
import math
from typing import List, Tuple, Dict

# A*路径规划算法（四维扩展），可用于局部路径搜索或在H4PSEA中进行局部细化
class AStarPlanner:
    # 初始化，cost_params中可包含多重代价权重，如{"w1":0.4, "w2":0.2, "w3":0.2, "w4":0.1, "w5":0.1, "lambda":1.0}
    def __init__(self, cost_params: Dict[str, float]):
        # 储存代价权重
        self.cost_params = cost_params
        # open_list为优先队列，closed_set为已访问节点集合
        self.open_list = []
        self.closed_set = set()
        # 用于回溯路径的映射
        self.came_from = {}
        # g和f为记录节点的代价
        self.g_values = {}
        self.f_values = {}

    # 计算四维距离，含时间维度，并加权(lambda)
    def distance_4d(self, node1: Tuple[float, float, float, float],
                    node2: Tuple[float, float, float, float]) -> float:
        # node格式(x, y, z, t)
        x1, y1, z1, t1 = node1
        x2, y2, z2, t2 = node2
        lam = self.cost_params.get("lambda", 1.0)
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2 + lam * (t2 - t1) ** 2)

    # 计算两个节点间的综合代价C_ij，根据开发文档中的公式:
    # C_ij = w1 * d^4D + w2 * E + w3 * R + w4 * Δt + w5 * L
    # 这里E、R、L等可根据项目需求或传感器数据实现，本示例仅演示基础框架
    def cost_4d(self, node1: Tuple[float, float, float, float],
                node2: Tuple[float, float, float, float]) -> float:
        w1 = self.cost_params.get("w1", 0.4)
        w2 = self.cost_params.get("w2", 0.2)
        w3 = self.cost_params.get("w3", 0.2)
        w4 = self.cost_params.get("w4", 0.1)
        w5 = self.cost_params.get("w5", 0.1)

        dist_4d = self.distance_4d(node1, node2)     # d^4D
        energy = dist_4d                             # 示例：能耗E和4D距离正相关
        risk = 0.0                                   # 示例：可视实际需求定义
        dt = abs(node2[3] - node1[3])                # Δt
        layer_penalty = 0.0                          # 示例：空域层级转换惩罚可在外部根据高度实现
        return w1 * dist_4d + w2 * energy + w3 * risk + w4 * dt + w5 * layer_penalty

    # 估计当前节点到目标节点的启发式距离(可使用4D距离或仅空间距离)
    def heuristic_4d(self, current: Tuple[float, float, float, float],
                     goal: Tuple[float, float, float, float]) -> float:
        return self.distance_4d(current, goal)

    # 获取当前节点在4D空间的相邻节点，可根据实际需求设置步长或安全检查
    def get_neighbors(self, node: Tuple[float, float, float, float]) -> List[Tuple[float, float, float, float]]:
        # 简易实现：在(x±1, y±1, z±1, t±1)的局部范围内搜索
        # 若需要更精细的步长或约束，可自行扩展
        neighbors = []
        x, y, z, t = node
        for dx in [-1.0, 0.0, 1.0]:
            for dy in [-1.0, 0.0, 1.0]:
                for dz in [-1.0, 0.0, 1.0]:
                    for dt in [-1.0, 0.0, 1.0]:
                        if dx == 0 and dy == 0 and dz == 0 and dt == 0:
                            continue
                        nx = x + dx
                        ny = y + dy
                        nz = z + dz
                        nt = t + dt
                        neighbors.append((nx, ny, nz, nt))
        return neighbors

    # 重建从start到goal的路径
    def reconstruct_path(self, current: Tuple[float, float, float, float]) -> List[Tuple[float, float, float, float]]:
        path = [current]
        while current in self.came_from:
            current = self.came_from[current]
            path.append(current)
        path.reverse()
        return path

    # A*搜索主体
    # start与goal为4D节点，obstacles可为需要屏蔽的节点列表
    def plan(self,
             start: Tuple[float, float, float, float],
             goal: Tuple[float, float, float, float],
             obstacles: List[Tuple[float, float, float, float]] = None
             ) -> List[Tuple[float, float, float, float]]:
        if obstacles is None:
            obstacles = []

        self.open_list.clear()
        self.closed_set.clear()
        self.came_from.clear()
        self.g_values.clear()
        self.f_values.clear()

        # 初始化start节点
        self.g_values[start] = 0.0
        self.f_values[start] = self.heuristic_4d(start, goal)
        heapq.heappush(self.open_list, (self.f_values[start], start))

        # 循环处理open_list直至为空或找到目标
        while self.open_list:
            _, current = heapq.heappop(self.open_list)

            # 若与goal足够接近，可认为到达目标
            if self.distance_4d(current, goal) < 1e-3:
                return self.reconstruct_path(current)

            self.closed_set.add(current)

            # 获取相邻节点
            for neighbor in self.get_neighbors(current):
                if neighbor in self.closed_set:
                    continue
                if neighbor in obstacles:
                    continue

                # 计算新的g值 = 父节点g + cost_4d(current, neighbor)
                tentative_g = self.g_values[current] + self.cost_4d(current, neighbor)

                # 若neighbor未访问或发现更优路径
                if (neighbor not in self.g_values) or (tentative_g < self.g_values[neighbor]):
                    self.came_from[neighbor] = current
                    self.g_values[neighbor] = tentative_g
                    self.f_values[neighbor] = tentative_g + self.heuristic_4d(neighbor, goal)
                    heapq.heappush(self.open_list, (self.f_values[neighbor], neighbor))

        # 若搜索失败，返回空
        return []
