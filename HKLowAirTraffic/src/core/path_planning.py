import math
import random
import heapq
from typing import List, Tuple, Dict, Any

# A*路径规划算法（4D扩展），用于局部路径优化
class AStarPlanner:
    # 初始化A*算法
    def __init__(self, cost_params: Dict[str, float]):
        # cost_params包含代价函数的权重参数，如{"w1":0.5, "w2":0.2, ...}
        self.cost_params = cost_params
        self.open_list = []
        self.closed_list = []
        self.came_from = {}
        self.g_scores = {}
        self.f_scores = {}

    # 计算4D距离：空间+时间
    def distance_4d(self, node1: Tuple[float, float, float, float],
                    node2: Tuple[float, float, float, float]) -> float:
        # node格式为(x, y, z, t)
        x1, y1, z1, t1 = node1
        x2, y2, z2, t2 = node2
        lam = self.cost_params.get("lambda", 1.0)
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2 + lam * (t2 - t1) ** 2)

    # 计算两节点之间的综合代价C_ij
    def cost_4d(self, node1: Tuple[float, float, float, float],
                node2: Tuple[float, float, float, float]) -> float:
        # 这里可以根据需要计算能耗E、风险R、空域转换L等
        # 这里仅给出典型的加权示例，实际E、R等可根据项目需求细化或调用外部函数
        w1 = self.cost_params.get("w1", 0.5)
        w2 = self.cost_params.get("w2", 0.2)
        w3 = self.cost_params.get("w3", 0.2)
        w4 = self.cost_params.get("w4", 0.1)
        w5 = self.cost_params.get("w5", 0.0)
        dist_4d = self.distance_4d(node1, node2)
        energy = dist_4d  # 示例：能耗E和4D距离正相关
        risk = 0.0        # 示例：安全风险可视情况定义
        dt = abs(node2[3] - node1[3])
        layer_penalty = 0.0
        # 综合代价公式: C_ij = w1*d^4D + w2*E + w3*R + w4*dt + w5*layer_penalty
        return w1 * dist_4d + w2 * energy + w3 * risk + w4 * dt + w5 * layer_penalty

    # 估计从当前节点到目标节点的启发式距离（可使用4D欧几里得距离）
    def heuristic_4d(self, current: Tuple[float, float, float, float],
                     goal: Tuple[float, float, float, float]) -> float:
        return self.distance_4d(current, goal)

    # 获取某节点在4D空间的相邻节点
    def get_neighbors(self, node: Tuple[float, float, float, float]) -> List[Tuple[float, float, float, float]]:
        # 简易实现：只考虑(x±1, y±1, z±1, t±1)等附近节点，实际可根据需求自定义
        x, y, z, t = node
        neighbors = []
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

    # 计算并返回从终点回溯到起点的最终路径
    def reconstruct_path(self, current: Tuple[float, float, float, float]) -> List[Tuple[float, float, float, float]]:
        path = [current]
        while current in self.came_from:
            current = self.came_from[current]
            path.append(current)
        path.reverse()
        return path

    # A*搜索主函数
    def plan(self,
             start: Tuple[float, float, float, float],
             goal: Tuple[float, float, float, float],
             obstacles: List[Tuple[float, float, float, float]]) -> List[Tuple[float, float, float, float]]:
        self.open_list = []
        self.closed_list = []
        self.came_from.clear()
        self.g_scores = {}
        self.f_scores = {}

        self.g_scores[start] = 0.0
        self.f_scores[start] = self.heuristic_4d(start, goal)
        heapq.heappush(self.open_list, (self.f_scores[start], start))

        while self.open_list:
            _, current = heapq.heappop(self.open_list)

            # 若已到达目标节点，则回溯路径
            if self.distance_4d(current, goal) < 1e-3:
                return self.reconstruct_path(current)

            self.closed_list.append(current)

            # 遍历相邻节点
            for neighbor in self.get_neighbors(current):
                if neighbor in self.closed_list:
                    continue
                if neighbor in obstacles:
                    continue

                tentative_g = self.g_scores[current] + self.cost_4d(current, neighbor)

                if neighbor not in self.g_scores or tentative_g < self.g_scores[neighbor]:
                    self.came_from[neighbor] = current
                    self.g_scores[neighbor] = tentative_g
                    self.f_scores[neighbor] = tentative_g + self.heuristic_4d(neighbor, goal)
                    heapq.heappush(self.open_list, (self.f_scores[neighbor], neighbor))
        return []

# 混合4D粒子群进化算法（H4PSEA），融合PSO、遗传算子及局部A*搜索
class H4PSEAPlanner:
    # 初始化H4PSEA，max_particles为粒子数，max_iterations为迭代次数
    def __init__(self,
                 cost_params: Dict[str, float],
                 max_particles: int = 30,
                 max_iterations: int = 50,
                 crossover_rate: float = 0.3,
                 mutation_rate: float = 0.1):
        # 保存代价权重等配置
        self.cost_params = cost_params
        # 粒子相关设置
        self.max_particles = max_particles
        self.max_iterations = max_iterations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        # 粒子集合
        self.particles = []
        # 全局最优解与最优代价
        self.best_global_path = []
        self.best_global_cost = float('inf')
        # 用于局部搜索的AStar
        self.a_star = AStarPlanner(cost_params=self.cost_params)
        # 记录上一次规划的路径（可供外部调用可视化）
        self.last_path = []

    # 计算4D距离
    def distance_4d(self, n1: Tuple[float, float, float, float],
                    n2: Tuple[float, float, float, float]) -> float:
        x1, y1, z1, t1 = n1
        x2, y2, z2, t2 = n2
        lam = self.cost_params.get("lambda", 1.0)
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2 + lam * (t2 - t1) ** 2)

    # 简化的路径表示与成本计算（可灵活扩展）
    def compute_path_cost(self, path: List[Tuple[float, float, float, float]]) -> float:
        total_cost = 0.0
        for i in range(len(path) - 1):
            total_cost += self.a_star.cost_4d(path[i], path[i + 1])
        return total_cost

    # 初始化粒子（随机路径），此处仅演示随机航点
    def initialize_particles(self,
                             start: Tuple[float, float, float, float],
                             goal: Tuple[float, float, float, float],
                             obstacles: List[Tuple[float, float, float, float]]) -> None:
        self.particles = []
        for _ in range(self.max_particles):
            random_path = [start]
            num_waypoints = random.randint(3, 6)
            for _j in range(num_waypoints):
                # 这里随机生成4D航点，可根据实际需求进行限制
                rx = random.uniform(min(start[0], goal[0]), max(start[0], goal[0]))
                ry = random.uniform(min(start[1], goal[1]), max(start[1], goal[1]))
                rz = random.uniform(min(start[2], goal[2]), max(start[2], goal[2]))
                rt = random.uniform(min(start[3], goal[3]), max(start[3], goal[3]))
                random_path.append((rx, ry, rz, rt))
            random_path.append(goal)
            cost = self.compute_path_cost(random_path)
            velocity = []  # 可自定义路径层级的“速度”
            self.particles.append({
                "path": random_path,
                "velocity": velocity,
                "best_path": random_path,
                "best_cost": cost
            })

    # 交叉算子：对两个粒子的路径进行交叉
    def crossover(self, path1: List[Tuple[float, float, float, float]],
                  path2: List[Tuple[float, float, float, float]]) -> List[Tuple[float, float, float, float]]:
        if len(path1) < 3 or len(path2) < 3:
            return path1
        cut1 = random.randint(1, len(path1) - 2)
        cut2 = random.randint(1, len(path2) - 2)
        new_path = path1[:cut1] + path2[cut2:]
        return new_path

    # 变异算子：随机扰动路径中的若干航点
    def mutate(self, path: List[Tuple[float, float, float, float]]) -> List[Tuple[float, float, float, float]]:
        if len(path) <= 2:
            return path
        idx = random.randint(1, len(path) - 2)
        px, py, pz, pt = path[idx]
        mutated_x = px + random.uniform(-1.0, 1.0)
        mutated_y = py + random.uniform(-1.0, 1.0)
        mutated_z = pz + random.uniform(-1.0, 1.0)
        mutated_t = pt + random.uniform(-1.0, 1.0)
        path[idx] = (mutated_x, mutated_y, mutated_z, mutated_t)
        return path

    # 利用A*对粒子路径中的局部段进行细化
    def local_search(self,
                     path: List[Tuple[float, float, float, float]],
                     obstacles: List[Tuple[float, float, float, float]]) -> List[Tuple[float, float, float, float]]:
        if len(path) < 3:
            return path
        # 简易局部搜索：随机选择两点，用A*在4D空间做细化
        i1 = random.randint(0, len(path) - 2)
        sub_start = path[i1]
        sub_goal = path[i1 + 1]
        refined_segment = self.a_star.plan(sub_start, sub_goal, obstacles)
        if len(refined_segment) >= 2:
            # 将 refined_segment 替换原先的两个航点
            new_path = path[:i1] + refined_segment + path[i1 + 2:]
            return new_path
        return path

    # 更新粒子集（PSO + 遗传 + 局部搜索）
    def update_particles(self,
                         start: Tuple[float, float, float, float],
                         goal: Tuple[float, float, float, float],
                         obstacles: List[Tuple[float, float, float, float]]) -> None:
        # 执行进化迭代，对每个粒子进行交叉、变异、局部搜索
        for i in range(self.max_particles):
            particle = self.particles[i]

            # 交叉
            if random.random() < self.crossover_rate:
                mate_idx = random.randint(0, self.max_particles - 1)
                if mate_idx != i:
                    particle["path"] = self.crossover(particle["path"], self.particles[mate_idx]["path"])

            # 变异
            if random.random() < self.mutation_rate:
                particle["path"] = self.mutate(particle["path"])

            # 局部搜索
            particle["path"] = self.local_search(particle["path"], obstacles)

            # 计算新成本
            new_cost = self.compute_path_cost(particle["path"])

            # 更新粒子自身的最好解
            if new_cost < particle["best_cost"]:
                particle["best_cost"] = new_cost
                particle["best_path"] = particle["path"]

            # 更新全局最好解
            if new_cost < self.best_global_cost:
                self.best_global_cost = new_cost
                self.best_global_path = particle["path"]

    # 主规划接口，返回全局最优路径
    def plan(self,
             start: Tuple[float, float, float, float],
             goal: Tuple[float, float, float, float],
             obstacles: List[Tuple[float, float, float, float]]) -> List[Tuple[float, float, float, float]]:
        self.best_global_path = []
        self.best_global_cost = float("inf")
        self.initialize_particles(start, goal, obstacles)

        for _iter in range(self.max_iterations):
            self.update_particles(start, goal, obstacles)

        self.last_path = self.best_global_path
        return self.best_global_path
