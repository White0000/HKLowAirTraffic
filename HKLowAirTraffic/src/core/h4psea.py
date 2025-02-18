import random
import math
import heapq
from typing import List, Tuple, Dict

# 粒子类, 用于在混合4D粒子群进化算法(H4PSEA)中表示候选路径
# 每个粒子包含一条4D路径(多个航点), 航点格式可为(x, y, z, t)
class Particle:
    # 初始化粒子, 包含路径、成本及局部搜索相关信息
    def __init__(self,
                 path: List[Tuple[float, float, float, float]],
                 cost: float,
                 velocity: float = 0.0):
        # path: 粒子当前的4D路径列表
        # cost: 粒子的当前路径总代价
        # velocity: 粒子的搜索“速度”(可根据需求定义)
        self.path = path
        self.cost = cost
        self.velocity = velocity
        self.best_path = path
        self.best_cost = cost

# 基于混合4D粒子群进化算法(H4PSEA)的全球路径规划器
# 结合PSO、遗传算子(交叉与变异)与局部A*搜索, 考虑4D距离、能耗、风险、时间延迟、空域层级转换等综合代价
class H4PSEA:
    # cost_params示例: {"w1":0.4, "w2":0.2, "w3":0.2, "w4":0.1, "w5":0.1, "lambda":1.0}
    # max_particles: 粒子数
    # max_iterations: 最大迭代次数
    # crossover_rate/mutation_rate: 遗传交叉与变异概率
    # local_search_rate: 执行局部A*搜索的概率
    # a_star_planner: 用于局部搜索的A*实例(需实现四维代价)
    def __init__(self,
                 cost_params: Dict[str, float] = None,
                 max_particles: int = 30,
                 max_iterations: int = 50,
                 crossover_rate: float = 0.3,
                 mutation_rate: float = 0.1,
                 local_search_rate: float = 0.2,
                 a_star_planner: object = None):
        # 如果未提供cost_params, 则使用默认权重
        if cost_params is None:
            cost_params = {
                "w1": 0.4,   # 距离代价权重
                "w2": 0.2,   # 能耗代价权重
                "w3": 0.2,   # 风险代价权重
                "w4": 0.1,   # 时间代价权重
                "w5": 0.1,   # 空域层级转换代价权重
                "lambda": 1.0
            }
        self.cost_params = cost_params
        self.max_particles = max_particles
        self.max_iterations = max_iterations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.local_search_rate = local_search_rate
        self.a_star_planner = a_star_planner

        # 粒子群
        self.particles: List[Particle] = []
        # 全局最优解及其代价
        self.best_global_path: List[Tuple[float, float, float, float]] = []
        self.best_global_cost = float("inf")
        # 记录最后一次规划结果, 供外部可视化或调用
        self.last_path: List[Tuple[float, float, float, float]] = []

    # 计算4D欧几里得距离(含lambda时间权重)
    def distance_4d(self,
                    n1: Tuple[float, float, float, float],
                    n2: Tuple[float, float, float, float]) -> float:
        x1, y1, z1, t1 = n1
        x2, y2, z2, t2 = n2
        lam = self.cost_params.get("lambda", 1.0)
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2 + lam * (t2 - t1) ** 2)

    # 计算任意两点之间的综合代价C_ij
    def cost_4d(self,
                n1: Tuple[float, float, float, float],
                n2: Tuple[float, float, float, float]) -> float:
        w1 = self.cost_params.get("w1", 0.4)
        w2 = self.cost_params.get("w2", 0.2)
        w3 = self.cost_params.get("w3", 0.2)
        w4 = self.cost_params.get("w4", 0.1)
        w5 = self.cost_params.get("w5", 0.1)
        dist_4d = self.distance_4d(n1, n2)
        energy = dist_4d
        risk = 0.0
        dt = abs(n2[3] - n1[3])
        layer_penalty = 0.0
        return w1 * dist_4d + w2 * energy + w3 * risk + w4 * dt + w5 * layer_penalty

    # 计算整条路径的总代价
    def compute_path_cost(self,
                          path: List[Tuple[float, float, float, float]]) -> float:
        total_cost = 0.0
        for i in range(len(path) - 1):
            total_cost += self.cost_4d(path[i], path[i + 1])
        return total_cost

    # 产生随机初始化路径(可根据需求限制空间或时间范围)
    def generate_random_path(self,
                             start: Tuple[float, float, float, float],
                             goal: Tuple[float, float, float, float]) -> List[Tuple[float, float, float, float]]:
        path = [start]
        num_waypoints = random.randint(3, 6)
        for _ in range(num_waypoints):
            rx = random.uniform(min(start[0], goal[0]), max(start[0], goal[0]))
            ry = random.uniform(min(start[1], goal[1]), max(start[1], goal[1]))
            rz = random.uniform(min(start[2], goal[2]), max(start[2], goal[2]))
            rt = random.uniform(min(start[3], goal[3]), max(start[3], goal[3]))
            path.append((rx, ry, rz, rt))
        path.append(goal)
        return path

    # 初始化粒子群
    def initialize_particles(self,
                             start: Tuple[float, float, float, float],
                             goal: Tuple[float, float, float, float]) -> None:
        self.particles.clear()
        self.best_global_path = []
        self.best_global_cost = float("inf")
        for _ in range(self.max_particles):
            random_path = self.generate_random_path(start, goal)
            cost = self.compute_path_cost(random_path)
            particle = Particle(path=random_path, cost=cost)
            if cost < self.best_global_cost:
                self.best_global_cost = cost
                self.best_global_path = random_path
            self.particles.append(particle)

    # 遗传交叉算子, 对两条路径进行交叉, 生成新路径
    def crossover(self,
                  path1: List[Tuple[float, float, float, float]],
                  path2: List[Tuple[float, float, float, float]]) -> List[Tuple[float, float, float, float]]:
        if len(path1) < 3 or len(path2) < 3:
            return path1
        cut1 = random.randint(1, len(path1) - 2)
        cut2 = random.randint(1, len(path2) - 2)
        new_path = path1[:cut1] + path2[cut2:]
        return new_path

    # 遗传变异算子, 随机扰动路径中的若干航点
    def mutate(self,
               path: List[Tuple[float, float, float, float]],
               start: Tuple[float, float, float, float],
               goal: Tuple[float, float, float, float]) -> List[Tuple[float, float, float, float]]:
        if len(path) <= 2:
            return path
        idx = random.randint(1, len(path) - 2)
        px, py, pz, pt = path[idx]
        mutated_x = px + random.uniform(-1.0, 1.0)
        mutated_y = py + random.uniform(-1.0, 1.0)
        mutated_z = pz + random.uniform(-1.0, 1.0)
        mutated_t = pt + random.uniform(-1.0, 1.0)
        clamped_x = max(min(mutated_x, max(start[0], goal[0])), min(start[0], goal[0]))
        clamped_y = max(min(mutated_y, max(start[1], goal[1])), min(start[1], goal[1]))
        clamped_z = max(min(mutated_z, max(start[2], goal[2])), min(start[2], goal[2]))
        clamped_t = max(min(mutated_t, max(start[3], goal[3])), min(start[3], goal[3]))
        path[idx] = (clamped_x, clamped_y, clamped_z, clamped_t)
        return path

    # 局部搜索: 使用A*对粒子部分路径进行细化(可随机选段或全段)
    def local_search(self,
                     particle_path: List[Tuple[float, float, float, float]],
                     obstacles: List[Tuple[float, float, float, float]]) -> List[Tuple[float, float, float, float]]:
        if not self.a_star_planner or len(particle_path) < 2:
            return particle_path
        i1 = random.randint(0, len(particle_path) - 2)
        sub_start = particle_path[i1]
        sub_goal = particle_path[i1 + 1]
        refined = self.a_star_planner.plan(sub_start, sub_goal, obstacles)
        if len(refined) >= 2:
            new_path = particle_path[:i1] + refined + particle_path[i1 + 2:]
            return new_path
        return particle_path

    # 进化过程(PSO+遗传+局部搜索)
    def update_particles(self,
                         start: Tuple[float, float, float, float],
                         goal: Tuple[float, float, float, float],
                         obstacles: List[Tuple[float, float, float, float]]) -> None:
        for i in range(self.max_particles):
            particle = self.particles[i]
            if random.random() < self.crossover_rate:
                partner_idx = random.randint(0, self.max_particles - 1)
                if partner_idx != i:
                    new_path = self.crossover(particle.path, self.particles[partner_idx].path)
                    particle.path = new_path
            if random.random() < self.mutation_rate:
                particle.path = self.mutate(particle.path, start, goal)
            if random.random() < self.local_search_rate:
                particle.path = self.local_search(particle.path, obstacles)
            new_cost = self.compute_path_cost(particle.path)
            particle.cost = new_cost
            if new_cost < particle.best_cost:
                particle.best_cost = new_cost
                particle.best_path = particle.path
            if new_cost < self.best_global_cost:
                self.best_global_cost = new_cost
                self.best_global_path = particle.path

    # 主流程, 输入起点/终点/障碍物, 返回全局最优4D路径
    def plan(self,
             start: Tuple[float, float, float, float],
             goal: Tuple[float, float, float, float],
             obstacles: List[Tuple[float, float, float, float]]) -> List[Tuple[float, float, float, float]]:
        self.initialize_particles(start, goal)
        for _iter in range(self.max_iterations):
            self.update_particles(start, goal, obstacles)
        self.last_path = self.best_global_path
        return self.best_global_path
