import math
from typing import List, Tuple

# 代价函数类，用于计算路径的不同代价：距离、能耗、安全性、时间及空域层级转换
class CostFunctions:
    # 初始化，接收地图分辨率、能耗系数、安全距离阈值、飞行速度和最大飞行高度
    def __init__(self, resolution: float, energy_factor: float, safety_threshold: float, speed: float,
                 max_altitude: float):
        # 地图网格分辨率
        self.resolution = resolution
        # 单位距离的能耗系数(瓦特·米/秒)
        self.energy_factor = energy_factor
        # 安全距离阈值(米), 小于该值则认为路径不安全
        self.safety_threshold = safety_threshold
        # 无人机飞行速度(米/秒)
        self.speed = speed
        # 最大飞行高度(米)
        self.max_altitude = max_altitude

    # 计算路径的距离代价(欧几里得距离求和)
    def calculate_distance_cost(self, path: List[Tuple[float, float, float]]) -> float:
        distance_cost = 0.0
        for i in range(1, len(path)):
            x1, y1, _ = path[i - 1]
            x2, y2, _ = path[i]
            distance_cost += math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        return distance_cost

    # 计算路径的能耗代价(距离×能耗系数)
    def calculate_energy_cost(self, path: List[Tuple[float, float, float]]) -> float:
        distance_cost = self.calculate_distance_cost(path)
        energy_cost = distance_cost * self.energy_factor
        return energy_cost

    # 计算路径的安全性代价，根据障碍物与路径段中点的最短距离
    def calculate_safety_cost(self, path: List[Tuple[float, float, float]],
                              obstacles: List[Tuple[float, float, float]]) -> float:
        safety_cost = 0.0
        for i in range(1, len(path)):
            x1, y1, alt1 = path[i - 1]
            x2, y2, alt2 = path[i]
            midpoint_x = (x1 + x2) / 2
            midpoint_y = (y1 + y2) / 2
            midpoint_alt = (alt1 + alt2) / 2
            min_distance = float('inf')
            for ox, oy, oalt in obstacles:
                dist = math.sqrt((midpoint_x - ox) ** 2 + (midpoint_y - oy) ** 2 + (midpoint_alt - oalt) ** 2)
                if dist < min_distance:
                    min_distance = dist
            if min_distance < self.safety_threshold:
                safety_cost += (self.safety_threshold - min_distance)
        return safety_cost

    # 计算路径的时间代价(距离÷速度)
    def calculate_time_cost(self, path: List[Tuple[float, float, float]]) -> float:
        distance_cost = self.calculate_distance_cost(path)
        time_cost = distance_cost / self.speed
        return time_cost

    # 计算路径的空域层级转换代价(当航点高度跨层时增加代价)
    def calculate_airspace_cost(self, path: List[Tuple[float, float, float]]) -> float:
        airspace_cost = 0.0
        previous_airspace_level = None
        for _, _, altitude in path:
            airspace_level = self.get_airspace_level(altitude)
            if previous_airspace_level is not None and previous_airspace_level != airspace_level:
                airspace_cost += 1.0
            previous_airspace_level = airspace_level
        return airspace_cost

    # 根据飞行高度获取空域层级
    def get_airspace_level(self, altitude: float) -> int:
        if 0 <= altitude < 10:
            return 1
        elif 10 <= altitude < 20:
            return 2
        elif 20 <= altitude < 30:
            return 3
        else:
            raise ValueError("Altitude exceeds maximum airspace range")

    # 计算路径的总代价(将距离、能耗、安全、时间及空域层级转换代价相加)
    def calculate_total_cost(self, path: List[Tuple[float, float, float]],
                             obstacles: List[Tuple[float, float, float]]) -> float:
        distance_cost = self.calculate_distance_cost(path)
        energy_cost = self.calculate_energy_cost(path)
        safety_cost = self.calculate_safety_cost(path, obstacles)
        time_cost = self.calculate_time_cost(path)
        airspace_cost = self.calculate_airspace_cost(path)
        total_cost = distance_cost + energy_cost + safety_cost + time_cost + airspace_cost
        return total_cost
