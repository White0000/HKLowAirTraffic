import math
from typing import List, Tuple

# 自定义异常类，用于表示空域冲突错误
class AirspaceConflictError(Exception):
    pass

# 空域管理器类，负责管理低空空域的分层、飞行调度及冲突检测
class AirspaceManager:
    # 初始化空域管理器，设置空域层级及飞行高度限制
    # airspace_zones: [(low_alt, high_alt), ...]
    # max_altitude: 最大飞行高度(米)
    def __init__(self, airspace_zones: List[Tuple[int, int]], max_altitude: float = 300.0):
        self.airspace_zones = airspace_zones
        self.max_altitude = max_altitude
        # 预留空域预约信息: [(start_time, end_time, zone_index), ...]
        self.reserved_slots = []

    # 根据高度altitude获取空域层级的索引(若超出范围则抛异常)
    def get_airspace_level(self, altitude: float) -> int:
        for i, (low, high) in enumerate(self.airspace_zones):
            if low <= altitude < high:
                return i
        raise ValueError("Altitude exceeds maximum airspace range.")

    # 为指定时间段和高度进行空域预约
    # start_time和end_time(秒), altitude(米)
    # 成功返回True, 若冲突则返回False
    def reserve_airspace_slot(self, start_time: float, end_time: float, altitude: float) -> bool:
        zone_index = self.get_airspace_level(altitude)
        for (res_start, res_end, res_zone) in self.reserved_slots:
            # 若同一层且时间段重叠即冲突
            if res_zone == zone_index and not (end_time <= res_start or start_time >= res_end):
                return False
        self.reserved_slots.append((start_time, end_time, zone_index))
        return True

    # 释放已预约的空域段
    def release_airspace_slot(self, start_time: float, end_time: float, altitude: float) -> None:
        zone_index = self.get_airspace_level(altitude)
        self.reserved_slots = [
            slot for slot in self.reserved_slots
            if slot != (start_time, end_time, zone_index)
        ]

    # 检查给定路径(一系列具有时间戳的航点)是否存在空域冲突
    # path中每个航点可格式化为(lat, lon, alt, t)
    # 如果冲突返回True, 否则False
    def detect_airspace_conflict(self, path: List[Tuple[float, float, float, float]]) -> bool:
        # 遍历路径的相邻时间段进行空域预约尝试
        for i in range(len(path) - 1):
            _, _, alt1, t1 = path[i]
            _, _, alt2, t2 = path[i + 1]
            start_time = min(t1, t2)
            end_time = max(t1, t2)
            mid_alt = (alt1 + alt2) / 2.0
            if not self.reserve_airspace_slot(start_time, end_time, mid_alt):
                return True
        return False

    # 调整路径以避免空域冲突(仅作示例, 实际逻辑可更复杂)
    # path中的每个航点格式(lat, lon, alt, t)
    def adjust_path_to_avoid_conflict(self, path: List[Tuple[float, float, float, float]]) -> List[Tuple[float, float, float, float]]:
        adjusted_path = []
        for (lat, lon, alt, t) in path:
            # 若此航点高度超出最大限制, 抛异常或尝试降低高度
            if alt > self.max_altitude:
                raise AirspaceConflictError("Flight altitude exceeds max altitude.")
            # 若对应层级已冲突, 则调整
            if not self.reserve_airspace_slot(t, t + 1, alt):
                new_alt = self.get_alternate_altitude(alt)
                if new_alt > self.max_altitude:
                    raise AirspaceConflictError("No available altitude for conflict avoidance.")
                adjusted_path.append((lat, lon, new_alt, t))
            else:
                adjusted_path.append((lat, lon, alt, t))
        return adjusted_path

    # 获取备选高度(示例: 简单+10米)
    def get_alternate_altitude(self, altitude: float) -> float:
        alt_candidate = altitude + 10.0
        if alt_candidate > self.max_altitude:
            raise AirspaceConflictError("Exceeded maximum altitude while searching alternate.")
        return alt_candidate
