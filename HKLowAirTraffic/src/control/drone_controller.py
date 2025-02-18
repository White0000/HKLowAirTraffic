import time
from typing import List, Tuple
from pymavlink import mavutil

# 无人机控制器类，通过 MAVLink 协议与无人机通信
class DroneController:
    # 初始化无人机控制器
    # connection_string: MAVLink 连接字符串 (如 'udpin:localhost:14550')
    # flight_speed: 无人机的飞行速度 (单位: 米/秒)
    # altitude: 无人机的目标飞行高度 (单位: 米)
    def __init__(self, connection_string: str, flight_speed: float = 5.0, altitude: float = 100.0):
        # 记录连接信息和默认飞行参数
        self.connection_string = connection_string
        self.flight_speed = flight_speed
        self.altitude = altitude
        # 通过 pymavlink 建立连接并等待心跳包，以确认与无人机正常通信
        self.master = mavutil.mavlink_connection(self.connection_string)
        self.master.wait_heartbeat()

    # 设置无人机飞行模式 (如 'GUIDED', 'AUTO', 'LOITER')
    def set_mode(self, mode: str) -> None:
        # 与 MAVLink 模式对应的模式 ID 映射
        mode_mapping = {
            'GUIDED': 4,
            'AUTO': 3,
            'LOITER': 5
        }
        mode_id = mode_mapping.get(mode.upper(), None)
        if mode_id is None:
            raise ValueError(f"Unknown mode: {mode}")
        self.master.mav.set_mode_send(
            self.master.target_system,
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            mode_id
        )

    # 解锁无人机
    def arm(self) -> None:
        # 通过 pymavlink 的 API 发送解锁命令
        self.master.arducopter_arm()

    # 加锁无人机
    def disarm(self) -> None:
        # 通过 pymavlink 的 API 发送加锁命令
        self.master.arducopter_disarm()

    # 起飞到指定高度
    # target_altitude: 目标飞行高度 (单位: 米)
    def takeoff(self, target_altitude: float) -> None:
        # 切换到 'GUIDED' 模式并解锁后发送起飞命令
        self.set_mode('GUIDED')
        self.arm()
        self.master.mav.command_long_send(
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
            0, 0, 0, 0, 0, 0, 0, target_altitude
        )
        time.sleep(5)

    # 降落无人机
    def land(self) -> None:
        # 切换到 'AUTO' 模式并发送降落命令
        self.set_mode('AUTO')
        self.master.mav.command_long_send(
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_CMD_NAV_LAND,
            0, 0, 0, 0, 0, 0, 0, 0
        )

    # 前往指定航点 (lat, lon, alt)
    def go_to_waypoint(self, waypoint: Tuple[float, float, float]) -> None:
        # 切换到 'GUIDED' 模式并发送 waypoint 命令
        lat, lon, altitude = waypoint
        self.set_mode('GUIDED')
        self.master.mav.mission_item_send(
            self.master.target_system,
            self.master.target_component,
            0,
            mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
            mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
            2, 0, 0, 0, 0, 0, lat, lon, altitude
        )

    # 执行路径 (多个航点)
    def execute_path(self, path: List[Tuple[float, float, float]]) -> None:
        # 依次前往路径中的各个航点
        for waypoint in path:
            self.go_to_waypoint(waypoint)
            time.sleep(5)

    # 调整飞行速度 (米/秒)
    def control_speed(self, speed: float) -> None:
        # 设置飞行速度 (具体实现可能需要发送额外 MAVLink 消息)
        self.flight_speed = speed

    # 调整飞行高度 (米)
    def adjust_altitude(self, altitude: float) -> None:
        # 发送相关命令以更改无人机当前的高度
        self.altitude = altitude
        self.master.mav.command_long_send(
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_CMD_NAV_ALTITUDE_WAIT,
            0, 0, 0, 0, 0, 0, 0, altitude
        )

    # 停止当前操作: 发送返航指令
    def stop(self) -> None:
        # 发送返航命令 (RTL)
        self.master.mav.command_long_send(
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_CMD_NAV_RETURN_TO_LAUNCH,
            0, 0, 0, 0, 0, 0, 0, 0
        )
