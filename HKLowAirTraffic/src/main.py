import sys
import os
import psutil
import yaml
from typing import Dict, List, Tuple
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QLabel,
    QHBoxLayout, QComboBox, QGroupBox, QFormLayout, QMessageBox, QFileDialog
)
from PyQt5.QtCore import QTimer, Qt, QThread, pyqtSignal
import open3d as o3d

# 调整 sys.path，确保可以正确导入项目内的模块
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# 尝试导入各核心模块并进行异常处理
try:
    from core.path_planning import AStarPlanner, H4PSEAPlanner
except Exception as e:
    AStarPlanner = None
    H4PSEAPlanner = None
    print(f"[Warning] Error loading path planning modules: {e}")

try:
    from perception.obstacle_detector import ObstacleDetector
except Exception as e:
    ObstacleDetector = None
    print(f"[Warning] Error loading obstacle detector: {e}")

try:
    from control.drone_controller import DroneController
except Exception as e:
    DroneController = None
    print(f"[Warning] Error loading drone controller: {e}")

try:
    from control.airspace_manager import AirspaceManager
except Exception as e:
    AirspaceManager = None
    print(f"[Warning] Error loading airspace manager: {e}")

try:
    from utils.visualizer import Visualizer
except Exception as e:
    Visualizer = None
    print(f"[Warning] Error loading visualizer: {e}")

try:
    from utils.logger import Logger
except Exception as e:
    Logger = None
    print(f"[Warning] Error loading logger: {e}")


class DroneControllerLoaderThread(QThread):
    """
    异步加载 DroneController 的线程，避免阻塞主界面
    """
    loaded = pyqtSignal(object, str)

    def __init__(self, connection_string: str):
        super().__init__()
        self.connection_string = connection_string

    def run(self):
        if not DroneController:
            self.loaded.emit(None, "DroneController module not available.")
            return

        try:
            controller = DroneController(connection_string=self.connection_string)
            self.loaded.emit(controller, "DroneController loaded successfully.")
        except Exception as e:
            self.loaded.emit(None, f"Failed to initialize DroneController: {e}")


class ModelLoaderThread(QThread):
    """
    异步加载障碍物检测模型的线程
    """
    loaded = pyqtSignal(str, bool)

    def __init__(self, model_path: str):
        super().__init__()
        self.model_path = model_path

    def run(self):
        if os.path.exists(self.model_path):
            self.loaded.emit(self.model_path, True)
        else:
            self.loaded.emit(self.model_path, False)


class MainWindow(QMainWindow):
    """
    系统主窗口，集成日志、无人机控制、路径规划算法切换与可视化界面
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Low Air Traffic Path Planning System")
        self.setGeometry(100, 100, 1200, 800)

        # 日志模块
        self.logger = Logger(log_dir="logs", log_level="DEBUG") if Logger else None
        if self.logger:
            self.logger.log("Logger initialized", "INFO")
        else:
            print("Logger not available.")

        # 默认障碍物检测模型
        self.obstacle_detector = None
        self.init_obstacle_detector()

        # DroneController 异步加载
        self.drone_controller = None
        if DroneController:
            connection_string = self.load_drone_connection()
            self.drone_loader_thread = DroneControllerLoaderThread(connection_string=connection_string)
            self.drone_loader_thread.loaded.connect(self.on_drone_controller_loaded)
            self.drone_loader_thread.start()

        # 空域管理与可视化
        self.airspace_manager = AirspaceManager(airspace_zones=[(0, 10), (10, 20), (20, 30)]) if AirspaceManager else None
        self.visualizer = Visualizer() if Visualizer else None

        # 路径规划器 (AStarPlanner 或 H4PSEAPlanner)
        self.path_planner = None

        # 初始化UI
        self.init_ui()

        # 定时器用于更新系统状态
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_system_status)
        self.timer.start(1000)

        # 设置算法选择默认选项为A*，并手动触发一次切换
        # 避免构造时就出错，我们先把下拉选择设为0，然后在UI绘制后再执行algorithm_changed
        # QTimer.singleShot 可以用来在事件循环开始后再执行
        QTimer.singleShot(0, lambda: self.algorithm_changed("A* Algorithm"))

    def init_obstacle_detector(self):
        """
        初始化默认的障碍物检测模型
        """
        if not ObstacleDetector:
            print("[Warning] ObstacleDetector module not available.")
            return

        default_model_path = os.path.join(parent_dir, "data", "models", "mask2former.pth")
        if os.path.exists(default_model_path):
            try:
                self.obstacle_detector = ObstacleDetector(mask_model_path=default_model_path)
            except Exception as e:
                print(f"[Warning] Failed to initialize ObstacleDetector: {e}")
                self.obstacle_detector = None
        else:
            self.obstacle_detector = None

    def init_ui(self):
        """
        构建UI界面：状态面板、算法选择、控制面板、资源监控等
        """
        self.layout = QVBoxLayout()

        # 状态面板
        self.status_panel = QVBoxLayout()
        self.label_status = QLabel("System Status: Ready", self)
        self.status_panel.addWidget(self.label_status)

        # 算法选择
        self.control_panel = QVBoxLayout()
        self.algorithm_selector = QComboBox(self)
        self.algorithm_selector.addItem("A* Algorithm")
        self.algorithm_selector.addItem("Hybrid 4D PSO Algorithm (H4PSEA)")
        self.algorithm_selector.currentTextChanged.connect(self.algorithm_changed)
        self.control_panel.addWidget(self.algorithm_selector)

        # 控制子面板
        self.control_box = QGroupBox("Control Panel")
        self.control_layout = QFormLayout()

        self.start_button = QPushButton("Start Path Planning", self)
        self.start_button.clicked.connect(self.start_path_planning)
        self.control_layout.addRow("Start Planning", self.start_button)

        self.update_button = QPushButton("Update Visualization", self)
        self.update_button.clicked.connect(self.update_visualization)
        self.control_layout.addRow("Update View", self.update_button)

        self.load_model_button = QPushButton("Load Obstacle Detector Model", self)
        self.load_model_button.clicked.connect(self.load_pretrained_model)
        self.control_layout.addRow("Load Model", self.load_model_button)

        self.quit_button = QPushButton("Quit", self)
        self.quit_button.clicked.connect(self.close)
        self.control_layout.addRow("Exit", self.quit_button)

        self.control_box.setLayout(self.control_layout)
        self.control_panel.addWidget(self.control_box)

        # 系统状态指示
        self.path_status = QLabel("Path Planning: Not Started", self)
        self.system_status = QLabel("System Running: Idle", self)
        self.status_panel.addWidget(self.path_status)
        self.status_panel.addWidget(self.system_status)

        # 资源监控面板（CPU/内存使用）
        self.resource_panel = QVBoxLayout()
        self.cpu_usage_label = QLabel("CPU Usage: 0%", self)
        self.memory_usage_label = QLabel("Memory Usage: 0%", self)
        self.resource_panel.addWidget(self.cpu_usage_label)
        self.resource_panel.addWidget(self.memory_usage_label)

        # 组合整体布局
        container = QWidget()
        self.layout.addLayout(self.status_panel)
        self.layout.addLayout(self.control_panel)
        self.layout.addLayout(self.resource_panel)
        container.setLayout(self.layout)
        self.setCentralWidget(container)

    def load_drone_connection(self) -> str:
        """
        加载无人机连接参数，如 'udpin:localhost:14550' 等
        """
        config_path = os.path.join(parent_dir, "configs", "drone_config.yaml")
        if not os.path.exists(config_path):
            print(f"[Warning] drone_config.yaml not found. Using default connection.")
            return "udpin:localhost:14550"

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                drone_cfg = yaml.safe_load(f)
            return drone_cfg.get("connection_string", "udpin:localhost:14550")
        except Exception as e:
            print(f"[Warning] Failed to load drone_config.yaml: {e}")
            return "udpin:localhost:14550"

    def load_path_planning_config(self) -> dict:
        """
        加载 path_planning.yaml 中的所有配置信息，并返回
        """
        config_path = os.path.join(parent_dir, "configs", "path_planning.yaml")
        if not os.path.exists(config_path):
            print("[Warning] path_planning.yaml not found.")
            return {}

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                scenario_config = yaml.safe_load(f)
            return scenario_config if scenario_config else {}
        except Exception as e:
            self.show_error_message(f"Failed to load path_planning.yaml: {e}")
            return {}

    def parse_cost_function(self, cost_dict: dict) -> Dict[str, float]:
        """
        根据 path_planning.yaml 的 'cost_function' 字段解析并返回给 H4PSEA 或 AStar 的 cost_params
        """
        if not cost_dict:
            # 默认
            return {"w1": 0.4, "w2": 0.2, "w3": 0.2, "w4": 0.1, "w5": 0.1, "lambda": 1.0}

        w1 = cost_dict.get("distance_weight", 0.4)
        w2 = cost_dict.get("energy_weight", 0.2)
        w3 = cost_dict.get("safety_weight", 0.2)
        w4 = cost_dict.get("time_weight", 0.1)
        w5 = cost_dict.get("airspace_weight", 0.1)
        return {
            "w1": w1,
            "w2": w2,
            "w3": w3,
            "w4": w4,
            "w5": w5,
            "lambda": 1.0
        }

    def load_current_scenario(self) -> Tuple[Tuple[float, ...], Tuple[float, ...], List[Tuple[float, ...]]]:
        """
        从 path_planning.yaml 读取规划场景，包括 start、goal、obstacles
        """
        scenario_config = self.load_path_planning_config()
        scenario_data = scenario_config.get("scenario", None)

        if not scenario_data:
            self.show_error_message("No scenario found in path_planning.yaml, using default scenario.")
            # 提供默认
            return (0, 0, 0, 0), (5, 5, 5, 5), [(2, 2, 2, 2)]

        try:
            start = tuple(scenario_data["start"])
            goal = tuple(scenario_data["goal"])
            raw_obs = scenario_data.get("obstacles", [])
            obstacles = [tuple(o) for o in raw_obs]
            return start, goal, obstacles
        except KeyError as e:
            self.show_error_message(f"Scenario file incomplete: missing {e}")
            # 回退默认
            return (0, 0, 0, 0), (5, 5, 5, 5), [(2, 2, 2, 2)]
        except Exception as e:
            self.show_error_message(f"Failed to load scenario: {e}")
            return (0, 0, 0, 0), (5, 5, 5, 5), [(2, 2, 2, 2)]

    def algorithm_changed(self, selected_algorithm: str):
        """
        当用户在下拉框中切换算法时执行的逻辑
        """
        # 加载配置文件(含 cost_function) 以便给A*或H4PSEA使用
        config_data = self.load_path_planning_config()
        cost_dict = config_data.get("cost_function", {})
        cost_params = self.parse_cost_function(cost_dict)

        if selected_algorithm == "A* Algorithm":
            if not AStarPlanner:
                self.show_error_message("A* Algorithm module not available.")
                self.label_status.setText("System Status: A* Algorithm Unavailable")
                self.path_planner = None
                return

            # 给 AStarPlanner 传入 cost_params(已在你的 AStarPlanner 里加了默认构造)
            self.path_planner = AStarPlanner(cost_params=cost_params)
            self.label_status.setText("System Status: A* Algorithm Selected")

        elif selected_algorithm == "Hybrid 4D PSO Algorithm (H4PSEA)":
            if not H4PSEAPlanner:
                self.show_error_message("H4PSEA Algorithm module not available.")
                self.label_status.setText("System Status: H4PSEA Algorithm Unavailable")
                self.path_planner = None
                return

            self.path_planner = H4PSEAPlanner(cost_params=cost_params)
            self.label_status.setText("System Status: H4PSEA Algorithm Selected")

        else:
            self.label_status.setText("System Status: Unknown Algorithm Selected")
            self.path_planner = None

    def start_path_planning(self):
        """
        启动路径规划
        """
        if not self.path_planner:
            self.show_error_message("No valid planning algorithm selected.")
            self.label_status.setText("System Status: Please select a valid planning algorithm")
            return

        self.label_status.setText("System Status: Planning Path...")
        if self.logger:
            self.logger.log("Path planning started", "INFO")

        start, goal, obstacles = self.load_current_scenario()
        try:
            path = self.path_planner.plan(start=start, goal=goal, obstacles=obstacles)
            if self.logger:
                self.logger.log(f"Path planned: {path}", "INFO")
            if self.visualizer:
                self.visualizer.visualize_path(path)
            self.label_status.setText(f"System Status: Path Planned ({len(path)} waypoints)")
            self.path_status.setText("Path Planning: Completed")
        except Exception as e:
            self.show_error_message(f"Path planning failed: {e}")
            if self.logger:
                self.logger.log(f"Path planning failed: {e}", "ERROR")
            self.path_status.setText("Path Planning: Failed")

    def update_visualization(self):
        """
        更新可视化内容
        """
        if not self.visualizer:
            self.show_error_message("No visualizer available.")
            return

        try:
            # 如果有 DroneController，可获取当前位置
            if self.drone_controller and hasattr(self.drone_controller, "get_position"):
                drone_position = self.drone_controller.get_position()
            else:
                drone_position = (0, 0, 0)

            # 如果有 ObstacleDetector，可获取障碍物信息
            if self.obstacle_detector and hasattr(self.obstacle_detector, "get_obstacles"):
                obstacles = self.obstacle_detector.get_obstacles()
            else:
                obstacles = []

            # 如果已有路径规划结果，则可视化
            path = []
            if self.path_planner and hasattr(self.path_planner, "last_path"):
                path = getattr(self.path_planner, "last_path", [])

            self.visualizer.update_visualization(None, path, obstacles, drone_position)
            self.label_status.setText("System Status: Visualization Updated")
        except Exception as e:
            self.show_error_message(f"Visualization update failed: {e}")
            if self.logger:
                self.logger.log(f"Visualization update failed: {e}", "ERROR")

    def update_system_status(self):
        """
        实时更新CPU、内存使用等系统状态
        """
        try:
            metrics = self.get_system_metrics()
            self.cpu_usage_label.setText(f"CPU Usage: {metrics['CPU']}%")
            self.memory_usage_label.setText(f"Memory Usage: {metrics['Memory']}%")
        except Exception as e:
            self.show_error_message(f"System status update failed: {e}")
            if self.logger:
                self.logger.log(f"System status update failed: {e}", "ERROR")

    def get_system_metrics(self) -> dict:
        """
        获取系统CPU、内存使用率
        """
        cpu_usage = psutil.cpu_percent(interval=None)
        mem_usage = psutil.virtual_memory().percent
        return {"CPU": cpu_usage, "Memory": mem_usage}

    def load_pretrained_model(self):
        """
        允许用户手动加载障碍物检测模型
        """
        if not ObstacleDetector:
            self.show_error_message("ObstacleDetector module not available.")
            return

        file_path, _ = QFileDialog.getOpenFileName(self, "Load Obstacle Detector Model", "", "Model Files (*.pth *.pt)")
        if file_path:
            self.label_status.setText("System Status: Loading Model...")
            self.model_loader_thread = ModelLoaderThread(file_path)
            self.model_loader_thread.loaded.connect(self.on_model_load_finished)
            self.model_loader_thread.start()
        else:
            self.show_error_message("No file selected for obstacle detector model.")

    def on_model_load_finished(self, model_path: str, success: bool):
        """
        模型加载完成的回调
        """
        if success:
            try:
                self.obstacle_detector = ObstacleDetector(mask_model_path=model_path)
                if self.logger:
                    self.logger.log(f"Obstacle detector loaded from {model_path}", "INFO")
                self.label_status.setText("System Status: Model Loaded")
            except Exception as e:
                self.show_error_message(f"Failed to update ObstacleDetector: {e}")
                if self.logger:
                    self.logger.log(f"Failed to update ObstacleDetector: {e}", "ERROR")
        else:
            self.show_error_message("Failed to load model file.")

    def on_drone_controller_loaded(self, controller, message):
        """
        DroneController加载完成后的回调函数
        """
        if controller is not None:
            self.drone_controller = controller
            self.label_status.setText("System Status: DroneController Loaded")
            if self.logger:
                self.logger.log(message, "INFO")
        else:
            self.show_error_message(message)

    def show_error_message(self, message: str):
        """
        错误提示弹窗，并在控制台打印错误信息
        """
        print(f"ERROR: {message}")
        msg_box = QMessageBox(self)
        msg_box.setIcon(QMessageBox.Critical)
        msg_box.setWindowTitle("Error")
        msg_box.setText(message)
        msg_box.exec_()

    def closeEvent(self, event):
        """
        关闭程序时记录日志，并执行必要的清理
        """
        if self.logger:
            self.logger.log("Application closed", "INFO")
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
