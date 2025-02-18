import yaml
import os

# 配置加载器类
class ConfigLoader:
    # 初始化配置加载器，指定配置文件的存放目录
    def __init__(self, config_dir: str):
        # 存放配置文件的目录
        self.config_dir = config_dir
        # 用于缓存已经加载的配置
        self.configs = {}

    # 加载指定的YAML配置文件
    def load_config(self, config_file: str) -> dict:
        config_path = os.path.join(self.config_dir, config_file)
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file '{config_file}' not found in {self.config_dir}")
        with open(config_path, 'r', encoding='utf-8') as file:
            config_data = yaml.safe_load(file)
            self.configs[config_file] = config_data
            return config_data

    # 获取相机标定参数配置
    def get_camera_params(self) -> dict:
        if 'camera_params.yaml' not in self.configs:
            self.load_config('camera_params.yaml')
        return self.configs['camera_params.yaml']

    # 获取无人机性能参数配置
    def get_drone_params(self) -> dict:
        if 'drone_config.yaml' not in self.configs:
            self.load_config('drone_config.yaml')
        return self.configs['drone_config.yaml']

    # 获取路径规划算法参数配置
    def get_path_planning_params(self) -> dict:
        if 'path_planning.yaml' not in self.configs:
            self.load_config('path_planning.yaml')
        return self.configs['path_planning.yaml']

    # 重新加载所有配置文件
    def reload_all_configs(self) -> None:
        self.configs.clear()
        self.load_config('camera_params.yaml')
        self.load_config('drone_config.yaml')
        self.load_config('path_planning.yaml')
