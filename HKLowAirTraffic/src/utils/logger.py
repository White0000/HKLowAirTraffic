import logging
import os
from datetime import datetime

class Logger:
    # 日志系统类
    def __init__(self, log_dir: str = "logs", log_level: str = "INFO"):
        # 初始化日志存储目录和日志级别
        self.log_dir = log_dir
        self.log_level = log_level.upper()

        # 如不存在则创建日志目录
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        # 设置日志格式
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

        # 创建当天的日志文件
        log_filename = os.path.join(self.log_dir, f"{datetime.now().strftime('%Y-%m-%d')}_log.txt")

        # 配置日志
        logging.basicConfig(
            level=self.log_level,
            format=log_format,
            handlers=[
                logging.FileHandler(log_filename),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def log(self, message: str, level: str = "INFO"):
        # 记录日志消息
        level = level.upper()
        if level == "INFO":
            self.logger.info(message)
        elif level == "DEBUG":
            self.logger.debug(message)
        elif level == "ERROR":
            self.logger.error(message)
        elif level == "WARNING":
            self.logger.warning(message)
        else:
            self.logger.info(message)
