"""日志配置 — 同时输出到控制台和轮转日志文件."""

import logging
import logging.handlers
from pathlib import Path

from core.utils.config import get_section, get_path


def setup_logging() -> None:
    """初始化全局日志配置."""
    log_config = get_section("logging")
    log_dir = Path(log_config.get("log_dir", "storage/logs"))
    log_dir.mkdir(parents=True, exist_ok=True)

    level = getattr(logging, log_config.get("level", "DEBUG").upper(), logging.DEBUG)
    console_level = getattr(
        logging, log_config.get("console_level", "INFO").upper(), logging.INFO
    )
    max_bytes = log_config.get("max_bytes", 10 * 1024 * 1024)
    backup_count = log_config.get("backup_count", 5)

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # 避免重复添加 handler
    if root_logger.handlers:
        return

    # 控制台 handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # 文件 handler（按天轮转）
    file_handler = logging.handlers.TimedRotatingFileHandler(
        filename=log_dir / "meteo.log",
        when="midnight",
        interval=1,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """获取命名 logger."""
    return logging.getLogger(name)
