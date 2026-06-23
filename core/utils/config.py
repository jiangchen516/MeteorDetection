"""配置加载器 — 读取 YAML 配置，支持环境变量覆盖."""

import os
from pathlib import Path
from typing import Any

import yaml

# 项目根目录（此文件向上两级）
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

_config: dict[str, Any] | None = None


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """加载 YAML 配置，返回完整配置字典.

    环境变量覆盖规则：METEO__SECTION__KEY=value
    例如：METEO__HOUGH_LINE__THRESHOLD=50
    """
    global _config
    if _config is not None:
        return _config

    if config_path is None:
        config_path = PROJECT_ROOT / "config.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 解析相对路径为绝对路径
    _resolve_paths(config)

    # 环境变量覆盖：METEO__SECTION__KEY=value（支持多级嵌套）
    _apply_env_overrides(config)

    _config = config
    return config


def _apply_env_overrides(config: dict) -> None:
    """用环境变量覆盖配置值.

    格式：METEO__SECTION__KEY=value（SECTION 和 KEY 自动转小写）
    例如：METEO__REDIS__HOST=redis → config["redis"]["host"] = "redis"
    """
    for env_key, env_value in os.environ.items():
        if not env_key.startswith("METEO__"):
            continue
        parts = env_key[len("METEO__"):].lower().split("__")
        # 只处理已知的顶级 section，避免意外注入
        if parts[0] not in config or not isinstance(config[parts[0]], dict):
            continue
        _set_nested(config, parts, env_value)


def _set_nested(config: dict, keys: list[str], value: str) -> None:
    """在嵌套字典中设置值，自动类型转换."""
    node = config
    for key in keys[:-1]:
        if key not in node or not isinstance(node[key], dict):
            node[key] = {}
        node = node[key]
    # 尝试类型转换
    node[keys[-1]] = _coerce(value)


def _coerce(value: str) -> Any:
    """将字符串环境变量值转换为合适的类型."""
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    if value.lower() == "null" or value.lower() == "none":
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def _resolve_paths(config: dict) -> None:
    """将配置中的相对路径解析为绝对路径."""
    paths = config.get("paths", {})
    for key, value in paths.items():
        if isinstance(value, str):
            p = Path(value)
            if not p.is_absolute():
                paths[key] = str(PROJECT_ROOT / p)

    # 解析 test_paths
    test_paths = config.get("test_paths", {})
    for key, value in test_paths.items():
        if isinstance(value, str):
            p = Path(value)
            if not p.is_absolute():
                test_paths[key] = str(PROJECT_ROOT / p)


def get_path(key: str) -> Path:
    """获取基础路径."""
    config = load_config()
    return Path(config["paths"][key])


def get_test_path(key: str) -> Path:
    """获取测试路径."""
    config = load_config()
    return Path(config["test_paths"][key])


def get_section(key: str) -> dict[str, Any]:
    """获取配置节."""
    config = load_config()
    return config.get(key, {})


def reset() -> None:
    """重置配置缓存（测试用）."""
    global _config
    _config = None
