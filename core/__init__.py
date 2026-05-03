"""Meteor Detection Core Engine."""

from core.utils.config import load_config
from core.utils.logging import setup_logging

# 初始化配置和日志
setup_logging()
load_config()
