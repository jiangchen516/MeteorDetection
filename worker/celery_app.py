"""Celery 应用配置."""

import sys
from pathlib import Path

# 确保项目根目录在 sys.path，使 api/ 等模块可导入
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from celery import Celery

from core.utils.config import get_section, get_path

redis_cfg = get_section("redis")
celery_cfg = get_section("celery")

# 构建 Redis URL
if redis_cfg.get("url"):
    broker_url = redis_cfg["url"]
else:
    host = redis_cfg.get("host", "localhost")
    port = redis_cfg.get("port", 6379)
    db = redis_cfg.get("db", 0)
    broker_url = f"redis://{host}:{port}/{db}"

app = Celery("meteo_worker")

app.conf.update(
    broker_url=celery_cfg.get("broker_url") or broker_url,
    result_backend=celery_cfg.get("result_backend") or broker_url,
    task_serializer=celery_cfg.get("task_serializer", "json"),
    result_serializer=celery_cfg.get("result_serializer", "json"),
    accept_content=celery_cfg.get("accept_content", ["json"]),
    timezone=celery_cfg.get("timezone", "UTC"),
    task_track_started=celery_cfg.get("task_track_started", True),
    worker_prefetch_multiplier=celery_cfg.get("worker_prefetch_multiplier", 1),
)

# 自动发现任务模块
app.autodiscover_tasks(["worker.tasks"])
