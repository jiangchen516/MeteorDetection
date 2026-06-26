"""Celery 任务定义."""

import json
import uuid
from datetime import datetime
from pathlib import Path

from worker.celery_app import app
from core.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


def _update_task(task_id: str, task_type: str | None = None, **kwargs) -> None:
    """更新任务状态到数据库."""
    from api.models.database import upsert_task, get_task
    if task_type is None:
        # 尝试从数据库获取已有 task_type
        existing = get_task(task_id)
        task_type = existing.get("task_type", "batch") if existing else "batch"
    upsert_task(task_id=task_id, task_type=task_type, **kwargs)


def _add_detection(task_id: str, det: dict) -> None:
    from api.models.database import add_detection
    add_detection(task_id, det)


@app.task(bind=True, name="worker.tasks.process_single")
def process_single(self, image_path: str, background_path: str | None = None,
                   output_dir: str | None = None, enable_rectify: bool = False) -> dict:
    """处理单张图片."""
    task_id = self.request.id
    setup_logging()

    _update_task(task_id=task_id, task_type="single", status="RUNNING",
                 message="Loading image...", progress=0.1)

    try:
        from core.pipeline import MeteorPipeline
        from core.utils.config import get_path

        if output_dir is None:
            output_dir = str(get_path("storage_dir") / "results" / task_id)

        bg = None
        if background_path:
            from core.utils.image_io import auto_load
            bg = auto_load(background_path)

        pipeline = MeteorPipeline(background_image=bg, enable_rectify=enable_rectify)

        def progress_cb(msg, frac):
            _update_task(task_id=task_id, progress=frac, message=msg)

        pipeline.progress_cb = progress_cb
        results = pipeline.run_single(image_path, output_dir)

        for r in results:
            _add_detection(task_id, r)

        annotated_path = str(Path(output_dir) / f"{Path(image_path).stem}_detected.jpg")
        _update_task(task_id=task_id, status="SUCCESS", progress=1.0,
                     message=f"Done: {len(results)} detections",
                     result_count=len(results),
                     result_image=annotated_path,
                     params=json.dumps(pipeline.get_params()))

        return {"detections": len(results), "output_dir": output_dir}

    except Exception as e:
        logger.exception(f"Single processing failed: {e}")
        _update_task(task_id=task_id, status="FAILED", message=str(e))
        raise


@app.task(bind=True, name="worker.tasks.process_batch")
def process_batch(self, image_dir: str, background_path: str | None = None,
                  output_dir: str | None = None, enable_rectify: bool = False) -> dict:
    """批量处理目录."""
    task_id = self.request.id
    setup_logging()

    _update_task(task_id=task_id, task_type="batch", status="RUNNING",
                 message="Starting batch...", progress=0.0)

    try:
        from core.pipeline import MeteorPipeline
        from core.utils.config import get_path
        from core.utils.image_io import auto_load

        if output_dir is None:
            output_dir = str(get_path("storage_dir") / "results" / task_id)

        bg = None
        if background_path:
            bg = auto_load(background_path)

        pipeline = MeteorPipeline(background_image=bg, enable_rectify=enable_rectify)

        def progress_cb(msg, frac):
            _update_task(task_id=task_id, progress=frac, message=msg)

        pipeline.progress_cb = progress_cb
        results = pipeline.run_batch(image_dir, output_dir)

        for r in results:
            _add_detection(task_id, r)

        # 收集所有标注图路径（递归搜索）
        from pathlib import Path
        result_images = sorted(Path(output_dir).rglob("*_detected.jpg"))
        result_images_str = [str(p) for p in result_images]

        _update_task(task_id=task_id, status="SUCCESS", progress=1.0,
                     message=f"Done: {len(results)} detections",
                     result_count=len(results),
                     result_images=result_images_str,
                     params=json.dumps(pipeline.get_params()))

        return {"detections": len(results), "output_dir": output_dir}

    except Exception as e:
        logger.exception(f"Batch processing failed: {e}")
        _update_task(task_id=task_id, status="FAILED", message=str(e))
        raise


@app.task(bind=True, name="worker.tasks.start_realtime")
def start_realtime(self, watch_dir: str, background_path: str | None = None,
                   output_dir: str | None = None, poll_interval: float = 2.0) -> dict:
    """启动实时检测."""
    task_id = self.request.id
    setup_logging()

    _update_task(task_id=task_id, task_type="realtime", status="RUNNING",
                 message=f"Watching {watch_dir}", progress=0.0)

    try:
        from core.utils.config import get_path
        from worker.runners.realtime import RealtimeDetector

        if output_dir is None:
            output_dir = str(get_path("storage_dir") / "results" / task_id)

        def on_detection(image_path: str, results: list):
            for r in results:
                _add_detection(task_id, r)

        def on_meteor(meteor_info: dict):
            _add_detection(task_id, meteor_info)

        detector = RealtimeDetector(
            watch_dir=watch_dir,
            output_dir=output_dir,
            background_path=background_path,
            poll_interval=poll_interval,
            on_detection=on_detection,
            on_meteor=on_meteor,
        )

        # 注意：这会阻塞，直到 stop_realtime 被调用
        detector.start()

        _update_task(task_id=task_id, status="SUCCESS", message="Realtime stopped")
        return {"status": "stopped"}

    except Exception as e:
        logger.exception(f"Realtime detection failed: {e}")
        _update_task(task_id=task_id, status="FAILED", message=str(e))
        raise
