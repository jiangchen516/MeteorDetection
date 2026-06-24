"""任务 API 路由."""

import json
from pathlib import Path

from celery.result import AsyncResult
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from api.models.database import (
    get_task, get_detections, get_logs, list_tasks, upsert_task,
    search_detections, get_classifications,
)
from api.models.schemas import TaskResponse, TaskDetail, TaskCreate, DetectionResult
from worker.tasks import process_single, process_batch, start_realtime

router = APIRouter()


@router.post("/", response_model=TaskResponse)
async def create_task(req: TaskCreate):
    """创建检测任务."""
    import uuid

    task_id = str(uuid.uuid4())
    task_type_str = req.task_type.value

    # 先写入数据库（保存 params 以便重试）
    upsert_task(
        task_id=task_id,
        task_type=task_type_str,
        status="PENDING",
        message="Task queued",
        params=json.dumps(req.params),
    )

    if task_type_str == "single":
        image_path = req.params.get("image_path")
        if not image_path:
            raise HTTPException(400, "image_path required for single task")
        process_single.apply_async(
            args=[image_path],
            kwargs={"background_path": req.params.get("background_path")},
            task_id=task_id,
        )
    elif task_type_str == "batch":
        image_dir = req.params.get("image_dir")
        if not image_dir:
            raise HTTPException(400, "image_dir required for batch task")
        process_batch.apply_async(
            args=[image_dir],
            kwargs={"background_path": req.params.get("background_path")},
            task_id=task_id,
        )
    else:
        raise HTTPException(400, f"Unknown task type: {task_type_str}")

    return _task_to_response(get_task(task_id))


@router.get("/", response_model=list[TaskResponse])
async def list_all_tasks(limit: int = 50, offset: int = 0):
    """列出所有任务."""
    return [_task_to_response(t) for t in list_tasks(limit, offset)]


@router.get("/{task_id}", response_model=TaskDetail)
async def get_task_detail(task_id: str):
    """获取任务详情."""
    task = get_task(task_id)
    if not task:
        raise HTTPException(404, "Task not found")

    task_resp = _task_to_response(task)
    return TaskDetail(
        id=task_resp.id,
        task_type=task_resp.task_type,
        status=task_resp.status,
        progress=task_resp.progress,
        message=task_resp.message,
        result_count=task_resp.result_count,
        created_at=task_resp.created_at,
        updated_at=task_resp.updated_at,
        params=task_resp.params,
        result_images=task_resp.result_images,
        results=[
            DetectionResult(
                x1=d["x1"], y1=d["y1"], x2=d["x2"], y2=d["y2"],
                brightness=d.get("brightness", 0),
                confidence=d.get("confidence", 0),
                score=d.get("score", 0),
                image=d.get("image", ""),
            )
            for d in get_detections(task_id)
        ],
        logs=[l["message"] for l in get_logs(task_id)],
    )


@router.get("/{task_id}/image")
async def get_task_image(task_id: str):
    """获取任务标注图."""
    task = get_task(task_id)
    if not task:
        raise HTTPException(404, "Task not found")
    image_path = task.get("result_image", "")
    if not image_path:
        raise HTTPException(404, "No result image")
    from pathlib import Path
    if not Path(image_path).exists():
        raise HTTPException(404, "Image file not found")
    return FileResponse(image_path, media_type="image/jpeg")


@router.post("/{task_id}/cancel")
async def cancel_task(task_id: str):
    """取消任务."""
    from worker.celery_app import app
    app.control.revoke(task_id, terminate=True)
    from api.models.database import upsert_task
    upsert_task(task_id=task_id, status="CANCELLED", message="Cancelled by user")
    return {"status": "cancelled"}


@router.post("/{task_id}/retry")
async def retry_task(task_id: str):
    """重试任务（任何状态都可以重试）."""
    task = get_task(task_id)
    if not task:
        raise HTTPException(404, "Task not found")

    # 重新提交任务
    task_type = task.get("task_type", "batch")
    params_str = task.get("params", "{}")
    try:
        params = json.loads(params_str)
    except json.JSONDecodeError:
        params = {}

    if task_type == "single":
        image_path = params.get("image_path")
        if not image_path:
            raise HTTPException(400, "Cannot retry: image_path not found in params (task may be created before v0.2)")
        process_single.apply_async(
            args=[image_path],
            kwargs={"background_path": params.get("background_path")},
            task_id=task_id,
        )
    elif task_type == "batch":
        image_dir = params.get("image_dir")
        if not image_dir:
            raise HTTPException(400, "Cannot retry: image_dir not found in params (task may be created before v0.2)")
        process_batch.apply_async(
            args=[image_dir],
            kwargs={"background_path": params.get("background_path")},
            task_id=task_id,
        )
    else:
        raise HTTPException(400, f"Unknown task type: {task_type}")

    # 更新 created_at 以便重试的任务排到列表最前面
    from datetime import datetime
    now = datetime.utcnow().isoformat()
    from api.models.database import get_connection
    conn = get_connection()
    conn.execute("UPDATE tasks SET status=?, message=?, progress=0.0, created_at=? WHERE id=?",
                 ("PENDING", "Retrying...", now, task_id))
    conn.commit()
    conn.close()
    return {"status": "retry", "task_id": task_id}


@router.delete("/{task_id}")
async def delete_task(task_id: str):
    """删除任务."""
    from api.models.database import get_connection
    conn = get_connection()
    conn.execute("DELETE FROM detections WHERE task_id = ?", (task_id,))
    conn.execute("DELETE FROM task_logs WHERE task_id = ?", (task_id,))
    conn.execute("DELETE FROM classifications WHERE task_id = ?", (task_id,))
    conn.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
    conn.commit()
    conn.close()
    return {"status": "deleted", "task_id": task_id}


@router.post("/{task_id}/save")
async def save_result(task_id: str, req: dict):
    """保存检测结果到任务."""
    params = req.get("params", {})

    # 运行检测
    task = get_task(task_id)
    if not task:
        raise HTTPException(404, "Task not found")

    # 获取图片路径
    image_path = task.get("params", "{}")
    try:
        image_params = json.loads(image_path)
        img_path = image_params.get("image_path", "")
    except:
        img_path = ""

    if not img_path:
        raise HTTPException(400, "No image path found in task params")

    # 运行检测
    import time
    start_time = time.time()

    import cv2
    import numpy as np
    from pathlib import Path
    from core.utils.image_io import auto_load, save_image
    from core.utils.visualization import draw_detections
    from core.detection.line_detector import DetectionResult

    path = Path(img_path)
    if not path.exists():
        raise HTTPException(404, f"Image not found: {img_path}")

    image = auto_load(path)
    if image is None:
        raise HTTPException(400, "Failed to load image")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    blur_ksize = params.get("blur", 5)
    blur = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
    binary = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        params.get("block_size", 21),
        params.get("c", -1),
    )

    # 形态学闭运算
    close_ksize = params.get("close", 3)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_kernel)

    # 形态学开运算
    morph_ksize = params.get("morph", 1)
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_ksize, morph_ksize))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, open_kernel)

    # 连通域分析
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

    det_results = []
    h, w = gray.shape[:2]
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh = stats[i, cv2.CC_STAT_HEIGHT]
        aspect = max(bw, bh) / max(min(bw, bh), 1)
        cx, cy = centroids[i]

        if area < params.get("min_area", 5):
            continue
        if aspect < params.get("min_aspect", 2.0):
            continue

        dist_to_edge = min(cx, cy, w - cx, h - cy)
        if dist_to_edge < 10:
            continue

        component_mask = (labels == i).astype(np.uint8) * 255
        contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        rect = cv2.minAreaRect(np.vstack(contours))
        (rcx, rcy), (rw, rh), angle = rect
        length = max(rw, rh)
        width = min(rw, rh)

        if length < params.get("min_line_length", 15) or width > 20:
            continue

        box = cv2.boxPoints(rect)
        box_pts = [(int(p[0]), int(p[1])) for p in box]
        dx_long = box_pts[2][0] - box_pts[1][0]
        dy_long = box_pts[2][1] - box_pts[1][1]
        x1, y1 = int(rcx - dx_long / 2), int(rcy - dy_long / 2)
        x2, y2 = int(rcx + dx_long / 2), int(rcy + dy_long / 2)

        mean_bright = float(gray[component_mask > 0].mean()) if area > 0 else 0

        det_results.append(DetectionResult(
            x1=x1, y1=y1, x2=x2, y2=y2,
            brightness=mean_bright,
            score=min(length / 100.0, 1.0),
            confidence=min(aspect / 10.0, 1.0),
            width=width,
            box_points=box_pts,
        ))

    elapsed = time.time() - start_time

    # 保存标注图
    output_dir = Path("storage/results") / task_id
    output_dir.mkdir(parents=True, exist_ok=True)
    annotated = draw_detections(image, det_results)
    result_path = output_dir / f"{path.stem}_detected.jpg"
    save_image(annotated, result_path)

    # 更新任务
    result_images = [str(result_path)]
    _update_task(
        task_id=task_id,
        status="SUCCESS",
        progress=1.0,
        message=f"Done: {len(det_results)} detections (tuned)",
        result_count=len(det_results),
        result_image=str(result_path),
        result_images=result_images,
    )

    return {
        "status": "saved",
        "detections": len(det_results),
        "time": elapsed,
        "result_image": str(result_path),
    }


@router.get("/{task_id}/images")
async def get_task_images(task_id: str):
    """获取批量任务的所有标注图路径."""
    task = get_task(task_id)
    if not task:
        raise HTTPException(404, "Task not found")
    raw = task.get("result_images", "[]")
    if isinstance(raw, str):
        try:
            images = json.loads(raw)
        except json.JSONDecodeError:
            images = []
    else:
        images = raw or []
    return {"images": images, "count": len(images)}


@router.get("/{task_id}/images/{index}")
async def get_task_image_by_index(task_id: str, index: int):
    """获取批量任务中指定索引的标注图."""
    task = get_task(task_id)
    if not task:
        raise HTTPException(404, "Task not found")
    raw = task.get("result_images", "[]")
    if isinstance(raw, str):
        try:
            images = json.loads(raw)
        except json.JSONDecodeError:
            images = []
    else:
        images = raw or []
    if index < 0 or index >= len(images):
        raise HTTPException(404, f"Image index {index} out of range (0-{len(images)-1})")
    image_path = Path(images[index])
    if not image_path.exists():
        raise HTTPException(404, "Image file not found")
    return FileResponse(str(image_path), media_type="image/jpeg")


@router.get("/search/detections")
async def search_detections_api(
    object_class: str | None = None,
    min_confidence: float = 0.0,
    min_brightness: float = 0.0,
    date_from: str | None = None,
    date_to: str | None = None,
    limit: int = 100,
    offset: int = 0,
):
    """搜索检测结果."""
    results = search_detections(
        object_class=object_class,
        min_confidence=min_confidence,
        min_brightness=min_brightness,
        date_from=date_from,
        date_to=date_to,
        limit=limit,
        offset=offset,
    )
    return {"results": results, "count": len(results)}


@router.post("/{task_id}/realtime")
async def start_realtime_api(
    task_id: str,
    watch_dir: str,
    background_path: str | None = None,
    output_dir: str | None = None,
    poll_interval: float = 2.0,
):
    """启动实时检测任务."""
    upsert_task(
        task_id=task_id,
        task_type="realtime",
        status="PENDING",
        message=f"Watching {watch_dir}",
    )
    start_realtime.apply_async(
        args=[watch_dir],
        kwargs={
            "background_path": background_path,
            "output_dir": output_dir,
            "poll_interval": poll_interval,
        },
        task_id=task_id,
    )
    return {"task_id": task_id, "status": "PENDING"}


def _task_to_response(task: dict | None) -> TaskResponse:
    if not task:
        return TaskResponse(id="", task_type="batch")
    # 解析 result_images JSON
    raw = task.get("result_images", "[]")
    if isinstance(raw, str):
        try:
            images = json.loads(raw)
        except json.JSONDecodeError:
            images = []
    else:
        images = raw or []
    return TaskResponse(
        id=task["id"],
        task_type=task.get("task_type", "batch"),
        status=task.get("status", "PENDING"),
        progress=task.get("progress", 0),
        message=task.get("message", ""),
        result_count=task.get("result_count", 0),
        result_image=task.get("result_image", ""),
        result_images=images,
    )
