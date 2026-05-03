"""任务 API 路由."""

from celery.result import AsyncResult
from fastapi import APIRouter, HTTPException

from api.models.database import get_task, get_detections, get_logs, list_tasks, upsert_task
from api.models.schemas import TaskResponse, TaskDetail, TaskCreate, DetectionResult
from worker.tasks import process_single, process_batch

router = APIRouter()


@router.post("/", response_model=TaskResponse)
async def create_task(req: TaskCreate):
    """创建检测任务."""
    import uuid

    task_id = str(uuid.uuid4())
    task_type_str = req.task_type.value

    # 先写入数据库
    upsert_task(
        task_id=task_id,
        task_type=task_type_str,
        status="PENDING",
        message="Task queued",
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


@router.post("/{task_id}/cancel")
async def cancel_task(task_id: str):
    """取消任务."""
    from worker.celery_app import app
    app.control.revoke(task_id, terminate=True)
    from api.models.database import upsert_task
    upsert_task(task_id=task_id, status="CANCELLED", message="Cancelled by user")
    return {"status": "cancelled"}


def _task_to_response(task: dict | None) -> TaskResponse:
    if not task:
        return TaskResponse(id="", task_type="batch")
    return TaskResponse(
        id=task["id"],
        task_type=task.get("task_type", "batch"),
        status=task.get("status", "PENDING"),
        progress=task.get("progress", 0),
        message=task.get("message", ""),
        result_count=task.get("result_count", 0),
    )
