"""FastAPI 应用入口."""

import json

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import RedirectResponse

from api.models.database import init_db
from core.utils.logging import setup_logging, get_logger

setup_logging()
logger = get_logger(__name__)

app = FastAPI(title="流星探测", version="0.2.0")

# 静态文件和模板
app.mount("/static", StaticFiles(directory="api/static"), name="static")
app.mount("/storage", StaticFiles(directory="storage"), name="storage")
templates = Jinja2Templates(directory="api/templates")

# 初始化数据库
init_db()

# 注册路由
from api.routes import tasks, upload, detect  # noqa: E404
app.include_router(tasks.router, prefix="/api/tasks", tags=["tasks"])
app.include_router(upload.router, prefix="/api/upload", tags=["upload"])
app.include_router(detect.router, prefix="/api/detect", tags=["detect"])


@app.get("/")
async def index():
    return RedirectResponse(url="/dashboard")


@app.get("/dashboard")
async def dashboard(request: Request):
    from api.models.database import list_tasks
    task_list = list_tasks()
    return templates.TemplateResponse(request, "dashboard.html", {"tasks": task_list})


@app.get("/tasks/{task_id}")
async def task_detail(request: Request, task_id: str):
    from api.models.database import get_task, get_detections, get_logs
    task = get_task(task_id)
    if task:
        # 解析 result_images JSON 字符串
        raw = task.get("result_images", "[]")
        if isinstance(raw, str) and raw.strip():
            try:
                task["result_images"] = json.loads(raw)
            except json.JSONDecodeError:
                task["result_images"] = []
        else:
            task["result_images"] = []
    else:
        task = None
    detections = get_detections(task_id) if task else []
    logs = get_logs(task_id) if task else []
    return templates.TemplateResponse(
        request, "task_detail.html",
        {"task": task, "detections": detections, "logs": logs},
    )


@app.get("/upload")
async def upload_page(request: Request):
    return templates.TemplateResponse(request, "upload.html", {})


@app.get("/tune")
async def tune_page(request: Request):
    return templates.TemplateResponse(request, "tune.html", {})


@app.get("/tune/{task_id}")
async def tune_task_page(request: Request, task_id: str):
    from api.models.database import get_task
    task = get_task(task_id)
    if not task:
        from fastapi.responses import RedirectResponse
        return RedirectResponse(url="/dashboard")

    # 获取原始图片路径（从 params 中读取）
    image_path = ""
    try:
        params = json.loads(task.get("params", "{}"))
        image_path = params.get("image_path", "")
    except:
        pass

    # 如果 params 中没有，则从 result_image 获取
    if not image_path:
        if task.get("result_image"):
            image_path = task["result_image"]
        elif task.get("result_images"):
            images = json.loads(task["result_images"]) if isinstance(task["result_images"], str) else task["result_images"]
            if images:
                image_path = images[0]

    # 将绝对路径转换为 URL 路径
    if image_path and "storage/" in image_path:
        idx = image_path.index("storage/")
        image_path = "/" + image_path[idx:]

    return templates.TemplateResponse(request, "tune_task.html", {
        "task": task,
        "image_path": image_path,
    })


@app.get("/history")
async def history_page(request: Request):
    return templates.TemplateResponse(request, "history.html", {})
