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

app = FastAPI(title="Meteor Detection", version="0.2.0")

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

    # 获取结果图片列表
    result_images = []
    if task.get("result_images"):
        result_images = json.loads(task["result_images"]) if isinstance(task["result_images"], str) else task["result_images"]
    elif task.get("result_image"):
        result_images = [task["result_image"]]

    # 获取原始图片路径
    try:
        params = json.loads(task.get("params", "{}"))
        image_dir = params.get("image_dir", "")
    except:
        params = {}
        image_dir = ""

    # 优先使用第一个结果图，否则使用原始图片目录中的第一张
    image_path = ""
    if result_images:
        image_path = result_images[0]
    elif image_dir:
        # 从原始目录获取第一张图片
        from pathlib import Path
        img_dir = Path(image_dir)
        if img_dir.exists():
            for f in sorted(img_dir.iterdir()):
                if f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp'):
                    image_path = str(f)
                    break

    # 将绝对路径转换为 URL 路径
    def to_url_path(path):
        if path and "storage/" in path:
            idx = path.index("storage/")
            return "/" + path[idx:]
        return path

    image_path = to_url_path(image_path)
    result_images_urls = [to_url_path(p) for p in result_images]

    # 获取原始图片路径（用于调参时重新检测）
    original_images_urls = []
    if image_dir:
        from pathlib import Path
        img_dir = Path(image_dir)
        if img_dir.exists():
            for f in sorted(img_dir.iterdir()):
                if f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp'):
                    original_images_urls.append(to_url_path(str(f)))

    # 获取保存的检测参数
    detection_params = {}
    if "block_size" in params:
        detection_params = params

    from datetime import datetime
    now = int(datetime.utcnow().timestamp())

    return templates.TemplateResponse(request, "tune_task.html", {
        "task": task,
        "image_path": image_path,
        "result_images": result_images_urls,
        "original_images": original_images_urls,
        "image_dir": image_dir,
        "detection_params": detection_params,
        "now": now,
    })


@app.get("/history")
async def history_page(request: Request):
    return templates.TemplateResponse(request, "history.html", {})
