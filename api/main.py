"""FastAPI 应用入口."""

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
templates = Jinja2Templates(directory="api/templates")

# 初始化数据库
init_db()

# 注册路由
from api.routes import tasks, upload  # noqa: E404
app.include_router(tasks.router, prefix="/api/tasks", tags=["tasks"])
app.include_router(upload.router, prefix="/api/upload", tags=["upload"])


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
    detections = get_detections(task_id)
    logs = get_logs(task_id)
    return templates.TemplateResponse(
        request, "task_detail.html",
        {"task": task, "detections": detections, "logs": logs},
    )


@app.get("/upload")
async def upload_page(request: Request):
    return templates.TemplateResponse(request, "upload.html", {})


@app.get("/history")
async def history_page(request: Request):
    return templates.TemplateResponse(request, "history.html", {})
