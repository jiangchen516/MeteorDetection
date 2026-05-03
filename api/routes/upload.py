"""文件上传 API 路由."""

import shutil
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import JSONResponse

from core.utils.config import get_section

router = APIRouter()


@router.post("/")
async def upload_image(
    file: UploadFile = File(...),
    background: UploadFile = File(None),
):
    """上传图片文件."""
    storage_cfg = get_section("paths")
    upload_dir = Path(storage_cfg.get("storage_dir", "storage")) / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

    file_path = upload_dir / file.filename
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    bg_path = None
    if background:
        bg_path = upload_dir / background.filename
        with open(bg_path, "wb") as f:
            shutil.copyfileobj(background.file, f)

    return JSONResponse({
        "filename": file.filename,
        "path": str(file_path),
        "background": str(bg_path) if bg_path else None,
        "size": file_path.stat().st_size,
    })


@router.post("/batch")
async def upload_batch(
    files: list[UploadFile] = File(...),
):
    """批量上传图片."""
    storage_cfg = get_section("paths")
    upload_dir = Path(storage_cfg.get("storage_dir", "storage")) / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

    uploaded = []
    for file in files:
        file_path = upload_dir / file.filename
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        uploaded.append({"filename": file.filename, "path": str(file_path)})

    return {"uploaded": uploaded, "count": len(uploaded), "paths": [u["path"] for u in uploaded]}
