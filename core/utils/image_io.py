"""图像读写工具 — 支持 JPG/PNG/RAW."""

from pathlib import Path

import cv2
import numpy as np

from core.utils.logging import get_logger

logger = get_logger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
RAW_EXTENSIONS = {".nef", ".cr2", ".arw", ".dng", ".raf", ".orf"}


def load_image(path: str | Path) -> np.ndarray:
    """加载图像（JPG/PNG 等标准格式）.

    Returns:
        BGR 格式的 numpy 数组.
    """
    path = Path(path)
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {path}")
    logger.debug(f"Loaded image: {path} ({img.shape})")
    return img


def load_raw(path: str | Path) -> np.ndarray:
    """加载 RAW 格式图像.

    Returns:
        BGR 格式的 numpy 数组.
    """
    import rawpy

    path = Path(path)
    logger.debug(f"Loading RAW image: {path}")
    with rawpy.imread(str(path)) as raw:
        rgb = raw.postprocess(use_camera_wb=True)
    # RGB → BGR
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    logger.debug(f"Loaded RAW: {path} ({bgr.shape})")
    return bgr


def auto_load(path: str | Path) -> np.ndarray:
    """根据扩展名自动选择加载方式."""
    path = Path(path)
    ext = path.suffix.lower()
    if ext in RAW_EXTENSIONS:
        return load_raw(path)
    return load_image(path)


def save_image(img: np.ndarray, path: str | Path) -> None:
    """保存图像."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)
    logger.debug(f"Saved image: {path}")


def is_supported_format(path: str | Path) -> bool:
    """检查文件格式是否支持."""
    ext = Path(path).suffix.lower()
    return ext in IMAGE_EXTENSIONS or ext in RAW_EXTENSIONS


def list_images(directory: str | Path) -> list[Path]:
    """列出目录中所有支持的图像文件."""
    directory = Path(directory)
    if not directory.exists():
        return []
    files = sorted(
        f
        for f in directory.iterdir()
        if f.is_file() and is_supported_format(f)
    )
    return files
