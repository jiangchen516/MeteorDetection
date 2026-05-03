"""图像叠加 — 使用 lighter 混合模式."""

import os
from pathlib import Path

from PIL import Image, ImageChops

from core.utils.logging import get_logger

logger = get_logger(__name__)

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


def merge_images(pic_dir: str | Path, out_path: str | Path) -> Image.Image:
    """将目录中的所有图像用 lighter 模式叠加.

    Args:
        pic_dir: 输入图像目录.
        out_path: 输出图像路径.

    Returns:
        叠加后的 PIL Image.

    Raises:
        FileNotFoundError: 目录不存在.
        ValueError: 目录中没有支持的图像文件.
    """
    pic_dir = Path(pic_dir)
    if not pic_dir.exists():
        raise FileNotFoundError(f"Directory not found: {pic_dir}")

    files = sorted(
        f
        for f in os.listdir(pic_dir)
        if Path(f).suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not files:
        raise ValueError(f"No supported images in {pic_dir}")

    logger.info(f"Merging {len(files)} images from {pic_dir}")
    final_image = Image.open(pic_dir / files[0])

    for filename in files[1:]:
        current_image = Image.open(pic_dir / filename)
        final_image = ImageChops.lighter(final_image, current_image)

    final_image.save(str(out_path), "JPEG")
    logger.info(f"Saved merged image: {out_path}")
    return final_image
