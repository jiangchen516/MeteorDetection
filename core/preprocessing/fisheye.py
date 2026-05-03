"""鱼眼镜头校正."""

from pathlib import Path

from defisheye import Defisheye

from core.utils.config import get_section
from core.utils.logging import get_logger

logger = get_logger(__name__)


def rectify_fisheye(
    img_path: str | Path,
    img_out_path: str | Path,
    dtype: str | None = None,
    format: str | None = None,
    fov: float | None = None,
    pfov: float | None = None,
) -> None:
    """校正鱼眼图像.

    Args:
        img_path: 输入图像路径.
        img_out_path: 输出图像路径.
        dtype: 校正类型，默认从配置读取.
        format: 输出格式，默认从配置读取.
        fov: 视场角，默认从配置读取.
        pfov: 输出视场角，默认从配置读取.
    """
    cfg = get_section("fisheye")
    dtype = dtype or cfg.get("dtype", "linear")
    format = format or cfg.get("format", "fullframe")
    fov = fov or cfg.get("fov", 180)
    pfov = pfov or cfg.get("pfov", 120)

    logger.info(f"Rectifying fisheye image: {img_path}")
    obj = Defisheye(str(img_path), dtype=dtype, format=format, fov=fov, pfov=pfov)
    obj.convert(outfile=str(img_out_path))
    logger.info(f"Saved rectified image: {img_out_path}")
