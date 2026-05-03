"""背景差分 / 去噪."""

from typing import Literal

import cv2
import numpy as np
import astroalign as aa

from core.utils.logging import get_logger

logger = get_logger(__name__)


def remove_background(
    light_image: np.ndarray,
    background_image: np.ndarray,
    method: Literal["simple", "registration"] = "registration",
    coords: tuple[int, int, int, int] | None = None,
) -> np.ndarray:
    """从光图像中移除背景噪声.

    Args:
        light_image: 输入图像（有流星的图像）.
        background_image: 背景图像（无流星）.
        method: 去噪方法，'simple' 为简单差分，'registration' 为配准+差分.
        coords: ROI 坐标 (x1, y1, x2, y2)，为 None 时处理整张图像.

    Returns:
        去噪后的图像.
    """
    if light_image.shape != background_image.shape:
        raise ValueError(
            f"Image sizes mismatch: {light_image.shape} vs {background_image.shape}"
        )

    if coords is not None:
        x1, y1, x2, y2 = coords
        light_roi = light_image[y1:y2, x1:x2]
        bg_roi = background_image[y1:y2, x1:x2]
    else:
        light_roi = light_image
        bg_roi = background_image

    if method == "registration":
        return _remove_with_registration(light_roi, bg_roi)
    else:
        return _remove_simple(light_roi, bg_roi)


def _remove_simple(light: np.ndarray, background: np.ndarray) -> np.ndarray:
    """简单差分."""
    result = np.subtract(light, background)
    black = np.zeros_like(result, dtype=np.uint8)
    return np.uint8(np.maximum(result, black))


def _remove_with_registration(
    light: np.ndarray, background: np.ndarray
) -> np.ndarray:
    """图像配准 + 差分."""
    target_gray = (
        cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
        if len(background.shape) == 3
        else background
    )
    source_gray = (
        cv2.cvtColor(light, cv2.COLOR_BGR2GRAY)
        if len(light.shape) == 3
        else light
    )

    try:
        registered, _ = aa.register(source_gray, target_gray)
    except (ValueError, RuntimeError) as e:
        logger.warning(f"Registration failed ({e}), falling back to simple subtraction")
        return _remove_simple(light, background)

    result = np.subtract(source_gray, registered)
    black = np.zeros_like(result, dtype=np.uint8)
    return np.uint8(np.maximum(result, black))
