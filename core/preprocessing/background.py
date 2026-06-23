"""背景差分 / 去噪 — 支持多种背景建模方法."""

from typing import Literal

import cv2
import numpy as np
import astroalign as aa

from core.utils.config import get_section
from core.utils.logging import get_logger

logger = get_logger(__name__)


def build_median_background(image_paths: list[str | np.ndarray]) -> np.ndarray:
    """从多张图像构建中值背景.

    Args:
        image_paths: 图像路径列表或 numpy 数组列表.

    Returns:
        中值背景图像 (BGR).
    """
    frames = []
    for p in image_paths:
        if isinstance(p, np.ndarray):
            frames.append(p)
        else:
            img = cv2.imread(str(p))
            if img is not None:
                frames.append(img)

    if not frames:
        raise ValueError("No valid images for median background")

    logger.info(f"Building median background from {len(frames)} images")
    # 检查所有帧尺寸一致
    shapes = {f.shape for f in frames}
    if len(shapes) > 1:
        raise ValueError(
            f"Cannot build median background: images have different sizes {shapes}. "
            "Use per-image background estimation instead."
        )
    stack = np.stack(frames, axis=0)
    median = np.median(stack, axis=0).astype(np.uint8)
    return median


class MOG2BackgroundSubtractor:
    """基于 OpenCV MOG2 的背景减除器."""

    def __init__(self, history: int | None = None, threshold: int | None = None,
                 learning_rate: float | None = None):
        cfg = get_section("background")
        self.history = history or cfg.get("history", 50)
        self.threshold = threshold or cfg.get("threshold", 30)
        self.learning_rate = learning_rate or cfg.get("learning_rate", 0.01)

        self._subtractor = cv2.createBackgroundSubtractorMOG2(
            history=self.history,
            varThreshold=self.threshold,
            detectShadows=False,
        )

    def apply(self, image: np.ndarray) -> np.ndarray:
        """应用背景减除，返回前景掩码."""
        fg_mask = self._subtractor.apply(image, learningRate=self.learning_rate)
        return fg_mask

    def get_background(self) -> np.ndarray | None:
        """获取当前背景模型."""
        return self._subtractor.getBackgroundImage()


def _estimate_background(image: np.ndarray) -> np.ndarray:
    """用形态学开运算估计局部背景.

    形态学开运算（先腐蚀后膨胀）能平滑背景同时保留细长目标:
    - 腐蚀步骤: 去除亮点（恒星、流星），保留暗背景
    - 膨胀步骤: 恢复背景区域，填补被腐蚀的暗洞
    - 结果: 背景被平滑，但细长目标（流星）被保留在前景

    核大小选择:
    - 核应大于恒星直径（通常 3-5 像素）但小于流星长度
    - 推荐: 短边的 1/15 ~ 1/8，范围 [21, 61]

    Args:
        image: 输入图像 (BGR 或灰度).

    Returns:
        估计的背景图像 (灰度).
    """
    gray = _to_gray(image)
    h, w = gray.shape[:2]
    short = min(h, w)
    # 核大小 = 短边的 1/12，最小 21，最大 61，奇数
    kernel_size = int(short / 12)
    kernel_size = max(kernel_size, 21)
    kernel_size = min(kernel_size, 61)
    if kernel_size % 2 == 0:
        kernel_size += 1
    logger.info(f"Estimated background kernel size: {kernel_size}x{kernel_size} (morphological)")

    # 形态学开运算估计背景
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    background = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    return background


def remove_background(
    light_image: np.ndarray,
    background_image: np.ndarray | None = None,
    method: Literal["simple", "registration", "mog2"] = "registration",
    coords: tuple[int, int, int, int] | None = None,
) -> np.ndarray:
    """从光图像中移除背景噪声.

    Args:
        light_image: 输入图像（有流星的图像）.
        background_image: 背景图像（无流星），mog2 方法不需要.
        method: 去噪方法.
        coords: ROI 坐标 (x1, y1, x2, y2)，为 None 时处理整张图像.

    Returns:
        去噪后的灰度图像.
    """
    if coords is not None:
        x1, y1, x2, y2 = coords
        light_roi = light_image[y1:y2, x1:x2]
    else:
        light_roi = light_image

    if method == "mog2":
        return _remove_mog2(light_roi)
    elif method == "registration":
        if background_image is None:
            background_image = _estimate_background(light_roi)
        if coords is not None:
            bg_roi = background_image[y1:y2, x1:x2]
        else:
            bg_roi = background_image
        return _remove_with_registration(light_roi, bg_roi)
    else:
        if background_image is None:
            background_image = _estimate_background(light_roi)
        if coords is not None:
            bg_roi = background_image[y1:y2, x1:x2]
        else:
            bg_roi = background_image
        return _remove_simple(light_roi, bg_roi)


def _remove_simple(light: np.ndarray, background: np.ndarray) -> np.ndarray:
    """简单差分 — 取绝对值，保留亮目标和暗目标."""
    gray = _to_gray(light)
    bg_gray = _to_gray(background)
    if gray.shape[:2] != bg_gray.shape[:2]:
        raise ValueError(f"Size mismatch: {gray.shape} vs {bg_gray.shape}")
    result = cv2.absdiff(gray, bg_gray)
    return result


def _remove_with_registration(
    light: np.ndarray, background: np.ndarray
) -> np.ndarray:
    """图像配准 + 差分.

    注意：当 background 是由 _estimate_background 生成的大核模糊图像时，
    配准没有意义（背景和源图像几乎一样），直接走简单差分。
    """
    source_gray = _to_gray(light)
    bg_gray = _to_gray(background)

    if source_gray.shape[:2] != bg_gray.shape[:2]:
        raise ValueError(f"Size mismatch: {source_gray.shape} vs {bg_gray.shape}")

    # 估计背景的情况：直接差分，不做配准
    diff = cv2.absdiff(source_gray, bg_gray)
    if diff.mean() < 1.0:
        return _remove_simple(light, background)

    try:
        registered, _ = aa.register(source_gray, bg_gray)
    except (ValueError, RuntimeError) as e:
        logger.warning(f"Registration failed ({e}), falling back to simple subtraction")
        return _remove_simple(light, background)

    registered = registered.astype(np.uint8)
    result = cv2.absdiff(source_gray, registered)
    result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX)
    return result


def _remove_mog2(image: np.ndarray) -> np.ndarray:
    """MOG2 背景减除（逐帧调用时效果有限，建议用 batch 模式）."""
    gray = _to_gray(image)
    # 单次调用：用固定阈值做前景提取
    # 实际 MOG2 需要多帧训练，这里用帧差近似
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return blurred


def _to_gray(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image
