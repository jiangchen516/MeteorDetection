"""流星检测器 — 自适应阈值 + 连通域分析 + 形态学筛选.

检测流程:
  1. 对差分图做高斯模糊 + 自适应阈值分割
  2. 形态学开运算去噪
  3. 连通域分析
  4. 筛选: 面积 > min_area AND 长宽比 > min_aspect
  5. 最小外接旋转矩形 → 线段
  6. 去重 + 合并共线

核心思路:
  - 差分图已经抑制了银河背景
  - 自适应阈值能适应局部亮度变化
  - 流星是拉长的连通域（长宽比 > 3）
  - 恒星边缘形成的碎片是点状的（长宽比 ≈ 1）
"""

from dataclasses import dataclass, field

import cv2
import numpy as np

from core.utils.config import get_section
from core.utils.logging import get_logger

logger = get_logger(__name__)

MERGE_ANGLE_THRESH = 10.0
MERGE_DIST_THRESH = 30.0
MERGE_GAP_THRESH = 50.0


@dataclass
class DetectionResult:
    """检测结果."""
    x1: int
    y1: int
    x2: int
    y2: int
    brightness: float = 0.0
    confidence: float = 0.0
    score: float = 0.0
    merged_count: int = 1
    width: float = 0.0
    continuity: float = 0.0
    # 旋转矩形四个角点 (cv2.minAreaRect 的 boxPoints)，用于画方框
    # 每个点是 (x, y) 元组，共4个点
    box_points: list[tuple[int, int]] = field(default_factory=list)


@dataclass
class LineDetector:
    """流星检测器 — 连通域分析 + 形态学筛选."""

    rho: float = 1.0
    theta: float = 0.017453292519943295
    threshold: int = 30
    min_line_length: int = 30
    max_line_gap: int = 10
    score_threshold: float = 0.5
    length_weight: float = 0.3
    brightness_weight: float = 0.4
    aspect_ratio_weight: float = 0.3
    merge_lines: bool = True
    _config_loaded: bool = field(default=False, repr=False)

    adaptive_block_size: int = 21    # 自适应阈值块大小（更小更灵敏）
    adaptive_c: float = -2.0         # 自适应阈值偏移
    min_line_length: int = 15        # 最小线长（检测更短的流星）
    morph_kernel_size: int = 1       # 形态学开运算核大小
    close_kernel_size: int = 3       # 形态学闭运算核大小（连接断裂轨迹）
    use_fixed_threshold: bool = False  # 使用自适应阈值
    fixed_threshold: int = 15         # 固定阈值（0-255），更低更灵敏

    def __post_init__(self) -> None:
        if not self._config_loaded:
            self._load_config()

    def _load_config(self) -> None:
        cfg = get_section("hough_line")
        meteor_cfg = get_section("meteor_detection")
        weights = meteor_cfg.get("score_weights", {})
        self.rho = cfg.get("rho", self.rho)
        self.theta = cfg.get("theta", self.theta)
        self.threshold = cfg.get("threshold", self.threshold)
        self.min_line_length = meteor_cfg.get("min_line_length", self.min_line_length)
        self.max_line_gap = meteor_cfg.get("max_line_gap", self.max_line_gap)
        self.score_threshold = meteor_cfg.get("score_threshold", self.score_threshold)
        self.length_weight = weights.get("length", self.length_weight)
        self.brightness_weight = weights.get("brightness", self.brightness_weight)
        self.aspect_ratio_weight = weights.get("aspect_ratio", self.aspect_ratio_weight)

    def detect(self, image: np.ndarray, original_image: np.ndarray | None = None) -> list[DetectionResult]:
        """检测图像中的流星候选.

        Args:
            image: 输入图（差分图或原图）.
            original_image: 原图（用于亮度验证，可选）.

        Returns:
            候选流星检测结果列表.
        """
        gray = _to_gray(image)
        h, w = gray.shape[:2]

        # 阈值分割 — 直接在原图上检测明亮的细长区域
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        if self.use_fixed_threshold:
            _, binary = cv2.threshold(blur, self.fixed_threshold, 255, cv2.THRESH_BINARY)
        else:
            binary = cv2.adaptiveThreshold(
                blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, self.adaptive_block_size, self.adaptive_c,
            )

        # 形态学闭运算（连接断裂的流星轨迹）
        if self.close_kernel_size > 0:
            close_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (self.close_kernel_size, self.close_kernel_size)
            )
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_kernel)

        # 形态学开运算去噪
        open_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self.morph_kernel_size, self.morph_kernel_size)
        )
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, open_kernel)

        # 连通域分析
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )

        # 筛选 + 转线段
        candidates = []
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            bw = stats[i, cv2.CC_STAT_WIDTH]
            bh = stats[i, cv2.CC_STAT_HEIGHT]

            # 面积过滤（太小是噪点）
            if area < 5:
                continue

            # 长宽比过滤（流星是拉长的，但断裂的流星可能长宽比低）
            aspect = max(bw, bh) / max(min(bw, bh), 1)
            if aspect < 2.0:
                continue

            # 边缘过滤（排除距边缘<10px的检测）
            cx, cy = centroids[i]
            dist_to_edge = min(cx, cy, w - cx, h - cy)
            if dist_to_edge < 10:
                continue

            # 最小外接旋转矩形
            component_mask = (labels == i).astype(np.uint8) * 255
            contours, _ = cv2.findContours(
                component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if not contours:
                continue

            rect = cv2.minAreaRect(np.vstack(contours))
            (cx, cy), (rw, rh), angle = rect
            length = max(rw, rh)
            width = min(rw, rh)

            if length < self.min_line_length or width > 20:
                continue

            # 获取旋转矩形四个角点用于画方框
            box = cv2.boxPoints(rect)  # 4x2 float32
            box_pts = [(int(p[0]), int(p[1])) for p in box]

            # 用 atan2 从 boxPoints 直接计算长边方向
            dx_long = box_pts[2][0] - box_pts[1][0]
            dy_long = box_pts[2][1] - box_pts[1][1]
            x1, y1 = int(cx - dx_long / 2), int(cy - dy_long / 2)
            x2, y2 = int(cx + dx_long / 2), int(cy + dy_long / 2)

            # 亮度（从原图或差分图）
            if original_image is not None:
                orig_gray = _to_gray(original_image)
            else:
                orig_gray = gray
            mean_bright = float(orig_gray[component_mask > 0].mean()) if area > 0 else 0

            # 边缘过滤：排除靠近图像边缘的检测
            dist_to_edge = min(x1, y1, w - x2, h - y2)
            if dist_to_edge < 5:
                continue

            candidates.append(DetectionResult(
                x1=x1, y1=y1, x2=x2, y2=y2,
                brightness=mean_bright,
                score=min(length / 100.0, 1.0),
                confidence=min(aspect / 10.0, 1.0),
                width=width,
                continuity=area / max(bw * bh, 1),
                box_points=box_pts,
            ))

        logger.info(f"Detected {len(candidates)} candidate meteor lines")
        return candidates


def _to_gray(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image
