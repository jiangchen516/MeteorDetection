"""直线检测 + 流星候选验证（评分制）+ 线段合并."""

from dataclasses import dataclass, field

import cv2
import numpy as np

from core.utils.config import get_section
from core.utils.logging import get_logger

logger = get_logger(__name__)

# 合并阈值
MERGE_ANGLE_THRESH = 10.0      # 角度差阈值（度）
MERGE_DIST_THRESH = 30.0       # 距离阈值（像素）
MERGE_GAP_THRESH = 50.0        # 间隙阈值（像素）


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
    merged_count: int = 1  # 合并了多少条原始线段


@dataclass
class LineDetector:
    """直线检测器 — 使用 Hough 变换 + 评分制验证 + 线段合并."""

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

    def detect(self, image: np.ndarray) -> list[DetectionResult]:
        """检测图像中的直线，返回候选流星列表（已合并）.

        Args:
            image: 输入图像（去噪后的灰度图或 BGR 图）.

        Returns:
            候选流星检测结果列表，无结果返回空列表.
        """
        gray = _to_gray(image)

        kernel_size = 5
        blur = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)

        # 自适应 Canny 阈值
        median_val = np.median(blur)
        low = int(max(0, 0.33 * median_val))
        high = int(min(255, 1.33 * median_val))
        edges = cv2.Canny(blur, low, high, apertureSize=3)

        # 膨胀连接断裂的边缘
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)

        raw_lines = cv2.HoughLinesP(
            edges,
            self.rho,
            self.theta,
            self.threshold,
            np.array([]),
            self.min_line_length,
            self.max_line_gap,
        )

        if raw_lines is None:
            logger.debug("No lines detected")
            return []

        # 评分
        scored = []
        for line in raw_lines:
            x1, y1, x2, y2 = line[0]
            score, brightness = self._score_line(x1, y1, x2, y2, gray)
            if score >= self.score_threshold:
                scored.append(
                    DetectionResult(
                        x1=int(x1), y1=int(y1), x2=int(x2), y2=int(y2),
                        brightness=brightness, confidence=score, score=score,
                    )
                )

        # 合并共线/近似线段
        if self.merge_lines and len(scored) > 1:
            scored = self._merge_collinear(scored, gray)

        scored.sort(key=lambda r: r.score, reverse=True)
        logger.info(f"Detected {len(results_after := scored)} candidate meteor lines")
        return scored

    def _score_line(
        self, x1: int, y1: int, x2: int, y2: int, gray: np.ndarray
    ) -> tuple[float, float]:
        """对检测到的直线评分."""
        length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        length_score = min(length / (3 * self.min_line_length), 1.0)
        brightness = self._sample_brightness(x1, y1, x2, y2, gray)
        brightness_score = min(brightness / 255.0, 1.0)

        line_width = 2.0
        aspect_ratio = length / line_width
        aspect_score = min(aspect_ratio / 50.0, 1.0)

        total = (
            self.length_weight * length_score
            + self.brightness_weight * brightness_score
            + self.aspect_ratio_weight * aspect_score
        )
        return total, brightness

    @staticmethod
    def _sample_brightness(
        x1: int, y1: int, x2: int, y2: int, gray: np.ndarray
    ) -> float:
        """沿直线采样平均亮度."""
        num_samples = max(abs(x2 - x1), abs(y2 - y1), 1)
        xs = np.linspace(x1, x2, num_samples, dtype=int)
        ys = np.linspace(y1, y2, num_samples, dtype=int)
        h, w = gray.shape
        xs = np.clip(xs, 0, w - 1)
        ys = np.clip(ys, 0, h - 1)
        pixels = gray[ys, xs]
        return float(np.mean(pixels))

    def _merge_collinear(
        self, lines: list[DetectionResult], gray: np.ndarray
    ) -> list[DetectionResult]:
        """合并共线或近似共线的线段."""
        if len(lines) <= 1:
            return lines

        merged = []
        used = set()

        for i, line_a in enumerate(lines):
            if i in used:
                continue

            # 找到所有与 line_a 共线的线段
            group = [line_a]
            used.add(i)

            for j, line_b in enumerate(lines):
                if j in used or j <= i:
                    continue
                if self._are_collinear(line_a, line_b):
                    group.append(line_b)
                    used.add(j)

            if len(group) == 1:
                merged.append(line_a)
            else:
                merged_line = self._merge_group(group, gray)
                merged.append(merged_line)
                logger.debug(f"Merged {len(group)} lines into one")

        return merged

    @staticmethod
    def _are_collinear(a: DetectionResult, b: DetectionResult) -> bool:
        """判断两条线段是否共线."""
        # 角度差
        angle_a = np.arctan2(a.y2 - a.y1, a.x2 - a.x1)
        angle_b = np.arctan2(b.y2 - b.y1, b.x2 - b.x1)
        angle_diff = abs(np.degrees(angle_a - angle_b))
        angle_diff = min(angle_diff, 180 - angle_diff) if angle_diff > 90 else angle_diff
        if angle_diff > MERGE_ANGLE_THRESH:
            return False

        # 距离：b 的中点到 a 线的距离
        mid_b = ((b.x1 + b.x2) / 2, (b.y1 + b.y2) / 2)
        dist = _point_to_line_distance(mid_b, (a.x1, a.y1), (a.x2, a.y2))
        if dist > MERGE_DIST_THRESH:
            return False

        # 间隙：最近端点距离
        gap = _min_endpoint_distance(a, b)
        if gap > MERGE_GAP_THRESH:
            return False

        return True

    @staticmethod
    def _merge_group(
        group: list[DetectionResult], gray: np.ndarray
    ) -> DetectionResult:
        """合并一组共线线段."""
        # 取最远两个端点
        all_x = [p for l in group for p in (l.x1, l.x2)]
        all_y = [p for l in group for p in (l.y1, l.y2)]

        # 用主方向投影找最远点
        main_line = group[0]
        angle = np.arctan2(main_line.y2 - main_line.y1, main_line.x2 - main_line.x1)
        cos_a, sin_a = np.cos(angle), np.sin(angle)

        projections = [x * cos_a + y * sin_a for x, y in zip(all_x, all_y)]
        min_idx = np.argmin(projections)
        max_idx = np.argmax(projections)

        x1, y1 = int(all_x[min_idx]), int(all_y[min_idx])
        x2, y2 = int(all_x[max_idx]), int(all_y[max_idx])

        # 重新评分
        length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        num_samples = max(abs(x2 - x1), abs(y2 - y1), 1)
        xs = np.linspace(x1, x2, num_samples, dtype=int)
        ys = np.linspace(y1, y2, num_samples, dtype=int)
        h, w = gray.shape
        xs = np.clip(xs, 0, w - 1)
        ys = np.clip(ys, 0, h - 1)
        brightness = float(np.mean(gray[ys, xs]))

        return DetectionResult(
            x1=x1, y1=y1, x2=x2, y2=y2,
            brightness=brightness,
            confidence=max(l.confidence for l in group),
            score=max(l.score for l in group) * 1.2,  # 合并加分
            merged_count=len(group),
        )


def _to_gray(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def _point_to_line_distance(
    point: tuple[float, float],
    line_p1: tuple[float, float],
    line_p2: tuple[float, float],
) -> float:
    """点到线段的距离."""
    px, py = point
    x1, y1 = line_p1
    x2, y2 = line_p2
    dx, dy = x2 - x1, y2 - y1
    if dx == 0 and dy == 0:
        return np.sqrt((px - x1) ** 2 + (py - y1) ** 2)
    t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)))
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    return np.sqrt((px - proj_x) ** 2 + (py - proj_y) ** 2)


def _min_endpoint_distance(a: DetectionResult, b: DetectionResult) -> float:
    """两条线段端点间的最小距离."""
    pts_a = [(a.x1, a.y1), (a.x2, a.y2)]
    pts_b = [(b.x1, b.y1), (b.x2, b.y2)]
    min_dist = float("inf")
    for pa in pts_a:
        for pb in pts_b:
            d = np.sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
            min_dist = min(min_dist, d)
    return min_dist
