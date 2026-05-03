"""直线检测 + 流星候选验证（评分制）."""

from dataclasses import dataclass, field

import cv2
import numpy as np

from core.utils.config import get_section
from core.utils.logging import get_logger

logger = get_logger(__name__)


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


@dataclass
class LineDetector:
    """直线检测器 — 使用 Hough 变换 + 评分制验证."""

    rho: float = 1.0
    theta: float = 0.017453292519943295
    threshold: int = 30
    min_line_length: int = 30
    max_line_gap: int = 10
    score_threshold: float = 0.5
    length_weight: float = 0.3
    brightness_weight: float = 0.4
    aspect_ratio_weight: float = 0.3
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
        """检测图像中的直线，返回候选流星列表.

        Args:
            image: 输入图像（去噪后的灰度图或 BGR 图）.

        Returns:
            候选流星检测结果列表，无结果返回空列表.
        """
        gray = (
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if len(image.shape) == 3
            else image
        )

        kernel_size = 5
        blur = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
        edges = cv2.Canny(blur, 50, 150, apertureSize=3)

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

        results = []
        for line in raw_lines:
            x1, y1, x2, y2 = line[0]
            score, brightness = self._score_line(x1, y1, x2, y2, gray)
            if score >= self.score_threshold:
                length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                results.append(
                    DetectionResult(
                        x1=int(x1),
                        y1=int(y1),
                        x2=int(x2),
                        y2=int(y2),
                        brightness=brightness,
                        confidence=score,
                        score=score,
                    )
                )

        results.sort(key=lambda r: r.score, reverse=True)
        logger.info(f"Detected {len(results)} candidate meteor lines")
        return results

    def _score_line(
        self, x1: int, y1: int, x2: int, y2: int, gray: np.ndarray
    ) -> tuple[float, float]:
        """对检测到的直线评分.

        Returns:
            (综合评分, 平均亮度).
        """
        length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        # 1. 长度评分（越长越好，饱和于 3 倍 min_line_length）
        length_score = min(length / (3 * self.min_line_length), 1.0)

        # 2. 亮度评分 — 沿直线采样像素均值
        brightness = self._sample_brightness(x1, y1, x2, y2, gray)
        brightness_score = min(brightness / 255.0, 1.0)

        # 3. 长宽比评分（流星是狭长的）
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
        num_samples = max(
            abs(x2 - x1), abs(y2 - y1), 1
        )
        xs = np.linspace(x1, x2, num_samples, dtype=int)
        ys = np.linspace(y1, y2, num_samples, dtype=int)

        h, w = gray.shape
        xs = np.clip(xs, 0, w - 1)
        ys = np.clip(ys, 0, h - 1)

        pixels = gray[ys, xs]
        return float(np.mean(pixels))
