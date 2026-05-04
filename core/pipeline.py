"""检测流水线编排."""

from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from core.detection.line_detector import LineDetector
from core.detection.tracker import MeteorTracker
from core.preprocessing.background import remove_background, build_median_background
from core.preprocessing.fisheye import rectify_fisheye
from core.utils.config import get_section
from core.utils.exif_reader import read_exif
from core.utils.image_io import auto_load, list_images, save_image
from core.utils.logging import get_logger
from core.utils.visualization import draw_detections
from core.classification.meteor_classifier import (
    extract_trajectory_features,
    classify_trajectory,
)

logger = get_logger(__name__)


class MeteorPipeline:
    """流星检测流水线.

    步骤:
        1. Load       → 读取图片序列（支持 EXIF 自动读取）
        2. Preprocess → 鱼眼校正（可选）→ 背景差分（支持多种方法）
        3. Detect     → 直线检测 + 线段合并
        4. Track      → 轨迹跟踪（多帧）
        5. Output     → 标注图 + CSV + JSON
    """

    def __init__(
        self,
        background_image: np.ndarray | None = None,
        background_path: str | Path | None = None,
        enable_rectify: bool = False,
        method: str = "registration",
        progress_callback: Callable[[str, float], None] | None = None,
    ):
        if background_image is not None:
            self.background = background_image
        elif background_path is not None:
            self.background = auto_load(background_path)
        else:
            self.background = None

        self.enable_rectify = enable_rectify
        self.method = method
        self.progress_cb = progress_callback
        self.detector = LineDetector()
        self.tracker = MeteorTracker()
        self.results: list[dict] = []

    def _report(self, message: str, fraction: float) -> None:
        logger.info(f"[{fraction:.0%}] {message}")
        if self.progress_cb:
            self.progress_cb(message, fraction)

    def run_single(self, image_path: str | Path, output_dir: str | Path) -> list[dict]:
        """处理单张图像.

        Returns:
            检测结果列表.
        """
        image_path = Path(image_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 读取 EXIF
        meta = read_exif(image_path)
        self._report(f"Loading {image_path.name} ({meta.width}x{meta.height})", 0.05)

        image = auto_load(image_path)

        if self.enable_rectify:
            self._report("Rectifying fisheye", 0.15)
            rectify_fisheye(image_path, image_path)

        # 背景差分（无外部背景时自动用大核模糊估计）
        self._report(f"Removing background (method={self.method})", 0.3)
        processed = remove_background(
            image, self.background, method=self.method
        )

        # 检测（在差分图上连通域分析，原图用于亮度验证）
        self._report("Detecting lines", 0.6)
        detections = self.detector.detect(processed, original_image=image)

        # 输出
        self._report("Generating output", 0.9)
        self._save_output(image, detections, output_dir, image_path.stem)

        self._report("Done", 1.0)
        return [
            {
                "x1": d.x1, "y1": d.y1, "x2": d.x2, "y2": d.y2,
                "brightness": d.brightness, "confidence": d.confidence,
                "score": d.score, "merged_count": d.merged_count,
            }
            for d in detections
        ]

    def run_batch(
        self, image_dir: str | Path, output_dir: str | Path,
        build_background: bool = True,
    ) -> list[dict]:
        """批量处理目录中的图像.

        Args:
            image_dir: 输入图像目录.
            output_dir: 输出目录.
            build_background: 是否从中值帧构建背景.

        Returns:
            所有图像的检测结果汇总.
        """
        image_dir = Path(image_dir)
        output_dir = Path(output_dir)
        images = list_images(image_dir)

        if not images:
            logger.warning(f"No images found in {image_dir}")
            return []

        # 中值背景（批量模式下优先用中值背景，比估计背景更准确）
        if build_background and self.background is None and len(images) >= 3:
            self._report("Building median background", 0.05)
            sample = images[:min(10, len(images))]
            self.background = build_median_background([str(p) for p in sample])
            logger.info("Using median background built from image sequence")

        all_results = []
        self.tracker.reset()

        for i, img_path in enumerate(images):
            fraction = 0.1 + 0.8 * (i / len(images))
            self._report(f"Processing {img_path.name} ({i+1}/{len(images)})", fraction)
            results = self.run_single(img_path, output_dir / img_path.stem)
            for r in results:
                r["image"] = str(img_path)
            all_results.extend(results)

            # 轨迹跟踪
            lines = [(r["x1"], r["y1"], r["x2"], r["y2"]) for r in results]
            self.tracker.update(lines, i)

        # 分类确认的流星
        self._report("Classifying trajectories", 0.95)
        for meteor in self.tracker.confirmed_meteors:
            features = extract_trajectory_features(meteor["points"])
            classification = classify_trajectory(features)
            meteor["classification"] = classification.object_class.value
            meteor["classification_confidence"] = classification.confidence
            meteor["classification_reasons"] = classification.reasons
            logger.info(
                f"Trajectory {meteor['id']}: {classification.object_class.value} "
                f"(conf={classification.confidence:.2f})"
            )

        self._report("Batch complete", 1.0)
        return all_results

    def _save_output(
        self,
        original: np.ndarray,
        detections: list,
        output_dir: Path,
        name: str,
    ) -> None:
        """保存检测结果图像（使用增强可视化）."""
        annotated = draw_detections(original, detections)
        save_image(annotated, output_dir / f"{name}_detected.jpg")
        logger.info(f"Saved annotated image: {output_dir / f'{name}_detected.jpg'}")
