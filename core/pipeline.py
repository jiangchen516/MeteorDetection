"""检测流水线编排."""

from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from core.detection.line_detector import LineDetector
from core.preprocessing.background import remove_background
from core.preprocessing.fisheye import rectify_fisheye
from core.utils.config import get_section
from core.utils.image_io import auto_load, list_images, save_image
from core.utils.logging import get_logger

logger = get_logger(__name__)


class MeteorPipeline:
    """流星检测流水线.

    步骤:
        1. Load       → 读取图片序列
        2. Preprocess → 鱼眼校正（可选）→ 背景差分
        3. Detect     → 直线检测
        4. Track      → 轨迹跟踪（多帧）
        5. Output     → 标注图 + CSV + JSON
    """

    def __init__(
        self,
        background_image: np.ndarray | None = None,
        background_path: str | Path | None = None,
        enable_rectify: bool = False,
        progress_callback: Callable[[str, float], None] | None = None,
    ):
        """
        Args:
            background_image: 背景图像（numpy 数组）.
            background_path: 背景图像路径（与 background_image 二选一）.
            enable_rectify: 是否启用鱼眼校正.
            progress_callback: 进度回调 (message, fraction).
        """
        if background_image is not None:
            self.background = background_image
        elif background_path is not None:
            self.background = auto_load(background_path)
        else:
            self.background = None

        self.enable_rectify = enable_rectify
        self.progress_cb = progress_callback
        self.detector = LineDetector()
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

        self._report(f"Loading {image_path.name}", 0.1)
        image = auto_load(image_path)

        if self.background is None:
            logger.warning("No background image provided, skipping background subtraction")
            processed = image
        else:
            self._report("Removing background", 0.3)
            processed = remove_background(image, self.background, method="registration")

        self._report("Detecting lines", 0.6)
        detections = self.detector.detect(processed)

        self._report("Generating output", 0.9)
        self._save_output(image, detections, output_dir, image_path.stem)

        self._report("Done", 1.0)
        return [
            {
                "x1": d.x1,
                "y1": d.y1,
                "x2": d.x2,
                "y2": d.y2,
                "brightness": d.brightness,
                "confidence": d.confidence,
                "score": d.score,
            }
            for d in detections
        ]

    def run_batch(
        self, image_dir: str | Path, output_dir: str | Path
    ) -> list[dict]:
        """批量处理目录中的图像.

        Returns:
            所有图像的检测结果汇总.
        """
        image_dir = Path(image_dir)
        output_dir = Path(output_dir)
        images = list_images(image_dir)

        if not images:
            logger.warning(f"No images found in {image_dir}")
            return []

        all_results = []
        for i, img_path in enumerate(images):
            fraction = i / len(images)
            self._report(f"Processing {img_path.name} ({i+1}/{len(images)})", fraction)
            results = self.run_single(img_path, output_dir / img_path.stem)
            for r in results:
                r["image"] = str(img_path)
            all_results.extend(results)

        self._report("Batch complete", 1.0)
        return all_results

    def _save_output(
        self,
        original: np.ndarray,
        detections: list,
        output_dir: Path,
        name: str,
    ) -> None:
        """保存检测结果图像."""
        annotated = original.copy()
        for d in detections:
            cv2.line(annotated, (d.x1, d.y1), (d.x2, d.y2), (0, 255, 0), 2)
            cv2.putText(
                annotated,
                f"{d.confidence:.2f}",
                (d.x1, d.y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )

        save_image(annotated, output_dir / f"{name}_detected.jpg")
        logger.info(f"Saved annotated image: {output_dir / f'{name}_detected.jpg'}")
