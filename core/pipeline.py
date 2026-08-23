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
    classify_single_detection,
)

logger = get_logger(__name__)


def _is_fisheye_image(meta) -> bool:
    """Auto-detect if an image is from a fisheye lens.

    Checks:
    - Camera model (known fisheye cameras)
    - Focal length (< 16mm is likely fisheye)
    - Image aspect ratio (circular fisheye is ~1:1)
    """
    # Check camera model for known fisheye cameras
    fisheye_keywords = ["fisheye", "sigma", "okinawa", "ronikon", "samyang"]
    camera_str = f"{meta.camera_make} {meta.camera_model}".lower()
    if any(kw in camera_str for kw in fisheye_keywords):
        return True

    # Check focal length (fisheye is typically < 16mm)
    if 0 < meta.focal_length < 16:
        return True

    # Check aspect ratio (circular fisheye is close to 1:1)
    if meta.width > 0 and meta.height > 0:
        ratio = meta.width / meta.height
        if 0.9 < ratio < 1.1 and meta.width == meta.height:
            return True

    return False


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

    def get_params(self) -> dict:
        """获取当前检测参数."""
        return {
            "block_size": self.detector.adaptive_block_size,
            "c": self.detector.adaptive_c,
            "min_line_length": self.detector.min_line_length,
            "min_area": self.detector.min_area,
            "min_aspect": self.detector.min_aspect,
            "blur": self.detector.blur_kernel_size,
            "close": self.detector.close_kernel_size,
            "morph": self.detector.morph_kernel_size,
        }

    def _classify_detections(self, image: np.ndarray, detections: list) -> None:
        """对每个检测结果做单帧分类，填充 object_class / 置信度 / 原因."""
        if not detections:
            return
        gray = image
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        orig_mean = float(gray.mean())
        orig_std = max(float(gray.std()), 1.0)
        for d in detections:
            line_length = float(np.hypot(d.x2 - d.x1, d.y2 - d.y1))
            result = classify_single_detection(
                 brightness=d.brightness,
                 score=d.score,
                 line_length=line_length,
                 width=getattr(d, "width", 0.0),
                 continuity=getattr(d, "continuity", 0.0),
                 orig_mean=orig_mean,
                 orig_std=orig_std,
              )
            d.object_class = result.object_class.value
            d.classification_confidence = result.confidence
            d.classification_reasons = result.reasons

    def _report(self, message: str, fraction: float) -> None:
        logger.info(f"[{fraction:.0%}] {message}")
        if self.progress_cb:
            self.progress_cb(message, fraction)

    def _check_uniform_size(self, image_paths: list[Path]) -> bool:
        """检查所有图片尺寸是否一致."""
        if len(image_paths) < 2:
            return True
        first = cv2.imread(str(image_paths[0]))
        if first is None:
            return False
        h, w = first.shape[:2]
        for p in image_paths[1:]:
            img = cv2.imread(str(p))
            if img is None:
                continue
            if img.shape[0] != h or img.shape[1] != w:
                logger.info(f"Size mismatch in batch: {w}x{h} vs {img.shape[1]}x{img.shape[0]}, using per-image background")
                return False
        return True

    def run_single(self, image_path: str | Path, output_dir: str | Path,
                   force_background: str | None = None) -> list[dict]:
        """处理单张图像.

        Args:
            image_path: 图片路径.
            output_dir: 输出目录.
            force_background: 强制背景模式，"estimate" 表示每张图用大核模糊估计背景，
                             None 则按正常逻辑（用中值背景或外部背景）.

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

        # Auto-detect fisheye and warn if not rectifying
        if not self.enable_rectify and _is_fisheye_image(meta):
            logger.warning(
                f"Fisheye image detected but rectification is disabled: {image_path.name}. "
                f"Camera: {meta.camera_make} {meta.camera_model}, "
                f"Focal length: {meta.focal_length}mm. "
                f"Consider enabling rectify=True for better results."
            )

        if self.enable_rectify:
            self._report("Rectifying fisheye", 0.15)
            rectify_fisheye(image_path, image_path)
            image = auto_load(image_path)  # Reload rectified image

        # 检测（直接在原图上连通域分析，不依赖背景差分）
        self._report("Detecting lines", 0.6)
        detections = self.detector.detect(image)
        self._classify_detections(image, detections)

        # 输出
        self._report("Generating output", 0.9)
        self._save_output(image, detections, output_dir, image_path.stem)

        self._report("Done", 1.0)
        return [
            {
                "x1": d.x1, "y1": d.y1, "x2": d.x2, "y2": d.y2,
                "brightness": d.brightness, "confidence": d.confidence,
                "score": d.score, "merged_count": d.merged_count,
                "object_class": d.object_class, "classification_confidence": d.classification_confidence,
                "classification_reasons": d.classification_reasons,
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

        # 检查所有图片尺寸是否一致，决定是否能用统一中值背景
        use_median_bg = (
            build_background
            and self.background is None
            and len(images) >= 3
            and self._check_uniform_size(images)
        )

        if use_median_bg:
            self._report("Building median background", 0.05)
            self.background = build_median_background([str(p) for p in images])
            logger.info("Using median background built from image sequence")
        elif build_background and self.background is None and len(images) >= 3:
            logger.info("Images have different sizes, using per-image background estimation")

        all_results = []
        self.tracker.reset()

        for i, img_path in enumerate(images):
            fraction = 0.1 + 0.8 * (i / len(images))
            self._report(f"Processing {img_path.name} ({i+1}/{len(images)})", fraction)
            # 尺寸不一致时，每张图用自己的估计背景（不用中值背景）
            results = self.run_single(
                img_path, output_dir / img_path.stem,
                force_background=None if use_median_bg else "estimate",
            )
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
