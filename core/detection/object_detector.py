"""AI 模型检测器 — TensorFlow/TensorRT."""

from pathlib import Path

import cv2
import numpy as np

from core.utils.config import get_section, get_path
from core.utils.logging import get_logger

logger = get_logger(__name__)

_detect_fn = None
_category_index = None


def _load_model():
    """懒加载模型和标签."""
    global _detect_fn, _category_index
    if _detect_fn is not None:
        return

    from object_detection.utils import label_map_util
    import tensorflow as tf

    model_cfg = get_section("model")
    model_dir = model_cfg.get("model_dir", "data/trt_fp16_dir/saved_model")
    label_map_path = model_cfg.get("label_map", "data/labelmap.txt")

    if not Path(model_dir).is_absolute():
        model_dir = str(get_path("base_dir") / model_dir)
    if not Path(label_map_path).is_absolute():
        label_map_path = str(get_path("base_dir") / label_map_path)

    logger.info(f"Loading model from {model_dir}")
    tf.keras.backend.clear_session()
    _detect_fn = tf.saved_model.load(model_dir)

    label_map = label_map_util.load_labelmap(label_map_path)
    categories = label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes=90, use_display_name=True
    )
    _category_index = label_map_util.create_category_index(categories)
    logger.info("Model loaded successfully")


def detect_from_image(
    image: np.ndarray, confidence_threshold: float | None = None
) -> list[list]:
    """使用 AI 模型检测图像中的目标.

    Args:
        image: BGR 格式的 numpy 数组.
        confidence_threshold: 置信度阈值，None 时使用配置值.

    Returns:
        检测结果列表 [[x_min, y_min, x_max, y_max, label, score], ...].
    """
    _load_model()

    model_cfg = get_section("model")
    if confidence_threshold is None:
        confidence_threshold = model_cfg.get("confidence_threshold", 0.1)

    im_height, im_width, _ = image.shape
    input_tensor = np.expand_dims(image, 0)
    detections = _detect_fn(input_tensor)

    bboxes = detections["detection_boxes"][0].numpy()
    bclasses = detections["detection_classes"][0].numpy().astype(np.int32)
    bscores = detections["detection_scores"][0].numpy()

    results = []
    for idx, bbox in enumerate(bboxes):
        if bscores[idx] >= confidence_threshold:
            y_min = int(bbox[0] * im_height)
            x_min = int(bbox[1] * im_width)
            y_max = int(bbox[2] * im_height)
            x_max = int(bbox[3] * im_width)
            label = _category_index[int(bclasses[idx])]["name"]
            results.append([x_min, y_min, x_max, y_max, label, float(bscores[idx])])

    logger.info(f"Object detector found {len(results)} objects")
    return results
