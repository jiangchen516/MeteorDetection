"""可视化测试 — draw_detections 支持分类标签."""

import numpy as np
import cv2
import pytest

from core.detection.line_detector import DetectionResult
from core.utils.visualization import draw_detections


def _image():
    return np.zeros((200, 200, 3), dtype=np.uint8)


def test_draw_detections_no_crash_with_classification():
    """带分类字段的检测应能正常绘制，不报错."""
    det = DetectionResult(
        x1=10, y1=50, x2=150, y2=50,
        brightness=120, confidence=0.8, score=0.7,
        object_class="meteor", classification_confidence=0.8,
        box_points=[(10, 45), (150, 45), (150, 55), (10, 55)],
    )
    out = draw_detections(_image(), [det])
    assert out.shape == (200, 200, 3)
    assert out.dtype == np.uint8


def test_draw_detections_empty():
    """无检测也应正常返回标注图."""
    out = draw_detections(_image(), [])
    assert out.shape == (200, 200, 3)


def test_draw_detections_mixed_classifications():
    """流星/噪点混合时也能绘制."""
    dets = [
        DetectionResult(x1=10, y1=50, x2=150, y2=50, brightness=120,
                        confidence=0.8, score=0.7, object_class="meteor",
                        classification_confidence=0.8,
                        box_points=[(10, 45), (150, 45), (150, 55), (10, 55)]),
        DetectionResult(x1=90, y1=90, x2=95, y2=95, brightness=10,
                        confidence=0.2, score=0.1, object_class="noise",
                        classification_confidence=0.6),
    ]
    out = draw_detections(_image(), dets)
    assert out.shape == (200, 200, 3)
