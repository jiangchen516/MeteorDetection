"""流水线单帧处理集成测试 — 验证检测 + 单帧分类贯通."""

import cv2
import numpy as np
import pytest

from core.pipeline import MeteorPipeline
from core.classification.meteor_classifier import ObjectClass


@pytest.fixture
def synthetic_image(tmp_path):
    """生成含一条明亮细线的 BGR 图，写入临时目录."""
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    cv2.line(img, (50, 150), (250, 150), (255, 255, 255), 2)
    img_path = tmp_path / "in.jpg"
    cv2.imwrite(str(img_path), img)
    return img_path


def test_run_single_detects_and_classifies(tmp_path, synthetic_image):
    """单帧处理应产出检测，且每个检测都带分类字段."""
    out = tmp_path / "out"
    results = MeteorPipeline().run_single(synthetic_image, out)

    assert len(results) >= 1
    for r in results:
        assert "object_class" in r
        assert r["object_class"] in {c.value for c in ObjectClass}
        assert "classification_confidence" in r
        assert "classification_reasons" in r
        # 单个像素噪点与检测结果是不同粒度的对象，避免误用字段
        assert "x1" in r and "y1" in r and "x2" in r and "y2" in r


def test_run_single_writes_annotated_image(tmp_path, synthetic_image):
    """单帧处理应写出标注图."""
    out = tmp_path / "out"
    MeteorPipeline().run_single(synthetic_image, out)
    annotated = list(out.rglob("*_detected.jpg"))
    assert len(annotated) >= 1


def test_run_single_empty_image_no_detections(tmp_path):
    """全黑图不应产生检测，也不应报错."""
    black = np.zeros((200, 200, 3), dtype=np.uint8)
    img_path = tmp_path / "blank.jpg"
    cv2.imwrite(str(img_path), black)
    out = tmp_path / "out"
    results = MeteorPipeline().run_single(img_path, out)
    assert results == []
