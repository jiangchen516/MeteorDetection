"""流水线批量处理集成测试 — 端到端冒烟."""

import cv2
import numpy as np
import pytest

from core.pipeline import MeteorPipeline
from core.classification.meteor_classifier import ObjectClass


def _make_frame(bright_line: bool, seed: int):
    """生成一张合成帧：暗背景上可选一条明亮细线，其余为弱噪声."""
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 25, (200, 200, 3), dtype=np.uint8)
    if bright_line:
        cv2.line(img, (30, 100), (170, 100), (255, 255, 255), 2)
    return img


@pytest.fixture
def batch_dir(tmp_path):
    """临时目录：3 帧中 1 帧含亮线（模拟流星出现一帧）."""
    d = tmp_path / "in"
    d.mkdir()
    for i in range(3):
        cv2.imwrite(str(d / f"frame_{i}.jpg"), _make_frame(bright_line=(i == 1), seed=i))
    return d


def test_run_batch_smoke(tmp_path, batch_dir):
    """批量处理应不报错、写出结果、且带分类字段."""
    out = tmp_path / "out"
    results = MeteorPipeline().run_batch(batch_dir, out)

    assert isinstance(results, list)
    for r in results:
        assert "image" in r
        assert "object_class" in r
        assert r["object_class"] in {c.value for c in ObjectClass}

    # 写出标注图
    annotated = list(out.rglob("*_detected.jpg"))
    assert len(annotated) >= 1


def test_run_batch_empty_dir(tmp_path):
    """空目录不应报错，返回空列表."""
    empty = tmp_path / "empty"
    empty.mkdir()
    results = MeteorPipeline().run_batch(empty, tmp_path / "out")
    assert results == []
