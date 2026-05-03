"""背景差分测试."""

import numpy as np
import cv2
import pytest

from core.preprocessing.background import remove_background


def test_simple_subtraction():
    """简单差分应能检出前景."""
    bg = np.zeros((50, 50, 3), dtype=np.uint8)
    fg = bg.copy()
    fg[20:30, 20:30] = 100  # 白色方块

    result = remove_background(fg, bg, method="simple")
    assert result.shape == bg.shape[:2] if len(result.shape) == 2 else bg.shape
    # 差分区域应有非零值
    assert np.any(result > 0)


def test_registration_fallback():
    """全黑背景 astroalign 无法配准时，应回退到简单差分."""
    bg = np.zeros((100, 100, 3), dtype=np.uint8)
    fg = bg.copy()
    cv2.line(fg, (10, 50), (90, 50), (255, 255, 255), 2)

    # 不应抛异常，而是回退到 simple 方法
    result = remove_background(fg, bg, method="registration")
    assert result.shape[0] > 0
    assert result.shape[1] > 0


def test_size_mismatch():
    """尺寸不匹配应抛出 ValueError."""
    img1 = np.zeros((50, 50, 3), dtype=np.uint8)
    img2 = np.zeros((60, 60, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="sizes mismatch"):
        remove_background(img1, img2)


def test_with_coords():
    """指定 ROI 坐标时应正确裁剪."""
    bg = np.zeros((100, 100, 3), dtype=np.uint8)
    fg = bg.copy()
    fg[40:60, 40:60] = 200

    result = remove_background(fg, bg, method="simple", coords=(30, 30, 70, 70))
    assert result.shape[:2] == (40, 40)
