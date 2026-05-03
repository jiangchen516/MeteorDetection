"""背景差分测试."""

import cv2
import numpy as np
import pytest

from core.preprocessing.background import (
    remove_background,
    build_median_background,
    MOG2BackgroundSubtractor,
)


class TestSimpleSubtraction:
    def test_detects_foreground(self):
        bg = np.zeros((50, 50, 3), dtype=np.uint8)
        fg = bg.copy()
        fg[20:30, 20:30] = 100
        result = remove_background(fg, bg, method="simple")
        assert result.shape[:2] == bg.shape[:2]
        assert np.any(result > 0)

    def test_size_mismatch(self):
        with pytest.raises(ValueError, match="Size mismatch"):
            remove_background(
                np.zeros((50, 50, 3), dtype=np.uint8),
                np.zeros((60, 60, 3), dtype=np.uint8),
                method="simple",
            )


class TestRegistration:
    def test_fallback_on_black_background(self):
        """全黑背景 astroalign 失败时应回退到 simple."""
        bg = np.zeros((100, 100, 3), dtype=np.uint8)
        fg = bg.copy()
        cv2.line(fg, (10, 50), (90, 50), (255, 255, 255), 2)
        result = remove_background(fg, bg, method="registration")
        assert result.shape[0] > 0

    def test_requires_background(self):
        with pytest.raises(ValueError, match="requires background"):
            remove_background(
                np.zeros((50, 50, 3), dtype=np.uint8),
                method="registration",
                background_image=None,
            )


class TestMedianBackground:
    def test_build_from_arrays(self):
        """从中值帧构建背景."""
        frames = [np.random.randint(0, 50, (50, 50, 3), dtype=np.uint8) for _ in range(5)]
        bg = build_median_background(frames)
        assert bg.shape == (50, 50, 3)

    def test_requires_valid_images(self):
        with pytest.raises(ValueError, match="No valid"):
            build_median_background([])


class TestMOG2:
    def test_create_subtractor(self):
        sub = MOG2BackgroundSubtractor(history=10, threshold=20)
        assert sub._subtractor is not None

    def test_apply_returns_mask(self):
        sub = MOG2BackgroundSubtractor(history=10, threshold=20)
        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        mask = sub.apply(frame)
        assert mask.shape == (100, 100)
