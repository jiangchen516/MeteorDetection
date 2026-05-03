"""直线检测测试."""

import numpy as np
import cv2
import pytest

from core.detection.line_detector import LineDetector


@pytest.fixture
def detector():
    return LineDetector(score_threshold=0.0)  # 阈值为 0，检出所有线


@pytest.fixture
def bright_line_image():
    """含一条明亮白线的图像."""
    img = np.zeros((200, 200), dtype=np.uint8)
    cv2.line(img, (20, 100), (180, 100), 255, 2)
    return img


@pytest.fixture
def dark_line_image():
    """含一条暗线的图像."""
    img = np.zeros((200, 200), dtype=np.uint8)
    cv2.line(img, (20, 100), (180, 100), 20, 2)
    return img


@pytest.fixture
def empty_image():
    """全黑图像."""
    return np.zeros((200, 200), dtype=np.uint8)


class TestLineDetector:
    def test_detects_bright_line(self, detector, bright_line_image):
        """应检出明亮直线."""
        results = detector.detect(bright_line_image)
        assert len(results) > 0

    def test_returns_empty_list_when_no_lines(self, detector, empty_image):
        """无直线时应返回空列表（不是 None）."""
        results = detector.detect(empty_image)
        assert results == []
        assert isinstance(results, list)

    def test_bright_line_scores_higher(self, detector, bright_line_image, dark_line_image):
        """亮线评分应高于暗线."""
        bright_results = detector.detect(bright_line_image)
        dark_results = detector.detect(dark_line_image)

        if bright_results and dark_results:
            assert bright_results[0].score >= dark_results[0].score

    def test_result_has_required_fields(self, detector, bright_line_image):
        """检测结果应包含所有必要字段."""
        results = detector.detect(bright_line_image)
        if results:
            r = results[0]
            assert hasattr(r, 'x1')
            assert hasattr(r, 'y1')
            assert hasattr(r, 'x2')
            assert hasattr(r, 'y2')
            assert hasattr(r, 'brightness')
            assert hasattr(r, 'confidence')
            assert hasattr(r, 'score')

    def test_results_sorted_by_score(self, detector, bright_line_image):
        """结果应按评分降序排列."""
        results = detector.detect(bright_line_image)
        if len(results) >= 2:
            for i in range(len(results) - 1):
                assert results[i].score >= results[i + 1].score

    def test_bgr_input(self, detector, bright_line_image):
        """应支持 BGR 彩色图输入."""
        bgr = cv2.cvtColor(bright_line_image, cv2.COLOR_GRAY2BGR)
        results = detector.detect(bgr)
        assert len(results) > 0
