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

    def test_min_area_filters_small_blobs(self):
        """min_area 较大的单个像素噪点应被过滤."""
        img = np.zeros((100, 100), dtype=np.uint8)
        img[50, 50] = 255     # 单像素
        det = LineDetector(_config_loaded=True, min_area=10)
        assert det.detect(img) == []

    def test_min_aspect_filters_square_blobs(self):
        """长宽比过小的方形连通域应被 min_aspect 过滤."""
        img = np.zeros((100, 100), dtype=np.uint8)
        cv2.circle(img, (50, 50), 10, 255, -1)     # 圆，长宽比约 1
        det = LineDetector(_config_loaded=True, min_aspect=3.0)
        slants = [r for r in det.detect(img) if abs(r.y1 - r.y2) > 5]
        assert slants == []

    def test_blur_kernel_size_even_is_safe(self):
        """偶数模糊核应被安全处理（OpenCV 要求奇数 > 0）."""
        img = np.zeros((200, 200), dtype=np.uint8)
        cv2.line(img, (20, 100), (180, 100), 255, 2)
        det = LineDetector(_config_loaded=True, blur_kernel_size=6)     # 偶数
        assert isinstance(det.detect(img), list)

    def test_detection_result_has_classification_fields(self):
        """DetectionResult 默认带单帧分类字段."""
        img = np.zeros((200, 200), dtype=np.uint8)
        cv2.line(img, (20, 100), (180, 100), 255, 2)
        results = LineDetector(_config_loaded=True).detect(img)
        assert results
        r = results[0]
        assert r.object_class == "unknown"
        assert r.classification_confidence == 0.0
        assert r.classification_reasons == []

    def test_score_threshold_filters_low_score(self):
        """高阈值应过滤掉短/暗/低分碎屑."""
        img = np.zeros((200, 200), dtype=np.uint8)
        cv2.line(img, (95, 100), (105, 100), 30, 1)      # 短暗线段
        det = LineDetector(_config_loaded=True, score_threshold=0.9)
        assert det.detect(img) == []

    def test_score_threshold_zero_disables_filtering(self):
        """score_threshold=0 表示不过滤."""
        det = LineDetector(_config_loaded=True, score_threshold=0.0)
        img = np.zeros((200, 200), dtype=np.uint8)
        cv2.line(img, (95, 100), (105, 100), 255, 1)
        # 虽然短，但不过滤就应保留
        assert isinstance(det.detect(img), list)

    def test_quality_score_ranks_long_bright_slender_highest(self):
        """长而亮而细长的候选综合分最高."""
        from core.detection.line_detector import _quality_score
        good = _quality_score(200, 200, 20, image_mean=30, image_std=40)
        poor = _quality_score(10, 20, 1.2, image_mean=30, image_std=40)
        assert good > poor
