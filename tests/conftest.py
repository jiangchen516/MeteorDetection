"""pytest fixtures."""

import numpy as np
import pytest

from core.utils.config import reset as reset_config


@pytest.fixture(autouse=True)
def _reset_config():
    """每个测试前重置配置缓存."""
    reset_config()
    yield
    reset_config()


@pytest.fixture
def synthetic_line_image():
    """创建含一条白线的合成图像."""
    img = np.zeros((200, 200), dtype=np.uint8)
    cv2_line = cv2.line(img, (20, 100), (180, 100), 255, 2)
    return cv2_line


@pytest.fixture
def black_image():
    """全黑图像."""
    return np.zeros((100, 100, 3), dtype=np.uint8)


@pytest.fixture
def white_image():
    """全白图像."""
    return np.ones((100, 100, 3), dtype=np.uint8) * 255


# 延迟导入 cv2（避免头文件依赖问题）
import cv2  # noqa: E402
