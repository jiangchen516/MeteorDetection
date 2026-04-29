"""
配置文件 - 统一管理项目路径和参数
"""
import os
from pathlib import Path

# 基础目录
BASE_DIR = Path(__file__).parent
TEST_DIR = BASE_DIR / "test"
OUTPUTS_DIR = BASE_DIR / "outputs"
DATA_DIR = BASE_DIR / "data"

# 测试数据路径
RECTIFY_TEST_DIR = TEST_DIR / "rectify_test"
MERGE_TEST_DIR = TEST_DIR / "merge_tests"
RECTIFIED_IMAGES_DIR = TEST_DIR / "rectify_test"  # 鱼眼校正输出
MERGED_IMAGE_PATH = TEST_DIR / "merged_rectified.jpg"

# 输出路径
COORDS_FILE = OUTPUTS_DIR / "coords.csv"
LINE_DETECTION_OUTPUT = TEST_DIR / "line_detection"

# 算法参数
HOUGH_LINE_PARAMS = {
    'rho': 1,
    'theta': np.pi / 180,
    'threshold': 30,
    'min_line_length': 30,
    'max_line_gap': 10
}

# 流星检测参数
METEOR_DETECTION = {
    'min_line_length': 30,  # 最小流星长度（像素）
    'max_line_gap': 10,     # 最大线段间隙
    'slope_tolerance': 0.05, # 斜率容差
    'min_brightness': 100,  # 最小亮度
    'min_aspect_ratio': 10  # 最小长宽比
}

# 轨迹跟踪参数
TRACKING = {
    'max_frames': 5,        # 最大跟踪帧数
    'max_gap': 2,           # 最大帧间间隔
    'min_trajectory_length': 3,  # 最小轨迹长度
    'linearity_threshold': 0.8   # 线性度阈值
}

# 背景建模参数
BACKGROUND = {
    'history': 50,          # 历史帧数
    'threshold': 30,        # 前景阈值
    'learning_rate': 0.01   # 学习率
}

# 确保目录存在
def ensure_directories():
    """创建必要的目录"""
    for directory in [TEST_DIR, OUTPUTS_DIR, DATA_DIR, RECTIFY_TEST_DIR, LINE_DETECTION_OUTPUT]:
        directory.mkdir(parents=True, exist_ok=True)

# 调用确保目录
ensure_directories()