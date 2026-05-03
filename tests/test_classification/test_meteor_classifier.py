"""流星分类器测试."""

import pytest
import numpy as np

from core.classification.meteor_classifier import (
    extract_trajectory_features,
    classify_trajectory,
    classify_single_detection,
    ObjectClass,
    TrajectoryFeatures,
)


class TestTrajectoryFeatures:
    def test_straight_line(self):
        """直线轨迹应有高线性度."""
        points = [(10, 10, 20, 20, 0), (30, 30, 40, 40, 1), (50, 50, 60, 60, 2)]
        features = extract_trajectory_features(points)
        assert features.linearity > 0.9
        assert features.velocity > 0

    def test_zigzag(self):
        """锯齿轨迹应有低线性度."""
        points = [(10, 10, 20, 20, 0), (30, 50, 40, 60, 1), (50, 10, 60, 20, 2)]
        features = extract_trajectory_features(points)
        assert features.linearity < 0.9

    def test_single_point(self):
        """单点轨迹."""
        points = [(10, 10, 20, 20, 0)]
        features = extract_trajectory_features(points)
        assert features.duration == 1
        assert features.velocity == 0
        assert features.linearity == 0  # 单点无法计算线性度

    def test_with_brightness(self):
        points = [(10, 10, 20, 20, 0), (30, 30, 40, 40, 1)]
        features = extract_trajectory_features(points, brightnesses=[100, 200])
        assert features.brightness_mean == 150
        assert features.brightness_std > 0


class TestClassifyTrajectory:
    def test_classify_satellite(self):
        """匀速直线应为卫星."""
        features = TrajectoryFeatures(
            velocity=20.0, velocity_std=1.0,
            linearity=0.99, duration=8,
            total_distance=200.0, direction_change=1.0,
            brightness_mean=50, brightness_std=5,
        )
        result = classify_trajectory(features)
        assert result.object_class == ObjectClass.SATELLITE
        assert result.confidence > 0.3

    def test_classify_meteor(self):
        """高亮度短轨迹应为流星."""
        features = TrajectoryFeatures(
            velocity=10.0, velocity_std=5.0,
            linearity=0.85, duration=2,
            total_distance=50.0, direction_change=10.0,
            brightness_mean=150, brightness_std=20,
        )
        result = classify_trajectory(features)
        assert result.object_class == ObjectClass.METEOR

    def test_classify_airplane(self):
        """闪烁+直线应为飞机."""
        features = TrajectoryFeatures(
            velocity=15.0, velocity_std=2.0,
            linearity=0.9, duration=5,
            total_distance=100.0, direction_change=3.0,
            brightness_mean=80, brightness_std=50,
        )
        result = classify_trajectory(features)
        assert result.object_class == ObjectClass.AIRPLANE

    def test_classify_noise(self):
        """低线性度+短轨迹应为噪点."""
        features = TrajectoryFeatures(
            velocity=5.0, velocity_std=10.0,
            linearity=0.3, duration=1,
            total_distance=5.0, direction_change=30.0,
            brightness_mean=10, brightness_std=2,
        )
        result = classify_trajectory(features)
        assert result.object_class == ObjectClass.NOISE

    def test_satellite_with_matches(self):
        """有卫星匹配时加分."""
        features = TrajectoryFeatures(
            velocity=20.0, velocity_std=1.0,
            linearity=0.99, duration=8,
            total_distance=200.0, direction_change=1.0,
            brightness_mean=50, brightness_std=5,
        )
        result = classify_trajectory(features, satellite_matches=["ISS"])
        assert result.object_class == ObjectClass.SATELLITE
        assert result.confidence > 0.5


class TestClassifySingleDetection:
    def test_bright_long_line(self):
        result = classify_single_detection(brightness=150, score=0.8, line_length=200)
        assert result.object_class in (ObjectClass.METEOR, ObjectClass.SATELLITE)

    def test_dim_short_line(self):
        result = classify_single_detection(brightness=10, score=0.1, line_length=20)
        assert result.object_class in (ObjectClass.NOISE, ObjectClass.UNKNOWN)
