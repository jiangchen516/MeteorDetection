"""流星/卫星/飞机/噪点分类器 — 基于规则 + 轨迹特征."""

from dataclasses import dataclass
from enum import Enum
from typing import Literal

import numpy as np

from core.utils.logging import get_logger

logger = get_logger(__name__)


class ObjectClass(str, Enum):
    METEOR = "meteor"
    SATELLITE = "satellite"
    AIRPLANE = "airplane"
    NOISE = "noise"
    UNKNOWN = "unknown"


@dataclass
class ClassificationResult:
    """分类结果."""
    object_class: ObjectClass
    confidence: float
    reasons: list[str]


@dataclass
class TrajectoryFeatures:
    """轨迹特征."""
    velocity: float = 0.0           # 像素/帧
    velocity_std: float = 0.0       # 速度标准差
    linearity: float = 0.0          # R²
    duration: int = 0               # 帧数
    total_distance: float = 0.0     # 总位移
    direction_change: float = 0.0   # 方向变化（度）
    brightness_mean: float = 0.0    # 平均亮度
    brightness_std: float = 0.0     # 亮度标准差


def extract_trajectory_features(
    points: list[tuple[int, int, int, int, int]],
    brightnesses: list[float] | None = None,
) -> TrajectoryFeatures:
    """从轨迹点提取特征."""
    features = TrajectoryFeatures()
    features.duration = len(points)

    if len(points) < 2:
        return features

    # 速度
    speeds = []
    directions = []
    for i in range(1, len(points)):
        x1 = (points[i-1][0] + points[i-1][2]) / 2
        y1 = (points[i-1][1] + points[i-1][3]) / 2
        x2 = (points[i][0] + points[i][2]) / 2
        y2 = (points[i][1] + points[i][3]) / 2

        dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        dt = points[i][4] - points[i-1][4]
        if dt > 0:
            speeds.append(dist / dt)

        direction = np.arctan2(y2 - y1, x2 - x1)
        directions.append(direction)

    if speeds:
        features.velocity = float(np.mean(speeds))
        features.velocity_std = float(np.std(speeds)) if len(speeds) > 1 else 0.0

    # 总位移
    first_mid = ((points[0][0] + points[0][2]) / 2, (points[0][1] + points[0][3]) / 2)
    last_mid = ((points[-1][0] + points[-1][2]) / 2, (points[-1][1] + points[-1][3]) / 2)
    features.total_distance = float(np.sqrt(
        (last_mid[0] - first_mid[0])**2 + (last_mid[1] - first_mid[1])**2
    ))

    # 方向变化
    if len(directions) > 1:
        dir_diffs = []
        for i in range(1, len(directions)):
            diff = abs(directions[i] - directions[i-1])
            diff = min(diff, 2*np.pi - diff)
            dir_diffs.append(np.degrees(diff))
        features.direction_change = float(np.mean(dir_diffs))

    # 线性度
    if len(points) >= 3:
        all_x, all_y = [], []
        for p in points:
            all_x.extend([p[0], p[2]])
            all_y.extend([p[1], p[3]])
        try:
            coeffs = np.polyfit(all_x, all_y, 1)
            poly = np.poly1d(coeffs)
            y_pred = poly(all_x)
            corr = np.corrcoef(all_y, y_pred)[0, 1]
            features.linearity = float(corr**2)
        except (np.RankWarning, ValueError):
            features.linearity = 0.0

    # 亮度
    if brightnesses:
        features.brightness_mean = float(np.mean(brightnesses))
        features.brightness_std = float(np.std(brightnesses)) if len(brightnesses) > 1 else 0.0

    return features


def classify_trajectory(
    features: TrajectoryFeatures,
    satellite_matches: list[str] | None = None,
) -> ClassificationResult:
    """基于特征分类轨迹.

    分类规则:
    - 卫星: 匀速直线 + 高线性度 + 可能匹配卫星数据库
    - 流星: 高亮度 + 短持续时间 + 可能有方向变化
    - 飞机: 闪烁（亮度变化大）+ 匀速 + 中等持续时间
    - 噪点: 低线性度 + 短持续时间 + 低亮度
    """
    scores = {cls: 0.0 for cls in ObjectClass}
    reasons = {cls: [] for cls in ObjectClass}

    # === 卫星特征 ===
    if features.velocity > 0 and features.velocity_std / max(features.velocity, 0.01) < 0.2:
        scores[ObjectClass.SATELLITE] += 2.0
        reasons[ObjectClass.SATELLITE].append("匀速运动")

    if features.linearity > 0.95:
        scores[ObjectClass.SATELLITE] += 1.5
        reasons[ObjectClass.SATELLITE].append(f"高线性度({features.linearity:.2f})")

    if features.duration >= 5:
        scores[ObjectClass.SATELLITE] += 1.0
        reasons[ObjectClass.SATELLITE].append(f"长轨迹({features.duration}帧)")

    if satellite_matches:
        scores[ObjectClass.SATELLITE] += 3.0
        reasons[ObjectClass.SATELLITE].append(f"卫星匹配: {', '.join(satellite_matches)}")

    # === 流星特征 ===
    if features.brightness_mean > 50:
        scores[ObjectClass.METEOR] += 1.5
        reasons[ObjectClass.METEOR].append(f"高亮度({features.brightness_mean:.0f})")

    if features.duration <= 3 and features.duration >= 2:
        scores[ObjectClass.METEOR] += 1.5
        reasons[ObjectClass.METEOR].append(f"短轨迹({features.duration}帧)")

    if features.direction_change > 5:
        scores[ObjectClass.METEOR] += 1.0
        reasons[ObjectClass.METEOR].append(f"方向变化({features.direction_change:.1f}°)")

    if features.linearity > 0.7 and features.linearity < 0.98:
        scores[ObjectClass.METEOR] += 1.0
        reasons[ObjectClass.METEOR].append("中等线性度")

    # === 飞机特征 ===
    if features.brightness_std > 30:
        scores[ObjectClass.AIRPLANE] += 2.0
        reasons[ObjectClass.AIRPLANE].append(f"亮度闪烁(std={features.brightness_std:.0f})")

    if 3 <= features.duration <= 10:
        scores[ObjectClass.AIRPLANE] += 1.0
        reasons[ObjectClass.AIRPLANE].append(f"中等轨迹({features.duration}帧)")

    if features.linearity > 0.8:
        scores[ObjectClass.AIRPLANE] += 1.0
        reasons[ObjectClass.AIRPLANE].append("直线运动")

    # === 噪点特征 ===
    if features.linearity < 0.5:
        scores[ObjectClass.NOISE] += 2.0
        reasons[ObjectClass.NOISE].append(f"低线性度({features.linearity:.2f})")

    if features.duration < 2:
        scores[ObjectClass.NOISE] += 1.5
        reasons[ObjectClass.NOISE].append("极短轨迹")

    if features.brightness_mean < 20:
        scores[ObjectClass.NOISE] += 1.0
        reasons[ObjectClass.NOISE].append(f"低亮度({features.brightness_mean:.0f})")

    if features.velocity > 0 and features.velocity_std / max(features.velocity, 0.01) > 0.5:
        scores[ObjectClass.NOISE] += 1.0
        reasons[ObjectClass.NOISE].append("速度不稳定")

    # 选择最高分
    best_class = max(scores, key=scores.get)
    best_score = scores[best_class]

    # 归一化置信度
    total_score = sum(scores.values())
    confidence = best_score / total_score if total_score > 0 else 0.0

    # 如果最高分太低，标记为 UNKNOWN
    if best_score < 1.0:
        best_class = ObjectClass.UNKNOWN
        confidence = 0.0

    result = ClassificationResult(
        object_class=best_class,
        confidence=min(confidence, 1.0),
        reasons=reasons[best_class] if reasons[best_class] else ["无显著特征"],
    )

    logger.debug(
        f"Classified as {best_class.value} (conf={confidence:.2f}): "
        f"{'; '.join(result.reasons)}"
    )
    return result


def classify_single_detection(
    brightness: float,
    score: float,
    line_length: float,
    width: float = 0.0,
    continuity: float = 0.0,
    orig_mean: float = 100.0,
    orig_std: float = 50.0,
) -> ClassificationResult:
    """对单帧检测结果分类（无轨迹信息时）.

    使用亮度 + 形状特征区分流星/恒星/噪点:
    - 流星: 细长亮线，宽度窄，亮度连续，显著高于背景
    - 恒星边缘: 亮度可能高但不连续，宽度不确定
    - 噪点: 短、暗、不连续
    """
    scores = {cls: 0.0 for cls in ObjectClass}
    reasons = {cls: [] for cls in ObjectClass}

    # 亮度相对于背景的显著性
    bright_sigma = (brightness - orig_mean) / max(orig_std, 1.0)

    # === 流星特征 ===
    # 显著高于背景的亮度
    if bright_sigma > 1.0:
        scores[ObjectClass.METEOR] += 1.5
        reasons[ObjectClass.METEOR].append(f"亮度显著(bright_sigma={bright_sigma:.1f})")

    # 高亮度 + 高评分
    if brightness > 60 and score > 0.4:
        scores[ObjectClass.METEOR] += 1.5
        reasons[ObjectClass.METEOR].append("高亮度+高评分")

    # 窄宽度 (流星宽度 2-8 像素)
    if 0 < width < 10:
        scores[ObjectClass.METEOR] += 1.0
        reasons[ObjectClass.METEOR].append(f"窄宽度({width:.0f}px)")

    # 高连续性
    if continuity > 0.35:
        scores[ObjectClass.METEOR] += 1.0
        reasons[ObjectClass.METEOR].append(f"连续(continuity={continuity:.2f})")

    # 长线段
    if line_length > 80:
        scores[ObjectClass.METEOR] += 1.0
        scores[ObjectClass.SATELLITE] += 0.5
        reasons[ObjectClass.METEOR].append(f"长线段({line_length:.0f}px)")

    # === 噪点特征 ===
    if bright_sigma < 0.3:
        scores[ObjectClass.NOISE] += 2.0
        reasons[ObjectClass.NOISE].append("低亮度")

    if score < 0.2:
        scores[ObjectClass.NOISE] += 1.5
        reasons[ObjectClass.NOISE].append("低评分")

    if line_length < 20:
        scores[ObjectClass.NOISE] += 1.0
        reasons[ObjectClass.NOISE].append("极短")

    # === 卫星特征 ===
    if line_length > 100 and bright_sigma > 0.5:
        scores[ObjectClass.SATELLITE] += 1.0
        reasons[ObjectClass.SATELLITE].append("长+稳定亮度")

    best_class = max(scores, key=scores.get)
    best_score = scores[best_class]
    total = sum(scores.values())
    confidence = best_score / total if total > 0 else 0.0

    if best_score < 1.0:
        best_class = ObjectClass.UNKNOWN
        confidence = 0.0

    return ClassificationResult(
        object_class=best_class,
        confidence=min(confidence, 1.0),
        reasons=reasons[best_class] if reasons[best_class] else ["无显著特征"],
    )
