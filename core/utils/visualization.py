"""结果可视化工具."""

from pathlib import Path

import cv2
import numpy as np

from core.detection.line_detector import DetectionResult
from core.utils.logging import get_logger

logger = get_logger(__name__)


def draw_detections(
    image: np.ndarray,
    detections: list[DetectionResult],
    show_score: bool = True,
    show_brightness: bool = True,
    color_by_score: bool = True,
) -> np.ndarray:
    """在图像上绘制检测结果（旋转矩形框）.

    Args:
        image: 原始图像.
        detections: 检测结果列表.
        show_score: 是否显示评分.
        show_brightness: 是否显示亮度.
        color_by_score: 是否按评分着色（高分绿色，低分红色）.

    Returns:
        标注后的图像.
    """
    annotated = image.copy()

    for det in detections:
        if color_by_score:
            score = min(det.score, 1.0)
            r = int(255 * (1 - score))
            g = int(255 * score)
            b = 0
            color = (b, g, r)
        else:
            color = (0, 255, 0)

        # 画旋转矩形框（包围流星的方框）
        if det.box_points and len(det.box_points) == 4:
            pts = np.array(det.box_points, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated, [pts], isClosed=True, color=color, thickness=2)
            # 在矩形中心画一个小圆点标记
            cx = int(sum(p[0] for p in det.box_points) / 4)
            cy = int(sum(p[1] for p in det.box_points) / 4)
            cv2.circle(annotated, (cx, cy), 3, color, -1)
        else:
            # 降级：没有 box_points 时画线段
            cv2.line(annotated, (det.x1, det.y1), (det.x2, det.y2), color, 2)
            cv2.circle(annotated, (det.x1, det.y1), 4, color, -1)
            cv2.circle(annotated, (det.x2, det.y2), 4, color, -1)

        # 信息文本
        texts = []
        if show_score:
            texts.append(f"S:{det.confidence:.2f}")
        if show_brightness:
            texts.append(f"B:{det.brightness:.0f}")
        if det.merged_count > 1:
            texts.append(f"M:{det.merged_count}")

        if texts:
            label = " ".join(texts)
            mid_x = (det.x1 + det.x2) // 2
            mid_y = (det.y1 + det.y2) // 2
            cv2.putText(
                annotated, label, (mid_x - 20, mid_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1,
            )

    # 汇总信息
    info = f"Detections: {len(detections)}"
    cv2.putText(
        annotated, info, (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
    )

    return annotated


def draw_trajectory(
    image: np.ndarray,
    points: list[tuple[int, int, int, int]],
    color: tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:
    """绘制流星轨迹."""
    annotated = image.copy()

    if len(points) < 2:
        return annotated

    for i in range(len(points) - 1):
        x1 = (points[i][0] + points[i][2]) // 2
        y1 = (points[i][1] + points[i][3]) // 2
        x2 = (points[i + 1][0] + points[i + 1][2]) // 2
        y2 = (points[i + 1][1] + points[i + 1][3]) // 2
        cv2.line(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.circle(annotated, (x1, y1), 3, color, -1)

    # 最后一个点
    last = points[-1]
    lx = (last[0] + last[2]) // 2
    ly = (last[1] + last[3]) // 2
    cv2.circle(annotated, (lx, ly), 5, color, -1)

    return annotated


def create_summary_image(
    detections: list[DetectionResult],
    image_size: tuple[int, int] = (800, 600),
) -> np.ndarray:
    """创建检测结果汇总图."""
    canvas = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)
    canvas[:] = (20, 20, 40)

    if not detections:
        cv2.putText(canvas, "No detections", (300, 300),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)
        return canvas

    # 评分分布直方图
    scores = [d.score for d in detections]
    brightnesses = [d.brightness for d in detections]

    # 标题
    cv2.putText(canvas, f"Meteor Detection Summary ({len(detections)} found)",
                (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 255), 2)

    # 评分最高的 5 个
    cv2.putText(canvas, "Top detections:", (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 200), 1)

    for i, det in enumerate(detections[:5]):
        y = 100 + i * 30
        text = (f"  #{i+1}: ({det.x1},{det.y1})-({det.x2},{det.y2}) "
                f"score={det.confidence:.2f} bright={det.brightness:.0f}")
        cv2.putText(canvas, text, (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

    return canvas


def save_annotated(
    image: np.ndarray,
    detections: list[DetectionResult],
    output_path: str | Path,
) -> None:
    """保存标注图像."""
    annotated = draw_detections(image, detections)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), annotated)
    logger.info(f"Saved annotated image: {output_path}")
