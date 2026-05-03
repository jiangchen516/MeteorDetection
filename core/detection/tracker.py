"""流星轨迹跟踪器 — 多帧轨迹验证."""

from collections import defaultdict

import cv2
import numpy as np

from core.utils.config import get_section
from core.utils.logging import get_logger

logger = get_logger(__name__)


class MeteorTracker:
    """多帧流星轨迹跟踪器."""

    def __init__(self, max_frames: int | None = None, max_gap: int | None = None):
        cfg = get_section("tracking")
        self.max_frames = max_frames or cfg.get("max_frames", 5)
        self.max_gap = max_gap or cfg.get("max_gap", 2)
        self.trajectories: dict[int, list[tuple]] = defaultdict(list)
        self.next_id = 0
        self.confirmed_meteors: list[dict] = []

    def update(self, lines: list[tuple[int, int, int, int]], frame_id: int) -> list[dict]:
        """更新轨迹.

        Args:
            lines: 检测到的直线列表 [(x1,y1,x2,y2), ...].
            frame_id: 当前帧 ID.

        Returns:
            确认的流星轨迹列表.
        """
        if not lines:
            return []

        for line in lines:
            if not self._is_assigned_to_trajectory(line, frame_id):
                self.trajectories[self.next_id].append((*line, frame_id))
                self.next_id += 1

        self._cleanup_trajectories(frame_id)
        return self.confirmed_meteors

    def _is_assigned_to_trajectory(
        self, line: tuple[int, int, int, int], frame_id: int
    ) -> bool:
        x1, y1, x2, y2 = line
        for traj_id, points in self.trajectories.items():
            if not points:
                continue
            last_point = points[-1]
            if frame_id - last_point[4] > self.max_gap:
                continue
            if self._is_consistent(last_point[:4], line):
                points.append((*line, frame_id))
                return True
        return False

    @staticmethod
    def _is_consistent(
        prev_line: tuple[int, int, int, int],
        curr_line: tuple[int, int, int, int],
        angle_thresh: float = 15.0,
        dist_thresh: float = 50.0,
    ) -> bool:
        x1p, y1p, x2p, y2p = prev_line
        x1c, y1c, x2c, y2c = curr_line

        angle_prev = np.arctan2(y2p - y1p, x2p - x1p)
        angle_curr = np.arctan2(y2c - y1c, x2c - x1c)
        angle_diff = abs(np.degrees(angle_curr - angle_prev))
        if angle_diff > 180:
            angle_diff = 360 - angle_diff

        dist1 = np.sqrt((x1c - x1p) ** 2 + (y1c - y1p) ** 2)
        dist2 = np.sqrt((x2c - x2p) ** 2 + (y2c - y2p) ** 2)

        return angle_diff < angle_thresh and dist1 < dist_thresh and dist2 < dist_thresh

    def _cleanup_trajectories(self, current_frame_id: int) -> None:
        cfg = get_section("tracking")
        min_length = cfg.get("min_trajectory_length", 3)
        linearity_thresh = cfg.get("linearity_threshold", 0.8)

        for traj_id in list(self.trajectories.keys()):
            points = self.trajectories[traj_id]
            if not points:
                continue

            last_frame = points[-1][4]
            if current_frame_id - last_frame > self.max_gap:
                if self._validate_trajectory(points, min_length, linearity_thresh):
                    meteor_info = {
                        "id": traj_id,
                        "points": points,
                        "length": len(points),
                        "duration": points[-1][4] - points[0][4],
                        "velocity": self._calculate_velocity(points),
                        "linearity": self._calculate_linearity(points),
                        "confidence": self._calculate_confidence(points),
                    }
                    self.confirmed_meteors.append(meteor_info)
                    logger.info(
                        f"Confirmed meteor {traj_id}: {len(points)} frames, "
                        f"velocity={meteor_info['velocity']:.2f} px/frame"
                    )
                del self.trajectories[traj_id]

    def _validate_trajectory(
        self, points: list[tuple], min_length: int, linearity_thresh: float
    ) -> bool:
        if len(points) < min_length:
            return False
        if self._calculate_linearity(points) < linearity_thresh:
            return False
        return True

    @staticmethod
    def _calculate_velocity(points: list[tuple]) -> float:
        if len(points) < 2:
            return 0.0
        speeds = []
        for i in range(1, len(points)):
            x1p, y1p, x2p, y2p, tp = points[i - 1]
            x1c, y1c, x2c, y2c, tc = points[i]
            mid_prev = ((x1p + x2p) / 2, (y1p + y2p) / 2)
            mid_curr = ((x1c + x2c) / 2, (y1c + y2c) / 2)
            distance = np.sqrt(
                (mid_curr[0] - mid_prev[0]) ** 2 + (mid_curr[1] - mid_prev[1]) ** 2
            )
            time_diff = tc - tp
            if time_diff > 0:
                speeds.append(distance / time_diff)
        return float(np.mean(speeds)) if speeds else 0.0

    @staticmethod
    def _calculate_linearity(points: list[tuple]) -> float:
        if len(points) < 3:
            return 1.0
        all_pts = []
        for x1, y1, x2, y2, _ in points:
            all_pts.extend([(x1, y1), (x2, y2)])
        xs = [p[0] for p in all_pts]
        ys = [p[1] for p in all_pts]
        try:
            coeffs = np.polyfit(xs, ys, 1)
            poly = np.poly1d(coeffs)
            y_pred = poly(xs)
            correlation = np.corrcoef(ys, y_pred)[0, 1]
            return float(correlation ** 2)
        except (np.RankWarning, ValueError):
            return 0.0

    def _calculate_confidence(self, points: list[tuple]) -> float:
        cfg = get_section("tracking")
        min_length = cfg.get("min_trajectory_length", 3)
        length_score = min(len(points) / min_length, 2.0)
        linearity_score = self._calculate_linearity(points) * 2
        velocity_score = min(self._calculate_velocity(points) / 10, 1.0)
        confidence = (length_score + linearity_score + velocity_score) / 3
        return min(confidence, 1.0)

    def reset(self) -> None:
        self.trajectories.clear()
        self.confirmed_meteors.clear()
        self.next_id = 0
