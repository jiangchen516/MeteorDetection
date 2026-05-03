"""卫星交叉验证 — 将检测结果与卫星 TLE 数据比对."""

import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from core.utils.config import get_path
from core.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SatellitePass:
    """卫星过境信息."""
    name: str
    rise_time: datetime.datetime
    set_time: datetime.datetime
    max_elevation: float
    direction: str  # N→S, S→N, etc.


class SatelliteFilter:
    """卫星过滤器 — 判断检测到的亮线是否可能是卫星."""

    def __init__(self, tle_dir: Optional[str | Path] = None):
        self.tle_dir = Path(tle_dir) if tle_dir else get_path("outputs")
        self._satellites = None

    def check_detection(
        self,
        detection_time: datetime.datetime,
        azimuth: float | None = None,
        elevation: float | None = None,
    ) -> list[SatellitePass]:
        """检查给定时间/方向是否有卫星过境.

        Args:
            detection_time: 检测时间.
            azimuth: 方位角（度），可选.
            elevation: 仰角（度），可选.

        Returns:
            匹配的卫星过境列表.
        """
        try:
            from skyfield.api import load, wgs84
        except ImportError:
            logger.warning("skyfield not installed, skipping satellite check")
            return []

        passes = []
        try:
            tle_files = sorted(self.tle_dir.glob("TLE*.txt"))
            if not tle_files:
                return []

            latest_tle = tle_files[-1]
            satellites = load.tle_file(str(latest_tle))

            ts = load.timescale()
            t = ts.from_astropy(
                detection_time.replace(tzinfo=datetime.timezone.utc)
            )

            for sat in satellites:
                try:
                    # 检查卫星是否在地平线以上
                    geocentric = sat.at(t)
                    subpoint = geocentric.subpoint()
                    lat = subpoint.latitude.degrees
                    lon = subpoint.longitude.degrees
                    elevation_km = subpoint.elevation.km

                    if elevation_km > 200:  # 低轨道卫星
                        passes.append(SatellitePass(
                            name=sat.name,
                            rise_time=detection_time,
                            set_time=detection_time,
                            max_elevation=elevation_km,
                            direction=f"lat={lat:.1f} lon={lon:.1f}",
                        ))
                except Exception:
                    continue

        except Exception as e:
            logger.warning(f"Satellite check failed: {e}")

        return passes

    def is_likely_satellite(
        self,
        detection_time: datetime.datetime,
        velocity: float,
        linearity: float,
    ) -> tuple[bool, list[str]]:
        """基于运动特征快速判断是否可能是卫星.

        Args:
            detection_time: 检测时间.
            velocity: 速度（像素/帧）.
            linearity: 线性度.

        Returns:
            (是否可能是卫星, 原因列表).
        """
        reasons = []

        # 卫星特征：匀速、高线性度
        if linearity > 0.98:
            reasons.append(f"极高线性度({linearity:.3f})")

        if velocity > 5 and velocity < 100:
            reasons.append(f"匀速运动({velocity:.1f} px/frame)")

        # 如果在夜间有卫星过境记录
        passes = self.check_detection(detection_time)
        if passes:
            reasons.append(f"卫星过境: {', '.join(p.name for p in passes[:3])}")

        is_satellite = len(reasons) >= 2
        return is_satellite, reasons
