"""卫星交叉验证 — 将检测结果与卫星 TLE 数据比对."""

import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from core.utils.config import get_path, get_section
from core.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SatellitePass:
    """卫星过境信息."""
    name: str
    rise_time: datetime.datetime
    set_time: datetime.datetime
    max_elevation: float  # 相对观测者的最大地平仰角（度）
    direction: str  # 过境方向，如 "N→S"，或 "az=xxx°→yyy°"


class SatelliteFilter:
    """卫星过滤器 — 判断检测到的亮线是否可能是卫星."""

    def __init__(
        self,
        tle_dir: Optional[str | Path] = None,
        observer: Optional[dict] = None,
    ):
        self.tle_dir = Path(tle_dir) if tle_dir else get_path("outputs_dir")
        # 观测点：显式传入 > config['observer']（缺省北京，需用户改为真实地点）
        obs = observer or get_section("observer")
        self.obs_lat = float(obs.get("latitude", 39.9042))
        self.obs_lon = float(obs.get("longitude", 116.4074))
        self.obs_elev = float(obs.get("elevation_m", 50.0))
        self.min_elevation = float(obs.get("min_elevation", 10.0))
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
            obs = wgs84.latlon(self.obs_lat, self.obs_lon, elevation_m=self.obs_elev)

            ts = load.timescale()
            # from_datetime 接受时区感知的 datetime；naive 视为 UTC
            aware = detection_time if detection_time.tzinfo else detection_time.replace(
                tzinfo=datetime.timezone.utc)
            t = ts.from_datetime(aware)

            # 过境窗口采样（前后各 5 分钟，寻找相对观测者的最大仰角）
            half_w = datetime.timedelta(minutes=5)
            t_prev = ts.from_datetime(aware - half_w)
            t_next = ts.from_datetime(aware + half_w)

            for sat in satellites:
                try:
                    # 相对观测者的地平坐标（topocentric altaz）
                    alt0, az0, _ = (sat - obs).at(t).altaz()
                    el0 = alt0.degrees
                    if el0 < self.min_elevation:
                        continue
                    # 峰值仰角
                    alt_prev, _, _ = (sat - obs).at(t_prev).altaz()
                    alt_next, _, _ = (sat - obs).at(t_next).altaz()
                    max_el = max(el0, alt_prev.degrees, alt_next.degrees)
                    # 过境方向（沿地平坐标系 az 变化）
                    az_prev, _, _ = (sat - obs).at(t_prev).altaz()
                    az_next, _, _ = (sat - obs).at(t_next).altaz()
                    az_from = az_prev.degrees
                    az_to = az_next.degrees
                    az_from = (az_from + 360) % 360
                    az_to = (az_to + 360) % 360
                    direction = f"az={az_from:.0f}°→{az_to:.0f}°"
                    # 若提供了检测方向，做粗匹配（可选）
                    if azimuth is not None and elevation is not None:
                        daz = abs((az0.degrees - azimuth + 180) % 360 - 180)
                        if daz > 90:  # 方向相差超过 90° 视为不匹配
                            continue
                    passes.append(SatellitePass(
                        name=sat.name,
                        rise_time=aware - half_w,
                        set_time=aware + half_w,
                        max_elevation=max_el,
                        direction=direction,
                    ))
                except Exception:
                    continue

        except Exception as e:
            logger.warning(f"Satellite check failed: {e}")

        # 按仰角降序
        passes.sort(key=lambda p: p.max_elevation, reverse=True)
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
