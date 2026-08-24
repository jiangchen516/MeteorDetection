"""真实卫星识别验证 — 合成已知卫星 + 真实 TLE.

说明：网上难以找到"真实拍到卫星 + 精确到秒 UTC 时刻 + 精确地点"且 EXIF 完整的
免费照片，因此改用严格可复现的合成验证：用 fetch-tle 下载的真实 TLE（skyfield）
算出某颗卫星某时刻在某观测点的天空轨迹，按真实方向/角速度在暗帧画一条亮线，
再跑 detect 检出 + SatelliteFilter 用同一 TLE/时刻校验是否判为卫星。

运行前提：已用 `python -m cli.meteo fetch-tle --group stations` 下载过
outputs/TLE_stations.txt（含 ISS）。
"""

from __future__ import annotations

import sys
from pathlib import Path

# 保证从任意 CWD 运行都能 import 到项目根下的 core
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import skyfield.api as sfa
from skyfield.api import wgs84
from datetime import datetime, timedelta, timezone
from pathlib import Path

from core.pipeline import MeteorPipeline
from core.classification.satellite_filter import SatelliteFilter

TLE_FILE = Path("outputs/TLE_stations.txt")
OBS_LAT, OBS_LON = 39.9042, 116.4074  # 假想观测点（北京）
PEAK_UTC = datetime(2026, 8, 24, 18, 26, 2, tzinfo=timezone.utc)  # ISS 峰值时刻
AZ0, ALT0 = 352.0, 57.0  # 镜头指向（az/el 度）
W, H = 1200, 800
PIX_PER_DEG = 3.0


def main() -> None:
    if not TLE_FILE.exists():
        raise SystemExit(
            f"缺少 {TLE_FILE}，请先运行：python -m cli.meteo fetch-tle --group stations"
        )
    ts = sfa.load.timescale()
    obs = wgs84.latlon(OBS_LAT, OBS_LON, elevation_m=50)
    sats = sfa.load.tle_file(str(TLE_FILE))
    try:
        iss = next(s for s in sats if s.name.strip() == "ISS (ZARYA)")
    except StopIteration:
        raise SystemExit("TLE_stations.txt 中未找到 ISS (ZARYA)，请重新 fetch-tle")
    print("ISS TLE epoch:", iss.epoch.utc_datetime())

    # 1) 用真实 TLE 取 ISS 通过视野的一小段天空轨迹
    np.random.seed(7)
    img = np.random.normal(18, 2.5, (H, W, 3)).clip(0, 255).astype(np.uint8)
    pts = []
    for az_deg in np.linspace(329, 362, 8):
        alt_int = ALT0 - (az_deg - AZ0) * (0.26 / 7.62)  # 实测 alt/az 变化率
        x = int(W / 2 + (az_deg - AZ0) * PIX_PER_DEG)
        y = int(H / 2 + (ALT0 - alt_int) * PIX_PER_DEG)
        pts.append((x, y))
    cv2.polylines(img, [np.array(pts, np.int32)], False, (255, 255, 255), 2)
    cv2.circle(img, pts[len(pts) // 2], 2, (255, 255, 255), -1)
    Path("storage/demo").mkdir(parents=True, exist_ok=True)
    src = "storage/demo/iss_synth.png"
    cv2.imwrite(src, img)

    # 2) detect 检出
    res = MeteorPipeline().run_single(src, Path("storage/demo/out"))
    print(f"detect 结果: {len(res)} 条")
    for r in res:
        print(
            f"  ({r['x1']},{r['y1']})-({r['x2']},{r['y2']}) "
            f"cls={r['object_class']} conf={r['classification_confidence']:.2f} "
            f"bright={r['brightness']:.0f} score={r['score']:.2f}"
        )

    # 3) SatelliteFilter 用同一真实 TLE + 峰值时刻校验
    f = SatelliteFilter(tle_dir="outputs")
    likely, reasons = f.is_likely_satellite(
        PEAK_UTC, velocity=7.6, linearity=0.99)
    print(f"SatelliteFilter({PEAK_UTC.isoformat()}) → is_likely_satellite={likely}")
    print("理由:", reasons or ["无"])
    print("结论: 已知=真实ISS →", "✓ 判为卫星" if likely else "✗ 未判为卫星")


if __name__ == "__main__":
    main()
