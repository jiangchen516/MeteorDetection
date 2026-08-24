"""卫星过滤器测试 — 离线、用最小合成 TLE 文件."""

from datetime import datetime, timezone

import pytest

from core.classification.satellite_filter import SatelliteFilter

MINIMAL_TLE = """TEST SAT 001
1 99999U 24001A   24032.50000000  .00000000  00000-0  00000-0 0  9999
2 99999  51.6000 200.0000 0001000 300.0000  60.0000 15.00000000 000001
"""


@pytest.fixture
def tle_dir(tmp_path):
    d = tmp_path / "tle"
    d.mkdir()
    (d / "TLE_test.txt").write_text(MINIMAL_TLE)
    return d


def test_is_likely_satellite_by_motion(tle_dir):
    """基于运动特征即可判断，不依赖网络/TLE 比对."""
    f = SatelliteFilter(tle_dir=tle_dir)
    likely, reasons = f.is_likely_satellite(
        datetime.now(timezone.utc), velocity=30.0, linearity=0.99)
    assert isinstance(likely, bool)
    assert isinstance(reasons, list)
    # 高线性度 + 匀速 → 极可能是卫星
    assert likely is True


def test_check_detection_returns_list(tle_dir):
    """check_detection 应返回过境列表而不抛异常（回归 from_datetime 修复）."""
    f = SatelliteFilter(tle_dir=tle_dir)
    passes = f.check_detection(datetime.now(timezone.utc))
    assert isinstance(passes, list)
