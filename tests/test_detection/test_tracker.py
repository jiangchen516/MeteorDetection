"""轨迹跟踪器测试 — 重点覆盖 flush()（序列结束时确认末段轨迹）."""

import pytest

from core.detection.tracker import MeteorTracker


def _tracker():
    return MeteorTracker()


def test_flush_confirms_trailing_trajectory():
    """连续帧（无超时）也能在 flush 时被确认，而不是丢失."""
    tr = _tracker()
    line = (100, 100, 200, 110)  # (x1,y1,x2,y2)
    for i in range(3):
        tr.update([line], i)
    # 连续帧、无 gap → 尚不确认
    assert tr.confirmed_meteors == []
    tr.flush()
    assert len(tr.confirmed_meteors) == 1
    m = tr.confirmed_meteors[0]
    assert m["length"] == 3
    assert m["velocity"] >= 0
    assert 0.0 <= m["linearity"] <= 1.0


def test_flush_drops_short_trajectory():
    """不足 min_trajectory_length(3) 的轨迹在 flush 时被丢弃."""
    tr = _tracker()
    tr.update([(100, 100, 200, 110)], 0)  # 单帧单点
    tr.flush()
    assert tr.confirmed_meteors == []


def test_flush_idempotent():
    """flush 后轨迹清零，再 flush 不会重复确认."""
    tr = _tracker()
    for i in range(3):
        tr.update([(100, 100, 200, 110)], i)
    tr.flush()
    n = len(tr.confirmed_meteors)
    tr.flush()
    assert len(tr.confirmed_meteors) == n


def test_stale_confirms_without_flush():
    """存在时序缺口（>max_gap）时，更新期即确认（无需 flush）."""
    tr = _tracker()
    for i in range(3):
        tr.update([(100, 100, 200, 110)], i)
    # 跳到很远的一帧制造超时
    tr.update([], 10)
    assert len(tr.confirmed_meteors) == 1
