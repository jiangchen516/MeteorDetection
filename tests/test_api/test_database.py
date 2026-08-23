"""数据库分类持久化与搜索测试."""

import json
import pathlib

import pytest

import api.models.database as db


@pytest.fixture
def clean_db(tmp_path):
    """把数据库指向临时文件，避免污染真实 storage/meteo.db."""
    db._db_path = pathlib.Path(tmp_path) / "test.db"
    db.init_db()
    yield
    db._db_path = None


def _task(clean_db, task_id="task-1", task_type="batch"):
    db.upsert_task(task_id=task_id, task_type=task_type)


class TestAddDetectionPersistsClassification:
    def test_returns_detection_id(self, clean_db):
        _task(clean_db)
        det_id = db.add_detection("task-1", {
            "x1": 0, "y1": 0, "x2": 10, "y2": 0,
            "object_class": "meteor",
        })
        assert isinstance(det_id, int) and det_id > 0

    def test_detection_carries_classification(self, clean_db):
        _task(clean_db)
        db.add_detection("task-1", {
            "x1": 0, "y1": 0, "x2": 10, "y2": 0,
            "brightness": 120, "confidence": 0.8, "score": 0.7,
            "object_class": "meteor", "classification_confidence": 0.83,
            "classification_reasons": ["高亮度", "细长"],
        })
        rows = db.get_detections("task-1")
        assert len(rows) == 1
        d = rows[0]
        assert d["class"] == "meteor"
        assert d["class_confidence"] == 0.83
        assert "高亮度" in json.loads(d["class_reasons"] or "[]")

    def test_detection_without_classification(self, clean_db):
        """无分类信息时仅存检测，不写分类."""
        _task(clean_db)
        db.add_detection("task-1", {"x1": 0, "y1": 0, "x2": 10, "y2": 0})
        d = db.get_detections("task-1")[0]
        assert d.get("class") is None
        assert db.get_classifications("task-1") == []


class TestSearchByObjectClass:
    def test_filters_by_object_class(self, clean_db):
        _task(clean_db)
        db.add_detection("task-1", {
            "x1": 0, "y1": 0, "x2": 10, "y2": 0, "confidence": 0.8,
            "object_class": "meteor",
        })
        db.add_detection("task-1", {
            "x1": 5, "y1": 5, "x2": 8, "y2": 8, "confidence": 0.2,
            "object_class": "noise",
        })
        meteors = db.search_detections(object_class="meteor")
        assert [r["object_class"] for r in meteors] == ["meteor"]
        assert len(meteors) == 1

    def test_search_all(self, clean_db):
        _task(clean_db)
        db.add_detection("task-1", {"x1": 0, "y1": 0, "x2": 10, "y2": 0,
                                    "object_class": "meteor"})
        db.add_detection("task-1", {"x1": 5, "y1": 5, "x2": 8, "y2": 8,
                                    "object_class": "noise"})
        assert len(db.search_detections()) == 2

    def test_trajectory_classification_not_in_search(self, clean_db):
        """轨迹级分类（无 detection_id）不进 per-detection 搜索."""
        _task(clean_db)
        db.add_classification("task-1", object_class="satellite",
                              confidence=0.9, reasons="[]", detection_id=None)
        assert db.get_classifications("task-1")
        assert db.search_detections(object_class="satellite") == []
