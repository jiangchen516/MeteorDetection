"""SQLite 数据库连接."""

import sqlite3
from datetime import datetime
from pathlib import Path

from core.utils.config import get_path

_db_path: Path | None = None


def get_db_path() -> Path:
    global _db_path
    if _db_path is None:
        _db_path = get_path("storage_dir") / "meteo.db"
        _db_path.parent.mkdir(parents=True, exist_ok=True)
    return _db_path


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(str(get_db_path()))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db() -> None:
    """初始化数据库表."""
    conn = get_connection()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS tasks (
            id TEXT PRIMARY KEY,
            task_type TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'PENDING',
            progress REAL DEFAULT 0.0,
            message TEXT DEFAULT '',
            result_count INTEGER DEFAULT 0,
            result_image TEXT DEFAULT '',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            params TEXT DEFAULT '{}'
        );

        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id TEXT NOT NULL,
            image TEXT DEFAULT '',
            x1 INTEGER NOT NULL,
            y1 INTEGER NOT NULL,
            x2 INTEGER NOT NULL,
            y2 INTEGER NOT NULL,
            brightness REAL DEFAULT 0.0,
            confidence REAL DEFAULT 0.0,
            score REAL DEFAULT 0.0,
            FOREIGN KEY (task_id) REFERENCES tasks(id)
        );

        CREATE TABLE IF NOT EXISTS task_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id TEXT NOT NULL,
            level TEXT DEFAULT 'INFO',
            message TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (task_id) REFERENCES tasks(id)
        );

        CREATE TABLE IF NOT EXISTS classifications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            detection_id INTEGER,
            task_id TEXT NOT NULL,
            object_class TEXT NOT NULL DEFAULT 'unknown',
            confidence REAL DEFAULT 0.0,
            reasons TEXT DEFAULT '[]',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (task_id) REFERENCES tasks(id)
        );

        CREATE INDEX IF NOT EXISTS idx_detections_task_id ON detections(task_id);
        CREATE INDEX IF NOT EXISTS idx_task_logs_task_id ON task_logs(task_id);
        CREATE INDEX IF NOT EXISTS idx_classifications_task_id ON classifications(task_id);
        CREATE INDEX IF NOT EXISTS idx_classifications_class ON classifications(object_class);
    """)
    conn.commit()
    conn.close()


def upsert_task(
    task_id: str,
    task_type: str,
    status: str = "PENDING",
    progress: float = 0.0,
    message: str = "",
    result_count: int = 0,
    params: str = "{}",
    result_image: str = "",
) -> None:
    now = datetime.utcnow().isoformat()
    conn = get_connection()
    conn.execute(
        """
        INSERT INTO tasks (id, task_type, status, progress, message, result_count, result_image, params, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            status=excluded.status,
            progress=excluded.progress,
            message=excluded.message,
            result_count=excluded.result_count,
            result_image=excluded.result_image,
            updated_at=excluded.updated_at
        """,
        (task_id, task_type, status, progress, message, result_count, result_image, params, now, now),
    )
    conn.commit()
    conn.close()


def get_task(task_id: str) -> dict | None:
    conn = get_connection()
    row = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def list_tasks(limit: int = 50, offset: int = 0) -> list[dict]:
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM tasks ORDER BY created_at DESC LIMIT ? OFFSET ?",
        (limit, offset),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def add_detection(task_id: str, det: dict) -> None:
    conn = get_connection()
    conn.execute(
        """
        INSERT INTO detections (task_id, image, x1, y1, x2, y2, brightness, confidence, score)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (task_id, det.get("image", ""), det["x1"], det["y1"], det["x2"], det["y2"],
         det.get("brightness", 0), det.get("confidence", 0), det.get("score", 0)),
    )
    conn.commit()
    conn.close()


def get_detections(task_id: str) -> list[dict]:
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM detections WHERE task_id = ?", (task_id,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def add_log(task_id: str, level: str, message: str) -> None:
    conn = get_connection()
    conn.execute(
        "INSERT INTO task_logs (task_id, level, message) VALUES (?, ?, ?)",
        (task_id, level, message),
    )
    conn.commit()
    conn.close()


def get_logs(task_id: str, limit: int = 200) -> list[dict]:
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM task_logs WHERE task_id = ? ORDER BY created_at DESC LIMIT ?",
        (task_id, limit),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def add_classification(
    task_id: str,
    object_class: str,
    confidence: float,
    reasons: str = "[]",
    detection_id: int | None = None,
) -> None:
    conn = get_connection()
    conn.execute(
        """
        INSERT INTO classifications (detection_id, task_id, object_class, confidence, reasons)
        VALUES (?, ?, ?, ?, ?)
        """,
        (detection_id, task_id, object_class, confidence, reasons),
    )
    conn.commit()
    conn.close()


def get_classifications(task_id: str) -> list[dict]:
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM classifications WHERE task_id = ?", (task_id,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def search_detections(
    object_class: str | None = None,
    min_confidence: float = 0.0,
    min_brightness: float = 0.0,
    date_from: str | None = None,
    date_to: str | None = None,
    limit: int = 100,
    offset: int = 0,
) -> list[dict]:
    """搜索检测结果."""
    conn = get_connection()
    conditions = ["d.confidence >= ?", "d.brightness >= ?"]
    params: list = [min_confidence, min_brightness]

    if object_class:
        conditions.append("c.object_class = ?")
        params.append(object_class)

    if date_from:
        conditions.append("t.created_at >= ?")
        params.append(date_from)

    if date_to:
        conditions.append("t.created_at <= ?")
        params.append(date_to)

    where = " AND ".join(conditions)
    query = f"""
        SELECT d.*, t.task_type, t.created_at as task_time,
               c.object_class, c.confidence as class_confidence
        FROM detections d
        JOIN tasks t ON d.task_id = t.id
        LEFT JOIN classifications c ON c.task_id = t.id AND c.detection_id = d.id
        WHERE {where}
        ORDER BY d.confidence DESC
        LIMIT ? OFFSET ?
    """
    params.extend([limit, offset])

    rows = conn.execute(query, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]
