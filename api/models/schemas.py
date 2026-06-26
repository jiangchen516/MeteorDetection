"""Pydantic 数据模型."""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class TaskStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class TaskType(str, Enum):
    SINGLE = "single"
    BATCH = "batch"
    REALTIME = "realtime"


class TaskCreate(BaseModel):
    task_type: TaskType = TaskType.BATCH
    params: dict[str, Any] = Field(default_factory=dict)
    enable_rectify: bool = False


class TaskResponse(BaseModel):
    id: str
    task_type: TaskType
    status: TaskStatus = TaskStatus.PENDING
    progress: float = 0.0
    message: str = ""
    result_count: int = 0
    created_at: datetime | None = None
    updated_at: datetime | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    result_image: str = ""
    result_images: list[str] = Field(default_factory=list)


class DetectionResult(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int
    brightness: float = 0.0
    confidence: float = 0.0
    score: float = 0.0
    image: str = ""


class TaskDetail(TaskResponse):
    results: list[DetectionResult] = Field(default_factory=list)
    logs: list[str] = Field(default_factory=list)
