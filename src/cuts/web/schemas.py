from __future__ import annotations

from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"


class JobCreateResponse(BaseModel):
    job_id: str
    status: JobStatus
    status_url: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: JobStatus
    stage: str | None = None
    brain_backend: str
    warnings: list[str] = Field(default_factory=list)
    error: str | None = None
    video_url: str | None = None
    edl_url: str | None = None
    video_path: Path | None = None
    edl_path: Path | None = None
