from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, ValidationInfo, field_validator, model_validator


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"


class JobCreateResponse(BaseModel):
    job_id: str
    status: JobStatus
    status_url: str
    version: int = 1


class JobStatusResponse(BaseModel):
    job_id: str
    status: JobStatus
    version: int = 1
    status_url: str | None = None
    stage: str | None = None
    brain_backend: str
    warnings: list[str] = Field(default_factory=list)
    error: str | None = None
    video_url: str | None = None
    edl_url: str | None = None
    video_path: Path | None = None
    edl_path: Path | None = None


class RerenderClipEdit(BaseModel):
    original_index: int
    source_in: float
    source_out: float
    transition_kind: Literal["cut", "fade"] = "cut"
    transition_duration: float = 0.0

    @field_validator("original_index")
    @classmethod
    def _validate_original_index(cls, value: int) -> int:
        if value < 0:
            raise ValueError("original_index must be non-negative")
        return value

    @field_validator("source_in", "source_out", "transition_duration")
    @classmethod
    def _validate_non_negative(cls, value: float, info: ValidationInfo) -> float:
        if value < 0:
            raise ValueError(f"{info.field_name} must be non-negative")
        return value

    @model_validator(mode="after")
    def _validate_bounds(self) -> RerenderClipEdit:
        if self.source_out <= self.source_in:
            raise ValueError("source_out must be greater than source_in")
        return self


class RerenderRequest(BaseModel):
    edits: list[RerenderClipEdit]
    captions: bool = True
    ducking: bool | None = None

    @model_validator(mode="after")
    def _validate_edits(self) -> RerenderRequest:
        if not self.edits:
            raise ValueError("at least one edit is required")
        return self
