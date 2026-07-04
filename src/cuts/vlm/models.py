from __future__ import annotations

from enum import Enum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Platform(str, Enum):
    REELS = "reels"
    SHORTS = "shorts"
    TIKTOK = "tiktok"


class VibeIntent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    vibe_prompt: str = ""
    platform: Platform = Platform.REELS
    target_duration: float | None = None

    @field_validator("target_duration")
    @classmethod
    def _target_duration_positive(cls, value: float | None) -> float | None:
        if value is not None and value <= 0:
            raise ValueError("target duration must be positive")
        return value


class ShotFrameSample(BaseModel):
    model_config = ConfigDict(extra="forbid")

    shot_index: int
    clip_id: str
    shot_start: float
    shot_end: float
    frame_index: int
    frame_time: float
    frame_path: Path
    frame_hash: str


class ShotObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    shot_index: int
    clip_id: str
    shot_start: float
    shot_end: float
    frames: list[ShotFrameSample] = Field(default_factory=list)


class ShotTags(BaseModel):
    model_config = ConfigDict(extra="forbid")

    shot_index: int
    clip_id: str
    shot_start: float
    shot_end: float
    subject: str
    action: str
    shot_type: str
    setting: str
    energy: float
    mood_tags: list[str] = Field(default_factory=list)
    role: str
    caption: str

    @field_validator("energy")
    @classmethod
    def _energy_normalized(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            if 1.0 < value <= 5.0:
                return (value - 1.0) / 4.0
            return max(0.0, min(value, 1.0))
        return value


class SequencePlanItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    shot_index: int
    clip_id: str
    shot_start: float
    shot_end: float
    keep: bool = True
    trim_in: float | None = None
    trim_out: float | None = None
    rationale: str = ""


class SequencePlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rationale: str
    ordered_shots: list[SequencePlanItem] = Field(default_factory=list)
