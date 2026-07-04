from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator

SCHEMA_VERSION = "0.2.0"


class Transition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["cut", "fade"] = "cut"
    duration: float = 0.0

    @field_validator("duration")
    @classmethod
    def _duration_non_negative(cls, value: float) -> float:
        if value < 0:
            raise ValueError("transition duration must be non-negative")
        return value


class TimelineClip(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_clip_id: str
    source_path: Path
    source_in: float
    source_out: float
    transition: Transition = Field(default_factory=Transition)
    has_audio: bool = True
    crop_aspect: float = 9.0 / 16.0
    crop_path: list[CropKeyframe] = Field(default_factory=list)

    @field_validator("source_out")
    @classmethod
    def _out_after_in(cls, value: float, info: ValidationInfo) -> float:
        source_in = info.data.get("source_in")
        if source_in is not None and value <= source_in:
            raise ValueError("source_out must be greater than source_in")
        return value


class CropKeyframe(BaseModel):
    model_config = ConfigDict(extra="forbid")

    t: float
    center_x: float
    center_y: float

    @field_validator("t")
    @classmethod
    def _time_non_negative(cls, value: float) -> float:
        if value < 0:
            raise ValueError("crop keyframe time must be non-negative")
        return value

    @field_validator("center_x", "center_y")
    @classmethod
    def _center_normalized(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("crop keyframe centers must be normalized")
        return value


class Caption(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_clip_id: str
    start: float
    end: float
    text: str


class CaptionTrack(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = "captions"
    captions: list[Caption] = Field(default_factory=list)


class OverlayItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_clip_id: str | None = None
    start: float
    end: float
    kind: str
    text: str


class OverlayTrack(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = "overlays"
    items: list[OverlayItem] = Field(default_factory=list)


class AudioTrack(BaseModel):
    model_config = ConfigDict(extra="forbid")

    music_path: Path | None = None
    ducking: bool = False
    normalize_lufs: float = -14.0


class Timeline(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = SCHEMA_VERSION
    target_width: int = 1080
    target_height: int = 1920
    target_fps: float = 30.0
    duration: float | None = None
    clips: list[TimelineClip] = Field(default_factory=list)
    caption_tracks: list[CaptionTrack] = Field(default_factory=list)
    overlay_tracks: list[OverlayTrack] = Field(default_factory=list)
    audio: AudioTrack = Field(default_factory=AudioTrack)


TimelineClip.model_rebuild()
Timeline.model_rebuild()
