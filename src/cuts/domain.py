from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass(slots=True)
class EditorConfig:
    scene_threshold: float = 27.0
    motion_window_seconds: float = 1.0
    motion_stride_seconds: float = 0.5
    motion_sharpness_threshold: float = 80.0
    motion_flow_threshold: float = 1.2
    silence_window_seconds: float = 0.5
    silence_rms_threshold: float = 250.0
    silence_merge_gap_seconds: float = 0.3
    speech_padding_seconds: float = 0.15
    assembler_min_segment_seconds: float = 0.5
    assembler_gap_snap_seconds: float = 0.2
    assembler_waste_penalty: float = 1.0
    assembler_speech_bonus: float = 2.0
    assembler_sharpness_bonus: float = 0.01


@dataclass(slots=True, frozen=True)
class Clip:
    clip_id: str
    path: Path
    duration: float
    fps: float
    width: int
    height: int
    rotation: int
    has_audio: bool
    creation_time: datetime | None


@dataclass(slots=True, frozen=True)
class Shot:
    clip_id: str
    start: float
    end: float


@dataclass(slots=True, frozen=True)
class MotionWasteSegment:
    clip_id: str
    start: float
    end: float
    score: float
    reason: str


@dataclass(slots=True, frozen=True)
class WordTimestamp:
    clip_id: str
    text: str
    start: float
    end: float
    probability: float | None = None


@dataclass(slots=True, frozen=True)
class SpeechRegion:
    clip_id: str
    start: float
    end: float
    speech: bool
    score: float
    source: str


@dataclass(slots=True, frozen=True)
class BeatGrid:
    music_path: Path
    tempo: float
    beats: tuple[float, ...] = field(default_factory=tuple)
