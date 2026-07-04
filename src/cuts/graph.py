from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

from cuts.domain import (
    BeatGrid,
    Clip,
    EditorConfig,
    MotionWasteSegment,
    Shot,
    SpeechRegion,
    WordTimestamp,
)
from cuts.edl import Timeline
from cuts.vlm.models import Platform, SequencePlan, ShotTags


@dataclass(slots=True)
class Context:
    source_paths: tuple[Path, ...]
    music_path: Path | None = None
    target_duration: float | None = None
    vibe_prompt: str = ""
    platform: Platform = Platform.REELS
    whisper_model: str = "base"
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"
    target_width: int = 1080
    target_height: int = 1920
    target_fps: float = 30.0
    output_path: Path | None = None
    config: EditorConfig = field(default_factory=EditorConfig)
    clips: list[Clip] = field(default_factory=list)
    shots: list[Shot] = field(default_factory=list)
    motion_segments: list[MotionWasteSegment] = field(default_factory=list)
    words: list[WordTimestamp] = field(default_factory=list)
    speech_regions: list[SpeechRegion] = field(default_factory=list)
    beat_grid: BeatGrid | None = None
    shot_tags: list[ShotTags] = field(default_factory=list)
    sequence_plan: SequencePlan | None = None
    timeline: Timeline | None = None
    warnings: list[str] = field(default_factory=list)
    extras: dict[str, object] = field(default_factory=dict)


class Node(ABC):
    name: ClassVar[str]
    requires: ClassVar[tuple[str, ...]] = ()
    provides: ClassVar[tuple[str, ...]] = ()

    @abstractmethod
    def run(self, context: Context) -> Context:
        raise NotImplementedError


class Pipeline:
    def __init__(self, nodes: Sequence[Node]) -> None:
        self._nodes = list(nodes)
        self._ordered_nodes = self._topological_order()

    def _topological_order(self) -> list[Node]:
        providers: dict[str, Node] = {}
        for node in self._nodes:
            for key in node.provides:
                if key in providers:
                    previous = providers[key]
                    raise ValueError(
                        f"field {key!r} is provided by both {previous.name!r} and {node.name!r}"
                    )
                providers[key] = node

        available = {
            "source_paths",
            "music_path",
            "target_duration",
            "vibe_prompt",
            "platform",
            "whisper_model",
            "whisper_device",
            "whisper_compute_type",
            "target_width",
            "target_height",
            "target_fps",
            "output_path",
            "config",
        }
        remaining = list(self._nodes)
        ordered: list[Node] = []

        while remaining:
            progress = False
            for node in list(remaining):
                missing = [name for name in node.requires if name not in available]
                if missing:
                    continue
                ordered.append(node)
                available.update(node.provides)
                remaining.remove(node)
                progress = True
            if not progress:
                missing_descriptions = []
                for node in remaining:
                    missing = [name for name in node.requires if name not in available]
                    missing_descriptions.append(
                        f"{node.name}: {', '.join(missing) if missing else 'unknown'}"
                    )
                raise ValueError(
                    "unresolvable pipeline dependencies: " + "; ".join(missing_descriptions)
                )
        return ordered

    @property
    def ordered_nodes(self) -> tuple[Node, ...]:
        return tuple(self._ordered_nodes)

    def run(self, context: Context) -> Context:
        for node in self._ordered_nodes:
            context = node.run(context)
        return context
