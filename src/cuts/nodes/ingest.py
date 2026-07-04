from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha1
from pathlib import Path
from typing import Any

from cuts.domain import Clip
from cuts.graph import Context, Node


@dataclass(slots=True)
class ProbeResult:
    duration: float
    fps: float
    width: int
    height: int
    rotation: int
    has_audio: bool
    creation_time: datetime | None


class IngestNode(Node):
    name = "ingest"
    requires = ("source_paths",)
    provides = ("clips",)

    def run(self, context: Context) -> Context:
        clips = [self._probe_clip(path) for path in context.source_paths]
        clips.sort(key=lambda clip: self._sort_key(clip))
        context.clips = clips
        return context

    def _sort_key(self, clip: Clip) -> tuple[int, float, str]:
        if clip.creation_time is None:
            return (1, float("inf"), clip.path.name)
        return (0, clip.creation_time.timestamp(), clip.path.name)

    def _probe_clip(self, path: Path) -> Clip:
        probe = self._ffprobe(path)
        clip_id = sha1(str(path).encode("utf-8")).hexdigest()[:12]
        return Clip(
            clip_id=clip_id,
            path=path,
            duration=probe.duration,
            fps=probe.fps,
            width=probe.width,
            height=probe.height,
            rotation=probe.rotation,
            has_audio=probe.has_audio,
            creation_time=probe.creation_time,
        )

    def _ffprobe(self, path: Path) -> ProbeResult:
        command = [
            "ffprobe",
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_streams",
            "-show_format",
            str(path),
        ]
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
        payload: dict[str, Any] = json.loads(completed.stdout)
        streams = payload.get("streams", [])
        format_info = payload.get("format", {})
        video_stream = next(stream for stream in streams if stream.get("codec_type") == "video")
        audio_stream = next(
            (stream for stream in streams if stream.get("codec_type") == "audio"), None
        )
        fps = self._parse_fps(
            video_stream.get("avg_frame_rate") or video_stream.get("r_frame_rate")
        )
        rotation = int(
            video_stream.get("tags", {}).get("rotate")
            or (video_stream.get("side_data_list") or [{}])[0].get("rotation", 0)
            or 0
        )
        creation_time = self._parse_datetime(
            video_stream.get("tags", {}).get("creation_time")
            or format_info.get("tags", {}).get("creation_time")
        )
        return ProbeResult(
            duration=float(format_info.get("duration") or video_stream.get("duration") or 0.0),
            fps=fps,
            width=int(video_stream.get("width") or 0),
            height=int(video_stream.get("height") or 0),
            rotation=rotation,
            has_audio=audio_stream is not None,
            creation_time=creation_time,
        )

    def _parse_fps(self, value: object) -> float:
        if not isinstance(value, str) or "/" not in value:
            return 0.0
        numerator_text, denominator_text = value.split("/", 1)
        denominator = float(denominator_text)
        if denominator == 0:
            return 0.0
        return float(numerator_text) / denominator

    def _parse_datetime(self, value: object) -> datetime | None:
        if not isinstance(value, str) or not value:
            return None
        normalized = value.replace("Z", "+00:00")
        parsed = datetime.fromisoformat(normalized)
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=UTC)
        return parsed.astimezone(UTC)
