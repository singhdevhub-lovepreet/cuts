from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

from cuts.domain import Clip, Shot
from cuts.graph import Context, Node
from cuts.vlm.cache import default_cache_dir, stable_json_hash
from cuts.vlm.client import VLMClient
from cuts.vlm.gemini import build_gemini_client
from cuts.vlm.mock import MockVLMClient
from cuts.vlm.models import ShotFrameSample, ShotObservation, ShotTags, VibeIntent


@dataclass(slots=True)
class _ShotSamplePlan:
    shot_index: int
    clip: Clip
    shot: Shot
    frame_times: tuple[float, ...]


class VibeTaggerNode(Node):
    name = "vibe_tagger"
    requires = ("clips", "shots")
    provides = ("shot_tags",)

    def __init__(
        self,
        client: VLMClient | None = None,
        cache_dir: Path | None = None,
    ) -> None:
        self._client = client
        self._cache_dir = cache_dir or default_cache_dir() / "tags"

    def run(self, context: Context) -> Context:
        intent = self._intent(context)
        client = self._client or self._default_client()
        if client is None:
            context.warnings.append("vibe tagging unavailable; using phase-0 path")
            return context
        samples = self._build_samples(context)
        if not samples:
            context.shot_tags = []
            return context

        frames_by_shot = [self._sample_frames(sample) for sample in samples]
        observations = [
            ShotObservation(
                shot_index=sample.shot_index,
                clip_id=sample.clip.clip_id,
                shot_start=sample.shot.start,
                shot_end=sample.shot.end,
                frames=frames,
            )
            for sample, frames in zip(samples, frames_by_shot, strict=True)
        ]
        cache_key = stable_json_hash(
            {
                "model": client.model_name,
                "intent": intent.model_dump(mode="json"),
                "samples": [
                    {
                        "shot_index": observation.shot_index,
                        "clip_id": observation.clip_id,
                        "shot_start": observation.shot_start,
                        "shot_end": observation.shot_end,
                        "frame_hashes": [frame.frame_hash for frame in observation.frames],
                    }
                    for observation in observations
                ],
            }
        )
        cache_path = self._cache_dir / f"{cache_key}.json"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        if cache_path.exists():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            context.shot_tags = [ShotTags.model_validate(item) for item in cached["shot_tags"]]
            return context

        tagged = client.describe_shots(observations, intent)
        context.shot_tags = tagged
        cache_path.write_text(
            json.dumps(
                {
                    "intent": intent.model_dump(mode="json"),
                    "model": client.model_name,
                    "shot_tags": [tag.model_dump(mode="json") for tag in tagged],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return context

    def _default_client(self) -> VLMClient | None:
        client = build_gemini_client()
        return client if client is not None else MockVLMClient()

    def _intent(self, context: Context) -> VibeIntent:
        return VibeIntent(
            vibe_prompt=context.vibe_prompt,
            platform=context.platform,
            target_duration=context.target_duration,
        )

    def _build_samples(self, context: Context) -> list[_ShotSamplePlan]:
        clip_by_id = {clip.clip_id: clip for clip in context.clips}
        samples: list[_ShotSamplePlan] = []
        for shot_index, shot in enumerate(context.shots):
            clip = clip_by_id.get(shot.clip_id)
            if clip is None:
                continue
            samples.append(
                _ShotSamplePlan(
                    shot_index=shot_index,
                    clip=clip,
                    shot=shot,
                    frame_times=self._frame_times(shot.start, shot.end),
                )
            )
        return samples

    def _sample_frames(self, sample: _ShotSamplePlan) -> list[ShotFrameSample]:
        frame_dir = self._frame_dir(sample.clip.path, sample.shot.start, sample.shot.end)
        frame_dir.mkdir(parents=True, exist_ok=True)
        frames: list[ShotFrameSample] = []
        for frame_index, frame_time in enumerate(sample.frame_times):
            frame_path = frame_dir / f"{frame_index:03d}.jpg"
            self._extract_frame(sample.clip.path, frame_time, frame_path)
            frame_hash = stable_json_hash(
                {
                    "clip": str(sample.clip.path),
                    "shot_index": sample.shot_index,
                    "frame_time": frame_time,
                    "bytes": frame_path.read_bytes().hex() if frame_path.exists() else "",
                }
            )
            frames.append(
                ShotFrameSample(
                    shot_index=sample.shot_index,
                    clip_id=sample.clip.clip_id,
                    shot_start=sample.shot.start,
                    shot_end=sample.shot.end,
                    frame_index=frame_index,
                    frame_time=frame_time,
                    frame_path=frame_path,
                    frame_hash=frame_hash,
                )
            )
        return frames

    def _frame_dir(self, clip_path: Path, shot_start: float, shot_end: float) -> Path:
        cache_key = stable_json_hash(
            {
                "clip": str(clip_path.resolve()),
                "start": round(shot_start, 6),
                "end": round(shot_end, 6),
            }
        )
        return self._cache_dir / "frames" / cache_key

    def _frame_times(self, shot_start: float, shot_end: float) -> tuple[float, ...]:
        duration = max(shot_end - shot_start, 1e-6)
        if duration <= 0.5:
            return (round((shot_start + shot_end) / 2.0, 6),)
        positions = (0.15, 0.5, 0.85)
        return tuple(round(shot_start + duration * position, 6) for position in positions)

    def _extract_frame(self, clip_path: Path, frame_time: float, output_path: Path) -> None:
        if output_path.exists():
            return
        command = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{frame_time:.6f}",
            "-i",
            str(clip_path),
            "-frames:v",
            "1",
            "-q:v",
            "2",
            str(output_path),
        ]
        subprocess.run(command, check=True)


def build_default_vlm_client() -> VLMClient:
    client = build_gemini_client()
    return client if client is not None else MockVLMClient()
