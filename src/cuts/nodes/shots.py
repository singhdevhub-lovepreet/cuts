from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from cuts.domain import Shot
from cuts.graph import Context, Node


@dataclass(slots=True)
class ShotDetectionResult:
    shots: list[Shot]


class ShotsNode(Node):
    name = "shots"
    requires = ("clips",)
    provides = ("shots",)

    def run(self, context: Context) -> Context:
        context.shots = []
        for clip in context.clips:
            context.shots.extend(
                self._detect_shots(
                    clip.path, clip.clip_id, clip.duration, context.config.scene_threshold
                )
            )
        return context

    def _detect_shots(
        self, path: Path, clip_id: str, duration: float, threshold: float
    ) -> list[Shot]:
        try:
            from scenedetect import SceneManager, open_video
            from scenedetect.detectors import ContentDetector
        except ImportError:
            return [Shot(clip_id=clip_id, start=0.0, end=duration)]

        video = open_video(str(path))
        manager = SceneManager()
        manager.add_detector(ContentDetector(threshold=threshold))
        manager.detect_scenes(video, show_progress=False)
        scene_list = manager.get_scene_list()
        if not scene_list:
            return [Shot(clip_id=clip_id, start=0.0, end=duration)]
        shots = [
            Shot(clip_id=clip_id, start=start.get_seconds(), end=end.get_seconds())
            for start, end in scene_list
        ]
        if shots[0].start > 0.0:
            shots.insert(0, Shot(clip_id=clip_id, start=0.0, end=shots[0].start))
        last_end = shots[-1].end
        if last_end < duration:
            shots.append(Shot(clip_id=clip_id, start=last_end, end=duration))
        return shots
