from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from cuts.domain import BeatGrid
from cuts.graph import Context, Node


@dataclass(slots=True)
class BeatAnalysisResult:
    beat_grid: BeatGrid | None


class BeatsNode(Node):
    name = "beats"
    requires = ()
    provides = ("beat_grid",)

    def run(self, context: Context) -> Context:
        if context.music_path is None:
            context.beat_grid = None
            return context
        context.beat_grid = self._analyze_music(context.music_path)
        return context

    def _analyze_music(self, path: Path) -> BeatGrid | None:
        try:
            import librosa
            import numpy as np
        except ImportError:
            return None
        audio, sample_rate = librosa.load(str(path), sr=None, mono=True)
        tempo, beats = librosa.beat.beat_track(y=audio, sr=sample_rate)
        beat_times = tuple(float(time) for time in librosa.frames_to_time(beats, sr=sample_rate))
        tempo_value = float(np.asarray(tempo, dtype=float).reshape(-1)[0])
        return BeatGrid(music_path=path, tempo=tempo_value, beats=beat_times)
