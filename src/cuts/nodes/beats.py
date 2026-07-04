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
            import librosa  # type: ignore[import-not-found]
        except ImportError:
            return None
        audio, sample_rate = librosa.load(str(path), sr=None, mono=True)
        tempo, beats = librosa.beat.beat_track(y=audio, sr=sample_rate)
        beat_times = tuple(float(time) for time in librosa.frames_to_time(beats, sr=sample_rate))
        return BeatGrid(music_path=path, tempo=float(tempo), beats=beat_times)
