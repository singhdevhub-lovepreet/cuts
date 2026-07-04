from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from cuts.domain import WordTimestamp
from cuts.graph import Context, Node


@dataclass(slots=True)
class TranscriptionResult:
    words: list[WordTimestamp]


class TranscribeNode(Node):
    name = "transcribe"
    requires = ("clips",)
    provides = ("words",)

    def __init__(
        self, model_size: str = "base", device: str = "cpu", compute_type: str = "int8"
    ) -> None:
        self._model_size = model_size
        self._device = device
        self._compute_type = compute_type
        self._model = None

    def run(self, context: Context) -> Context:
        context.words = []
        for clip in context.clips:
            context.words.extend(self._transcribe_clip(clip.path, clip.clip_id, clip.has_audio))
        return context

    def _load_model(self):
        if self._model is not None:
            return self._model
        try:
            from faster_whisper import WhisperModel  # type: ignore[import-not-found]
        except ImportError:
            self._model = None
            return None
        self._model = WhisperModel(
            self._model_size, device=self._device, compute_type=self._compute_type
        )
        return self._model

    def _transcribe_clip(self, path: Path, clip_id: str, has_audio: bool) -> list[WordTimestamp]:
        if not has_audio:
            return []
        model = self._load_model()
        if model is None:
            return []
        segments, _info = model.transcribe(str(path), word_timestamps=True)
        words: list[WordTimestamp] = []
        for segment in segments:
            for word in segment.words or []:
                words.append(
                    WordTimestamp(
                        clip_id=clip_id,
                        text=str(word.word).strip(),
                        start=float(word.start),
                        end=float(word.end),
                        probability=float(word.probability)
                        if word.probability is not None
                        else None,
                    )
                )
        return words
