from __future__ import annotations

import audioop
import io
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path

from cuts.domain import SpeechRegion, WordTimestamp
from cuts.graph import Context, Node


@dataclass(slots=True)
class SilenceAnalysisResult:
    regions: list[SpeechRegion]


class SilenceNode(Node):
    name = "silence"
    requires = ("clips", "words")
    provides = ("speech_regions",)

    def run(self, context: Context) -> Context:
        context.speech_regions = []
        for clip in context.clips:
            transcript_regions = self._regions_from_words(clip.clip_id, context.words)
            vad_regions = self._energy_vad(clip.path, clip.clip_id, clip.duration, context)
            regions = self._merge_regions(
                transcript_regions + vad_regions, context.config.speech_padding_seconds
            )
            context.speech_regions.extend(regions)
        return context

    def _regions_from_words(self, clip_id: str, words: list[WordTimestamp]) -> list[SpeechRegion]:
        clip_words = [word for word in words if word.clip_id == clip_id]
        if not clip_words:
            return []
        ordered = sorted(clip_words, key=lambda word: (word.start, word.end))
        regions: list[SpeechRegion] = []
        start = ordered[0].start
        end = ordered[0].end
        for word in ordered[1:]:
            if word.start <= end + 0.3:
                end = max(end, word.end)
            else:
                regions.append(
                    SpeechRegion(
                        clip_id=clip_id,
                        start=start,
                        end=end,
                        speech=True,
                        score=1.0,
                        source="transcript",
                    )
                )
                start = word.start
                end = word.end
        regions.append(
            SpeechRegion(
                clip_id=clip_id, start=start, end=end, speech=True, score=1.0, source="transcript"
            )
        )
        return regions

    def _energy_vad(
        self, path: Path, clip_id: str, duration: float, context: Context
    ) -> list[SpeechRegion]:
        try:
            command = [
                "ffmpeg",
                "-v",
                "error",
                "-i",
                str(path),
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-f",
                "wav",
                "pipe:1",
            ]
            completed = subprocess.run(command, check=True, capture_output=True)
        except Exception:
            return []
        with wave.open(io.BytesIO(completed.stdout), "rb") as wav_file:
            sample_rate = wav_file.getframerate()
            frame_count = wav_file.getnframes()
            audio = wav_file.readframes(frame_count)
        if sample_rate <= 0:
            return []
        window = max(1, int(sample_rate * context.config.silence_window_seconds))
        samples: list[tuple[float, bool, float]] = []
        for offset in range(0, len(audio), window * 2):
            chunk = audio[offset : offset + window * 2]
            if not chunk:
                continue
            rms = float(audioop.rms(chunk, 2))
            samples.append(
                (offset / 2 / sample_rate, rms > context.config.silence_rms_threshold, rms)
            )
        regions: list[SpeechRegion] = []
        active_start: float | None = None
        active_score = 0.0
        for start, is_speech, score in samples:
            end = min(duration, start + context.config.silence_window_seconds)
            if is_speech:
                if active_start is None:
                    active_start = start
                active_score = max(active_score, score)
            elif active_start is not None:
                regions.append(
                    SpeechRegion(
                        clip_id=clip_id,
                        start=active_start,
                        end=end,
                        speech=True,
                        score=active_score,
                        source="vad",
                    )
                )
                active_start = None
                active_score = 0.0
        if active_start is not None:
            regions.append(
                SpeechRegion(
                    clip_id=clip_id,
                    start=active_start,
                    end=duration,
                    speech=True,
                    score=active_score,
                    source="vad",
                )
            )
        return regions

    def _merge_regions(self, regions: list[SpeechRegion], padding: float) -> list[SpeechRegion]:
        speech_regions = [region for region in regions if region.speech]
        if not speech_regions:
            return []
        ordered = sorted(
            speech_regions, key=lambda region: (region.clip_id, region.start, region.end)
        )
        merged: list[SpeechRegion] = []
        current = ordered[0]
        for region in ordered[1:]:
            if region.clip_id != current.clip_id:
                merged.append(current)
                current = region
                continue
            if region.start <= current.end + padding:
                current = SpeechRegion(
                    clip_id=current.clip_id,
                    start=current.start,
                    end=max(current.end, region.end),
                    speech=True,
                    score=max(current.score, region.score),
                    source=f"{current.source}+{region.source}",
                )
            else:
                merged.append(current)
                current = region
        merged.append(current)
        return merged
