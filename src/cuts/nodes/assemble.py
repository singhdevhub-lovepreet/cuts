from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from cuts.domain import BeatGrid, MotionWasteSegment, Shot, SpeechRegion, WordTimestamp
from cuts.edl import AudioTrack, Caption, CaptionTrack, Timeline, TimelineClip
from cuts.graph import Context, Node


@dataclass(slots=True)
class AssemblerDecision:
    clip_id: str
    source_in: float
    source_out: float
    score: float


class AssembleNode(Node):
    name = "assemble"
    requires = ("clips", "shots", "motion_segments", "words", "speech_regions")
    provides = ("timeline",)

    def run(self, context: Context) -> Context:
        timeline = self.assemble(context)
        context.timeline = timeline
        return context

    def assemble(self, context: Context) -> Timeline:
        decisions = self._select_segments(context)
        clips = [
            TimelineClip(
                source_clip_id=decision.clip_id,
                source_path=self._clip_path(context, decision.clip_id),
                source_in=decision.source_in,
                source_out=decision.source_out,
                has_audio=self._clip_has_audio(context, decision.clip_id),
            )
            for decision in decisions
        ]
        captions = self._build_captions(clips, context.words)
        duration = sum(clip.source_out - clip.source_in for clip in clips)
        return Timeline(
            target_width=context.target_width,
            target_height=context.target_height,
            target_fps=context.target_fps,
            duration=duration,
            clips=clips,
            caption_tracks=[CaptionTrack(captions=captions)] if captions else [],
            overlay_tracks=[],
            audio=AudioTrack(music_path=context.music_path, ducking=context.music_path is not None),
        )

    def _select_segments(self, context: Context) -> list[AssemblerDecision]:
        decisions: list[AssemblerDecision] = []
        target_duration = context.target_duration
        motion_by_clip = self._group_motion(context.motion_segments)
        speech_by_clip = self._group_speech(context.speech_regions)
        shot_by_clip = self._group_shots(context.shots)
        for clip in context.clips:
            clip_speech_regions = speech_by_clip.get(clip.clip_id, [])
            clip_has_speech = bool(clip_speech_regions)
            for shot in shot_by_clip.get(
                clip.clip_id, [Shot(clip_id=clip.clip_id, start=0.0, end=clip.duration)]
            ):
                if shot.end <= shot.start:
                    continue
                overlap_waste = self._overlap_with_motion(
                    shot, motion_by_clip.get(clip.clip_id, [])
                )
                speech_overlap = self._overlap_with_speech(shot, clip_speech_regions)
                sharpness_bonus = (
                    max(0.0, shot.end - shot.start) * context.config.assembler_sharpness_bonus
                )
                score = (
                    speech_overlap * context.config.assembler_speech_bonus
                    + sharpness_bonus
                    - overlap_waste * context.config.assembler_waste_penalty
                )
                if clip_has_speech and speech_overlap <= 0.0:
                    continue
                if not clip_has_speech and overlap_waste > 0.0:
                    continue
                if shot.end - shot.start < context.config.assembler_min_segment_seconds:
                    continue
                decisions.append(
                    AssemblerDecision(
                        clip_id=clip.clip_id,
                        source_in=shot.start,
                        source_out=shot.end,
                        score=score,
                    )
                )
        selected: list[AssemblerDecision] = []
        total = 0.0
        for decision in decisions:
            duration = decision.source_out - decision.source_in
            if target_duration is not None and total + duration > target_duration:
                remaining = target_duration - total
                if remaining <= 0:
                    break
                selected.append(
                    AssemblerDecision(
                        clip_id=decision.clip_id,
                        source_in=decision.source_in,
                        source_out=decision.source_in + remaining,
                        score=decision.score,
                    )
                )
                total = target_duration
                break
            selected.append(decision)
            total += duration
        if context.beat_grid is not None and selected:
            selected = self._snap_to_beats(selected, context.beat_grid)
        return selected

    def _snap_to_beats(
        self, decisions: list[AssemblerDecision], beat_grid: BeatGrid
    ) -> list[AssemblerDecision]:
        if not beat_grid.beats:
            return decisions
        # Heuristic Phase-0 placeholder: this still snaps source-relative cut
        # points to beat timestamps and should be revisited when cut timing is
        # aligned on the output timeline.
        snapped: list[AssemblerDecision] = []
        for decision in decisions:
            start = self._snap_point(decision.source_in, beat_grid.beats)
            end = self._snap_point(decision.source_out, beat_grid.beats)
            if end <= start:
                end = decision.source_out
            snapped.append(
                AssemblerDecision(
                    clip_id=decision.clip_id,
                    source_in=start,
                    source_out=end,
                    score=decision.score,
                )
            )
        return snapped

    def _snap_point(self, value: float, beats: tuple[float, ...]) -> float:
        return min(beats, key=lambda beat: (abs(beat - value), beat))

    def _build_captions(
        self, clips: list[TimelineClip], words: list[WordTimestamp]
    ) -> list[Caption]:
        captions: list[Caption] = []
        ordered_words = sorted(
            words, key=lambda item: (item.clip_id, item.start, item.end, item.text)
        )
        output_offset = 0.0
        for clip in clips:
            for word in ordered_words:
                if word.clip_id != clip.source_clip_id:
                    continue
                if word.start < clip.source_in or word.end > clip.source_out:
                    continue
                start = output_offset + (word.start - clip.source_in)
                end = output_offset + (min(word.end, clip.source_out) - clip.source_in)
                if end <= start:
                    continue
                captions.append(
                    Caption(
                        source_clip_id=word.clip_id,
                        start=start,
                        end=end,
                        text=word.text,
                    )
                )
            output_offset += clip.source_out - clip.source_in
        return captions

    def _group_shots(self, shots: list[Shot]) -> dict[str, list[Shot]]:
        grouped: dict[str, list[Shot]] = {}
        for shot in shots:
            grouped.setdefault(shot.clip_id, []).append(shot)
        for items in grouped.values():
            items.sort(key=lambda shot: (shot.start, shot.end))
        return grouped

    def _group_motion(
        self, segments: list[MotionWasteSegment]
    ) -> dict[str, list[MotionWasteSegment]]:
        grouped: dict[str, list[MotionWasteSegment]] = {}
        for segment in segments:
            grouped.setdefault(segment.clip_id, []).append(segment)
        return grouped

    def _group_speech(self, regions: list[SpeechRegion]) -> dict[str, list[SpeechRegion]]:
        grouped: dict[str, list[SpeechRegion]] = {}
        for region in regions:
            if region.speech:
                grouped.setdefault(region.clip_id, []).append(region)
        return grouped

    def _overlap_with_motion(self, shot: Shot, motion_segments: list[MotionWasteSegment]) -> float:
        return sum(
            self._overlap(shot.start, shot.end, segment.start, segment.end)
            for segment in motion_segments
        )

    def _overlap_with_speech(self, shot: Shot, speech_regions: list[SpeechRegion]) -> float:
        return sum(
            self._overlap(shot.start, shot.end, region.start, region.end)
            for region in speech_regions
        )

    def _overlap(self, start_a: float, end_a: float, start_b: float, end_b: float) -> float:
        start = max(start_a, start_b)
        end = min(end_a, end_b)
        return max(0.0, end - start)

    def _clip_path(self, context: Context, clip_id: str) -> Path:
        for clip in context.clips:
            if clip.clip_id == clip_id:
                return clip.path
        raise KeyError(f"unknown clip id: {clip_id}")

    def _clip_has_audio(self, context: Context, clip_id: str) -> bool:
        for clip in context.clips:
            if clip.clip_id == clip_id:
                return clip.has_audio
        raise KeyError(f"unknown clip id: {clip_id}")
