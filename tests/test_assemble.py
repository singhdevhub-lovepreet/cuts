from __future__ import annotations

from pathlib import Path

import pytest

from cuts.domain import (
    BeatGrid,
    Clip,
    EditorConfig,
    MotionWasteSegment,
    Shot,
    SpeechRegion,
    WordTimestamp,
)
from cuts.graph import Context
from cuts.nodes.assemble import AssembleNode
from cuts.vlm.models import SequencePlan, SequencePlanItem


def test_assemble_is_deterministic_and_prefers_speech() -> None:
    clip = Clip(
        clip_id="clip-1",
        path=Path("clip1.mp4"),
        duration=10.0,
        fps=30.0,
        width=1080,
        height=1920,
        rotation=0,
        has_audio=True,
        creation_time=None,
    )
    context = Context(source_paths=(clip.path,), target_duration=6.0)
    context.config = EditorConfig(
        assembler_min_segment_seconds=0.5,
        assembler_waste_penalty=10.0,
        assembler_speech_bonus=5.0,
        assembler_sharpness_bonus=0.0,
    )
    context.clips = [clip]
    context.shots = [
        Shot(clip_id="clip-1", start=0.0, end=3.0),
        Shot(clip_id="clip-1", start=3.0, end=6.0),
        Shot(clip_id="clip-1", start=6.0, end=10.0),
    ]
    context.motion_segments = [
        MotionWasteSegment(clip_id="clip-1", start=6.0, end=10.0, score=1.0, reason="blur")
    ]
    context.words = [WordTimestamp(clip_id="clip-1", text="hello", start=3.2, end=3.5)]
    context.speech_regions = [
        SpeechRegion(
            clip_id="clip-1", start=3.0, end=4.0, speech=True, score=1.0, source="transcript"
        )
    ]

    timeline = AssembleNode().assemble(context)
    assert len(timeline.clips) == 1
    assert timeline.clips[0].source_in == 3.0
    assert timeline.clips[0].source_out == 6.0
    assert timeline.caption_tracks[0].captions[0].text == "hello"
    assert timeline.caption_tracks[0].captions[0].start == pytest.approx(0.2)
    assert timeline.caption_tracks[0].captions[0].end == pytest.approx(0.5)
    assert timeline.duration == 3.0


def test_assemble_offsets_captions_and_drops_unselected_words() -> None:
    clip = Clip(
        clip_id="clip-1",
        path=Path("clip1.mp4"),
        duration=6.0,
        fps=30.0,
        width=1080,
        height=1920,
        rotation=0,
        has_audio=True,
        creation_time=None,
    )
    context = Context(source_paths=(clip.path,), target_duration=4.0)
    context.config = EditorConfig(
        assembler_min_segment_seconds=0.5,
        assembler_waste_penalty=0.0,
        assembler_speech_bonus=1.0,
        assembler_sharpness_bonus=0.0,
    )
    context.clips = [clip]
    context.shots = [
        Shot(clip_id="clip-1", start=0.0, end=2.0),
        Shot(clip_id="clip-1", start=2.0, end=4.0),
        Shot(clip_id="clip-1", start=4.0, end=6.0),
    ]
    context.speech_regions = [
        SpeechRegion(clip_id="clip-1", start=0.0, end=4.0, speech=True, score=1.0, source="vad")
    ]
    context.words = [
        WordTimestamp(clip_id="clip-1", text="first", start=0.4, end=0.7),
        WordTimestamp(clip_id="clip-1", text="second", start=2.2, end=2.5),
        WordTimestamp(clip_id="clip-1", text="dropped", start=4.4, end=4.7),
    ]

    timeline = AssembleNode().assemble(context)
    assert [clip.source_in for clip in timeline.clips] == [0.0, 2.0]
    assert [caption.text for caption in timeline.caption_tracks[0].captions] == ["first", "second"]
    assert [caption.start for caption in timeline.caption_tracks[0].captions] == [
        pytest.approx(0.4),
        pytest.approx(2.2),
    ]
    assert [caption.end for caption in timeline.caption_tracks[0].captions] == [
        pytest.approx(0.7),
        pytest.approx(2.5),
    ]
    assert all(caption.text != "dropped" for caption in timeline.caption_tracks[0].captions)


def test_beat_sync_aligns_on_output_timeline_across_clips() -> None:
    first = Clip(
        clip_id="clip-a",
        path=Path("clip-a.mp4"),
        duration=5.0,
        fps=30.0,
        width=1080,
        height=1920,
        rotation=0,
        has_audio=True,
        creation_time=None,
    )
    second = Clip(
        clip_id="clip-b",
        path=Path("clip-b.mp4"),
        duration=5.0,
        fps=30.0,
        width=1080,
        height=1920,
        rotation=0,
        has_audio=True,
        creation_time=None,
    )
    context = Context(source_paths=(first.path, second.path), target_duration=6.0)
    context.config = EditorConfig(
        assembler_min_segment_seconds=0.5,
        assembler_beat_sync=True,
        assembler_beat_snap_max_seconds=1.0,
        assembler_waste_penalty=0.0,
        assembler_speech_bonus=0.0,
        assembler_sharpness_bonus=0.0,
    )
    context.clips = [first, second]
    context.shots = [
        Shot(clip_id="clip-a", start=0.0, end=2.0),
        Shot(clip_id="clip-b", start=1.2, end=3.2),
    ]
    context.beat_grid = BeatGrid(
        music_path=Path("music.mp3"),
        tempo=120.0,
        beats=(0.0, 1.0, 2.5, 4.0),
    )

    timeline = AssembleNode().assemble(context)
    assert [clip.source_in for clip in timeline.clips] == [0.0, 1.2]
    assert [clip.source_out for clip in timeline.clips] == [1.0, pytest.approx(2.7)]


def test_beat_sync_can_be_disabled() -> None:
    clip = Clip(
        clip_id="clip-1",
        path=Path("clip1.mp4"),
        duration=4.0,
        fps=30.0,
        width=1080,
        height=1920,
        rotation=0,
        has_audio=True,
        creation_time=None,
    )
    context = Context(source_paths=(clip.path,), target_duration=4.0)
    context.config = EditorConfig(
        assembler_min_segment_seconds=0.5,
        assembler_beat_sync=False,
        assembler_beat_snap_max_seconds=1.0,
        assembler_waste_penalty=0.0,
        assembler_speech_bonus=0.0,
        assembler_sharpness_bonus=0.0,
    )
    context.clips = [clip]
    context.shots = [Shot(clip_id="clip-1", start=0.2, end=3.4)]
    context.beat_grid = BeatGrid(
        music_path=Path("music.mp3"), tempo=120.0, beats=(0.0, 1.0, 2.5, 4.0)
    )

    timeline = AssembleNode().assemble(context)
    assert len(timeline.clips) == 1
    assert timeline.clips[0].source_in == 0.2
    assert timeline.clips[0].source_out == 3.4


def test_beat_sync_applies_to_sequence_plan_path() -> None:
    clip = Clip(
        clip_id="clip-1",
        path=Path("clip1.mp4"),
        duration=4.0,
        fps=30.0,
        width=1080,
        height=1920,
        rotation=0,
        has_audio=True,
        creation_time=None,
    )
    context = Context(source_paths=(clip.path,), target_duration=4.0)
    context.config = EditorConfig(
        assembler_min_segment_seconds=0.5,
        assembler_beat_sync=True,
        assembler_beat_snap_max_seconds=1.0,
    )
    context.clips = [clip]
    context.sequence_plan = SequencePlan(
        rationale="test",
        ordered_shots=[
            SequencePlanItem(
                shot_index=0,
                clip_id="clip-1",
                shot_start=0.0,
                shot_end=4.0,
                keep=True,
                trim_in=1.2,
                trim_out=3.2,
                rationale="keep",
            )
        ],
    )
    context.beat_grid = BeatGrid(music_path=Path("music.mp3"), tempo=120.0, beats=(0.0, 1.0, 2.5))

    timeline = AssembleNode().assemble(context)
    assert len(timeline.clips) == 1
    assert timeline.clips[0].source_in == 1.2
    assert timeline.clips[0].source_out == pytest.approx(2.2)
