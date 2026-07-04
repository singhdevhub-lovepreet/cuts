from __future__ import annotations

from pathlib import Path

from cuts.domain import Clip, EditorConfig, Shot, SpeechRegion, WordTimestamp
from cuts.graph import Context
from cuts.nodes.assemble import AssembleNode
from cuts.nodes.sequence import SequencerNode
from cuts.nodes.vibe import VibeTaggerNode
from cuts.vlm.mock import MockVLMClient
from cuts.vlm.models import SequencePlan, SequencePlanItem, ShotFrameSample, ShotTags


def test_shot_tags_round_trip_json() -> None:
    tags = ShotTags(
        shot_index=1,
        clip_id="clip-1",
        shot_start=1.0,
        shot_end=3.0,
        subject="person",
        action="talking",
        shot_type="close-up",
        setting="indoors",
        energy=0.75,
        mood_tags=["warm", "reels"],
        role="hero",
        caption="person talking in indoors",
    )
    loaded = ShotTags.model_validate_json(tags.model_dump_json())
    assert loaded == tags


def test_shot_tags_energy_normalizes_likert_scale() -> None:
    tags = ShotTags(
        shot_index=0,
        clip_id="clip-1",
        shot_start=0.0,
        shot_end=1.0,
        subject="person",
        action="talking",
        shot_type="close-up",
        setting="indoors",
        energy=4.0,
        mood_tags=[],
        role="hero",
        caption="caption",
    )
    assert tags.energy == 0.75


def test_vibe_tagger_node_with_mock_provider_uses_cached_samples(tmp_path: Path) -> None:
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
    context = Context(source_paths=(clip.path,), vibe_prompt="warm and upbeat")
    context.clips = [clip]
    context.shots = [Shot(clip_id="clip-1", start=0.0, end=2.0)]
    node = VibeTaggerNode(client=MockVLMClient(), cache_dir=tmp_path / "cache")

    frame_samples = [
        ShotFrameSample(
            shot_index=0,
            clip_id="clip-1",
            shot_start=0.0,
            shot_end=2.0,
            frame_index=0,
            frame_time=0.3,
            frame_path=tmp_path / "frame-0.jpg",
            frame_hash="frame-hash-0",
        ),
        ShotFrameSample(
            shot_index=0,
            clip_id="clip-1",
            shot_start=0.0,
            shot_end=2.0,
            frame_index=1,
            frame_time=1.0,
            frame_path=tmp_path / "frame-1.jpg",
            frame_hash="frame-hash-1",
        ),
    ]
    node._sample_frames = lambda sample: frame_samples  # type: ignore[method-assign]

    updated = node.run(context)
    assert len(updated.shot_tags) == 1
    assert updated.shot_tags[0].shot_index == 0
    assert updated.shot_tags[0].caption
    assert any(path.suffix == ".json" for path in (tmp_path / "cache").iterdir())


def test_sequencer_orders_and_drops_waste() -> None:
    context = Context(source_paths=(Path("clip1.mp4"),), vibe_prompt="energetic edits")
    context.shot_tags = [
        ShotTags(
            shot_index=0,
            clip_id="clip-1",
            shot_start=0.0,
            shot_end=2.0,
            subject="person",
            action="talking",
            shot_type="close-up",
            setting="indoors",
            energy=0.9,
            mood_tags=["hero"],
            role="hero",
            caption="hero shot",
        ),
        ShotTags(
            shot_index=1,
            clip_id="clip-1",
            shot_start=2.0,
            shot_end=4.0,
            subject="detail",
            action="moving",
            shot_type="detail",
            setting="indoors",
            energy=0.7,
            mood_tags=["broll"],
            role="broll",
            caption="broll shot",
        ),
        ShotTags(
            shot_index=2,
            clip_id="clip-1",
            shot_start=4.0,
            shot_end=6.0,
            subject="junk",
            action="static",
            shot_type="wide",
            setting="outside",
            energy=0.1,
            mood_tags=["waste"],
            role="waste",
            caption="waste shot",
        ),
    ]
    SequencerNode(client=MockVLMClient()).run(context)
    assert context.sequence_plan is not None
    assert [item.shot_index for item in context.sequence_plan.ordered_shots] == [0, 1]


def test_sequence_plan_flows_into_assemble() -> None:
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
    context.motion_segments = []
    context.words = [
        WordTimestamp(clip_id="clip-1", text="hello", start=2.5, end=2.8),
        WordTimestamp(clip_id="clip-1", text="again", start=0.5, end=0.8),
    ]
    context.speech_regions = [
        SpeechRegion(clip_id="clip-1", start=0.0, end=4.0, speech=True, score=1.0, source="vad")
    ]
    context.sequence_plan = SequencePlan(
        rationale="mock narrative",
        ordered_shots=[
            SequencePlanItem(
                shot_index=1,
                clip_id="clip-1",
                shot_start=2.0,
                shot_end=4.0,
                keep=True,
                trim_in=2.0,
                trim_out=4.0,
                rationale="second first",
            ),
            SequencePlanItem(
                shot_index=0,
                clip_id="clip-1",
                shot_start=0.0,
                shot_end=2.0,
                keep=True,
                trim_in=0.0,
                trim_out=2.0,
                rationale="first second",
            ),
        ],
    )

    timeline = AssembleNode().assemble(context)
    assert [clip.source_in for clip in timeline.clips] == [2.0, 0.0]
    assert timeline.duration == 4.0
    assert [caption.text for caption in timeline.caption_tracks[0].captions] == ["hello", "again"]
    assert timeline.caption_tracks[0].captions[0].start == 0.5
    assert timeline.caption_tracks[0].captions[1].start == 2.5
