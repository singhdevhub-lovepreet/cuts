from __future__ import annotations

from pathlib import Path

import pytest

from cuts.domain import WordTimestamp
from cuts.edl import (
    AudioTrack,
    Caption,
    CaptionTrack,
    CropKeyframe,
    Timeline,
    TimelineClip,
    Transition,
)
from cuts.pipeline import RerenderClipEdit, rebuild_timeline_for_rerender


def test_rebuild_timeline_for_rerender_reorders_trims_and_recomputes_captions() -> None:
    original = Timeline(
        clips=[
            TimelineClip(
                source_clip_id="clip-1",
                source_path=Path("clip1.webm"),
                source_in=0.0,
                source_out=4.0,
                crop_path=[CropKeyframe(t=0.0, center_x=0.4, center_y=0.5)],
            ),
            TimelineClip(
                source_clip_id="clip-2",
                source_path=Path("clip2.webm"),
                source_in=1.0,
                source_out=5.0,
                crop_path=[CropKeyframe(t=0.0, center_x=0.6, center_y=0.5)],
            ),
            TimelineClip(
                source_clip_id="clip-3",
                source_path=Path("clip3.webm"),
                source_in=0.0,
                source_out=3.0,
                crop_path=[CropKeyframe(t=0.0, center_x=0.5, center_y=0.6)],
            ),
        ],
        caption_tracks=[
            CaptionTrack(
                captions=[Caption(source_clip_id="clip-1", start=0.2, end=0.5, text="old")]
            )
        ],
        audio=AudioTrack(music_path=Path("music.mp3"), ducking=True),
    )
    words = [
        WordTimestamp(clip_id="clip-2", text="two", start=1.2, end=1.5),
        WordTimestamp(clip_id="clip-1", text="one", start=0.5, end=0.8),
    ]
    timeline = rebuild_timeline_for_rerender(
        original,
        words,
        [
            RerenderClipEdit(
                original_index=1,
                source_in=1.0,
                source_out=5.0,
                transition_kind="cut",
                transition_duration=0.0,
            ),
            RerenderClipEdit(
                original_index=0,
                source_in=0.5,
                source_out=3.0,
                transition_kind="fade",
                transition_duration=1.0,
            ),
        ],
        ducking_override=False,
    )

    assert [clip.source_clip_id for clip in timeline.clips] == ["clip-2", "clip-1"]
    assert [clip.source_in for clip in timeline.clips] == [1.0, 0.5]
    assert [clip.source_out for clip in timeline.clips] == [5.0, 3.0]
    assert timeline.clips[0].crop_path == original.clips[1].crop_path
    assert timeline.clips[1].crop_path == []
    assert timeline.clips[1].transition == Transition(kind="fade", duration=1.0)
    assert timeline.duration == pytest.approx(5.5)
    assert timeline.audio.ducking is False
    assert [caption.text for caption in timeline.caption_tracks[0].captions] == ["two", "one"]
    assert [caption.start for caption in timeline.caption_tracks[0].captions] == [
        pytest.approx(0.2),
        pytest.approx(3.0),
    ]
    assert [caption.end for caption in timeline.caption_tracks[0].captions] == [
        pytest.approx(0.5),
        pytest.approx(3.3),
    ]
