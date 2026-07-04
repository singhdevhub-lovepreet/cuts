from __future__ import annotations

from pathlib import Path

from cuts.edl import (
    AudioTrack,
    Caption,
    CaptionTrack,
    CropKeyframe,
    Timeline,
    TimelineClip,
)


def test_timeline_round_trip_json() -> None:
    timeline = Timeline(
        target_width=1080,
        target_height=1920,
        target_fps=30.0,
        duration=12.5,
        clips=[
            TimelineClip(
                source_clip_id="clip-1",
                source_path=Path("clip1.mp4"),
                source_in=1.0,
                source_out=5.0,
                crop_aspect=9.0 / 16.0,
                crop_path=[
                    CropKeyframe(t=0.0, center_x=0.5, center_y=0.5),
                    CropKeyframe(t=1.0, center_x=0.6, center_y=0.5),
                ],
            )
        ],
        caption_tracks=[
            CaptionTrack(
                captions=[
                    Caption(source_clip_id="clip-1", start=1.2, end=1.5, text="hello"),
                ]
            )
        ],
        audio=AudioTrack(music_path=Path("music.mp3"), ducking=True),
    )
    dumped = timeline.model_dump_json()
    loaded = Timeline.model_validate_json(dumped)
    assert loaded == timeline
    assert loaded.schema_version == "0.2.0"
