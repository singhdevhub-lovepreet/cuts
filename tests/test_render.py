from __future__ import annotations

from pathlib import Path

from cuts.edl import AudioTrack, Caption, CaptionTrack, CropKeyframe, Timeline, TimelineClip
from cuts.render import build_ffmpeg_plan


def test_ffmpeg_command_construction_includes_ducking_and_subtitles() -> None:
    timeline = Timeline(
        clips=[
            TimelineClip(
                source_clip_id="clip-1",
                source_path=Path("clip1.mp4"),
                source_in=1.0,
                source_out=4.0,
                crop_aspect=9.0 / 16.0,
                crop_path=[
                    CropKeyframe(t=0.0, center_x=0.45, center_y=0.5),
                    CropKeyframe(t=1.5, center_x=0.60, center_y=0.5),
                ],
            )
        ],
        caption_tracks=[
            CaptionTrack(
                captions=[Caption(source_clip_id="clip-1", start=1.2, end=1.6, text="hello")]
            )
        ],
        audio=AudioTrack(music_path=Path("music.mp3"), ducking=True),
    )
    plan = build_ffmpeg_plan(timeline, Path("out.mp4"), work_dir=Path("/tmp"))
    command = " ".join(plan.command)
    assert "ffmpeg" in command
    assert "crop='" in plan.filter_complex
    assert "min(max((" in plan.filter_complex
    assert "[acatf]asplit=2[speech_mix][speech_sc]" in plan.filter_complex
    assert (
        "[musicf][speech_sc]sidechaincompress=threshold=0.02:ratio=8:attack=5:release=250[ducked_music]"
        in plan.filter_complex
    )
    assert (
        "[speech_mix][ducked_music]amix=inputs=2:duration=first:dropout_transition=0[mixed]"
        in plan.filter_complex
    )
    assert "subtitles=" in plan.filter_complex
    assert "-map [vout] -map [aout]" in command
