from __future__ import annotations

import subprocess
from dataclasses import dataclass
from hashlib import sha1
from pathlib import Path
from shlex import join as shell_join

from cuts.edl import Caption, CropKeyframe, Timeline, TimelineClip


@dataclass(slots=True)
class RenderPlan:
    command: list[str]
    filter_complex: str
    subtitles_path: Path | None


class Renderer:
    def build_plan(
        self, timeline: Timeline, output_path: Path, work_dir: Path | None = None
    ) -> RenderPlan:
        temp_dir = self._work_dir_for(output_path, work_dir)
        subtitles_path = self._write_subtitles(timeline, temp_dir)
        filter_complex = self._build_filter_complex(timeline, subtitles_path)
        command = self._build_command(timeline, output_path, filter_complex)
        return RenderPlan(
            command=command, filter_complex=filter_complex, subtitles_path=subtitles_path
        )

    def render(self, timeline: Timeline, output_path: Path, work_dir: Path | None = None) -> Path:
        plan = self.build_plan(timeline, output_path, work_dir)
        print(shell_join(plan.command))
        subprocess.run(plan.command, check=True)
        return output_path

    def _build_command(
        self, timeline: Timeline, output_path: Path, filter_complex: str
    ) -> list[str]:
        command = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
        for clip in timeline.clips:
            command.extend(["-i", str(clip.source_path)])
        if timeline.audio.music_path is not None:
            command.extend(["-stream_loop", "-1", "-i", str(timeline.audio.music_path)])
        command.extend(["-filter_complex", filter_complex, "-map", "[vout]", "-map", "[aout]"])
        command.extend(
            [
                "-r",
                self._format_number(timeline.target_fps),
                "-s",
                f"{timeline.target_width}x{timeline.target_height}",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-movflags",
                "+faststart",
                str(output_path),
            ]
        )
        return command

    def _build_filter_complex(self, timeline: Timeline, subtitles_path: Path | None) -> str:
        parts: list[str] = []
        video_labels: list[str] = []
        audio_labels: list[str] = []
        for index, clip in enumerate(timeline.clips):
            video_label = f"v{index}"
            audio_label = f"a{index}"
            crop_w_expr, crop_h_expr = self._crop_dimensions_expression(clip.crop_aspect)
            crop_x_expr, crop_y_expr = self._crop_position_expression(clip)
            parts.append(
                f"[{index}:v]trim=start={self._format_number(clip.source_in)}:end={self._format_number(clip.source_out)},"
                f"setpts=PTS-STARTPTS,crop='{crop_w_expr}':'{crop_h_expr}':{crop_x_expr}:{crop_y_expr},"
                f"scale={timeline.target_width}:{timeline.target_height}:flags=lanczos,fps={self._format_number(timeline.target_fps)},setsar=1[{video_label}]"
            )
            if clip.has_audio:
                parts.append(
                    f"[{index}:a]atrim=start={self._format_number(clip.source_in)}:end={self._format_number(clip.source_out)},"
                    f"asetpts=PTS-STARTPTS,aresample=48000,volume=1[{audio_label}]"
                )
            else:
                duration = clip.source_out - clip.source_in
                parts.append(
                    f"anullsrc=r=48000:cl=stereo,atrim=duration={self._format_number(duration)},"
                    f"asetpts=PTS-STARTPTS[{audio_label}]"
                )
            video_labels.append(f"[{video_label}]")
            audio_labels.append(f"[{audio_label}]")
        if timeline.clips:
            if self._has_transitions(timeline):
                vcat, acat = self._build_transition_chain(parts, timeline)
            else:
                concat_inputs = "".join(
                    label for pair in zip(video_labels, audio_labels, strict=True) for label in pair
                )
                parts.append(f"{concat_inputs}concat=n={len(timeline.clips)}:v=1:a=1[vcat][acat]")
                vcat, acat = "vcat", "acat"
        else:
            parts.append("color=c=black:s=1080x1920:d=1[vcat]")
            parts.append("anullsrc=r=48000:cl=stereo:d=1[acat]")
            vcat, acat = "vcat", "acat"
        if subtitles_path is not None:
            parts.append(f"[{vcat}]subtitles={self._escape_filter_path(subtitles_path)}[vout]")
        else:
            parts.append(f"[{vcat}]setpts=PTS-STARTPTS[vout]")
        if timeline.audio.music_path is not None:
            parts.append(
                f"[{len(timeline.clips)}:a]atrim=start=0:end={self._format_number(self._timeline_duration(timeline))},"
                f"asetpts=PTS-STARTPTS,aresample=48000,volume=1[music]"
            )
            parts.append(f"[{acat}]aformat=sample_fmts=fltp:channel_layouts=stereo[acatf]")
            parts.append("[music]aformat=sample_fmts=fltp:channel_layouts=stereo[musicf]")
            if timeline.audio.ducking:
                parts.append("[acatf]asplit=2[speech_mix][speech_sc]")
                parts.append(
                    "[musicf][speech_sc]sidechaincompress=threshold=0.02:ratio=8:attack=5:release=250[ducked_music]"
                )
                parts.append(
                    "[speech_mix][ducked_music]amix=inputs=2:duration=first:dropout_transition=0[mixed]"
                )
            else:
                parts.append(
                    "[acatf][musicf]amix=inputs=2:duration=first:dropout_transition=0[mixed]"
                )
            parts.append(
                f"[mixed]loudnorm=I={self._format_number(timeline.audio.normalize_lufs)}:TP=-1.5:LRA=11[aout]"
            )
        else:
            parts.append(f"[{acat}]aformat=sample_fmts=fltp:channel_layouts=stereo[acatf]")
            parts.append(
                f"[acatf]loudnorm=I={self._format_number(timeline.audio.normalize_lufs)}:TP=-1.5:LRA=11[aout]"
            )
        return ";".join(parts)

    def _build_transition_chain(self, parts: list[str], timeline: Timeline) -> tuple[str, str]:
        video_label = "v0"
        audio_label = "a0"
        current_duration = self._clip_duration(timeline.clips[0])
        for index, clip in enumerate(timeline.clips[1:], start=1):
            clip_duration = self._clip_duration(clip)
            transition_duration = min(
                max(0.0, clip.transition.duration),
                current_duration,
                clip_duration,
            )
            if transition_duration > 0.0:
                transition_name = self._transition_filter_name(clip.transition.kind)
                offset = max(0.0, current_duration - transition_duration)
                next_video_label = f"vxf{index}"
                next_audio_label = f"axf{index}"
                parts.append(
                    f"[{video_label}][v{index}]xfade=transition={transition_name}:duration={self._format_number(transition_duration)}:offset={self._format_number(offset)}[{next_video_label}]"
                )
                parts.append(
                    f"[{audio_label}][a{index}]acrossfade=d={self._format_number(transition_duration)}[{next_audio_label}]"
                )
                video_label = next_video_label
                audio_label = next_audio_label
                current_duration = current_duration + clip_duration - transition_duration
                continue
            next_video_label = f"vcat{index}"
            next_audio_label = f"acat{index}"
            parts.append(f"[{video_label}][v{index}]concat=n=2:v=1:a=0[{next_video_label}]")
            parts.append(f"[{audio_label}][a{index}]concat=n=2:v=0:a=1[{next_audio_label}]")
            video_label = next_video_label
            audio_label = next_audio_label
            current_duration = current_duration + clip_duration
        return video_label, audio_label

    def _has_transitions(self, timeline: Timeline) -> bool:
        return any(clip.transition.duration > 0.0 for clip in timeline.clips[1:])

    def _transition_filter_name(self, kind: str) -> str:
        normalized = kind.strip().lower()
        if normalized in {"crossfade", "fade"}:
            return "fade"
        if normalized == "dip_to_black":
            return "fadeblack"
        return "fade"

    def _clip_duration(self, clip: TimelineClip) -> float:
        return clip.source_out - clip.source_in

    def _crop_dimensions_expression(self, crop_aspect: float) -> tuple[str, str]:
        aspect = self._format_number(crop_aspect)
        return (
            f"if(gte(iw/ih,{aspect}),ih*{aspect},iw)",
            f"if(gte(iw/ih,{aspect}),ih,iw/{aspect})",
        )

    def _crop_position_expression(self, clip: TimelineClip) -> tuple[str, str]:
        if not clip.crop_path:
            return "'(iw-ow)/2'", "'(ih-oh)/2'"
        x_expr = self._normalized_position_expression(clip.crop_path, "center_x")
        y_expr = self._normalized_position_expression(clip.crop_path, "center_y")
        return (
            f"'min(max(({x_expr})*iw-ow/2,0),iw-ow)'",
            f"'min(max(({y_expr})*ih-oh/2,0),ih-oh)'",
        )

    def _normalized_position_expression(self, keyframes: list[CropKeyframe], attribute: str) -> str:
        ordered = sorted(keyframes, key=lambda item: (item.t, item.center_x, item.center_y))
        if len(ordered) == 1:
            return self._format_number(self._keyframe_value(ordered[0], attribute))
        tail = self._format_number(self._keyframe_value(ordered[-1], attribute))
        expression = tail
        for left, right in reversed(list(zip(ordered[:-1], ordered[1:], strict=True))):
            left_value = self._format_number(self._keyframe_value(left, attribute))
            right_value = self._format_number(self._keyframe_value(right, attribute))
            duration = max(right.t - left.t, 1e-6)
            interpolated = (
                f"({left_value})+(({right_value})-({left_value}))*((t-{self._format_number(left.t)})/"
                f"{self._format_number(duration)})"
            )
            expression = f"if(lte(t,{self._format_number(right.t)}),{interpolated},{expression})"
        return expression

    def _keyframe_value(self, keyframe: CropKeyframe, attribute: str) -> float:
        if attribute == "center_x":
            return keyframe.center_x
        if attribute == "center_y":
            return keyframe.center_y
        raise ValueError(f"unknown crop attribute: {attribute}")

    def _write_subtitles(self, timeline: Timeline, work_dir: Path) -> Path | None:
        captions: list[Caption] = []
        for track in timeline.caption_tracks:
            for caption in track.captions:
                captions.append(caption)
        if not captions:
            return None
        path = work_dir / "captions.ass"
        lines = [
            "[Script Info]",
            "ScriptType: v4.00+",
            "PlayResX: 1080",
            "PlayResY: 1920",
            "WrapStyle: 2",
            "ScaledBorderAndShadow: yes",
            "",
            "[V4+ Styles]",
            (
                "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
                "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
                "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
                "Alignment, MarginL, MarginR, MarginV, Encoding"
            ),
            (
                "Style: Default,Arial,68,&H00FFFFFF,&H000000FF,&H00000000,"
                "&H64000000,1,0,0,0,100,100,0,0,1,3,1,2,40,40,60,1"
            ),
            "",
            "[Events]",
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
        ]
        for track in timeline.caption_tracks:
            for caption in track.captions:
                text = self._caption_text(caption.text, track.animation)
                dialogue = (
                    f"Dialogue: 0,{self._ass_time(caption.start)},"
                    f"{self._ass_time(caption.end)},Default,,0,0,0,,"
                    f"{text}"
                )
                lines.append(dialogue)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    def _caption_text(self, text: str, animation: str) -> str:
        escaped_text = self._escape_ass_text(text)
        if animation != "pop":
            return escaped_text
        return r"{\fad(50,0)\t(0,150,\fscx120\fscy120)\t(0,150,\fscx100\fscy100)}" + escaped_text

    def _timeline_duration(self, timeline: Timeline) -> float:
        if timeline.duration is not None:
            return timeline.duration
        return sum(clip.source_out - clip.source_in for clip in timeline.clips)

    def _work_dir_for(self, output_path: Path, work_dir: Path | None) -> Path:
        if work_dir is not None:
            target = work_dir
        else:
            key = sha1(str(output_path).encode("utf-8")).hexdigest()[:12]
            target = Path("/tmp") / "cuts-render" / key
        target.mkdir(parents=True, exist_ok=True)
        return target

    def _format_number(self, value: float) -> str:
        text = f"{value:.6f}"
        return text.rstrip("0").rstrip(".") if "." in text else text

    def _ass_time(self, value: float) -> str:
        total_centiseconds = int(round(value * 100))
        centiseconds = total_centiseconds % 100
        total_seconds = total_centiseconds // 100
        seconds = total_seconds % 60
        total_minutes = total_seconds // 60
        minutes = total_minutes % 60
        hours = total_minutes // 60
        return f"{hours:d}:{minutes:02d}:{seconds:02d}.{centiseconds:02d}"

    def _escape_ass_text(self, text: str) -> str:
        return text.replace("\\", r"\\").replace("{", r"\{").replace("}", r"\}")

    def _escape_filter_path(self, path: Path) -> str:
        return str(path).replace("\\", r"\\\\").replace(":", r"\:")


def build_ffmpeg_plan(
    timeline: Timeline, output_path: Path, work_dir: Path | None = None
) -> RenderPlan:
    return Renderer().build_plan(timeline, output_path, work_dir)


def render_timeline(timeline: Timeline, output_path: Path, work_dir: Path | None = None) -> Path:
    return Renderer().render(timeline, output_path, work_dir)
