from __future__ import annotations

import subprocess
from dataclasses import dataclass
from hashlib import sha1
from pathlib import Path
from shlex import join as shell_join

from cuts.edl import Caption, Timeline


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
        concat_inputs: list[str] = []
        for index, clip in enumerate(timeline.clips):
            video_label = f"v{index}"
            audio_label = f"a{index}"
            # TODO: replace the center crop with subject-aware reframing.
            parts.append(
                f"[{index}:v]trim=start={self._format_number(clip.source_in)}:end={self._format_number(clip.source_out)},"
                f"setpts=PTS-STARTPTS,crop='if(gte(iw/ih,9/16),ih*9/16,iw)':"
                f"'if(gte(iw/ih,9/16),ih,iw*16/9)':(iw-ow)/2:(ih-oh)/2,"
                f"scale={timeline.target_width}:{timeline.target_height}:flags=lanczos,setsar=1[{video_label}]"
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
            concat_inputs.extend([f"[{video_label}]", f"[{audio_label}]"])
        if concat_inputs:
            parts.append(
                f"{''.join(concat_inputs)}concat=n={len(timeline.clips)}:v=1:a=1[vcat][acat]"
            )
        else:
            parts.append("color=c=black:s=1080x1920:d=1[vcat]")
            parts.append("anullsrc=r=48000:cl=stereo:d=1[acat]")
        if subtitles_path is not None:
            parts.append(f"[vcat]subtitles={self._escape_filter_path(subtitles_path)}[vout]")
        else:
            parts.append("[vcat]setpts=PTS-STARTPTS[vout]")
        if timeline.audio.music_path is not None:
            parts.append(
                f"[{len(timeline.clips)}:a]atrim=start=0:end={self._format_number(self._timeline_duration(timeline))},"
                f"asetpts=PTS-STARTPTS,aresample=48000,volume=1[music]"
            )
            if timeline.audio.ducking:
                parts.append(
                    "[acat][music]sidechaincompress=threshold=0.02:ratio=8:attack=5:release=250[mixed]"
                )
            else:
                parts.append(
                    "[acat][music]amix=inputs=2:duration=first:dropout_transition=0[mixed]"
                )
            parts.append(
                f"[mixed]loudnorm=I={self._format_number(timeline.audio.normalize_lufs)}:TP=-1.5:LRA=11[aout]"
            )
        else:
            parts.append(
                f"[acat]loudnorm=I={self._format_number(timeline.audio.normalize_lufs)}:TP=-1.5:LRA=11[aout]"
            )
        return ";".join(parts)

    def _write_subtitles(self, timeline: Timeline, work_dir: Path) -> Path | None:
        captions: list[Caption] = []
        for track in timeline.caption_tracks:
            captions.extend(track.captions)
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
        for caption in captions:
            dialogue = (
                f"Dialogue: 0,{self._ass_time(caption.start)},"
                f"{self._ass_time(caption.end)},Default,,0,0,0,,"
                f"{self._escape_ass_text(caption.text)}"
            )
            lines.append(dialogue)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

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
