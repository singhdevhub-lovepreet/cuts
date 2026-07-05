from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal, Protocol

from cuts.config import load_editor_config
from cuts.domain import EditorConfig, WordTimestamp
from cuts.edl import AudioTrack, CaptionTrack, Timeline, TimelineClip, Transition
from cuts.graph import Context, Pipeline
from cuts.nodes.assemble import AssembleNode, build_captions
from cuts.nodes.beats import BeatsNode
from cuts.nodes.ingest import IngestNode
from cuts.nodes.motion import MotionNode
from cuts.nodes.reframe import ReframeNode
from cuts.nodes.sequence import SequencerNode
from cuts.nodes.shots import ShotsNode
from cuts.nodes.silence import SilenceNode
from cuts.nodes.transcribe import TranscribeNode
from cuts.nodes.vibe import VibeTaggerNode, build_default_vlm_client
from cuts.render import render_timeline
from cuts.vlm.client import VLMClient
from cuts.vlm.models import Platform


class ProgressReporter(Protocol):
    def set_stage(self, stage: str) -> None:
        raise NotImplementedError


@dataclass(slots=True)
class PipelineOptions:
    source_paths: tuple[Path, ...]
    music_path: Path | None = None
    target_duration: float | None = None
    vibe_prompt: str = ""
    platform: Platform = Platform.REELS
    brain: bool = False
    whisper_model: str = "base"
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"
    config: EditorConfig = field(default_factory=EditorConfig)


@dataclass(slots=True)
class PipelineRunResult:
    context: Context
    brain_backend: str
    smart_path_enabled: bool


@dataclass(slots=True)
class RenderedJobResult:
    run: PipelineRunResult
    edl_path: Path
    video_path: Path


@dataclass(slots=True, frozen=True)
class RerenderClipEdit:
    original_index: int
    source_in: float
    source_out: float
    transition_kind: Literal["cut", "fade"] = "cut"
    transition_duration: float = 0.0


def build_context(options: PipelineOptions, output_path: Path | None = None) -> Context:
    return Context(
        source_paths=options.source_paths,
        music_path=options.music_path,
        target_duration=options.target_duration,
        vibe_prompt=options.vibe_prompt,
        platform=options.platform,
        whisper_model=options.whisper_model,
        whisper_device=options.whisper_device,
        whisper_compute_type=options.whisper_compute_type,
        output_path=output_path,
        config=options.config,
    )


def build_pipeline(
    options: PipelineOptions,
    client: VLMClient | None = None,
) -> tuple[Pipeline, str]:
    nodes = [
        IngestNode(),
        ShotsNode(),
        MotionNode(),
        TranscribeNode(model_size=options.whisper_model),
        SilenceNode(),
        BeatsNode(),
    ]
    smart_requested = bool(options.brain or options.vibe_prompt.strip())
    brain_backend = "phase0"
    if smart_requested:
        vlm_client = client or build_default_vlm_client()
        brain_backend = vlm_client.model_name
        nodes.extend(
            [
                VibeTaggerNode(client=vlm_client),
                SequencerNode(client=vlm_client),
            ]
        )
    nodes.extend([AssembleNode(), ReframeNode()])
    return Pipeline(nodes), brain_backend


def run_pipeline(
    options: PipelineOptions,
    *,
    progress: ProgressReporter | None = None,
    client: VLMClient | None = None,
) -> PipelineRunResult:
    context = build_context(options)
    pipeline, brain_backend = build_pipeline(options, client=client)
    for node in pipeline.ordered_nodes:
        if progress is not None:
            progress.set_stage(node.name)
        context = node.run(context)
    return PipelineRunResult(
        context=context,
        brain_backend=brain_backend,
        smart_path_enabled=brain_backend != "phase0",
    )


def render_job(
    options: PipelineOptions,
    *,
    job_dir: Path,
    progress: ProgressReporter | None = None,
    client: VLMClient | None = None,
) -> RenderedJobResult:
    run = run_pipeline(options, progress=progress, client=client)
    if run.context.timeline is None:
        raise RuntimeError("analysis pipeline did not produce a timeline")
    edl_path = job_dir / "result.edl.json"
    video_path = job_dir / "result.mp4"
    edl_path.write_text(run.context.timeline.model_dump_json(indent=2), encoding="utf-8")
    (job_dir / "words.json").write_text(
        json.dumps([asdict(word) for word in run.context.words], indent=2),
        encoding="utf-8",
    )
    if progress is not None:
        progress.set_stage("render")
    render_timeline(run.context.timeline, video_path, job_dir / "render-work")
    return RenderedJobResult(run=run, edl_path=edl_path, video_path=video_path)


def load_pipeline_config(config_path: Path | None) -> EditorConfig:
    return load_editor_config(config_path)


def load_words(words_path: Path) -> list[WordTimestamp]:
    raw_words = json.loads(words_path.read_text(encoding="utf-8"))
    return [WordTimestamp(**item) for item in raw_words]


def rebuild_timeline_for_rerender(
    original: Timeline,
    words: Sequence[WordTimestamp],
    edits: Sequence[RerenderClipEdit],
    *,
    captions: bool = True,
    ducking_override: bool | None = None,
) -> Timeline:
    new_clips: list[TimelineClip] = []
    previous_length: float | None = None
    for edit in edits:
        try:
            original_clip = original.clips[edit.original_index]
        except IndexError as exc:
            raise ValueError(f"invalid original clip index: {edit.original_index}") from exc
        source_length = edit.source_out - edit.source_in
        transition = _clamp_transition(
            edit.transition_kind,
            edit.transition_duration,
            previous_length,
            source_length,
        )
        new_clips.append(
            TimelineClip(
                source_clip_id=original_clip.source_clip_id,
                source_path=original_clip.source_path,
                source_in=edit.source_in,
                source_out=edit.source_out,
                transition=transition,
                has_audio=original_clip.has_audio,
                crop_aspect=original_clip.crop_aspect,
                crop_path=list(original_clip.crop_path)
                if original_clip.source_in == edit.source_in
                and original_clip.source_out == edit.source_out
                else [],
            )
        )
        previous_length = source_length

    caption_tracks = (
        [CaptionTrack(captions=build_captions(new_clips, list(words)))] if captions else []
    )
    return Timeline(
        target_width=original.target_width,
        target_height=original.target_height,
        target_fps=original.target_fps,
        duration=_timeline_duration(new_clips),
        clips=new_clips,
        caption_tracks=caption_tracks,
        overlay_tracks=list(original.overlay_tracks),
        audio=AudioTrack(
            music_path=original.audio.music_path,
            ducking=original.audio.ducking if ducking_override is None else ducking_override,
            normalize_lufs=original.audio.normalize_lufs,
        ),
    )


def _clamp_transition(
    kind: str,
    duration: float,
    previous_length: float | None,
    current_length: float,
) -> Transition:
    if kind.strip().lower() != "fade":
        return Transition()
    if previous_length is None:
        return Transition()
    clamped = min(duration, 0.5 * min(previous_length, current_length))
    if clamped <= 0.02:
        return Transition()
    return Transition(kind="fade", duration=clamped)


def _timeline_duration(clips: Sequence[TimelineClip]) -> float:
    output_offset = 0.0
    for index, clip in enumerate(clips):
        clip_start = output_offset if index == 0 else output_offset - clip.transition.duration
        output_offset = clip_start + (clip.source_out - clip.source_in)
    return output_offset
