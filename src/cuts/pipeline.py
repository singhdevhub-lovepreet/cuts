from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from cuts.config import load_editor_config
from cuts.domain import EditorConfig
from cuts.graph import Context, Pipeline
from cuts.nodes.assemble import AssembleNode
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
    if progress is not None:
        progress.set_stage("render")
    render_timeline(run.context.timeline, video_path, job_dir / "render-work")
    return RenderedJobResult(run=run, edl_path=edl_path, video_path=video_path)


def load_pipeline_config(config_path: Path | None) -> EditorConfig:
    return load_editor_config(config_path)
