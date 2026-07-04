from __future__ import annotations

import argparse
from pathlib import Path

from cuts.config import load_editor_config
from cuts.edl import Timeline
from cuts.graph import Context, Pipeline
from cuts.nodes.assemble import AssembleNode
from cuts.nodes.beats import BeatsNode
from cuts.nodes.ingest import IngestNode
from cuts.nodes.motion import MotionNode
from cuts.nodes.shots import ShotsNode
from cuts.nodes.silence import SilenceNode
from cuts.nodes.transcribe import TranscribeNode
from cuts.render import render_timeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cuts", description="Deterministic short-video editor")
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze = subparsers.add_parser("analyze", help="Analyze clips and write an EDL JSON file")
    analyze.add_argument("clips", nargs="+", type=Path)
    analyze.add_argument("--music", type=Path)
    analyze.add_argument("--target-duration", type=float)
    analyze.add_argument("--whisper-model", default="base")
    analyze.add_argument("--output", type=Path, required=True)
    analyze.add_argument("--config", type=Path)

    render = subparsers.add_parser("render", help="Render an EDL JSON file to MP4")
    render.add_argument("edl", type=Path)
    render.add_argument("--output", type=Path, required=True)
    render.add_argument("--work-dir", type=Path)

    run = subparsers.add_parser("run", help="Analyze clips and render the final MP4")
    run.add_argument("clips", nargs="+", type=Path)
    run.add_argument("--music", type=Path)
    run.add_argument("--target-duration", type=float)
    run.add_argument("--whisper-model", default="base")
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--config", type=Path)
    run.add_argument("--work-dir", type=Path)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "analyze":
        context = Context(
            source_paths=tuple(args.clips),
            music_path=args.music,
            target_duration=args.target_duration,
            whisper_model=args.whisper_model,
            config=load_editor_config(args.config),
        )
        pipeline = Pipeline(
            [
                IngestNode(),
                ShotsNode(),
                MotionNode(),
                TranscribeNode(model_size=args.whisper_model),
                SilenceNode(),
                BeatsNode(),
                AssembleNode(),
            ]
        )
        context = pipeline.run(context)
        if context.timeline is None:
            raise RuntimeError("analysis pipeline did not produce a timeline")
        args.output.write_text(context.timeline.model_dump_json(indent=2), encoding="utf-8")
        return 0

    if args.command == "render":
        timeline = Timeline.model_validate_json(args.edl.read_text(encoding="utf-8"))
        render_timeline(timeline, args.output, args.work_dir)
        return 0

    if args.command == "run":
        context = Context(
            source_paths=tuple(args.clips),
            music_path=args.music,
            target_duration=args.target_duration,
            whisper_model=args.whisper_model,
            config=load_editor_config(args.config),
            output_path=args.output,
        )
        pipeline = Pipeline(
            [
                IngestNode(),
                ShotsNode(),
                MotionNode(),
                TranscribeNode(model_size=args.whisper_model),
                SilenceNode(),
                BeatsNode(),
                AssembleNode(),
            ]
        )
        context = pipeline.run(context)
        if context.timeline is None:
            raise RuntimeError("analysis pipeline did not produce a timeline")
        render_timeline(context.timeline, args.output, args.work_dir)
        return 0

    raise AssertionError(f"unexpected command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
