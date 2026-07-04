from __future__ import annotations

import argparse
from pathlib import Path

from cuts.edl import Timeline
from cuts.pipeline import PipelineOptions, load_pipeline_config, run_pipeline
from cuts.render import render_timeline
from cuts.vlm.models import Platform


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cuts", description="Deterministic short-video editor")
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze = subparsers.add_parser("analyze", help="Analyze clips and write an EDL JSON file")
    analyze.add_argument("clips", nargs="+", type=Path)
    analyze.add_argument("--music", type=Path)
    analyze.add_argument("--target-duration", type=float)
    analyze.add_argument("--vibe", default="")
    analyze.add_argument(
        "--platform",
        choices=[platform.value for platform in Platform],
        default=Platform.REELS.value,
    )
    analyze.add_argument("--brain", action="store_true")
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
    run.add_argument("--vibe", default="")
    run.add_argument(
        "--platform",
        choices=[platform.value for platform in Platform],
        default=Platform.REELS.value,
    )
    run.add_argument("--brain", action="store_true")
    run.add_argument("--whisper-model", default="base")
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--config", type=Path)
    run.add_argument("--work-dir", type=Path)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "analyze":
        options = PipelineOptions(
            source_paths=tuple(args.clips),
            music_path=args.music,
            target_duration=args.target_duration,
            vibe_prompt=args.vibe,
            platform=Platform(args.platform),
            brain=args.brain,
            whisper_model=args.whisper_model,
            config=load_pipeline_config(args.config),
        )
        result = run_pipeline(options)
        if result.context.timeline is None:
            raise RuntimeError("analysis pipeline did not produce a timeline")
        args.output.write_text(
            result.context.timeline.model_dump_json(indent=2),
            encoding="utf-8",
        )
        return 0

    if args.command == "render":
        timeline = Timeline.model_validate_json(args.edl.read_text(encoding="utf-8"))
        render_timeline(timeline, args.output, args.work_dir)
        return 0

    if args.command == "run":
        options = PipelineOptions(
            source_paths=tuple(args.clips),
            music_path=args.music,
            target_duration=args.target_duration,
            vibe_prompt=args.vibe,
            platform=Platform(args.platform),
            brain=args.brain,
            whisper_model=args.whisper_model,
            config=load_pipeline_config(args.config),
        )
        result = run_pipeline(options)
        if result.context.timeline is None:
            raise RuntimeError("analysis pipeline did not produce a timeline")
        render_timeline(result.context.timeline, args.output, args.work_dir)
        return 0

    raise AssertionError(f"unexpected command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
