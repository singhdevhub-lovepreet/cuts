from __future__ import annotations

# ruff: noqa: E402, I001

import threading
import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from cuts.domain import WordTimestamp
from cuts.edl import AudioTrack, Caption, CaptionTrack, Timeline, TimelineClip
from cuts.graph import Context
from cuts.pipeline import PipelineOptions, PipelineRunResult, RenderedJobResult
from cuts.web.app import create_app
from cuts.web.schemas import JobStatus
from cuts.web.service import JobProgress, WebJobService
from cuts.vlm.models import Platform


def test_web_job_lifecycle_with_mock_executor(tmp_path: Path) -> None:
    ready = threading.Event()
    proceed = threading.Event()

    def executor(
        options: PipelineOptions, *, job_dir: Path, progress: JobProgress | None = None
    ) -> RenderedJobResult:
        if progress is not None:
            progress.set_stage("analysis")
        ready.set()
        assert proceed.wait(timeout=5.0)
        _write_artifacts(job_dir)
        context = Context(source_paths=options.source_paths)
        context.words = [WordTimestamp(clip_id="clip-1", text="hello", start=0.2, end=0.5)]
        return RenderedJobResult(
            run=PipelineRunResult(
                context=context,
                brain_backend="mock-vlm",
                smart_path_enabled=True,
            ),
            edl_path=job_dir / "result.edl.json",
            video_path=job_dir / "result.mp4",
        )

    service = WebJobService(tmp_path, executor=executor)
    app = create_app(work_root=tmp_path, service=service)

    try:
        with TestClient(app) as client:
            response = client.post(
                "/api/jobs",
                files=[
                    ("clips", ("clip-a.webm", b"clip-a", "video/webm")),
                    ("clips", ("clip-b.webm", b"clip-b", "video/webm")),
                    ("music", ("music.mp3", b"music", "audio/mpeg")),
                ],
                data={
                    "vibe": "punchy uplifting travel diary",
                    "platform": Platform.REELS.value,
                    "target_duration": "8",
                    "brain": "true",
                    "whisper_model": "base",
                },
            )
            assert response.status_code == 200
            payload = response.json()
            job_id = payload["job_id"]
            assert payload["version"] == 1

            assert ready.wait(timeout=5.0)
            status = _wait_for_status(client, job_id, JobStatus.RUNNING)
            assert status["stage"] == "analysis"
            assert status["brain_backend"] == "phase0"
            assert (
                client.post(
                    f"/api/jobs/{job_id}/rerender",
                    json={
                        "edits": [
                            {
                                "original_index": 0,
                                "source_in": 0.0,
                                "source_out": 1.0,
                            }
                        ]
                    },
                ).status_code
                == 409
            )

            proceed.set()
            done = _wait_for_status(client, job_id, JobStatus.DONE)
            assert done["version"] == 1
            assert done["brain_backend"] == "mock-vlm"
            assert done["video_url"] == f"/api/jobs/{job_id}/video?v=1"
            assert done["edl_url"] == f"/api/jobs/{job_id}/edl?v=1"
            assert done["status_url"] == f"/api/jobs/{job_id}"
            assert client.get(f"/api/jobs/{job_id}/video").status_code == 200
            assert client.get(f"/api/jobs/{job_id}/edl").status_code == 200
    finally:
        service.close()


def test_web_rerender_updates_artifacts_and_versions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ready = threading.Event()
    proceed = threading.Event()
    rendered: list[Timeline] = []

    def executor(
        options: PipelineOptions, *, job_dir: Path, progress: JobProgress | None = None
    ) -> RenderedJobResult:
        if progress is not None:
            progress.set_stage("analysis")
        ready.set()
        assert proceed.wait(timeout=5.0)
        timeline = Timeline(
            clips=[
                TimelineClip(
                    source_clip_id="clip-1",
                    source_path=Path("clip1.webm"),
                    source_in=0.0,
                    source_out=4.0,
                ),
                TimelineClip(
                    source_clip_id="clip-2",
                    source_path=Path("clip2.webm"),
                    source_in=1.0,
                    source_out=5.0,
                ),
            ],
            caption_tracks=[
                CaptionTrack(
                    captions=[
                        Caption(source_clip_id="clip-1", start=0.5, end=0.8, text="one"),
                        Caption(source_clip_id="clip-2", start=4.5, end=4.8, text="two"),
                    ]
                )
            ],
            audio=AudioTrack(music_path=Path("music.mp3"), ducking=True),
        )
        job_dir.joinpath("result.edl.json").write_text(
            timeline.model_dump_json(indent=2), encoding="utf-8"
        )
        job_dir.joinpath("words.json").write_text(
            """
            [
              {"clip_id": "clip-1", "text": "one", "start": 0.5, "end": 0.8},
              {"clip_id": "clip-2", "text": "two", "start": 1.5, "end": 1.8}
            ]
            """.strip(),
            encoding="utf-8",
        )
        job_dir.joinpath("result.mp4").write_bytes(b"fake-mp4")
        context = Context(source_paths=options.source_paths)
        context.words = [
            WordTimestamp(clip_id="clip-1", text="one", start=0.5, end=0.8),
            WordTimestamp(clip_id="clip-2", text="two", start=1.5, end=1.8),
        ]
        return RenderedJobResult(
            run=PipelineRunResult(
                context=context,
                brain_backend="mock-vlm",
                smart_path_enabled=True,
            ),
            edl_path=job_dir / "result.edl.json",
            video_path=job_dir / "result.mp4",
        )

    def fake_render_timeline(timeline: Timeline, out: Path, work_dir: Path) -> None:
        rendered.append(timeline)
        out.write_bytes(b"rerendered-mp4")

    monkeypatch.setattr("cuts.web.service.render_timeline", fake_render_timeline)

    service = WebJobService(tmp_path, executor=executor)
    app = create_app(work_root=tmp_path, service=service)

    try:
        with TestClient(app) as client:
            response = client.post(
                "/api/jobs",
                files=[("clips", ("clip-a.webm", b"clip-a", "video/webm"))],
                data={
                    "vibe": "punchy",
                    "platform": Platform.REELS.value,
                    "target_duration": "4",
                    "brain": "false",
                    "whisper_model": "base",
                },
            )
            assert response.status_code == 200
            job_id = response.json()["job_id"]
            assert ready.wait(timeout=5.0)
            proceed.set()
            done = _wait_for_status(client, job_id, JobStatus.DONE)
            assert done["version"] == 1
            assert done["video_url"] == f"/api/jobs/{job_id}/video?v=1"

            rerender_response = client.post(
                f"/api/jobs/{job_id}/rerender",
                json={
                    "edits": [
                        {
                            "original_index": 1,
                            "source_in": 1.0,
                            "source_out": 4.0,
                            "transition_kind": "cut",
                            "transition_duration": 0.0,
                        },
                        {
                            "original_index": 0,
                            "source_in": 0.5,
                            "source_out": 3.0,
                            "transition_kind": "fade",
                            "transition_duration": 1.0,
                        },
                    ],
                    "captions": True,
                    "ducking": False,
                },
            )
            assert rerender_response.status_code == 200
            queued = rerender_response.json()
            assert queued["status"] in {JobStatus.QUEUED.value, JobStatus.RUNNING.value}

            rerendered = _wait_for_status(client, job_id, JobStatus.DONE)
            assert rerendered["version"] == 2
            assert rerendered["video_url"] == f"/api/jobs/{job_id}/video?v=2"
            assert rerendered["edl_url"] == f"/api/jobs/{job_id}/edl?v=2"
            assert rendered[-1].clips[0].source_clip_id == "clip-2"
            assert rendered[-1].clips[1].source_clip_id == "clip-1"
            assert rendered[-1].clips[1].crop_path == []
            assert client.get(f"/api/jobs/{job_id}/video").status_code == 200
            assert client.get(f"/api/jobs/{job_id}/edl").status_code == 200
    finally:
        service.close()


def test_unknown_job_404(tmp_path: Path) -> None:
    service = WebJobService(tmp_path, executor=_instant_executor)
    app = create_app(work_root=tmp_path, service=service)
    try:
        with TestClient(app) as client:
            assert client.get("/api/jobs/missing").status_code == 404
            assert client.get("/api/jobs/missing/video").status_code == 404
            assert client.get("/api/jobs/missing/edl").status_code == 404
            assert (
                client.post(
                    "/api/jobs/missing/rerender",
                    json={
                        "edits": [
                            {
                                "original_index": 0,
                                "source_in": 0.0,
                                "source_out": 1.0,
                            }
                        ]
                    },
                ).status_code
                == 404
            )
    finally:
        service.close()


def _instant_executor(
    options: PipelineOptions, *, job_dir: Path, progress: JobProgress | None = None
) -> RenderedJobResult:
    _write_artifacts(job_dir)
    context = Context(source_paths=options.source_paths)
    context.words = [WordTimestamp(clip_id="clip-1", text="hello", start=0.2, end=0.5)]
    return RenderedJobResult(
        run=PipelineRunResult(
            context=context,
            brain_backend="phase0",
            smart_path_enabled=False,
        ),
        edl_path=job_dir / "result.edl.json",
        video_path=job_dir / "result.mp4",
    )


def _write_artifacts(job_dir: Path) -> None:
    timeline = Timeline(
        clips=[
            TimelineClip(
                source_clip_id="clip-1",
                source_path=Path("clip1.webm"),
                source_in=0.0,
                source_out=4.0,
            )
        ],
        caption_tracks=[
            CaptionTrack(
                captions=[Caption(source_clip_id="clip-1", start=0.2, end=0.5, text="hello")]
            )
        ],
        audio=AudioTrack(music_path=Path("music.mp3"), ducking=True),
    )
    job_dir.joinpath("result.edl.json").write_text(
        timeline.model_dump_json(indent=2), encoding="utf-8"
    )
    job_dir.joinpath("words.json").write_text(
        """
        [
          {"clip_id": "clip-1", "text": "hello", "start": 0.2, "end": 0.5}
        ]
        """.strip(),
        encoding="utf-8",
    )
    job_dir.joinpath("result.mp4").write_bytes(b"fake-mp4")


def _wait_for_status(client: TestClient, job_id: str, status: JobStatus) -> dict[str, object]:
    deadline = time.time() + 5.0
    while time.time() < deadline:
        response = client.get(f"/api/jobs/{job_id}")
        assert response.status_code == 200
        payload = response.json()
        if payload["status"] == status.value:
            return payload
        time.sleep(0.1)
    raise AssertionError(f"job {job_id} did not reach status {status.value}")
