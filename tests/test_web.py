from __future__ import annotations

# ruff: noqa: E402, I001

import threading
import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

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
        edl_path = job_dir / "result.edl.json"
        video_path = job_dir / "result.mp4"
        edl_path.write_text("{}", encoding="utf-8")
        video_path.write_bytes(b"fake-mp4")
        context = Context(source_paths=options.source_paths)
        return RenderedJobResult(
            run=PipelineRunResult(
                context=context,
                brain_backend="mock-vlm",
                smart_path_enabled=True,
            ),
            edl_path=edl_path,
            video_path=video_path,
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

            assert ready.wait(timeout=5.0)
            status = _wait_for_status(client, job_id, JobStatus.RUNNING)
            assert status["stage"] == "analysis"
            assert status["brain_backend"] == "phase0"

            proceed.set()
            done = _wait_for_status(client, job_id, JobStatus.DONE)
            assert done["brain_backend"] == "mock-vlm"
            assert done["video_url"] == f"/api/jobs/{job_id}/video"
            assert done["edl_url"] == f"/api/jobs/{job_id}/edl"
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
    finally:
        service.close()


def _instant_executor(
    options: PipelineOptions, *, job_dir: Path, progress: JobProgress | None = None
) -> RenderedJobResult:
    edl_path = job_dir / "result.edl.json"
    video_path = job_dir / "result.mp4"
    edl_path.write_text("{}", encoding="utf-8")
    video_path.write_bytes(b"fake-mp4")
    context = Context(source_paths=options.source_paths)
    return RenderedJobResult(
        run=PipelineRunResult(
            context=context,
            brain_backend="phase0",
            smart_path_enabled=False,
        ),
        edl_path=edl_path,
        video_path=video_path,
    )


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
