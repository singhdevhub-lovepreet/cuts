from __future__ import annotations

import queue
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Protocol
from uuid import uuid4

from cuts.domain import EditorConfig
from cuts.edl import Timeline
from cuts.pipeline import (
    PipelineOptions,
    RenderedJobResult,
    RerenderClipEdit,
    load_pipeline_config,
    load_words,
    rebuild_timeline_for_rerender,
    render_job,
)
from cuts.render import render_timeline
from cuts.vlm.models import Platform
from cuts.web.schemas import JobStatus, JobStatusResponse, RerenderRequest


class JobProgress(Protocol):
    def set_stage(self, stage: str) -> None:
        raise NotImplementedError


class JobExecutor(Protocol):
    def __call__(
        self, options: PipelineOptions, *, job_dir: Path, progress: JobProgress | None = None
    ) -> RenderedJobResult:
        raise NotImplementedError


@dataclass(slots=True)
class WebJobRequest:
    source_paths: tuple[Path, ...]
    music_path: Path | None
    target_duration: float | None
    vibe_prompt: str
    platform: Platform
    brain: bool
    whisper_model: str
    config_path: Path | None = None


@dataclass(slots=True, frozen=True)
class WebJobTask:
    kind: Literal["pipeline", "rerender"]
    job_id: str


@dataclass(slots=True)
class JobRecord:
    job_id: str
    request: WebJobRequest
    job_dir: Path
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    status: JobStatus = JobStatus.QUEUED
    stage: str | None = None
    version: int = 1
    brain_backend: str = "phase0"
    warnings: list[str] = field(default_factory=list)
    error: str | None = None
    edl_path: Path | None = None
    video_path: Path | None = None
    pending_rerender: RerenderRequest | None = None


class _RecordProgress:
    def __init__(self, service: WebJobService, job_id: str) -> None:
        self._service = service
        self._job_id = job_id

    def set_stage(self, stage: str) -> None:
        self._service.update_stage(self._job_id, stage)


class WebJobService:
    def __init__(
        self,
        work_root: Path,
        executor: JobExecutor | None = None,
    ) -> None:
        self._work_root = work_root
        self._executor = executor or _default_executor
        self._jobs: dict[str, JobRecord] = {}
        self._lock = threading.Lock()
        self._queue: queue.Queue[WebJobTask | None] = queue.Queue()
        self._stop = threading.Event()
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

    def close(self) -> None:
        self._stop.set()
        self._queue.put(None)
        self._worker.join(timeout=5.0)

    def submit(
        self,
        request: WebJobRequest,
        *,
        job_dir: Path | None = None,
        job_id: str | None = None,
    ) -> JobRecord:
        job_id = job_id or uuid4().hex
        job_dir = job_dir or (self._work_root / job_id)
        job_dir.mkdir(parents=True, exist_ok=True)
        record = JobRecord(job_id=job_id, request=request, job_dir=job_dir)
        with self._lock:
            self._jobs[job_id] = record
        self._queue.put(WebJobTask(kind="pipeline", job_id=job_id))
        return record

    def enqueue_rerender(self, job_id: str, request: RerenderRequest) -> JobRecord:
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None:
                raise KeyError(job_id)
            record.pending_rerender = request
            record.status = JobStatus.QUEUED
            record.stage = "queued"
            record.error = None
        self._queue.put(WebJobTask(kind="rerender", job_id=job_id))
        return record

    def get(self, job_id: str) -> JobRecord | None:
        with self._lock:
            return self._jobs.get(job_id)

    def update_stage(self, job_id: str, stage: str) -> None:
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None:
                return
            record.stage = stage
            record.status = JobStatus.RUNNING

    def mark_done(
        self,
        job_id: str,
        *,
        brain_backend: str,
        warnings: list[str],
        edl_path: Path,
        video_path: Path,
        bump_version: bool = False,
    ) -> None:
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None:
                return
            record.status = JobStatus.DONE
            record.stage = "done"
            record.brain_backend = brain_backend
            record.warnings = list(warnings)
            record.edl_path = edl_path
            record.video_path = video_path
            record.error = None
            record.pending_rerender = None
            if bump_version:
                record.version += 1

    def mark_error(self, job_id: str, error: str) -> None:
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None:
                return
            record.status = JobStatus.ERROR
            record.stage = "error"
            record.error = error

    def snapshot(self, job_id: str) -> JobStatusResponse | None:
        record = self.get(job_id)
        if record is None:
            return None
        return self._snapshot_record(record)

    def _snapshot_record(self, record: JobRecord) -> JobStatusResponse:
        video_url = (
            f"/api/jobs/{record.job_id}/video?v={record.version}"
            if record.video_path is not None
            else None
        )
        edl_url = (
            f"/api/jobs/{record.job_id}/edl?v={record.version}"
            if record.edl_path is not None
            else None
        )
        return JobStatusResponse(
            job_id=record.job_id,
            status=record.status,
            version=record.version,
            status_url=f"/api/jobs/{record.job_id}",
            stage=record.stage,
            brain_backend=record.brain_backend,
            warnings=list(record.warnings),
            error=record.error,
            video_url=video_url,
            edl_url=edl_url,
            video_path=record.video_path,
            edl_path=record.edl_path,
        )

    def _worker_loop(self) -> None:
        while not self._stop.is_set():
            try:
                task = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if task is None:
                continue
            record = self.get(task.job_id)
            if record is None:
                continue
            try:
                if task.kind == "pipeline":
                    self.update_stage(task.job_id, "queued")
                    progress = _RecordProgress(self, task.job_id)
                    options = self._options_from_record(record)
                    result = self._executor(options, job_dir=record.job_dir, progress=progress)
                    self.mark_done(
                        task.job_id,
                        brain_backend=result.run.brain_backend,
                        warnings=list(result.run.context.warnings),
                        edl_path=result.edl_path,
                        video_path=result.video_path,
                    )
                else:
                    self._rerender(task.job_id, record)
            except Exception as exc:  # pragma: no cover - exercised in integration only
                self.mark_error(task.job_id, str(exc))

    def _rerender(self, job_id: str, record: JobRecord) -> None:
        request = record.pending_rerender
        if request is None:
            raise RuntimeError("rerender request missing")
        edl_path = record.job_dir / "result.edl.json"
        words_path = record.job_dir / "words.json"
        if not edl_path.exists():
            raise FileNotFoundError(edl_path)
        if not words_path.exists():
            raise FileNotFoundError(words_path)
        self.update_stage(job_id, "rerender")
        original = Timeline.model_validate_json(edl_path.read_text(encoding="utf-8"))
        words = load_words(words_path)
        timeline = rebuild_timeline_for_rerender(
            original,
            words,
            [
                RerenderClipEdit(
                    original_index=edit.original_index,
                    source_in=edit.source_in,
                    source_out=edit.source_out,
                    transition_kind=edit.transition_kind,
                    transition_duration=edit.transition_duration,
                )
                for edit in request.edits
            ],
            captions=request.captions,
            ducking_override=request.ducking,
        )
        edl_path.write_text(timeline.model_dump_json(indent=2), encoding="utf-8")
        self.update_stage(job_id, "render")
        render_timeline(timeline, record.job_dir / "result.mp4", record.job_dir / "render-work")
        self.mark_done(
            job_id,
            brain_backend=record.brain_backend,
            warnings=list(record.warnings),
            edl_path=edl_path,
            video_path=record.job_dir / "result.mp4",
            bump_version=True,
        )

    def _options_from_record(self, record: JobRecord) -> PipelineOptions:
        return PipelineOptions(
            source_paths=record.request.source_paths,
            music_path=record.request.music_path,
            target_duration=record.request.target_duration,
            vibe_prompt=record.request.vibe_prompt if record.request.brain else "",
            platform=record.request.platform,
            brain=record.request.brain,
            whisper_model=record.request.whisper_model,
            config=self._load_config(record.request.config_path),
        )

    def _load_config(self, config_path: Path | None) -> EditorConfig:
        return load_pipeline_config(config_path)


def _default_executor(
    options: PipelineOptions, *, job_dir: Path, progress: JobProgress | None = None
) -> RenderedJobResult:
    return render_job(options, job_dir=job_dir, progress=progress)
