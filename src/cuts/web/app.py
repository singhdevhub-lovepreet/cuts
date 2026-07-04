from __future__ import annotations

# ruff: noqa: E501, I001

import os
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse

from cuts.vlm.models import Platform
from cuts.web.schemas import JobCreateResponse, JobStatusResponse
from cuts.web.service import WebJobRequest, WebJobService


HTML_PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>cuts</title>
  <style>
    body { font-family: system-ui, sans-serif; max-width: 960px; margin: 2rem auto; padding: 0 1rem; background: #0f172a; color: #e2e8f0; }
    .card { background: #111827; border: 1px solid #334155; border-radius: 12px; padding: 1rem; margin-bottom: 1rem; }
    label { display: block; margin: 0.75rem 0 0.25rem; }
    input, select, button, textarea { width: 100%; box-sizing: border-box; padding: 0.75rem; border-radius: 8px; border: 1px solid #475569; background: #0b1220; color: #e2e8f0; }
    button { cursor: pointer; background: #2563eb; border: none; font-weight: 700; }
    button:disabled { opacity: 0.6; cursor: wait; }
    small, .muted { color: #94a3b8; }
    pre { white-space: pre-wrap; word-break: break-word; }
    video { width: 100%; border-radius: 12px; margin-top: 1rem; background: black; }
    .row { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }
    @media (max-width: 700px) { .row { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <h1>cuts</h1>
  <p class="muted">Upload clips, set a vibe, and render a 9:16 short.</p>
  <div class="card">
    <form id="job-form">
      <label>Clips</label>
      <input name="clips" id="clips" type="file" multiple required />
      <small>Required. Upload at least one clip.</small>
      <label>Music file</label>
      <input name="music" id="music" type="file" />
      <label>Vibe</label>
      <textarea name="vibe" id="vibe" rows="3" placeholder="punchy uplifting travel diary"></textarea>
      <div class="row">
        <div>
          <label>Platform</label>
          <select name="platform" id="platform">
            <option value="reels">reels</option>
            <option value="shorts">shorts</option>
            <option value="tiktok">tiktok</option>
          </select>
        </div>
        <div>
          <label>Target duration (seconds)</label>
          <input name="target_duration" id="target_duration" type="number" min="1" step="0.1" value="8" />
        </div>
      </div>
      <label style="margin-top: 1rem;">
        <input id="brain" name="brain" type="checkbox" />
        Use AI brain
      </label>
      <button type="submit" id="run-button">Run</button>
    </form>
  </div>
  <div class="card">
    <h2>Status</h2>
    <pre id="status">Idle.</pre>
    <div id="links"></div>
    <video id="preview" controls playsinline hidden></video>
  </div>
  <script>
    const form = document.getElementById("job-form");
    const statusEl = document.getElementById("status");
    const linksEl = document.getElementById("links");
    const preview = document.getElementById("preview");
    const runButton = document.getElementById("run-button");

    async function pollJob(jobId) {
      while (true) {
        const response = await fetch(`/api/jobs/${jobId}`);
        const data = await response.json();
        statusEl.textContent = JSON.stringify(data, null, 2);
        if (data.status === "done") {
          preview.src = data.video_url;
          preview.hidden = false;
          linksEl.innerHTML = `<p><a href="${data.video_url}">Download MP4</a> · <a href="${data.edl_url}">Download EDL JSON</a></p>`;
          runButton.disabled = false;
          return;
        }
        if (data.status === "error") {
          linksEl.innerHTML = "";
          preview.hidden = true;
          runButton.disabled = false;
          return;
        }
        await new Promise((resolve) => setTimeout(resolve, 3000));
      }
    }

    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      runButton.disabled = true;
      linksEl.innerHTML = "";
      preview.hidden = true;
      const formData = new FormData();
      for (const file of document.getElementById("clips").files) {
        formData.append("clips", file);
      }
      const music = document.getElementById("music").files[0];
      if (music) {
        formData.append("music", music);
      }
      formData.append("vibe", document.getElementById("vibe").value);
      formData.append("platform", document.getElementById("platform").value);
      formData.append("target_duration", document.getElementById("target_duration").value);
      formData.append("brain", document.getElementById("brain").checked ? "true" : "false");
      statusEl.textContent = "Submitting...";
      const response = await fetch("/api/jobs", { method: "POST", body: formData });
      const data = await response.json();
      statusEl.textContent = JSON.stringify(data, null, 2);
      await pollJob(data.job_id);
    });
  </script>
</body>
</html>
"""


def create_app(work_root: Path | None = None, service: WebJobService | None = None) -> FastAPI:
    root = work_root or _default_work_root()
    app = FastAPI(title="cuts")
    job_service = service or WebJobService(root)

    @app.on_event("shutdown")
    def _shutdown() -> None:
        if service is None:
            job_service.close()

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        return HTMLResponse(HTML_PAGE)

    @app.post("/api/jobs", response_model=JobCreateResponse)
    async def create_job(
        clips: list[UploadFile] = File(...),
        music: UploadFile | None = File(None),
        vibe: str = Form(""),
        platform: str = Form(Platform.REELS.value),
        target_duration: float | None = Form(None),
        brain: bool = Form(False),
        whisper_model: str = Form("base"),
    ) -> JobCreateResponse:
        if not clips:
            raise HTTPException(status_code=400, detail="at least one clip is required")
        job_root = root / "jobs"
        job_root.mkdir(parents=True, exist_ok=True)
        job_id = uuid4().hex
        job_dir = job_root / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        saved_clips = await _save_uploads(clips, job_dir / "inputs" / "clips")
        music_path = None
        if music is not None and music.filename:
            music_dir = job_dir / "inputs" / "music"
            music_dir.mkdir(parents=True, exist_ok=True)
            music_path = music_dir / _sanitize_filename(music.filename)
            await _save_upload(music, music_path)
        try:
            platform_value = Platform(platform)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="invalid platform") from exc
        request = WebJobRequest(
            source_paths=tuple(saved_clips),
            music_path=music_path,
            target_duration=target_duration,
            vibe_prompt=vibe,
            platform=platform_value,
            brain=brain,
            whisper_model=whisper_model,
        )
        record = job_service.submit(request, job_dir=job_dir, job_id=job_id)
        return JobCreateResponse(
            job_id=record.job_id,
            status=record.status,
            status_url=f"/api/jobs/{record.job_id}",
        )

    @app.get("/api/jobs/{job_id}", response_model=JobStatusResponse)
    def get_job(job_id: str) -> JobStatusResponse:
        snapshot = job_service.snapshot(job_id)
        if snapshot is None:
            raise HTTPException(status_code=404, detail="job not found")
        return snapshot

    @app.get("/api/jobs/{job_id}/video")
    def get_job_video(job_id: str) -> FileResponse:
        record = job_service.get(job_id)
        if record is None:
            raise HTTPException(status_code=404, detail="job not found")
        if record.video_path is None or not record.video_path.exists():
            raise HTTPException(status_code=409, detail="video not ready")
        return FileResponse(
            record.video_path, media_type="video/mp4", filename=record.video_path.name
        )

    @app.get("/api/jobs/{job_id}/edl")
    def get_job_edl(job_id: str) -> FileResponse:
        record = job_service.get(job_id)
        if record is None:
            raise HTTPException(status_code=404, detail="job not found")
        if record.edl_path is None or not record.edl_path.exists():
            raise HTTPException(status_code=409, detail="edl not ready")
        return FileResponse(
            record.edl_path, media_type="application/json", filename=record.edl_path.name
        )

    return app


def _default_work_root() -> Path:
    configured = os.environ.get("CUTS_WEB_WORK_ROOT")
    if configured:
        return Path(configured)
    return Path.cwd() / "work" / "web"


def _sanitize_filename(filename: str) -> str:
    return Path(filename).name or "upload.bin"


async def _save_uploads(files: list[UploadFile], destination_dir: Path) -> list[Path]:
    destination_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for index, upload in enumerate(files):
        path = destination_dir / f"{index:02d}_{_sanitize_filename(upload.filename or 'clip.bin')}"
        await _save_upload(upload, path)
        saved.append(path)
    return saved


async def _save_upload(upload: UploadFile, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    contents = await upload.read()
    destination.write_bytes(contents)
