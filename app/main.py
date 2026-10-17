"""HTTP layer: jury API (/api/v1, needs X-API-Key) + web UI (/, /ui/*, no key)."""
import logging
import shutil
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from . import config, jobs
from .pipeline import analyze

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
app = FastAPI(title="ForteBank cash-desk violation detector", docs_url=None, redoc_url=None)

STATIC = Path(__file__).parent / "static"
ALLOWED_EXT = {".mov", ".mp4", ".m4v", ".avi", ".mkv"}


@app.on_event("startup")
def _startup() -> None:
    jobs.start(analyze)


def _check_key(key: str | None) -> None:
    if key != config.API_KEY:
        raise HTTPException(status_code=401, detail="invalid or missing X-API-Key")


def _store_upload(file: UploadFile) -> Path:
    if file is None or not file.filename:
        raise HTTPException(status_code=400, detail="field 'file' is required")
    ext = Path(file.filename).suffix.lower() or ".mov"
    if ext not in ALLOWED_EXT:
        raise HTTPException(status_code=400, detail=f"unsupported file type {ext}")
    dest = config.UPLOAD_DIR / f"{uuid.uuid4().hex}{ext}"
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    if dest.stat().st_size == 0:
        dest.unlink()
        raise HTTPException(status_code=400, detail="empty file")
    return dest


def _public_view(job: dict) -> dict:
    """Exact jury schema: job_id, status, and verdict/violations once done."""
    out = {"job_id": job["job_id"], "status": job["status"]}
    if job["status"] == "done":
        out["verdict"] = job["verdict"]
        out["violations"] = [
            {k: v[k] for k in ("type", "timecode_sec", "description", "evidence_frames")}
            for v in job["violations"]
        ]
    elif job["status"] == "failed":
        out["error"] = job.get("error", "processing failed")
    return out


# ---------- Jury API ----------

@app.post("/api/v1/analyze")
def api_analyze(file: UploadFile = File(None), x_api_key: str | None = Header(None)):
    _check_key(x_api_key)
    path = _store_upload(file)
    job = jobs.create_job(path, source="api", filename=file.filename)
    return {"job_id": job["job_id"], "status": "processing"}


@app.get("/api/v1/jobs/{job_id}")
def api_job(job_id: str, x_api_key: str | None = Header(None)):
    _check_key(x_api_key)
    job = jobs.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    return _public_view(job)


# ---------- Web UI ----------

@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.post("/ui/analyze")
def ui_analyze(file: UploadFile = File(None)):
    path = _store_upload(file)
    job = jobs.create_job(path, source="ui", filename=file.filename)
    return {"job_id": job["job_id"], "status": "processing"}


@app.get("/ui/jobs/{job_id}")
def ui_job(job_id: str):
    job = jobs.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    view = _public_view(job)
    view.update(progress=job.get("progress", 0), stage=job.get("stage", ""),
                queue_position=jobs.queue_position(job_id), elapsed_sec=job.get("elapsed_sec"))
    if job["status"] == "done":
        view["violations"] = job["violations"]  # UI also gets rule/regulation fields
        view["summary"] = job.get("summary", "")
        view["duration"] = job.get("duration")
        view["segments"] = job.get("segments", [])
        view["pipeline"] = job.get("pipeline", {})
        view["filename"] = job.get("filename", "")
        view["video_available"] = job["source"] == "ui" and Path(job["video_path"]).exists()
    return view


@app.get("/ui/video/{job_id}")
def ui_video(job_id: str):
    job = jobs.get_job(job_id)
    if not job or job["source"] != "ui" or not Path(job["video_path"]).exists():
        raise HTTPException(status_code=404, detail="video not available")
    return FileResponse(job["video_path"], media_type="video/mp4")


@app.get("/ui/status")
def ui_status():
    from .pipeline import vlm
    return {"queue": jobs.queue_length(), "scan_model": vlm.available("scan"), "judge_model": vlm.available("judge")}


@app.get("/health")
def health():
    return JSONResponse({"ok": True})
