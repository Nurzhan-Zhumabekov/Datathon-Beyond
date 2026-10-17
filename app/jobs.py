"""Job store + single background worker.

Jobs are persisted as JSON files so results survive a server restart.
Status values follow the jury spec exactly: processing | done | failed
(a job waiting in the queue is reported as "processing").
"""
import json
import logging
import queue
import re
import threading
import time
import traceback
import uuid
from pathlib import Path

from . import config

log = logging.getLogger("jobs")

_jobs: dict[str, dict] = {}
_lock = threading.Lock()
# Priority queue: jury API jobs (0) always run before website uploads (1); ties by upload time.
_queue: "queue.PriorityQueue[tuple[int, float, str]]" = queue.PriorityQueue()


def _enqueue(job: dict) -> None:
    _queue.put((0 if job["source"] == "api" else 1, job["created_at"], job["job_id"]))


_ID_RE = re.compile(r"[A-Za-z0-9_-]{1,64}")


def _job_path(job_id: str) -> Path:
    # job_id comes from the URL of unauthenticated routes: never let it carry path separators (on Windows a
    # backslash or a UNC path would otherwise escape the jobs folder).
    if not _ID_RE.fullmatch(job_id):
        raise ValueError(f"bad job id {job_id!r}")
    return config.JOB_DIR / f"{job_id}.json"


def _save(job: dict) -> None:
    tmp = _job_path(job["job_id"]).with_suffix(".tmp")
    # Explicit UTF-8: the Windows default (cp1251) cannot store characters like "→" from the analysis log.
    tmp.write_text(json.dumps(job, ensure_ascii=False), encoding="utf-8")
    tmp.replace(_job_path(job["job_id"]))


def create_job(video_path: Path, source: str, filename: str = "") -> dict:
    job_id = uuid.uuid4().hex[:12]
    job = {
        "job_id": job_id,
        "status": "processing",
        "source": source,  # "api" (jury) or "ui"
        "filename": filename,
        "video_path": str(video_path),
        "created_at": time.time(),
        "progress": 0.0,
        "stage": "в очереди",
    }
    with _lock:
        _jobs[job_id] = job
        _save(job)
    _enqueue(job)
    return job


def waiting_ui_jobs() -> int:
    with _lock:
        return sum(1 for j in _jobs.values()
                   if j["status"] == "processing" and j.get("source") == "ui" and j.get("stage") == "в очереди")


def get_job(job_id: str) -> dict | None:
    if not _ID_RE.fullmatch(job_id):
        return None
    with _lock:
        job = _jobs.get(job_id)
        if job is None and _job_path(job_id).exists():
            job = json.loads(_job_path(job_id).read_text(encoding="utf-8"))
            _jobs[job_id] = job
        return dict(job) if job else None


def update_job(job_id: str, **fields) -> None:
    with _lock:
        job = _jobs[job_id]
        job.update(fields)
        try:
            _save(job)
        except OSError as e:  # the in-memory job stays the source of truth; a disk problem must not kill the worker
            log.error("could not save job %s: %s", job_id, e)


def queue_length() -> int:
    with _lock:
        return sum(1 for j in _jobs.values() if j["status"] == "processing")


def queue_position(job_id: str) -> int:
    with _lock:
        waiting = [j for j in _jobs.values() if j["status"] == "processing" and j.get("stage") == "в очереди"]
    waiting.sort(key=lambda j: (0 if j.get("source") == "api" else 1, j["created_at"]))
    ids = [j["job_id"] for j in waiting]
    return ids.index(job_id) + 1 if job_id in ids else 0


def _worker(analyze_fn) -> None:
    while True:
        _prio, _ts, job_id = _queue.get()
        job = get_job(job_id)
        if not job:
            continue
        started = time.time()
        update_job(job_id, stage="анализ", started_at=started)
        # Jury limit: 5 min per video counted from upload, queue included. Split the time left for this job
        # between it and every job waiting behind it, so the last one in the queue still finishes in time.
        # Only jury jobs waiting behind this one take its time: website uploads run after every jury video.
        with _lock:
            waiting = sum(1 for j in _jobs.values()
                          if j["status"] == "processing" and j.get("source") == "api" and j.get("stage") == "в очереди")
        left = job["created_at"] + config.JURY_LIMIT_SEC - started
        budget = max(25.0, min(config.TIME_BUDGET_SEC, left / (1 + waiting)))
        if job["source"] != "api":
            budget = min(budget, config.UI_BUDGET_SEC)  # a website upload must not hold up a jury video for long

        log_lines: list[dict] = []

        def progress(frac: float, stage: str, log_line: str | None = None, src: str = "", **live) -> None:
            """Live status for the dashboard: stage, active model, counters and a short analysis log."""
            if log_line:
                log_lines.append({"t": round(time.time() - started, 1), "src": src, "text": log_line})
                del log_lines[:-40]
            update_job(job_id, progress=round(float(frac), 3), stage=stage, live=live, log=list(log_lines))

        try:
            result = analyze_fn(Path(job["video_path"]), progress, budget)
            update_job(job_id, status="done", progress=1.0, stage="готово",
                       finished_at=time.time(), elapsed_sec=round(time.time() - started, 1), **result)
        except Exception as e:  # never leave a job hanging in "processing"
            log.error("job %s failed: %s\n%s", job_id, e, traceback.format_exc())
            update_job(job_id, status="failed", stage="ошибка", error=str(e), finished_at=time.time())
        finally:
            if job["source"] == "api":
                Path(job["video_path"]).unlink(missing_ok=True)


def _cleanup_loop() -> None:
    """Delete UI-uploaded videos after a while (they are kept briefly for playback)."""
    while True:
        now = time.time()
        for p in config.UPLOAD_DIR.glob("*"):
            if now - p.stat().st_mtime > config.KEEP_UI_VIDEOS_SEC:
                p.unlink(missing_ok=True)
        time.sleep(300)


def start(analyze_fn) -> None:
    # Re-queue jobs that were interrupted by a restart.
    for p in sorted(config.JOB_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime):
        job = json.loads(p.read_text(encoding="utf-8"))
        _jobs[job["job_id"]] = job
        if job["status"] == "processing":
            if Path(job["video_path"]).exists():
                job["stage"] = "в очереди"
                _enqueue(job)
            else:
                job.update(status="failed", error="video lost after restart")
            _save(job)
    threading.Thread(target=_worker, args=(analyze_fn,), daemon=True).start()
    threading.Thread(target=_cleanup_loop, daemon=True).start()
