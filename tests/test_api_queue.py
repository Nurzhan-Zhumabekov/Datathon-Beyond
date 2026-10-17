"""API contract and queue protection. No models are loaded: the analysis function is replaced by a stub."""
import time

from fastapi.testclient import TestClient

from app import config, jobs
from app.main import _public_view, app

client = TestClient(app)  # no startup event -> no real worker
KEY = {"X-API-Key": config.API_KEY}
MOV = ("a.mov", b"\x00" * 1000, "video/quicktime")


def test_api_requires_key():
    assert client.post("/api/v1/analyze", files={"file": MOV}).status_code == 401
    assert client.get("/api/v1/jobs/x", headers={"X-API-Key": "wrong"}).status_code == 401


def test_unknown_job_404_and_missing_file_400():
    assert client.get("/api/v1/jobs/nope", headers=KEY).status_code == 404
    assert client.post("/api/v1/analyze", headers=KEY).status_code == 400


def test_upload_size_limit(monkeypatch):
    monkeypatch.setattr(config, "MAX_UPLOAD_MB", 0.0001)
    r = client.post("/api/v1/analyze", headers=KEY, files={"file": ("a.mov", b"x" * 500, "video/quicktime")})
    assert r.status_code == 413


def test_website_queue_limit(monkeypatch):
    monkeypatch.setattr(jobs, "waiting_ui_jobs", lambda: 1)
    assert client.post("/ui/analyze", files={"file": MOV}).status_code == 429


def test_public_view_has_exact_jury_schema():
    job = {"job_id": "j1", "status": "done", "verdict": "violation", "violations": [{
        "type": "t", "timecode_sec": 1.0, "description": "d", "evidence_frames": [{"timestamp_sec": 1.0, "image_base64": "x"}],
        "confidence": 0.5, "reasoning": ["r"], "regulation": {}}]}
    v = _public_view(job)
    assert set(v) == {"job_id", "status", "verdict", "violations"}
    assert set(v["violations"][0]) == {"type", "timecode_sec", "description", "evidence_frames"}
    assert _public_view({"job_id": "j2", "status": "processing"}) == {"job_id": "j2", "status": "processing"}


def test_jury_jobs_run_before_waiting_website_uploads():
    order = []

    def stub(path, progress, budget):
        order.append((path.stem, budget))
        time.sleep(0.3)
        if path.stem.startswith("bad"):
            raise ValueError("cannot decode video")
        return {"verdict": "no_violation", "violations": []}

    jobs.start(stub)

    def mk(name, src):
        p = config.UPLOAD_DIR / f"{name}.mov"
        p.write_bytes(b"x")
        return jobs.create_job(p, source=src, filename=name)

    mk("ui1", "ui")
    time.sleep(0.1)
    mk("ui2", "ui")
    mk("api1", "api")
    mk("api2", "api")
    bad = mk("bad1", "api")
    deadline = time.time() + 10
    while len(order) < 5 and time.time() < deadline:
        time.sleep(0.1)
    assert [n for n, _ in order] == ["ui1", "api1", "api2", "bad1", "ui2"]
    assert all(b <= config.UI_BUDGET_SEC for n, b in order if n.startswith("ui"))
    # an exception inside the analysis never leaves a job hanging: it becomes status "failed"
    while jobs.get_job(bad["job_id"])["status"] == "processing" and time.time() < deadline:
        time.sleep(0.05)
    j = jobs.get_job(bad["job_id"])
    assert j["status"] == "failed" and "decode" in j["error"]
    assert _public_view(j) == {"job_id": bad["job_id"], "status": "failed", "error": j["error"]}
