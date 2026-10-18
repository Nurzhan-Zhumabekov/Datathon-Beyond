"""Admission/cleanup regressions using synthetic bytes, never the running server or GPU."""
import io
import queue
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi import UploadFile
from fastapi.testclient import TestClient

from app import config, jobs
from app import main


@pytest.fixture
def isolated(monkeypatch, tmp_path):
    uploads, persisted = tmp_path / "uploads", tmp_path / "jobs"
    uploads.mkdir()
    persisted.mkdir()
    monkeypatch.setattr(config, "UPLOAD_DIR", uploads)
    monkeypatch.setattr(config, "JOB_DIR", persisted)
    monkeypatch.setattr(jobs, "_jobs", {})
    monkeypatch.setattr(jobs, "_queue", queue.PriorityQueue())
    # No TestClient lifespan: the production worker and models are never started.
    yield TestClient(main.app)


def post(client, source="ui"):
    route = "/ui/analyze" if source == "ui" else "/api/v1/analyze"
    headers = {} if source == "ui" else {"X-API-Key": config.API_KEY}
    return client.post(route, headers=headers, files={"file": ("synthetic.mov", b"synthetic")})


def test_parallel_ui_admission_cannot_bypass_waiting_limit(isolated, monkeypatch):
    # Force all requests past the old preflight count before any job is created.
    ready = threading.Barrier(8)
    original = main._store_upload
    def store(file):
        path = original(file)
        ready.wait(timeout=5)
        return path
    monkeypatch.setattr(main, "_store_upload", store)
    with ThreadPoolExecutor(max_workers=8) as pool:
        statuses = list(pool.map(lambda _: post(isolated).status_code, range(8)))
    assert statuses.count(200) == 1
    assert statuses.count(429) == 7
    assert jobs.waiting_ui_jobs() == 1
    assert len(list(config.UPLOAD_DIR.iterdir())) == 1
    assert len(list(config.JOB_DIR.iterdir())) == 1


def test_running_ui_allows_one_waiting_ui_but_not_a_second(isolated):
    first = post(isolated).json()["job_id"]
    jobs._queue.get_nowait()  # emulate the existing worker taking the first job
    jobs.update_job(first, stage="анализ")
    assert post(isolated).status_code == 200
    assert post(isolated).status_code == 429
    assert len(list(config.UPLOAD_DIR.iterdir())) == 2


def test_jury_api_is_admitted_when_ui_waiting_slot_is_full(isolated):
    assert post(isolated).status_code == 200
    assert post(isolated, "api").status_code == 200
    assert jobs.queue_length() == 2
    assert jobs._queue.get_nowait()[0] == 0


def test_read_error_removes_partial_upload(isolated):
    class BrokenStream(io.BytesIO):
        def read(self, size=-1):
            if self.tell():
                raise OSError("synthetic interrupted upload")
            return super().read(3)
    file = UploadFile(filename="synthetic.mov", file=BrokenStream(b"abcdef"))
    with pytest.raises(OSError, match="synthetic interrupted upload"):
        main._store_upload(file)
    assert not list(config.UPLOAD_DIR.iterdir())


@pytest.mark.parametrize("source", ["api", "ui"])
def test_persistence_failure_leaves_no_phantom_job_or_upload(isolated, monkeypatch, source):
    def cannot_save(job):
        raise OSError("synthetic full disk")
    monkeypatch.setattr(jobs, "_save", cannot_save)
    with pytest.raises(OSError, match="synthetic full disk"):
        post(isolated, source)
    assert jobs.queue_length() == 0
    assert jobs._queue.empty()
    assert not list(config.UPLOAD_DIR.iterdir())
