"""Fixes from the independent review: a model-server error must not fail the whole video, the time budget counts only
jury jobs behind the current one, and URL job ids cannot escape the jobs folder."""
import importlib
import sys
import types

from fastapi.testclient import TestClient

from app import config, jobs
from app.main import app

client = TestClient(app)
analyze = importlib.import_module("app.pipeline.analyze")


def test_job_routes_answer_404_for_path_like_ids():
    for bad in ("..%5Csecret", "%5C%5Chost%5Cshare%5Cx", "C%3A%5Cx"):
        assert client.get(f"/ui/jobs/{bad}").status_code == 404
        assert client.get(f"/ui/video/{bad}").status_code == 404
        assert client.get(f"/api/v1/jobs/{bad}", headers={"X-API-Key": config.API_KEY}).status_code == 404


def _states():
    # cashier present, then gone for four samples, then back
    def st(t, present):
        return {"t": t, "people": 1 if present else 0, "cashier_present": present, "phone": False,
                "personal_items": False, "eating": False}
    return [st(t, t < 4 or t > 12) for t in (0, 1.5, 3, 4.5, 6, 7.5, 9, 10.5, 12, 13.5, 15, 16.5)]


def test_failed_desk_check_skips_the_departure_instead_of_failing(monkeypatch):
    from PIL import Image
    from app.video import Frame
    frames = [Frame(t=i * 0.5, image=Image.new("RGB", (32, 32))) for i in range(36)]

    def boom(*args, **kwargs):
        raise RuntimeError("model server answered 500")

    monkeypatch.setattr(analyze, "check_left_behind", boom)
    monkeypatch.setattr(analyze, "note", lambda *a, **k: None)
    events = analyze.detect_events(_states(), frames, deadline=1e12)  # must not raise
    assert events == []


def test_failed_desk_check_still_reports_leaving_during_service(monkeypatch):
    from PIL import Image
    from app.video import Frame
    frames = [Frame(t=i * 0.5, image=Image.new("RGB", (32, 32))) for i in range(36)]
    states = [dict(s, people=(2 if s["cashier_present"] else 0)) for s in _states()]  # a visitor is in the room

    def boom(*args, **kwargs):
        raise RuntimeError("model server answered 500")

    monkeypatch.setattr(analyze, "check_left_behind", boom)
    monkeypatch.setattr(analyze, "note", lambda *a, **k: None)
    events = analyze.detect_events(states, frames, deadline=1e12)
    assert [e["id"] for e in events] == ["left_during_service"]
