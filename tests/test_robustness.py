"""Bad inputs must end as status 'failed' (never hang), and tiny valid videos must still produce a valid answer.
Models are replaced by stubs, so these tests run in seconds without loading anything."""
import importlib

import av
import numpy as np
import pytest

from app import config

A = importlib.import_module("app.pipeline.analyze")
GOOD = '{"people": 1, "cashier_present": true, "phone": false, "personal_items": false, "eating": false}'


def _stub_models(monkeypatch):
    monkeypatch.setattr(A.vlm, "load", lambda kind="scan": None)
    monkeypatch.setattr(A.vlm, "ask", lambda *a, **k: GOOD)


def _make_video(path, seconds=1.0, fps=10, size=(64, 96)):
    with av.open(str(path), "w") as c:
        s = c.add_stream("libx264", rate=fps)
        s.width, s.height, s.pix_fmt = size[0], size[1], "yuv420p"
        for i in range(int(seconds * fps)):
            img = np.full((size[1], size[0], 3), 40 + i, dtype=np.uint8)
            for pkt in s.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
                c.mux(pkt)
        for pkt in s.encode():
            c.mux(pkt)


def test_corrupt_file_fails_cleanly(monkeypatch):
    _stub_models(monkeypatch)
    p = config.UPLOAD_DIR / "broken.mov"
    p.write_bytes(b"\x00\x00\x00\x18ftypqt  " + b"garbage" * 100)
    with pytest.raises(Exception):  # the worker turns any exception into status "failed" (see test_api_queue)
        A.analyze(p, lambda *a, **k: None, 60)


def test_not_a_video_fails_cleanly(monkeypatch):
    _stub_models(monkeypatch)
    p = config.UPLOAD_DIR / "notes.mov"
    p.write_text("this is not a video")
    with pytest.raises(Exception):
        A.analyze(p, lambda *a, **k: None, 60)


def test_one_second_video_gives_valid_answer(monkeypatch, tmp_path):
    _stub_models(monkeypatch)
    p = config.UPLOAD_DIR / "short.mp4"
    _make_video(p, seconds=1.0)
    res = A.analyze(p, lambda *a, **k: None, 60)
    assert res["verdict"] in ("violation", "no_violation")
    assert (res["verdict"] == "violation") == bool(res["violations"])
    assert res["duration"] > 0 and res["segments"]


def test_short_budget_still_scans_frames(monkeypatch):
    """Several jury videos at once give each a short budget: the scanner must still read frames."""
    _stub_models(monkeypatch)
    p = config.UPLOAD_DIR / "busy.mp4"
    _make_video(p, seconds=8.0)
    for budget in (25, 40, 60):
        res = A.analyze(p, lambda *a, **k: None, budget)
        assert res["pipeline"]["frames_scanned"] >= A.MIN_FRAMES, budget
