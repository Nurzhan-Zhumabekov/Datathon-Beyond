"""An unreadable model answer must never be read as 'the cashier left' (that would invent a departure)."""
import importlib

from PIL import Image

from app.video import Frame

A = importlib.import_module("app.pipeline.analyze")
FRAME = Frame(5.0, Image.new("RGB", (464, 848)))
GOOD = '{"people": 2, "cashier_present": true, "phone": false, "personal_items": false, "eating": false}'


def test_parsed_answer(monkeypatch):
    monkeypatch.setattr(A.vlm, "ask", lambda *a, **k: GOOD)
    st = A.read_state_safe(FRAME, None)
    assert st["parsed"] and st["cashier_present"] and st["people"] == 2


def test_garbage_keeps_previous_state(monkeypatch):
    calls = []
    monkeypatch.setattr(A.vlm, "ask", lambda *a, **k: calls.append(1) or "sorry, I cannot")
    prev = {"t": 3.5, "people": 1, "cashier_present": True, "phone": False, "personal_items": False, "eating": False}
    st = A.read_state_safe(FRAME, prev)
    assert len(calls) == 2  # retried once
    assert st["cashier_present"] is True and st["t"] == 5.0 and st["unparsed"]


def test_garbage_on_first_frame_is_neutral(monkeypatch):
    monkeypatch.setattr(A.vlm, "ask", lambda *a, **k: "")
    st = A.read_state_safe(FRAME, None)
    assert st["cashier_present"] is True and st["unparsed"]


def test_no_departure_from_one_unreadable_frame():
    base = {"people": 1, "cashier_present": True, "phone": False, "personal_items": False, "eating": False}
    states = [dict(base, t=t * 1.5) for t in range(8)]
    assert A.find_departures(states) == []


def test_valuables_filter():
    import re
    def counts(o):
        return bool(re.search(A.VALUABLES, o)) and not any(w in o for w in A.STANDARD_EQUIPMENT)
    assert counts("money bundle") and counts("envelope") and counts("green object")
    assert not counts("notepad") and not counts("cash register") and not counts("banknote counting machine")
