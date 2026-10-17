"""Timeline rules on synthetic scanner states: if someone changes a threshold, these show what broke."""
import importlib

A = importlib.import_module("app.pipeline.analyze")


def st(t, people=1, present=True, phone=False, items=False, eating=False):
    return {"t": t, "people": people, "cashier_present": present, "phone": phone, "personal_items": items, "eating": eating}


def timeline(pattern, step=1.5, **kw):
    """pattern: list of dicts of overrides per sample."""
    return [st(i * step, **{**kw, **o}) for i, o in enumerate(pattern)]


def test_departure_needs_two_absent_samples():
    stays = timeline([{}] * 4 + [{"present": False, "people": 0}] + [{}] * 4)
    leaves = timeline([{}] * 4 + [{"present": False, "people": 0}] * 4)
    assert A.find_departures(stays) == []
    assert A.find_departures(leaves) == [4]


def test_crowd_rule_fires_on_three_people():
    s = timeline([{}] * 3 + [{"people": 3}] + [{"people": 2}] * 3)
    ev = A.detect_events(s, frames=[], deadline=0)  # deadline 0: no model-backed checks run
    assert any(e["id"] == "outsider_in_cash_area" for e in ev)


def test_short_visit_rule_and_its_limit():
    quick = timeline([{}] * 3 + [{"people": 2}] * 3 + [{}] * 3)        # visitor ~4.5 s
    long_ = timeline([{}] * 3 + [{"people": 2}] * 8 + [{}] * 3)        # visitor ~12 s: a normal client
    assert any(e["id"] == "collection_bag" for e in A.detect_events(quick, [], 0))
    assert not any(e["id"] == "collection_bag" for e in A.detect_events(long_, [], 0))


def test_personal_items_need_three_frames_and_cashier_alone():
    two = timeline([{"items": True}] * 2 + [{}] * 4)
    three = timeline([{"items": True}] * 3 + [{}] * 4)
    with_client = timeline([{"items": True, "people": 2}] * 4 + [{}] * 3)
    ids = lambda s: [e["id"] for e in A.detect_events(s, [], 0)]
    assert "personal_items" not in ids(two)
    assert "personal_items" in ids(three)
    assert "personal_items" not in ids(with_client)


def test_clean_timeline_has_no_events():
    s = timeline([{}] * 12)
    assert A.detect_events(s, [], 0) == []


def test_departure_timecode_uses_lead():
    assert A.DEPARTURE_LEAD_SEC == 3.0
