"""The legal wording was checked against the acts on adilet.zan.kz (docs/rules-check.md). These tests keep it honest:
a type either cites a real document with a clause or says plainly that the source is not determined."""
import importlib

A = importlib.import_module("app.pipeline.analyze")
REG = A.REGISTRY
V, D = REG["violations"], REG["documents"]


def st(t, eating=False, **kw):
    base = {"t": t, "people": 1, "cashier_present": True, "phone": False, "personal_items": False, "eating": eating}
    return {**base, **kw}


def test_every_cited_document_exists_and_has_a_clause():
    for key, spec in V.items():
        if spec["doc"] is not None:
            assert spec["doc"] in D, key
            assert spec["clause"].strip(), key


def test_repealed_security_requirements_are_not_a_basis_for_any_type():
    # Requirements No. 48 were repealed (AFRD resolution of 03.04.2026 No. 53) and their item 71 is about automatic
    # locking by the system, not about the cashier leaving the desk.
    assert all(spec["doc"] != "R48" for spec in V.values())
    assert V["unlocked_workstation"]["doc"] is None
    assert "не определён" in V["unlocked_workstation"]["clause"]


def test_eating_is_its_own_type_without_an_invented_source():
    assert V["eating_at_desk"]["doc"] is None
    assert "еда" not in V["personal_items"]["explain"].lower()
    states = [st(i * 1.5, eating=i in (3, 4, 5)) for i in range(9)]
    ids = {e["id"] for e in A.detect_events(states, frames=[], deadline=0)}
    assert "eating_at_desk" in ids and "personal_items" not in ids


def test_every_event_id_in_the_code_exists_in_the_registry():
    import re
    from pathlib import Path
    code = (Path(A.__file__)).read_text(encoding="utf-8")
    used = set(re.findall(r'"id": "([a-z_]+)"', code)) | set(A.SECOND_LOOK_TYPES.values())
    assert used <= set(V), used - set(V)
