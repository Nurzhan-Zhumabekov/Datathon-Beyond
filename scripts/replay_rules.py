"""Replay the scanner rules on stored model answers — no models are run.

Every finished job keeps the scanner's per-frame states (people, cashier present, phone, personal items,
eating). This script re-applies the timeline rules to those stored states with different parameters and
scores them against the labels, so thresholds and timecode offsets can be tuned in seconds.

usage: .venv/bin/python scripts/replay_rules.py [--labels ~/Downloads/open/open_labels.xlsx]
"""
import argparse
import collections
import glob
import importlib
import itertools
import json
import os
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from labels import norm, read_labels  # noqa: E402

A = importlib.import_module("app.pipeline.analyze")
DUP = {"X-47": "X-4", "X-48": "X-5"}
NEAR = 6.0  # a finding counts as a hit when it is within this many seconds of the labelled second


def load_runs():
    runs = collections.defaultdict(list)
    for p in glob.glob(str(ROOT / "data/jobs/*.json")) + glob.glob(str(ROOT / "data/old_jobs/*.json")):
        try:
            j = json.load(open(p))
        except (OSError, json.JSONDecodeError):
            continue
        if j.get("status") == "done" and j.get("filename") and j.get("states"):
            name = norm(j["filename"].rsplit(".", 1)[0])
            runs[DUP.get(name, name)].append(j["states"])
    return runs


def departures(states):
    return [states[i]["t"] for i in A.find_departures(states)]


def scanner_events(states, p):
    """The no-VLM rules (phone, personal items, eating, outsider, short visit) with parameters p."""
    ev = []
    ts = [s["t"] for s in states]
    cnt = [s["people"] or 0 for s in states]
    alone = [c <= 1 and s["cashier_present"] for s, c in zip(states, cnt)]
    for a, _ in A._runs(A._smooth([s["phone"] and al for s, al in zip(states, alone)]), p["phone"]):
        ev.append(("phone", ts[a]))
    for a, _ in A._runs(A._smooth([s["personal_items"] and al for s, al in zip(states, alone)]), p["items"]):
        ev.append(("items", ts[a]))
    for a, _ in A._runs(A._smooth([s["eating"] and al for s, al in zip(states, alone)]), p["eat"]):
        ev.append(("eating", ts[a]))
    for a, _ in A._runs([c >= 3 for c in cnt], p["crowd"]):
        ev.append(("outsider", ts[a]))
    for a, b in A._runs([c >= 2 for c in cnt], 2):
        if a < 2 or b + 2 >= len(states):
            continue
        if all(c == 1 for c in cnt[a - 2:a]) and all(c == 1 for c in cnt[b + 1:b + 3]) and ts[b + 1] - ts[a] <= p["visit"]:
            ev.append(("visit", ts[a] + (ts[b] - ts[a]) / 2))
    return ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=os.path.expanduser("~/Downloads/open/open_labels.xlsx"))
    a = ap.parse_args()
    labels = read_labels(a.labels)
    runs = load_runs()
    vids = sorted(v for v in runs if v in labels and v not in DUP)
    nviol = sum(labels[v]["violation"] for v in vids)
    print(f"videos with stored scanner answers: {len(vids)} ({nviol} with violation), runs: {sum(len(runs[v]) for v in vids)}\n")

    # ---- 1. Timecode offset for "cashier left" findings: label second vs the moment the cashier is gone ----
    diffs = collections.defaultdict(list)
    for v in vids:
        if not labels[v]["violation"]:
            continue
        secs = [labels[v]["sec"]] + ([labels["X-47"]["sec"]] if v == "X-4" else [])
        for st in runs[v]:
            for t_abs in departures(st):
                d = min((s - t_abs for s in secs), key=abs)
                if -12 <= d <= 6:
                    diffs[v].append(d)
    per_video = {v: statistics.median(d) for v, d in diffs.items()}
    if per_video:
        offs = sorted(per_video.values())
        best = statistics.median(offs)
        print("1) TIMECODE for 'cashier left': label − moment the cashier disappears")
        print(f"   per video (median over runs): {', '.join(f'{v} {d:+.1f}' for v, d in sorted(per_video.items()))}")
        print(f"   median {best:+.1f} s (now the code uses −3.5 s)")
        for off in (-2.0, -2.5, -3.0, -3.5, -4.0, -4.5):
            errs = [abs(d - off) for d in per_video.values()]
            print(f"   offset {off:+.1f}: mean |error| {statistics.mean(errs):.2f} s, within 2 s: {sum(e <= 2 for e in errs)}/{len(errs)}")
        print()

    # ---- 2. Scanner rules: which settings fire on violation videos near the label vs on clean videos ----
    print("2) SCANNER RULES replayed on stored answers (share of runs, averaged per video)")
    current = {"phone": 2, "items": 3, "eat": 2, "crowd": A.CROWD_MIN_FRAMES, "visit": A.VISIT_MAX_SEC}
    grid = {"phone": (1, 2, 3), "items": (2, 3, 4), "eat": (1, 2, 3), "crowd": (1, 2, 3), "visit": (6.0, 9.0, 12.0)}

    def score(p, kinds):
        hit = fp = 0.0
        for v in vids:
            lab = labels[v]
            secs = [lab["sec"]] + ([labels["X-47"]["sec"]] if v == "X-4" else []) if lab["violation"] else []
            fired_hit = fired_any = 0
            for st in runs[v]:
                ev = [t for k, t in scanner_events(st, p) if k in kinds]
                fired_any += bool(ev)
                fired_hit += any(abs(t - s) <= NEAR for t in ev for s in secs)
            n = len(runs[v])
            if lab["violation"]:
                hit += fired_hit / n
            else:
                fp += fired_any / n
        return hit, fp

    rows = []
    for kind, key in (("phone", "phone"), ("items", "items"), ("eating", "eat"), ("outsider", "crowd"), ("visit", "visit")):
        for val in grid[key]:
            p = dict(current, **{key: val})
            h, f = score(p, {kind})
            rows.append((kind, key, val, h, f, val == current[key]))
    print(f"   {'rule':9} {'setting':12} {'hits on violation videos':>26} {'fires on clean videos':>23}")
    for kind, key, val, h, f, cur in rows:
        print(f"   {kind:9} {key + '=' + str(val):12} {h:20.1f} / {nviol:<4} {f:16.1f} / {len(vids) - nviol:<4}{'  ← now' if cur else ''}")
    print()

    # ---- 3. Error map: what the scanner rules alone say per video (current settings) ----
    print("3) PER VIDEO with current settings (scanner rules only, before the 8B check): fired types / runs")
    for v in vids:
        lab = labels[v]
        kinds = collections.Counter()
        for st in runs[v]:
            for k in {k for k, _ in scanner_events(st, current)}:
                kinds[k] += 1
        dep = sum(bool(departures(st)) for st in runs[v])
        tag = f"V@{lab['sec']:.0f}s" if lab["violation"] else "clean"
        fired = ", ".join(f"{k} {n}/{len(runs[v])}" for k, n in kinds.most_common()) or "—"
        print(f"   {v:6} {tag:8} runs {len(runs[v]):2}  cashier-left in {dep}/{len(runs[v])}  | {fired}")


if __name__ == "__main__":
    main()
