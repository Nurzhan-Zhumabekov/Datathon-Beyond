"""Mimic the jury: upload videos through the public API, poll, validate schema, score.

usage: .venv/bin/python scripts/jury_sim.py --base http://127.0.0.1:8000 --videos ~/Downloads/open/videos \
          --labels ~/Downloads/open/open_labels.xlsx [--only X-1,X-48] [--parallel 1]
"""
import argparse
import base64
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).parent))
from labels import norm, read_labels  # noqa: E402

# Per violation type: in how many videos it fired, split by ground truth (violation video / clean video).
BY_TYPE: dict[str, list[int]] = {}

# Byte-identical duplicates in the dataset: evaluate each pair once.
DUPLICATES = {"X-47": "X-4", "X-48": "X-5"}


def validate(res: dict) -> list[str]:
    errs = []
    if res.get("status") != "done":
        return [f"status={res.get('status')}"]
    if res.get("verdict") not in ("violation", "no_violation"):
        errs.append(f"bad verdict {res.get('verdict')}")
    vs = res.get("violations")
    if not isinstance(vs, list):
        return errs + ["violations not a list"]
    if (res["verdict"] == "violation") != bool(vs):
        errs.append("verdict/violations mismatch")
    for v in vs:
        for k in ("type", "timecode_sec", "description", "evidence_frames"):
            if k not in v:
                errs.append(f"missing {k}")
        fr = v.get("evidence_frames", [])
        if not 1 <= len(fr) <= 3:
            errs.append(f"{len(fr)} evidence frames")
        for f in fr:
            b = base64.b64decode(f["image_base64"])
            if not b.startswith(b"\xff\xd8"):
                errs.append("frame is not JPEG")
    return errs


def _retry(fn, tries: int = 5):
    """The jury's client would retry a dropped connection; so do we."""
    for k in range(tries):
        try:
            return fn()
        except httpx.TransportError:
            if k == tries - 1:
                raise
            time.sleep(2)


def run_one(client: httpx.Client, base: str, key: str, path: Path) -> dict:
    t0 = time.time()
    def upload():
        with path.open("rb") as f:
            return client.post(f"{base}/api/v1/analyze", headers={"X-API-Key": key}, files={"file": (path.name, f, "video/quicktime")})
    r = _retry(upload)
    r.raise_for_status()
    job_id = r.json()["job_id"]
    while True:
        r = _retry(lambda: client.get(f"{base}/api/v1/jobs/{job_id}", headers={"X-API-Key": key}))
        r.raise_for_status()
        res = r.json()
        if res["status"] != "processing":
            break
        if time.time() - t0 > 300:
            res = {"status": "timeout"}
            break
        time.sleep(2)
    res["_elapsed"] = round(time.time() - t0, 1)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--key", default=os.environ.get("API_KEY"))
    ap.add_argument("--videos", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--only", default="")
    ap.add_argument("--parallel", type=int, default=1)
    a = ap.parse_args()
    if not a.key:
        for line in (Path(__file__).parent.parent / ".env").read_text().splitlines():
            if line.startswith("API_KEY="):
                a.key = line.split("=", 1)[1].strip()

    labels = read_labels(a.labels)
    vids = sorted(Path(a.videos).expanduser().glob("*.mov"), key=lambda p: int(norm(p.stem).split("-")[1]))
    vids = [p for p in vids if norm(p.stem) not in DUPLICATES]
    if a.only:
        want = {norm(x) for x in a.only.split(",")}
        vids = [p for p in vids if norm(p.stem) in want]

    client = httpx.Client(timeout=120)
    tp = fp = tn = fn = 0
    errs_tc = []
    results = []
    with ThreadPoolExecutor(a.parallel) as ex:
        stream = ex.map(lambda p: (p, run_one(client, a.base, a.key, p)), vids)
        for p, res in stream:
            results.append((p, res))
            tp, fp, tn, fn = report(p, res, labels, errs_tc, tp, fp, tn, fn)
    n = tp + tn + fp + fn
    print(f"\naccuracy {(tp + tn) / n:.2f}  recall {tp / max(1, tp + fn):.2f}  FP {fp}/{fp + tn}  FN {fn}/{tp + fn}")
    if errs_tc:
        errs_tc.sort()
        print(f"timecode error: median {errs_tc[len(errs_tc) // 2]:.1f}s, within 2s: {sum(e <= 2 for e in errs_tc)}/{len(errs_tc)}")
    print(f"max time per video: {max(r['_elapsed'] for _, r in results)}s")
    if BY_TYPE:
        print("\nby type (videos where it fired): on violation videos / on clean videos")
        for typ, (on_v, on_c) in sorted(BY_TYPE.items(), key=lambda kv: -sum(kv[1])):
            print(f"  {on_v:3d} / {on_c:3d}  {typ}")


def report(p, res, labels, errs_tc, tp, fp, tn, fn):
    if True:
        name = norm(p.stem)
        lab = labels.get(name, {})
        truth = lab.get("violation")
        secs = [lab.get("sec")]
        if name in DUPLICATES.values():
            dup = [k for k, v in DUPLICATES.items() if v == name][0]
            secs.append(labels.get(dup, {}).get("sec"))
        secs = [s for s in secs if s is not None]
        problems = validate(res)
        pred = res.get("verdict") == "violation"
        if truth and pred: tp += 1
        elif truth and not pred: fn += 1
        elif not truth and pred: fp += 1
        else: tn += 1
        for typ in {v["type"] for v in res.get("violations", [])}:
            row = BY_TYPE.setdefault(typ, [0, 0])
            row[0 if truth else 1] += 1
        tcs = [v["timecode_sec"] for v in res.get("violations", [])]
        tc_err = None
        if truth and pred and secs and tcs:
            tc_err = min(abs(t - s) for t in tcs for s in secs)
            errs_tc.append(tc_err)
        mark = "OK " if truth == pred else "ERR"
        types = "; ".join(f"{v['timecode_sec']}s {v['type']}" for v in res.get("violations", []))
        print(f"{mark} {name:6} truth={'V' if truth else '-'}@{secs} pred={'V' if pred else '-'} "
              f"tc_err={tc_err if tc_err is None else round(tc_err, 1)} {res['_elapsed']}s {problems or ''} | {types}", flush=True)
    return tp, fp, tn, fn


if __name__ == "__main__":
    main()
