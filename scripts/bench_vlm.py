"""Quick check of VLM speed and answers on chosen frames.

usage: .venv/bin/python scripts/bench_vlm.py <video> <t1,t2,...> [max_side]
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import importlib  # noqa: E402
A = importlib.import_module("app.pipeline.analyze")
from app.pipeline import vlm  # noqa: E402
from app.video import sample_frames  # noqa: E402

video = Path(sys.argv[1])
times = [float(x) for x in sys.argv[2].split(",")]
if len(sys.argv) > 3:
    A.config_max_side = lambda: int(sys.argv[3])

t0 = time.time()
vlm.load()
print(f"load {time.time() - t0:.1f}s")
frames = sample_frames(video, fps=2.0)
for t in times:
    f = min(frames, key=lambda f: abs(f.t - t))
    t1 = time.time()
    st = A.read_state(f)
    print(f"t={f.t:5.1f} {time.time() - t1:4.1f}s ", {k: st[k] for k in A.STATE_KEYS})
    if not any(st[k] is not None for k in A.STATE_KEYS):
        print("   RAW:", st["raw"][:300])
