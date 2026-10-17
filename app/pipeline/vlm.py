"""Local vision-language models (Qwen3-VL via MLX on Apple Silicon). No network calls at inference.

Two models form a cascade:
  scan  — Qwen3-VL-4B: fast, reads every sampled frame
  judge — Qwen3-VL-8B: slower and stronger, only re-checks suspicious moments
"""
import json
import logging
import re
import tempfile
import threading
import time
from pathlib import Path

from PIL import Image

from .. import config

log = logging.getLogger("vlm")
_lock = threading.Lock()
_models: dict[str, tuple] = {}


def _path(kind: str) -> str:
    return config.VLM_MODEL if kind == "scan" else config.JUDGE_MODEL


def available(kind: str) -> bool:
    p = Path(_path(kind))
    # The judge weights are optional; ".complete" is written once the download finished.
    return p.joinpath("config.json").exists() and (kind == "scan" or p.joinpath(".complete").exists())


def load(kind: str = "scan"):
    with _lock:
        if kind not in _models:
            from mlx_vlm import load as mlx_load
            t0 = time.time()
            _models[kind] = mlx_load(_path(kind))
            log.info("VLM %s (%s) loaded in %.1fs", kind, _path(kind), time.time() - t0)
    return _models[kind]


def ask(images: list[Image.Image], question: str, max_tokens: int = 120, kind: str = "scan") -> str:
    """Ask a model about one or more frames. Returns raw text."""
    from mlx_vlm import generate
    from mlx_vlm.prompt_utils import apply_chat_template

    model, processor = load(kind)
    with tempfile.TemporaryDirectory() as td:
        paths = []
        for i, im in enumerate(images):
            p = Path(td) / f"f{i}.jpg"
            im.convert("RGB").save(p, quality=90)
            paths.append(str(p))
        prompt = apply_chat_template(processor, model.config, question, num_images=len(paths))
        with _lock:
            out = generate(model, processor, prompt, image=paths, max_tokens=max_tokens,
                           temperature=0.0, verbose=False)
    return getattr(out, "text", out) if not isinstance(out, str) else out


def parse_json(text: str):
    m = re.search(r"[\[{].*[\]}]", text, re.S)
    if not m:
        return None
    raw = m.group(0)
    for candidate in (raw, raw.replace("'", '"').replace("True", "true").replace("False", "false")):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    m = re.search(r"\{.*?\}", text, re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    return None


def parse_bbox(text: str) -> list[float] | None:
    """Qwen3-VL returns boxes on a 0..1000 grid; convert to 0..1 fractions."""
    m = re.search(r"\[\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*\]", text)
    if not m:
        return None
    x1, y1, x2, y2 = (min(1.0, max(0.0, float(v) / 1000)) for v in m.groups())
    if x2 - x1 < 0.01 or y2 - y1 < 0.01:
        return None
    return [x1, y1, x2, y2]
