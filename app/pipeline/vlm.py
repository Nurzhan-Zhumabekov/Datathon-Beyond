"""Local vision-language models (Qwen3-VL). Frames never leave this machine.

Two models form a cascade:
  scan  — Qwen3-VL-4B: fast, reads every sampled frame
  judge — Qwen3-VL-8B: slower and stronger, only re-checks suspicious moments

Two interchangeable backends, chosen by VLM_BACKEND:
  mlx    — in-process MLX (Apple Silicon only), weights in ./models
  openai — a local OpenAI-compatible server on this machine (llama.cpp `llama-server`, Ollama, ...).
           Works on Windows, Linux and Mac. Only loopback addresses are accepted, so a typo in VLM_URL
           can never send video frames to a cloud service.
"""
import base64
import io
import json
import logging
import re
import tempfile
import threading
import time
from pathlib import Path
from urllib.parse import urlparse

from PIL import Image

from .. import config

log = logging.getLogger("vlm")
_lock = threading.Lock()
_models: dict[str, tuple] = {}
_reach: dict[str, tuple[float, bool]] = {}
_LOOPBACK = {"127.0.0.1", "localhost", "::1"}


def _openai() -> bool:
    return config.VLM_BACKEND == "openai"


def _url(kind: str) -> str:
    return (config.VLM_URL if kind == "scan" else config.JUDGE_URL).rstrip("/")


def _local_url(kind: str) -> str:
    url = _url(kind)
    if urlparse(url).hostname not in _LOOPBACK:
        raise RuntimeError(f"{kind} model URL {url!r} is not a local address: video frames must stay on this machine")
    return url


def _reachable(url: str) -> bool:
    """Is the model server answering? Cached for 5 s because the status endpoint asks often."""
    now = time.time()
    hit = _reach.get(url)
    if hit and now - hit[0] < 5.0:
        return hit[1]
    import httpx
    try:
        ok = httpx.get(url + "/v1/models", timeout=2.0).status_code == 200
    except httpx.HTTPError:
        ok = False
    _reach[url] = (now, ok)
    return ok


def _path(kind: str) -> str:
    return config.VLM_MODEL if kind == "scan" else config.JUDGE_MODEL


def available(kind: str) -> bool:
    if _openai():
        if not _url(kind):  # JUDGE_URL is optional: without it the scanner works alone
            return False
        try:
            return _reachable(_local_url(kind))
        except RuntimeError as e:
            log.error("%s", e)
            return False
    p = Path(_path(kind))
    # The judge weights are optional; ".complete" is written once the download finished.
    return p.joinpath("config.json").exists() and (kind == "scan" or p.joinpath(".complete").exists())


def load(kind: str = "scan"):
    if _openai():
        if not available(kind):
            raise RuntimeError(f"the '{kind}' model server is not reachable at {_url(kind) or '(URL not set)'}")
        return None
    with _lock:
        if kind not in _models:
            from mlx_vlm import load as mlx_load
            t0 = time.time()
            _models[kind] = mlx_load(_path(kind))
            log.info("VLM %s (%s) loaded in %.1fs", kind, _path(kind), time.time() - t0)
    return _models[kind]


def _ask_openai(images: list[Image.Image], question: str, max_tokens: int, kind: str) -> str:
    import httpx
    url = _local_url(kind)
    parts = []
    for im in images:
        buf = io.BytesIO()
        im.convert("RGB").save(buf, format="JPEG", quality=90)
        b64 = base64.b64encode(buf.getvalue()).decode()
        parts.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
    parts.append({"type": "text", "text": question})
    body = {"model": config.VLM_API_MODEL if kind == "scan" else config.JUDGE_API_MODEL,
            "messages": [{"role": "user", "content": parts}],
            "max_tokens": max_tokens, "temperature": 0.0, "stream": False}
    with _lock:
        r = httpx.post(url + "/v1/chat/completions", json=body, timeout=config.VLM_TIMEOUT_SEC)
    if r.status_code != 200:
        raise RuntimeError(f"the '{kind}' model server answered {r.status_code}: {r.text[:200]}")
    content = r.json()["choices"][0]["message"]["content"]
    if isinstance(content, list):
        content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return content or ""


def ask(images: list[Image.Image], question: str, max_tokens: int = 120, kind: str = "scan") -> str:
    """Ask a model about one or more frames. Returns raw text."""
    if _openai():
        return _ask_openai(images, question, max_tokens, kind)
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
