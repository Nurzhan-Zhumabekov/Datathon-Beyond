"""Video decoding with real presentation timestamps (videos may have variable frame rate)."""
import io
from dataclasses import dataclass
from pathlib import Path

import av
from PIL import Image


@dataclass
class Frame:
    t: float          # seconds from start of video (from PTS, not frame_index / fps)
    image: Image.Image


def probe(path: Path) -> dict:
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        duration = float(s.duration * s.time_base) if s.duration else float(c.duration or 0) / av.time_base
        return {
            "duration": duration,
            "fps": float(s.average_rate or 0),
            "width": s.codec_context.width,
            "height": s.codec_context.height,
        }


def _rotate(img: Image.Image, frame) -> Image.Image:
    rot = getattr(frame, "rotation", 0) or 0
    if rot % 360:
        img = img.rotate(rot, expand=True)
    return img


def sample_frames(path: Path, fps: float = 2.0) -> list[Frame]:
    """Decode the whole video once and keep the frame closest to every 1/fps step."""
    step = 1.0 / fps
    out: list[Frame] = []
    next_t = 0.0
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        s.thread_type = "AUTO"
        start = float(s.start_time * s.time_base) if s.start_time is not None else 0.0
        for frame in c.decode(s):
            if frame.pts is None:
                continue
            t = float(frame.pts * s.time_base) - start
            if t + 1e-6 >= next_t:
                out.append(Frame(round(t, 3), _rotate(frame.to_image(), frame)))
                next_t += step
                while next_t <= t:
                    next_t += step
    return out


def to_jpeg_bytes(img: Image.Image, max_side: int = 960, quality: int = 85) -> bytes:
    im = img.copy()
    im.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    im.convert("RGB").save(buf, format="JPEG", quality=quality)
    return buf.getvalue()
