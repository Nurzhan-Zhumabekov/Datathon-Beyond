"""Re-encode the training videos with a changed viewpoint, to measure how much the service depends on the camera angle.

The final test videos are shot from a different angle. We cannot film them, so this makes two proxies of the 40 open
videos (the labels stay valid because the timing does not change):
  hflip  mirror image
  angle  keystone (as if the camera looked from another height) + 5 degree tilt + 10% zoom
Output goes to <out>/<variant>/<same file name>. Nothing is uploaded anywhere.

usage: python scripts/make_augmented.py <videos dir> <out dir> [hflip angle]
then:  python scripts/jury_sim.py --videos <out dir>/angle --labels <xlsx>   (same labels, same service)
"""
import sys
from pathlib import Path

import av
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.video import _rotate  # noqa: E402  (same upright orientation the service uses)


def perspective_coeffs(src_quad, dst_quad):
    """PIL wants, for every OUTPUT point, where to read the INPUT: solve the 8 coefficients."""
    rows, rhs = [], []
    for (x, y), (u, v) in zip(dst_quad, src_quad):
        rows.append([x, y, 1, 0, 0, 0, -x * u, -y * u]); rhs.append(u)
        rows.append([0, 0, 0, x, y, 1, -x * v, -y * v]); rhs.append(v)
    return np.linalg.solve(np.array(rows, dtype=float), np.array(rhs, dtype=float))


def transform(img: Image.Image, variant: str) -> Image.Image:
    w, h = img.size
    if variant == "hflip":
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    if variant == "angle":
        k = 0.14 * w
        coeffs = perspective_coeffs([(k, 0), (w - k, 0), (w, h), (0, h)], [(0, 0), (w, 0), (w, h), (0, h)])
        img = img.transform((w, h), Image.PERSPECTIVE, tuple(coeffs), Image.BICUBIC)
        img = img.rotate(5, resample=Image.BICUBIC)
        cw, ch = int(w / 1.10), int(h / 1.10)
        left, top = (w - cw) // 2, (h - ch) // 2
        return img.crop((left, top, left + cw, top + ch)).resize((w, h), Image.BICUBIC)
    raise ValueError(variant)


def convert(src: Path, dst: Path, variant: str) -> None:
    with av.open(str(src)) as c:
        s = c.streams.video[0]
        fps = s.average_rate or 30
        dst.parent.mkdir(parents=True, exist_ok=True)
        with av.open(str(dst), "w", format="mov") as out:
            o = out.add_stream("libx264", rate=fps)
            o.pix_fmt = "yuv420p"
            o.options = {"crf": "20", "preset": "veryfast"}
            for frame in c.decode(s):
                img = transform(_rotate(frame.to_image(), frame), variant).convert("RGB")
                if not o.width:
                    o.width, o.height = img.width // 2 * 2, img.height // 2 * 2
                img = img.crop((0, 0, o.width, o.height))
                for pkt in o.encode(av.VideoFrame.from_image(img)):
                    out.mux(pkt)
            for pkt in o.encode():
                out.mux(pkt)


if __name__ == "__main__":
    videos, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    variants = sys.argv[3:] or ["hflip", "angle"]
    files = sorted(videos.glob("*.mov"))
    for variant in variants:
        for i, p in enumerate(files, 1):
            convert(p, out_dir / variant / p.name, variant)
            print(f"{variant}: {i}/{len(files)} {p.name}", flush=True)
