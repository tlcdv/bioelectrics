"""Build the bioelectrics emblem: a two-headed planarian at the centre of the project emblem.

The two-headed planarian is the best-known result of developmental bioelectricity:
Levin's laboratory changed a flatworm's bioelectric state for a few hours, and it
regenerated a head at both ends. The planarian illustration keeps its dark halo
and the contour lines drawn around it, which read as the field around the body,
and loses only its flat brown background and the measurement lines outside the
halo. It is set at the centre of the project emblem (assets/logo.png), whose rim,
spikes and starfield stay.

Run:    python scripts/build_emblem.py <planarian image>
Output: assets/emblem.png (transparent background)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage

ASSETS = Path(__file__).resolve().parent.parent / "assets"


def planarian_cutout(path: str) -> Image.Image:
    """The planarian and its dark halo, on a transparent background."""
    im = Image.open(path).convert("RGB")
    a = np.asarray(im).astype(float)
    # The background is one flat brown. Estimate it from the border, then keep
    # every pixel that clearly differs from it: the dark halo, the contour lines
    # and the pale body alike. Thresholding on darkness alone lost the body,
    # which is lighter than the background and whose tail touches the edge.
    border = np.concatenate([a[:6].reshape(-1, 3), a[-6:].reshape(-1, 3),
                             a[:, :6].reshape(-1, 3), a[:, -6:].reshape(-1, 3)])
    bg = np.median(border, axis=0)
    # Background is the brown that is connected to the image border. The body's
    # interior is a mid-brown close to the background colour too, but the body
    # outline separates it from the border, so it is not reached and stays.
    near_bg = np.linalg.norm(a - bg, axis=-1) < 30
    lab_bg, _ = ndimage.label(near_bg)
    edge_labels = np.unique(np.concatenate([lab_bg[0], lab_bg[-1], lab_bg[:, 0], lab_bg[:, -1]]))
    background = np.isin(lab_bg, edge_labels[edge_labels > 0])
    foreground = ~background
    # An opening removes the thin measurement lines drawn on the background,
    # which are a few pixels wide; the halo and body are far wider.
    solid = ndimage.binary_opening(foreground, structure=np.ones((7, 7)))
    solid = ndimage.binary_closing(solid, structure=np.ones((9, 9)))
    lab, n = ndimage.label(solid)
    sizes = ndimage.sum(np.ones_like(lab), lab, index=range(1, n + 1))
    body = lab == (int(np.argmax(sizes)) + 1)     # the halo and body, nothing else
    body = ndimage.binary_fill_holes(body)
    body = ndimage.binary_dilation(body, iterations=2)  # recover the edge the opening took
    alpha = Image.fromarray((body * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(2.5))
    # The halo's fill is a muddy brown, lighter than the emblem's disc, so it read
    # as a pasted patch. Darken only the dark fill toward the disc; the pale body
    # and the contour lines are brighter and keep their full strength.
    lum = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
    keep = np.clip((lum - 35.0) / 55.0, 0.0, 1.0)
    a = a * (0.40 + 0.60 * keep)[..., None]
    im = Image.fromarray(a.clip(0, 255).astype(np.uint8), "RGB")
    out = im.convert("RGBA")
    out.putalpha(alpha)
    ys, xs = np.nonzero(body)
    return out.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))


def build(planarian_path: str) -> Path:
    base = Image.open(ASSETS / "logo.png").convert("RGBA")
    # Work at a size where the planarian is not enlarged: its source is small.
    scale = 0.5
    base = base.resize((round(base.width * scale), round(base.height * scale)), Image.LANCZOS)
    cx, cy = 596 * scale, 517 * scale          # disc centre in logo.png coordinates
    rim = 490 * scale                          # disc radius

    p = planarian_cutout(planarian_path)
    target_h = 1.62 * rim                      # the halo spans most of the disc height
    k = target_h / p.height
    p = p.resize((round(p.width * k), round(p.height * k)), Image.LANCZOS)

    # Darken the emblem inside the disc, so the planarian is the subject and the
    # rings and rays recede to a texture behind it.
    a = np.asarray(base).astype(float)
    yy, xx = np.indices(a.shape[:2])
    r = np.hypot(xx - cx, yy - cy)
    inside = r < rim * 0.965
    fade = np.clip((rim * 0.965 - r) / (rim * 0.12), 0, 1) * inside
    a[..., :3] *= (1 - 0.72 * fade)[..., None]
    base = Image.fromarray(a.astype(np.uint8), "RGBA")

    base.alpha_composite(p, (round(cx - p.width / 2), round(cy - p.height / 2)))
    out = ASSETS / "emblem.png"
    base.save(out)
    return out


if __name__ == "__main__":
    print("wrote", build(sys.argv[1]))
