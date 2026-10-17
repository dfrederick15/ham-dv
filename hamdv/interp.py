"""Motion-compensated frame interpolation for smooth display.

The link carries 2-4 real frames per second.  The viewer renders 30 frames
per second by warping the two real frames that bracket the display time
towards each other and blending:

    F_t(p) = (1-t) F0(p + t v(p)) + t F1(p - (1-t) v(p)),   0 <= t <= 1,

where v(p) is the displacement field with F1(p) ~ F0(p + v(p)).  For the
digital codec v comes for free from the decoded macroblock motion vectors
(bilinearly upsampled to a dense field); for the analog transport a global
translation is estimated with the least-squares (Lucas-Kanade) solver.  No
extra bits are transmitted for this.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy.ndimage import map_coordinates

from .videocodec import global_motion_ls


def dense_field(mvs: np.ndarray, modes: np.ndarray, mbw: int, mbh: int, h: int, w: int) -> np.ndarray:
    """Macroblock vectors (n_mb, 2) [dx, dy] -> dense (2, h, w) field [dy, dx], bilinearly upsampled.

    INTRA macroblocks have no vector; they borrow the median of their neighbours.
    """
    fx = mvs[:, 0].astype(np.float64).reshape(mbh, mbw)
    fy = mvs[:, 1].astype(np.float64).reshape(mbh, mbw)
    intra = (modes == 2).reshape(mbh, mbw)
    if intra.any() and not intra.all():
        for (r, c) in zip(*np.nonzero(intra)):
            ys, xs = [], []
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < mbh and 0 <= cc < mbw and not intra[rr, cc]:
                        ys.append(fy[rr, cc])
                        xs.append(fx[rr, cc])
            if ys:
                fy[r, c], fx[r, c] = np.median(ys), np.median(xs)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    coords = [(yy + 0.5) / (h / mbh) - 0.5, (xx + 0.5) / (w / mbw) - 0.5]
    return np.stack([map_coordinates(fy, coords, order=1, mode="nearest"),
                     map_coordinates(fx, coords, order=1, mode="nearest")])


def global_field(f0: np.ndarray, f1: np.ndarray) -> np.ndarray:
    """Dense field from a least-squares global translation between two RGB frames."""
    y0 = f0.astype(np.float64).mean(axis=2)
    y1 = f1.astype(np.float64).mean(axis=2)
    u, v = global_motion_ls(y1, y0)      # f1(p) ~ f0(p + (u, v))
    h, w = y0.shape
    return np.stack([np.full((h, w), v), np.full((h, w), u)])


def interpolate(f0: np.ndarray, f1: np.ndarray, t: float, field: Optional[np.ndarray] = None) -> np.ndarray:
    """Frame at fraction t between f0 (t=0) and f1 (t=1), RGB uint8 (h, w, 3)."""
    if t <= 0:
        return f0
    if t >= 1:
        return f1
    h, w = f0.shape[:2]
    if field is None:
        return np.clip(np.round((1 - t) * f0.astype(np.float64) + t * f1.astype(np.float64)), 0, 255).astype(np.uint8)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    c0 = [yy + t * field[0], xx + t * field[1]]
    c1 = [yy - (1 - t) * field[0], xx - (1 - t) * field[1]]
    out = np.empty((h, w, 3))
    for ch in range(3):
        a = map_coordinates(f0[..., ch].astype(np.float64), c0, order=1, mode="nearest")
        b = map_coordinates(f1[..., ch].astype(np.float64), c1, order=1, mode="nearest")
        out[..., ch] = (1 - t) * a + t * b
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


def smooth_sequence(frames, fields, fps_in: float, fps_out: float = 30.0):
    """Offline: expand a list of real frames (with per-frame fields relative to the previous frame)
    into an fps_out sequence."""
    out = []
    if not frames:
        return out
    step = fps_in / fps_out
    pos = 0.0
    last = len(frames) - 1
    while pos <= last + 1e-9:
        i = int(np.floor(pos))
        t = pos - i
        if i >= last:
            out.append(frames[last])
        else:
            fld = fields[i + 1] if fields[i + 1] is not None else global_field(frames[i], frames[i + 1])
            out.append(interpolate(frames[i], frames[i + 1], t, fld))
        pos += step
    return out
