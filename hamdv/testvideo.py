"""Synthetic test video: a moving 'operator' silhouette with callsign-like bars and noise-free background."""
from __future__ import annotations

import numpy as np


def synth_frame(t: float, width: int = 96, height: int = 72, seed: int = 0) -> np.ndarray:
    """Return an RGB uint8 frame at time t (seconds): a shaded head that sways, a moving bar, static background."""
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float64)
    img = np.zeros((height, width, 3), dtype=np.float64)
    # background gradient + a few static "shack" objects
    img[..., 0] = 60 + 40 * xx / width
    img[..., 1] = 70 + 30 * yy / height
    img[..., 2] = 90 + 20 * (xx + yy) / (width + height)
    img[int(0.1 * height):int(0.3 * height), int(0.65 * width):int(0.95 * width)] = (40, 40, 45)   # a shelf
    for k in range(4):
        x0 = int((0.68 + 0.06 * k) * width)
        img[int(0.12 * height):int(0.28 * height), x0:x0 + 2] = (200, 180, 60)                     # books
    # head: ellipse that sways and nods
    cx = width * (0.45 + 0.06 * np.sin(2 * np.pi * 0.25 * t))
    cy = height * (0.5 + 0.03 * np.sin(2 * np.pi * 0.4 * t + 1))
    rx, ry = width * 0.14, height * 0.26
    d = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2
    face = d < 1
    shade = np.clip(1.0 - 0.5 * d, 0.3, 1.0)
    img[face] = np.stack([220 * shade[face], 180 * shade[face], 150 * shade[face]], -1)
    # eyes and mouth (mouth opens/closes while "talking")
    for ex in (-0.4, 0.4):
        eye = ((xx - (cx + ex * rx)) ** 2 + (yy - (cy - 0.25 * ry)) ** 2) < (0.08 * rx * ry)
        img[eye] = (30, 30, 40)
    mouth_h = 0.08 + 0.06 * (0.5 + 0.5 * np.sin(2 * np.pi * 2.0 * t))
    mouth = (np.abs(xx - cx) < 0.35 * rx) & (np.abs(yy - (cy + 0.45 * ry)) < mouth_h * ry)
    img[mouth] = (120, 40, 50)
    # body / shoulders
    body = (yy > cy + 0.9 * ry) & (np.abs(xx - cx) < rx * 2.2)
    img[body] = (50, 70, 120)
    # a moving caption bar (tests motion compensation)
    bx = int((t * 12) % (width + 20)) - 10
    img[height - 8:height - 3, max(bx, 0):max(min(bx + 14, width), 0)] = (240, 240, 240)
    return np.clip(img, 0, 255).astype(np.uint8)
