"""Gray-mapped QAM constellations and max-log LLR computation."""
from __future__ import annotations

from functools import lru_cache

import numpy as np


def _gray(n: int) -> np.ndarray:
    return np.array([i ^ (i >> 1) for i in range(n)])


@lru_cache(maxsize=None)
def constellation(bits_per_symbol: int):
    """Return (points complex array of size 2^b, bit labels array (2^b, b)).

    Square QAM with independent Gray coding on I and Q, normalised to unit
    average power.  b = 2 (QPSK), 4 (16-QAM), 6 (64-QAM).
    """
    b = bits_per_symbol
    if b % 2:
        raise ValueError("only square QAM supported")
    m = 1 << (b // 2)
    levels = np.arange(m) * 2 - (m - 1)   # -(m-1) ... (m-1)
    gray = _gray(m)
    pts = []
    labels = []
    for ii, li in enumerate(levels):
        for qq, lq in enumerate(levels):
            pts.append(li + 1j * lq)
            lab = (int(gray[ii]) << (b // 2)) | int(gray[qq])
            labels.append([(lab >> (b - 1 - j)) & 1 for j in range(b)])
    pts = np.array(pts, dtype=np.complex128)
    pts /= np.sqrt(np.mean(np.abs(pts) ** 2))
    return pts, np.array(labels, dtype=np.uint8)


def map_bits(bits: np.ndarray, bits_per_symbol: int) -> np.ndarray:
    pts, labels = constellation(bits_per_symbol)
    b = bits_per_symbol
    bits = np.asarray(bits, dtype=np.int64).reshape(-1, b)
    weights = 1 << np.arange(b - 1, -1, -1)
    idx = bits @ weights
    # labels are in index order by construction? build lookup
    lut = np.zeros(1 << b, dtype=np.int64)
    lab_idx = labels.astype(np.int64) @ weights
    lut[lab_idx] = np.arange(1 << b)
    return pts[lut[idx]]


def llrs(sym: np.ndarray, bits_per_symbol: int, noise_var: float) -> np.ndarray:
    """Max-log LLRs (positive favours bit 0) for received symbols.

    LLR_j = (min_{s: b_j(s)=1} |y-s|^2 - min_{s: b_j(s)=0} |y-s|^2) / sigma^2
    """
    pts, labels = constellation(bits_per_symbol)
    d2 = np.abs(sym[:, None] - pts[None, :]) ** 2    # (n, M)
    out = np.empty((len(sym), bits_per_symbol))
    for j in range(bits_per_symbol):
        mask0 = labels[:, j] == 0
        d0 = np.min(d2[:, mask0], axis=1)
        d1 = np.min(d2[:, ~mask0], axis=1)
        out[:, j] = (d1 - d0) / max(noise_var, 1e-6)
    return out.reshape(-1)


def hard_decide(sym: np.ndarray, bits_per_symbol: int) -> np.ndarray:
    pts, labels = constellation(bits_per_symbol)
    idx = np.argmin(np.abs(sym[:, None] - pts[None, :]) ** 2, axis=1)
    return labels[idx].reshape(-1)
