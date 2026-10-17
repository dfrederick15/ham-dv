"""Analog ("SoftCast"-style) video transport: linear mapping of transform coefficients.

Instead of quantising and entropy coding, a group of pictures (GOP) is
3-D DCT transformed and the coefficients themselves are sent as the real and
imaginary parts of channel symbols.  The received picture quality then
degrades *gracefully* with the channel SNR - exactly like analog FM voice -
instead of falling off a digital cliff.

Mathematics
-----------
* GOP volume V (T x H x W) -> coefficients D = DCT3(V).
* D is divided into chunks (rectangular tiles of the coefficient plane,
  spanning all T temporal positions).  Chunk variance lambda_c is the only
  side information (sent digitally, entropy coded).
* Chunks are ranked by lambda_c; the top ones that fit the symbol budget are
  transmitted (the rest are set to zero at the decoder).
* Power allocation: scaling g_c minimising total distortion under a total
  power constraint gives the classical water-filling-free solution
        g_c = (P / sum_i sqrt(lambda_i))^(1/2) * lambda_c^(-1/4),
  i.e. every chunk is scaled so that its transmitted power is proportional
  to sqrt(lambda_c)  (Lagrangian of  min sum lambda_c / g_c^2  s.t.
  sum g_c^2 lambda_c = P).
* The scaled values are mixed by an orthonormal Hadamard matrix (64 x 64)
  so that the loss of any individual symbol or block is spread evenly over
  all coefficients, and every transmitted value is near Gaussian with equal
  power (good PAPR behaviour).
* Decoder: linear least-squares estimate (LLSE) of the coefficients
        x_hat = Lambda G^T (G Lambda G^T + Sigma)^-1 y,
  with G = Hadamard * diag(g), Lambda = diag(lambda) and Sigma the per-symbol
  noise variances reported by the modem (missing symbols: infinite variance).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .dsp import dct3d, idct3d, hadamard
from .videocodec import rgb_to_ycbcr420, ycbcr420_to_rgb

MIX = 64                         # Hadamard group size
CLIP = 3.5                       # clip transmitted values at +-CLIP sigma


@dataclass
class SoftConfig:
    width: int = 64
    height: int = 48
    gop: int = 4
    chunk: int = 8               # luma chunk size; chroma uses chunk/2

    def plane_size(self, plane: int) -> Tuple[int, int]:
        return (self.height, self.width) if plane == 0 else (self.height // 2, self.width // 2)

    def rects(self, plane: int) -> List[Tuple[int, int, int, int, int]]:
        """Chunk rectangles (t, y0, y1, x0, x1) of the coefficient volume of a plane.

        Each chunk spans ONE temporal index (so the high-energy temporal-DC
        plane never shares a chunk with the temporal-AC planes) and a square
        tile of spatial frequencies.  The tile containing the spatial DC is
        split dyadically down to single coefficients, because the variance of
        the lowest frequencies changes by orders of magnitude over a few
        coefficients; everywhere else the spectrum is smooth enough for
        uniform tiles.  Keeping the values inside a chunk near-Gaussian is
        what makes the fixed clipping threshold harmless.
        """
        h, w = self.plane_size(plane)
        base = self.chunk if plane == 0 else self.chunk // 2
        out = []
        for t in range(self.gop):
            for y0 in range(0, h, base):
                for x0 in range(0, w, base):
                    if y0 == 0 and x0 == 0:
                        out.extend(_dyadic(t, base))
                    else:
                        out.append((t, y0, y0 + base, x0, x0 + base))
        return out


def _dyadic(t: int, size: int) -> List[Tuple[int, int, int, int, int]]:
    """Split the [0,size)^2 tile: the corner recursively, the other three quadrants as blocks."""
    if size == 1:
        return [(t, 0, 1, 0, 1)]
    half = size // 2
    out = _dyadic(t, half)
    out.append((t, 0, half, half, size))
    out.append((t, half, size, 0, half))
    out.append((t, half, size, half, size))
    return out


def _chunks(coef: np.ndarray, rects) -> List[np.ndarray]:
    return [coef[t, y0:y1, x0:x1].reshape(-1) for (t, y0, y1, x0, x1) in rects]


def _unchunks(chunks: List[np.ndarray], rects, t: int, h: int, w: int) -> np.ndarray:
    out = np.zeros((t, h, w))
    for c, (ti, y0, y1, x0, x1) in zip(chunks, rects):
        out[ti, y0:y1, x0:x1] = c.reshape(y1 - y0, x1 - x0)
    return out


class SoftEncoder:
    def __init__(self, cfg: SoftConfig):
        self.cfg = cfg
        self.H = hadamard(MIX)
        self.gop_seq = 0

    def encode_gop(self, frames: List[np.ndarray], n_values: int):
        """frames: list of `gop` RGB frames. n_values: real values available.

        Returns (values: float array (<= n_values, unit average power), meta: dict)
        """
        cfg = self.cfg
        planes = [[], [], []]
        for f in frames:
            y, cb, cr = rgb_to_ycbcr420(f)
            planes[0].append(y.astype(np.float64))
            planes[1].append(cb.astype(np.float64))
            planes[2].append(cr.astype(np.float64))
        chunk_vals, lam, plane_of = [], [], []
        means = []
        for p in range(3):
            vol = np.stack(planes[p])
            m = int(np.clip(np.round(np.mean(vol)), 0, 255))   # plane mean is sent digitally
            means.append(m)
            d = dct3d(vol - m)
            c = _chunks(d, cfg.rects(p))
            chunk_vals.append(c)
            lam.append(np.array([np.mean(ci ** 2) for ci in c]))
            plane_of.append(np.full(len(c), p))
        lam_all = np.concatenate(lam)
        plane_all = np.concatenate(plane_of)
        sizes = np.concatenate([np.array([len(ci) for ci in cv]) for cv in chunk_vals])
        # quantise variances on a 3 dB log scale (this is the side information)
        lq = np.clip(np.round(np.log2(lam_all + 1e-3) * 2), 0, 63).astype(np.int64)
        lam_q = 2.0 ** (lq / 2.0)
        # select chunks by variance until the budget is exhausted (in Hadamard groups)
        order = np.argsort(-lam_q)
        budget = (n_values // MIX) * MIX
        selected = []
        used = 0
        for i in order:
            if used + sizes[i] > budget or lam_q[i] < 0.5:
                continue
            selected.append(int(i))
            used += int(sizes[i])
        selected.sort()
        sel_mask = np.zeros(len(lam_all), dtype=bool)
        sel_mask[selected] = True
        # power allocation g_c = (P / sum sqrt(lambda))^(1/2) lambda_c^(-1/4) with P = 1 per value
        if selected:
            lam_s = lam_q[selected]
            w_s = sizes[selected]
            s = np.sum(w_s * np.sqrt(lam_s)) / np.sum(w_s)
            g = (1.0 / s) ** 0.5 * lam_s ** (-0.25)
        values = []
        for j, i in enumerate(selected):
            p = plane_all[i]
            local = i - (0 if p == 0 else (len(lam[0]) if p == 1 else len(lam[0]) + len(lam[1])))
            values.append(chunk_vals[p][local] * g[j])
        if not values:
            return np.zeros(0), dict(lq=lq, mask=sel_mask, gop_seq=self.gop_seq, means=means, tdc=tdc_flags(cfg))
        v = np.concatenate(values)
        n_groups = len(v) // MIX
        # interleave so that every Hadamard group mixes values from many different
        # chunks (equalises the group variances -> near-Gaussian, equal-power output)
        v = v[: n_groups * MIX].reshape(MIX, n_groups).T
        mixed = (v @ self.H.T).reshape(-1)           # y = H x per group
        mixed = np.clip(mixed, -CLIP, CLIP)
        meta = dict(lq=lq, mask=sel_mask, gop_seq=self.gop_seq, means=means, tdc=tdc_flags(cfg))
        self.gop_seq = (self.gop_seq + 1) & 0xFF
        return mixed, meta


class SoftDecoder:
    def __init__(self, cfg: SoftConfig):
        self.cfg = cfg
        self.H = hadamard(MIX)

    def decode_gop(self, values: np.ndarray, noise_var: np.ndarray, meta: dict) -> List[np.ndarray]:
        """values: received real values (same order as sent), noise_var: per-value noise variance
        (np.inf for missing). Returns list of `gop` RGB frames."""
        cfg = self.cfg
        lq, mask = meta["lq"], meta["mask"]
        lam_q = 2.0 ** (lq / 2.0)
        sizes = []
        n_chunks = []
        rects = [cfg.rects(p) for p in range(3)]
        for p in range(3):
            sizes += [(y1 - y0) * (x1 - x0) for (_, y0, y1, x0, x1) in rects[p]]
            n_chunks.append(len(rects[p]))
        sizes = np.array(sizes)
        selected = np.nonzero(mask)[0]
        if len(selected) == 0:
            m = meta.get("means", [128, 128, 128])
            y8 = np.full((cfg.height, cfg.width), m[0], np.uint8)
            c8 = np.full((cfg.height // 2, cfg.width // 2), m[1], np.uint8)
            r8 = np.full((cfg.height // 2, cfg.width // 2), m[2], np.uint8)
            return [ycbcr420_to_rgb(y8, c8, r8)] * cfg.gop
        lam_s = lam_q[selected]
        w_s = sizes[selected]
        s = np.sum(w_s * np.sqrt(lam_s)) / np.sum(w_s)
        g = (1.0 / s) ** 0.5 * lam_s ** (-0.25)
        # per transmitted value: its chunk variance and scale
        lam_v = np.repeat(lam_s, w_s)
        g_v = np.repeat(g, w_s)
        n_total = len(lam_v)
        n_groups = n_total // MIX
        lam_v = lam_v[: n_groups * MIX].reshape(MIX, n_groups).T
        g_v = g_v[: n_groups * MIX].reshape(MIX, n_groups).T
        y = np.zeros(n_groups * MIX)
        nv = np.full(n_groups * MIX, np.inf)
        m = min(len(values), n_groups * MIX)
        y[:m] = values[:m]
        nv[:m] = noise_var[:m]
        y = y.reshape(n_groups, MIX)
        nv = nv.reshape(n_groups, MIX)
        x_hat = np.zeros_like(y)
        H = self.H
        for gi in range(n_groups):
            # LLSE: x = Lambda G^T (G Lambda G^T + Sigma)^-1 y, G = H diag(g)
            G = H * g_v[gi][None, :]
            lam_d = lam_v[gi]
            ok = np.isfinite(nv[gi])
            if not np.any(ok):
                continue
            Gk = G[ok]
            A = (Gk * lam_d[None, :]) @ Gk.T + np.diag(nv[gi][ok])
            rhs = np.linalg.solve(A, y[gi][ok])
            x_hat[gi] = lam_d * (Gk.T @ rhs)
        x_hat = x_hat.T.reshape(-1)          # undo the interleaving
        # scatter back into chunks
        out_planes = []
        pos = 0
        chunk_arrays = []
        for p in range(3):
            chunk_arrays.append([np.zeros((y1 - y0) * (x1 - x0)) for (_, y0, y1, x0, x1) in rects[p]])
        offs = [0, n_chunks[0], n_chunks[0] + n_chunks[1]]
        for j, i in enumerate(selected):
            p = 0 if i < offs[1] else (1 if i < offs[2] else 2)
            local = i - offs[p]
            size = sizes[i]
            if pos + size <= len(x_hat):
                chunk_arrays[p][local] = x_hat[pos: pos + size]
            pos += size
        frames_y, frames_cb, frames_cr = None, None, None
        recon = []
        for p in range(3):
            h, w = cfg.plane_size(p)
            d = _unchunks(chunk_arrays[p], rects[p], cfg.gop, h, w)
            recon.append(idct3d(d) + meta.get("means", [128, 128, 128])[p])
        frames = []
        for t in range(cfg.gop):
            y8 = np.clip(np.round(recon[0][t]), 0, 255).astype(np.uint8)
            cb8 = np.clip(np.round(recon[1][t]), 0, 255).astype(np.uint8)
            cr8 = np.clip(np.round(recon[2][t]), 0, 255).astype(np.uint8)
            frames.append(ycbcr420_to_rgb(y8, cb8, cr8))
        return frames


# ----------------------------------------------------------- side information
def pack_meta(meta: dict) -> bytes:
    """Entropy-code the chunk mask and the 6-bit log-variances of selected chunks."""
    from .rangecoder import Encoder, PROB_INIT
    enc = Encoder()
    probs = [PROB_INIT] * 80
    mask, lq = meta["mask"], meta["lq"]
    tdc = meta.get("tdc")
    prev = 1
    for i, m in enumerate(mask):
        ctx = 10 + prev * 2 + (1 if tdc is None or tdc[i] else 0)
        enc.encode(probs, ctx, 1 if m else 0)
        prev = 1 if m else 0
    prev_v = 20
    for i in np.nonzero(mask)[0]:
        d = int(lq[i]) - prev_v
        # signed delta: zero flag, sign, unary/eg magnitude
        enc.encode(probs, 2, 0 if d == 0 else 1)
        if d != 0:
            enc.encode_direct(1 if d < 0 else 0, 1)
            mag = abs(d) - 1
            for k in range(6):
                if mag == k:
                    enc.encode(probs, 3 + k, 0)
                    break
                enc.encode(probs, 3 + k, 1)
            else:
                enc.encode_direct(mag - 6, 6)
        prev_v = int(lq[i])
    means = meta.get("means", [128, 128, 128])
    return bytes([meta["gop_seq"] & 0xFF, len(mask) & 0xFF, len(mask) >> 8] + [int(m) & 0xFF for m in means]) + enc.finish()


def tdc_flags(cfg: SoftConfig) -> np.ndarray:
    """Per chunk: True if it belongs to the temporal-DC plane (t == 0) of its colour plane."""
    out = []
    for p in range(3):
        out += [r[0] == 0 for r in cfg.rects(p)]
    return np.array(out, dtype=bool)


def unpack_meta(data: bytes, cfg: Optional[SoftConfig] = None) -> Optional[dict]:
    from .rangecoder import Decoder, PROB_INIT
    if len(data) < 7:
        return None
    tdc = tdc_flags(cfg) if cfg is not None else None
    gop_seq = data[0]
    n = data[1] | (data[2] << 8)
    means = [int(data[3]), int(data[4]), int(data[5])]
    dec = Decoder(data[6:])
    probs = [PROB_INIT] * 80
    mask = np.zeros(n, dtype=bool)
    if tdc is not None and len(tdc) != n:
        return None
    prev = 1
    for i in range(n):
        ctx = 10 + prev * 2 + (1 if tdc is None or tdc[i] else 0)
        b = dec.decode(probs, ctx)
        mask[i] = bool(b)
        prev = b
    lq = np.zeros(n, dtype=np.int64)
    prev_v = 20
    for i in np.nonzero(mask)[0]:
        if dec.decode(probs, 2) == 0:
            d = 0
        else:
            sign = dec.decode_direct(1)
            mag = 0
            for k in range(6):
                if dec.decode(probs, 3 + k) == 0:
                    mag = k
                    break
            else:
                mag = dec.decode_direct(6) + 6
            d = -(mag + 1) if sign else (mag + 1)
        prev_v += d
        lq[i] = prev_v
    if dec.overrun():
        return None
    return dict(lq=lq, mask=mask, gop_seq=gop_seq, means=means)
