"""Low-bit-rate hybrid video codec for radio links ("HamVid").

Design goals: 1-6 kbit/s, 1-5 frames/s, small frames (64x48 ... 160x120),
no key frames (receivers can join at any time and bursts of loss heal
themselves), bit-exact integer reconstruction on every platform.

Structure
---------
* YCbCr 4:2:0.  A macroblock (MB) is an 8x8 luma block with two 4x4 chroma
  blocks.
* Three MB modes: SKIP (copy co-located reference), INTER (integer-pel
  motion vector + coded residual), INTRA (DC prediction from reconstructed
  neighbours + coded residual).
* Transform: HEVC-style integer DCT cores (8x8 for luma, 4x4 for chroma).
  Forward:  Y = C X C^T.  Inverse:  X = (C^T Y C + rounding) >> shift.
  The cores are scaled orthogonal matrices (C C^T ~= 2^15 I for 8x8,
  2^14 I for 4x4) so the inverse is an exact integer operation.
* Quantisation step  delta = 2^(q/4)  (q = 0..47), dead-zone rounding, flat
  matrix for luma and 1.25x for chroma.
* Entropy coding: context-adaptive binary arithmetic coding of modes,
  motion vector differences (median predictor) and zig-zag scanned
  coefficients (significance / greater-than-1 / greater-than-2 flags with
  position and plane dependent contexts, Exp-Golomb bypass remainders).
* Motion estimation: least-squares global motion (Lucas-Kanade normal
  equations) seeds a +-4 pel vectorised block search with a Lagrangian
  cost J = SAD + lambda * bits(mv).
* Mode decision: Lagrangian J = SSE + lambda(q) * bits.
* Intra refresh: every frame forces a fixed number of MBs to INTRA, walking
  the MB array with a stride derived from the golden ratio so that refresh
  blocks are spread evenly in space and time; a lost frame is healed within
  one refresh period.  The decoder repeats the last frame on loss.
* Rate control: multiplicative QP update towards the bit budget supplied by
  the link layer plus one re-encode when the budget is badly exceeded.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from .rangecoder import Encoder, Decoder, PROB_INIT, encode_unary_eg, decode_unary_eg

# ---------------------------------------------------------------- transforms
C8 = np.array([
    [64, 64, 64, 64, 64, 64, 64, 64],
    [89, 75, 50, 18, -18, -50, -75, -89],
    [83, 36, -36, -83, -83, -36, 36, 83],
    [75, -18, -89, -50, 50, 89, 18, -75],
    [64, -64, -64, 64, 64, -64, -64, 64],
    [50, -89, 18, 75, -75, -18, 89, -50],
    [36, -83, 83, -36, -36, 83, -83, 36],
    [18, -50, 75, -89, 89, -75, 50, -18]], dtype=np.int64)
C4 = np.array([
    [64, 64, 64, 64],
    [83, 36, -36, -83],
    [64, -64, -64, 64],
    [36, -83, 83, -36]], dtype=np.int64)

# zig-zag scan orders
def _zigzag(n: int) -> np.ndarray:
    order = sorted(((i + j, j if (i + j) % 2 else i, i, j) for i in range(n) for j in range(n)))
    return np.array([i * n + j for (_, _, i, j) in order], dtype=np.int64)


ZZ8 = _zigzag(8)
ZZ4 = _zigzag(4)


def fdct_blocks(blocks: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Forward integer DCT of (n, k, k) blocks -> coefficient domain scaled to orthonormal units (float)."""
    k = c.shape[0]
    y = np.einsum('ij,njk,lk->nil', c, blocks.astype(np.int64), c)
    return y / float(1 << (15 if k == 8 else 14))


def idct_blocks_int(coef8: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Exact integer inverse: coef8 holds coefficients in 1/8 units (int64). Returns int64 residuals."""
    k = c.shape[0]
    shift = (15 if k == 8 else 14) + 3
    y = np.einsum('ji,njk,kl->nil', c, coef8.astype(np.int64), c)   # C^T Y C
    return (y + (1 << (shift - 1))) >> shift


def qstep8(q: int) -> int:
    """Quantiser step in 1/8 pixel units for quality index q."""
    return int(round(8.0 * 2.0 ** (q / 4.0)))


# ---------------------------------------------------------------- contexts
class Ctx:
    """Context layout for the arithmetic coder (shared by encoder and decoder)."""
    SKIP = 0          # 3 contexts (number of skipped neighbours)
    INTRA = 3         # 3
    MVD_ZERO = 6      # 2 (x / y)
    MVD_MAG = 8       # 2 * 6 unary contexts
    CBP = 20          # 3 planes * 2 (neighbour coded?)
    SIG = 26          # plane(2) * 8 position buckets
    GT1 = 42          # plane(2) * 3
    GT2 = 48          # plane(2) * 3
    LAST = 54         # plane(2) * 8
    TOTAL = 70

    @staticmethod
    def bucket(i: int) -> int:
        if i < 3:
            return i
        if i < 6:
            return 3
        if i < 10:
            return 4
        if i < 20:
            return 5
        if i < 40:
            return 6
        return 7


def new_probs() -> list:
    return [PROB_INIT] * Ctx.TOTAL


# ---------------------------------------------------------------- coefficient coding
def encode_block(enc: Encoder, probs: list, q: np.ndarray, zz: np.ndarray, plane: int):
    """Encode quantised block q (flattened k*k ints) in zig-zag order."""
    coefs = q[zz]
    nz = np.nonzero(coefs)[0]
    if len(nz) == 0:
        raise ValueError("encode_block called with empty block")
    last = int(nz[-1])
    n_gt1 = 0
    for i in range(last + 1):
        v = int(coefs[i])
        sig_ctx = Ctx.SIG + plane * 8 + Ctx.bucket(i)
        if v == 0:
            enc.encode(probs, sig_ctx, 0)
            continue
        enc.encode(probs, sig_ctx, 1)
        mag = abs(v)
        g1 = Ctx.GT1 + plane * 3 + min(n_gt1, 2)
        if mag == 1:
            enc.encode(probs, g1, 0)
        else:
            enc.encode(probs, g1, 1)
            n_gt1 += 1
            g2 = Ctx.GT2 + plane * 3 + min(n_gt1 - 1, 2)
            if mag == 2:
                enc.encode(probs, g2, 0)
            else:
                enc.encode(probs, g2, 1)
                r = mag - 3
                v1 = r + 1
                n = v1.bit_length() - 1
                for _ in range(n):
                    enc.encode_direct(1, 1)
                enc.encode_direct(0, 1)
                enc.encode_direct(v1 & ((1 << n) - 1), n)
        enc.encode_direct(1 if v < 0 else 0, 1)
        last_ctx = Ctx.LAST + plane * 8 + Ctx.bucket(i)
        enc.encode(probs, last_ctx, 1 if i == last else 0)


def decode_block(dec: Decoder, probs: list, k: int, zz: np.ndarray, plane: int) -> np.ndarray:
    out = np.zeros(k * k, dtype=np.int64)
    n_gt1 = 0
    for i in range(k * k):
        sig_ctx = Ctx.SIG + plane * 8 + Ctx.bucket(i)
        if dec.decode(probs, sig_ctx) == 0:
            continue
        g1 = Ctx.GT1 + plane * 3 + min(n_gt1, 2)
        if dec.decode(probs, g1) == 0:
            mag = 1
        else:
            n_gt1 += 1
            g2 = Ctx.GT2 + plane * 3 + min(n_gt1 - 1, 2)
            if dec.decode(probs, g2) == 0:
                mag = 2
            else:
                n = 0
                while dec.decode_direct(1) == 1 and n < 20:
                    n += 1
                v1 = (1 << n) | dec.decode_direct(n)
                mag = v1 - 1 + 3
        sign = dec.decode_direct(1)
        out[zz[i]] = -mag if sign else mag
        last_ctx = Ctx.LAST + plane * 8 + Ctx.bucket(i)
        if dec.decode(probs, last_ctx) == 1:
            break
    return out


# ---------------------------------------------------------------- helpers
def rgb_to_ycbcr420(rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rgb = rgb.astype(np.float64)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 128 - 0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 128 + 0.5 * r - 0.418688 * g - 0.081312 * b
    def sub(c):
        return (c[0::2, 0::2] + c[1::2, 0::2] + c[0::2, 1::2] + c[1::2, 1::2]) / 4.0
    return (np.clip(np.round(y), 0, 255).astype(np.uint8),
            np.clip(np.round(sub(cb)), 0, 255).astype(np.uint8),
            np.clip(np.round(sub(cr)), 0, 255).astype(np.uint8))


def ycbcr420_to_rgb(y: np.ndarray, cb: np.ndarray, cr: np.ndarray) -> np.ndarray:
    cbu = np.repeat(np.repeat(cb.astype(np.float64), 2, 0), 2, 1) - 128
    cru = np.repeat(np.repeat(cr.astype(np.float64), 2, 0), 2, 1) - 128
    yy = y.astype(np.float64)
    r = yy + 1.402 * cru
    g = yy - 0.344136 * cbu - 0.714136 * cru
    b = yy + 1.772 * cbu
    return np.clip(np.round(np.stack([r, g, b], -1)), 0, 255).astype(np.uint8)


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 10 * np.log10(255.0 ** 2 / max(mse, 1e-9))


def blockify(img: np.ndarray, k: int) -> np.ndarray:
    h, w = img.shape
    return img.reshape(h // k, k, w // k, k).transpose(0, 2, 1, 3).reshape(-1, k, k)


def unblockify(blocks: np.ndarray, h: int, w: int, k: int) -> np.ndarray:
    return blocks.reshape(h // k, w // k, k, k).transpose(0, 2, 1, 3).reshape(h, w)


def shift_image(img: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """Return img sampled at (x+dx, y+dy) with edge clamping."""
    h, w = img.shape
    ys = np.clip(np.arange(h) + dy, 0, h - 1)
    xs = np.clip(np.arange(w) + dx, 0, w - 1)
    return img[ys][:, xs]


def mc_block(ref: np.ndarray, y0: int, x0: int, k: int, dx: int, dy: int) -> np.ndarray:
    """Motion-compensated k x k block from ref at (y0+dy, x0+dx), clamped."""
    h, w = ref.shape
    ys = np.clip(np.arange(y0 + dy, y0 + dy + k), 0, h - 1)
    xs = np.clip(np.arange(x0 + dx, x0 + dx + k), 0, w - 1)
    return ref[ys][:, xs]


def golden_order(n: int) -> np.ndarray:
    """Visit order of n items with stride ~ n/phi (coprime with n): evenly spread refresh pattern."""
    phi = (1 + 5 ** 0.5) / 2
    g = max(1, int(round(n / phi)))
    while np.gcd(g, n) != 1:
        g += 1
    return (np.arange(n) * g) % n


def global_motion_ls(cur: np.ndarray, ref: np.ndarray, iters: int = 3) -> Tuple[float, float]:
    """Lucas-Kanade global translation: solve the 2x2 normal equations
        [sum Ix^2   sum IxIy] [u]   = -[sum Ix It]
        [sum IxIy   sum Iy^2] [v]      [sum Iy It]
    iterated with integer re-warping of the reference."""
    u = v = 0.0
    cur = cur.astype(np.float64)
    reff = ref.astype(np.float64)
    for _ in range(iters):
        w = shift_image(reff, int(round(u)), int(round(v)))
        ix = np.zeros_like(w)
        iy = np.zeros_like(w)
        ix[:, 1:-1] = (w[:, 2:] - w[:, :-2]) / 2
        iy[1:-1, :] = (w[2:, :] - w[:-2, :]) / 2
        it = cur - w
        a = np.array([[np.sum(ix * ix), np.sum(ix * iy)], [np.sum(ix * iy), np.sum(iy * iy)]])
        b = -np.array([np.sum(ix * it), np.sum(iy * it)])
        if np.linalg.cond(a) > 1e8:
            break
        du, dv = np.linalg.solve(a, b)
        u += du
        v += dv
        if abs(du) < 0.05 and abs(dv) < 0.05:
            break
    return float(np.clip(u, -8, 8)), float(np.clip(v, -8, 8))


# ---------------------------------------------------------------- codec core (shared)
@dataclass
class FrameHeader:
    frame_num: int
    q: int
    width: int
    height: int


class _State:
    def __init__(self, width: int, height: int):
        if width % 8 or height % 8:
            raise ValueError("width and height must be multiples of 8")
        self.w, self.h = width, height
        self.mbw, self.mbh = width // 8, height // 8
        self.n_mb = self.mbw * self.mbh
        self.y = np.full((height, width), 128, dtype=np.uint8)
        self.cb = np.full((height // 2, width // 2), 128, dtype=np.uint8)
        self.cr = np.full((height // 2, width // 2), 128, dtype=np.uint8)
        self.frame_count = 0


def _mv_pred(mvs: np.ndarray, i: int, mbw: int) -> Tuple[int, int]:
    """Median of left, top and top-right motion vectors (unavailable -> 0)."""
    r, c = divmod(i, mbw)
    cands = []
    cands.append(mvs[i - 1] if c > 0 else np.zeros(2, np.int64))
    cands.append(mvs[i - mbw] if r > 0 else np.zeros(2, np.int64))
    cands.append(mvs[i - mbw + 1] if (r > 0 and c < mbw - 1) else np.zeros(2, np.int64))
    c = np.array(cands)
    return int(np.median(c[:, 0])), int(np.median(c[:, 1]))


def _dc_pred(plane: np.ndarray, y0: int, x0: int, k: int) -> int:
    vals = []
    if y0 > 0:
        vals.append(plane[y0 - 1, x0: x0 + k].astype(np.int64))
    if x0 > 0:
        vals.append(plane[y0: y0 + k, x0 - 1].astype(np.int64))
    if not vals:
        return 128
    v = np.concatenate(vals)
    return int((np.sum(v) + len(v) // 2) // len(v))


CHROMA_NUM, CHROMA_DEN = 5, 8     # chroma step = 5/8 of the luma step (4x4 blocks have half the DC gain of 8x8)


def _quant(coef: np.ndarray, step8: int, chroma: bool) -> np.ndarray:
    """Dead-zone quantiser on orthonormal-scale coefficients (float) -> int levels."""
    step = step8 / 8.0 * (CHROMA_NUM / CHROMA_DEN if chroma else 1.0)
    rnd = np.full(coef.shape, 1.0 / 3.0)
    rnd.reshape(coef.shape[0], -1)[:, 0] = 0.5   # DC
    q = np.sign(coef) * np.floor(np.abs(coef) / step + rnd)
    return q.astype(np.int64)


def _dequant8(levels: np.ndarray, step8: int, chroma: bool) -> np.ndarray:
    s = (step8 * CHROMA_NUM + CHROMA_DEN // 2) // CHROMA_DEN if chroma else step8
    return levels.astype(np.int64) * s


# ---------------------------------------------------------------- encoder
@dataclass
class EncodeStats:
    bits: int = 0
    q: int = 0
    n_skip: int = 0
    n_inter: int = 0
    n_intra: int = 0
    psnr: float = 0.0


class VideoEncoder:
    def __init__(self, width: int, height: int, refresh_period: int = 12, q: int = 24,
                 q_min: int = 8, q_max: int = 44, search: int = 4):
        self.st = _State(width, height)
        self.refresh_period = max(1, refresh_period)
        self.refresh_order = golden_order(self.st.n_mb)
        self.refresh_ptr = 0
        self.q = float(q)
        self.q_min, self.q_max = q_min, q_max
        self.search = search
        self.frame_num = 0
        self.last_stats = EncodeStats()
        self.mvs = np.zeros((self.st.n_mb, 2), dtype=np.int64)

    def snapshot(self):
        st = self.st
        return (st.y.copy(), st.cb.copy(), st.cr.copy(), self.frame_num, self.refresh_ptr, self.q, self.mvs.copy())

    def restore(self, snap):
        """Undo the last encode_frame (used when the link layer has to drop the frame)."""
        st = self.st
        st.y, st.cb, st.cr, self.frame_num, self.refresh_ptr, self.q, self.mvs = snap
        st.y, st.cb, st.cr, self.mvs = st.y.copy(), st.cb.copy(), st.cr.copy(), self.mvs.copy()

    # -------------------------------------------------------------- rate control
    def _update_q(self, bits: int, target_bits: int):
        if target_bits <= 0:
            return
        ratio = bits / max(target_bits, 1)
        self.q += float(np.clip(3.0 * np.log2(ratio), -3.0, 4.0))
        self.q = float(np.clip(self.q, self.q_min, self.q_max))

    def encode_frame(self, rgb: np.ndarray, target_bits: int = 0, max_bits: int = 0) -> Tuple[bytes, EncodeStats]:
        """Encode one RGB frame (h, w, 3) uint8. Returns (payload bytes, stats).

        target_bits drives the rate control; max_bits (0 = none) triggers one
        re-encode at a coarser quantiser when exceeded.
        """
        y, cb, cr = rgb_to_ycbcr420(rgb)
        q = int(round(self.q))
        data, stats, recon = self._encode(y, cb, cr, q)
        if max_bits and len(data) * 8 > max_bits * 1.25 and q < self.q_max:
            q = min(self.q_max, q + int(np.ceil(4 * np.log2(len(data) * 8 / max_bits))))
            data, stats, recon = self._encode(y, cb, cr, q)
        self.st.y, self.st.cb, self.st.cr = recon
        self.q = float(q)
        self._update_q(len(data) * 8, target_bits)
        self.frame_num = (self.frame_num + 1) & 0xFF
        self.refresh_ptr = (self.refresh_ptr + self._n_refresh()) % self.st.n_mb
        self.last_stats = stats
        return data, stats

    def _n_refresh(self) -> int:
        return int(np.ceil(self.st.n_mb / self.refresh_period))

    def _encode(self, y, cb, cr, q):
        st = self.st
        k = 8
        step8 = qstep8(q)
        lam = 0.1 * (step8 / 8.0) ** 2          # Lagrangian multiplier (SSE units per bit)
        ref_y, ref_cb, ref_cr = st.y, st.cb, st.cr
        cur_blocks = blockify(y, 8).astype(np.int64)
        n_mb = st.n_mb
        # forced intra refresh set
        forced = np.zeros(n_mb, dtype=bool)
        nr = self._n_refresh()
        idx = [(self.refresh_ptr + i) % n_mb for i in range(nr)]
        forced[self.refresh_order[idx]] = True
        # ---------------- motion search (vectorised over all MBs per candidate)
        gu, gv = global_motion_ls(y, ref_y)
        gmv = (int(round(gu)), int(round(gv)))
        cands = {(0, 0), gmv}
        s = self.search
        for dy in range(-s, s + 1):
            for dx in range(-s, s + 1):
                cands.add((dx, dy))
                cands.add((gmv[0] + dx, gmv[1] + dy))
        cands = sorted(cands)
        best_cost = np.full(n_mb, np.inf)
        best_mv = np.zeros((n_mb, 2), dtype=np.int64)
        lam_mv = 0.5 * np.sqrt(lam) * 8
        for (dx, dy) in cands:
            shifted = blockify(shift_image(ref_y, dx, dy), 8).astype(np.int64)
            sad = np.sum(np.abs(cur_blocks - shifted), axis=(1, 2)).astype(np.float64)
            cost = sad + lam_mv * (abs(dx) + abs(dy)) * 1.0
            better = cost < best_cost
            best_cost[better] = cost[better]
            best_mv[better] = (dx, dy)
        # ---------------- per-MB coding with RDO
        enc = Encoder()
        probs = new_probs()
        rec_y = np.empty_like(y)
        rec_cb = np.empty_like(cb)
        rec_cr = np.empty_like(cr)
        modes = np.zeros(n_mb, dtype=np.int64)   # 0 skip 1 inter 2 intra
        mvs = np.zeros((n_mb, 2), dtype=np.int64)
        cbp_prev = [0, 0, 0]
        stats = EncodeStats(q=q)
        for i in range(n_mb):
            r, c = divmod(i, st.mbw)
            y0, x0 = r * 8, c * 8
            cy0, cx0 = r * 4, c * 4
            cur_y = y[y0:y0 + 8, x0:x0 + 8].astype(np.int64)
            cur_cb = cb[cy0:cy0 + 4, cx0:cx0 + 4].astype(np.int64)
            cur_cr = cr[cy0:cy0 + 4, cx0:cx0 + 4].astype(np.int64)
            # --- candidate: SKIP
            sk_y = ref_y[y0:y0 + 8, x0:x0 + 8].astype(np.int64)
            sk_cb = ref_cb[cy0:cy0 + 4, cx0:cx0 + 4].astype(np.int64)
            sk_cr = ref_cr[cy0:cy0 + 4, cx0:cx0 + 4].astype(np.int64)
            sse_skip = _sse(cur_y, sk_y) + _sse(cur_cb, sk_cb) + _sse(cur_cr, sk_cr)
            # --- candidate: INTER
            mv = best_mv[i]
            pmv = _mv_pred(mvs, i, st.mbw)
            pr_y = mc_block(ref_y, y0, x0, 8, int(mv[0]), int(mv[1])).astype(np.int64)
            cmx, cmy = int(np.floor(mv[0] / 2 + 0.5)), int(np.floor(mv[1] / 2 + 0.5))
            pr_cb = mc_block(ref_cb, cy0, cx0, 4, cmx, cmy).astype(np.int64)
            pr_cr = mc_block(ref_cr, cy0, cx0, 4, cmx, cmy).astype(np.int64)
            inter = _code_residual(cur_y, cur_cb, cur_cr, pr_y, pr_cb, pr_cr, step8)
            mvd_bits = _mvd_bits(int(mv[0]) - pmv[0]) + _mvd_bits(int(mv[1]) - pmv[1])
            j_inter = inter.sse + lam * (inter.bits + mvd_bits + 2)
            j_skip = sse_skip + lam * 1
            # --- candidate: INTRA (DC prediction from reconstructed neighbours of current frame)
            dc_y = _dc_pred(rec_y, y0, x0, 8)
            dc_cb = _dc_pred(rec_cb, cy0, cx0, 4)
            dc_cr = _dc_pred(rec_cr, cy0, cx0, 4)
            intra = _code_residual(cur_y, cur_cb, cur_cr, np.full((8, 8), dc_y, np.int64),
                                   np.full((4, 4), dc_cb, np.int64), np.full((4, 4), dc_cr, np.int64), step8)
            j_intra = intra.sse + lam * (intra.bits + 3)
            if forced[i]:
                mode = 2
            else:
                js = [j_skip, j_inter, j_intra]
                mode = int(np.argmin(js))
            # --- write syntax
            n_skip_nb = (1 if c > 0 and modes[i - 1] == 0 else 0) + (1 if r > 0 and modes[i - st.mbw] == 0 else 0)
            enc.encode(probs, Ctx.SKIP + n_skip_nb, 1 if mode == 0 else 0)
            if mode == 0:
                rec_y[y0:y0 + 8, x0:x0 + 8] = sk_y
                rec_cb[cy0:cy0 + 4, cx0:cx0 + 4] = sk_cb
                rec_cr[cy0:cy0 + 4, cx0:cx0 + 4] = sk_cr
                modes[i] = 0
                mvs[i] = 0
                stats.n_skip += 1
                continue
            n_intra_nb = (1 if c > 0 and modes[i - 1] == 2 else 0) + (1 if r > 0 and modes[i - st.mbw] == 2 else 0)
            enc.encode(probs, Ctx.INTRA + n_intra_nb, 1 if mode == 2 else 0)
            if mode == 1:
                _encode_mvd(enc, probs, 0, int(mv[0]) - pmv[0])
                _encode_mvd(enc, probs, 1, int(mv[1]) - pmv[1])
                mvs[i] = mv
                res = inter
                stats.n_inter += 1
            else:
                mvs[i] = 0
                res = intra
                stats.n_intra += 1
            modes[i] = mode
            for pl in range(3):
                coded = 1 if res.levels[pl] is not None else 0
                enc.encode(probs, Ctx.CBP + pl * 2 + cbp_prev[pl], coded)
                cbp_prev[pl] = coded
                if coded:
                    encode_block(enc, probs, res.levels[pl].reshape(-1), ZZ8 if pl == 0 else ZZ4, 0 if pl == 0 else 1)
            rec_y[y0:y0 + 8, x0:x0 + 8] = res.rec[0]
            rec_cb[cy0:cy0 + 4, cx0:cx0 + 4] = res.rec[1]
            rec_cr[cy0:cy0 + 4, cx0:cx0 + 4] = res.rec[2]
        payload = bytes([self.frame_num & 0xFF, q & 0xFF]) + enc.finish()
        stats.bits = len(payload) * 8
        stats.psnr = psnr(rec_y, y)
        self.mvs = mvs
        return payload, stats, (rec_y, rec_cb, rec_cr)


def _sse(a, b) -> float:
    d = a - b
    return float(np.sum(d * d))


def _mvd_bits(v: int) -> float:
    v = abs(v)
    return 1.0 if v == 0 else 2.0 + 2.0 * np.log2(v + 1)


def _encode_mvd(enc: Encoder, probs: list, axis: int, v: int):
    enc.encode(probs, Ctx.MVD_ZERO + axis, 0 if v == 0 else 1)
    if v == 0:
        return
    enc.encode_direct(1 if v < 0 else 0, 1)
    encode_unary_eg(enc, probs, Ctx.MVD_MAG + axis * 6, 6, abs(v) - 1)


def _decode_mvd(dec: Decoder, probs: list, axis: int) -> int:
    if dec.decode(probs, Ctx.MVD_ZERO + axis) == 0:
        return 0
    sign = dec.decode_direct(1)
    mag = decode_unary_eg(dec, probs, Ctx.MVD_MAG + axis * 6, 6) + 1
    return -mag if sign else mag


@dataclass
class _Residual:
    levels: list            # per plane: int64 array (k,k) or None if all zero
    rec: list               # per plane reconstructed uint8 block
    sse: float
    bits: float


def _code_residual(cur_y, cur_cb, cur_cr, pr_y, pr_cb, pr_cr, step8) -> _Residual:
    levels, rec, bits, sse = [], [], 0.0, 0.0
    for pl, (cur, pr) in enumerate(((cur_y, pr_y), (cur_cb, pr_cb), (cur_cr, pr_cr))):
        k = cur.shape[0]
        c = C8 if k == 8 else C4
        chroma = pl > 0
        coef = fdct_blocks((cur - pr)[None], c)[0]
        lv = _quant(coef[None], step8, chroma)[0]
        if not np.any(lv):
            levels.append(None)
            r = np.clip(pr, 0, 255).astype(np.uint8)
            bits += 1
        else:
            levels.append(lv)
            res = idct_blocks_int(_dequant8(lv, step8, chroma)[None], c)[0]
            r = np.clip(pr + res, 0, 255).astype(np.uint8)
            nz = np.count_nonzero(lv)
            bits += 2 + 5.5 * nz + 1.5 * np.sum(np.log2(np.abs(lv[lv != 0]) + 1))
        rec.append(r)
        sse += _sse(cur, r.astype(np.int64))
    return _Residual(levels, rec, sse, bits)


# ---------------------------------------------------------------- decoder
class VideoDecoder:
    def __init__(self, width: int, height: int, postfilter: bool = True):
        self.st = _State(width, height)
        self.postfilter = postfilter
        self.last_frame_num: Optional[int] = None
        self.frames_decoded = 0
        self.frames_lost = 0
        self.last_mvs: Optional[np.ndarray] = None      # (n_mb, 2) [dx, dy] of the last decoded frame
        self.last_modes: Optional[np.ndarray] = None    # 0 skip, 1 inter, 2 intra

    def decode_frame(self, payload: bytes) -> Optional[np.ndarray]:
        """Decode a frame payload; returns RGB uint8 (h, w, 3) or None on error."""
        st = self.st
        if len(payload) < 3:
            return None
        frame_num, q = payload[0], payload[1]
        if self.last_frame_num is not None:
            gap = (frame_num - self.last_frame_num) & 0xFF
            if gap > 1 and gap < 128:
                self.frames_lost += gap - 1
        self.last_frame_num = frame_num
        step8 = qstep8(q)
        dec = Decoder(payload[2:])
        probs = new_probs()
        ref_y, ref_cb, ref_cr = st.y, st.cb, st.cr
        rec_y = np.empty_like(ref_y)
        rec_cb = np.empty_like(ref_cb)
        rec_cr = np.empty_like(ref_cr)
        modes = np.zeros(st.n_mb, dtype=np.int64)
        mvs = np.zeros((st.n_mb, 2), dtype=np.int64)
        cbp_prev = [0, 0, 0]
        try:
            for i in range(st.n_mb):
                r, c = divmod(i, st.mbw)
                y0, x0 = r * 8, c * 8
                cy0, cx0 = r * 4, c * 4
                n_skip_nb = (1 if c > 0 and modes[i - 1] == 0 else 0) + (1 if r > 0 and modes[i - st.mbw] == 0 else 0)
                if dec.decode(probs, Ctx.SKIP + n_skip_nb) == 1:
                    rec_y[y0:y0 + 8, x0:x0 + 8] = ref_y[y0:y0 + 8, x0:x0 + 8]
                    rec_cb[cy0:cy0 + 4, cx0:cx0 + 4] = ref_cb[cy0:cy0 + 4, cx0:cx0 + 4]
                    rec_cr[cy0:cy0 + 4, cx0:cx0 + 4] = ref_cr[cy0:cy0 + 4, cx0:cx0 + 4]
                    modes[i] = 0
                    continue
                n_intra_nb = (1 if c > 0 and modes[i - 1] == 2 else 0) + (1 if r > 0 and modes[i - st.mbw] == 2 else 0)
                intra = dec.decode(probs, Ctx.INTRA + n_intra_nb) == 1
                if intra:
                    modes[i] = 2
                    pr = [np.full((8, 8), _dc_pred(rec_y, y0, x0, 8), np.int64),
                          np.full((4, 4), _dc_pred(rec_cb, cy0, cx0, 4), np.int64),
                          np.full((4, 4), _dc_pred(rec_cr, cy0, cx0, 4), np.int64)]
                else:
                    modes[i] = 1
                    pmv = _mv_pred(mvs, i, st.mbw)
                    mvx = _decode_mvd(dec, probs, 0) + pmv[0]
                    mvy = _decode_mvd(dec, probs, 1) + pmv[1]
                    mvs[i] = (mvx, mvy)
                    cmx, cmy = int(np.floor(mvx / 2 + 0.5)), int(np.floor(mvy / 2 + 0.5))
                    pr = [mc_block(ref_y, y0, x0, 8, mvx, mvy).astype(np.int64),
                          mc_block(ref_cb, cy0, cx0, 4, cmx, cmy).astype(np.int64),
                          mc_block(ref_cr, cy0, cx0, 4, cmx, cmy).astype(np.int64)]
                outs = []
                for pl in range(3):
                    coded = dec.decode(probs, Ctx.CBP + pl * 2 + cbp_prev[pl])
                    cbp_prev[pl] = coded
                    k = 8 if pl == 0 else 4
                    if coded:
                        lv = decode_block(dec, probs, k, ZZ8 if pl == 0 else ZZ4, 0 if pl == 0 else 1).reshape(k, k)
                        res = idct_blocks_int(_dequant8(lv, step8, pl > 0)[None], C8 if k == 8 else C4)[0]
                        outs.append(np.clip(pr[pl] + res, 0, 255).astype(np.uint8))
                    else:
                        outs.append(np.clip(pr[pl], 0, 255).astype(np.uint8))
                rec_y[y0:y0 + 8, x0:x0 + 8] = outs[0]
                rec_cb[cy0:cy0 + 4, cx0:cx0 + 4] = outs[1]
                rec_cr[cy0:cy0 + 4, cx0:cx0 + 4] = outs[2]
                if dec.overrun():
                    return None
        except Exception:
            return None
        st.y, st.cb, st.cr = rec_y, rec_cb, rec_cr
        self.frames_decoded += 1
        self.last_mvs, self.last_modes = mvs, modes
        if self.postfilter:
            thr = 1.5 * step8 / 8.0 / 8.0 * 2 + 2     # ~ pixel-domain size of one quantiser step on the DC
            return ycbcr420_to_rgb(deblock(rec_y, 8, thr), deblock(rec_cb, 4, thr), deblock(rec_cr, 4, thr))
        return ycbcr420_to_rgb(rec_y, rec_cb, rec_cr)

    def current_rgb(self) -> np.ndarray:
        return ycbcr420_to_rgb(self.st.y, self.st.cb, self.st.cr)


def deblock(plane: np.ndarray, k: int, thr: float) -> np.ndarray:
    """Display-time deblocking: smooth k x k block edges whose step is below thr (a quantiser-sized
    discontinuity is an artefact; a larger one is a real edge).  Operates outside the prediction
    loop, so it never affects the bit-exact reconstruction."""
    p = plane.astype(np.float64)
    out = p.copy()
    h, w = p.shape
    for x in range(k, w, k):           # vertical edges
        a, b = p[:, x - 1], p[:, x]
        d = b - a
        m = np.abs(d) < thr
        out[:, x - 1][m] += d[m] * 0.25
        out[:, x][m] -= d[m] * 0.25
        if x >= 2 and x + 1 < w:
            out[:, x - 2][m] += d[m] * 0.125
            out[:, x + 1][m] -= d[m] * 0.125
    p = out.copy()
    for y in range(k, h, k):           # horizontal edges
        a, b = p[y - 1, :], p[y, :]
        d = b - a
        m = np.abs(d) < thr
        out[y - 1, :][m] += d[m] * 0.25
        out[y, :][m] -= d[m] * 0.25
        if y >= 2 and y + 1 < h:
            out[y - 2, :][m] += d[m] * 0.125
            out[y + 1, :][m] -= d[m] * 0.125
    return np.clip(np.round(out), 0, 255).astype(np.uint8)
