"""Quasi-cyclic LDPC codes with a vectorised normalised min-sum decoder.

Construction
------------
The parity-check matrix is built from p x p circulant permutation matrices
(p prime).  A base matrix B (rows x cols) holds either -1 (all-zero block)
or a shift s in [0, p).  Every block column has weight 3 and the shifts are
drawn by a seeded generator subject to the girth-6 condition: for every
2 x 2 sub-array of non-empty blocks with shifts (a b; c d) we require
    a - b + d - c  !=  0  (mod p),
which rules out length-4 cycles in the Tanner graph.

Structure and encoding
----------------------
H = [H_u | H_p] where H_u (information part) has block-column weight 3 and
H_p is a block "staircase": identity circulants on the diagonal and the
sub-diagonal (an irregular repeat-accumulate structure, as in DVB-S2 and
IEEE 802.16e).  H_p is therefore invertible and encoding is linear time:
    s_i = sum_j C(shift_ij) u_j           (syndrome of the information bits)
    p_0 = s_0,   p_i = p_{i-1} + s_i      (accumulate, mod 2)
No Gaussian elimination is needed, which makes codewords spanning a whole
superframe (n up to ~14000) practical.

Decoding
--------
Normalised min-sum belief propagation.  Check-node messages use the first and
second minimum of the incoming magnitudes (computed with reduceat), variable
nodes sum the channel LLR with incoming messages, and the syndrome is checked
every iteration for early termination.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np


def _no_four_cycle(base: np.ndarray, col: int, rows: np.ndarray, shifts: np.ndarray, p: int) -> bool:
    """Check candidate column (rows, shifts) against all already placed columns."""
    placed = np.nonzero((base >= 0).any(axis=0))[0]
    for c2 in placed:
        if c2 == col:
            continue
        other = base[:, c2]
        # rows where both columns are non-empty
        common = [(r, s) for r, s in zip(rows, shifts) if other[r] >= 0]
        if len(common) < 2:
            continue
        for i in range(len(common)):
            for j in range(i + 1, len(common)):
                r1, a = common[i]
                r2, d = common[j]
                b = other[r1]
                c = other[r2]
                if (a - b + d - c) % p == 0:
                    return False
    return True


def make_base_matrix(block_rows: int, block_cols: int, p: int, col_weight: int = 3, seed: int = 1234) -> np.ndarray:
    """Base matrix [info part | staircase parity part], block_cols = total block columns."""
    rng = np.random.default_rng(seed)
    base = -np.ones((block_rows, block_cols), dtype=np.int64)
    row_fill = np.zeros(block_rows, dtype=np.int64)
    n_info = block_cols - block_rows
    # parity part: staircase of identity circulants (shift 0)
    for r in range(block_rows):
        base[r, n_info + r] = 0
        if r > 0:
            base[r, n_info + r - 1] = 0
        row_fill[r] = 2 if r > 0 else 1
    for c in range(n_info):
        for attempt in range(20000):
            # prefer rows with the fewest entries so row weights stay balanced
            order = np.argsort(row_fill + rng.random(block_rows) * 0.5)
            cand = order[: min(col_weight + 1, block_rows)]
            rows = np.sort(rng.choice(cand, size=min(col_weight, block_rows), replace=False))
            shifts = rng.integers(0, p, size=len(rows))
            if _no_four_cycle(base, c, rows, shifts, p):
                base[rows, c] = shifts
                row_fill[rows] += 1
                break
        else:  # pragma: no cover
            raise RuntimeError("could not place column without 4-cycles; increase p")
    return base


def expand_base(base: np.ndarray, p: int) -> np.ndarray:
    """Expand a base matrix of circulant shifts into the binary H (uint8)."""
    br, bc = base.shape
    h = np.zeros((br * p, bc * p), dtype=np.uint8)
    for r in range(br):
        for c in range(bc):
            s = base[r, c]
            if s < 0:
                continue
            for i in range(p):
                h[r * p + i, c * p + (i + s) % p] = 1
    return h


def gf2_rref(h: np.ndarray):
    """Row-reduce H over GF(2). Returns (reduced matrix, pivot column list)."""
    a = h.copy().astype(np.uint8)
    m, n = a.shape
    pivots = []
    row = 0
    for col in range(n):
        if row >= m:
            break
        pr = np.nonzero(a[row:, col])[0]
        if len(pr) == 0:
            continue
        pr = pr[0] + row
        if pr != row:
            a[[row, pr]] = a[[pr, row]]
        mask = a[:, col].astype(bool)
        mask[row] = False
        a[mask] ^= a[row]
        pivots.append(col)
        row += 1
    return a[:row], pivots


@dataclass
class LDPCCode:
    n: int
    k: int
    p: int
    h: np.ndarray            # parity-check matrix (m x n) uint8
    base: np.ndarray         # base matrix of circulant shifts
    free_cols: np.ndarray    # information positions in the codeword (0..k-1)
    pivot_cols: np.ndarray   # parity positions (k..n-1)
    # decoder tables (edges sorted by check node)
    edge_row: np.ndarray
    edge_col: np.ndarray
    row_start: np.ndarray

    def encode(self, u: np.ndarray) -> np.ndarray:
        u = np.asarray(u, dtype=np.uint8)
        if len(u) != self.k:
            raise ValueError(f"expected {self.k} info bits, got {len(u)}")
        p = self.p
        br = self.base.shape[0]
        n_info = self.k // p
        ub = u.reshape(n_info, p)
        par = np.zeros((br, p), dtype=np.uint8)
        acc = np.zeros(p, dtype=np.uint8)
        for r in range(br):
            srow = np.zeros(p, dtype=np.uint8)
            for c in range(n_info):
                sft = self.base[r, c]
                if sft >= 0:
                    # circulant C(s)[i, (i+s) mod p] = 1  ->  (C u)[i] = u[(i+s) mod p]
                    srow ^= np.roll(ub[c], -sft)
            acc = acc ^ srow
            par[r] = acc
        return np.concatenate([u, par.reshape(-1)])

    def syndrome_ok(self, c: np.ndarray) -> bool:
        return not np.any((self.h.astype(np.int64) @ c.astype(np.int64)) & 1)

    def extract(self, c: np.ndarray) -> np.ndarray:
        return c[: self.k]

    def decode(self, llr: np.ndarray, max_iter: int = 40, alpha: float = 0.8):
        """Normalised min-sum decoding of channel LLRs (positive = bit 0).

        Returns (codeword bits, converged flag, iterations used).
        """
        llr = np.asarray(llr, dtype=np.float64)
        er, ec, rs = self.edge_row, self.edge_col, self.row_start
        m_v2c = llr[ec].copy()
        hard = (llr < 0).astype(np.uint8)
        ok = self._check(hard)
        if ok:
            return hard, True, 0
        for it in range(1, max_iter + 1):
            s = np.where(m_v2c < 0, -1.0, 1.0)
            a = np.abs(m_v2c)
            min1 = np.minimum.reduceat(a, rs)
            is_min = a == min1[er]
            a2 = np.where(is_min, np.inf, a)
            min2 = np.minimum.reduceat(a2, rs)
            cnt = np.add.reduceat(is_min.astype(np.int64), rs)
            min2 = np.where(cnt > 1, min1, min2)
            min2 = np.where(np.isinf(min2), min1, min2)
            prod = np.multiply.reduceat(s, rs)
            mag = np.where(is_min, min2[er], min1[er]) * alpha
            m_c2v = prod[er] * s * mag
            tot = llr + np.bincount(ec, weights=m_c2v, minlength=self.n)
            m_v2c = tot[ec] - m_c2v
            hard = (tot < 0).astype(np.uint8)
            if self._check(hard):
                return hard, True, it
        return hard, False, max_iter

    def _check(self, hard: np.ndarray) -> bool:
        par = np.bitwise_xor.reduceat(hard[self.edge_col], self.row_start)
        return not np.any(par)


@lru_cache(maxsize=None)
def get_code(n: int, rate_num: int, rate_den: int, p: int = 31, seed: int = 1234) -> LDPCCode:
    """Build (and cache) the code with codeword length n and nominal rate."""
    if n % p:
        raise ValueError("n must be a multiple of p")
    bc = n // p
    br = bc - (bc * rate_num) // rate_den
    base = make_base_matrix(br, bc, p, 3, seed + 1000 * bc + br)
    h = expand_base(base, p)
    k = n - br * p
    rows, cols = np.nonzero(h)
    order = np.lexsort((cols, rows))
    rows, cols = rows[order], cols[order]
    row_start = np.concatenate([[0], np.nonzero(np.diff(rows))[0] + 1])
    return LDPCCode(n=n, k=k, p=p, h=h, base=base, free_cols=np.arange(k), pivot_cols=np.arange(k, n),
                    edge_row=rows, edge_col=cols, row_start=row_start)
