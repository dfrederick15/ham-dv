"""Probabilistic amplitude shaping (PAS) for 256-QAM.

Square 256-QAM is 16-PAM on each axis: a sign bit and one of eight
amplitudes a in {1, 3, ..., 15}.  Uniformly used, the outer points cost a lot
of power for little extra information.  Shaping sends the amplitudes with a
Maxwell-Boltzmann distribution

        P(a) = exp(-nu a^2) / sum_a' exp(-nu a'^2),

the distribution that maximises entropy for a given mean power.  In the
limit this recovers the 1.53 dB "ultimate shaping gain" (the gap between a
square and a Gaussian signal); at the block lengths used here about 1 dB is
obtained, and the information rate becomes continuously tunable through nu
instead of jumping in steps of a whole bit per symbol.

Architecture (Boecherer, Steiner, Schulte 2015):

    user bits --> distribution matcher --> amplitudes --> amplitude labels (3 bits each)
                                                              |
                                     systematic LDPC, rate 3/4: info = amplitude bits,
                                     parity bits = sign bits (parity is uniform, so the
                                     signs are uniform and the amplitudes keep their shape)

Distribution matcher
--------------------
A constant-composition matcher: every block of n = 496 amplitudes contains
exactly n_a copies of amplitude a.  The number of such sequences is the
multinomial coefficient M = n! / prod_a n_a!, and k = floor(log2 M) user bits
select one of them by *enumerative coding* (ranking / unranking of multiset
permutations):

    rank(seq) = sum_i sum_{a' < seq[i]}  #completions after choosing a' at position i,

where the number of completions of a remaining composition starting with
amplitude a' is  M_rem * n_a' / n_rem  (exact integer).  Python's arbitrary
precision integers make this exact and perfectly invertible, with a rate loss
of only ~0.06 bit per amplitude relative to the entropy H(P) at n = 496.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import comb
from typing import List, Sequence, Tuple

import numpy as np

from .constellation import constellation, _gray

N_AMP_LEVELS = 8                 # amplitudes 1,3,...,15 (index a -> level 2a+1)
AMPS_PER_BLOCK = 496             # 248 symbols x 2 dimensions

# Constant compositions for n = 496 amplitudes (index 0..7 = levels 1..15), chosen as the
# most strongly shaped Maxwell-Boltzmann composition whose multinomial coefficient still
# admits k_dm user bits.  Rates are in user bits per *symbol* (2 amplitudes + sign bits
# carry only parity).  Hard-coded so that every implementation uses identical tables.
PAS_TABLE = {
    # bits/symbol: (k_dm user bits per block, composition n_0..n_7)
    4.5: (1116, [161, 136, 96, 57, 29, 12, 4, 1]),
    5.0: (1240, [134, 119, 94, 67, 42, 23, 12, 5]),
    5.5: (1364, [104, 98, 86, 71, 54, 39, 27, 17]),
}


def multinomial(counts: Sequence[int]) -> int:
    m = 1
    rem = sum(counts)
    for c in counts:
        m *= comb(rem, c)
        rem -= c
    return m


def mb_composition(nu: float, n: int = AMPS_PER_BLOCK) -> List[int]:
    """Maxwell-Boltzmann probabilities quantised to integer counts summing to n."""
    lv = 2 * np.arange(N_AMP_LEVELS) + 1
    p = np.exp(-nu * lv.astype(np.float64) ** 2)
    p /= p.sum()
    counts = np.floor(p * n).astype(int)
    # distribute the remainder to the largest fractional parts (deterministic)
    frac = p * n - counts
    for i in np.argsort(-frac, kind="stable")[: n - counts.sum()]:
        counts[i] += 1
    return [int(c) for c in counts]


def design_composition(k_dm: int, n: int = AMPS_PER_BLOCK) -> List[int]:
    """Largest nu (strongest shaping) whose composition still has log2(M) >= k_dm."""
    lo, hi = 0.0, 0.2
    for _ in range(60):
        mid = (lo + hi) / 2
        if multinomial(mb_composition(mid, n)).bit_length() - 1 >= k_dm:
            lo = mid
        else:
            hi = mid
    return mb_composition(lo, n)


def unrank(rank: int, counts: Sequence[int]) -> List[int]:
    counts = list(counts)
    rem = sum(counts)
    m = multinomial(counts)
    seq = []
    for _ in range(rem, 0, -1):
        for a in range(len(counts)):
            if counts[a] == 0:
                continue
            cnt = m * counts[a] // rem
            if rank < cnt:
                seq.append(a)
                m = cnt
                counts[a] -= 1
                rem -= 1
                break
            rank -= cnt
        else:  # pragma: no cover
            raise ValueError("rank out of range")
    return seq


def rank(seq: Sequence[int], counts: Sequence[int]) -> int:
    counts = list(counts)
    rem = sum(counts)
    m = multinomial(counts)
    r = 0
    for s in seq:
        for a in range(s):
            if counts[a]:
                r += m * counts[a] // rem
        if counts[s] <= 0:
            return 0          # composition violated (decoding failure) - caller checks CRC
        m = m * counts[s] // rem
        counts[s] -= 1
        rem -= 1
    return r


def bits_to_int(bits: np.ndarray) -> int:
    return int("".join("1" if b else "0" for b in bits), 2) if len(bits) else 0


def int_to_bits(v: int, n: int) -> np.ndarray:
    s = bin(v)[2:].zfill(n)
    return np.frombuffer(s.encode(), dtype=np.uint8) - 48


# ----------------------------------------------------------------- labels and points
@lru_cache(maxsize=None)
def pam16_tables():
    """Gray labels of 16-PAM: returns (amp_label[a] (3 bits as int), sign bit for positive)."""
    g = _gray(16)
    amp_label = np.array([int(g[8 + a]) & 7 for a in range(8)])
    # the reflected Gray code is mirror symmetric apart from its MSB
    for a in range(8):
        assert (int(g[7 - a]) & 7) == amp_label[a]
    sign_pos = int(g[8]) >> 3
    label_to_amp = np.zeros(8, dtype=np.int64)
    label_to_amp[amp_label] = np.arange(8)
    return amp_label, sign_pos, label_to_amp


@dataclass
class PASMode:
    bits_per_symbol: float
    k_dm: int
    composition: List[int]

    def __post_init__(self):
        counts = np.array(self.composition, dtype=np.float64)
        self.p_amp = counts / counts.sum()
        lv = 2 * np.arange(8) + 1
        e = 2 * np.sum(self.p_amp * lv ** 2)
        self.scale = 1.0 / np.sqrt(e)                       # unit average power with the shaped distribution
        pts, labels = constellation(8)
        # points in the same label order as constellation(8), rescaled for the shaped power
        pts_u = pts * np.sqrt(np.mean(np.abs(np.array([complex(2 * i - 15, 2 * q - 15) for i in range(16) for q in range(16)])) ** 2))
        self.points = pts_u * self.scale
        self.labels = labels
        _, _, label_to_amp = pam16_tables()
        lab_i = (labels[:, 1] << 2) | (labels[:, 2] << 1) | labels[:, 3]
        lab_q = (labels[:, 5] << 2) | (labels[:, 6] << 1) | labels[:, 7]
        a_i, a_q = label_to_amp[lab_i], label_to_amp[lab_q]
        self.log_prior = np.log(self.p_amp[a_i] * self.p_amp[a_q] / 4.0)
        weights = 1 << np.arange(7, -1, -1)
        self.lut = np.zeros(256, dtype=np.int64)
        self.lut[labels.astype(np.int64) @ weights] = np.arange(256)

    def amplitude_bits(self, user_bits: np.ndarray) -> np.ndarray:
        """k_dm user bits -> 496 amplitudes -> 1488 label bits (amplitude-major, MSB first)."""
        amp_label, _, _ = pam16_tables()
        amps = unrank(bits_to_int(user_bits[: self.k_dm]), self.composition)
        lab = amp_label[np.array(amps)]
        return np.stack([(lab >> 2) & 1, (lab >> 1) & 1, lab & 1], axis=1).reshape(-1).astype(np.uint8)

    def user_bits(self, amp_bits: np.ndarray) -> Tuple[np.ndarray, bool]:
        _, _, label_to_amp = pam16_tables()
        lab = amp_bits.reshape(-1, 3).astype(np.int64)
        amps = label_to_amp[(lab[:, 0] << 2) | (lab[:, 1] << 1) | lab[:, 2]]
        ok = [int(np.sum(amps == a)) for a in range(8)] == list(self.composition)
        return int_to_bits(rank(amps.tolist(), self.composition), self.k_dm), ok

    def map_labels(self, label_bits: np.ndarray) -> np.ndarray:
        """(248, 8) label bits -> complex symbols."""
        weights = 1 << np.arange(7, -1, -1)
        return self.points[self.lut[label_bits.astype(np.int64) @ weights]]


@lru_cache(maxsize=None)
def pas_mode(bits_per_symbol: float) -> PASMode:
    k_dm = int(round(bits_per_symbol * AMPS_PER_BLOCK / 2))
    key = round(bits_per_symbol, 2)
    comp = PAS_TABLE[key][1] if key in PAS_TABLE else design_composition(k_dm)
    return PASMode(bits_per_symbol, k_dm, comp)
