"""Low-rate LPC/LSP mixed-excitation speech vocoder ("HamVoice").

Modes: 1200, 1600 and 2400 bit/s with 30 ms frames (240 samples at 8 kHz).

Analysis (per frame)
--------------------
1. Pre-emphasis x'[n] = x[n] - 0.9 x[n-1]; 40 ms Hamming window.
2. Autocorrelation r[k], lag window, white-noise correction, then the
   Levinson-Durbin recursion solves the Toeplitz normal equations
        R a = -r     (order 10)
   for the linear-prediction polynomial A(z) = 1 + sum a_i z^-i, followed
   by bandwidth expansion a_i <- a_i 0.994^i.
3. Line spectral pairs: roots of the symmetric / antisymmetric polynomials
        P(z) = A(z) + z^-(p+1) A(z^-1),   Q(z) = A(z) - z^-(p+1) A(z^-1)
   lie on the unit circle and interlace; their angles w_1 < ... < w_10 are
   the LSPs.  The differences w_i - w_(i-1) are quantised on a log scale
   (the quantiser only has to preserve ordering, so stability is guaranteed).
4. Pitch: normalised autocorrelation of the LPC residual over lags 20..160
   with sub-multiple (octave error) checking; voicing from its peak value
   and (2400 mode) per-band voicing strengths.
5. Gain: RMS of the residual on a log scale.

Synthesis
---------
Per 7.5 ms sub-frame the LSPs are interpolated and converted back to A(z)
(polynomial products of the LSP root pairs).  Excitation is a dispersed
pulse train in voiced bands plus white noise in unvoiced bands (4-band
filter bank), scaled to the transmitted gain and filtered by 1/A(z).  An
adaptive post filter A(z/0.5)/A(z/0.8) with tilt compensation sharpens the
formants before de-emphasis.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from scipy import signal

FS = 8000
FRAME = 240
ORDER = 10
SUB = 4                       # sub-frames per frame
SUBLEN = FRAME // SUB
WIN = 320
PITCH_MIN, PITCH_MAX = 20, 160

# bit allocation per mode: lsp bits (10), pitch, gain(s), voicing bits
MODES = {
    1200: dict(lsp=[3, 3, 3, 3, 3, 2, 2, 2, 2, 2], pitch=6, gain=[4], voicing=1),
    1600: dict(lsp=[4, 4, 4, 4, 4, 3, 3, 3, 3, 3], pitch=7, gain=[5], voicing=1),
    2400: dict(lsp=[5, 5, 5, 5, 5, 4, 4, 4, 4, 4], pitch=7, gain=[5, 5], voicing=4 + 4),
}
# log-domain ranges of the LSP differences (radians)
D_MIN = 0.015
D_MAX = [0.55, 0.65, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.9]


def bits_per_frame(mode: int) -> int:
    m = MODES[mode]
    return sum(m["lsp"]) + m["pitch"] + sum(m["gain"]) + m["voicing"]


# --------------------------------------------------------------------- LPC
def levinson(r: np.ndarray, order: int) -> np.ndarray:
    """Levinson-Durbin: returns a[0..order] with a[0] = 1 (prediction polynomial)."""
    a = np.zeros(order + 1)
    a[0] = 1.0
    e = r[0]
    for i in range(1, order + 1):
        acc = r[i] + np.dot(a[1:i], r[i - 1:0:-1])
        k = -acc / e
        a_new = a.copy()
        a_new[i] = k
        a_new[1:i] = a[1:i] + k * a[i - 1:0:-1]
        a = a_new
        e *= (1 - k * k)
        if e <= 1e-12:
            break
    return a


def lpc_to_lsp(a: np.ndarray) -> np.ndarray:
    p = len(a) - 1
    a_rev = a[::-1]
    P = np.concatenate([a, [0.0]]) + np.concatenate([[0.0], a_rev])
    Q = np.concatenate([a, [0.0]]) - np.concatenate([[0.0], a_rev])
    # remove the trivial roots z = -1 from P and z = +1 from Q (p even)
    P = np.polydiv(P, [1.0, 1.0])[0]
    Q = np.polydiv(Q, [1.0, -1.0])[0]
    rp = np.angle(np.roots(P))
    rq = np.angle(np.roots(Q))
    w = np.sort(np.concatenate([rp[rp > 0], rq[rq > 0]]))
    w = w[(w > 0) & (w < np.pi)]
    if len(w) != p:  # numerical trouble: fall back to a flat spectrum
        w = np.arange(1, p + 1) * np.pi / (p + 1)
    return w


def lsp_to_lpc(w: np.ndarray) -> np.ndarray:
    p = len(w)
    P = np.array([1.0])
    Q = np.array([1.0])
    for i in range(0, p, 2):
        P = np.convolve(P, [1.0, -2.0 * np.cos(w[i]), 1.0])
    for i in range(1, p, 2):
        Q = np.convolve(Q, [1.0, -2.0 * np.cos(w[i]), 1.0])
    P = np.convolve(P, [1.0, 1.0])
    Q = np.convolve(Q, [1.0, -1.0])
    a = 0.5 * (P + Q)
    return a[:p + 1]


# --------------------------------------------------------------- quantisers
def _q_uniform(x: float, lo: float, hi: float, bits: int) -> int:
    n = (1 << bits) - 1
    return int(np.clip(round((x - lo) / (hi - lo) * n), 0, n))


def _dq_uniform(i: int, lo: float, hi: float, bits: int) -> float:
    n = (1 << bits) - 1
    return lo + (hi - lo) * i / n


def quant_lsp(w: np.ndarray, bits: List[int]) -> Tuple[List[int], np.ndarray]:
    idx = []
    prev = 0.0
    out = np.zeros(len(w))
    for i in range(len(w)):
        d = max(w[i] - prev, D_MIN)
        u = np.log(d / D_MIN) / np.log(D_MAX[i] / D_MIN)
        k = _q_uniform(u, 0.0, 1.0, bits[i])
        idx.append(k)
        dq = D_MIN * np.exp(_dq_uniform(k, 0.0, 1.0, bits[i]) * np.log(D_MAX[i] / D_MIN))
        prev = prev + dq
        out[i] = prev
    return idx, out


def dequant_lsp(idx: List[int], bits: List[int]) -> np.ndarray:
    prev = 0.0
    out = np.zeros(len(idx))
    for i, k in enumerate(idx):
        dq = D_MIN * np.exp(_dq_uniform(k, 0.0, 1.0, bits[i]) * np.log(D_MAX[i] / D_MIN))
        prev += dq
        out[i] = prev
    # keep inside (0, pi) with minimum separation
    out = np.minimum(out, np.pi - 0.02 * (len(idx) - np.arange(len(idx))))
    return out


GAIN_LO, GAIN_HI = -75.0, -3.0     # dB re full scale (input in [-1, 1])


# -------------------------------------------------------------- bit packing
class BitWriter:
    def __init__(self):
        self.bits: List[int] = []

    def write(self, v: int, n: int):
        for i in range(n - 1, -1, -1):
            self.bits.append((v >> i) & 1)

    def tobytes(self) -> bytes:
        b = self.bits + [0] * ((-len(self.bits)) % 8)
        return bytes(int("".join(map(str, b[i:i + 8])), 2) for i in range(0, len(b), 8))


class BitReader:
    def __init__(self, data: bytes, nbits: int = None):
        self.bits = [(byte >> (7 - i)) & 1 for byte in data for i in range(8)]
        self.pos = 0

    def read(self, n: int) -> int:
        v = 0
        for _ in range(n):
            v = (v << 1) | (self.bits[self.pos] if self.pos < len(self.bits) else 0)
            self.pos += 1
        return v


# ------------------------------------------------------------------ bands
def _bandfilters():
    edges = [(0, 500), (500, 1000), (1000, 2000), (2000, 3999)]
    taps = []
    for lo, hi in edges:
        if lo == 0:
            taps.append(signal.firwin(33, hi, fs=FS))
        else:
            taps.append(signal.firwin(33, [lo, hi], fs=FS, pass_zero=False))
    return taps


BANDS = _bandfilters()


# ------------------------------------------------------------------ encoder
class VocoderEncoder:
    def __init__(self, mode: int = 1200):
        self.mode = mode
        self.cfg = MODES[mode]
        self.buf = np.zeros(WIN + FRAME)
        self.pre_state = 0.0
        self.win = np.hamming(WIN)
        self.lagwin = np.exp(-0.5 * (2 * np.pi * 60.0 / FS * np.arange(ORDER + 1)) ** 2)

    @property
    def frame_samples(self) -> int:
        return FRAME

    @property
    def bits_per_frame(self) -> int:
        return bits_per_frame(self.mode)

    def encode_frame(self, x: np.ndarray) -> List[int]:
        """x: 240 float samples in [-1, 1]. Returns the list of parameter bits."""
        x = np.asarray(x, dtype=np.float64)
        pre = np.empty_like(x)
        prev = self.pre_state
        pre = x - 0.9 * np.concatenate([[prev], x[:-1]])
        self.pre_state = x[-1]
        self.buf = np.concatenate([self.buf[FRAME:], pre])
        seg = self.buf[-WIN - FRAME // 4: -FRAME // 4] if False else self.buf[FRAME - (WIN - FRAME) // 2: FRAME - (WIN - FRAME) // 2 + WIN]
        xw = seg * self.win
        r = np.array([np.dot(xw[:WIN - k], xw[k:]) for k in range(ORDER + 1)])
        r[0] = r[0] * 1.0001 + 1e-9
        r *= self.lagwin
        a = levinson(r, ORDER)
        a *= 0.994 ** np.arange(ORDER + 1)
        w = lpc_to_lsp(a)
        # residual for pitch & gain (on the un-windowed centre frame)
        resid = signal.lfilter(a, [1.0], seg)
        T, strength = self._pitch(resid)
        gains = []
        for g_i in range(len(self.cfg["gain"])):
            n = len(self.cfg["gain"])
            s = resid[(WIN - FRAME) // 2 + g_i * FRAME // n: (WIN - FRAME) // 2 + (g_i + 1) * FRAME // n]
            gains.append(20 * np.log10(np.sqrt(np.mean(s ** 2)) + 1e-9))
        if self.cfg["voicing"] == 1:
            voicing = [1 if strength > 0.5 else 0]
        else:
            # per band voicing strengths at the chosen lag, two sub-frames (4 bits each)
            voicing = []
            for half in range(2):
                s = seg[half * WIN // 2:(half + 1) * WIN // 2]
                for bi, taps in enumerate(BANDS):
                    sb = signal.lfilter(taps, 1.0, s)
                    c = self._nac(sb, T)
                    voicing.append(1 if c > (0.45 if bi == 0 else 0.55) else 0)
        # quantise
        bw = BitWriter()
        lsp_idx, _ = quant_lsp(w, self.cfg["lsp"])
        for k, b in zip(lsp_idx, self.cfg["lsp"]):
            bw.write(k, b)
        bw.write(_q_uniform(np.log2(T / PITCH_MIN), 0.0, 3.0, self.cfg["pitch"]), self.cfg["pitch"])
        for g, b in zip(gains, self.cfg["gain"]):
            bw.write(_q_uniform(g, GAIN_LO, GAIN_HI, b), b)
        for v in voicing:
            bw.write(v, 1)
        return bw.bits

    @staticmethod
    def _nac(x: np.ndarray, T: int) -> float:
        n = len(x) - T
        if n < 8:
            return 0.0
        a, b = x[:n], x[T:T + n]
        return float(np.dot(a, b) / (np.sqrt(np.dot(a, a) * np.dot(b, b)) + 1e-12))

    def _pitch(self, resid: np.ndarray) -> Tuple[int, float]:
        # low-pass the residual to 1 kHz to make the correlation peak cleaner
        b, a = signal.butter(2, 1000 / (FS / 2))
        x = signal.lfilter(b, a, resid)
        n = len(x)
        best_T, best = PITCH_MIN, -1.0
        cors = {}
        for T in range(PITCH_MIN, PITCH_MAX + 1):
            c = self._nac(x, T)
            cors[T] = c
            if c > best:
                best, best_T = c, T
        # octave-error check: prefer a sub-multiple if nearly as good
        for div in (2, 3):
            Ts = best_T // div
            if Ts >= PITCH_MIN:
                c = max(cors.get(t, -1) for t in range(max(PITCH_MIN, Ts - 1), Ts + 2))
                if c > 0.8 * best:
                    best_T = Ts
                    best = c
        return best_T, best


# ------------------------------------------------------------------ decoder
class VocoderDecoder:
    def __init__(self, mode: int = 1200):
        self.mode = mode
        self.cfg = MODES[mode]
        self.prev_w = np.arange(1, ORDER + 1) * np.pi / (ORDER + 1)
        self.prev_gain = GAIN_LO
        self.prev_T = 80.0
        self.phase = 0.0
        self.syn_state = np.zeros(ORDER)
        self.pf_state_num = np.zeros(ORDER)
        self.pf_state_den = np.zeros(ORDER)
        self.deemph_state = 0.0
        self.tilt_state = 0.0
        self.rng = np.random.default_rng(7)
        self.band_states = [np.zeros(32) for _ in BANDS]
        self.band_states_n = [np.zeros(32) for _ in BANDS]
        self.dispersion = self._dispersion_filter()

    @staticmethod
    def _dispersion_filter() -> np.ndarray:
        # spreads each pitch pulse in time (reduces buzziness) - a short chirp-like FIR
        n = 24
        t = np.arange(n)
        h = np.cos(np.pi * t * t / (2 * n)) * np.hanning(2 * n)[n:]
        return h / np.sqrt(np.sum(h ** 2))

    @property
    def frame_samples(self) -> int:
        return FRAME

    def decode_frame(self, bits: List[int]) -> np.ndarray:
        br = BitReader(b"")
        br.bits = list(bits)
        lsp_idx = [br.read(b) for b in self.cfg["lsp"]]
        w = dequant_lsp(lsp_idx, self.cfg["lsp"])
        T = PITCH_MIN * 2.0 ** _dq_uniform(br.read(self.cfg["pitch"]), 0.0, 3.0, self.cfg["pitch"])
        gains = [_dq_uniform(br.read(b), GAIN_LO, GAIN_HI, b) for b in self.cfg["gain"]]
        if self.cfg["voicing"] == 1:
            v = br.read(1)
            band_voicing = [[v, v, v, 0] if v else [0, 0, 0, 0]] * 2
        else:
            band_voicing = [[br.read(1) for _ in range(4)] for _ in range(2)]
        return self._synth(w, T, gains, band_voicing)

    def decode_lost(self) -> np.ndarray:
        """Comfort output for a lost frame: fade the previous parameters."""
        g = self.prev_gain - 6.0
        return self._synth(self.prev_w, self.prev_T, [g] * len(self.cfg["gain"]), [[0, 0, 0, 0]] * 2)

    def _synth(self, w, T, gains, band_voicing) -> np.ndarray:
        out = np.zeros(FRAME)
        n_g = len(gains)
        for s in range(SUB):
            frac = (s + 1) / SUB
            ws = (1 - frac) * self.prev_w + frac * w
            a = lsp_to_lpc(ws)
            # gain interpolation in dB
            gi = gains[min(s * n_g // SUB, n_g - 1)]
            g_db = (1 - frac) * self.prev_gain + frac * gi if n_g == 1 else gi
            g = 10 ** (g_db / 20)
            Ts = (1 - frac) * self.prev_T + frac * T
            bv = band_voicing[0] if s < SUB // 2 else band_voicing[1]
            # excitation: pulse train + noise, mixed per band
            pulses = np.zeros(SUBLEN)
            for n in range(SUBLEN):
                self.phase += 1.0
                if self.phase >= Ts:
                    self.phase -= Ts
                    pulses[n] = np.sqrt(Ts)        # unit RMS pulse train
            pulses = np.convolve(pulses, self.dispersion)[:SUBLEN]
            noise = self.rng.standard_normal(SUBLEN)
            exc = np.zeros(SUBLEN)
            for bi, taps in enumerate(BANDS):
                pb, self.band_states[bi] = signal.lfilter(taps, 1.0, pulses, zi=self.band_states[bi])
                nb, self.band_states_n[bi] = signal.lfilter(taps, 1.0, noise, zi=self.band_states_n[bi])
                exc += pb if bv[bi] else nb
            exc *= g
            y, self.syn_state = signal.lfilter([1.0], a, exc, zi=self.syn_state)
            # post filter: A(z/0.5) / A(z/0.8), then tilt compensation and gain normalisation
            num = a * 0.5 ** np.arange(ORDER + 1)
            den = a * 0.8 ** np.arange(ORDER + 1)
            pf, self.pf_state_num = signal.lfilter(num, [1.0], y, zi=self.pf_state_num)
            pf, self.pf_state_den = signal.lfilter([1.0], den, pf, zi=self.pf_state_den)
            e_in = np.sqrt(np.mean(y ** 2)) + 1e-9
            e_out = np.sqrt(np.mean(pf ** 2)) + 1e-9
            pf *= e_in / e_out
            out[s * SUBLEN:(s + 1) * SUBLEN] = pf
        # de-emphasis 1 / (1 - 0.9 z^-1)
        out, st = signal.lfilter([1.0], [1.0, -0.9], out, zi=[self.deemph_state])
        self.deemph_state = st[0]
        self.prev_w, self.prev_gain, self.prev_T = w, gains[-1], T
        return np.clip(out, -1.0, 1.0)


# ------------------------------------------------------------------ helpers
def pack_frames(frames: List[List[int]]) -> bytes:
    bw = BitWriter()
    for f in frames:
        bw.bits.extend(f)
    return bw.tobytes()


def unpack_frames(data: bytes, n_frames: int, bits_per: int) -> List[List[int]]:
    br = BitReader(data)
    return [[br.read(1) for _ in range(bits_per)] for _ in range(n_frames)]


def synth_test_speech(seconds: float = 3.0, seed: int = 3) -> np.ndarray:
    """Synthetic 'speech': pitch-modulated pulse train through time-varying formant filters, with unvoiced bursts."""
    rng = np.random.default_rng(seed)
    n = int(seconds * FS)
    out = np.zeros(n)
    t = 0
    f0 = 120.0
    phase = 0.0
    vowels = [(730, 1090, 2440), (270, 2290, 3010), (300, 870, 2240), (530, 1840, 2480), (570, 840, 2410)]
    while t < n:
        dur = int(rng.uniform(0.12, 0.3) * FS)
        seg = np.zeros(dur)
        if rng.random() < 0.75:
            f0 = np.clip(f0 * rng.uniform(0.9, 1.1), 80, 200)
            fv = vowels[rng.integers(len(vowels))]
            for i in range(dur):
                phase += f0 * (1 + 0.03 * np.sin(2 * np.pi * 3 * i / FS)) / FS
                if phase >= 1:
                    phase -= 1
                    seg[i] = 1.0
            seg = np.convolve(seg, [1, -0.6, 0.2])[:dur]
            for fc_, bw_ in zip(fv, (80, 120, 160)):
                r = np.exp(-np.pi * bw_ / FS)
                seg = signal.lfilter([1 - r], [1, -2 * r * np.cos(2 * np.pi * fc_ / FS), r * r], seg)
        else:
            seg = rng.standard_normal(dur) * 0.3
            b, a = signal.butter(2, [1500 / 4000, 3800 / 4000], "band")
            seg = signal.lfilter(b, a, seg)
        env = np.hanning(dur) ** 0.5
        seg = seg * env
        seg /= (np.max(np.abs(seg)) + 1e-9)
        seg *= rng.uniform(0.3, 0.7)
        end = min(t + dur, n)
        out[t:end] = seg[: end - t]
        t = end
    return out
