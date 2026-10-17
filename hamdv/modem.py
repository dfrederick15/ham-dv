"""SC-FDE modem: symbols <-> audio for an FM voice channel.

Transmit side
-------------
    audio(t) = Re{ sum_k s_k g(t - kT) e^{j 2 pi f_c t} }
with g the root-raised-cosine pulse.  A transmission is

    [preamble: ZC(127) x 2] [superframe 0] [superframe 1] ... [tail]

and every superframe is one *pilot block* followed by B *data blocks*.  A
block is a cyclic prefix of L symbols followed by N = 256 symbols; the pilot
block carries a 256-point Zadoff-Chu sequence, data blocks carry 248 payload
symbols plus 8 scattered pilot symbols.

Receive side
------------
1. Mix to complex baseband, matched filter (RRC).
2. Acquire with a normalised correlation against the preamble; the two
   repetitions give a carrier-offset estimate from their phase difference.
3. For every superframe: re-anchor timing on the pilot block (correlation,
   parabolic sub-sample refinement), track the sample-rate offset, and
   estimate the channel in the frequency domain:
        H_ls[k] = Y[k] / X_p[k]           (least squares; |X_p| is constant)
        h = IDFT(H_ls);  h[k] = 0 for k >= L   (projection onto short-IR subspace)
        H = DFT(h)
   The projection is the orthogonal projector P = F S F^H where S selects the
   first L taps; it removes most of the estimation noise.
4. Equalise every data block with the MMSE weights
        W[k] = H*[k] / (|H[k]|^2 + sigma^2)
   and remove residual phase (and phase slope) using the scattered pilots.

Because the FFT window of every block starts `win_lead` symbols inside the
cyclic prefix, the circular-convolution model holds for channels with
pre-cursor taps up to win_lead and post-cursor taps up to L - win_lead.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from scipy import signal

from .dsp import rrc_taps, zadoff_chu, fd_interp
from .params import ModemParams, DEFAULT


def pilot_block_symbols(p: ModemParams) -> np.ndarray:
    return zadoff_chu(p.n_fft, p.pilot_root)


def scattered_pilots(p: ModemParams) -> np.ndarray:
    """Known QPSK symbols placed at the scattered pilot positions of every data block."""
    z = zadoff_chu(p.n_pilot_sym + 1 if (p.n_pilot_sym % 2 == 0) else p.n_pilot_sym, 1)[: p.n_pilot_sym]
    return z / np.abs(z)


def preamble_symbols(p: ModemParams) -> np.ndarray:
    z = zadoff_chu(p.preamble_len, p.preamble_root)
    return np.tile(z, p.preamble_reps)


def build_data_block(p: ModemParams, payload: np.ndarray) -> np.ndarray:
    """Insert scattered pilots and prepend the cyclic prefix. payload: 248 symbols."""
    if len(payload) != p.data_sym_per_block:
        raise ValueError("payload length mismatch")
    blk = np.empty(p.n_fft, dtype=np.complex128)
    pos = np.array(p.pilot_positions)
    mask = np.ones(p.n_fft, dtype=bool)
    mask[pos] = False
    blk[pos] = scattered_pilots(p)
    blk[mask] = payload
    return np.concatenate([blk[-p.cp:], blk])


def build_pilot_block(p: ModemParams) -> np.ndarray:
    blk = pilot_block_symbols(p)
    return np.concatenate([blk[-p.cp:], blk])


# --------------------------------------------------------------------------
class Modulator:
    """Streaming symbol -> audio converter with continuous filter state."""

    def __init__(self, p: ModemParams = DEFAULT):
        self.p = p
        self.taps = rrc_taps(p.sps, p.alpha, p.rrc_span) * np.sqrt(p.sps)
        self.zi = np.zeros(len(self.taps) - 1, dtype=np.complex128)
        self.n = 0  # sample counter for the carrier phase

    def reset(self):
        self.zi[:] = 0
        self.n = 0

    def modulate(self, symbols: np.ndarray) -> np.ndarray:
        """Return real audio samples (float64, unit average symbol power -> RMS ~ 1/sqrt(2))."""
        p = self.p
        up = np.zeros(len(symbols) * p.sps, dtype=np.complex128)
        up[:: p.sps] = symbols
        bb, self.zi = signal.lfilter(self.taps, 1.0, up, zi=self.zi)
        t = (np.arange(len(bb)) + self.n) / p.fs
        self.n += len(bb)
        return np.real(bb * np.exp(2j * np.pi * p.fc * t))

    def flush(self) -> np.ndarray:
        return self.modulate(np.zeros(self.p.rrc_span, dtype=np.complex128))


# --------------------------------------------------------------------------
@dataclass
class EqualisedBlock:
    sf_index: int           # superframe counter since acquisition
    block: int              # data block index within the superframe (0..B-1)
    symbols: np.ndarray     # 248 equalised payload symbols
    noise_var: float        # estimated post-equalisation noise variance
    snr_db: float


@dataclass
class SuperframeInfo:
    sf_index: int
    anchor: float
    corr_quality: float
    snr_db: float
    sro_ppm: float
    cfo_hz: float


class Demodulator:
    """Streaming audio -> equalised symbol blocks.

    feed() audio chunks (8 kHz float); collect results from .blocks (list of
    EqualisedBlock) and .events (acquisition / loss messages).
    """

    SEARCH, LOCKED = 0, 1

    def __init__(self, p: ModemParams = DEFAULT, corr_threshold: float = 0.45, verbose: bool = False):
        self.p = p
        self.verbose = verbose
        self.corr_threshold = corr_threshold
        self.taps = rrc_taps(p.sps, p.alpha, p.rrc_span)
        self.zi = np.zeros(len(self.taps) - 1, dtype=np.complex128)
        self.n_in = 0                               # audio samples consumed
        self.bb = np.zeros(0, dtype=np.complex128)  # baseband buffer
        self.bb_base = 0                            # absolute index of bb[0]
        self.state = self.SEARCH
        self.search_pos = 0                         # absolute index up to which search was done
        # templates (symbol spaced -> sparse at sample rate)
        self.pre_sym = preamble_symbols(p)
        self.zc1 = zadoff_chu(p.preamble_len, p.preamble_root)
        self.pilot_blk = build_pilot_block(p)
        self.pilot_fft = np.fft.fft(pilot_block_symbols(p))
        self.spilots = scattered_pilots(p)
        self.ppos = np.array(p.pilot_positions)
        self.dmask = np.ones(p.n_fft, dtype=bool)
        self.dmask[self.ppos] = False
        # tracking state
        self.anchor = 0.0
        self.next_anchor = 0.0
        self.sro = 0.0          # fractional sample-rate offset (rx/tx - 1)
        self.cfo = 0.0          # Hz
        self.sf_index = 0
        self.misses = 0
        self.H = None
        self.W = None
        self.noise_t = 1.0
        self.blocks: List[EqualisedBlock] = []
        self.events: List[str] = []
        self.sf_info: List[SuperframeInfo] = []
        self.last_snr_db = float("nan")
        self.stop_after_sf: Optional[int] = None

    # ------------------------------------------------------------------ input
    def feed(self, audio: np.ndarray):
        p = self.p
        audio = np.asarray(audio, dtype=np.float64)
        if len(audio) == 0:
            return
        t = (np.arange(len(audio)) + self.n_in) / p.fs
        self.n_in += len(audio)
        mixed = audio * np.exp(-2j * np.pi * p.fc * t) * np.sqrt(2.0)
        bb, self.zi = signal.lfilter(self.taps, 1.0, mixed, zi=self.zi)
        self.bb = np.concatenate([self.bb, bb])
        self._process()
        self._trim()

    def _avail(self) -> int:
        return self.bb_base + len(self.bb)

    def _get(self, start: int, length: int) -> np.ndarray:
        s = start - self.bb_base
        return self.bb[s: s + length]

    def _trim(self):
        keep_from = self.search_pos if self.state == self.SEARCH else int(self.anchor)
        keep_from -= 4 * self.p.block_len * self.p.sps
        cut = keep_from - self.bb_base
        if cut > 0:
            self.bb = self.bb[cut:]
            self.bb_base += cut

    # ---------------------------------------------------------------- search
    def _process(self):
        while True:
            if self.state == self.SEARCH:
                if not self._search():
                    return
            else:
                if not self._process_superframe():
                    return

    def _search(self) -> bool:
        p = self.p
        sps = p.sps
        L = p.preamble_len
        tmpl_span = (2 * L) * sps   # both repetitions
        avail = self._avail()
        chunk = 4096
        start = max(self.search_pos, self.bb_base)
        if avail - start < tmpl_span + chunk:
            return False
        seg = self._get(start, min(avail - start, 8 * chunk + tmpl_span))
        n_eval = len(seg) - tmpl_span
        # correlation of each repetition at sample resolution via strided sums
        zc = np.conj(self.zc1)
        c1 = np.zeros(n_eval, dtype=np.complex128)
        c2 = np.zeros(n_eval, dtype=np.complex128)
        e1 = np.zeros(n_eval)
        e2 = np.zeros(n_eval)
        seg_e = np.abs(seg) ** 2
        for k in range(L):
            c1 += zc[k] * seg[k * sps: k * sps + n_eval]
            c2 += zc[k] * seg[(k + L) * sps: (k + L) * sps + n_eval]
            e1 += seg_e[k * sps: k * sps + n_eval]
            e2 += seg_e[(k + L) * sps: (k + L) * sps + n_eval]
        metric = (np.abs(c1) + np.abs(c2)) ** 2 / (2 * L * (e1 + e2) + 1e-12)
        idx = int(np.argmax(metric))
        if metric[idx] >= self.corr_threshold:
            # refine: parabolic interpolation around the peak
            if 0 < idx < n_eval - 1:
                a, b, c = metric[idx - 1], metric[idx], metric[idx + 1]
                den = a - 2 * b + c
                delta = 0.5 * (a - c) / den if abs(den) > 1e-12 else 0.0
            else:
                delta = 0.0
            self.cfo = float(np.angle(c2[idx] * np.conj(c1[idx])) / (2 * np.pi * L / p.baud))
            t0 = start + idx + delta
            self.anchor = t0 + len(self.pre_sym) * sps
            self.next_anchor = self.anchor
            self.sf_index = 0
            self.misses = 0
            self.sro = 0.0
            self.state = self.LOCKED
            self.events.append(f"ACQ at sample {t0:.1f} quality {metric[idx]:.2f} cfo {self.cfo:+.2f} Hz")
            if self.verbose:
                print(self.events[-1])
            self.search_pos = int(t0) + 1
            return True
        self.search_pos = start + n_eval
        return True

    # ------------------------------------------------------------- superframe
    def _sym_positions(self, start_sym: int, count: int) -> np.ndarray:
        """Absolute (fractional) sample positions of `count` symbols starting at symbol start_sym of the SF."""
        step = self.p.sps * (1.0 + self.sro)
        return self.next_anchor + (start_sym + np.arange(count)) * step

    def _extract(self, start_sym: int, count: int) -> np.ndarray:
        pos = self._sym_positions(start_sym, count)
        rel = pos - self.bb_base
        y = fd_interp(self.bb, rel)
        if abs(self.cfo) > 1e-3:
            y = y * np.exp(-2j * np.pi * self.cfo * pos / self.p.fs)
        return y

    def _find_pilot(self, expected: float):
        """Correlate against the pilot block around `expected`; return (anchor, quality)."""
        p = self.p
        sps = p.sps
        W = 2 * sps + 2
        step = sps * (1.0 + self.sro)
        qs = np.empty(2 * W + 1)
        for j, off in enumerate(range(-W, W + 1)):
            pos = expected + off + np.arange(p.block_len) * step
            y = fd_interp(self.bb, pos - self.bb_base)
            c = np.vdot(self.pilot_blk, y)
            qs[j] = abs(c) ** 2 / (np.sum(np.abs(y) ** 2) * p.block_len + 1e-12)
        i = int(np.argmax(qs))
        if 0 < i < len(qs) - 1:
            a, b, c = qs[i - 1], qs[i], qs[i + 1]
            den = a - 2 * b + c
            delta = 0.5 * (a - c) / den if abs(den) > 1e-12 else 0.0
        else:
            delta = 0.0
        return expected + (i - W) + delta, float(qs[i])

    def _process_superframe(self) -> bool:
        p = self.p
        sps = p.sps
        lookahead = 1 if self.sf_index == 0 else 0
        need = int(self.next_anchor + (p.sf_len * (1 + lookahead) + p.block_len) * sps * (1 + self.sro)) + 8 * sps + 4
        if self._avail() < need:
            return False
        if self.next_anchor - 8 * sps < self.bb_base:
            self._lose("buffer underrun")
            return True
        # --- re-anchor on the pilot block -------------------------------
        new_anchor, best_q = self._find_pilot(self.next_anchor)
        if best_q < self.corr_threshold * 0.5:
            self.misses += 1
            if self.verbose:
                print(f"SF {self.sf_index}: pilot miss ({best_q:.2f})")
            if self.misses >= 3 or self.sf_index == 0:
                self._lose("pilot lost")
                return True
            self.next_anchor += p.sf_len * sps * (1.0 + self.sro)
            self.sf_index += 1
            return True
        self.misses = 0
        if self.sf_index == 0:
            # one-time look-ahead: the next pilot block gives the sample-rate offset
            # before any data block of this superframe is equalised
            nxt, q2 = self._find_pilot(new_anchor + p.sf_len * sps)
            if q2 >= self.corr_threshold * 0.5:
                self.sro = float(np.clip((nxt - new_anchor - p.sf_len * sps) / (p.sf_len * sps), -5e-4, 5e-4))
        else:
            expected = self.anchor + p.sf_len * sps
            drift = (new_anchor - expected) / (p.sf_len * sps)
            self.sro = float(np.clip(0.7 * self.sro + 0.3 * drift, -5e-4, 5e-4))
        self.anchor = new_anchor
        self.next_anchor = new_anchor
        step = sps * (1.0 + self.sro)
        # --- channel estimation from the pilot block --------------------
        y = self._extract(p.cp - p.win_lead, p.n_fft)
        Y = np.fft.fft(y)
        H_ls = Y / self.pilot_fft
        h = np.fft.ifft(H_ls)
        h[p.cp:] = 0.0
        H = np.fft.fft(h)
        resid = Y - H * self.pilot_fft
        noise_t = float(np.mean(np.abs(resid) ** 2) / p.n_fft)
        sig = float(np.mean(np.abs(H) ** 2))
        noise_t = max(noise_t, sig * 1e-5)
        self.H = H
        self.noise_t = noise_t
        self.W = np.conj(H) / (np.abs(H) ** 2 + noise_t)
        eq_noise = float(np.mean(np.abs(self.W) ** 2) * noise_t + np.mean(np.abs(self.W * H - 1.0) ** 2))
        snr_db = 10 * np.log10(sig / noise_t)
        self.last_snr_db = snr_db
        self.sf_info.append(SuperframeInfo(self.sf_index, new_anchor, best_q, snr_db, self.sro * 1e6, self.cfo))
        if self.verbose:
            print(f"SF {self.sf_index}: q={best_q:.2f} snr={snr_db:.1f} dB sro={self.sro*1e6:+.0f} ppm")
        # --- data blocks --------------------------------------------------
        for b in range(p.blocks_per_sf):
            start_sym = (b + 1) * p.block_len + p.cp - p.win_lead
            yb = self._extract(start_sym, p.n_fft)
            x = np.fft.ifft(self.W * np.fft.fft(yb))
            xp = x[self.ppos]
            z = xp * np.conj(self.spilots)
            phi0 = np.angle(np.sum(z))
            slope = np.angle(np.sum(z[1:] * np.conj(z[:-1]))) / (self.ppos[1] - self.ppos[0])
            nbar = np.mean(self.ppos)
            x = x * np.exp(-1j * (phi0 + slope * (np.arange(p.n_fft) - nbar)))
            err = float(np.mean(np.abs(x[self.ppos] - self.spilots) ** 2))
            nv = max(eq_noise, 0.8 * err)
            self.blocks.append(EqualisedBlock(self.sf_index, b, x[self.dmask], nv, 10 * np.log10(1.0 / nv)))
        self.next_anchor = new_anchor + p.sf_len * step
        self.sf_index += 1
        if self.stop_after_sf is not None and self.sf_index > self.stop_after_sf:
            self._lose("end of transmission")
        return True

    def _lose(self, why: str):
        self.events.append(f"LOST ({why}) after {self.sf_index} superframes")
        if self.verbose:
            print(self.events[-1])
        self.state = self.SEARCH
        self.search_pos = max(int(self.anchor), self.bb_base)
        self.stop_after_sf = None
        self.H = None

    def end_of_transmission(self):
        """Called by the frame parser when a header announces the final superframe."""
        self.stop_after_sf = self.sf_index - 1
