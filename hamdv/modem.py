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
    """Two Zadoff-Chu chirps with opposite slopes: roots u and N-u.

    A single ZC sequence is a chirp, so a carrier offset is indistinguishable
    from a time shift (its delay-Doppler ambiguity is a ridge along f = u tau).
    The conjugate root has the opposite slope; requiring both correlations to
    peak at the same (tau, f) makes the joint estimate unambiguous.
    """
    z1 = zadoff_chu(p.preamble_len, p.preamble_root)
    z2 = zadoff_chu(p.preamble_len, p.preamble_len - p.preamble_root)
    return np.concatenate([z1, z2])


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
    fll_gain = 0.1

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
        self.zc2 = zadoff_chu(p.preamble_len, p.preamble_len - p.preamble_root)
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
        self.H_next = None            # channel estimate of the next pilot block (interpolation mode)
        self.next_pilot_anchor = None
        # constellation points per data block (set by the link layer from the last header) used for
        # decision-directed channel tracking on fading channels; block 0 is always QPSK
        from .constellation import constellation as _const
        self.block_points: dict = {0: _const(2)[0]}
        self.H_track = None

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
        """Preamble acquisition with a bank of carrier-offset hypotheses.

        The 127-symbol correlation is computed in segments of SEG symbols;
        combining the segment partials with a DFT across the segment index
        evaluates the correlation for every offset hypothesis at once:
            C_f[t] = sum_m c_m[t] exp(-j 2 pi f m SEG T).
        Hypotheses span +- baud/(2 SEG) (+-167 Hz at 1333 Bd); the residual
        is measured from the phase between the two preamble repetitions.
        """
        p = self.p
        sps = p.sps
        L = p.preamble_len
        tmpl_span = (2 * L) * sps
        avail = self._avail()
        chunk = 4096
        start = max(self.search_pos, self.bb_base)
        if avail - start < tmpl_span + chunk:
            return False
        seg = self._get(start, min(avail - start, 8 * chunk + tmpl_span))
        n_eval = len(seg) - tmpl_span
        zc_a = np.conj(self.zc1)
        zc_b = np.conj(self.zc2)
        SEG = 4
        n_seg = L // SEG
        seg_e = np.abs(seg) ** 2
        c1 = np.zeros((n_eval, n_seg), dtype=np.complex128)
        c2 = np.zeros((n_eval, n_seg), dtype=np.complex128)
        e1 = np.zeros(n_eval)
        e2 = np.zeros(n_eval)
        for k in range(n_seg * SEG):
            c1[:, k // SEG] += zc_a[k] * seg[k * sps: k * sps + n_eval]
            c2[:, k // SEG] += zc_b[k] * seg[(k + L) * sps: (k + L) * sps + n_eval]
            e1 += seg_e[k * sps: k * sps + n_eval]
            e2 += seg_e[(k + L) * sps: (k + L) * sps + n_eval]
        if p.cfo_search_hz > 0:
            n_fft = 64
            # hypothesis spacing baud / (SEG * n_fft); keep those inside +- cfo_search_hz
            f_hyp = np.fft.fftfreq(n_fft, d=SEG / p.baud)
            keep = np.abs(f_hyp) <= p.cfo_search_hz
            C1 = np.fft.fft(c1, n=n_fft, axis=1)[:, keep]
            C2 = np.fft.fft(c2, n=n_fft, axis=1)[:, keep]
            f_hyp = f_hyp[keep]
        else:
            C1 = c1.sum(axis=1, keepdims=True)
            C2 = c2.sum(axis=1, keepdims=True)
            f_hyp = np.zeros(1)
        n_used = n_seg * SEG
        metric = (np.abs(C1) + np.abs(C2)) ** 2 / (2 * n_used * (e1 + e2)[:, None] + 1e-12)
        idx, hi = np.unravel_index(int(np.argmax(metric)), metric.shape)
        best = metric[idx, hi]
        if best >= self.corr_threshold:
            col = metric[:, hi]
            if 0 < idx < n_eval - 1:
                a, b, c = col[idx - 1], col[idx], col[idx + 1]
                den = a - 2 * b + c
                delta = 0.5 * (a - c) / den if abs(den) > 1e-12 else 0.0
            else:
                delta = 0.0
            # residual offset from the phase between the two chirps (unambiguous +- baud/(2L)),
            # after removing the phase the coarse hypothesis accumulates over one chirp length
            ph = np.angle(C2[idx, hi] * np.conj(C1[idx, hi])) - 2 * np.pi * f_hyp[hi] * L / p.baud
            ph = (ph + np.pi) % (2 * np.pi) - np.pi
            resid = float(ph / (2 * np.pi * L / p.baud))
            self.cfo = float(f_hyp[hi]) + resid
            t0 = start + idx + delta
            self.anchor = t0 + len(self.pre_sym) * sps
            self.next_anchor = self.anchor
            self.sf_index = 0
            self.misses = 0
            self.sro = 0.0
            self.H_next = None
            self.state = self.LOCKED
            self.events.append(f"ACQ at sample {t0:.1f} quality {best:.2f} cfo {self.cfo:+.2f} Hz")
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
            if abs(self.cfo) > 1e-3:
                y = y * np.exp(-2j * np.pi * self.cfo * pos / self.p.fs)
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

    def _estimate_channel(self, anchor: float):
        """LS channel estimate from the pilot block at `anchor`, projected onto the short-IR subspace."""
        p = self.p
        saved = self.next_anchor
        self.next_anchor = anchor
        y = self._extract(p.cp - p.win_lead, p.n_fft)
        self.next_anchor = saved
        Y = np.fft.fft(y)
        H_ls = Y / self.pilot_fft
        h = np.fft.ifft(H_ls)
        h[p.cp:] = 0.0
        H = np.fft.fft(h)
        resid = Y - H * self.pilot_fft
        noise_t = float(np.mean(np.abs(resid) ** 2) / p.n_fft)
        sig = float(np.mean(np.abs(H) ** 2))
        return H, max(noise_t, sig * 1e-5), sig

    def _process_superframe(self) -> bool:
        p = self.p
        sps = p.sps
        lookahead = 1 if (self.sf_index == 0 or p.interp_channel) else 0
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
            if self.misses >= (6 if p.interp_channel else 3) or self.sf_index == 0:
                self._lose("pilot lost")
                return True
            self.next_anchor += p.sf_len * sps * (1.0 + self.sro)
            self.sf_index += 1
            return True
        self.misses = 0
        nxt_anchor = None
        if p.interp_channel and self.sf_index > 0:
            # fading channel: the correlation peak hops between propagation paths, so the
            # anchor follows the clock prediction and is only nudged (the CP window absorbs
            # path-delay differences); the drift loop is slow and clipped
            nudge = float(np.clip(new_anchor - self.next_anchor, -0.3 * sps, 0.3 * sps))
            new_anchor = self.next_anchor + nudge
        if self.sf_index == 0 or p.interp_channel:
            # look-ahead: the next pilot block gives the sample-rate offset (and, in
            # interpolation mode, the channel at the end of this superframe)
            predicted = new_anchor + p.sf_len * sps * (1.0 + self.sro)
            nxt, q2 = self._find_pilot(predicted)
            if q2 >= self.corr_threshold * 0.5:
                drift = (nxt - new_anchor - p.sf_len * sps) / (p.sf_len * sps)
                if p.interp_channel:
                    drift = float(np.clip(drift, -1.5e-4, 1.5e-4))
                    self.sro = float(np.clip(drift if self.sf_index == 0 else 0.9 * self.sro + 0.1 * drift, -5e-4, 5e-4))
                    # the look-ahead channel must share the data blocks' timing reference: on a
                    # multipath channel the correlation peak hops between paths, so use the
                    # clock prediction (the CP window absorbs the path-delay difference)
                    nxt_anchor = new_anchor + p.sf_len * sps * (1.0 + self.sro)
                else:
                    self.sro = float(np.clip(drift if self.sf_index == 0 else 0.7 * self.sro + 0.3 * drift, -5e-4, 5e-4))
                    nxt_anchor = nxt
        else:
            expected = self.anchor + p.sf_len * sps
            drift = (new_anchor - expected) / (p.sf_len * sps)
            self.sro = float(np.clip(0.7 * self.sro + 0.3 * drift, -5e-4, 5e-4))
        self.anchor = new_anchor
        self.next_anchor = new_anchor
        step = sps * (1.0 + self.sro)
        # --- channel estimation from the pilot block(s) -----------------
        H, noise_t, sig = self._estimate_channel(new_anchor)
        H_next = None
        if p.interp_channel and nxt_anchor is not None:
            H_next, noise_n, sig_n = self._estimate_channel(nxt_anchor)
            noise_t = 0.5 * (noise_t + noise_n)
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
        if p.interp_channel:
            self._equalise_tracking(H, H_next, noise_t, eq_noise)
        else:
            for b in range(p.blocks_per_sf):
                start_sym = (b + 1) * p.block_len + p.cp - p.win_lead
                yb = self._extract(start_sym, p.n_fft)
                x = self._phase_correct(np.fft.ifft(self.W * np.fft.fft(yb)))
                xp = x[self.ppos]
                z = xp * np.conj(self.spilots)
                slope = np.angle(np.sum(z[1:] * np.conj(z[:-1]))) / (self.ppos[1] - self.ppos[0])
                if p.track_gain:
                    g = float(np.abs(np.sum(z)) / np.sum(np.abs(self.spilots) ** 2))
                    x = x / max(0.3 + 0.7 * g, 0.2)
                err = float(np.mean(np.abs(x[self.ppos] - self.spilots) ** 2))
                if p.cfo_search_hz > 0 and err < 0.4:
                    self.cfo += float(np.clip(self.fll_gain * slope * p.baud / (2 * np.pi), -0.2, 0.2))
                nv = max(eq_noise, 0.8 * err)
                self.blocks.append(EqualisedBlock(self.sf_index, b, x[self.dmask], nv, 10 * np.log10(1.0 / nv)))
        self.next_anchor = new_anchor + p.sf_len * step
        self.sf_index += 1
        if self.stop_after_sf is not None and self.sf_index > self.stop_after_sf:
            self._lose("end of transmission")
        return True

    def _dd_estimate(self, Yb: np.ndarray, x: np.ndarray, H_prior: np.ndarray, pts: np.ndarray) -> np.ndarray:
        """Decision-directed channel estimate of one block (Bayesian LS with H_prior as prior)."""
        p = self.p
        full = np.empty(p.n_fft, dtype=np.complex128)
        d = x[self.dmask]
        full[self.dmask] = pts[np.argmin(np.abs(d[:, None] - pts[None, :]) ** 2, axis=1)]
        full[self.ppos] = self.spilots
        Xd = np.fft.fft(full)
        lam = 0.1 * np.mean(np.abs(Xd) ** 2)
        h = np.fft.ifft((Yb * np.conj(Xd) + lam * H_prior) / (np.abs(Xd) ** 2 + lam))
        h[p.cp:] = 0.0
        return np.fft.fft(h)

    def _equalise_tracking(self, H, H_next, noise_t, eq_noise_static):
        """Fading-channel equalisation of the data blocks of the current superframe.

        Pass 1: equalise every block with the channel interpolated between the two
        pilot blocks, make decisions, and re-estimate the channel of each block
        (decision directed).  Pass 2: treat the channel as piecewise linear in time
        through the per-block estimates, equalise every block four times with the
        channel at each quarter, and keep the matching quarter of each output.
        That approximates a time-varying equaliser at the cost of a few FFTs.
        """
        p = self.p
        B = p.blocks_per_sf
        Ys, firsts = [], []
        H_end = H_next if H_next is not None else H
        Hc = [None] * B
        for b in range(B):
            start_sym = (b + 1) * p.block_len + p.cp - p.win_lead
            Yb = np.fft.fft(self._extract(start_sym, p.n_fft))
            Ys.append(Yb)
            w = (b + 1) / (B + 1)
            Hb = (1 - w) * H + w * H_end
            if b > 0 and Hc[b - 1] is not None:
                Hb = 0.5 * (Hb + Hc[b - 1])
            Wb = np.conj(Hb) / (np.abs(Hb) ** 2 + noise_t)
            x = self._phase_correct(np.fft.ifft(Wb * Yb))
            pts = self.block_points.get(b)
            Hc[b] = self._dd_estimate(Yb, x, Hb, pts) if pts is not None else Hb
            firsts.append(x)
        # channel samples at block centres: pilot (t=-0.5 .. relative), blocks 0..B-1, next pilot
        knots_t = [-1.0] + [float(b) for b in range(B)] + [float(B)]
        knots_H = [H] + Hc + [H_end]
        n = p.n_fft
        Q = 4
        for b in range(B):
            Yb = Ys[b]
            x = np.empty(n, dtype=np.complex128)
            nv_parts = []
            for q in range(Q):
                tq = b - 0.5 + (q + 0.5) / Q           # time of this quarter in block units
                # piecewise-linear interpolation between the knots
                j = int(np.clip(np.searchsorted(knots_t, tq) - 1, 0, len(knots_t) - 2))
                a = (tq - knots_t[j]) / (knots_t[j + 1] - knots_t[j])
                Hq = (1 - a) * knots_H[j] + a * knots_H[j + 1]
                Wq = np.conj(Hq) / (np.abs(Hq) ** 2 + noise_t)
                xq = np.fft.ifft(Wq * Yb)
                sl = slice(q * n // Q, (q + 1) * n // Q)
                x[sl] = xq[sl]
                nv_parts.append(float(np.mean(np.abs(Wq) ** 2) * noise_t + np.mean(np.abs(Wq * Hq - 1.0) ** 2)))
            x = self._phase_correct(x)
            z = x[self.ppos] * np.conj(self.spilots)
            slope = np.angle(np.sum(z[1:] * np.conj(z[:-1]))) / (self.ppos[1] - self.ppos[0])
            if p.track_gain:
                g = float(np.abs(np.sum(z)) / np.sum(np.abs(self.spilots) ** 2))
                x = x / max(0.3 + 0.7 * g, 0.2)
            err = float(np.mean(np.abs(x[self.ppos] - self.spilots) ** 2))
            if p.cfo_search_hz > 0 and err < 0.4:
                # frequency-locked loop, gated on pilot fit (a moving multipath notch corrupts the
                # slope estimate), small gain, small step
                self.cfo += float(np.clip(self.fll_gain * slope * p.baud / (2 * np.pi), -0.2, 0.2))
            nv = max(float(np.mean(nv_parts)), 0.8 * err)
            self.blocks.append(EqualisedBlock(self.sf_index, b, x[self.dmask], nv, 10 * np.log10(1.0 / nv)))

    def _phase_correct(self, x: np.ndarray) -> np.ndarray:
        """Remove the common phase and the phase slope of a block using the scattered pilots."""
        z = x[self.ppos] * np.conj(self.spilots)
        phi0 = np.angle(np.sum(z))
        slope = np.angle(np.sum(z[1:] * np.conj(z[:-1]))) / (self.ppos[1] - self.ppos[0])
        nbar = np.mean(self.ppos)
        return x * np.exp(-1j * (phi0 + slope * (np.arange(self.p.n_fft) - nbar)))

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
