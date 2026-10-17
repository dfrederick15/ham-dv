"""Simulation of a narrow-band FM voice channel through a repeater.

The audio chain seen by the modem is modelled as:

    [TX radio]  HPF 300 Hz, pre-emphasis, soft limiter, LPF ~3 kHz
    [repeater]  de-emphasis, CTCSS high-pass, LPF, pre-emphasis again
    [RX radio]  de-emphasis, LPF
    + white noise (flat after de-emphasis), optional impulse noise near squelch
    + sample-rate offset between the two sound cards
    + unknown gain, unknown delay, optional repeater ID tone bursts
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal

from .dsp import resample_ratio, bandpass_power

HF_CONDITIONS = {
    # name: (list of (delay_ms, gain_db), doppler spread Hz)   -- CCIR 520 / ITU-R F.1487 style
    "none":     ([(0.0, 0.0)], 0.0),
    "good":     ([(0.0, 0.0), (0.5, 0.0)], 0.1),
    "moderate": ([(0.0, 0.0), (1.0, 0.0)], 0.5),
    "poor":     ([(0.0, 0.0), (2.0, 0.0)], 1.0),
    "flutter":  ([(0.0, 0.0), (0.5, 0.0)], 10.0),
}


@dataclass
class ChannelConfig:
    fs: int = 8000
    flat: bool = False             # 9600-baud data port: flat DC-6 kHz path, no emphasis, no mic limiter
    snr_db: float = 25.0
    sro_ppm: float = 40.0          # sample-rate offset
    gain: float = 0.8
    delay_s: float = 0.137
    limiter_drive: float = 0.5     # 1.0 = TX audio limiter driven into compression
    id_tone: bool = False          # mix a repeater CW ID over the signal
    id_tone_start_s: float = 3.0
    id_tone_level: float = 0.3
    impulse_rate: float = 0.0      # impulses per second (popcorn noise near threshold)
    # HF options
    hf: str = ""                   # "" (not HF), "ssb" or "am"
    hf_condition: str = "good"     # key of HF_CONDITIONS (Watterson two-path model)
    cfo_hz: float = 0.0            # SSB mistuning
    am_depth: float = 0.9          # AM peak modulation index
    am_sync: bool = False          # synchronous (coherent) AM detector instead of envelope
    hpf_hz: float = 300.0
    lpf_hz: float = 3000.0
    seed: int = 0


def _radio_chain(x: np.ndarray, fs: int, cfg: ChannelConfig) -> np.ndarray:
    # TX: CTCSS high-pass (HPF), pre-emphasis (+6 dB/oct), limiter, LPF
    b, a = signal.butter(2, cfg.hpf_hz / (fs / 2), "high")
    x = signal.lfilter(b, a, x)
    pre_b, pre_a = [1.0, -0.95], [1.0]   # ~ +6 dB/octave
    de_b, de_a = [1.0], [1.0, -0.95]     # exact inverse
    x = signal.lfilter(pre_b, pre_a, x)
    x = np.tanh(cfg.limiter_drive * x * 2.0) / (2.0 * cfg.limiter_drive)
    b, a = signal.ellip(5, 0.5, 50, cfg.lpf_hz / (fs / 2))
    x = signal.lfilter(b, a, x)
    # repeater RX de-emphasis then its own TX chain (another HPF/pre/LPF)
    x = signal.lfilter(de_b, de_a, x)
    b, a = signal.butter(2, cfg.hpf_hz / (fs / 2), "high")
    x = signal.lfilter(b, a, x)
    x = signal.lfilter(pre_b, pre_a, x)
    b, a = signal.ellip(5, 0.5, 50, (cfg.lpf_hz - 100) / (fs / 2))
    x = signal.lfilter(b, a, x)
    # receiving radio de-emphasis and audio LPF
    x = signal.lfilter(de_b, de_a, x)
    b, a = signal.butter(4, cfg.lpf_hz / (fs / 2))
    x = signal.lfilter(b, a, x)
    return x


def _flat_chain(x: np.ndarray, fs: int, cfg: ChannelConfig) -> np.ndarray:
    """9600-port path: modulator input -> IF filter -> discriminator output, twice (via a repeater)."""
    # deviation limiter is far away at sensible drive; model a gentle soft clip
    x = np.tanh(0.3 * x * cfg.limiter_drive * 2.0) / (0.3 * cfg.limiter_drive * 2.0)
    b, a = signal.butter(1, 20.0 / (fs / 2), "high")          # AC coupling
    x = signal.lfilter(b, a, x)
    b, a = signal.ellip(5, 0.3, 50, 5800.0 / (fs / 2))        # IF / post-discriminator low-pass
    x = signal.lfilter(b, a, x)
    x = signal.lfilter(b, a, x)                                # repeater receive + transmit
    b, a = signal.butter(1, 20.0 / (fs / 2), "high")
    x = signal.lfilter(b, a, x)
    return x


def _watterson(xa: np.ndarray, fs: int, condition: str, rng) -> np.ndarray:
    """Two-path Watterson fading applied to an analytic signal: y = sum_i g_i(t) x(t - tau_i)
    with g_i complex Gaussian processes of Gaussian Doppler spectrum (2-sigma = spread)."""
    paths, spread = HF_CONDITIONS[condition]
    n = len(xa)
    y = np.zeros(n, dtype=np.complex128)
    for (d_ms, g_db) in paths:
        d = int(round(d_ms * 1e-3 * fs))
        xd = np.concatenate([np.zeros(d, dtype=np.complex128), xa[: n - d]])
        if spread > 0:
            # low-pass filtered complex Gaussian: Gaussian Doppler spectrum with sigma = spread / 2
            sigma_f = spread / 2.0
            m = n
            f = np.fft.fftfreq(m, d=1.0 / fs)
            spec = np.exp(-f ** 2 / (2 * sigma_f ** 2))
            w = rng.standard_normal(m) + 1j * rng.standard_normal(m)
            g = np.fft.ifft(np.fft.fft(w) * spec)
            g /= np.sqrt(np.mean(np.abs(g) ** 2))
        else:
            g = np.ones(n, dtype=np.complex128)
        y += 10 ** (g_db / 20) * g * xd
    y /= np.sqrt(len(paths))
    return y


def _ssb_chain(x: np.ndarray, fs: int, cfg: ChannelConfig, rng) -> np.ndarray:
    """TX SSB filter -> ionosphere (fading, mistuning) -> RX SSB filter."""
    b, a = signal.ellip(6, 0.5, 60, [300.0 / (fs / 2), 2700.0 / (fs / 2)], "band")
    x = signal.lfilter(b, a, x)
    xa = signal.hilbert(x)
    y = _watterson(xa, fs, cfg.hf_condition, rng)
    t = np.arange(len(y)) / fs
    y = y * np.exp(2j * np.pi * cfg.cfo_hz * t)       # receiver mistuning shifts every audio tone
    y = np.real(y)
    y = signal.lfilter(b, a, y)
    return y


def _am_chain(x: np.ndarray, fs: int, cfg: ChannelConfig, rng, snr_db: float):
    """AM: envelope (or synchronous) detection of a double-sideband-with-carrier signal through fading.

    The complex envelope of the RF signal is  s(t) = 1 + m x(t)  (carrier + sidebands); after the
    channel the receiver sees  r(t) = sum_i g_i(t) s(t - tau_i) + n(t).  An envelope detector
    outputs |r(t)|: linear while the paths add in phase, severely distorted when the carrier
    fades out (the classic AM selective-fading distortion).  A synchronous detector outputs
    Re{ r(t) e^{-j arg g_0(t)} }, which is much more robust.
    """
    b, a = signal.butter(4, [300.0 / (fs / 2), 3100.0 / (fs / 2)], "band")
    x = signal.lfilter(b, a, x)
    m = cfg.am_depth / (np.max(np.abs(x)) + 1e-12)
    s = 1.0 + m * x
    paths, spread = HF_CONDITIONS[cfg.hf_condition]
    # path gains applied to the full AM complex envelope (carrier included)
    r = _watterson(s.astype(np.complex128), fs, cfg.hf_condition, rng)
    # noise relative to the sideband power (so snr_db means the same as for the other channels)
    side_p = np.mean((m * x) ** 2) / len(paths)
    nv = side_p / 10 ** (snr_db / 10)
    r = r + np.sqrt(nv / 2) * (rng.standard_normal(len(r)) + 1j * rng.standard_normal(len(r)))
    if cfg.am_sync:
        # coherent detector locked to the (slowly fading) carrier phase: narrow-band PLL approximated
        # by low-pass filtering the received carrier
        bb, ab = signal.butter(2, 30.0 / (fs / 2))
        carrier = signal.filtfilt(bb, ab, r)
        y = np.real(r * np.exp(-1j * np.angle(carrier + 1e-12)))
    else:
        y = np.abs(r)
    y = y - np.mean(y)
    y = signal.lfilter(b, a, y)
    return y


def simulate(audio: np.ndarray, cfg: ChannelConfig) -> np.ndarray:
    """Pass TX audio (float, at cfg.fs) through the modelled channel; returns RX audio."""
    rng = np.random.default_rng(cfg.seed)
    fs = cfg.fs
    x = np.asarray(audio, dtype=np.float64)
    x = np.concatenate([np.zeros(int(cfg.delay_s * fs)), x, np.zeros(int(0.5 * fs))])
    if cfg.hf == "am":
        y = _am_chain(x, fs, cfg, rng, cfg.snr_db) * cfg.gain
        if cfg.sro_ppm:
            y = resample_ratio(y, 1.0 + cfg.sro_ppm * 1e-6)
        return y
    if cfg.hf == "ssb":
        y = _ssb_chain(x, fs, cfg, rng) * cfg.gain
        band = (400.0, 2600.0)
        sig_p = bandpass_power(y, fs, *band)
        noise = rng.standard_normal(len(y))
        b, a = signal.ellip(6, 0.5, 60, [300.0 / (fs / 2), 2700.0 / (fs / 2)], "band")
        noise = signal.lfilter(b, a, noise)
        noise *= np.sqrt(sig_p / bandpass_power(noise, fs, *band) / 10 ** (cfg.snr_db / 10))
        y = y + noise
        if cfg.sro_ppm:
            y = resample_ratio(y, 1.0 + cfg.sro_ppm * 1e-6)
        return y
    if cfg.flat:
        y = _flat_chain(x, fs, cfg) * cfg.gain
        band = (500.0, 5500.0)
        sig_p = bandpass_power(y, fs, *band)
        # discriminator noise rises with frequency (~f^2 power spectrum) with a small flat floor
        white = rng.standard_normal(len(y))
        noise = signal.lfilter([1.0, -1.0], [1.0], white) * 3.0 + 0.3 * white
        b, a = signal.ellip(5, 0.3, 50, 5800.0 / (fs / 2))
        noise = signal.lfilter(b, a, noise)
        noise *= np.sqrt(sig_p / bandpass_power(noise, fs, *band) / 10 ** (cfg.snr_db / 10))
        y = y + noise
        if cfg.sro_ppm:
            y = resample_ratio(y, 1.0 + cfg.sro_ppm * 1e-6)
        return y
    y = _radio_chain(x, fs, cfg) * cfg.gain
    if cfg.id_tone:
        t = np.arange(len(y)) / fs
        start = cfg.id_tone_start_s
        # Morse-ish keyed 1 kHz tone for ~3 s
        key = (((t - start) * 8).astype(int) % 3 != 2) & (t >= start) & (t < start + 3.0)
        y = y + cfg.id_tone_level * np.sin(2 * np.pi * 1000 * t) * key
    # noise: white, then shaped by the receiver audio LPF (de-emphasised FM noise ~ flat)
    sig_p = bandpass_power(y, fs, 500, 2900)
    noise = rng.standard_normal(len(y))
    b, a = signal.butter(4, cfg.lpf_hz / (fs / 2))
    noise = signal.lfilter(b, a, noise)
    noise_p = bandpass_power(noise, fs, 500, 2900)
    noise *= np.sqrt(sig_p / noise_p / 10 ** (cfg.snr_db / 10))
    y = y + noise
    if cfg.impulse_rate > 0:
        n_imp = rng.poisson(cfg.impulse_rate * len(y) / fs)
        for _ in range(n_imp):
            i = rng.integers(0, len(y) - 40)
            y[i: i + 40] += rng.standard_normal(40) * np.sqrt(sig_p) * 4 * np.exp(-np.arange(40) / 8)
    if cfg.sro_ppm:
        y = resample_ratio(y, 1.0 + cfg.sro_ppm * 1e-6)
    return y
