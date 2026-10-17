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


def simulate(audio: np.ndarray, cfg: ChannelConfig) -> np.ndarray:
    """Pass TX audio (float, at cfg.fs) through the modelled channel; returns RX audio."""
    rng = np.random.default_rng(cfg.seed)
    fs = cfg.fs
    x = np.asarray(audio, dtype=np.float64)
    x = np.concatenate([np.zeros(int(cfg.delay_s * fs)), x, np.zeros(int(0.5 * fs))])
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
