"""Signal-processing primitives used by the modem, the codecs and the channel model.

Everything here is plain mathematics: pulse shaping, Zadoff-Chu sequences,
Hadamard matrices, polynomial (Farrow) interpolation, DCT matrices and a
few helpers for resampling.
"""
from __future__ import annotations

import numpy as np
from scipy import signal


# --------------------------------------------------------------------------
# Pulse shaping
# --------------------------------------------------------------------------
def rrc_taps(sps: int, alpha: float, span: int) -> np.ndarray:
    """Root-raised-cosine impulse response, unit energy.

    g(t) = [sin(pi t (1-a)/T) + 4 a t/T cos(pi t (1+a)/T)] /
           [pi t/T (1 - (4 a t/T)^2)]
    with the two removable singularities (t = 0 and t = +-T/(4a)) evaluated
    by their limits.
    """
    n = span * sps
    t = (np.arange(n + 1) - n / 2) / sps  # in symbol periods
    a = alpha
    g = np.zeros_like(t)
    for i, ti in enumerate(t):
        if abs(ti) < 1e-9:
            g[i] = 1.0 - a + 4.0 * a / np.pi
        elif a > 0 and abs(abs(ti) - 1.0 / (4.0 * a)) < 1e-9:
            g[i] = (a / np.sqrt(2.0)) * (
                (1 + 2 / np.pi) * np.sin(np.pi / (4 * a))
                + (1 - 2 / np.pi) * np.cos(np.pi / (4 * a))
            )
        else:
            num = np.sin(np.pi * ti * (1 - a)) + 4 * a * ti * np.cos(np.pi * ti * (1 + a))
            den = np.pi * ti * (1 - (4 * a * ti) ** 2)
            g[i] = num / den
    g /= np.sqrt(np.sum(g ** 2))
    return g


# --------------------------------------------------------------------------
# Sequences with ideal correlation properties
# --------------------------------------------------------------------------
def zadoff_chu(n: int, u: int, q: int = 0) -> np.ndarray:
    """Zadoff-Chu sequence x_u[k] = exp(-j pi u k (k + c_f + 2q) / n), c_f = n mod 2.

    For gcd(u, n) = 1 the sequence has constant amplitude and zero periodic
    autocorrelation (CAZAC), and its DFT is again CAZAC. That makes it ideal
    both for timing acquisition (sharp correlation peak) and for channel
    estimation (no spectral nulls, so a least-squares estimate has uniform
    noise amplification).
    """
    if np.gcd(u, n) != 1:
        raise ValueError("u must be coprime with n")
    k = np.arange(n)
    cf = n % 2
    return np.exp(-1j * np.pi * u * k * (k + cf + 2 * q) / n).astype(np.complex128)


def hadamard(n: int) -> np.ndarray:
    """Orthonormal Sylvester-Hadamard matrix, n a power of two: H H^T = I."""
    if n & (n - 1):
        raise ValueError("n must be a power of two")
    h = np.array([[1.0]])
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    return h / np.sqrt(n)


# --------------------------------------------------------------------------
# Interpolation (Farrow structure, cubic Lagrange)
# --------------------------------------------------------------------------
def farrow_interp(x: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Evaluate the band-limited signal x at fractional sample positions t.

    Uses the 4-tap cubic Lagrange polynomial through samples i-1, i, i+1, i+2
    where i = floor(t) and mu = t - i:
        w(-1) = -mu (mu-1)(mu-2)/6,  w(0) = (mu+1)(mu-1)(mu-2)/2,
        w(1)  = -(mu+1) mu (mu-2)/2, w(2) = (mu+1) mu (mu-1)/6.
    Positions outside the signal are clamped (contribute edge samples).
    """
    t = np.asarray(t, dtype=np.float64)
    i = np.floor(t).astype(np.int64)
    mu = t - i
    n = len(x)
    idx = np.stack([i - 1, i, i + 1, i + 2], axis=0)
    idx = np.clip(idx, 0, n - 1)
    xs = x[idx]
    w0 = -mu * (mu - 1) * (mu - 2) / 6.0
    w1 = (mu + 1) * (mu - 1) * (mu - 2) / 2.0
    w2 = -(mu + 1) * mu * (mu - 2) / 2.0
    w3 = (mu + 1) * mu * (mu - 1) / 6.0
    return w0 * xs[0] + w1 * xs[1] + w2 * xs[2] + w3 * xs[3]


# --------------------------------------------------------------------------
# Polyphase windowed-sinc fractional-delay interpolation
# --------------------------------------------------------------------------
_PP_TAPS = 16
_PP_PHASES = 128


def _polyphase_table(taps: int = _PP_TAPS, phases: int = _PP_PHASES, beta: float = 8.0) -> np.ndarray:
    """Table[phase] = Kaiser-windowed sinc taps for fractional delay phase/phases.

    Row m interpolates x at position i + m/phases using samples
    x[i - taps/2 + 1 ... i + taps/2].
    """
    half = taps // 2
    table = np.zeros((phases + 1, taps))
    win_n = np.arange(taps) - (half - 1)
    for m in range(phases + 1):
        mu = m / phases
        t = win_n - mu
        h = np.sinc(t)
        w = np.kaiser(2 * taps + 1, beta)
        # window centred on mu: evaluate kaiser at fractional positions
        x = (t / half)
        wk = np.i0(beta * np.sqrt(np.clip(1 - x ** 2, 0, None))) / np.i0(beta)
        wk[np.abs(x) > 1] = 0
        h = h * wk
        table[m] = h / np.sum(h)
    return table


_PP_TABLE = _polyphase_table()


def fd_interp(x: np.ndarray, t: np.ndarray) -> np.ndarray:
    """High-accuracy band-limited interpolation of x at fractional positions t.

    16-tap Kaiser-windowed sinc, 128 polyphase branches with linear
    interpolation between adjacent branches.  Gain ripple is below 0.05 dB
    up to 0.3 cycles/sample, which keeps the equaliser's channel estimate
    valid while the sampling phase drifts.
    """
    t = np.asarray(t, dtype=np.float64)
    i = np.floor(t).astype(np.int64)
    mu = t - i
    ph = mu * _PP_PHASES
    p0 = np.floor(ph).astype(np.int64)
    frac = ph - p0
    w = _PP_TABLE[p0] * (1 - frac)[:, None] + _PP_TABLE[p0 + 1] * frac[:, None]   # (n, taps)
    half = _PP_TAPS // 2
    idx = i[:, None] + (np.arange(_PP_TAPS) - (half - 1))[None, :]
    idx = np.clip(idx, 0, len(x) - 1)
    return np.sum(x[idx] * w, axis=1)


def resample_ratio(x: np.ndarray, ratio: float) -> np.ndarray:
    """Resample x by an arbitrary real ratio (output rate = input rate * ratio)."""
    n_out = int(np.floor((len(x) - 9) * ratio))
    t = np.arange(n_out) / ratio + 8.0
    return fd_interp(x, t)


def resample_int(x: np.ndarray, fs_in: int, fs_out: int) -> np.ndarray:
    """Rational resampling between integer sample rates (polyphase FIR)."""
    if fs_in == fs_out:
        return x
    g = np.gcd(fs_in, fs_out)
    return signal.resample_poly(x, fs_out // g, fs_in // g)


# --------------------------------------------------------------------------
# Transforms
# --------------------------------------------------------------------------
def dct_matrix(n: int) -> np.ndarray:
    """Orthonormal DCT-II matrix C (n x n): C C^T = I.

    C[k, i] = s_k sqrt(2/n) cos(pi (2i+1) k / (2n)),  s_0 = 1/sqrt(2), s_k = 1.
    """
    i = np.arange(n)
    k = i[:, None]
    c = np.sqrt(2.0 / n) * np.cos(np.pi * (2 * i + 1) * k / (2 * n))
    c[0, :] /= np.sqrt(2.0)
    return c


def dct3d(vol: np.ndarray) -> np.ndarray:
    """Separable orthonormal 3-D DCT of a (t, y, x) volume."""
    out = vol.astype(np.float64)
    for axis in range(3):
        c = dct_matrix(out.shape[axis])
        out = np.tensordot(c, out, axes=([1], [axis]))
        out = np.moveaxis(out, 0, axis)
    return out


def idct3d(coef: np.ndarray) -> np.ndarray:
    out = coef.astype(np.float64)
    for axis in range(3):
        c = dct_matrix(out.shape[axis]).T
        out = np.tensordot(c, out, axes=([1], [axis]))
        out = np.moveaxis(out, 0, axis)
    return out


# --------------------------------------------------------------------------
# Misc
# --------------------------------------------------------------------------
def db(x: float) -> float:
    return 10.0 * np.log10(max(float(x), 1e-30))


def bandpass_power(x: np.ndarray, fs: int, f_lo: float, f_hi: float) -> float:
    """Mean power of x inside [f_lo, f_hi] (Welch estimate)."""
    f, p = signal.welch(x, fs=fs, nperseg=1024)
    m = (f >= f_lo) & (f <= f_hi)
    return float(np.trapezoid(p[m], f[m]))
