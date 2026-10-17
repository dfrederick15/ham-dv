"""Sound-card input/output at the device rate with streaming resampling to/from 8 kHz, and PTT control."""
from __future__ import annotations

import queue
import threading
import time
import wave
from typing import Optional

import numpy as np
from scipy import signal

INTERNAL_FS = 8000


class StreamResampler:
    """Streaming integer-ratio resampler (FIR low-pass with carried filter state)."""

    def __init__(self, fs_in: int, fs_out: int):
        g = np.gcd(fs_in, fs_out)
        self.up, self.down = fs_out // g, fs_in // g
        cutoff = 0.9 / max(self.up, self.down)
        self.taps = signal.firwin(32 * max(self.up, self.down) + 1, cutoff) * self.up
        self.zi = np.zeros(len(self.taps) - 1)
        self.phase = 0

    def process(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if self.up > 1:
            u = np.zeros(len(x) * self.up)
            u[:: self.up] = x
        else:
            u = x
        y, self.zi = signal.lfilter(self.taps, 1.0, u, zi=self.zi)
        if self.down > 1:
            idx = np.arange(self.phase, len(y), self.down)
            out = y[idx]
            self.phase = (idx[-1] + self.down - len(y)) if len(idx) else (self.phase - len(y)) % self.down
            return out
        return y


def list_devices() -> str:
    import sounddevice as sd
    return str(sd.query_devices())


class AudioInput:
    """Microphone / radio audio input, delivered as 8 kHz float samples."""

    def __init__(self, device=None, fs_dev: int = 48000, blocksize: int = 4800, fs_internal: int = INTERNAL_FS):
        import sounddevice as sd
        self.q: "queue.Queue[np.ndarray]" = queue.Queue()
        self.fs_internal = fs_internal
        self.rs = StreamResampler(fs_dev, fs_internal)
        self.buf = np.zeros(0)
        self.stream = sd.InputStream(device=device, samplerate=fs_dev, channels=1, dtype="float32",
                                     blocksize=blocksize, callback=self._cb)
        self.stream.start()

    def _cb(self, indata, frames, t, status):
        self.q.put(indata[:, 0].copy())

    def read(self, n: Optional[int] = None, timeout: float = 5.0) -> np.ndarray:
        """Return n samples at 8 kHz (blocking), or whatever is available if n is None."""
        deadline = time.time() + timeout
        while n is None or len(self.buf) < n:
            try:
                chunk = self.q.get(timeout=max(0.01, deadline - time.time()))
            except queue.Empty:
                if n is None:
                    break
                if time.time() > deadline:
                    break
                continue
            self.buf = np.concatenate([self.buf, self.rs.process(chunk)])
            if n is None and self.q.empty():
                break
        if n is None:
            out, self.buf = self.buf, np.zeros(0)
            return out
        out, self.buf = self.buf[:n], self.buf[n:]
        if len(out) < n:
            out = np.concatenate([out, np.zeros(n - len(out))])
        return out

    def close(self):
        self.stream.stop()
        self.stream.close()


class AudioOutput:
    """Speaker / radio microphone output fed with 8 kHz float samples."""

    def __init__(self, device=None, fs_dev: int = 48000, blocksize: int = 2400, fs_internal: int = INTERNAL_FS):
        import sounddevice as sd
        self.fs_internal = fs_internal
        self.rs = StreamResampler(fs_internal, fs_dev)
        self.lock = threading.Lock()
        self.buf = np.zeros(0, dtype=np.float32)
        self.stream = sd.OutputStream(device=device, samplerate=fs_dev, channels=1, dtype="float32",
                                      blocksize=blocksize, callback=self._cb)
        self.stream.start()

    def _cb(self, outdata, frames, t, status):
        with self.lock:
            n = min(frames, len(self.buf))
            outdata[:n, 0] = self.buf[:n]
            outdata[n:, 0] = 0
            self.buf = self.buf[n:]

    def write(self, samples_8k: np.ndarray):
        y = self.rs.process(samples_8k).astype(np.float32)
        with self.lock:
            self.buf = np.concatenate([self.buf, y])

    def pending_seconds(self) -> float:
        with self.lock:
            return len(self.buf) / self.stream.samplerate

    def drain(self):
        while self.pending_seconds() > 0.05:
            time.sleep(0.05)

    def close(self):
        self.drain()
        self.stream.stop()
        self.stream.close()


class WavWriter:
    def __init__(self, path: str, fs: int = INTERNAL_FS):
        self.w = wave.open(path, "wb")
        self.w.setnchannels(1)
        self.w.setsampwidth(2)
        self.w.setframerate(fs)

    def write(self, samples: np.ndarray):
        self.w.writeframes((np.clip(samples, -1, 1) * 32767).astype(np.int16).tobytes())

    def pending_seconds(self) -> float:
        return 0.0

    def close(self):
        self.w.close()


def read_wav(path: str, fs_out: int = INTERNAL_FS) -> np.ndarray:
    """Read a mono/stereo wav of any rate; return float samples at fs_out."""
    with wave.open(path, "rb") as w:
        fs = w.getframerate()
        n = w.getnframes()
        ch = w.getnchannels()
        sw = w.getsampwidth()
        raw = w.readframes(n)
    if sw == 2:
        x = np.frombuffer(raw, dtype=np.int16).astype(np.float64) / 32768
    elif sw == 1:
        x = (np.frombuffer(raw, dtype=np.uint8).astype(np.float64) - 128) / 128
    elif sw == 4:
        x = np.frombuffer(raw, dtype=np.int32).astype(np.float64) / 2 ** 31
    else:
        raise ValueError("unsupported sample width")
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    if fs != fs_out:
        g = np.gcd(fs, fs_out)
        x = signal.resample_poly(x, fs_out // g, fs // g)
    return x


class PTT:
    """Push-to-talk via a serial port control line ("/dev/ttyUSB0:rts" or "COM3:dtr"), or none."""

    def __init__(self, spec: Optional[str]):
        self.ser = None
        self.line = "rts"
        if spec and spec.lower() not in ("none", "vox"):
            import serial
            port, _, line = spec.partition(":")
            self.line = (line or "rts").lower()
            self.ser = serial.Serial(port)
            self.unkey()

    def key(self):
        if self.ser:
            setattr(self.ser, self.line, True)

    def unkey(self):
        if self.ser:
            setattr(self.ser, self.line, False)

    def close(self):
        self.unkey()
        if self.ser:
            self.ser.close()
