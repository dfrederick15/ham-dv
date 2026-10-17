"""Minimal ctypes binding of libopus (no Python package needed) for the DV mode.

Opus at 6-16 kbit/s (SILK / hybrid, 16 kHz wideband) is what "sounds like
you are in the room": a 1.2 kbit/s LPC vocoder cannot.  It needs the
throughput of the 2000-baud voice profile or the flat 9600-port profile.

Frames: 20 ms (320 samples at 16 kHz), constant bit rate, in-band FEC on
so a lost frame can be partially recovered from the next one.
"""
from __future__ import annotations

import ctypes
import ctypes.util
from typing import List, Optional

import numpy as np

OPUS_APPLICATION_VOIP = 2048
OPUS_SET_BITRATE_REQUEST = 4002
OPUS_SET_INBAND_FEC_REQUEST = 4012
OPUS_SET_PACKET_LOSS_PERC_REQUEST = 4014
OPUS_SET_VBR_REQUEST = 4006
OPUS_SET_COMPLEXITY_REQUEST = 4010
OPUS_SET_BANDWIDTH_REQUEST = 4008
OPUS_BANDWIDTH_WIDEBAND = 1103

_lib = None


def available() -> bool:
    return _load() is not None


def _load():
    global _lib
    if _lib is not None:
        return _lib
    name = ctypes.util.find_library("opus") or "libopus.so.0"
    try:
        lib = ctypes.CDLL(name)
    except OSError:
        return None
    lib.opus_encoder_create.restype = ctypes.c_void_p
    lib.opus_encoder_create.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
    lib.opus_encode.restype = ctypes.c_int
    lib.opus_encode.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int16), ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
    lib.opus_encoder_ctl.restype = ctypes.c_int
    lib.opus_decoder_create.restype = ctypes.c_void_p
    lib.opus_decoder_create.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
    lib.opus_decode.restype = ctypes.c_int
    lib.opus_decode.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int16), ctypes.c_int, ctypes.c_int]
    _lib = lib
    return lib


class OpusEncoder:
    def __init__(self, bitrate: int = 8000, fs: int = 16000, frame_ms: int = 20, fec: bool = True, loss_pct: int = 20):
        lib = _load()
        if lib is None:
            raise RuntimeError("libopus not found")
        self.lib, self.fs = lib, fs
        self.frame = fs * frame_ms // 1000
        err = ctypes.c_int()
        self.enc = lib.opus_encoder_create(fs, 1, OPUS_APPLICATION_VOIP, ctypes.byref(err))
        if err.value:
            raise RuntimeError(f"opus encoder error {err.value}")
        lib.opus_encoder_ctl(self.enc, OPUS_SET_BITRATE_REQUEST, ctypes.c_int(bitrate))
        lib.opus_encoder_ctl(self.enc, OPUS_SET_VBR_REQUEST, ctypes.c_int(0))            # constant bit rate
        lib.opus_encoder_ctl(self.enc, OPUS_SET_INBAND_FEC_REQUEST, ctypes.c_int(1 if fec else 0))
        lib.opus_encoder_ctl(self.enc, OPUS_SET_PACKET_LOSS_PERC_REQUEST, ctypes.c_int(loss_pct))
        lib.opus_encoder_ctl(self.enc, OPUS_SET_COMPLEXITY_REQUEST, ctypes.c_int(10))
        lib.opus_encoder_ctl(self.enc, OPUS_SET_BANDWIDTH_REQUEST, ctypes.c_int(OPUS_BANDWIDTH_WIDEBAND))
        self.buf = ctypes.create_string_buffer(4000)

    def encode(self, pcm: np.ndarray) -> bytes:
        """pcm: self.frame float samples in [-1, 1] -> opus packet bytes."""
        s = np.clip(pcm * 32767, -32768, 32767).astype(np.int16)
        arr = s.ctypes.data_as(ctypes.POINTER(ctypes.c_int16))
        n = self.lib.opus_encode(self.enc, arr, self.frame, self.buf, 4000)
        if n < 0:
            raise RuntimeError(f"opus_encode error {n}")
        return self.buf.raw[:n]


class OpusDecoder:
    def __init__(self, fs: int = 16000, frame_ms: int = 20):
        lib = _load()
        if lib is None:
            raise RuntimeError("libopus not found")
        self.lib, self.fs = lib, fs
        self.frame = fs * frame_ms // 1000
        err = ctypes.c_int()
        self.dec = lib.opus_decoder_create(fs, 1, ctypes.byref(err))
        if err.value:
            raise RuntimeError(f"opus decoder error {err.value}")
        self.out = (ctypes.c_int16 * (self.frame * 6))()

    def decode(self, packet: Optional[bytes], fec: bool = False) -> np.ndarray:
        """packet=None -> packet loss concealment; fec=True decodes the FEC data of the next packet."""
        if packet is None:
            n = self.lib.opus_decode(self.dec, None, 0, self.out, self.frame, 0)
        else:
            n = self.lib.opus_decode(self.dec, packet, len(packet), self.out, self.frame, 1 if fec else 0)
        if n < 0:
            raise RuntimeError(f"opus_decode error {n}")
        return np.frombuffer(bytes(self.out)[: 2 * n], dtype=np.int16).astype(np.float64) / 32768
