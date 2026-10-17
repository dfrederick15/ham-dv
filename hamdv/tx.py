"""Transmitter: camera frames + microphone samples -> superframes -> audio for the radio."""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .framing import (Header, FLAG_LAST, FLAG_ANALOG_VIDEO, FLAG_LONGCODE, FLAG_IL_SHIFT, IL_DEPTHS, AUDIO_MODE_IDX,
                      PKT_AUDIO, PKT_VIDEO, PKT_VMETA, PKT_TEXT, PKT_INFO, pack_packet, plan, build_superframe,
                      SymbolInterleaver)
from .modem import Modulator, preamble_symbols
from .params import ModemParams, DEFAULT, MODE_BY_NAME, PROFILES, BAUD_PROFILE
from .vocoder import FS as AUDIO_FS
from .softvideo import SoftEncoder, SoftConfig, pack_meta
from .videocodec import VideoEncoder
from .vocoder import VocoderEncoder, pack_frames, FRAME as AFRAME
from .framing import OPUS_MODES

PEAK_LIMIT = 3.0          # TX waveform peak limit in multiples of its RMS (~9.5 dB PAPR)


@dataclass
class TxConfig:
    callsign: str = "N0CALL"
    width: int = 96
    height: int = 72
    frames_per_sf: int = 3            # video frames per superframe (3 -> ~2 fps)
    audio_mode: int = 1200            # 0 = no audio
    dig_mode: str = "16qam-3/4"
    analog: bool = False              # analog (SoftCast-style) video transport
    analog_blocks: int = 3            # analog data blocks per superframe when analog=True
    refresh: int = 12                 # intra refresh period (frames), digital video
    level: float = 0.25               # TX audio RMS level (0..1 full scale)
    text: Optional[str] = None        # station text sent once per superframe
    keyup_s: float = 0.4              # silence after PTT before the preamble (repeater key-up)
    baud: int = 1600                  # 1600 / 2000 (voice channel) or 4000 (flat 9600-baud data port)
    long_code: bool = False           # one LDPC codeword over all digital payload blocks (~1 dB gain)
    profile: Optional[str] = None     # explicit waveform profile name (voice, voice2k, flat, ssb, am); overrides baud
    interleave: int = 1               # interleaving depth in superframes (1, 2, 4, 8): time diversity for fading


class Transmitter:
    def __init__(self, cfg: TxConfig, p: Optional[ModemParams] = None):
        self.cfg = cfg
        if p is None:
            p = PROFILES[cfg.profile] if cfg.profile else PROFILES[BAUD_PROFILE[cfg.baud]]
        self.p = p
        if cfg.width % 8 or cfg.height % 8:
            raise ValueError("video size must be a multiple of 8")
        self.dig_mode = MODE_BY_NAME[cfg.dig_mode]
        self.tx_id = random.randrange(256)
        self.sf_seq = 0
        self.mod = Modulator(p)
        bt = 0
        if cfg.analog:
            for b in range(p.blocks_per_sf - cfg.analog_blocks, p.blocks_per_sf):
                if b >= 1:
                    bt |= 1 << b
        self.block_types = bt
        self.audio_buf = np.zeros(0)
        self.aenc = None
        self.opus = None
        self.audio_fs = 8000
        if cfg.audio_mode in OPUS_MODES:
            from .opuscodec import OpusEncoder
            self.opus = OpusEncoder(bitrate=cfg.audio_mode, fs=16000, frame_ms=20, fec=False)
            self.audio_fs = 16000
        elif cfg.audio_mode:
            self.aenc = VocoderEncoder(cfg.audio_mode)
        if cfg.analog:
            self.senc = SoftEncoder(SoftConfig(cfg.width, cfg.height, gop=cfg.frames_per_sf))
            self.venc = None
        else:
            self.venc = VideoEncoder(cfg.width, cfg.height, refresh_period=cfg.refresh)
            self.senc = None
        self.last_frame = np.full((cfg.height, cfg.width, 3), 128, np.uint8)
        if cfg.interleave not in IL_DEPTHS:
            raise ValueError("interleave must be 1, 2, 4 or 8")
        pl = plan(self.header(), p)
        self.il = SymbolInterleaver(cfg.interleave, len(pl.digital_blocks) * p.data_sym_per_block) if cfg.interleave > 1 else None
        need = 0
        if self.aenc is not None:
            need += 7 + (int(np.ceil(self.sf_audio_samples / AFRAME)) * self.aenc.bits_per_frame + 7) // 8
        if self.opus is not None:
            need += 8 + int(np.ceil(self.sf_audio_samples / self.opus.frame)) * (cfg.audio_mode // 400)
        if cfg.text:
            need += 5 + min(len(cfg.text.encode("utf-8")), 120)
        if need > pl.payload_bytes:
            raise ValueError(f"layout {cfg.dig_mode} offers {pl.payload_bytes} bytes per superframe but audio/text need {need}: "
                             f"use a higher-order mode, fewer analog blocks, or --audio-rate 0")
        self.stats = dict(sf=0, video_bits=0, audio_bits=0, frames_sent=0, frames_dropped=0, q=0)
        self.last_sent_mask: List[bool] = []      # per input frame of the last superframe: transmitted?
        self.scale = cfg.level / 0.72      # modulator output RMS for unit-power symbols is ~0.72

    # ----------------------------------------------------------------- sizes
    @property
    def sf_samples(self) -> int:
        """Modem output samples per superframe (at p.fs)."""
        return self.p.sf_samples

    @property
    def sf_audio_samples(self) -> int:
        """Microphone samples (at audio_fs: 8 kHz vocoder, 16 kHz Opus) consumed per superframe."""
        return int(round(self.p.sf_seconds * self.audio_fs))

    @property
    def sf_seconds(self) -> float:
        return self.p.sf_seconds

    def header(self, last: bool = False) -> Header:
        c = self.cfg
        flags = ((FLAG_LAST if last else 0) | (FLAG_ANALOG_VIDEO if c.analog else 0)
                 | (FLAG_LONGCODE if c.long_code else 0) | (IL_DEPTHS.index(c.interleave) << FLAG_IL_SHIFT))
        fps_x10 = int(round(10 * c.frames_per_sf / self.p.sf_seconds))
        return Header(c.callsign, self.tx_id, self.sf_seq, flags, self.dig_mode, self.block_types,
                      c.width, c.height, fps_x10, AUDIO_MODE_IDX[c.audio_mode], c.frames_per_sf, c.refresh)

    # ----------------------------------------------------------------- audio shaping
    def _audio(self, symbols: np.ndarray) -> np.ndarray:
        a = self.mod.modulate(symbols) * self.scale
        lim = PEAK_LIMIT * self.cfg.level
        return np.clip(a, -lim, lim)

    def start_transmission(self) -> np.ndarray:
        """Audio to play right after keying PTT: key-up silence + preamble."""
        self.mod.reset()
        if self.il is not None:
            self.il = SymbolInterleaver(self.cfg.interleave, self.il.n)
        silence = np.zeros(int(self.cfg.keyup_s * self.p.fs))
        return np.concatenate([silence, self._audio(preamble_symbols(self.p))])

    def end_transmission(self) -> np.ndarray:
        tail = self._audio(np.zeros(self.p.tail_sym, dtype=np.complex128))
        return np.concatenate([tail, self.mod.flush() * self.scale, np.zeros(int(0.1 * self.p.fs))])

    # ----------------------------------------------------------------- one superframe
    def superframe(self, audio: np.ndarray, frames: List[np.ndarray], last: bool = False) -> np.ndarray:
        """Encode one superframe. audio: 8 kHz samples captured since the last call
        (ideally sf_audio_samples); frames: up to frames_per_sf RGB frames (h, w, 3).

        With interleaving, a final superframe is followed by depth-1 flush
        superframes (empty) so that the last codewords can be deinterleaved."""
        out = self._superframe(audio, frames, last and self.il is None)
        if last and self.il is not None:
            parts = [out]
            for i in range(self.cfg.interleave - 1):
                parts.append(self._superframe(np.zeros(0), [self.last_frame], last=(i == self.cfg.interleave - 2), flush=True))
            out = np.concatenate(parts)
        return out

    def _superframe(self, audio: np.ndarray, frames: List[np.ndarray], last: bool = False, flush: bool = False) -> np.ndarray:
        c, p = self.cfg, self.p
        hdr = self.header(last)
        if flush:
            syms = build_superframe(hdr, [], np.zeros(0), p, self.il)
            self.sf_seq = (self.sf_seq + 1) & 0xFFFF
            self.stats["sf"] += 1
            return self._audio(syms)
        pl = plan(hdr, p)
        packets: List[bytes] = []
        # --- audio
        if self.opus is not None:
            self.audio_buf = np.concatenate([self.audio_buf, np.asarray(audio, dtype=np.float64)])
            fr = self.opus.frame
            n_frames = len(self.audio_buf) // fr
            pk = [self.opus.encode(self.audio_buf[i * fr:(i + 1) * fr]) for i in range(n_frames)]
            self.audio_buf = self.audio_buf[n_frames * fr:]
            if n_frames:
                ln = max(len(x) for x in pk)
                payload = bytes([AUDIO_MODE_IDX[c.audio_mode], n_frames, ln]) + b"".join(x.ljust(ln, b"\0") for x in pk)
                packets.append(pack_packet(PKT_AUDIO, payload))
                self.stats["audio_bits"] += len(payload) * 8
        if self.aenc is not None:
            self.audio_buf = np.concatenate([self.audio_buf, np.asarray(audio, dtype=np.float64)])
            n_frames = len(self.audio_buf) // AFRAME
            bits = []
            for i in range(n_frames):
                bits.append(self.aenc.encode_frame(self.audio_buf[i * AFRAME:(i + 1) * AFRAME]))
            self.audio_buf = self.audio_buf[n_frames * AFRAME:]
            if n_frames:
                payload = bytes([AUDIO_MODE_IDX[c.audio_mode], n_frames]) + pack_frames(bits)
                packets.append(pack_packet(PKT_AUDIO, payload))
                self.stats["audio_bits"] += len(payload) * 8
        if c.text:
            packets.append(pack_packet(PKT_TEXT, c.text.encode("utf-8")[:120]))
        # --- video
        frames = list(frames)[: c.frames_per_sf]
        while len(frames) < c.frames_per_sf:
            frames.append(frames[-1] if frames else self.last_frame)
        self.last_frame = frames[-1]
        analog_values = np.zeros(0)
        used = sum(len(x) for x in packets)
        self.last_sent_mask = [True] * len(frames)
        if c.analog:
            budget = pl.analog_values
            for _ in range(12):
                vals, meta = self.senc.encode_gop(frames, budget)
                mpkt = pack_packet(PKT_VMETA, pack_meta(meta))
                if used + len(mpkt) <= pl.payload_bytes or budget < 128:
                    break
                budget = int(budget * 0.8)          # side information must fit: send fewer chunks
            analog_values = vals
            if used + len(mpkt) <= pl.payload_bytes:
                packets.append(mpkt)
                used += len(mpkt)
                self.stats["frames_sent"] += len(frames)
                self.stats["video_bits"] += len(mpkt) * 8
            else:
                self.stats["frames_dropped"] += len(frames)
        else:
            budget_bytes = pl.payload_bytes - used - 5 * len(frames)
            remaining_bits = max(budget_bytes, 0) * 8
            for i, f in enumerate(frames):
                per_frame = remaining_bits // (len(frames) - i)
                snap = self.venc.snapshot()
                data, st = self.venc.encode_frame(f, target_bits=per_frame, max_bits=per_frame)
                if len(data) * 8 > remaining_bits:
                    self.venc.restore(snap)       # does not fit: drop it, decoder repeats the previous frame
                    self.stats["frames_dropped"] += 1
                    self.last_sent_mask[i] = False
                    continue
                remaining_bits -= len(data) * 8
                packets.append(pack_packet(PKT_VIDEO, data))
                self.stats["frames_sent"] += 1
                self.stats["video_bits"] += len(data) * 8
                self.stats["q"] = st.q
        syms = build_superframe(hdr, packets, analog_values, p, self.il)
        self.sf_seq = (self.sf_seq + 1) & 0xFFFF
        self.stats["sf"] += 1
        return self._audio(syms)
