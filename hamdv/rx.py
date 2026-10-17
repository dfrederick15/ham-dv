"""Receiver: radio audio -> demodulator -> superframes -> video frames, speech, text."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

from .framing import (Header, PKT_AUDIO, PKT_VIDEO, PKT_VMETA, PKT_TEXT, PKT_INFO, AUDIO_MODES, OPUS_MODES,
                      decode_superframe, decode_header, decode_payload, DecodedSuperframe, SymbolDeinterleaver)
from .modem import Demodulator, EqualisedBlock
from .params import ModemParams, DEFAULT, DIGITAL_MODES, PROFILES, PAS_RATES
from .audioio import StreamResampler
from .constellation import constellation
from .shaping import pas_mode
from .framing import plan
from .softvideo import SoftDecoder, SoftConfig, unpack_meta
from .videocodec import VideoDecoder
from .interp import dense_field
from .vocoder import VocoderDecoder, unpack_frames, bits_per_frame, FRAME as AFRAME


@dataclass
class FrameInfo:
    callsign: str
    sf_seq: int
    snr_db: float
    analog: bool
    width: int
    height: int
    frame_index: int
    fps: float = 2.0                      # nominal frame rate announced by the transmitter
    field: Optional[np.ndarray] = None    # dense (2, h, w) motion field relative to the previous frame


class Receiver:
    """Audio in (any rate) -> decoded media.  Listens for every waveform profile
    (1600 / 2000 Bd voice, 4000 Bd flat) at once and follows whichever acquires."""

    def __init__(self, p: Optional[ModemParams] = None, verbose: bool = False,
                 on_frame: Optional[Callable[[np.ndarray, FrameInfo], None]] = None,
                 on_audio: Optional[Callable[[np.ndarray], None]] = None,
                 on_text: Optional[Callable[[str, str], None]] = None,
                 profiles: Optional[List[str]] = None, fs_in: int = 8000):
        if p is not None:
            self.profiles = {p.name: p}
        else:
            self.profiles = {name: PROFILES[name] for name in (profiles or list(PROFILES))}
        self.verbose = verbose
        self.fs_in = fs_in
        self.dems: Dict[str, Demodulator] = {n: Demodulator(pp, verbose=verbose) for n, pp in self.profiles.items()}
        self.resamplers: Dict[str, StreamResampler] = {n: StreamResampler(fs_in, pp.fs) for n, pp in self.profiles.items()}
        self.n_events_by: Dict[str, int] = {n: 0 for n in self.profiles}
        self.active: Optional[str] = None
        first = next(iter(self.profiles))
        self.p = self.profiles[first]
        self.dem = self.dems[first]
        self.on_frame = on_frame
        self.on_audio = on_audio
        self.on_text = on_text
        self.pending: Dict[int, List[EqualisedBlock]] = {}
        self.last_header: Optional[Header] = None
        self.last_dispatched: Optional[Header] = None
        self.deil: Optional[SymbolDeinterleaver] = None
        self.pending_hdr: Dict[int, tuple] = {}
        self._il_k: Optional[int] = None
        self._il_last_sf: Optional[int] = None
        self.vdec: Optional[VideoDecoder] = None
        self.sdec: Optional[SoftDecoder] = None
        self.adec: Optional[VocoderDecoder] = None
        self.odec = None                      # Opus decoder (DV modes)
        self.audio_fs = 8000                  # sample rate of the samples given to on_audio
        self.audio_frames_per_sf = 0
        self.frame_index = 0
        self.n_events = 0
        self.status = dict(state="searching", callsign="", snr_db=float("nan"), sf_total=0, sf_ok=0,
                           blocks_ok=0, blocks_total=0, frames=0, audio_frames=0, text="",
                           mode="", video="", last_rx=0.0, sro_ppm=0.0, cfo_hz=0.0, profile="", baud=0)

    # ------------------------------------------------------------------ input
    def feed(self, samples: np.ndarray, fs_in: Optional[int] = None):
        """Feed audio samples (at fs_in, default the rate given at construction)."""
        if fs_in is not None and fs_in != self.fs_in:
            self.fs_in = fs_in
            self.resamplers = {n: StreamResampler(fs_in, pp.fs) for n, pp in self.profiles.items()}
        samples = np.asarray(samples, dtype=np.float64)
        # every profile's demodulator runs all the time; the receiver follows the locked one
        # with the best pilot correlation (a wrong profile can false-lock on a preamble for
        # one superframe, then loses its pilot)
        for name in self.profiles:
            x = samples if self.profiles[name].fs == self.fs_in else self.resamplers[name].process(samples)
            self.dems[name].feed(x)
            self.n_events_by[name] = len(self.dems[name].events)

        def quality(n):
            d = self.dems[n]
            return d.sf_info[-1].corr_quality if d.sf_info else 0.0
        locked = [n for n in self.profiles if self.dems[n].state == Demodulator.LOCKED]
        if self.active is not None and self.dems[self.active].state != Demodulator.LOCKED:
            self.active = None
            self.pending.clear()
            self.status["state"] = "searching"
        if self.active is None and locked:
            self.active = max(locked, key=quality)
            self.p = self.profiles[self.active]
            self.dem = self.dems[self.active]
            self.status["state"] = "locked"
            self.status["profile"] = self.active
            self.status["baud"] = int(round(self.p.baud))
        for name in self.profiles:
            if name != self.active:
                self.dems[name].blocks.clear()
        if self.active is None:
            return
        self.dem = self.dems[self.active]
        if self.dem.blocks:
            for blk in self.dem.blocks:
                self.pending.setdefault(blk.sf_index, []).append(blk)
            self.dem.blocks.clear()
            done = [k for k, v in self.pending.items() if len(v) >= self.p.blocks_per_sf]
            for k in sorted(done):
                blocks = sorted(self.pending.pop(k), key=lambda b: b.block)
                self._handle(blocks)
            # drop stale partial superframes
            for k in list(self.pending):
                if k < self.dem.sf_index - 2:
                    self.pending.pop(k)
        if self.dem.sf_info:
            info = self.dem.sf_info[-1]
            self.status["sro_ppm"] = info.sro_ppm
            self.status["cfo_hz"] = info.cfo_hz

    # ------------------------------------------------------------------ superframe
    def _handle(self, blocks: List[EqualisedBlock]):
        st = self.status
        st["sf_total"] += 1
        hdr, hdr_ok = decode_header(blocks, self.p, self.last_header)
        if hdr is None:
            if self.verbose:
                print("superframe: header undecodable")
            self._conceal_audio()
            return
        st["callsign"] = hdr.callsign
        st["mode"] = DIGITAL_MODES[hdr.dig_mode][0]
        st["video"] = f"{hdr.vid_w}x{hdr.vid_h} {'analog' if hdr.analog_video else 'digital'} {hdr.vid_fps_x10/10:.1f} fps"
        st["last_rx"] = time.time()
        if hdr.is_last:
            self.dem.end_of_transmission()
        new_stream = (self.last_header is None or hdr.tx_id != self.last_header.tx_id
                      or (hdr.vid_w, hdr.vid_h, hdr.analog_video, hdr.frames_per_sf) !=
                      (self.last_header.vid_w, self.last_header.vid_h, self.last_header.analog_video, self.last_header.frames_per_sf))
        if new_stream or (self.vdec is None and self.sdec is None):
            if hdr.analog_video:
                self.sdec = SoftDecoder(SoftConfig(hdr.vid_w, hdr.vid_h, gop=hdr.frames_per_sf))
                self.vdec = None
            else:
                self.vdec = VideoDecoder(hdr.vid_w, hdr.vid_h)
                self.sdec = None
        amode = AUDIO_MODES.get(hdr.audio_mode, 0)
        if amode in OPUS_MODES:
            if self.odec is None:
                from .opuscodec import OpusDecoder
                self.odec = OpusDecoder(fs=16000, frame_ms=20)
            self.audio_fs = 16000
        elif amode and (self.adec is None or self.adec.mode != amode):
            self.adec = VocoderDecoder(amode)
            self.audio_fs = 8000
        # tell the demodulator which constellation each block carries (decision-directed tracking)
        pl = plan(hdr, self.p)
        pts = pas_mode(PAS_RATES[hdr.dig_mode]).points if hdr.dig_mode in PAS_RATES else constellation(DIGITAL_MODES[hdr.dig_mode][1])[0]
        hints = {0: constellation(2)[0]}
        for b in pl.digital_blocks:
            hints[b] = pts
        self.dem.block_points = hints
        D = hdr.interleave
        if D == 1 or not pl.digital_blocks:
            self.deil = None
            self._dispatch(decode_payload(hdr, blocks, self.p, hdr_ok), new_stream)
        else:
            n = len(pl.digital_blocks) * self.p.data_sym_per_block
            # superframe index for deinterleaving: the header's sequence number when it decoded,
            # otherwise continued from the demodulator's own counter (contiguous across skipped
            # pilots); the deinterleaver survives a re-acquisition so codewords straddling a
            # fade are decoded with erasures instead of being thrown away
            if self.deil is None or self.deil.D != D or self.deil.n != n or new_stream:
                self.deil = SymbolDeinterleaver(D, n)
                self.pending_hdr = {}
                self._il_k = None
            if hdr_ok:
                k = hdr.sf_seq
            elif self._il_k is not None and self._il_last_sf is not None and blocks[0].sf_index > self._il_last_sf:
                k = self._il_k + (blocks[0].sf_index - self._il_last_sf)
            elif self._il_k is not None:
                k = self._il_k + 1
            else:
                k = hdr.sf_seq
            self._il_k, self._il_last_sf = k, blocks[0].sf_index
            self.pending_hdr[k] = (hdr, hdr_ok, blocks)
            flat = np.concatenate([blocks[b].symbols for b in pl.digital_blocks])
            nv = np.concatenate([np.full(self.p.data_sym_per_block, float(np.mean(blocks[b].noise_var))) for b in pl.digital_blocks])
            for (m, s, nvm) in self.deil.push(k, flat, nv):
                if m not in self.pending_hdr:
                    continue
                hm, okm, blm = self.pending_hdr.pop(m)
                plm = plan(hm, self.p)
                bl = list(blm)
                nsym = self.p.data_sym_per_block
                for i, b in enumerate(plm.digital_blocks):
                    sl = slice(i * nsym, (i + 1) * nsym)
                    bl[b] = EqualisedBlock(m, b, s[sl], nvm[sl], float(10 * np.log10(1.0 / max(np.mean(nvm[sl]), 1e-9))))
                self._dispatch(decode_payload(hm, bl, self.p, okm), False)
            for old in [i for i in self.pending_hdr if i < k - 2 * D]:
                self.pending_hdr.pop(old)
        self.last_header = hdr

    def _dispatch(self, d: DecodedSuperframe, new_stream: bool):
        """Deliver the packets of a decoded (possibly deinterleaved) superframe payload."""
        st = self.status
        hdr = d.header
        st["sf_ok"] += 1 if d.blocks_ok == d.blocks_total else 0
        st["blocks_ok"] += d.blocks_ok
        st["blocks_total"] += d.blocks_total
        st["snr_db"] = d.snr_db
        # audio loss concealment for skipped superframes
        if self.last_dispatched is not None and not new_stream:
            gap = (hdr.sf_seq - self.last_dispatched.sf_seq - 1) & 0xFFFF
            if 0 < gap <= 3:
                for _ in range(gap):
                    self._conceal_audio()
        got_audio = False
        for t, payload in d.packets:
            if t == PKT_AUDIO and len(payload) >= 3 and AUDIO_MODES.get(payload[0], 0) in OPUS_MODES and self.odec is not None:
                n, ln = payload[1], payload[2]
                outs = []
                for i in range(n):
                    pk = payload[3 + i * ln: 3 + (i + 1) * ln]
                    try:
                        outs.append(self.odec.decode(pk if len(pk) == ln else None))
                    except RuntimeError:
                        outs.append(self.odec.decode(None))
                self.audio_frames_per_sf = n
                st["audio_frames"] += n
                got_audio = True
                if self.on_audio is not None and outs:
                    self.on_audio(np.concatenate(outs))
            elif t == PKT_AUDIO and self.adec is not None and len(payload) >= 2:
                mode = AUDIO_MODES.get(payload[0], 0)
                n = payload[1]
                if mode != self.adec.mode and mode:
                    self.adec = VocoderDecoder(mode)
                if mode:
                    frames = unpack_frames(payload[2:], n, bits_per_frame(mode))
                    out = np.concatenate([self.adec.decode_frame(f) for f in frames]) if frames else np.zeros(0)
                    self.audio_frames_per_sf = n
                    st["audio_frames"] += n
                    got_audio = True
                    if self.on_audio is not None and len(out):
                        self.on_audio(out)
            elif t == PKT_VIDEO and self.vdec is not None:
                rgb = self.vdec.decode_frame(payload)
                if rgb is not None:
                    st_ = self.vdec.st
                    field = dense_field(self.vdec.last_mvs, self.vdec.last_modes, st_.mbw, st_.mbh, st_.h, st_.w)
                    self._emit(rgb, hdr, d.snr_db, False, field)
            elif t == PKT_VMETA and self.sdec is not None:
                meta = unpack_meta(payload, self.sdec.cfg)
                if meta is not None and len(d.analog_values):
                    for rgb in self.sdec.decode_gop(d.analog_values, d.analog_noise, meta):
                        self._emit(rgb, hdr, d.snr_db, True)
            elif t == PKT_TEXT:
                txt = payload.decode("utf-8", "replace")
                st["text"] = txt
                if self.on_text is not None:
                    self.on_text(hdr.callsign, txt)
        if not got_audio and (self.adec is not None or self.odec is not None) and hdr.audio_mode and d.packets:
            self._conceal_audio()
        self.last_dispatched = hdr

    def _conceal_audio(self):
        if not self.audio_frames_per_sf:
            return
        if self.odec is not None and self.audio_fs == 16000:
            out = np.concatenate([self.odec.decode(None) for _ in range(self.audio_frames_per_sf)])
        elif self.adec is not None:
            out = np.concatenate([self.adec.decode_lost() for _ in range(self.audio_frames_per_sf)])
        else:
            return
        if self.on_audio is not None:
            self.on_audio(out)

    def _emit(self, rgb: np.ndarray, hdr: Header, snr: float, analog: bool, field: Optional[np.ndarray] = None):
        self.frame_index += 1
        self.status["frames"] += 1
        if self.on_frame is not None:
            self.on_frame(rgb, FrameInfo(hdr.callsign, hdr.sf_seq, snr, analog, hdr.vid_w, hdr.vid_h,
                                         self.frame_index, max(hdr.vid_fps_x10, 1) / 10.0, field))
