"""Receiver: radio audio -> demodulator -> superframes -> video frames, speech, text."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

from .framing import (Header, PKT_AUDIO, PKT_VIDEO, PKT_VMETA, PKT_TEXT, PKT_INFO, AUDIO_MODES,
                      decode_superframe, DecodedSuperframe)
from .modem import Demodulator, EqualisedBlock
from .params import ModemParams, DEFAULT, DIGITAL_MODES
from .softvideo import SoftDecoder, SoftConfig, unpack_meta
from .videocodec import VideoDecoder
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


class Receiver:
    def __init__(self, p: ModemParams = DEFAULT, verbose: bool = False,
                 on_frame: Optional[Callable[[np.ndarray, FrameInfo], None]] = None,
                 on_audio: Optional[Callable[[np.ndarray], None]] = None,
                 on_text: Optional[Callable[[str, str], None]] = None):
        self.p = p
        self.verbose = verbose
        self.dem = Demodulator(p, verbose=verbose)
        self.on_frame = on_frame
        self.on_audio = on_audio
        self.on_text = on_text
        self.pending: Dict[int, List[EqualisedBlock]] = {}
        self.last_header: Optional[Header] = None
        self.vdec: Optional[VideoDecoder] = None
        self.sdec: Optional[SoftDecoder] = None
        self.adec: Optional[VocoderDecoder] = None
        self.audio_frames_per_sf = 0
        self.frame_index = 0
        self.n_events = 0
        self.status = dict(state="searching", callsign="", snr_db=float("nan"), sf_total=0, sf_ok=0,
                           blocks_ok=0, blocks_total=0, frames=0, audio_frames=0, text="",
                           mode="", video="", last_rx=0.0, sro_ppm=0.0, cfo_hz=0.0)

    # ------------------------------------------------------------------ input
    def feed(self, samples: np.ndarray):
        self.dem.feed(samples)
        for e in self.dem.events[self.n_events:]:
            self.n_events += 1
            if e.startswith("LOST"):
                self.pending.clear()
                self.status["state"] = "searching"
            elif e.startswith("ACQ"):
                self.status["state"] = "locked"
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
        d = decode_superframe(blocks, self.p, self.last_header)
        if d is None:
            if self.verbose:
                print("superframe: header undecodable")
            self._conceal_audio()
            return
        hdr = d.header
        st["sf_ok"] += 1 if d.blocks_ok == d.blocks_total else 0
        st["blocks_ok"] += d.blocks_ok
        st["blocks_total"] += d.blocks_total
        st["callsign"] = hdr.callsign
        st["snr_db"] = d.snr_db
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
        if amode and (self.adec is None or self.adec.mode != amode):
            self.adec = VocoderDecoder(amode)
        # audio loss concealment for skipped superframes
        if self.last_header is not None and not new_stream:
            gap = (hdr.sf_seq - self.last_header.sf_seq - 1) & 0xFFFF
            if 0 < gap <= 3:
                for _ in range(gap):
                    self._conceal_audio()
        got_audio = False
        for t, payload in d.packets:
            if t == PKT_AUDIO and self.adec is not None and len(payload) >= 2:
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
                    self._emit(rgb, hdr, d.snr_db, False)
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
        if not got_audio and self.adec is not None and hdr.audio_mode:
            self._conceal_audio()
        self.last_header = hdr

    def _conceal_audio(self):
        if self.adec is None or not self.audio_frames_per_sf:
            return
        out = np.concatenate([self.adec.decode_lost() for _ in range(self.audio_frames_per_sf)])
        if self.on_audio is not None:
            self.on_audio(out)

    def _emit(self, rgb: np.ndarray, hdr: Header, snr: float, analog: bool):
        self.frame_index += 1
        self.status["frames"] += 1
        if self.on_frame is not None:
            self.on_frame(rgb, FrameInfo(hdr.callsign, hdr.sf_seq, snr, analog, hdr.vid_w, hdr.vid_h, self.frame_index))
