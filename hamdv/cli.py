"""hamdv command line: sim / encode / decode / tx / rx / devices."""
from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from typing import List, Optional

import numpy as np

from .params import DEFAULT, MODE_BY_NAME, PROFILES, BAUD_PROFILE
from .tx import Transmitter, TxConfig
from .rx import Receiver, FrameInfo
from .channel import simulate, ChannelConfig
from .camera import VideoSource
from .vocoder import synth_test_speech
from .videocodec import psnr


def _size(s: str):
    w, h = s.lower().split("x")
    return int(w), int(h)


def add_media_args(ap: argparse.ArgumentParser):
    ap.add_argument("--callsign", default="N0CALL")
    ap.add_argument("--size", default="96x72", help="video size WxH (multiples of 8), e.g. 64x48, 96x72, 128x96")
    ap.add_argument("--frames-per-sf", type=int, default=3, help="video frames per superframe (3 = ~2 fps)")
    ap.add_argument("--mode", default="16qam-3/4", choices=sorted(MODE_BY_NAME), help="digital block modulation/code rate")
    ap.add_argument("--audio-rate", type=int, default=1200, choices=[0, 1200, 1600, 2400], help="vocoder bit rate (0 = no audio)")
    ap.add_argument("--analog", action="store_true", help="use the analog (SoftCast-style) video transport")
    ap.add_argument("--analog-blocks", type=int, default=3, help="analog data blocks per superframe (analog mode)")
    ap.add_argument("--refresh", type=int, default=12, help="intra refresh period in frames (digital video)")
    ap.add_argument("--level", type=float, default=0.25, help="TX audio RMS level, 0..1")
    ap.add_argument("--text", default=None, help="short station text sent with every superframe")
    ap.add_argument("--video", default="test", help="video source: 'test', camera index, video file or image")
    ap.add_argument("--audio", default="synth", help="audio source: 'synth', 'mic', 'none' or a .wav file")
    ap.add_argument("--baud", type=int, default=1600, choices=[1600, 2000, 4000],
                    help="1600/2000: voice channel (mic/speaker path, repeater); 4000: flat 9600-baud data port (16 kHz audio)")
    ap.add_argument("--long-code", action="store_true", help="one LDPC codeword per superframe (~1 dB gain, all-or-nothing)")
    ap.add_argument("--profile", default=None, choices=sorted(PROFILES),
                    help="waveform profile (overrides --baud): voice, voice2k, flat, ssb (HF single sideband), am (HF AM)")


def make_tx(args) -> Transmitter:
    w, h = _size(args.size)
    cfg = TxConfig(callsign=args.callsign, width=w, height=h, frames_per_sf=args.frames_per_sf,
                   audio_mode=args.audio_rate, dig_mode=args.mode, analog=args.analog,
                   analog_blocks=args.analog_blocks, refresh=args.refresh, level=args.level, text=args.text,
                   baud=args.baud, long_code=args.long_code, profile=args.profile)
    return Transmitter(cfg)


# ------------------------------------------------------------------------------------------------
def generate_transmission(tx: Transmitter, video: VideoSource, audio_src: np.ndarray, seconds: float,
                          sent_frames: Optional[list] = None) -> np.ndarray:
    """Encode `seconds` of media into one transmission (file / simulation use)."""
    n_sf = max(1, int(round(seconds / tx.sf_seconds)))
    parts = [tx.start_transmission()]
    fps = tx.cfg.frames_per_sf / tx.sf_seconds
    for s in range(n_sf):
        frames = []
        for i in range(tx.cfg.frames_per_sf):
            t = (s * tx.cfg.frames_per_sf + i) / fps
            f = video.read(t)
            frames.append(f)
        na = tx.sf_audio_samples
        a = audio_src[s * na:(s + 1) * na] if tx.aenc is not None else np.zeros(0)
        if tx.aenc is not None and len(a) < na:
            a = np.concatenate([a, np.zeros(na - len(a))])
        parts.append(tx.superframe(a, frames, last=(s == n_sf - 1)))
        if sent_frames is not None:
            sent_frames.extend(f for f, ok in zip(frames, tx.last_sent_mask) if ok)
        print(f"\r  superframe {s + 1}/{n_sf}  q={tx.stats['q']} frames={tx.stats['frames_sent']} dropped={tx.stats['frames_dropped']}", end="", flush=True)
    parts.append(tx.end_transmission())
    print()
    return np.concatenate(parts)


def load_audio_source(spec: str, seconds: float) -> np.ndarray:
    from .audioio import read_wav
    if spec in ("synth", None, ""):
        return synth_test_speech(seconds + 2)
    if spec == "none":
        return np.zeros(int((seconds + 2) * 8000))
    return read_wav(spec)


class FrameSink:
    """Collects decoded frames, writes PNGs and a contact sheet / MP4."""

    def __init__(self, out_dir: Optional[str], upscale: int = 4):
        self.out_dir = out_dir
        self.frames: List[np.ndarray] = []
        self.infos: List[FrameInfo] = []
        self.upscale = upscale
        self.fps = 2.0
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

    def __call__(self, rgb: np.ndarray, info: FrameInfo):
        self.frames.append(rgb)
        self.infos.append(info)
        self.fps = info.fps
        if self.out_dir:
            from PIL import Image
            im = Image.fromarray(rgb).resize((rgb.shape[1] * self.upscale, rgb.shape[0] * self.upscale), Image.BILINEAR)
            im.save(os.path.join(self.out_dir, f"rx_{info.frame_index:05d}.png"))

    def finish(self, fps: float, originals: Optional[List[np.ndarray]] = None):
        if not self.out_dir or not self.frames:
            return
        from PIL import Image
        n = len(self.frames)
        cols = min(6, n)
        rows = (n + cols - 1) // cols
        h, w = self.frames[0].shape[:2]
        sheet = Image.new("RGB", (cols * w * 2, rows * h * 2), (0, 0, 0))
        for i, f in enumerate(self.frames):
            im = Image.fromarray(f).resize((w * 2, h * 2), Image.BILINEAR)
            sheet.paste(im, ((i % cols) * w * 2, (i // cols) * h * 2))
        sheet.save(os.path.join(self.out_dir, "contact_sheet.png"))
        if originals:
            m = min(len(originals), n)
            pair = Image.new("RGB", (w * 4, h * 2 * ((m + 1) // 2)), (0, 0, 0))
            for i in range(m):
                a = Image.fromarray(originals[i]).resize((w * 2, h * 2), Image.BILINEAR)
                b = Image.fromarray(self.frames[i]).resize((w * 2, h * 2), Image.BILINEAR)
                pair.paste(a, (0, (i // 2) * h * 2)) if i % 2 == 0 else pair.paste(a, (w * 2, (i // 2) * h * 2))
            # simpler: side-by-side of the last 4 frames
            sbs = Image.new("RGB", (w * 4, h * 2 * min(m, 4)), (0, 0, 0))
            for j, i in enumerate(range(max(0, m - 4), m)):
                sbs.paste(Image.fromarray(originals[i]).resize((w * 2, h * 2), Image.BILINEAR), (0, j * h * 2))
                sbs.paste(Image.fromarray(self.frames[i]).resize((w * 2, h * 2), Image.BILINEAR), (w * 2, j * h * 2))
            sbs.save(os.path.join(self.out_dir, "original_vs_received.png"))
        # motion-interpolated 30 fps sequence (what the live viewer shows)
        from .interp import smooth_sequence
        smooth = smooth_sequence(self.frames, [i.field for i in self.infos], fps, 30.0)
        sdir = os.path.join(self.out_dir, "smooth30")
        os.makedirs(sdir, exist_ok=True)
        for k, f in enumerate(smooth):
            Image.fromarray(f).resize((w * self.upscale, h * self.upscale), Image.BILINEAR).save(os.path.join(sdir, f"f_{k:05d}.png"))
        # mp4 via ffmpeg if available
        try:
            import subprocess
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", f"{fps:.3f}", "-i",
                            os.path.join(self.out_dir, "rx_%05d.png"), "-pix_fmt", "yuv420p",
                            os.path.join(self.out_dir, "received.mp4")], check=False, timeout=120)
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "30", "-i",
                            os.path.join(sdir, "f_%05d.png"), "-pix_fmt", "yuv420p",
                            os.path.join(self.out_dir, "received_smooth30.mp4")], check=False, timeout=300)
        except Exception:
            pass


def report(rx: Receiver, sink: FrameSink, originals: Optional[List[np.ndarray]] = None):
    s = rx.status
    print(f"  profile {s['profile'] or '-'} ({s['baud']} Bd)")
    print(f"  station {s['callsign']!r}  SNR {s['snr_db']:.1f} dB  superframes OK {s['sf_ok']}/{s['sf_total']}  "
          f"blocks OK {s['blocks_ok']}/{s['blocks_total']}  frames {s['frames']}  audio frames {s['audio_frames']}  "
          f"drift {s['sro_ppm']:+.0f} ppm")
    if originals and sink.frames:
        m = min(len(originals), len(sink.frames))
        ps = [psnr(sink.frames[i], originals[i]) for i in range(m)]
        if ps:
            print(f"  video PSNR (RGB, vs source): mean {np.mean(ps):.1f} dB, last {ps[-1]:.1f} dB")
    if s["text"]:
        print(f"  text: {s['text']}")


# ------------------------------------------------------------------------------------------------
def cmd_sim(args):
    tx = make_tx(args)
    w, h = _size(args.size)
    video = VideoSource(args.video, w, h)
    audio_src = load_audio_source(args.audio, args.seconds)
    sent = []
    flat = args.flat or args.baud == 4000
    hf = tx.p.name if tx.p.name in ("ssb", "am") else ""
    chan_name = {"ssb": f"HF SSB ({args.hf_condition}, mistuned {args.cfo:+.0f} Hz)",
                 "am": f"HF AM ({args.hf_condition}, {'sync' if args.am_sync else 'envelope'} detector)"}.get(
        hf, 'flat 9600-port' if flat else 'voice repeater')
    print(f"HamDV simulation: profile {tx.p.name} {tx.p.baud:.0f} Bd{' long code' if args.long_code else ''}, "
          f"{args.mode}, {w}x{h}, {'analog' if args.analog else 'digital'} video, audio {args.audio_rate} bps, "
          f"{chan_name} channel, SNR {args.snr} dB, drift {args.sro} ppm")
    t0 = time.time()
    tx_audio = generate_transmission(tx, video, audio_src, args.seconds, sent)
    t_enc = time.time() - t0
    rx_audio = simulate(tx_audio, ChannelConfig(fs=tx.p.fs, flat=flat, snr_db=args.snr, sro_ppm=args.sro,
                                                limiter_drive=args.limiter, id_tone=args.id_tone,
                                                impulse_rate=args.impulses, seed=args.seed,
                                                hf=hf, hf_condition=args.hf_condition, cfo_hz=args.cfo,
                                                am_sync=args.am_sync))
    sink = FrameSink(args.out_dir)
    audio_out = []
    rx = Receiver(verbose=args.verbose, on_frame=sink, on_audio=audio_out.append,
                  on_text=lambda c, t: None, fs_in=tx.p.fs)
    t0 = time.time()
    for i in range(0, len(rx_audio), 4000):
        rx.feed(rx_audio[i:i + 4000])
    rx.feed(np.zeros(2 * tx.p.fs))
    t_dec = time.time() - t0
    fps = tx.cfg.frames_per_sf / tx.sf_seconds
    sink.finish(fps, sent)
    sink_fps = fps
    pl_bits = tx.stats["video_bits"] + tx.stats["audio_bits"]
    print(f"TX: {len(tx_audio)/8000:.1f} s of audio, {tx.stats['sf']} superframes, video {tx.stats['video_bits']/(tx.stats['sf']*tx.sf_seconds):.0f} bps, "
          f"audio {tx.stats['audio_bits']/(tx.stats['sf']*tx.sf_seconds):.0f} bps, PAPR {20*np.log10(np.abs(tx_audio).max()/np.sqrt(np.mean(tx_audio**2))):.1f} dB "
          f"(encode {t_enc:.1f} s, decode {t_dec:.1f} s)")
    print("RX:")
    report(rx, sink, sent)
    if args.out_dir:
        from .audioio import WavWriter
        if audio_out:
            ww = WavWriter(os.path.join(args.out_dir, "received_audio.wav"))
            ww.write(np.concatenate(audio_out))
            ww.close()
        ww = WavWriter(os.path.join(args.out_dir, "tx_audio.wav"), tx.p.fs)
        ww.write(tx_audio)
        ww.close()
        ww = WavWriter(os.path.join(args.out_dir, "rx_audio.wav"), tx.p.fs)
        ww.write(rx_audio / max(1.0, np.abs(rx_audio).max()))
        ww.close()
        print(f"  outputs written to {args.out_dir}/ (rx_*.png, contact_sheet.png, original_vs_received.png, received.mp4, "
              f"received_smooth30.mp4 = motion-interpolated 30 fps as shown by the viewer, *.wav)")


def cmd_encode(args):
    from .audioio import WavWriter
    tx = make_tx(args)
    w, h = _size(args.size)
    video = VideoSource(args.video, w, h)
    audio_src = load_audio_source(args.audio, args.seconds)
    tx_audio = generate_transmission(tx, video, audio_src, args.seconds)
    ww = WavWriter(args.out, tx.p.fs)
    ww.write(tx_audio)
    ww.close()
    print(f"wrote {args.out}: {len(tx_audio)/tx.p.fs:.1f} s at {tx.p.fs} Hz, {tx.stats['sf']} superframes, {args.baud} Bd")


def cmd_decode(args):
    from .audioio import read_wav, WavWriter
    rx_audio = read_wav(args.inp, 16000)
    sink = FrameSink(args.out_dir)
    audio_out = []
    rx = Receiver(verbose=args.verbose, on_frame=sink, on_audio=audio_out.append,
                  on_text=lambda c, t: print(f"  [{c}] {t}"), fs_in=16000)
    for i in range(0, len(rx_audio), 8000):
        rx.feed(rx_audio[i:i + 8000])
    rx.feed(np.zeros(32000))
    sink.finish(sink.fps)
    report(rx, sink)
    if args.out_dir and audio_out:
        ww = WavWriter(os.path.join(args.out_dir, "received_audio.wav"))
        ww.write(np.concatenate(audio_out))
        ww.close()


def cmd_devices(args):
    from .audioio import list_devices
    print(list_devices())


def cmd_tx(args):
    from .audioio import AudioInput, AudioOutput, WavWriter, PTT
    tx = make_tx(args)
    w, h = _size(args.size)
    video = VideoSource(args.video, w, h)
    if args.out_wav:
        out = WavWriter(args.out_wav, tx.p.fs)
    else:
        out = AudioOutput(device=args.out_device, fs_dev=args.device_rate, fs_internal=tx.p.fs)
    mic = None
    file_audio = None
    if tx.aenc is not None:
        if args.audio == "mic":
            mic = AudioInput(device=args.in_device, fs_dev=args.device_rate, fs_internal=8000)
        else:
            file_audio = load_audio_source(args.audio, args.seconds or 3600)
    ptt = PTT(args.ptt)
    fps = tx.cfg.frames_per_sf / tx.sf_seconds
    print(f"HamDV TX {tx.cfg.callsign}: {args.baud} Bd {args.mode}{' long code' if args.long_code else ''}, {w}x{h} "
          f"{'analog' if args.analog else 'digital'} video {fps:.1f} fps, audio {args.audio_rate} bps"
          f"{' (flat 9600-port audio)' if args.baud == 4000 else ''}. Ctrl-C to stop.")
    stop = threading.Event()
    t_start = time.time()
    sf_counter = 0
    try:
        while not stop.is_set():
            # one transmission (PTT period), limited by the repeater time-out timer
            ptt.key()
            out.write(tx.start_transmission())
            t_key = time.time()
            last = False
            while not last:
                frames = []
                t_sf0 = time.time()
                for i in range(tx.cfg.frames_per_sf):
                    # spread frame grabs over the superframe period for live sources
                    if mic is not None and video.kind == "camera":
                        target = t_sf0 + (i + 0.5) * tx.sf_seconds / tx.cfg.frames_per_sf
                        while time.time() < target:
                            time.sleep(0.01)
                    frames.append(video.read((sf_counter * tx.cfg.frames_per_sf + i) / fps))
                na = tx.sf_audio_samples
                if mic is not None:
                    a = mic.read(na)
                elif file_audio is not None:
                    a = file_audio[sf_counter * na:(sf_counter + 1) * na]
                    if len(a) < na:
                        a = np.concatenate([a, np.zeros(na - len(a))])
                else:
                    a = np.zeros(0)
                sf_counter += 1
                elapsed = time.time() - t_key
                last = (elapsed + 2 * tx.sf_seconds > args.tot) or (args.seconds and (time.time() - t_start) >= args.seconds)
                out.write(tx.superframe(a, frames, last=last))
                print(f"\r  SF {sf_counter}  q={tx.stats['q']}  frames={tx.stats['frames_sent']} dropped={tx.stats['frames_dropped']}  "
                      f"video {tx.stats['video_bits']/(tx.stats['sf']*tx.sf_seconds):.0f} bps", end="", flush=True)
                # for a live mic the output buffer stays ~1 superframe ahead; for files keep it bounded
                while mic is None and hasattr(out, "pending_seconds") and out.pending_seconds() > 2.5 * tx.sf_seconds:
                    time.sleep(0.05)
            out.write(tx.end_transmission())
            if hasattr(out, "drain"):
                out.drain()
            ptt.unkey()
            print("\n  unkeyed (time-out guard / end)")
            if args.seconds and (time.time() - t_start) >= args.seconds:
                break
            time.sleep(args.gap)
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        try:
            out.write(tx.end_transmission())
        except Exception:
            pass
        ptt.close()
        if mic is not None:
            mic.close()
        out.close()
        video.close()


def cmd_rx(args):
    from .audioio import AudioInput, AudioOutput, read_wav
    from .viewer import Viewer
    viewer = Viewer(args.http, smooth_fps=args.smooth_fps) if args.http else None
    spk = AudioOutput(device=args.out_device, fs_dev=args.device_rate) if args.out_device is not None or not args.no_speaker else None
    sink = FrameSink(args.save_dir) if args.save_dir else None

    def on_frame(rgb, info):
        if viewer:
            viewer.push_frame(rgb, info.fps, info.field)
        if sink:
            sink(rgb, info)
        if args.window:
            import cv2
            big = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), (rgb.shape[1] * 4, rgb.shape[0] * 4), interpolation=cv2.INTER_LINEAR)
            cv2.imshow(f"HamDV RX {info.callsign}", big)
            cv2.waitKey(1)

    def on_audio(samples):
        if spk is not None:
            spk.write(samples * args.volume)

    rx = Receiver(verbose=args.verbose, on_frame=on_frame, on_audio=on_audio,
                  on_text=lambda c, t: print(f"\n  [{c}] {t}"), fs_in=16000)
    if args.in_wav:
        src = read_wav(args.in_wav, 16000)
        print(f"HamDV RX from {args.in_wav}")
        for i in range(0, len(src), 8000):
            rx.feed(src[i:i + 8000])
            if viewer:
                viewer.update_status(rx.status)
            if spk is not None:
                while spk.pending_seconds() > 1.0:
                    time.sleep(0.05)
        rx.feed(np.zeros(32000))
    else:
        mic = AudioInput(device=args.in_device, fs_dev=args.device_rate, fs_internal=16000)
        print(f"HamDV RX listening on audio device {args.in_device}" + (f", viewer at http://localhost:{args.http}/" if viewer else "") + ". Ctrl-C to stop.")
        try:
            while True:
                rx.feed(mic.read(None, timeout=0.2))
                if viewer:
                    viewer.update_status(rx.status)
                s = rx.status
                print(f"\r  {s['state']:9s} {s['profile'] or '':8s} {s['callsign']:8s} SNR {s['snr_db']:5.1f} dB  SF {s['sf_ok']}/{s['sf_total']}  frames {s['frames']}   ", end="", flush=True)
                time.sleep(0.05)
        except KeyboardInterrupt:
            print("\nstopping")
        finally:
            mic.close()
    report(rx, sink or FrameSink(None))
    if sink:
        sink.finish(sink.fps)
    if spk is not None:
        spk.close()


# ------------------------------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(prog="hamdv", description="Live video and audio over an amateur radio FM voice channel")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("sim", help="simulate a transmission through a modelled FM repeater channel")
    add_media_args(s)
    s.add_argument("--seconds", type=float, default=12.0)
    s.add_argument("--snr", type=float, default=20.0, help="channel SNR in dB (audio band)")
    s.add_argument("--sro", type=float, default=50.0, help="sound-card sample-rate offset in ppm")
    s.add_argument("--limiter", type=float, default=0.5, help="TX audio limiter drive (0 = linear, 1 = heavy)")
    s.add_argument("--id-tone", action="store_true", help="mix a repeater CW ID over the signal")
    s.add_argument("--impulses", type=float, default=0.0, help="impulse noise events per second")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--flat", action="store_true", help="model the flat 9600-port path instead of the voice repeater")
    s.add_argument("--hf-condition", default="good", choices=["none", "good", "moderate", "poor", "flutter"],
                   help="HF ionospheric condition (Watterson two-path model) for the ssb/am profiles")
    s.add_argument("--cfo", type=float, default=0.0, help="SSB receiver mistuning in Hz")
    s.add_argument("--am-sync", action="store_true", help="AM: synchronous detector instead of envelope detector")
    s.add_argument("--out-dir", default=None, help="write received frames, contact sheet, mp4 and wav files here")
    s.add_argument("--verbose", action="store_true")
    s.set_defaults(func=cmd_sim)

    e = sub.add_parser("encode", help="encode media into a transmission .wav (8 kHz) to play into a radio")
    add_media_args(e)
    e.add_argument("--seconds", type=float, default=12.0)
    e.add_argument("--out", required=True)
    e.set_defaults(func=cmd_encode)

    d = sub.add_parser("decode", help="decode a recorded .wav (any sample rate)")
    d.add_argument("inp")
    d.add_argument("--out-dir", default=None)
    d.add_argument("--verbose", action="store_true")
    d.set_defaults(func=cmd_decode)

    v = sub.add_parser("devices", help="list sound devices")
    v.set_defaults(func=cmd_devices)

    t = sub.add_parser("tx", help="transmit live: camera + microphone -> radio")
    add_media_args(t)
    t.add_argument("--out-device", default=None, help="sound device feeding the radio microphone input")
    t.add_argument("--in-device", default=None, help="microphone device (when --audio mic)")
    t.add_argument("--device-rate", type=int, default=48000)
    t.add_argument("--out-wav", default=None, help="write TX audio to a wav instead of a device")
    t.add_argument("--ptt", default=None, help="PTT control: /dev/ttyUSB0:rts, COM3:dtr, or none (VOX)")
    t.add_argument("--tot", type=float, default=120.0, help="max seconds per key-down (repeater time-out guard)")
    t.add_argument("--gap", type=float, default=3.0, help="seconds to stay unkeyed between transmissions")
    t.add_argument("--seconds", type=float, default=0, help="stop after this many seconds (0 = until Ctrl-C)")
    t.set_defaults(func=cmd_tx)

    r = sub.add_parser("rx", help="receive live: radio -> browser viewer + speaker")
    r.add_argument("--in-device", default=None, help="sound device connected to the radio speaker/line out")
    r.add_argument("--out-device", default=None, help="speaker device for decoded audio")
    r.add_argument("--no-speaker", action="store_true")
    r.add_argument("--device-rate", type=int, default=48000)
    r.add_argument("--in-wav", default=None, help="decode from a wav file in real time instead of a device")
    r.add_argument("--http", type=int, default=8080, help="viewer port (0 = off)")
    r.add_argument("--window", action="store_true", help="also show an OpenCV window")
    r.add_argument("--save-dir", default=None)
    r.add_argument("--smooth-fps", type=float, default=30.0, help="viewer display rate with motion interpolation (0 = show real frames only)")
    r.add_argument("--volume", type=float, default=1.0)
    r.add_argument("--verbose", action="store_true")
    r.set_defaults(func=cmd_rx)

    args = ap.parse_args(argv)
    for name in ("in_device", "out_device"):
        if hasattr(args, name) and getattr(args, name) is not None and str(getattr(args, name)).isdigit():
            setattr(args, name, int(getattr(args, name)))
    args.func(args)


if __name__ == "__main__":
    main()
