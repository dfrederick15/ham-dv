import numpy as np
import pytest
from hamdv.tx import Transmitter, TxConfig
from hamdv.rx import Receiver
from hamdv.channel import simulate, ChannelConfig
from hamdv.testvideo import synth_frame
from hamdv.vocoder import synth_test_speech
from hamdv.videocodec import psnr


def run_link(cfg: TxConfig, n_sf=4, snr_db=25, **chan):
    tx = Transmitter(cfg)
    speech = synth_test_speech(n_sf * tx.sf_seconds + 1)
    parts = [tx.start_transmission()]
    sent_frames = []
    for s in range(n_sf):
        frames = [synth_frame((s * cfg.frames_per_sf + i) / (cfg.frames_per_sf / tx.sf_seconds), cfg.width, cfg.height)
                  for i in range(cfg.frames_per_sf)]
        a = speech[s * tx.sf_samples:(s + 1) * tx.sf_samples]
        parts.append(tx.superframe(a, frames, last=(s == n_sf - 1)))
        sent_frames += [f for f, ok in zip(frames, tx.last_sent_mask) if ok]
    parts.append(tx.end_transmission())
    audio = np.concatenate(parts)
    rx_audio = simulate(audio, ChannelConfig(snr_db=snr_db, **chan))
    got_frames, got_audio, texts = [], [], []
    rx = Receiver(on_frame=lambda f, i: got_frames.append((f, i)), on_audio=lambda a: got_audio.append(a),
                  on_text=lambda c, t: texts.append(t))
    for i in range(0, len(rx_audio), 2000):
        rx.feed(rx_audio[i:i + 2000])
    rx.feed(np.zeros(16000))
    return tx, rx, sent_frames, got_frames, got_audio, texts, audio


def test_digital_link_16qam():
    cfg = TxConfig(callsign="K4OZI", text="hello from the shack", dig_mode="16qam-3/4")
    tx, rx, sent, got, aud, texts, audio = run_link(cfg, n_sf=4, snr_db=24, sro_ppm=60)
    assert rx.status["sf_ok"] == 4, rx.status
    assert rx.status["callsign"] == "K4OZI"
    assert texts and texts[0] == "hello from the shack"
    assert len(got) >= 10
    # decoded frames should resemble the transmitted ones
    ps = [psnr(f, sent[i.frame_index - 1]) for f, i in got]
    assert np.mean(ps[3:]) > 23.5, ps
    assert sum(len(a) for a in aud) >= 4 * tx.sf_samples - 240
    # PAPR of the transmitted audio is bounded
    assert np.abs(audio).max() <= 3.0 * cfg.level + 1e-6


def test_analog_link_and_graceful_degradation():
    # analog video with its side information in the most robust digital mode and no voice:
    # the picture must survive down to a few dB SNR and improve smoothly with signal strength
    cfg = TxConfig(callsign="W1AW", analog=True, analog_blocks=4, width=64, height=48, dig_mode="qpsk-1/2", audio_mode=0)
    res = {}
    for snr in (25, 6):
        tx, rx, sent, got, aud, texts, audio = run_link(cfg, n_sf=3, snr_db=snr)
        assert rx.status["sf_total"] == 3
        assert len(got) == 9, len(got)
        res[snr] = np.mean([psnr(f, sent[i.frame_index - 1]) for f, i in got])
    assert res[25] > 22, res
    assert res[6] > 18, res
    assert res[25] >= res[6]


def test_infeasible_layout_is_rejected():
    with pytest.raises(ValueError):
        Transmitter(TxConfig(dig_mode="qpsk-1/2", audio_mode=1200))


def test_qpsk_low_snr():
    cfg = TxConfig(callsign="VK2ABC", dig_mode="qpsk-3/4", width=64, height=48, frames_per_sf=2)
    tx, rx, sent, got, aud, texts, audio = run_link(cfg, n_sf=3, snr_db=8)
    assert rx.status["sf_ok"] >= 2, rx.status
    assert len(got) >= 4


def test_64qam_full_quieting():
    cfg = TxConfig(callsign="JA1XYZ", dig_mode="64qam-3/4", width=128, height=96, frames_per_sf=3)
    tx, rx, sent, got, aud, texts, audio = run_link(cfg, n_sf=3, snr_db=30, limiter_drive=0.3)
    assert rx.status["sf_ok"] == 3, rx.status
    assert len(got) >= 7


def test_256qam_shaped_strong_signal():
    cfg = TxConfig(callsign="K4OZI", dig_mode="256qam-s5.5", width=128, height=96, frames_per_sf=4)
    tx, rx, sent, got, aud, texts, audio = run_link(cfg, n_sf=3, snr_db=32, limiter_drive=0.2)
    assert rx.status["sf_ok"] == 3, rx.status
    assert len(got) >= 9
