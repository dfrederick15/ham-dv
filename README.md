# HamDV — live video and voice over an FM voice repeater

HamDV sends **live, low-rate video with digital voice through an ordinary
amateur-radio FM voice channel**, including through a voice repeater.  The
radio only transmits and receives audio; everything else is software:

* a **single-carrier, frequency-domain-equalised (SC-FDE) waveform** that fits
  in the 300–3000 Hz passband of a voice repeater (1600 baud, 1700 Hz carrier,
  root-raised-cosine pulses, low peak-to-average ratio so the radio's mic
  limiter stays out of the way),
* **quasi-cyclic LDPC** forward error correction with adaptive QPSK / 16-QAM /
  64-QAM,
* a built-in **1200/1600/2400 bit/s LPC vocoder** for speech,
* two video transports:
  * **digital** — a hybrid DCT video codec with motion compensation, golden-
    ratio intra refresh (no key frames: anyone can tune in at any time and the
    picture builds up within a few seconds) and context-adaptive binary
    arithmetic coding;
  * **analog** — a linear "SoftCast" style transport (3-D DCT, power
    allocation, Hadamard mixing, linear least-squares decoding) whose picture
    degrades *gracefully* with signal strength instead of falling off a
    digital cliff;
* a **browser viewer** so every computer on your LAN can watch the station
  that is transmitting, with callsign, SNR and link statistics.

Typical throughput through a repeater with a full-quieting signal:
16-QAM rate 3/4 carries about 3.4 kbit/s net, used as ~1.2 kbit/s voice plus
~1.8–2 kbit/s video (96×72 at ~2 frames/s), or at 64-QAM about 5 kbit/s.

The full mathematics of the design is in [DESIGN.md](DESIGN.md).

## Quick start

```bash
pip install -e ".[all]"        # numpy, scipy, sounddevice, pyserial, opencv, pillow
python -m pytest -q            # self test (~10 s)

# 1. Simulate a 15 s transmission through a modelled repeater channel
hamdv sim --callsign K4OZI --text "73 de K4OZI" --seconds 15 --snr 20 --out-dir out/
#    -> out/contact_sheet.png, out/original_vs_received.png, out/received.mp4,
#       out/tx_audio.wav (what the radio would transmit), out/received_audio.wav

# 2. Analog video, weak signal (6 dB), most robust digital mode for the side information
hamdv sim --analog --analog-blocks 4 --mode qpsk-1/2 --audio-rate 0 --size 64x48 --snr 6 --out-dir out_an/

# 3. Make a transmission file and decode it (play tx.wav into any radio)
hamdv encode --callsign K4OZI --seconds 20 --out tx.wav
hamdv decode tx.wav --out-dir decoded/
```

### Live operation

```bash
hamdv devices                                    # find your sound-card interface

# Transmit: webcam 0 + microphone -> radio (PTT on the serial RTS line of a USB interface)
hamdv tx --callsign K4OZI --video 0 --audio mic --out-device 3 --in-device 1 \
         --ptt /dev/ttyUSB0:rts --mode 16qam-3/4 --size 96x72 --tot 120

# Receive: radio -> decoder -> browser viewer at http://<this computer>:8080/ and speaker
hamdv rx --in-device 1 --out-device 2 --http 8080
```

Any computer on the LAN can open the viewer page; the page shows the
transmitting station's callsign, SNR, mode and the live picture.  Several
stations on a repeater simply take turns transmitting ("video round table"):
the receivers follow whoever is keyed.

`--video` accepts a camera index, a video file, a still image, or `test`
(a synthetic talking head used by the simulator).  `--audio` accepts `mic`,
`none`, `synth` or a `.wav` file.

## Radio hook-up and levels

* Use a sound-card interface (SignaLink, DigiRig, RigBlaster, a USB radio
  cable…) on the **microphone/speaker voice path**: no data port is required.
* Set the TX audio level (`--level`, default 0.25 RMS full scale, and the
  interface's drive pot) so that the radio does **not** engage its limiter.
  The waveform's peaks are bounded at 3× its RMS (about 9.5 dB PAPR); aim for
  about 2.5–3 kHz peak deviation on a 5 kHz channel.  The simulator's
  `--limiter 1.0` shows what an overdriven mic input does (it mostly hurts the
  64-QAM and analog modes).
* RX audio from the radio's speaker jack or data-out; disable noise
  reduction / compression features in the radio's audio path.
* `--tot 120 --gap 3`: the transmitter drops PTT before the repeater's
  time-out timer, waits for the courtesy tone, and re-keys with a new
  preamble; receivers re-acquire automatically.
* There is no carrier-frequency offset on FM, so radios need no fine tuning.
  Sound-card clock offsets of hundreds of ppm are tracked automatically.

## Modes and link budget

| digital mode | net bytes per superframe (7 blocks) | needs SNR ≈ |
|--------------|-------------------------------------|-------------|
| qpsk-1/2     | 217                                 | 3 dB        |
| qpsk-3/4     | 322                                 | 6 dB        |
| 16qam-1/2    | 434                                 | 8 dB        |
| 16qam-3/4    | 651                                 | 11 dB       |
| 64qam-1/2    | 651                                 | 13 dB       |
| 64qam-3/4    | 973                                 | 17 dB       |

A superframe lasts 1.53 s.  Voice at 1200 bit/s costs 237 bytes of it; the
rest goes to video.  Block 0 of every superframe is always QPSK rate-1/2 and
carries the header (callsign, sequence, layout), so the station identity is
decodable well below the payload threshold.  A full-quieting FM signal
through a repeater typically gives 25–35 dB audio SNR, enough for 64-QAM.

## Legal notes (US Part 97, similar elsewhere)

* Every superframe header contains the operator's callsign in plain ASCII,
  which satisfies station identification throughout the transmission.
* The encoding is fully specified in this repository (open source, documented
  in DESIGN.md), as required for digital codes on amateur frequencies.
* The emission occupies only the audio passband of a voice channel, so it is
  no wider than voice on the same channel; check your band plan and repeater
  owner's policy before using a repeater for extended data transmissions.

## Repository layout

```
hamdv/dsp.py          pulse shaping, Zadoff-Chu, Hadamard, polyphase interpolation, DCT
hamdv/modem.py        SC-FDE modulator / demodulator (sync, channel estimation, MMSE equaliser)
hamdv/ldpc.py         QC-LDPC construction, GF(2) encoder, min-sum decoder
hamdv/constellation.py QAM mapping and max-log LLRs
hamdv/framing.py      superframes, header, packets, CRC, block coding
hamdv/videocodec.py   digital video codec (integer DCT, motion, RDO, arithmetic coding)
hamdv/softvideo.py    analog video transport (3-D DCT, power allocation, Hadamard, LLSE)
hamdv/vocoder.py      LPC/LSP mixed-excitation speech codec
hamdv/rangecoder.py   adaptive binary range coder
hamdv/channel.py      FM repeater channel simulator
hamdv/tx.py, rx.py    transmitter / receiver pipelines
hamdv/audioio.py      sound devices, resampling, PTT;  camera.py  video sources;  viewer.py  HTTP viewer
hamdv/cli.py          command line
tests/                unit and end-to-end tests
```
