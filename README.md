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
16-QAM rate 3/4 carries about 3.4 kbit/s net at 1600 baud, used as
~1.2 kbit/s voice plus ~1.8–2 kbit/s video (96×72 at ~2 frames/s); 2000 baud
and 64-QAM / shaped 256-QAM raise that to 5–7 kbit/s.  On a radio's flat
9600-baud data port the 4000-baud profile carries 8–16 kbit/s (128×96 or
160×120 at 4–5 frames/s).

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
transmitting station's callsign, SNR, mode and the live picture.  The
viewer displays **30 frames per second**: the 2–4 real frames per second
that the channel carries are motion-interpolated for display using the
motion vectors the codec already transmits (no extra bits on the air, see
DESIGN.md §6.1).  `--smooth-fps 0` shows only the real frames.  Several
stations on a repeater simply take turns transmitting ("video round table"):
the receivers follow whoever is keyed.

`--video` accepts a camera index, a video file, a still image, or `test`
(a synthetic talking head used by the simulator).  `--audio` accepts `mic`,
`none`, `synth` or a `.wav` file.

### Waveform profiles: `--baud 1600 | 2000 | 4000`

| profile | symbol rate | audio band | internal rate | use |
|---------|-------------|------------|---------------|-----|
| `--baud 1600` (default) | 1600 Bd | 620–2780 Hz | 8 kHz | any voice radio / repeater, mic & speaker jacks |
| `--baud 2000` | 2000 Bd | 450–2950 Hz | 8 kHz | voice path with good audio response (+25 % symbols, +~45 % video) |
| `--baud 4000` | 4000 Bd | 500–5500 Hz | 16 kHz | **flat 9600-baud data port** (unfiltered audio): 2.5× the symbols |

The receiver listens for all three profiles at once and follows whichever
acquires, so stations can switch profiles without coordination.

**9600-port operation (`--baud 4000`).**  Connect the sound card to the
radio's 6-pin mini-DIN data port (9600 mode: pins "PKD/TXD" and "RXD9600")
or an equivalent flat discriminator-out / modulator-in path.  That path has
no pre-emphasis, no microphone limiter and no 300–3000 Hz audio filters,
only the IF filter, so a 4000-baud waveform up to 5.5 kHz fits.  Set the
drive for about 3 kHz peak deviation on a 25 kHz channel (the TX level
`--level` plus the interface pot, as for 9600-baud packet) and use the
radio's wide (25 kHz) setting.  Through a repeater this needs a repeater
built with flat audio (as for 9600 packet); most voice repeaters are not,
so `--baud 4000` is mainly for simplex or data-capable repeaters.
Discriminator noise rises with frequency, so a few dB more signal is needed
than the band-average SNR suggests.

### HF: `--profile ssb` and `--profile am`

| profile | symbol rate | audio band | designed for |
|---------|-------------|------------|--------------|
| `ssb` | 1333 Bd | 633–2367 Hz (fits a 2.4 kHz SSB filter) | HF single sideband: ±150 Hz mistuning, drift, ionospheric multipath and fading |
| `am`  | 2000 Bd | 500–3100 Hz | HF (or any) AM with an envelope or synchronous detector |

**SSB.**  Feed the transceiver's mic/line input and take audio from its
line/speaker output as for any digital mode.  Set the drive so ALC barely
moves (the waveform has a 9 dB peak-to-average ratio; the shaped and
analog modes need a linear amplifier).  No fine tuning is required: the
receiver searches ±150 Hz on acquisition and then tracks drift.  Pilot
blocks come every 0.86 s, the channel is interpolated between them and
re-estimated decision-directed inside every block, and a piecewise
time-varying equaliser follows fading within a block.  Use `--long-code`
(time diversity across the superframe) and the QPSK modes on real HF
paths; voice at 1200 bit/s fits with `qpsk-3/4` and leaves ~0.5 kbit/s
for video, so most HF users will run `--audio-rate 0` for video or the
analog video mode for voice-free pictures.  Measured through a Watterson
two-path simulator (12 s, mistuned 73 Hz, 60 ppm clock offset, long code):

| condition (CCIR) | qpsk-1/2 | qpsk-3/4 | 16qam-3/4 |
|---|---|---|---|
| good (0.5 ms, 0.1 Hz), 6 / 9 / 12 / 18 dB | 10/11, 11/11, 11/11, 11/11 | 10/11, 11/11, 11/11, 11/11 | 0, 0, 8/11, 11/11 |
| moderate (1 ms, 0.5 Hz), 9 / 12 / 18 dB | 8/11, 8/11, 9/11 | 7/11, 7/11, 8/11 | unusable |
| poor (2 ms, 1 Hz), 12 / 18 dB | 6/11, 6/11 | 5/11, 6/11 | unusable |

(superframes decoded / sent).  Poor conditions are limited by the 190 ms
block length; shorter blocks or interleaving across superframes would be
the next step.

**AM.**  The carrier cancels in the detector, so no frequency search is
needed and the voice-channel machinery applies.  What matters is
*modulation depth*: the modulator must never reach 100 % on negative peaks,
so set the audio drive for about 90 % peak modulation (`--level 0.3` gives
3× RMS peaks, i.e. ~30 % average modulation).  On a stable path the AM
profile runs 16-QAM 3/4 at 20 dB SNR (12/12 superframes in the model).  On
a selectively fading path an **envelope detector** distorts whenever the
carrier fades below the sidebands (4/12 superframes in "good" fading);
a **synchronous detector** (`--am-sync` in the simulator; sync-AM / SDR
receivers in practice) recovers most of it (10/12).  Use QPSK modes on
fading AM paths.

### Long codes: `--long-code`

One LDPC codeword spans all seven payload blocks of a superframe instead of
one per block.  Measured gain is about 1 dB (16-QAM 3/4 decodes 6/6
superframes at 11 dB where the short code manages 1/6).  The price is
all-or-nothing decoding of the superframe and about 20 ms more latency.

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
| 256qam-s4.5  | 973                                 | 16 dB       |
| 256qam-s5.0  | 1085                                | 18 dB       |
| 256qam-s5.5  | 1190                                | 20 dB       |
| 256qam-3/4   | 1302                                | 22 dB, clean audio path only |

The `256qam-s*` modes are the **strong-signal modes with probabilistic
amplitude shaping**: 256-QAM whose amplitudes follow a Maxwell–Boltzmann
distribution produced by an exact enumerative distribution matcher, with the
LDPC parity carried on the sign bits (see DESIGN.md §4.4).  They fill the
6 dB gap between 64-QAM and uniform 256-QAM in finer rate steps, gain about
1 dB over a uniform constellation at the same rate, and because the outer
points are sent rarely they tolerate a radio's microphone limiter where
uniform 256-QAM does not.  Example at 128×96, 4 frames per superframe:

| mode | video bit rate (with 1200 bit/s voice) | PSNR, test scene |
|------|-----------------------------------------|------------------|
| 64qam-3/4   | 3.2 kbit/s | 28.0 dB |
| 256qam-s5.0 | 3.6 kbit/s | 29.1 dB |
| 256qam-s5.5 | 4.2 kbit/s | 29.9 dB |
| 256qam-3/4  | 4.7 kbit/s | 30.5 dB |

A superframe lasts 1.53 s at 1600 Bd (1.22 s at 2000 Bd, 0.61 s at 4000
Bd); the byte counts above scale with the symbol rate.  Voice at 1200 bit/s
costs 237 bytes of a 1600-Bd superframe; the rest goes to video.  Measured
video rates with voice on (test scene, through the modelled channels):

| profile / mode | video bit rate | picture |
|----------------|----------------|---------|
| 1600 Bd, 16qam-3/4 | 1.7 kbit/s | 96×72, 2 fps, 26.6 dB |
| 2000 Bd, 16qam-3/4 | 2.5 kbit/s | 96×72, 2.5 fps, 27.5 dB |
| 2000 Bd, 256qam-s5.5, long code | 5.4 kbit/s | 128×96, 3.3 fps, 30.7 dB |
| 4000 Bd flat, 16qam-3/4, long code | 6.1 kbit/s | 128×96, 5 fps, 27.0 dB |
| 4000 Bd flat, 64qam-3/4, long code | 10.1 kbit/s | 128×96, 5 fps, 29.5 dB |
| 4000 Bd flat, 256qam-s5.5, long code | 12.9 kbit/s | 160×120, 5 fps, 29.3 dB |  Block 0 of every superframe is always QPSK rate-1/2 and
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
hamdv/constellation.py QAM mapping and max-log LLRs (with shaping priors)
hamdv/shaping.py      probabilistic amplitude shaping: enumerative distribution matcher, 256-QAM PAS modes
hamdv/framing.py      superframes, header, packets, CRC, block coding
hamdv/videocodec.py   digital video codec (integer DCT, motion, RDO, arithmetic coding)
hamdv/softvideo.py    analog video transport (3-D DCT, power allocation, Hadamard, LLSE)
hamdv/vocoder.py      LPC/LSP mixed-excitation speech codec
hamdv/rangecoder.py   adaptive binary range coder
hamdv/channel.py      channel simulators: FM repeater, flat 9600-port, HF SSB (Watterson), HF AM
hamdv/tx.py, rx.py    transmitter / receiver pipelines
hamdv/audioio.py      sound devices, resampling, PTT;  camera.py  video sources;  viewer.py  HTTP viewer
hamdv/cli.py          command line
tests/                unit and end-to-end tests
```
