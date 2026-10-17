# HamDV plan: DV (digital voice) and DD (digital data) modes

This document plans two further operating modes on top of the HamDV physical
layer (SC-FDE waveform, LDPC, profiles for FM voice, 9600-port, SSB and AM).
It records what is already in place, the design decisions, the numbers that
drive them, and a build order.  Sections marked *built* are in the repository.

## 1. Goals

**DV — digital voice.**  Conversational voice that sounds like a good VoIP
call ("in the room"), as robust against fading and impulse noise as the
channel allows, with bandwidth that adapts to conditions, and a half-duplex
link that pauses to let the far end report link quality (VARA-style).

**DD — digital data.**  Reliable file / message transfer with automatic
repeat request (ARQ), adaptive modulation, and compression.

## 2. What exists already (built)

| building block | status |
|---|---|
| Waveform profiles 1600/2000 Bd voice, 4000 Bd flat, SSB, AM | built |
| LDPC rates 1/2, 3/4; long codes per superframe (~1 dB) | built |
| Cross-superframe symbol interleaving (depth 2/4/8) for fading and impulse bursts | built |
| Shaped 256-QAM strong-signal modes | built |
| Opus codec binding (libopus via ctypes), audio modes 6/8/12/16 kbit/s, 16 kHz wideband | built |
| Built-in LPC vocoder 1200/1600/2400 bit/s | built |
| Receiver profile auto-detection, per-block SNR, link statistics | built |

## 3. Voice quality versus channel: the numbers

Opus wideband (SILK, 16 kHz) is the "in the room" codec; it needs
6 kbit/s minimum and 12–16 kbit/s to be transparent.  Net link capacity
per profile (voice packet overhead ~1 %):

| profile / mode | net kbit/s | Opus rate that fits (with ~1 kbit/s margin for control / video) |
|---|---|---|
| 1600 Bd voice, 64qam-3/4 long | 5.1 | none (6 kbit/s needs 6.1) |
| 2000 Bd voice, 64qam-3/4 long | 6.4 | — (tight) |
| 2000 Bd voice, 256qam-s5.5 long | 7.8 | 6 kbit/s (+1.5 kbit/s video, measured) |
| 4000 Bd flat, 16qam-3/4 long | 8.5 | 6 kbit/s |
| 4000 Bd flat, 64qam-3/4 long | 12.7 | 12 kbit/s (measured, every frame delivered) |
| 4000 Bd flat, 256qam-s5.5 long | 15.6 | 12 kbit/s + video, or 16 kbit/s voice only |
| SSB 1333 Bd, qpsk-3/4 long | 1.3 | LPC vocoder only |

Conclusions that shape the design:

1. **"In the room" voice needs the flat 9600-port profile** (or a strong
   signal on 2000 Bd with shaped 256-QAM).  On a plain FM voice channel the
   ceiling is Opus 6 kbit/s with a full-quieting signal, otherwise the LPC
   vocoder.  On HF the LPC vocoder is the only option.
2. **Rate adaptation is a codec ladder, not just a modulation ladder:**
   Opus 16 → 12 → 8 → 6 kbit/s → LPC 2400 → 1600 → 1200.  Each step is
   about 3 dB of required SNR.
3. The codec frames are 20 ms but the link delivers them per superframe
   (0.6–1.5 s).  Loss is therefore bursty (whole superframes); Opus in-band
   FEC (designed for single-packet loss) does not help, so it is switched
   off and the bits go to quality.  The link-level protection is the LDPC
   code, interleaving, and the rate ladder.

## 4. DV mode design

### 4.1 Robustness stack

* **Codec:** Opus CBR, 20 ms frames, wideband, complexity 10, FEC off
  (reason above).  LPC vocoder below 6 kbit/s.
* **Voice packets** carry the mode byte, frame count and constant frame
  length; a superframe carries 30–75 frames.  (built)
* **Long code + interleaving depth 2–4** on the voice profiles (impulse
  noise: a 40 ms burst erases < 3 % of a depth-4 codeword instead of 30 %
  of a short one).  Depth 2 keeps latency under 2 superframes.
* **Concealment:** Opus packet-loss concealment for lost frames, a fade on
  lost superframes (built); jitter buffer of 2 superframes at the receiver.
* **Impulse noise at the demodulator:** blank samples whose magnitude
  exceeds 4× the running RMS before the matched filter (cheap and
  effective against ignition/popcorn noise on FM; *to build*), and the
  per-block noise estimate from the scattered pilots already de-weights a
  hit block in the LDPC decoder (built).

### 4.2 Dynamic bandwidth (adaptive rate)

Every superframe header already announces the modulation; the receiver
measures per-block SNR, block error rate and pilot quality (built).  The
transmitter picks the mode from a table of required SNR (measured in
DESIGN.md) with 3 dB hysteresis, and the codec rate from the mode's net
capacity.  What is missing is the *feedback path* (4.3).

### 4.3 Half-duplex link control (VARA-style turnarounds)

A DV session is a pair of stations in alternating "overs" with short
**control bursts** between them:

```
A: [preamble][voice superframes ... N][END+REQ]   -- PTT drops
B:            [preamble][CTRL burst: SNR, block errors, suggested mode]   ~0.5 s
A: [preamble][voice superframes ...]  (new mode)
```

* **When to pause.**  The transmitter talks continuously while the
  receiver's playout buffer is below a target (e.g. 2 superframes).  Once
  it holds enough audio to cover the turnaround, the transmitter finishes
  the current superframe, sets the LAST flag with a REQ bit, and unkeys.
  The receiver keeps playing from its buffer during the gap.  Target gap:
  key-up 0.3 s + preamble 0.2 s + one control superframe 0.6 s + key-down
  0.2 s ≈ 1.3 s, so the buffer target is 2 superframes (≥ 1.2 s at 4000 Bd,
  3 s at 1600 Bd).
* **Control burst.**  A one-superframe transmission (block 0 QPSK-1/2 only,
  which decodes 8 dB below the payload modes) carrying: measured SNR,
  block error rate, pilot quality, suggested mode, and a 'hold' or
  'resume' request.  It reuses the existing header + a new CTRL packet
  type; nothing new on the air except the REQ flag.
* **Rate decision** is made by the *receiver* (it has the measurements) and
  only acknowledged by the transmitter, as in VARA.  Fallback: if no
  control burst arrives within 2 s, the transmitter resumes at one step
  lower.
* **Repeaters.**  The courtesy tone and hang time fit inside the gap; the
  TOT guard (built) already forces periodic unkeying, so link checks are
  scheduled at those boundaries too.
* **Voice latency** then equals buffer target + one superframe ≈ 2–4 s.
  That is the price of robustness; a "low-latency" option with a 1-superframe
  buffer and no link checks remains available (today's behaviour).

### 4.4 Stream compression

Opus and the vocoder are already near entropy (measured: LZMA expands the
video stream, compresses the vocoder stream only on repetitive synthetic
speech).  Compression is therefore applied only to text/control payloads
and to DD mode (5.3).  A future gain inside the vocoder is predictive
coding of the line-spectral pairs (10–20 % on real speech), not a stream
compressor.

## 5. DD mode design

### 5.1 Framing and ARQ

* **Transport units:** DD packets of up to 1 kB with a 16-bit sequence
  number and CRC-32 (the existing CRC-16 packet layer stays for framing).
* **Selective-repeat ARQ:** the receiver answers each over with a bitmap of
  received sequence numbers (in the control burst of 4.3); the transmitter
  retransmits only missing packets.  Window = what fits in one over
  (e.g. 10 superframes).
* **Session:** CONNECT (callsigns, capabilities) → DATA overs → DISCONNECT;
  a 3-way handshake using control bursts.  Unconnected broadcast mode
  (no ACKs, interleave 8, QPSK-1/2) for bulletins.
* **Reliability under fading:** long code + interleaving depth 4–8 per
  over, and the ARQ catches what remains.  Measured basis: SSB moderate
  conditions deliver 87/88 blocks with depth 8; poor conditions need the
  shorter-block work in 6.

### 5.2 Adaptive modulation

Same mechanism as DV (4.2/4.3) with a more aggressive policy: step up
after two clean overs, step down after any over with block errors above
5 %, and an explicit per-over speed report in the UI.

### 5.3 Compression

Data is compressed **before** framing and FEC (never after: the LDPC
parity must reach the decoder bit for bit).  zstd level 19 if available,
otherwise LZMA2 (`lzma.FORMAT_RAW`), per file / message, with a 1-byte
method tag; incompressible files are sent raw.  Python's `lzma` is built
in, so no new dependency.

### 5.4 Throughput expectations

| profile | mode | DD net rate (ARQ overhead ~10 %) |
|---|---|---|
| FM voice 1600 Bd | 16qam-3/4 .. 64qam-3/4 long | 3.0 – 4.6 kbit/s |
| FM 2000 Bd | 256qam-s5.5 long | 7.0 kbit/s |
| 9600-port 4000 Bd | 64qam-3/4 .. 256qam-s5.5 long | 11 – 14 kbit/s |
| SSB good | qpsk-3/4 .. 16qam-3/4 long | 1.2 – 2.3 kbit/s |
| SSB moderate | qpsk-1/2 long, interleave 8 | 0.7 kbit/s |

## 6. Known limits and the work they imply

* **HF poor conditions (2 ms, 1 Hz):** the 190 ms block is too long for
  1 Hz Doppler; interleaving does not help because most symbols are already
  bad.  Fix: a 128-symbol block variant for HF (header spans two blocks),
  estimated 2× faster tracking.
* **Opus on a plain FM voice channel:** needs 2000 Bd + shaped 256-QAM, i.e.
  a strong signal and a clean audio path.  A 40 ms-frame Opus NB mode at
  5 kbit/s would fit 64-QAM on 1600 Bd; worth testing by ear.
* **libopus dependency:** the binding loads the system library; when it is
  missing the LPC vocoder is used and the DV modes are refused with a
  message.

## 7. Build order

1. Impulse blanker in the demodulator front end (small, measurable with
   `--impulses`).  
2. CTRL packet + REQ flag + receiver link-quality report; `hamdv dv` command
   with the buffer-driven pause/resume state machine and PTT control
   (two instances can be tested back-to-back through the simulator).  
3. Rate ladder (modulation + codec) driven by the control reports.  
4. DD: packetiser, selective-repeat ARQ, CONNECT/DISCONNECT, compression,
   `hamdv dd send|receive`.  
5. HF 128-symbol block variant for poor conditions.  
6. Field tests on a repeater (FM), a 9600-port simplex link, and an HF path.
