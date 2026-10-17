# HamDV design and mathematics

This document specifies the HamDV waveform and codecs completely, so that an
independent implementation can interoperate and so that the "digital code" is
publicly documented.  Notation: $T$ = symbol period, $N$ = 256 symbols per
block, $L$ = 16 cyclic-prefix symbols, $B$ = 8 data blocks per superframe,
$f_s$ = 8000 Hz internal sample rate.

## 1. Why this design

An FM voice repeater is a strange data channel: a 300–3000 Hz audio passband
with group-delay ripple at the edges, pre-/de-emphasis, a microphone limiter,
no carrier-frequency offset (FM carries audio at baseband) but unknown delay,
gain and sound-card clock offsets, a time-out timer, courtesy tones and the
occasional CW identification mixed *over* the signal.  When the signal is
full quieting the audio SNR is 25–35 dB; it collapses quickly below the FM
threshold.

The design therefore

* uses a **single carrier** (not OFDM) for a low peak-to-average ratio, so the
  radio's limiter is not triggered, and recovers the frequency-selective
  passband with **frequency-domain equalisation** (the FFT is used for the
  equaliser, not for the waveform);
* fits the whole waveform inside **620–2780 Hz**, well clear of the CTCSS
  high-pass and the audio low-pass of a repeater;
* re-synchronises every 1.5 s on a **pilot block**, so bursts (ID tones,
  squelch crashes) cost one superframe, never the link;
* gives the operator a choice between a **digital video codec** (best picture
  at good SNR) and an **analog linear transport** (no cliff, pictures down to
  a few dB SNR), both sharing the same voice and control channel.

## 2. Waveform

Transmitted audio:

$$a(t) = \Re\Big\{ \sum_k s_k\, g(t - kT)\, e^{j 2\pi f_c t} \Big\},\qquad
f_c = 1700\ \mathrm{Hz},\quad 1/T = 1600\ \mathrm{Bd}$$

with $g$ the root-raised-cosine pulse, roll-off $\alpha = 0.35$, span 10
symbols, unit energy:

$$g(t)=\frac{\sin\!\big(\pi \tfrac{t}{T}(1-\alpha)\big)+4\alpha\tfrac{t}{T}\cos\!\big(\pi\tfrac{t}{T}(1+\alpha)\big)}{\pi\tfrac{t}{T}\big(1-(4\alpha \tfrac{t}{T})^2\big)}.$$

Occupied band: $f_c \pm (1+\alpha)/(2T) = 1700 \pm 1080$ Hz.  All symbol
alphabets have unit average power; the TX output is scaled to the configured
RMS and peak-limited at $3\times$RMS.

Three profiles share the block structure and differ only in these
constants (the receiver runs all three acquisition correlators in parallel
and follows the one that locks):

| profile | $f_s$ | samples/symbol | $1/T$ | $f_c$ | $\alpha$ | band |
|---|---|---|---|---|---|---|
| voice | 8 kHz | 5 | 1600 Bd | 1700 Hz | 0.35 | 620–2780 Hz |
| voice2k | 8 kHz | 4 | 2000 Bd | 1700 Hz | 0.25 | 450–2950 Hz |
| flat (9600-baud data port) | 16 kHz | 4 | 4000 Bd | 3000 Hz | 0.25 | 500–5500 Hz |

The flat profile assumes the radio's unfiltered modulator-in /
discriminator-out path (no pre-emphasis, no limiter, no voice-band
filters).  Discriminator noise after FM demodulation has a power spectrum
rising as $f^2$, so the upper tones see a worse SNR than the lower ones; the
per-tone MMSE weights (§3.4) account for that automatically and the
channel simulator models it.

### 2.1 Transmission structure

```
[key-up silence][preamble: ZC127 ZC127][superframe 0][superframe 1]...[tail]
superframe  = [pilot block][data block 0][data block 1] ... [data block 7]
block       = [CP: last 16 symbols of the body][body: 256 symbols]
pilot body  = Zadoff-Chu sequence of length 256, root 29
data body   = 248 payload symbols + 8 scattered pilot symbols at 16 + 32 i
```

A superframe is $9 \times 272 = 2448$ symbols $= 1.53$ s; 12240 audio
samples at 8 kHz.

**Zadoff–Chu sequences.**  $x_u[n] = \exp\!\big(-j\pi u\, n(n+c_f)/N_{zc}\big)$,
$c_f = N_{zc} \bmod 2$, $\gcd(u,N_{zc})=1$.  They have constant amplitude
(unit PAPR before pulse shaping), zero periodic autocorrelation, and their DFT
is again constant-amplitude.  The preamble (two repetitions of $N_{zc}=127$,
$u=7$) gives timing by correlation and a carrier-offset estimate
$\hat f = \arg(c_2 c_1^*)/(2\pi \cdot 127 T)$ from the phase between the two
repetitions; the pilot block ($N_{zc}=256$, $u=29$) gives a least-squares
channel estimate without spectral nulls.

## 3. Receiver

### 3.1 Front end and synchronisation

The audio is mixed with $\sqrt{2}\,e^{-j2\pi f_c t}$ and matched-filtered
with $g$.  Acquisition evaluates, at every sample offset $t$,

$$m[t] = \frac{\big(|c_1[t]| + |c_2[t]|\big)^2}{2 N_{zc}\,(E_1[t]+E_2[t])},\qquad
c_i[t] = \sum_{k} x^*[k]\, y[t + kT + (i-1)N_{zc}T],$$

a correlation normalised by the energy under the template, so the threshold
(0.45) is independent of gain.  The peak is refined by parabolic
interpolation to sub-sample precision.

### 3.2 Timing drift (sound-card clocks)

Two sound cards differ by up to a few hundred ppm.  Each pilot block is
re-located by the same normalised correlation in a $\pm 2T$ window; the drift
between consecutive anchors gives the sample-rate offset $\rho$,
smoothed with a first-order filter, and symbol $n$ of superframe $k$ is read
at the fractional sample position $t_{k,n} = A_k + n\,(1+\rho)\,T f_s$.  On
acquisition the *next* pilot block is located before the first superframe is
equalised, so $\rho$ is known from the start.

Fractional positions are evaluated with a 16-tap Kaiser-windowed sinc
polyphase interpolator (128 phases, linear interpolation between phases).
Its gain ripple is below 0.05 dB over the signal band, which matters: a cubic
interpolator's 1.4 dB band-edge droop *varies with the fractional phase* and
would make the channel estimate stale as the timing drifts.

### 3.3 Channel estimation (least squares + subspace projection)

With $Y = \mathrm{DFT}(y_{\text{pilot}})$ and $X_p = \mathrm{DFT}(\text{ZC}_{256})$,
$|X_p[k]| = \sqrt{N}$ for all $k$, the least-squares estimate is
$\hat H_{ls}[k] = Y[k]/X_p[k]$.  Its noise is reduced by projecting onto the
subspace of impulse responses shorter than the cyclic prefix:

$$\hat H = F\, S\, F^{H} \hat H_{ls},\qquad S = \mathrm{diag}(\underbrace{1,\dots,1}_{L},0,\dots,0),$$

i.e. inverse DFT, keep the first $L$ taps, DFT.  $P = F S F^H$ is an
orthogonal projector of rank $L$, so the estimation noise variance drops by
$L/N = 1/16$ (12 dB).  The FFT window of every block starts 4 symbols into
the cyclic prefix, so the circular-convolution model holds for channels with
up to 4 pre-cursor and 12 post-cursor symbols (7.5 ms).  The residual
$Y - \hat H X_p$ gives the noise variance $\sigma^2$ used everywhere else.

### 3.4 MMSE frequency-domain equalisation

For every data block, $\hat X = W \odot \mathrm{DFT}(y)$,
$\hat x = \mathrm{IDFT}(\hat X)$ with

$$W[k] = \frac{\hat H^*[k]}{|\hat H[k]|^2 + \sigma^2},$$

the per-tone minimum-mean-square-error solution (the channel matrix is
circulant, $C = F^H \Lambda F$, so the $N\times N$ MMSE equaliser
$(C^H C + \sigma^2 I)^{-1} C^H$ diagonalises into these scalar weights).
The post-equalisation noise variance

$$\sigma_{eq}^2 = \frac{1}{N}\sum_k |W_k|^2 \sigma^2 + \frac{1}{N}\sum_k |W_k H_k - 1|^2$$

(noise enhancement plus residual ISI bias) feeds the LDPC log-likelihoods and
the analog decoder.  Scattered pilots remove the residual phase
$\phi_0 = \arg \sum_i \hat x_{p_i} p_i^*$ and the phase slope
$\omega = \arg\sum_i (\hat x_{p_{i+1}} p_{i+1}^*)(\hat x_{p_i} p_i^*)^*/32$
of each block, and their error power is a per-block sanity check on $\sigma^2_{eq}$
(bursts such as a CW ID raise it, which de-weights that block in the decoders).

## 4. Forward error correction

### 4.1 Quasi-cyclic LDPC construction

$H = [H_u \,|\, H_p]$ is built from $31\times31$ circulant permutation
matrices.  The information part $H_u$ has block-column weight 3 with shifts
drawn pseudo-randomly (fixed seed, so both ends build the same code) subject
to the girth-6 condition on every $2\times2$ sub-array of non-empty blocks
with shifts $\begin{psmallmatrix}a&b\\c&d\end{psmallmatrix}$:
$a - b + d - c \not\equiv 0 \pmod{31}$ (no 4-cycles).  The parity part $H_p$
is a block staircase of identity circulants on the diagonal and
sub-diagonal (an irregular repeat–accumulate structure as in DVB-S2 and
IEEE 802.16e), which is invertible by construction.

Encoding is therefore linear time and needs no matrix inversion:
$s_i = \sum_j C(\sigma_{ij})\,u_j$ for block row $i$, then
$p_0 = s_0$, $p_i = p_{i-1} \oplus s_i$ (accumulate).  Codeword length equals
the bit capacity of a block or of a group of blocks: $n = 248\,b\,n_b$ for
$b$ bits per symbol and $n_b$ blocks coded together, rates 1/2 and 3/4.

**Long codes.**  With the header flag `LONGCODE` one codeword spans all
digital payload blocks of the superframe ($n_b = 7$, $n$ up to 13 888).
Longer codes decode closer to capacity: measured AWGN thresholds improve
from 3.0 / 11.5 / 17.0 dB (QPSK-1/2, 16-QAM-3/4, 64-QAM-3/4) to
2.0 / 10.5 / 16.0 dB, and the error-rate cliff becomes steeper.  Decoding
costs 11–23 ms per superframe.  The shaped modes (§4.4) use the same
scheme with the amplitude labels of all $n_b$ blocks as information bits.

### 4.2 Decoding

Normalised min-sum belief propagation ($\alpha = 0.8$, up to 40 iterations,
early termination on $H\hat c = 0$).  Check-node messages use the first and
second minimum of the incoming magnitudes and the product of signs; the
implementation is fully vectorised (reduceat over edges sorted by check
node) and decodes a block in 1–7 ms.  Measured thresholds (AWGN, frame error
rate → 0): QPSK-1/2 3 dB, 16QAM-3/4 11 dB, 64QAM-3/4 17 dB.

### 4.3 Constellations and soft demapping

Square QAM, Gray coded independently on I and Q, unit average power.
Max-log LLRs: $\Lambda_j = \big(\min_{s: b_j(s)=1}|y-s|^2 - \min_{s: b_j(s)=0}|y-s|^2\big)/\sigma^2_{eq}$.
Codeword bits are interleaved by a fixed pseudo-random permutation before
mapping so a cluster of bad symbols does not hit adjacent code bits.

### 4.4 Strong-signal modes: 256-QAM with probabilistic amplitude shaping

Square 256-QAM is 16-PAM per axis: one sign bit and one of eight amplitudes
$a \in \{1,3,\dots,15\}$.  Used uniformly, the outer points spend a lot of
power for little information; a Gaussian-like distribution over the points
is up to $\tfrac{\pi e}{6}$ (1.53 dB) more efficient.  HamDV implements
probabilistic amplitude shaping (Böcherer, Steiner, Schulte 2015):

1. **Target distribution.**  Amplitudes are sent with the Maxwell–Boltzmann
   distribution $P(a) = e^{-\nu a^2}/\sum_{a'} e^{-\nu a'^2}$, the
   maximum-entropy distribution for a given mean power.  $\nu$ sets the
   information rate continuously; three operating points are defined
   (4.5, 5.0 and 5.5 user bits per symbol).

2. **Distribution matcher.**  A block of $n = 496$ amplitudes (248 symbols ×
   2 axes) has a fixed *composition* $(n_0,\dots,n_7)$, $\sum n_a = n$,
   obtained by quantising $nP(a)$.  The number of sequences with that
   composition is the multinomial coefficient $M = n!/\prod_a n_a!$, and
   $k = \lfloor \log_2 M\rfloor$ user bits select one by enumerative coding:
   reading the bits as an integer $r < 2^k$, the sequence is *unranked*
   symbol by symbol, where the number of completions that start with
   amplitude $a'$ given the remaining composition is
   $M_{rem}\, n_{a'}/n_{rem}$ (an exact integer).  The inverse (ranking) sums
   the completion counts of all smaller amplitudes at each position.  With
   arbitrary-precision integers the matcher is exact and perfectly
   invertible; the rate loss relative to the entropy $H(P)$ is about 0.05 bit
   per amplitude at $n = 496$.  Compositions for the three rates are fixed in
   `shaping.py`; the strongest shaping whose $\log_2 M$ still admits $k$ bits is
   chosen by bisection on $\nu$:

   | user bits/symbol | $k$ per block | composition $n_0..n_7$ | $H(A)$ | power saving vs. uniform |
   |---|---|---|---|---|
   | 4.5 | 1116 | 161 136 96 57 29 12 4 1 | 2.30 bit | 5.7 dB |
   | 5.0 | 1240 | 134 119 94 67 42 23 12 5 | 2.55 bit | 4.2 dB |
   | 5.5 | 1364 | 104 98 86 71 54 39 27 17 | 2.81 bit | 2.4 dB |

3. **Systematic code, parity on the signs.**  The amplitude labels (the 3
   low bits of the reflected Gray code, identical for $\pm a$) are the 1488
   information bits of the systematic rate-3/4 LDPC code with $n = 1984$; its
   496 parity bits are placed on the sign bits (the Gray MSB) of the 248
   symbols.  Parity bits are uniform, so the signs are uniform and the
   amplitude distribution is untouched.  The constellation is rescaled so
   the *shaped* average power is 1.

4. **Demapping with priors.**  The receiver computes max-log LLRs with the
   symbol prior,
   $\Lambda_j = \max_{s:b_j=0}\big[-|y-s|^2/\sigma^2 + \log P(s)\big] - \max_{s:b_j=1}[\cdot]$,
   $P(s) = P(a_I)P(a_Q)/4$; after LDPC decoding the amplitude labels are
   mapped back to amplitudes and *ranked* to recover the user bits.  A
   decoded sequence whose composition is wrong flags the block as failed.

Measured thresholds (AWGN, frame-error-free): 64-QAM-3/4 16 dB,
256-QAM-s4.5 16 dB, s5.0 18 dB, s5.5 19 dB, uniform 256-QAM-3/4 22 dB; add
about 2 dB through the modelled repeater chain.  The shaped modes also pass
the modelled microphone limiter (drive 0.5) that breaks uniform 256-QAM:
the outer points are transmitted so rarely that clipping them costs little.
Peak-to-average ratio of the shaped symbol stream is 7.2 dB versus 4.3 dB
for uniform 256-QAM, but the *audio* PAPR after pulse shaping is unchanged
(8.7–8.9 dB) because the RRC filter dominates it.

## 5. Link layer

Block 0 of every superframe is always QPSK rate-1/2 and carries the
**header packet**: magic, version, callsign (8 ASCII), transmission id,
superframe sequence, flags (last superframe, analog video), the digital mode
of blocks 1–7, a bitmap of which blocks are analog, video size, frame rate,
frames per superframe, audio mode and refresh period.  Blocks 1–7 form one
contiguous byte area filled with packets `[type][len16][payload][crc16]`
(types: audio, video frame, analog video side-information, text, info,
padding).  CRC-16/CCITT.  Each digital block is one LDPC codeword; analog
blocks carry 496 real values as $I$ and $Q$ of 248 symbols, backed off by
0.75 (−2.5 dB) because Gaussian-distributed symbols have a higher PAPR than
QAM.

Audio frames are 30 ms; a superframe carries $\lfloor 12240/240\rfloor = 51$
of them (a running sample counter handles other symbol rates).  Video frames
are packets inside the same area; the transmitter gives each frame
$(\text{bytes left})/(\text{frames left})$ as its target and drops a frame
that still does not fit (the encoder state is rolled back, so the decoder's
reference stays in sync and simply repeats the previous frame).

## 6. Digital video codec

Frames are YCbCr 4:2:0; a macroblock is 8×8 luma + two 4×4 chroma blocks.

**Transform.**  Integer cores $C_8$ (HEVC-style, entries ±64, ±89, ±83, ±75,
±50, ±36, ±18) and $C_4$.  $C_8 C_8^T \approx 2^{15} I$, $C_4 C_4^T \approx 2^{14} I$.
Forward (encoder only): $Y = C X C^T / 2^{15}$ gives coefficients in
orthonormal units.  Inverse (both ends, exact integer arithmetic):
$\hat X = \big(C^T \tilde Y_8 C + 2^{17}\big) \gg 18$ where $\tilde Y_8$
holds de-quantised coefficients in 1/8 units.  Because the reconstruction
uses only integer operations, encoder and decoder references are bit-exact on
every platform — a requirement for a prediction loop with no key frames.

**Quantisation.**  Step $\Delta(q) = 2^{q/4}$ in orthonormal units,
$q\in[0,47]$ (stored as $\Delta_8 = \mathrm{round}(8\cdot 2^{q/4})$); chroma uses
$5/8\,\Delta$ (a 4×4 DC has half the gain of an 8×8 DC).  Dead-zone rounding
offsets 1/2 (DC) and 1/3 (AC).

**Prediction.**  SKIP (co-located copy), INTER (integer-pel motion vector,
chroma vector $=\mathrm{round}(mv/2)$), INTRA (DC predicted from the mean of the
reconstructed top row and left column).  Motion vectors are coded as
differences from the median of the left, top and top-right vectors.

**Global motion by least squares.**  Before the block search, a Lucas–Kanade
translation is solved from the normal equations

$$\begin{pmatrix}\sum I_x^2 & \sum I_x I_y\\ \sum I_x I_y & \sum I_y^2\end{pmatrix}
\begin{pmatrix}u\\v\end{pmatrix} = -\begin{pmatrix}\sum I_x I_t\\ \sum I_y I_t\end{pmatrix}$$

(iterated three times with re-warping).  The block search then tests a ±4
window around both $(0,0)$ and $(u,v)$ with cost $\mathrm{SAD} + \lambda_{mv}(|dx|+|dy|)$,
vectorised over all macroblocks per candidate.

**Mode decision.**  Lagrangian $J = D + \lambda R$, $D$ = SSE of the actual
integer reconstruction, $R$ = estimated bits, $\lambda = 0.1\,\Delta^2$.

**Intra refresh.**  Every frame forces $\lceil n_{MB}/P\rceil$ macroblocks to
INTRA, visited in the order $i_k = k\,g \bmod n_{MB}$ with
$g \approx n_{MB}/\varphi$ (golden ratio, adjusted to be coprime with
$n_{MB}$).  Successive refresh blocks are therefore spread evenly over the
picture, every block is refreshed exactly once per period $P$ (default 12
frames ≈ 6 s), a lost frame heals within one period, and a receiver joining
mid-stream sees the picture build up progressively.  There are no key frames.

**Entropy coding.**  LZMA-style adaptive binary range coder (11-bit
probabilities, shift-5 adaptation, carry-less).  Syntax elements: skip flag
(context: number of skipped neighbours), intra flag, motion-vector
differences (zero flag, sign, unary with 6 contexts + Exp-Golomb), coded-
block flags per plane, and zig-zag coefficients with significance,
greater-than-1, greater-than-2 flags (contexts by scan position bucket,
plane, and count of large coefficients so far), Exp-Golomb remainders, signs
in bypass mode and a "last" flag after every significant coefficient.

**Rate control.**  $q \leftarrow q + \mathrm{clip}(3\log_2(\text{bits}/\text{target}),-3,4)$
after every frame, plus one re-encode at a coarser $q$ when a frame exceeds
its budget by more than 25 %.

**Display post-filter.**  A deblocking filter smooths block edges whose
discontinuity is smaller than a quantiser step, outside the prediction loop.

### 6.1 Motion-interpolated display

The channel carries 2–4 real frames per second; the viewer renders 30.  Let
$F_0$ and $F_1$ be consecutive decoded frames and $v(p)$ the displacement
field with $F_1(p) \approx F_0(p + v(p))$.  For the digital codec $v$ is the
decoded macroblock motion field, bilinearly upsampled to pixel resolution
(INTRA blocks borrow the median of their neighbours); for the analog
transport a global translation is estimated with the Lucas–Kanade
least-squares solver of §6.  The frame at fraction $t$ of the interval is the
bidirectional motion-compensated blend

$$F_t(p) = (1-t)\,F_0\big(p + t\,v(p)\big) + t\,F_1\big(p - (1-t)\,v(p)\big),$$

evaluated with bilinear sampling.  Real frames arrive in bursts of one
superframe, so the viewer runs a playout clock delayed by about one
superframe (1.7 s): every real frame is scheduled at its nominal media time
and the render thread always has the bracketing pair available.  On a
translated test scene the interpolated midpoint is more than 6 dB closer to
the true intermediate frame than a cross-fade.  Nothing is transmitted for
this; it costs the receiver a few milliseconds per displayed frame.

## 7. Analog video transport

Let $V$ be the YCbCr GOP volume ($T\times H\times W$ per plane, $T$ = frames
per superframe) with the plane mean removed (the mean is sent digitally).
Coefficients $D = \mathcal{C}_T V \mathcal{C}_H^T \ldots$ (separable
orthonormal 3-D DCT).  $D$ is partitioned into chunks: one temporal index by
an 8×8 (luma) or 4×4 (chroma) tile of spatial frequencies, the tile at the
spatial DC split dyadically down to single coefficients.  Each chunk $c$ has
variance $\lambda_c$ (quantised in 3 dB steps and entropy coded as side
information with the selection mask).

**Chunk selection.**  Chunks are sent in order of decreasing $\lambda_c$ until
the symbol budget is used; the rest are zero at the receiver.  This is the
optimal allocation for a linear transmitter under a bandwidth constraint.

**Power allocation.**  With scale $g_c$ per chunk, the received distortion of
chunk $c$ is $\sigma^2/g_c^2$ per coefficient and its transmit power
$g_c^2\lambda_c$.  Minimising $\sum_c n_c \sigma^2/g_c^2$ subject to
$\sum_c n_c g_c^2 \lambda_c = P$ with a Lagrange multiplier gives
$g_c^4 \lambda_c = \text{const}$, i.e.

$$g_c = \Big(\frac{P}{\sum_i n_i\sqrt{\lambda_i}}\Big)^{1/2}\lambda_c^{-1/4}.$$

**Mixing.**  Scaled values are interleaved across chunks and multiplied in
groups of 64 by the orthonormal Sylvester–Hadamard matrix $\mathcal{H}$
($\mathcal{H}\mathcal{H}^T=I$).  Every transmitted value is then a sum of 64
values from different chunks: near-Gaussian, equal power, so a fixed clip at
$\pm3.5$ affects < 0.2 % of values, and the loss of any block is spread
evenly over all coefficients.

**Decoding (LLSE).**  For each group, with $G = \mathcal{H}\,\mathrm{diag}(g)$,
$\Lambda = \mathrm{diag}(\lambda)$ (per value) and $\Sigma$ the diagonal of
per-value noise variances reported by the modem (infinite for missing
values),

$$\hat x = \Lambda G^T \big(G\Lambda G^T + \Sigma\big)^{-1} y .$$

This is the linear minimum-mean-square estimator; it automatically shrinks
noisy or missing components towards zero, which is why the picture degrades
smoothly rather than failing.  Measured on the synthetic scene at 64×48 with
four analog blocks per superframe: 25.5 dB PSNR at 25 dB SNR, 23 dB at 6 dB
SNR, 19 dB at 3 dB SNR (digital side information in QPSK-1/2).

## 8. Speech vocoder

30 ms frames, order-10 LPC.  Pre-emphasis $1 - 0.9z^{-1}$, 40 ms Hamming
window, autocorrelation with a Gaussian lag window and white-noise
correction, **Levinson–Durbin** recursion for $A(z)$, bandwidth expansion
$0.994^i$.  Line spectral pairs from the roots of
$P(z) = A(z) + z^{-11}A(z^{-1})$ and $Q(z) = A(z) - z^{-11}A(z^{-1})$
(after removing $z=\mp1$); the LSP *differences* are quantised on a log scale
with per-index bit allocations (3,3,3,3,3,2,2,2,2,2 bits at 1200 bit/s),
which guarantees a stable, ordered reconstruction.  Pitch from the
normalised autocorrelation of the low-passed LPC residual over lags 20–160
with sub-multiple checking (6–7 bits, log scale), gain in dB (4–5 bits),
voicing 1 bit (2400 bit/s: 4 bands × 2 sub-frames).  Synthesis interpolates
LSPs per 7.5 ms sub-frame, excites $1/A(z)$ with a dispersed pulse train in
voiced bands and noise in unvoiced bands (4-band filter bank), applies the
post-filter $A(z/0.5)/A(z/0.8)$ with gain normalisation and de-emphasis.
Lost frames are concealed by fading the previous parameters.

## 9. Budgets, latency, performance

| item | value |
|------|-------|
| superframe | 2448 symbols, 1.53 s, 12240 samples |
| pilot + CP + scattered-pilot overhead | 11 % + 5.9 % + 3.1 % |
| header block | QPSK-1/2, 31 bytes |
| payload, 16-QAM 3/4 | 7 × 93 = 651 bytes (3.4 kbit/s) |
| voice 1200 bit/s | 237 bytes per superframe |
| video, 16-QAM 3/4 with voice | ≈ 1.8–2.0 kbit/s, 96×72 at 1.96 fps, ≈ 27 dB PSNR |
| end-to-end latency | ≈ 2 superframes + key-up (3.5–4 s) |
| CPU (Python/NumPy, one core) | encode ≈ 0.2 × real time, decode ≈ 0.1 × real time |

## 10. Interoperability checklist

* Sample rate 8000, symbol rate 1600, carrier 1700, RRC α 0.35 span 10.
* Preamble ZC(127, u=7) ×2; pilot ZC(256, u=29); scattered pilots: the first
  8 elements of ZC(9, u=1) normalised, at body positions 16+32i.
* Blocks: CP 16 + body 256; 8 data blocks per superframe; tail 24 zeros.
* LDPC: p = 31, column weight 3, seeds 1234 + 1000·bc + br (see `ldpc.py`),
  interleaver = NumPy default_rng(0xC0DE + n).permutation(n).
* Constellations: Gray square QAM, unit power; header block mode 0.
* Shaped modes 7–9: compositions in `shaping.PAS_TABLE`; amplitude slot
  $j$ (0..1487) = amplitude $j/3$ → symbol $(j/3)/2$, axis $(j/3) \bmod 2$, label
  bit $1 + j \bmod 3$ (+4 for Q); the first 1488 free columns of the LDPC code
  in ascending order are the amplitude slots, the pivot columns in ascending
  order are the sign slots (symbol $s/2$, axis $s \bmod 2$).
* Packets and header fields: see `framing.py`.
* Video codec syntax: see `videocodec.py` (contexts in class `Ctx`).
* Analog transport: see `softvideo.py` (chunk rectangles in `SoftConfig.rects`).
