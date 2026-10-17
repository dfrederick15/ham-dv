"""Physical-layer and framing constants.

All symbol-level quantities are chosen so that the waveform fits inside the
audio passband of a narrow-band FM voice repeater (roughly 300-3000 Hz):

    symbol rate 1600 Bd, carrier 1700 Hz, RRC roll-off 0.35
    => occupied band 1700 +- 1080 Hz = 620 ... 2780 Hz
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModemParams:
    fs: int = 8000               # internal sample rate (Hz)
    sps: int = 5                 # samples per symbol -> 1600 Bd
    fc: float = 1700.0           # carrier (Hz)
    alpha: float = 0.35          # RRC roll-off
    rrc_span: int = 10           # RRC length in symbols
    n_fft: int = 256             # symbols per block (FFT size)
    cp: int = 16                 # cyclic prefix length (symbols)
    win_lead: int = 4            # FFT window starts this many symbols into the CP (pre-cursor room)
    n_pilot_sym: int = 8         # scattered pilots per data block
    blocks_per_sf: int = 8       # data blocks per superframe (plus one pilot block)
    preamble_len: int = 127      # Zadoff-Chu length of each preamble repetition
    preamble_reps: int = 2
    preamble_root: int = 7
    pilot_root: int = 29         # ZC root for the 256-symbol pilot block (odd => coprime with 256)
    tail_sym: int = 24           # idle symbols after the last superframe

    @property
    def baud(self) -> float:
        return self.fs / self.sps

    @property
    def block_len(self) -> int:
        return self.n_fft + self.cp

    @property
    def sf_len(self) -> int:
        """Symbols per superframe = pilot block + data blocks."""
        return (self.blocks_per_sf + 1) * self.block_len

    @property
    def sf_samples(self) -> int:
        return self.sf_len * self.sps

    @property
    def sf_seconds(self) -> float:
        return self.sf_len / self.baud

    @property
    def data_sym_per_block(self) -> int:
        return self.n_fft - self.n_pilot_sym

    @property
    def pilot_positions(self):
        step = self.n_fft // self.n_pilot_sym
        return [step // 2 + i * step for i in range(self.n_pilot_sym)]


# Digital block modes: (name, bits per symbol, rate numerator, rate denominator)
DIGITAL_MODES = {
    0: ("qpsk-1/2", 2, 1, 2),
    1: ("qpsk-3/4", 2, 3, 4),
    2: ("16qam-1/2", 4, 1, 2),
    3: ("16qam-3/4", 4, 3, 4),
    4: ("64qam-1/2", 6, 1, 2),
    5: ("64qam-3/4", 6, 3, 4),
}
MODE_BY_NAME = {v[0]: k for k, v in DIGITAL_MODES.items()}
HEADER_MODE = 0          # block 0 of every superframe always uses the most robust mode

BLOCK_DIGITAL = 0
BLOCK_ANALOG = 1

DEFAULT = ModemParams()
