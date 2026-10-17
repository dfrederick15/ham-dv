"""Link layer: superframe layout, header, packets, block coding.

Superframe (one pilot block + B data blocks, ~1.53 s at 1600 Bd):

    block 0        : always digital, QPSK rate-1/2 (robust) -> header packet
    blocks 1..B-1  : digital (mode announced in the header) or analog
                     (bit set in the header's block-type bitmap)

Digital bytes of a superframe form one contiguous area that is filled with
packets:

    [type u8][length u16][payload][crc16]    type 0 = padding (stop)

Every digital block is one LDPC codeword, bit-interleaved with a fixed
pseudo-random permutation and mapped to Gray-coded QAM.  Analog blocks carry
496 real values as the I and Q components of 248 symbols.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import numpy as np

from .constellation import map_bits, llrs
from .ldpc import get_code
from .modem import build_data_block, build_pilot_block, preamble_symbols, EqualisedBlock
from .params import ModemParams, DEFAULT, DIGITAL_MODES, HEADER_MODE

MAGIC = b"HV"
VERSION = 1

PKT_PAD, PKT_HEADER, PKT_AUDIO, PKT_VIDEO, PKT_VMETA, PKT_TEXT, PKT_INFO = 0, 1, 2, 3, 4, 5, 6
FLAG_LAST = 1
FLAG_ANALOG_VIDEO = 2
AUDIO_MODES = {0: 0, 1: 1200, 2: 1600, 3: 2400}
ANALOG_BACKOFF = 0.75      # analog symbols are Gaussian: back them off 2.5 dB so the radio's limiter stays out
AUDIO_MODE_IDX = {v: k for k, v in AUDIO_MODES.items()}


# ------------------------------------------------------------------ CRC-16
def crc16(data: bytes, poly: int = 0x1021, init: int = 0xFFFF) -> int:
    crc = init
    for b in data:
        crc ^= b << 8
        for _ in range(8):
            crc = ((crc << 1) ^ poly) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc


# ------------------------------------------------------------------ packets
def pack_packet(ptype: int, payload: bytes) -> bytes:
    head = struct.pack("<BH", ptype, len(payload))
    body = head + payload
    return body + struct.pack("<H", crc16(body))


def parse_packets(area: bytes) -> List[Tuple[int, bytes]]:
    """Walk the digital area; returns list of (type, payload) of packets with good CRC.

    A CRC failure resynchronises by scanning forward for the next plausible
    header, so one corrupted block does not necessarily hide every packet
    after it.
    """
    out = []
    pos = 0
    n = len(area)
    while pos + 5 <= n:
        ptype, ln = struct.unpack_from("<BH", area, pos)
        if ptype == PKT_PAD:
            break
        end = pos + 3 + ln + 2
        if ptype > PKT_INFO or end > n:
            pos += 1
            continue
        body = area[pos:pos + 3 + ln]
        (crc,) = struct.unpack_from("<H", area, pos + 3 + ln)
        if crc16(body) == crc:
            out.append((ptype, bytes(area[pos + 3:pos + 3 + ln])))
            pos = end
        else:
            pos += 1
    return out


# ------------------------------------------------------------------ header
@dataclass
class Header:
    callsign: str
    tx_id: int
    sf_seq: int
    flags: int
    dig_mode: int
    block_types: int           # bit b set -> data block b is analog (bit 0 unused)
    vid_w: int
    vid_h: int
    vid_fps_x10: int
    audio_mode: int            # index into AUDIO_MODES
    frames_per_sf: int
    refresh: int = 12          # digital video refresh period (frames)

    SIZE = 22

    def pack(self) -> bytes:
        cs = self.callsign.upper().encode("ascii", "replace")[:8].ljust(8)
        return (MAGIC + bytes([VERSION]) + cs + struct.pack("<BHBBBBBBBBB", self.tx_id & 0xFF, self.sf_seq & 0xFFFF,
                self.flags, self.dig_mode, self.block_types & 0xFF, self.vid_w // 8, self.vid_h // 8,
                self.vid_fps_x10, self.audio_mode, self.frames_per_sf, self.refresh))

    @classmethod
    def unpack(cls, data: bytes) -> Optional["Header"]:
        if len(data) < cls.SIZE or data[:2] != MAGIC or data[2] != VERSION:
            return None
        cs = data[3:11].decode("ascii", "replace").strip()
        (tx_id, sf_seq, flags, dig_mode, bt, w8, h8, fps, am, fpsf, refresh) = struct.unpack_from("<BHBBBBBBBBB", data, 11)
        if dig_mode not in DIGITAL_MODES or am not in AUDIO_MODES:
            return None
        return cls(cs, tx_id, sf_seq, flags, dig_mode, bt, w8 * 8, h8 * 8, fps, am, fpsf, refresh)

    @property
    def is_last(self) -> bool:
        return bool(self.flags & FLAG_LAST)

    @property
    def analog_video(self) -> bool:
        return bool(self.flags & FLAG_ANALOG_VIDEO)

    def n_analog_blocks(self, p: ModemParams) -> int:
        return sum(1 for b in range(1, p.blocks_per_sf) if (self.block_types >> b) & 1)


# ------------------------------------------------------------------ block coding
@lru_cache(maxsize=None)
def interleaver(n: int) -> np.ndarray:
    return np.random.default_rng(0xC0DE + n).permutation(n)


def mode_code(mode: int, p: ModemParams = DEFAULT):
    name, bps, rn, rd = DIGITAL_MODES[mode]
    n = p.data_sym_per_block * bps
    return get_code(n, rn, rd), bps


def block_capacity_bytes(mode: int, p: ModemParams = DEFAULT) -> int:
    code, _ = mode_code(mode, p)
    return code.k // 8


def encode_digital_block(data: bytes, mode: int, p: ModemParams = DEFAULT) -> np.ndarray:
    """bytes (<= capacity) -> 248 QAM symbols."""
    code, bps = mode_code(mode, p)
    bits = np.unpackbits(np.frombuffer(data.ljust(code.k // 8, b"\0"), dtype=np.uint8))
    u = np.zeros(code.k, dtype=np.uint8)
    u[: min(len(bits), code.k)] = bits[: code.k]
    cw = code.encode(u)
    cw = cw[interleaver(code.n)]
    return map_bits(cw, bps)


def decode_digital_block(blk: EqualisedBlock, mode: int, p: ModemParams = DEFAULT) -> Tuple[bytes, bool]:
    code, bps = mode_code(mode, p)
    l = llrs(blk.symbols, bps, blk.noise_var)
    de = np.empty_like(l)
    de[interleaver(code.n)] = l
    cw, ok, _ = code.decode(de)
    u = code.extract(cw)[: (code.k // 8) * 8]
    return np.packbits(u).tobytes(), ok


def encode_analog_block(values: np.ndarray, p: ModemParams = DEFAULT) -> np.ndarray:
    n = p.data_sym_per_block
    v = np.zeros(2 * n)
    m = min(len(values), 2 * n)
    v[:m] = values[:m]
    return (v[0::2] + 1j * v[1::2]) * ANALOG_BACKOFF


def decode_analog_block(blk: EqualisedBlock) -> Tuple[np.ndarray, np.ndarray]:
    """-> (496 real values, per-value noise variance)"""
    v = np.empty(2 * len(blk.symbols))
    v[0::2] = blk.symbols.real
    v[1::2] = blk.symbols.imag
    v /= ANALOG_BACKOFF
    return v, np.full(len(v), blk.noise_var / 2.0 / ANALOG_BACKOFF ** 2)


# ------------------------------------------------------------------ superframe assembly
@dataclass
class SuperframePlan:
    """How many bytes / values a superframe offers for the given layout."""
    header_bytes: int
    payload_bytes: int          # digital bytes in blocks 1..B-1 (contiguous area, after header block)
    analog_values: int
    digital_blocks: List[int]
    analog_blocks: List[int]


def plan(header: Header, p: ModemParams = DEFAULT) -> SuperframePlan:
    dig, ana = [], []
    for b in range(1, p.blocks_per_sf):
        (ana if (header.block_types >> b) & 1 else dig).append(b)
    return SuperframePlan(block_capacity_bytes(HEADER_MODE, p),
                          len(dig) * block_capacity_bytes(header.dig_mode, p),
                          len(ana) * 2 * p.data_sym_per_block, dig, ana)


def build_superframe(header: Header, packets: List[bytes], analog_values: np.ndarray,
                     p: ModemParams = DEFAULT) -> np.ndarray:
    """Return the superframe's symbols (pilot block + B data blocks)."""
    pl = plan(header, p)
    area0 = pack_packet(PKT_HEADER, header.pack())
    if len(area0) > pl.header_bytes:
        raise ValueError("header does not fit")
    area = b"".join(packets)
    if len(area) > pl.payload_bytes:
        raise ValueError(f"payload {len(area)} exceeds superframe capacity {pl.payload_bytes}")
    area = area.ljust(pl.payload_bytes, b"\0")
    blocks = [build_pilot_block(p)]
    syms = {0: encode_digital_block(area0.ljust(pl.header_bytes, b"\0"), HEADER_MODE, p)}
    cap = block_capacity_bytes(header.dig_mode, p)
    for i, b in enumerate(pl.digital_blocks):
        syms[b] = encode_digital_block(area[i * cap:(i + 1) * cap], header.dig_mode, p)
    nv = 2 * p.data_sym_per_block
    for i, b in enumerate(pl.analog_blocks):
        syms[b] = encode_analog_block(analog_values[i * nv:(i + 1) * nv], p)
    for b in range(p.blocks_per_sf):
        blocks.append(build_data_block(p, syms[b]))
    return np.concatenate(blocks)


@dataclass
class DecodedSuperframe:
    header: Header
    packets: List[Tuple[int, bytes]]
    analog_values: np.ndarray
    analog_noise: np.ndarray
    blocks_ok: int
    blocks_total: int
    snr_db: float


def decode_superframe(blocks: List[EqualisedBlock], p: ModemParams = DEFAULT,
                      last_header: Optional[Header] = None) -> Optional[DecodedSuperframe]:
    """blocks: the B equalised data blocks of one superframe (block index order)."""
    if len(blocks) != p.blocks_per_sf:
        return None
    area0, ok0 = decode_digital_block(blocks[0], HEADER_MODE, p)
    header = None
    for t, payload in parse_packets(area0):
        if t == PKT_HEADER:
            header = Header.unpack(payload)
            break
    if header is None:
        if last_header is None:
            return None
        # header lost: assume the layout did not change (sequence number unknown)
        header = Header(**{**last_header.__dict__, "sf_seq": (last_header.sf_seq + 1) & 0xFFFF, "flags": last_header.flags & ~FLAG_LAST})
        hdr_ok = False
    else:
        hdr_ok = True
    pl = plan(header, p)
    area = bytearray()
    n_ok = 1 if hdr_ok else 0
    for b in pl.digital_blocks:
        data, ok = decode_digital_block(blocks[b], header.dig_mode, p)
        n_ok += 1 if ok else 0
        area += data
    vals, noise = [], []
    for b in pl.analog_blocks:
        v, nv = decode_analog_block(blocks[b])
        vals.append(v)
        noise.append(nv)
        n_ok += 1
    packets = parse_packets(bytes(area))
    snr = float(np.mean([b.snr_db for b in blocks]))
    return DecodedSuperframe(header, packets,
                             np.concatenate(vals) if vals else np.zeros(0),
                             np.concatenate(noise) if noise else np.zeros(0),
                             n_ok, p.blocks_per_sf, snr)


def transmission_symbols(superframes: List[np.ndarray], p: ModemParams = DEFAULT) -> np.ndarray:
    return np.concatenate([preamble_symbols(p)] + superframes + [np.zeros(p.tail_sym, dtype=np.complex128)])
