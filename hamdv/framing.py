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
from .params import ModemParams, DEFAULT, DIGITAL_MODES, HEADER_MODE, PAS_RATES
from .shaping import pas_mode, AMPS_PER_BLOCK

MAGIC = b"HV"
VERSION = 1

PKT_PAD, PKT_HEADER, PKT_AUDIO, PKT_VIDEO, PKT_VMETA, PKT_TEXT, PKT_INFO = 0, 1, 2, 3, 4, 5, 6
FLAG_LAST = 1
FLAG_ANALOG_VIDEO = 2
FLAG_LONGCODE = 4          # one LDPC codeword spans all digital payload blocks of the superframe
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

    @property
    def long_code(self) -> bool:
        return bool(self.flags & FLAG_LONGCODE)

    def n_analog_blocks(self, p: ModemParams) -> int:
        return sum(1 for b in range(1, p.blocks_per_sf) if (self.block_types >> b) & 1)


# ------------------------------------------------------------------ block coding
@lru_cache(maxsize=None)
def interleaver(n: int) -> np.ndarray:
    return np.random.default_rng(0xC0DE + n).permutation(n)


def mode_code(mode: int, p: ModemParams = DEFAULT, nb: int = 1):
    """LDPC code for `nb` data blocks coded together (nb > 1 = long code spanning the superframe)."""
    name, bps, rn, rd = DIGITAL_MODES[mode]
    n = p.data_sym_per_block * bps * nb
    return get_code(n, rn, rd), bps


@lru_cache(maxsize=None)
def pas_layout(mode: int, p: ModemParams = DEFAULT, nb: int = 1):
    """Column -> (block, symbol, bit) assignment for a shaped mode over nb blocks.

    The first 3 x 496 x nb (information) columns of the systematic LDPC code
    carry the amplitude labels (block-major), the parity columns land on the
    two sign bits of each symbol.  Returns (code, pas, n_amp_bits, sym_bit (n, 3)).
    """
    code, bps = mode_code(mode, p, nb)
    pas = pas_mode(PAS_RATES[mode])
    n_amp_bits = 3 * AMPS_PER_BLOCK * nb
    if code.k != n_amp_bits:
        raise RuntimeError("PAS layout needs k == 3 x 496 x nb")
    nsym = p.data_sym_per_block
    sym_bit = np.zeros((code.n, 3), dtype=np.int64)
    j = np.arange(n_amp_bits)
    i = j // 3                                   # amplitude index (block-major)
    blk, il = i // AMPS_PER_BLOCK, i % AMPS_PER_BLOCK
    sym_bit[:n_amp_bits, 0] = blk
    sym_bit[:n_amp_bits, 1] = il // 2
    sym_bit[:n_amp_bits, 2] = 1 + (j % 3) + 4 * (il % 2)
    s = np.arange(code.n - n_amp_bits)           # sign slots (block-major)
    sym_bit[n_amp_bits:, 0] = s // AMPS_PER_BLOCK
    sym_bit[n_amp_bits:, 1] = (s % AMPS_PER_BLOCK) // 2
    sym_bit[n_amp_bits:, 2] = 4 * (s % 2)
    return code, pas, n_amp_bits, sym_bit


def block_capacity_bytes(mode: int, p: ModemParams = DEFAULT, nb: int = 1) -> int:
    """User bytes carried by a group of nb blocks coded together."""
    if mode in PAS_RATES:
        return (pas_mode(PAS_RATES[mode]).k_dm * nb) // 8
    code, _ = mode_code(mode, p, nb)
    return code.k // 8


def encode_digital_group(data: bytes, mode: int, nb: int = 1, p: ModemParams = DEFAULT) -> List[np.ndarray]:
    """bytes (<= capacity) -> nb arrays of 248 QAM symbols (one LDPC codeword over all of them)."""
    if mode in PAS_RATES:
        return _encode_pas_group(data, mode, nb, p)
    code, bps = mode_code(mode, p, nb)
    bits = np.unpackbits(np.frombuffer(data.ljust(code.k // 8, b"\0"), dtype=np.uint8))
    u = np.zeros(code.k, dtype=np.uint8)
    u[: min(len(bits), code.k)] = bits[: code.k]
    cw = code.encode(u)[interleaver(code.n)]
    syms = map_bits(cw, bps)
    return [syms[i * p.data_sym_per_block:(i + 1) * p.data_sym_per_block] for i in range(nb)]


def encode_digital_block(data: bytes, mode: int, p: ModemParams = DEFAULT) -> np.ndarray:
    return encode_digital_group(data, mode, 1, p)[0]


def _encode_pas_group(data: bytes, mode: int, nb: int, p: ModemParams) -> List[np.ndarray]:
    code, pas, n_amp_bits, sym_bit = pas_layout(mode, p, nb)
    cap = (pas.k_dm * nb) // 8
    bits = np.unpackbits(np.frombuffer(data.ljust(cap, b"\0")[:cap], dtype=np.uint8))
    bits = np.concatenate([bits, np.zeros(pas.k_dm * nb - len(bits), dtype=np.uint8)])
    u = np.concatenate([pas.amplitude_bits(bits[b * pas.k_dm:(b + 1) * pas.k_dm]) for b in range(nb)])
    cw = code.encode(u)
    labels = np.zeros((nb, p.data_sym_per_block, 8), dtype=np.uint8)
    labels[sym_bit[:, 0], sym_bit[:, 1], sym_bit[:, 2]] = cw
    return [pas.map_labels(labels[b]) for b in range(nb)]


def _decode_pas_group(blocks: List[EqualisedBlock], mode: int, p: ModemParams) -> Tuple[bytes, bool]:
    nb = len(blocks)
    code, pas, n_amp_bits, sym_bit = pas_layout(mode, p, nb)
    l = np.stack([llrs(b.symbols, 8, b.noise_var, points=pas.points, log_prior=pas.log_prior).reshape(-1, 8) for b in blocks])
    col_llr = l[sym_bit[:, 0], sym_bit[:, 1], sym_bit[:, 2]]
    cw, ok, _ = code.decode(col_llr)
    u = code.extract(cw)
    parts, comp_ok = [], True
    for b in range(nb):
        ub, cok = pas.user_bits(u[b * 3 * AMPS_PER_BLOCK:(b + 1) * 3 * AMPS_PER_BLOCK])
        parts.append(ub)
        comp_ok &= cok
    bits = np.concatenate(parts)
    cap = (pas.k_dm * nb) // 8
    return np.packbits(bits[: cap * 8]).tobytes(), ok and comp_ok


def decode_digital_group(blocks: List[EqualisedBlock], mode: int, p: ModemParams = DEFAULT) -> Tuple[bytes, bool]:
    if mode in PAS_RATES:
        return _decode_pas_group(blocks, mode, p)
    nb = len(blocks)
    code, bps = mode_code(mode, p, nb)
    l = np.concatenate([llrs(b.symbols, bps, b.noise_var) for b in blocks])
    de = np.empty_like(l)
    de[interleaver(code.n)] = l
    cw, ok, _ = code.decode(de)
    u = code.extract(cw)[: (code.k // 8) * 8]
    return np.packbits(u).tobytes(), ok


def decode_digital_block(blk: EqualisedBlock, mode: int, p: ModemParams = DEFAULT) -> Tuple[bytes, bool]:
    return decode_digital_group([blk], mode, p)


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
    groups: List[List[int]]     # digital blocks coded together (one LDPC codeword each)


def plan(header: Header, p: ModemParams = DEFAULT) -> SuperframePlan:
    dig, ana = [], []
    for b in range(1, p.blocks_per_sf):
        (ana if (header.block_types >> b) & 1 else dig).append(b)
    groups = [dig] if (header.long_code and dig) else [[b] for b in dig]
    payload = sum(block_capacity_bytes(header.dig_mode, p, len(g)) for g in groups)
    return SuperframePlan(block_capacity_bytes(HEADER_MODE, p), payload,
                          len(ana) * 2 * p.data_sym_per_block, dig, ana, groups)


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
    pos = 0
    for g in pl.groups:
        cap = block_capacity_bytes(header.dig_mode, p, len(g))
        for b, s in zip(g, encode_digital_group(area[pos:pos + cap], header.dig_mode, len(g), p)):
            syms[b] = s
        pos += cap
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
    for g in pl.groups:
        data, ok = decode_digital_group([blocks[b] for b in g], header.dig_mode, p)
        n_ok += len(g) if ok else 0
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
