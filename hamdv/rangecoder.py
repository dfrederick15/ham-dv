"""Adaptive binary range coder (LZMA-style) used by the video codec.

Each binary decision is coded against an 11-bit probability estimate that
adapts with a shift-5 exponential update:
    p0 <- p0 + (2048 - p0) >> 5   after a 0
    p0 <- p0 - p0 >> 5            after a 1
The coder is carry-less in the output (cache + pending 0xFF bytes) and uses
a 32-bit range; integer arithmetic only, so encoder and decoder are
bit-exact on every platform.
"""
from __future__ import annotations

PROB_BITS = 11
PROB_INIT = 1 << (PROB_BITS - 1)
MOVE_BITS = 5
TOP = 1 << 24
MASK32 = 0xFFFFFFFF


class Encoder:
    def __init__(self):
        self.low = 0
        self.range = MASK32
        self.cache = 0
        self.cache_size = 1
        self.out = bytearray()

    def _shift_low(self):
        if self.low < 0xFF000000 or self.low > MASK32:
            carry = self.low >> 32
            temp = self.cache
            while True:
                self.out.append((temp + carry) & 0xFF)
                temp = 0xFF
                self.cache_size -= 1
                if self.cache_size == 0:
                    break
            self.cache = (self.low >> 24) & 0xFF
        self.cache_size += 1
        self.low = (self.low & 0x00FFFFFF) << 8

    def encode(self, probs: list, idx: int, bit: int):
        p = probs[idx]
        bound = (self.range >> PROB_BITS) * p
        if bit == 0:
            self.range = bound
            probs[idx] = p + (((1 << PROB_BITS) - p) >> MOVE_BITS)
        else:
            self.low += bound
            self.range -= bound
            probs[idx] = p - (p >> MOVE_BITS)
        while self.range < TOP:
            self.range = (self.range << 8) & MASK32
            self._shift_low()

    def encode_direct(self, value: int, nbits: int):
        for i in range(nbits - 1, -1, -1):
            self.range >>= 1
            if (value >> i) & 1:
                self.low += self.range
            while self.range < TOP:
                self.range = (self.range << 8) & MASK32
                self._shift_low()

    def finish(self) -> bytes:
        for _ in range(5):
            self._shift_low()
        return bytes(self.out)


class Decoder:
    def __init__(self, data: bytes):
        self.data = data
        self.pos = 0
        self.range = MASK32
        self.code = 0
        for _ in range(5):
            self.code = ((self.code << 8) | self._next()) & MASK32

    def _next(self) -> int:
        if self.pos < len(self.data):
            b = self.data[self.pos]
            self.pos += 1
            return b
        self.pos += 1
        return 0

    def overrun(self) -> bool:
        return self.pos > len(self.data) + 8

    def decode(self, probs: list, idx: int) -> int:
        p = probs[idx]
        bound = (self.range >> PROB_BITS) * p
        if self.code < bound:
            self.range = bound
            probs[idx] = p + (((1 << PROB_BITS) - p) >> MOVE_BITS)
            bit = 0
        else:
            self.code -= bound
            self.range -= bound
            probs[idx] = p - (p >> MOVE_BITS)
            bit = 1
        while self.range < TOP:
            self.range = (self.range << 8) & MASK32
            self.code = ((self.code << 8) | self._next()) & MASK32
        return bit

    def decode_direct(self, nbits: int) -> int:
        v = 0
        for _ in range(nbits):
            self.range >>= 1
            if self.code >= self.range:
                self.code -= self.range
                bit = 1
            else:
                bit = 0
            v = (v << 1) | bit
            while self.range < TOP:
                self.range = (self.range << 8) & MASK32
                self.code = ((self.code << 8) | self._next()) & MASK32
        return v


# ------------------------------------------------------------------ helpers
def encode_unary_eg(enc: Encoder, probs: list, base: int, nctx: int, value: int):
    """Unary prefix with adaptive contexts (up to nctx), Exp-Golomb(0) suffix in bypass mode."""
    for i in range(nctx):
        if value == i:
            enc.encode(probs, base + i, 0)
            return
        enc.encode(probs, base + i, 1)
    v = value - nctx
    # exp-golomb order 0 of v
    v1 = v + 1
    n = v1.bit_length() - 1
    for _ in range(n):
        enc.encode_direct(1, 1)
    enc.encode_direct(0, 1)
    enc.encode_direct(v1 & ((1 << n) - 1), n)


def decode_unary_eg(dec: Decoder, probs: list, base: int, nctx: int) -> int:
    for i in range(nctx):
        if dec.decode(probs, base + i) == 0:
            return i
    n = 0
    while dec.decode_direct(1) == 1 and n < 24:
        n += 1
    v1 = (1 << n) | dec.decode_direct(n)
    return v1 - 1 + nctx
