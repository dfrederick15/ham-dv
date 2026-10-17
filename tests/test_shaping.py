import numpy as np
from hamdv.shaping import (PAS_TABLE, design_composition, multinomial, unrank, rank, bits_to_int, int_to_bits,
                           pas_mode, AMPS_PER_BLOCK)
from hamdv.framing import encode_digital_block, decode_digital_block, block_capacity_bytes
from hamdv.modem import EqualisedBlock
from hamdv.params import PAS_RATES, DIGITAL_MODES


def test_table_matches_design_and_is_invertible():
    rng = np.random.default_rng(0)
    for rate, (k, comp) in PAS_TABLE.items():
        assert sum(comp) == AMPS_PER_BLOCK
        assert design_composition(k) == comp
        assert multinomial(comp).bit_length() - 1 >= k
        for _ in range(3):
            bits = rng.integers(0, 2, k)
            seq = unrank(bits_to_int(bits), comp)
            assert [seq.count(a) for a in range(8)] == comp
            assert np.array_equal(int_to_bits(rank(seq, comp), k), bits)


def test_shaped_points_have_unit_power_and_prior_sums_to_one():
    for r in PAS_TABLE:
        m = pas_mode(r)
        pr = np.exp(m.log_prior)
        assert abs(pr.sum() - 1) < 1e-9
        assert abs(np.sum(pr * np.abs(m.points) ** 2) - 1) < 1e-9
        # shaped power is lower than uniform at the same point spacing
        assert np.mean(np.abs(m.points) ** 2) > 1.5


def test_pas_block_round_trip_and_gain():
    rng = np.random.default_rng(3)
    for mode in PAS_RATES:
        cap = block_capacity_bytes(mode)
        data = rng.bytes(cap)
        s = encode_digital_block(data, mode)
        assert abs(np.mean(np.abs(s) ** 2) - 1) < 0.1
        nv = 10 ** (-21 / 10)
        y = s + np.sqrt(nv / 2) * (rng.standard_normal(len(s)) + 1j * rng.standard_normal(len(s)))
        out, ok = decode_digital_block(EqualisedBlock(0, 0, y, nv, 21.0), mode)
        assert ok and out == data, DIGITAL_MODES[mode][0]
