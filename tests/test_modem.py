import numpy as np
from hamdv.params import DEFAULT as P
from hamdv.modem import Modulator, Demodulator, build_data_block, build_pilot_block, preamble_symbols
from hamdv.constellation import map_bits, hard_decide
from hamdv.channel import ChannelConfig, simulate


def make_tx(n_sf, bits_per_sym, rng):
    syms = [preamble_symbols(P)]
    payloads = []
    for s in range(n_sf):
        syms.append(build_pilot_block(P))
        for b in range(P.blocks_per_sf):
            bits = rng.integers(0, 2, P.data_sym_per_block * bits_per_sym)
            pl = map_bits(bits, bits_per_sym)
            payloads.append(pl)
            syms.append(build_data_block(P, pl))
    syms.append(np.zeros(P.tail_sym, dtype=complex))
    mod = Modulator(P)
    audio = np.concatenate([mod.modulate(np.concatenate(syms)), mod.flush()])
    return audio, payloads


def run(snr_db, bits_per_sym, n_sf=3, **kw):
    rng = np.random.default_rng(5)
    audio, payloads = make_tx(n_sf, bits_per_sym, rng)
    rx = simulate(audio * 0.3, ChannelConfig(snr_db=snr_db, **kw))
    dem = Demodulator(P)
    for i in range(0, len(rx), 1600):
        dem.feed(rx[i:i + 1600])
    dem.feed(np.zeros(8000))
    assert any(e.startswith("ACQ") for e in dem.events), dem.events
    assert len(dem.blocks) >= n_sf * P.blocks_per_sf, (len(dem.blocks), dem.events)
    errs = 0
    evm = []
    for blk, pl in zip(dem.blocks, payloads):
        errs += np.sum(hard_decide(blk.symbols, bits_per_sym) != hard_decide(pl, bits_per_sym))
        evm.append(np.mean(np.abs(blk.symbols - pl) ** 2))
    return errs, float(np.mean(evm)), dem


def test_clean_channel_qpsk():
    errs, evm, dem = run(30, 2)
    assert errs == 0
    assert evm < 0.01


def test_16qam_with_sro_and_delay():
    errs, evm, dem = run(24, 4, n_sf=4, sro_ppm=120, delay_s=0.0731, gain=0.5)
    assert errs == 0, errs
    assert abs(dem.sf_info[-1].sro_ppm - 120) < 40


def test_64qam_full_quieting():
    errs, evm, dem = run(32, 6, n_sf=3, sro_ppm=-60)
    # a handful of raw errors from the modelled mic limiter is fine: the LDPC code removes them
    assert errs <= 5, errs


def test_low_snr_qpsk_symbol_errors_small():
    errs, evm, dem = run(8, 2, n_sf=3)
    total = 3 * P.blocks_per_sf * P.data_sym_per_block * 2
    assert errs / total < 0.02
