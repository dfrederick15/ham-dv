import numpy as np
from hamdv.ldpc import get_code
from hamdv.constellation import map_bits, llrs, hard_decide, constellation


def test_codes_build_and_encode():
    for n, rn, rd in [(496, 1, 2), (496, 3, 4), (992, 1, 2), (992, 3, 4), (1488, 3, 4)]:
        code = get_code(n, rn, rd)
        assert code.n == n
        assert abs(code.k / n - rn / rd) < 0.06
        rng = np.random.default_rng(0)
        u = rng.integers(0, 2, code.k)
        c = code.encode(u)
        assert code.syndrome_ok(c)
        assert np.array_equal(code.extract(c), u)


def test_constellations():
    for b in (2, 4, 6):
        pts, labels = constellation(b)
        assert abs(np.mean(np.abs(pts) ** 2) - 1) < 1e-9
        rng = np.random.default_rng(1)
        bits = rng.integers(0, 2, 60 * b)
        s = map_bits(bits, b)
        assert np.array_equal(hard_decide(s, b), bits)


def test_decode_awgn():
    code = get_code(992, 3, 4)
    rng = np.random.default_rng(2)
    b = 4
    ebn0_db = 8.0
    esn0 = 10 ** (ebn0_db / 10) * b * code.k / code.n
    nv = 1.0 / esn0
    errs = 0
    fails = 0
    for trial in range(20):
        u = rng.integers(0, 2, code.k)
        c = code.encode(u)
        s = map_bits(c, b)
        y = s + np.sqrt(nv / 2) * (rng.standard_normal(len(s)) + 1j * rng.standard_normal(len(s)))
        hard_err = np.sum(hard_decide(y, b) != c)
        l = llrs(y, b, nv)
        chat, ok, it = code.decode(l)
        errs += np.sum(code.extract(chat) != u)
        fails += 0 if ok else 1
        assert hard_err > 0  # channel really was noisy
    assert errs == 0, errs
    assert fails == 0
