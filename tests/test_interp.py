import numpy as np
from hamdv.interp import interpolate, dense_field, smooth_sequence
from hamdv.videocodec import psnr, shift_image
from hamdv.testvideo import synth_frame


def test_motion_interpolation_beats_crossfade():
    base = synth_frame(0.0, 96, 72)
    # f1 is f0 moved right by 6 px: f1(p) = f0(p + (-6, 0)) -> block vectors dx = -6
    f0 = base
    f1 = np.stack([shift_image(base[..., c], -6, 0) for c in range(3)], -1)
    truth = np.stack([shift_image(base[..., c], -3, 0) for c in range(3)], -1)
    n_mb = (96 // 8) * (72 // 8)
    mvs = np.tile([[-6, 0]], (n_mb, 1))
    field = dense_field(mvs, np.ones(n_mb), 12, 9, 72, 96)
    mid_mc = interpolate(f0, f1, 0.5, field)
    mid_fade = interpolate(f0, f1, 0.5, None)
    assert psnr(mid_mc, truth) > psnr(mid_fade, truth) + 6
    assert psnr(mid_mc, truth) > 35


def test_smooth_sequence_length():
    frames = [synth_frame(i / 2, 64, 48) for i in range(4)]
    out = smooth_sequence(frames, [None] * 4, 2.0, 30.0)
    assert 44 <= len(out) <= 47
    assert np.array_equal(out[0], frames[0]) and np.array_equal(out[-1], frames[-1])
