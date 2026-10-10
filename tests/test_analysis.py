import numpy as np
import pytest

from fusi_tribe.insilico.analysis import block_amplitudes, bootstrap_ci

BLOCKS = [{"trial_type": "Speech" if i % 2 else "Audio only", "onset": 10.0 + 35 * i, "duration": 15.0}
          for i in range(8)]
T = np.arange(0, 300.0)


def response(scale_speech=1.0, scale_control=0.0):
    y = np.full(T.size, -0.5)                      # arbitrary baseline offset
    for b in BLOCKS:
        on = (T >= b["onset"] + 3) & (T < b["onset"] + b["duration"] + 4)
        y[on] += scale_speech if b["trial_type"] == "Speech" else scale_control
    return y[:, None]


def retention(y):
    amp = block_amplitudes(y, BLOCKS, T)[:, 0]
    sp = np.array([b["trial_type"] == "Speech" for b in BLOCKS])
    return amp[~sp].mean() / amp[sp].mean()


def test_silent_control_gives_zero_retention():
    assert retention(response(1.0, 0.0)) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("ratio", [0.3, 1.0, 1.4])
def test_retention_recovers_amplitude_ratio(ratio):
    assert retention(response(2.0, 2.0 * ratio)) == pytest.approx(ratio)


def test_baseline_offset_does_not_matter():
    assert retention(response(1.0, 0.5) + 7.0) == pytest.approx(0.5)


def test_bootstrap_ci_brackets_mean():
    x = np.random.default_rng(0).normal(1.0, 0.1, 50)
    lo, hi = bootstrap_ci(np.mean, [x], 2000, np.random.default_rng(1))
    assert lo < x.mean() < hi and hi - lo < 0.1
