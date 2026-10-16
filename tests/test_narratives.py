import numpy as np
import pytest

from fusi_tribe.narratives.analysis import best_lag, loo_isc, split_half_ceiling


def test_isc_high_for_shared_signal_and_low_for_noise():
    rng = np.random.default_rng(0)
    signal = rng.normal(size=300)
    shared = signal + 0.5 * rng.normal(size=(20, 300))
    noise = rng.normal(size=(20, 300))
    assert loo_isc(shared).mean() > 0.7
    assert abs(loo_isc(noise).mean()) < 0.05


def test_split_half_ceiling_tracks_snr():
    rng = np.random.default_rng(1)
    signal = rng.normal(size=300)
    clean = signal + 0.5 * rng.normal(size=(20, 300))
    noisy = signal + 3.0 * rng.normal(size=(20, 300))
    assert split_half_ceiling(clean, rng) > split_half_ceiling(noisy, rng) > 0


@pytest.mark.parametrize("shift", [-3, 0, 2])
def test_best_lag_recovers_planted_shift(shift):
    rng = np.random.default_rng(2)
    x = np.convolve(rng.normal(size=400), np.ones(5) / 5, mode="same")
    target = np.roll(x, shift)
    assert best_lag(x, target) == shift
