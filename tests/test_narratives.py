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


def test_subject_correlations_rank_a_true_predictor_above_noise():
    from fusi_tribe.narratives.analysis import subject_correlations
    rng = np.random.default_rng(4)
    signal = rng.normal(size=300)
    ts = signal + rng.normal(size=(15, 300))
    r_true, r_others = subject_correlations(ts, signal, lag=0)
    r_noise, _ = subject_correlations(ts, rng.normal(size=300), lag=0)
    assert r_true.mean() > 0.5 and abs(r_noise.mean()) < 0.1 and r_others.mean() > 0.5


def test_acf_timescale_and_slow_fraction_rank_slow_above_fast_signals():
    from fusi_tribe.narratives.analysis import acf_timescale, slow_fraction
    rng = np.random.default_rng(5)
    fast = rng.normal(size=400)
    slow = np.convolve(rng.normal(size=440), np.ones(40) / 40, mode="valid")[:400]
    assert acf_timescale(slow) > 5 * acf_timescale(fast)
    assert slow_fraction(slow) > slow_fraction(fast)


def test_matched_filter_removes_slow_drift_and_keeps_fast_signal():
    from fusi_tribe.narratives.analysis import matched_filter, temporal_filter_basis
    n = 400
    tt = np.arange(n) * 1.5
    fast = np.sin(2 * np.pi * tt / 20.0)            # 20 s period, well above the 128 s cutoff
    drift = 3.0 * (tt / tt[-1]) ** 2 + np.cos(2 * np.pi * tt / 600.0)
    out = matched_filter(fast + drift, temporal_filter_basis(n))
    assert np.corrcoef(out, fast)[0, 1] > 0.98
