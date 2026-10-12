"""The timing analysis must recover effects planted in synthetic data."""

import numpy as np
import pandas as pd
import pytest
from nilearn.glm.first_level import make_first_level_design_matrix

from fusi_tribe.fusi import timing as tm

TR = 1.6


@pytest.fixture
def paradigm():
    events = pd.DataFrame({
        "trial_type": ["Audio only", "Speech"] * 7,
        "onset": 8.0 + np.arange(14) * 28.0,
        "duration": [12, 17, 15, 15, 14, 15, 17, 15, 15, 17, 10, 13, 15, 19][:14],
    })
    frame_times = np.arange(254) * TR
    drift = make_first_level_design_matrix(frame_times, events=None, drift_model="cosine",
                                           high_pass=0.01).to_numpy()
    return events, frame_times, drift


def test_gamma_hrf_peak_and_width():
    t = np.arange(0, 40, 0.01)
    h = tm.gamma_hrf(6.0, 5.0, t)
    above = t[h >= 0.5]
    assert t[np.argmax(h)] == pytest.approx(6.0, abs=0.02)
    assert above[-1] - above[0] == pytest.approx(5.0, rel=0.1)


def test_gamma_hrf_no_overflow_for_narrow_late_response():
    h = tm.gamma_hrf(14.0, 2.0, np.arange(0, 40, 0.05))
    assert np.isfinite(h).all() and h.max() == pytest.approx(1.0)


@pytest.mark.parametrize("planted", [-2.0, 0.0, 3.0])
def test_shift_scan_recovers_planted_onset_shift(paradigm, planted):
    events, ft, drift = paradigm
    dm = make_first_level_design_matrix(ft, events.assign(onset=events.onset + planted),
                                        hrf_model="glover", drift_model=None)
    rng = np.random.default_rng(1)
    y = dm[["Speech", "Audio only"]].to_numpy() @ [2.0, 1.0] + rng.normal(0, 0.3, ft.size)
    shifts = np.round(np.arange(-6, 6.01, 0.2), 2)
    r2 = tm.shift_scan(y, events, ft, np.arange(ft.size), drift, shifts)
    assert shifts[np.argmax(r2)] == pytest.approx(planted, abs=0.4)


def test_gamma_grid_recovers_planted_peak(paradigm):
    events, ft, drift = paradigm
    t = np.arange(0, 40, tm.DT)
    X = tm.convolved_blocks(events, ft, tm.gamma_hrf(8.0, 6.0, t))
    y = X @ [2.0, 1.0] + np.random.default_rng(2).normal(0, 0.2, ft.size)
    grid = tm.gamma_grid(y, events, ft, np.arange(ft.size), drift)
    i, _ = np.unravel_index(np.argmax(grid), grid.shape)
    assert tm.PEAKS[i] == pytest.approx(8.0, abs=0.75)


def test_f_statistic_finds_responsive_voxels(paradigm):
    events, ft, drift = paradigm
    X = tm.fir_columns(events, ft, TR, 8)
    rng = np.random.default_rng(3)
    Y = rng.normal(size=(ft.size, 50))
    Y[:, :5] += X @ rng.uniform(0.5, 1.0, size=(8, 5))
    F = tm.f_statistic(Y, X, drift)
    assert set(np.argsort(F)[-5:]) == set(range(5))


def test_split_halves_alternates_within_condition(paradigm):
    events, *_ = paradigm
    half = tm.split_halves(events)
    for c in tm.CONDS:
        assert "".join(half[events.trial_type.to_numpy() == c]).startswith("ABAB")


def test_held_out_mask_drops_only_selection_half_windows(paradigm):
    events, ft, _ = paradigm
    half = tm.split_halves(events)
    keep = tm.held_out_mask(events, half, "A", ft, tail_s=12.0)
    in_a = np.zeros(ft.size, bool)
    for ev, h in zip(events.itertuples(), half):
        if h == "A":
            in_a |= (ft >= ev.onset) & (ft < ev.onset + ev.duration + 12.0)
    np.testing.assert_array_equal(keep, ~in_a)
    # Every held-out (B) block still contributes frames.
    for ev, h in zip(events.itertuples(), half):
        if h == "B":
            assert keep[(ft >= ev.onset) & (ft < ev.onset + ev.duration + 12.0)].any()
