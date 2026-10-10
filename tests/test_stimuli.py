"""The acoustic controls must do what their names claim."""

import numpy as np
import pandas as pd
import pytest
from scipy.signal import hilbert

from fusi_tribe.insilico import stimuli as st

SR = 16000


def tone(freq, dur=1.0):
    t = np.arange(int(dur * SR)) / SR
    return np.sin(2 * np.pi * freq * t)


def peak_freq(x):
    spec = np.abs(np.fft.rfft(x * np.hanning(x.size)))
    return np.fft.rfftfreq(x.size, 1 / SR)[np.argmax(spec)]


def band_power(x, lo, hi):
    spec = np.abs(np.fft.rfft(x)) ** 2
    f = np.fft.rfftfreq(x.size, 1 / SR)
    return spec[(f >= lo) & (f < hi)].sum() / spec.sum()


@pytest.mark.parametrize("f_in", [500.0, 1200.0, 3000.0])
def test_rotation_mirrors_frequency_around_2khz(f_in):
    assert peak_freq(st.spectrally_rotate(tone(f_in), SR)) == pytest.approx(4000 - f_in, abs=10)


def test_lowpass_and_rotation_share_the_same_band():
    mix = tone(1000) + tone(6000)
    for y in (st.lowpass(mix, SR), st.spectrally_rotate(mix, SR)):
        assert band_power(y, 4200, 8000) < 1e-3


def test_vocoder_keeps_envelope_but_not_harmonics():
    t = np.arange(SR * 2) / SR
    env = 0.5 * (1 + np.sin(2 * np.pi * 3 * t))                   # 3 Hz "syllable" rhythm
    harmonic = sum(np.sin(2 * np.pi * 150 * k * t) for k in range(1, 15))
    x = env * harmonic
    y = st.noise_vocode(x, SR)
    env_y = np.abs(hilbert(y))
    smooth = np.convolve(env_y, np.ones(800) / 800, mode="same")
    assert np.corrcoef(smooth[2000:-2000], env[2000:-2000])[0, 1] > 0.8
    # Harmonic peaks are gone: energy at 150 Hz multiples no longer dominates.
    assert peak_freq(y) % 150 > 5 or band_power(y, 145, 155) < 0.05


def test_reversed_is_time_reversal():
    x = np.arange(10.0)
    np.testing.assert_array_equal(st.manipulate(x, "reversed", SR), x[::-1])


def test_fit_to_duration_trims_fades_and_rejects_short_audio():
    y = st.fit_to_duration(np.ones(SR * 3), 2.0, SR)
    assert y.size == 2 * SR and y[0] == 0 and y[-1] == 0 and y[SR] == 1
    with pytest.raises(ValueError, match="too short"):
        st.fit_to_duration(np.ones(SR), 2.0, SR)


def test_assign_passages_keeps_speech_and_control_sets_disjoint():
    ev = pd.DataFrame({"trial_type": ["Audio only", "Speech", "Speech", "Audio only"]})
    m = st.assign_passages(ev)
    assert {m[1], m[2]} == {0, 1} and {m[0], m[3]} == {2, 3}


def test_unknown_condition_raises():
    with pytest.raises(ValueError):
        st.manipulate(np.zeros(4), "pitch_shift", SR)
