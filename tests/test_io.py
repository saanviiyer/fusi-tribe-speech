import numpy as np
import pandas as pd
import pytest

from fusi_tribe.io import frame_times_from_logs, load_events


def test_frame_times_sorted_zeroed_and_tr_is_median():
    ns = np.array([3.2e9, 0.0, 1.6e9, 4.8e9, 6.5e9]).astype(np.int64) + 10**12
    ft, tr = frame_times_from_logs(pd.DataFrame({"time_ns": ns}))
    np.testing.assert_allclose(ft, [0, 1.6, 3.2, 4.8, 6.5])
    assert tr == pytest.approx(1.6)


def test_frame_times_requires_time_ns():
    with pytest.raises(ValueError, match="time_ns"):
        frame_times_from_logs(pd.DataFrame({"t": [0, 1]}))


def test_load_events_checks_columns(tmp_path):
    p = tmp_path / "events.tsv"
    p.write_text("onset\ttrial_type\n0\tSpeech\n")
    with pytest.raises(ValueError, match="duration"):
        load_events(p)
