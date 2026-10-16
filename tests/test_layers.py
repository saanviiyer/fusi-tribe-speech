import numpy as np
import pytest

from fusi_tribe.insilico.layers import linear_cka


def test_cka_is_one_for_rotated_and_scaled_copies():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 20))
    Q, _ = np.linalg.qr(rng.normal(size=(20, 20)))
    assert linear_cka(X, 3.0 * X @ Q) == pytest.approx(1.0, abs=1e-9)


def test_cka_is_near_zero_for_independent_data():
    rng = np.random.default_rng(1)
    assert linear_cka(rng.normal(size=(5000, 10)), rng.normal(size=(5000, 10))) < 0.02
