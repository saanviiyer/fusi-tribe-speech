import numpy as np
import pytest
from scipy.ndimage import shift as nd_shift

from fusi_tribe.fusi import preprocess as pp


def test_framewise_displacement_scales_rotations_by_radius():
    rigid = np.zeros((6, 3))
    rigid[0, 1] = 0.2                  # 0.2 mm translation at frame 1
    rigid[3, 2] = 0.01                 # 0.01 rad rotation at frame 2
    fd = pp.framewise_displacement(rigid, radius_mm=50)
    np.testing.assert_allclose(fd, [0.0, 0.2, 0.2 + 0.5])


def test_boxcar_ignores_nan_frames_and_noops_on_even_window():
    x = np.array([1.0, np.nan, 3.0, 5.0]).reshape(1, 1, 1, 4)
    np.testing.assert_allclose(pp.boxcar_smooth(x, 3).ravel(), [1.0, 2.0, 4.0, 4.0])
    assert pp.boxcar_smooth(x, 4) is x


def test_fsl24_columns():
    cols, names = pp.fsl24(np.random.default_rng(0).normal(size=(6, 10)))
    assert cols.shape == (10, 24) and len(names) == 24
    np.testing.assert_allclose(cols[:, 12:18], cols[:, :6] ** 2)


def _blob(shape, center, widths):
    grids = np.meshgrid(*[np.arange(n) for n in shape], indexing="ij")
    return np.exp(-sum(((g - c) / w) ** 2 for g, c, w in zip(grids, center, widths)))


def test_registration_recovers_translation_with_anisotropic_voxels():
    """A 1 mm shift along numpy axis 0 must come back as tx = 1 mm, not tz.

    Voxels are 0.5 x 0.5 x 1.0 mm, so mixing up axis order (the bug fixed in
    preprocess.register_volumes) would report the wrong axis and magnitude.
    """
    shape, spacing = (32, 28, 16), np.array([0.5, 0.5, 1.0])
    ref = _blob(shape, (15, 13, 8), (4, 6, 3)) + 0.6 * _blob(shape, (21, 9, 5), (3, 3, 2))
    moved = nd_shift(ref, (2.0, 0, 0), order=1)          # 2 voxels * 0.5 mm = 1 mm
    stack = np.stack([ref] * 5 + [moved], axis=-1).astype(np.float32)
    reg = pp.register_volumes(stack, spacing=spacing)
    tx, ty, tz = reg.params[:3, -1]
    assert abs(tx) == pytest.approx(1.0, abs=0.15)
    assert abs(ty) < 0.15 and abs(tz) < 0.15
    # Registered frame should match the reference better than the moved one did.
    err_before = np.abs(moved - ref).mean()
    err_after = np.abs(reg.data[..., -1] - ref).mean()
    assert err_after < 0.5 * err_before


def test_to_full_length_places_frames_and_nans():
    reg = pp.Registration(data=np.ones((2, 2, 2, 3), np.float32), params=np.ones((6, 3), np.float32),
                          metric=np.zeros(3), kept=np.array([0, 2, 3]))
    data, params = pp.to_full_length(reg, n_frames=5, keep=np.array([0, 3]))
    assert np.isnan(data[..., [1, 2, 4]]).all() and (data[..., [0, 3]] == 1).all()
    assert np.isnan(params[:, 2]).all()
