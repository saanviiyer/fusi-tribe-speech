"""fUSI preprocessing: QC signals, censoring, rigid registration, smoothing.

Adapted from the lab-provided `fusiproc_utils3.py` (author not stated in the
source). Changes from the original:

* `register_volumes` passes voxel spacing to SimpleITK in its own axis order.
  `sitk.GetImageFromArray` reverses numpy's axes, so the original passed
  (x, y, z) spacing to a (z, y, x) image; harmless for isotropic voxels, wrong
  otherwise. Returned parameters are reordered to numpy (x, y, z) order.
* DVARS is computed directly in numpy instead of through nipype.
* Plotting is separated from computation.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import SimpleITK as sitk
from nilearn.signal import clean
from scipy.signal import savgol_filter

# ---------------------------------------------------------------- QC signals


def nan_mask(data: np.ndarray) -> np.ndarray:
    """Voxels finite in every frame."""
    return np.all(np.isfinite(data), axis=-1)


def global_signal(data: np.ndarray) -> np.ndarray:
    return np.nanmean(data, axis=(0, 1, 2))


def global_signal_outliers(sig: np.ndarray, t_r: float, k: float = 4.0) -> np.ndarray:
    """Frames whose detrended global signal exceeds median + k * MAD."""
    detr = clean(sig.reshape(-1, 1), detrend=True, standardize=False, t_r=t_r).ravel()
    med = np.median(detr)
    mad = np.median(np.abs(detr - med))
    return np.flatnonzero(detr > med + k * mad)


def framewise_displacement(rigid: np.ndarray, radius_mm: float = 50.0) -> np.ndarray:
    """Power et al. FD from (6, t) rigid parameters [tx, ty, tz, rx, ry, rz] (mm, rad)."""
    diff = np.diff(rigid.astype(float), axis=1)
    diff[3:6] *= radius_mm
    return np.insert(np.abs(diff).sum(axis=0), 0, 0.0)


def dvars(data: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """RMS over the mask of the frame-to-frame signal change, as % of the mean (t,)."""
    ts = data[mask]                                   # (V, t)
    mu = ts.mean(axis=1, keepdims=True)
    pct = 100 * ts / np.where(mu == 0, 1, mu)
    return np.insert(np.sqrt(np.mean(np.diff(pct, axis=1) ** 2, axis=0)), 0, np.nan)


# -------------------------------------------------------------- registration


@dataclass
class Registration:
    data: np.ndarray        # (x, y, z, n_kept) registered frames
    params: np.ndarray      # (6, n_kept): tx, ty, tz (mm) and rx, ry, rz (rad), numpy axis order
    metric: np.ndarray      # (n_kept,) final correlation metric
    kept: np.ndarray        # (n_kept,) original frame indices


def _to_sitk(vol: np.ndarray, spacing_xyz: np.ndarray) -> sitk.Image:
    img = sitk.GetImageFromArray(vol.astype(np.float32))   # sitk axes = numpy axes reversed
    img.SetSpacing(tuple(float(s) for s in spacing_xyz[::-1]))
    return img


def register_volumes(
    stack: np.ndarray,
    *,
    spacing: Sequence[float],
    skip: Sequence[int] = (),
    mask: np.ndarray | None = None,
    reference: str = "median",
    progress: bool = False,
) -> Registration:
    """Rigidly register every non-skipped frame to the mean or median volume."""
    if stack.ndim != 4:
        raise ValueError("stack must have shape (x, y, z, t)")
    n = stack.shape[-1]
    skip_set = set(int(i) for i in skip)
    kept = np.array([i for i in range(n) if i not in skip_set], dtype=int)
    if kept.size == 0:
        raise ValueError("No frames left after skipping")
    spacing_xyz = np.asarray(spacing, dtype=float)

    stack = np.where(np.isfinite(stack), stack, 0.0)
    agg = {"median": np.median, "mean": np.mean}[reference]
    fixed = _to_sitk(agg(stack[..., kept], axis=-1), spacing_xyz)

    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsCorrelation()
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetShrinkFactorsPerLevel([4, 2, 1])
    reg.SetSmoothingSigmasPerLevel([2, 1, 0])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    reg.SetOptimizerAsRegularStepGradientDescent(
        learningRate=0.1, minStep=1e-4, numberOfIterations=100, relaxationFactor=0.5)
    if mask is not None:
        m = _to_sitk(mask.astype(np.uint8), spacing_xyz)
        reg.SetMetricFixedMask(sitk.Cast(m, sitk.sitkUInt8))

    out = np.empty(stack.shape[:3] + (kept.size,), dtype=np.float32)
    params = np.empty((6, kept.size), dtype=np.float32)
    metric = np.empty(kept.size)
    for j, t in enumerate(kept):
        moving = _to_sitk(stack[..., t], spacing_xyz)
        init = sitk.CenteredTransformInitializer(
            fixed, moving, sitk.Euler3DTransform(),
            sitk.CenteredTransformInitializerFilter.MOMENTS)
        reg.SetInitialTransform(init, inPlace=True)
        reg.SetOptimizerScalesFromPhysicalShift()
        tx = reg.Execute(fixed, moving)
        metric[j] = reg.GetMetricValue()
        # sitk (x, y, z) == numpy (z, y, x): reverse to report in numpy order.
        params[:3, j] = np.asarray(tx.GetTranslation())[::-1]
        params[3:, j] = np.array([tx.GetAngleX(), tx.GetAngleY(), tx.GetAngleZ()])[::-1]
        resampled = sitk.Resample(moving, fixed, tx, sitk.sitkLinear, 0.0, moving.GetPixelID())
        out[..., j] = sitk.GetArrayFromImage(resampled)
        if progress and (j % 25 == 0 or j == kept.size - 1):
            print(f"  registered {j + 1}/{kept.size} frames", flush=True)
    return Registration(data=out, params=params, metric=metric, kept=kept)


def to_full_length(reg: Registration, n_frames: int, keep: np.ndarray):
    """Place registered frames listed in `keep` back on the full time axis (NaN elsewhere)."""
    sel = np.isin(reg.kept, keep)
    data = np.full(reg.data.shape[:3] + (n_frames,), np.nan, dtype=np.float32)
    data[..., reg.kept[sel]] = reg.data[..., sel]
    params = np.full((6, n_frames), np.nan, dtype=np.float32)
    params[:, reg.kept[sel]] = reg.params[:, sel]
    return data, params


# ------------------------------------------------------------------ smoothing


def boxcar_smooth(data: np.ndarray, window: int) -> np.ndarray:
    """Centered moving average over time, ignoring NaN frames. Even/short windows: no-op."""
    if window < 3 or window % 2 == 0:
        return data
    half = window // 2
    out = np.empty_like(data)
    n = data.shape[-1]
    for t in range(n):
        lo, hi = max(0, t - half), min(n, t + half + 1)
        out[..., t] = np.nanmean(data[..., lo:hi], axis=-1)
    return out


def savgol_smooth(data: np.ndarray, window: int, polyorder: int = 2) -> np.ndarray:
    """Savitzky-Golay over time; NaN frames are interpolated for the fit, then restored."""
    if window < 3 or window % 2 == 0 or polyorder >= window:
        raise ValueError("window must be odd, >= 3 and > polyorder")
    *spatial, n = data.shape
    flat = data.reshape(-1, n).copy()
    t = np.arange(n)
    for i, ts in enumerate(flat):
        ok = np.isfinite(ts)
        if ok.sum() < polyorder + 2:
            continue
        filled = np.interp(t, t[ok], ts[ok])
        y = savgol_filter(filled, window, polyorder, mode="interp")
        y[~ok] = np.nan
        flat[i] = y
    return flat.reshape(*spatial, n)


def fsl24(params: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Friston-24 motion regressors from (6, t) parameters; NaN frames set to 0."""
    m = np.nan_to_num(params.T)
    d = np.vstack([np.zeros((1, 6)), np.diff(m, axis=0)])
    names = ["tx", "ty", "tz", "rx", "ry", "rz"]
    cols = np.hstack([m, d, m ** 2, d ** 2])
    labels = names + [f"d{n}" for n in names] + [f"{n}2" for n in names] + [f"d{n}2" for n in names]
    return cols, labels
