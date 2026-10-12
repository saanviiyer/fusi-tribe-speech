"""When does the fUSI response happen, relative to the logged stimulus onsets?

The GLM assumes the Glover (BOLD fMRI) HRF and the original notebook shifted
every onset by -1 TR. Two data-driven checks instead:

1. Onset-shift scan: variance explained by Glover-convolved blocks as onsets
   are shifted from -6 s to +6 s.
2. Gamma fit: impulse response with free time-to-peak and width (grid search).

Both use an ROI picked by a timing-agnostic F-test on finite-impulse-response
(FIR) regressors from half of the blocks, and a leave-one-block-out jackknife
for uncertainty. A cross-validated shift scan then measures timing only on the
blocks that were *not* used to pick the voxels, in both directions. Peak latency and an event-clock offset are not separable from
logged onsets alone; everything is reported relative to the logged onsets.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.glm.first_level import make_first_level_design_matrix
from nilearn.glm.first_level.hemodynamic_models import glover_hrf
from scipy.ndimage import gaussian_filter

from ..config import Config

DT = 0.05                                       # high-resolution grid (s)
PEAKS = np.round(np.arange(1.0, 14.01, 0.25), 2)
FWHMS = np.round(np.arange(2.0, 16.01, 0.5), 2)
CONDS = ("Speech", "Audio only")


# ----------------------------------------------------------------- building blocks


def ols(X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    return beta, Y - X @ beta


def boxcar(onsets, durations, t: np.ndarray) -> np.ndarray:
    x = np.zeros_like(t)
    for o, d in zip(onsets, durations):
        x[(t >= o) & (t < o + d)] = 1.0
    return x


def gamma_hrf(peak_s: float, fwhm_s: float, t: np.ndarray) -> np.ndarray:
    """Unit-peak gamma impulse response with a given mode and (approximate) FWHM."""
    sd = fwhm_s / 2.3548
    k = 1 + (peak_s ** 2 + peak_s * np.sqrt(peak_s ** 2 + 4 * sd ** 2)) / (2 * sd ** 2)
    theta = peak_s / (k - 1)
    pos = t > 0
    log_h = np.full(t.shape, -np.inf)          # log space: t**(k-1) overflows for large k
    log_h[pos] = (k - 1) * np.log(t[pos]) - t[pos] / theta
    return np.exp(log_h - log_h[pos].max())


def convolved_blocks(events: pd.DataFrame, frame_times: np.ndarray, hrf: np.ndarray) -> np.ndarray:
    """One column per condition: boxcar * hrf, sampled at frame_times."""
    t_hr = np.arange(0, frame_times[-1] + 2.0, DT)
    cols = []
    for c in CONDS:
        ev = events[events.trial_type == c]
        x = np.convolve(boxcar(ev.onset, ev.duration, t_hr), hrf)[: t_hr.size] * DT
        cols.append(np.interp(frame_times, t_hr, x))
    return np.column_stack(cols)


def fir_columns(events: pd.DataFrame, frame_times: np.ndarray, tr: float, n_lags: int) -> np.ndarray:
    """Boxcar delayed by k*TR for k < n_lags (window-averaged to frame times)."""
    t_hr = np.arange(-tr, frame_times[-1] + tr, DT)
    cols = []
    for k in range(n_lags):
        x = boxcar(events.onset + k * tr, events.duration, t_hr)
        cols.append([x[(t_hr >= t - tr / 2) & (t_hr < t + tr / 2)].mean() for t in frame_times])
    return np.array(cols).T


def split_halves(events: pd.DataFrame) -> np.ndarray:
    """Alternate occurrences of each condition into halves 'A' and 'B'."""
    half = np.empty(len(events), dtype="<U1")
    for c in CONDS:
        idx = np.flatnonzero(events.trial_type.to_numpy() == c)
        half[idx[0::2]], half[idx[1::2]] = "A", "B"
    return half


def f_statistic(Y: np.ndarray, X_test: np.ndarray, X_other: np.ndarray) -> np.ndarray:
    """Voxelwise F for adding X_test to a model containing X_other."""
    _, r_full = ols(np.column_stack([X_test, X_other]), Y)
    _, r_red = ols(X_other, Y)
    df1, df2 = X_test.shape[1], Y.shape[0] - X_test.shape[1] - X_other.shape[1]
    rss_f = (r_full ** 2).sum(0)
    return ((r_red ** 2).sum(0) - rss_f) / df1 / (rss_f / df2)


def r2_gain(y: np.ndarray, X_task: np.ndarray, X_nuis: np.ndarray) -> float:
    _, res = ols(np.column_stack([X_task, X_nuis]), y)
    _, res0 = ols(X_nuis, y)
    return float(1 - (res ** 2).sum() / (res0 ** 2).sum())


def shift_scan(y, events, frame_times, rows, X_nuis, shifts, keep=None) -> np.ndarray:
    """R² of Glover-convolved blocks (beyond nuisance) for each onset shift."""
    keep = np.ones(rows.size, bool) if keep is None else keep
    out = []
    for s in shifts:
        ev = events.assign(onset=events.onset + s)
        dm = make_first_level_design_matrix(frame_times, ev, hrf_model="glover", drift_model=None)
        X = dm[list(CONDS)].to_numpy()[rows]
        out.append(r2_gain(y[keep], X[keep], X_nuis[keep]))
    return np.array(out)


def gamma_grid(y, events, frame_times, rows, X_nuis, keep=None, cache=None) -> np.ndarray:
    """R² over the (time-to-peak, width) grid."""
    keep = np.ones(rows.size, bool) if keep is None else keep
    cache = {} if cache is None else cache
    t = np.arange(0, 40, DT)
    grid = np.empty((PEAKS.size, FWHMS.size))
    for i, p in enumerate(PEAKS):
        for j, w in enumerate(FWHMS):
            if (p, w) not in cache:
                cache[(p, w)] = convolved_blocks(events, frame_times, gamma_hrf(p, w, t))[rows]
            grid[i, j] = r2_gain(y[keep], cache[(p, w)][keep], X_nuis[keep])
    return grid


def leave_block_out_masks(events, frame_times_rows, tail_s: float = 12.0) -> list[np.ndarray]:
    return [~((frame_times_rows >= ev.onset) & (frame_times_rows < ev.onset + ev.duration + tail_s))
            for ev in events.itertuples()]


def held_out_mask(events, half, selection_half: str, frame_times_rows, tail_s: float = 12.0) -> np.ndarray:
    """Frames outside every block window of `selection_half` (shared baseline kept)."""
    keep = np.ones(frame_times_rows.size, bool)
    for ev, h in zip(events.itertuples(), half):
        if h == selection_half:
            keep &= ~((frame_times_rows >= ev.onset) & (frame_times_rows < ev.onset + ev.duration + tail_s))
    return keep


def jackknife_se(x) -> float:
    x = np.asarray(x, float)
    return float(np.sqrt((x.size - 1) / x.size * ((x - x.mean()) ** 2).sum()))


def centroid(h: np.ndarray, t: np.ndarray) -> float:
    w = np.clip(h, 0, None)
    return float((t * w).sum() / w.sum())


# --------------------------------------------------------------------- pipeline


@dataclass
class RoiFit:
    roi: np.ndarray
    peak_s: float
    fwhm_s: float
    centroid_s: float
    r2: float
    peak_se_s: float
    centroid_se_s: float


def run_timing(cfg: Config) -> dict:
    tc = cfg.timing
    out = cfg.output_dir / cfg.run_name
    res = cfg.results_dir / "fusi"
    res.mkdir(parents=True, exist_ok=True)
    if not (out / "registered.nii.gz").exists():
        raise FileNotFoundError("Run `fusi-tribe fusi-glm` first")

    img = nib.load(out / "registered.nii.gz")
    data = img.get_fdata(dtype=np.float32)
    mask = nib.load(out / "mask.nii.gz").get_fdata() > 0
    rows = np.load(out / "keep_frames.npy")
    frame_times = np.load(out / "frame_times.npy")
    tr = float(np.median(np.diff(frame_times)))
    events = pd.read_csv(cfg.data_dir / "task_events.tsv", sep="\t").rename(
        columns={cfg.fusi.event_col: "trial_type"})
    motion = pd.read_csv(out / "confounds.tsv", sep="\t").to_numpy()[rows]
    drift = make_first_level_design_matrix(frame_times, events=None, drift_model="cosine",
                                           high_pass=cfg.fusi.high_pass_hz).to_numpy()[rows]
    X_nuis = np.column_stack([drift, motion])

    def pct(y):
        mu = y.mean(axis=0, keepdims=True)
        return 100 * (y - mu) / np.where(mu == 0, 1, mu)

    vol = data[..., rows]
    Y = pct(vol[mask].T)
    sigma = cfg.fusi.spatial_fwhm_mm / (2.3548 * float(np.sqrt((img.affine[:3, 0] ** 2).sum())))
    Ys = pct(np.stack([gaussian_filter(vol[..., i], sigma) for i in range(rows.size)], -1)[mask].T)
    del data, vol

    # FIR regressors per (condition, half): timing-agnostic ROI selection.
    half = split_halves(events)
    fir = {(c, h): fir_columns(events[(events.trial_type == c) & (half == h)], frame_times, tr,
                               tc.fir_lags)[rows] for c in CONDS for h in "AB"}
    shifts = np.round(np.arange(tc.shift_min_s, tc.shift_max_s + 1e-9, tc.shift_step_s), 2)
    ft_rows = frame_times[rows]
    lbo = leave_block_out_masks(events, ft_rows)
    t_fine = np.arange(0, 40, DT)
    cache: dict = {}

    fits: dict[str, RoiFit] = {}
    for h in "AB":
        other = "B" if h == "A" else "A"
        X_test = np.column_stack([fir[(c, h)] for c in CONDS])
        X_rest = np.column_stack([fir[(c, other)] for c in CONDS] + [X_nuis])
        roi = np.argsort(f_statistic(Ys, X_test, X_rest))[-tc.n_roi_voxels:]
        y = Y[:, roi].mean(axis=1)
        grid = gamma_grid(y, events, frame_times, rows, X_nuis, cache=cache)
        i, j = np.unravel_index(np.argmax(grid), grid.shape)
        jk = [np.unravel_index(np.argmax(gamma_grid(y, events, frame_times, rows, X_nuis, k, cache)),
                               grid.shape) for k in lbo]
        jk_peaks = [PEAKS[a] for a, _ in jk]
        jk_cent = [centroid(gamma_hrf(PEAKS[a], FWHMS[b], t_fine), t_fine) for a, b in jk]
        fits[h] = RoiFit(roi=roi, peak_s=float(PEAKS[i]), fwhm_s=float(FWHMS[j]),
                         centroid_s=centroid(gamma_hrf(PEAKS[i], FWHMS[j], t_fine), t_fine),
                         r2=float(grid[i, j]), peak_se_s=jackknife_se(jk_peaks),
                         centroid_se_s=jackknife_se(jk_cent))

    y_a = Y[:, fits["A"].roi].mean(axis=1)
    r2 = shift_scan(y_a, events, frame_times, rows, X_nuis, shifts)
    jk_shift = [shifts[np.argmax(shift_scan(y_a, events, frame_times, rows, X_nuis, shifts, k))]
                for k in lbo]
    glover = glover_hrf(DT, oversampling=1, time_length=40.0)[: t_fine.size]
    legacy = -tr

    # Cross-validated: pick voxels on one half, measure timing on the other half's blocks.
    cross = {}
    shared = np.intersect1d(fits["A"].roi, fits["B"].roi)
    for sel in "AB":
        keep = held_out_mask(events, half, sel, ft_rows)
        for name, voxels in ((f"roi_{sel}", fits[sel].roi), ("roi_shared", shared)):
            y = Y[:, voxels].mean(axis=1)
            r2_cv = shift_scan(y, events, frame_times, rows, X_nuis, shifts, keep)
            jk_cv = [shifts[np.argmax(shift_scan(y, events, frame_times, rows, X_nuis, shifts, keep & k))]
                     for k in lbo]
            cross[f"select_{sel}_measure_{'B' if sel == 'A' else 'A'}_{name}"] = {
                "n_voxels": int(voxels.size), "n_frames": int(keep.sum()),
                "best_shift_s": float(shifts[np.argmax(r2_cv)]),
                "best_shift_jackknife_se_s": jackknife_se(jk_cv),
                "r2_best": float(r2_cv.max()),
                "r2_no_shift": float(np.interp(0.0, shifts, r2_cv)),
                "r2_legacy_minus_one_tr": float(np.interp(legacy, shifts, r2_cv)),
            }

    summary = {
        "roi_voxels": tc.n_roi_voxels,
        "roi_overlap_A_B": float(np.intersect1d(fits["A"].roi, fits["B"].roi).size / tc.n_roi_voxels),
        "glover": {"peak_s": float(t_fine[np.argmax(glover)]), "centroid_s": centroid(glover, t_fine)},
        "gamma_fit": {h: {k: v for k, v in vars(f).items() if k != "roi"} for h, f in fits.items()},
        "shift_scan": {
            "best_shift_s": float(shifts[np.argmax(r2)]),
            "best_shift_jackknife_se_s": jackknife_se(jk_shift),
            "r2_best": float(r2.max()),
            "r2_no_shift": float(np.interp(0.0, shifts, r2)),
            "r2_legacy_minus_one_tr": float(np.interp(legacy, shifts, r2)),
        },
        "cross_validated_shift": cross,
    }
    (res / "timing_summary.json").write_text(json.dumps(summary, indent=2))

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    a1.plot(t_fine, glover / glover.max(), "k--", lw=1.5,
            label=f"Glover HRF (fMRI): centroid {summary['glover']['centroid_s']:.1f} s")
    for h, color in (("A", "#c0392b"), ("B", "#8e44ad")):
        f = fits[h]
        a1.plot(t_fine, gamma_hrf(f.peak_s, f.fwhm_s, t_fine), color=color, lw=2.5,
                label=f"fUSI ROI {h} (R²={f.r2:.2f}): centroid {f.centroid_s:.1f} ± {f.centroid_se_s:.1f} s")
    a1.set(xlim=(0, 30), xlabel="Time after logged onset (s)", ylabel="Normalised response",
           title="Best-fitting impulse response")
    a1.legend(frameon=False, fontsize=8)
    ss = summary["shift_scan"]
    a2.plot(shifts, r2, color="k", lw=2)
    a2.axvline(ss["best_shift_s"], color="#c0392b", lw=1.5,
               label=f"best {ss['best_shift_s']:+.1f} s (± {ss['best_shift_jackknife_se_s']:.1f})")
    a2.axvline(legacy, color="grey", ls="--", lw=1.5, label=f"original notebook: −1 TR ({legacy:.1f} s)")
    a2.axvline(0, color="grey", ls=":", lw=1, label="no shift")
    a2.set(xlabel="Onset shift (s)", ylabel="Variance explained beyond nuisance (R²)",
           title="Glover model fit vs. onset shift")
    a2.legend(frameon=False, fontsize=8)
    fig.savefig(res / "timing.png", dpi=200)
    plt.close(fig)
    return summary
