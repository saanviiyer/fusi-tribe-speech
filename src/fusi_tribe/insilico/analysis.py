"""Analyse TRIBE v2 predictions for the simulated runs.

Per block and ROI, a model-free amplitude: mean predicted activity in the
block's response window minus the mean over silent baseline frames. Then

    retention = mean amplitude(control blocks) / mean amplitude(speech blocks)

1 means TRIBE treats the control like speech; 0 means it evokes nothing.
Paired contrasts (manipulation minus matched reference) are computed per
passage, since the same passages fill the control blocks in every run.
Bootstrap intervals resample passages: TRIBE is deterministic, so passage-to-
passage variability is the only noise, and claims are about this stimulus set.

Timing: TRIBE is trained with fMRI sampled 5 s after each stimulus second, so
prediction row t is fMRI at t + 5 s (`insilico.tribe_offset_s`).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from nilearn import datasets
from nilearn.glm.first_level import make_first_level_design_matrix

from ..config import Config
from ..fusi.timing import DT, FWHMS, PEAKS, centroid, gamma_grid, gamma_hrf

ROIS = {
    "HG": ["G_temp_sup-G_T_transv", "S_temporal_transverse"],   # primary auditory
    "PT": ["G_temp_sup-Plan_tempo"],
    "STG": ["G_temp_sup-Lateral"],
    "STS": ["S_temporal_sup"],
    "MTG": ["G_temporal_middle"],
    "IFG": ["G_front_inf-Opercular", "G_front_inf-Triangul"],
    "V1": ["S_calcarine", "Pole_occipital"],                     # control region
}
PAIRED = (("rotated", "lowpass"), ("reversed", "intact"), ("vocoded", "intact"))
PAIRED_ROIS = ("HG", "PT", "STG", "STS", "MTG")
WIN_START_S, WIN_END_S = 4.0, 3.0      # response window: onset + 4 s .. offset + 3 s
BASELINE_GAP_S = 8.0                   # baseline frames start this long after a block ends
MIN_SPEECH_AMP = 0.15                  # weaker speech responses: ratio not interpretable
STYLES = {"intact": ("#7f8c8d", "o"), "lowpass": ("#c0392b", "P"), "rotated": ("#8e44ad", "^"),
          "vocoded": ("#d35400", "s"), "reversed": ("#16a085", "D"), "silence": ("#2c3e50", "v")}


def roi_masks(atlas_dir: Path) -> dict[str, np.ndarray]:
    atlas = datasets.fetch_atlas_surf_destrieux(data_dir=str(atlas_dir), verbose=0)
    labels = [str(lab) for lab in atlas["labels"]]
    both = np.concatenate([atlas["map_left"], atlas["map_right"]])   # fsaverage5: left, right
    return {roi: np.isin(both, [labels.index(n) for n in names]) for roi, names in ROIS.items()}


def block_amplitudes(Y: np.ndarray, blocks: list[dict], t: np.ndarray) -> np.ndarray:
    """(n_blocks, n_signals): window mean minus shared silent baseline. t: stimulus clock (s)."""
    base = np.ones(t.size, bool)
    for b in blocks:
        base &= ~((t >= b["onset"]) & (t < b["onset"] + b["duration"] + BASELINE_GAP_S))
    if not base.any():
        raise ValueError("No baseline frames: blocks leave no silent gaps")
    win = [(t >= b["onset"] + WIN_START_S) & (t < b["onset"] + b["duration"] + WIN_END_S)
           for b in blocks]
    return np.array([Y[w].mean(axis=0) for w in win]) - Y[base].mean(axis=0)


def bootstrap_ci(stat, samples: list[np.ndarray], n_boot: int, rng) -> tuple[float, float]:
    boots = [stat(*[s[rng.integers(0, s.size, s.size)] for s in samples]) for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(lo), float(hi)


def load_predictions(path: Path, total_s: float) -> tuple[np.ndarray, np.ndarray]:
    d = np.load(path)
    keep = d["starts"] < total_s                     # TRIBE pads the last audio chunk
    return d["preds"][keep].astype(np.float64), d["starts"][keep]


def analyse_mode(cfg: Config, mode: str) -> dict:
    ic = cfg.insilico
    manifest = json.loads((cfg.output_dir / "stimuli" / "manifest.json").read_text())
    pred_dir = cfg.output_dir / "predictions" / mode
    res = cfg.results_dir / "insilico" / mode
    res.mkdir(parents=True, exist_ok=True)
    masks = roi_masks(cfg.output_dir / "atlas")
    rng = np.random.default_rng(0)

    rows, per_block, roi_ts_by_cond = [], {}, {}
    for cond in ic.conditions:
        path = pred_dir / f"{cond}.npz"
        if not path.exists():
            print(f"  {mode}: no predictions for {cond}, skipping")
            continue
        preds, starts = load_predictions(path, manifest["total_s"])
        t = starts + ic.tribe_offset_s
        blocks = manifest["conditions"][cond]["blocks"]
        speech = np.array([b["trial_type"] == "Speech" for b in blocks])
        roi_ts = np.column_stack([preds[:, m].mean(axis=1) for m in masks.values()])
        roi_ts_by_cond[cond] = (t, roi_ts, blocks)
        amp = block_amplitudes(roi_ts, blocks, t)
        per_block[cond] = amp[~speech] / amp[speech].mean(axis=0)
        for k, roi in enumerate(masks):
            sp, ct = amp[speech, k], amp[~speech, k]
            lo, hi = bootstrap_ci(lambda c, s: c.mean() / s.mean(), [ct, sp], ic.n_boot, rng)
            rows.append({"mode": mode, "condition": cond, "roi": roi,
                         "retention": float(ct.mean() / sp.mean()), "ci_lo": lo, "ci_hi": hi,
                         "amp_speech": float(sp.mean()), "amp_control": float(ct.mean()),
                         "reliable": bool(sp.mean() >= MIN_SPEECH_AMP)})
    retention = pd.DataFrame(rows)
    retention.to_csv(res / "retention.tsv", sep="\t", index=False)

    roi_names = list(masks)
    paired_rows = []
    for num, den in PAIRED:
        if num in per_block and den in per_block:
            for roi in PAIRED_ROIS:
                k = roi_names.index(roi)
                diff = per_block[num][:, k] - per_block[den][:, k]
                lo, hi = bootstrap_ci(np.mean, [diff], ic.n_boot, rng)
                paired_rows.append({"mode": mode, "contrast": f"{num} - {den}", "roi": roi,
                                    "diff": float(diff.mean()), "ci_lo": lo, "ci_hi": hi})
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(res / "paired_contrasts.tsv", sep="\t", index=False)

    timing = tribe_timing(roi_ts_by_cond, roi_names) if "silence" in roi_ts_by_cond else None
    if timing:
        (res / "tribe_timing.json").write_text(json.dumps(timing, indent=2))
    _figure(retention, paired, mode, res / "retention.png")
    return {"retention": retention, "paired": paired, "timing": timing}


def tribe_timing(roi_ts_by_cond: dict, roi_names: list[str], roi: str = "HG") -> dict:
    """Gamma fit to TRIBE's predicted response in `roi` (speech vs silence run),
    with the same model used for fUSI in fusi.timing."""
    t, ts, blocks = roi_ts_by_cond["silence"]
    events = pd.DataFrame(blocks)
    y = ts[:, roi_names.index(roi)]
    rows = np.arange(t.size)
    drift = make_first_level_design_matrix(t, events=None, drift_model="cosine",
                                           high_pass=0.01).to_numpy()
    grid = gamma_grid(y, events, t, rows, drift)
    i, j = np.unravel_index(np.argmax(grid), grid.shape)
    tf = np.arange(0, 40, DT)
    return {"roi": roi, "peak_s": float(PEAKS[i]), "fwhm_s": float(FWHMS[j]),
            "centroid_s": centroid(gamma_hrf(PEAKS[i], FWHMS[j], tf), tf), "r2": float(grid[i, j])}


def _figure(retention: pd.DataFrame, paired: pd.DataFrame, mode: str, out_png: Path) -> None:
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(14, 4.4), layout="constrained",
                                 gridspec_kw={"width_ratios": [1.25, 1]})
    # Plot an ROI only if its speech response is large enough in every condition.
    rois = [r for r in ROIS if retention[retention.roi == r].reliable.all()]
    x = np.arange(len(rois))
    conds = list(dict.fromkeys(retention.condition))
    for i, cond in enumerate(conds):
        sub = retention[retention.condition == cond].set_index("roi").reindex(rois)
        sub = sub.where(sub.reliable.astype(bool))
        off = (i - (len(conds) - 1) / 2) * 0.1
        color, marker = STYLES.get(cond, ("grey", "o"))
        ax.errorbar(x + off, sub.retention, yerr=[sub.retention - sub.ci_lo, sub.ci_hi - sub.retention],
                    color=color, marker=marker, lw=1.2, capsize=2, label=cond)
    ax.axhline(1, color="grey", lw=0.8, ls="--")
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xticks(x, rois)
    ax.set_ylim(-0.3, 2.0)
    ax.set(ylabel="Retention (control / speech response)",
           title=f"(a) TRIBE v2 [{mode}]: response to each control, relative to speech")
    ax.legend(frameon=False, ncol=3, fontsize=8, loc="upper left")

    xp = np.arange(len(PAIRED_ROIS))
    names = list(dict.fromkeys(paired.contrast)) if len(paired) else []
    for i, name in enumerate(names):
        sub = paired[paired.contrast == name].set_index("roi").reindex(list(PAIRED_ROIS))
        off = (i - (len(names) - 1) / 2) * 0.22
        bx.bar(xp + off, sub["diff"], width=0.2, color=STYLES[name.split(" - ")[0]][0], label=name)
        bx.errorbar(xp + off, sub["diff"], yerr=[sub["diff"] - sub.ci_lo, sub.ci_hi - sub["diff"]],
                    fmt="none", ecolor="k", capsize=2, lw=1)
    bx.axhline(0, color="k", lw=0.8)
    bx.set_xticks(xp, PAIRED_ROIS)
    bx.set(ylabel="Change in retention (paired over passages)",
           title="(b) Each manipulation vs. its matched reference")
    if names:
        bx.legend(frameon=False, fontsize=8)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
