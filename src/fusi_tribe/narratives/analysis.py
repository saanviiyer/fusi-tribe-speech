"""Held-out test of TRIBE v2 on real brains: "It's Not the Fall That Gets You" (Narratives).

Narratives (Nastase et al. 2021, OpenNeuro ds002345, CC0) is not in TRIBE's
training set. Participants heard this story intact and, in the same session,
one temporally scrambled version: sentence-level segments reordered coarsely
(longscram) or finely (shortscram). Word order within sentences is preserved.

For each ROI along the auditory-language hierarchy and each version:

* Human reliability: leave-one-out inter-subject correlation (ISC) of the ROI
  time course, computed within each group so intact vs scrambled compares the
  same people.
* TRIBE accuracy: correlation between TRIBE's prediction from the stimulus
  audio and the group-mean ROI time course, and that correlation divided by a
  split-half (Spearman-Brown) noise ceiling of the group mean.

Data: preprocessed afni-nosmooth surface data (fsaverage6), reduced to
fsaverage5 by keeping the first 10,242 vertices per hemisphere (the fsaverage
meshes are nested). TR = 1.5 s. TRIBE prediction row t is fMRI at t + 5 s; one
extra lag, shared by all versions, is fitted on HG in the intact condition.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..config import Config

TR = 1.5
N_FS5 = 10242
VERSIONS = ("notthefallintact", "notthefallshortscram", "notthefalllongscram")
SHORT = {"notthefallintact": "intact", "notthefallshortscram": "shortscram", "notthefalllongscram": "longscram"}
GROUPS = {"short": "notthefallshortscram", "long": "notthefalllongscram"}
EXTRA_ROIS = {"AG": ["G_pariet_inf-Angular"], "PCun": ["G_precuneus"]}   # higher-order, scrambling-sensitive
MAX_LAG_TR = 6
N_SPLITS = 200


# --------------------------------------------------------------------- metrics


def zscore(x: np.ndarray, axis: int = 0) -> np.ndarray:
    return (x - x.mean(axis=axis, keepdims=True)) / (x.std(axis=axis, keepdims=True) + 1e-12)


def loo_isc(ts: np.ndarray) -> np.ndarray:
    """Leave-one-out ISC. ts: (subjects, time). Returns one r per subject."""
    z = zscore(ts, axis=1)
    out = np.empty(len(z))
    for i in range(len(z)):
        others = np.delete(z, i, axis=0).mean(axis=0)
        out[i] = np.corrcoef(z[i], others)[0, 1]
    return out


def split_half_ceiling(ts: np.ndarray, rng: np.random.Generator, n_splits: int = N_SPLITS) -> float:
    """Reliability of the full-group mean: mean split-half r, Spearman-Brown corrected."""
    n = len(ts)
    rs = []
    for _ in range(n_splits):
        perm = rng.permutation(n)
        a, b = ts[perm[: n // 2]].mean(0), ts[perm[n // 2:]].mean(0)
        rs.append(np.corrcoef(a, b)[0, 1])
    r = float(np.mean(rs))
    return 2 * r / (1 + r) if r > -1 else float("nan")


def best_lag(pred: np.ndarray, target: np.ndarray, max_lag: int = MAX_LAG_TR) -> int:
    """Shift (in samples) of `pred` that best correlates with `target` (positive = pred later)."""
    best, best_r = 0, -np.inf
    for lag in range(-max_lag, max_lag + 1):
        p = np.roll(pred, lag)
        sl = slice(max(lag, 0), len(p) + min(lag, 0))
        r = np.corrcoef(p[sl], target[sl])[0, 1]
        if r > best_r:
            best, best_r = lag, r
    return best


# ------------------------------------------------------------------------ data


def load_subject(path_l: Path, path_r: Path) -> np.ndarray:
    """(time, 20484) fsaverage5 time series from fsaverage6 left/right GIfTI files."""
    import nibabel as nib

    parts = []
    for p in (path_l, path_r):
        data = np.asarray(nib.load(p).agg_data())
        if data.shape[0] != 40962:          # agg_data gives (vertices, time) for func.gii
            data = data.T
        parts.append(data[:N_FS5].T)
    return np.concatenate(parts, axis=1).astype(np.float32)


def subjects_by_group(participants_tsv: Path, exclude: set[str]) -> dict[str, list[str]]:
    df = pd.read_csv(participants_tsv, sep="\t")
    out = {}
    for g, task in GROUPS.items():
        out[g] = sorted(s for s, t in zip(df.participant_id, df.task)
                        if s not in exclude and "notthefallintact" in t and task in t)
    return out


def tribe_timecourses(pred_path: Path, n_tr: int, masks: dict, offset_s: float) -> dict[str, np.ndarray]:
    d = np.load(pred_path)
    t = d["starts"] + offset_s                       # when each predicted sample is measured
    frame_times = np.arange(n_tr) * TR
    out = {}
    for roi, m in masks.items():
        out[roi] = np.interp(frame_times, t, d["preds"][:, m].mean(axis=1))
    return out


# --------------------------------------------------------------------- driver


def prepare_stimuli(cfg: Config, narr_dir: Path) -> Path:
    """Copy the three story recordings into this config's stimuli folder with a manifest,
    so `run-tribe --mode full` can predict them."""
    import shutil

    import soundfile as sf

    from ..insilico.stimuli import md5

    stim = cfg.output_dir / "stimuli"
    stim.mkdir(parents=True, exist_ok=True)
    manifest = {"conditions": {}}
    for v in VERSIONS:
        src = narr_dir / "stimuli" / f"{v}_audio.wav"
        dst = stim / f"{v}.wav"
        shutil.copyfile(src, dst)
        manifest["conditions"][v] = {"path": dst.name, "md5": md5(dst), "blocks": []}
    manifest["total_s"] = max(sf.info(stim / f"{v}.wav").duration for v in VERSIONS)
    (stim / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"prepared {len(VERSIONS)} stimuli, {manifest['total_s']:.1f} s each at most")
    return stim


def run_narratives(cfg: Config, narr_dir: Path, pred_dir: Path) -> pd.DataFrame:
    from ..insilico.analysis import ROIS, roi_masks
    from nilearn import datasets

    atlas_dir = cfg.output_dir / "atlas"
    masks = roi_masks(atlas_dir)
    atlas = datasets.fetch_atlas_surf_destrieux(data_dir=str(atlas_dir), verbose=0)
    labels = [str(lab) for lab in atlas["labels"]]
    both = np.concatenate([atlas["map_left"], atlas["map_right"]])
    for roi, names in EXTRA_ROIS.items():
        masks[roi] = np.isin(both, [labels.index(n) for n in names])
    masks.pop("V1", None)
    roi_names = list(masks)

    exclude = {"sub-066", "sub-317", "sub-333", "sub-335"}
    groups = subjects_by_group(narr_dir / "participants.tsv", exclude)
    func = narr_dir / "afni-nosmooth"

    # Per version: subjects x time x ROI
    data: dict[tuple[str, str], np.ndarray] = {}
    for g, task in GROUPS.items():
        for version in ("notthefallintact", task):
            stack = []
            for s in groups[g]:
                base = func / s / "func" / f"{s}_task-{version}_space-fsaverage6_hemi-{{}}_desc-clean.func.gii"
                X = load_subject(Path(str(base).format("L")), Path(str(base).format("R")))
                stack.append(np.column_stack([X[:, masks[r]].mean(axis=1) for r in roi_names]))
            n_tr = min(len(x) for x in stack)
            data[(g, version)] = np.stack([x[:n_tr] for x in stack])
            print(f"{g:5s} {SHORT[version]:10s} {len(stack)} subjects x {n_tr} TRs", flush=True)

    # Lag: fit once on HG, intact, all intact listeners pooled.
    intact_all = np.concatenate([data[(g, "notthefallintact")] for g in GROUPS])
    n_tr = intact_all.shape[1]
    hg = roi_names.index("HG")
    pred_intact = tribe_timecourses(pred_dir / "notthefallintact.npz", n_tr, masks, cfg.insilico.tribe_offset_s)
    lag = best_lag(zscore(pred_intact["HG"]), zscore(intact_all[:, :, hg].mean(0)))
    print(f"extra lag fitted on HG (intact): {lag} TR ({lag * TR:+.1f} s)", flush=True)

    rng = np.random.default_rng(0)
    rows = []
    for g, task in GROUPS.items():
        for version in ("notthefallintact", task):
            X = data[(g, version)]
            pred = tribe_timecourses(pred_dir / f"{version}.npz", X.shape[1], masks, cfg.insilico.tribe_offset_s)
            sl = slice(max(lag, 0), X.shape[1] + min(lag, 0))
            for k, roi in enumerate(roi_names):
                ts = X[:, :, k]
                isc = loo_isc(ts)
                p = np.roll(zscore(pred[roi]), lag)[sl]
                group_mean = ts.mean(0)[sl]
                r = float(np.corrcoef(p, group_mean)[0, 1])
                ceil = split_half_ceiling(ts[:, sl], rng)
                rows.append({"group": g, "version": SHORT[version], "roi": roi, "n_subjects": len(ts),
                             "isc_mean": float(isc.mean()), "isc_sem": float(isc.std(ddof=1) / np.sqrt(len(isc))),
                             "tribe_r": r, "noise_ceiling": ceil,
                             "tribe_r_normalised": r / ceil if ceil > 0 else float("nan"),
                             "lag_tr": lag})

    out = pd.DataFrame(rows)
    res = cfg.results_dir / "narratives"
    res.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cfg.output_dir / "narratives_roi_timeseries.npz", roi_names=np.array(roi_names), lag=lag,
                        **{f"{g}__{v}": arr for (g, v), arr in data.items()})
    preds = {v: tribe_timecourses(pred_dir / f"{v}.npz", n_tr, masks, cfg.insilico.tribe_offset_s) for v in VERSIONS}
    tribe_change = bootstrap_tribe_change(data, preds, roi_names, lag, rng)
    tribe_change.round(4).to_csv(res / "tribe_vs_human_change.tsv", sep="\t", index=False)
    plot_comparison(tribe_change, roi_names, res / "tribe_vs_human.png")
    out.round(4).to_csv(res / "narratives_summary.tsv", sep="\t", index=False)
    paired = paired_isc_change(data, roi_names, rng)
    paired.round(4).to_csv(res / "isc_change.tsv", sep="\t", index=False)
    plot_narratives(out, paired, roi_names, res / "narratives.png")
    return out


def subject_correlations(ts: np.ndarray, pred: np.ndarray, lag: int) -> tuple[np.ndarray, np.ndarray]:
    """Per person: r(TRIBE prediction, person) and r(mean of everyone else, person)."""
    sl = slice(max(lag, 0), ts.shape[1] + min(lag, 0))
    p = np.roll(zscore(pred), lag)[sl]
    z = zscore(ts[:, sl], axis=1)
    r_tribe = np.array([np.corrcoef(p, zi)[0, 1] for zi in z])
    r_others = np.array([np.corrcoef(np.delete(z, i, axis=0).mean(0), z[i])[0, 1] for i in range(len(z))])
    return r_tribe, r_others


def bootstrap_tribe_change(data: dict, preds: dict, roi_names: list[str], lag: int,
                           rng: np.random.Generator, n_boot: int = 5000) -> pd.DataFrame:
    """Scrambled minus intact, paired within people, on one scale (correlation with each person):
    delta_tribe for TRIBE's prediction, delta_others for the average of the other listeners.
    Bootstrap over people. Per-person ratios are avoided because r_others can be near 0."""
    rows = []
    for g, task in GROUPS.items():
        A, B = data[(g, "notthefallintact")], data[(g, task)]
        for k, roi in enumerate(roi_names):
            ta, oa = subject_correlations(A[:, :, k], preds["notthefallintact"][roi], lag)
            tb, ob = subject_correlations(B[:, :, k], preds[task][roi], lag)
            dt, do = tb - ta, ob - oa
            idx = rng.integers(0, len(dt), (n_boot, len(dt)))
            ci = lambda d: np.percentile(d[idx].mean(axis=1), [2.5, 97.5])
            (tlo, thi), (olo, ohi), (glo, ghi) = ci(dt), ci(do), ci(dt - do)
            rows.append({"group": g, "roi": roi,
                         "r_tribe_intact": float(ta.mean()), "r_tribe_scrambled": float(tb.mean()),
                         "delta_tribe": float(dt.mean()), "delta_tribe_lo": float(tlo), "delta_tribe_hi": float(thi),
                         "r_others_intact": float(oa.mean()), "r_others_scrambled": float(ob.mean()),
                         "delta_others": float(do.mean()), "delta_others_lo": float(olo), "delta_others_hi": float(ohi),
                         "gap": float((dt - do).mean()), "gap_lo": float(glo), "gap_hi": float(ghi)})
    return pd.DataFrame(rows)


def tribe_change_from_saved(cfg: Config) -> pd.DataFrame:
    """Recompute the TRIBE vs human comparison from the saved ROI time courses (no GIfTI reload)."""
    from ..insilico.analysis import roi_masks
    from nilearn import datasets

    z = np.load(cfg.output_dir / "narratives_roi_timeseries.npz")
    roi_names = [str(r) for r in z["roi_names"]]
    lag = int(z["lag"])
    data = {tuple(k.split("__")): z[k] for k in z.files if "__" in k}
    masks = roi_masks(cfg.output_dir / "atlas")
    atlas = datasets.fetch_atlas_surf_destrieux(data_dir=str(cfg.output_dir / "atlas"), verbose=0)
    labels = [str(lab) for lab in atlas["labels"]]
    both = np.concatenate([atlas["map_left"], atlas["map_right"]])
    for roi, names in EXTRA_ROIS.items():
        masks[roi] = np.isin(both, [labels.index(n) for n in names])
    masks = {r: masks[r] for r in roi_names}
    n_tr = next(iter(data.values())).shape[1]
    pred_dir = cfg.output_dir / "predictions" / "full"
    preds = {v: tribe_timecourses(pred_dir / f"{v}.npz", n_tr, masks, cfg.insilico.tribe_offset_s) for v in VERSIONS}
    out = bootstrap_tribe_change(data, preds, roi_names, lag, np.random.default_rng(0))
    res = cfg.results_dir / "narratives"
    res.mkdir(parents=True, exist_ok=True)
    out.round(4).to_csv(res / "tribe_vs_human_change.tsv", sep="\t", index=False)
    plot_comparison(out, roi_names, res / "tribe_vs_human.png")
    return out


def plot_comparison(df: pd.DataFrame, rois: list[str], out_png: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), layout="constrained", sharey=True)
    x = np.arange(len(rois))
    for ax, (g, sub) in zip(axes, df.groupby("group", sort=False)):
        sub = sub.set_index("roi").loc[rois]
        for off, col, color, label in ((-0.17, "delta_others", "#c0392b", "other listeners (human ceiling)"),
                                       (0.17, "delta_tribe", "#14707c", "TRIBE v2 prediction")):
            ax.bar(x + off, sub[col], 0.32, color=color, label=label)
            ax.errorbar(x + off, sub[col], yerr=[sub[col] - sub[f"{col}_lo"], sub[f"{col}_hi"] - sub[col]],
                        fmt="none", ecolor="k", capsize=2, lw=1)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(x, rois)
        ax.set_title(f"{g}-scrambled minus intact (same people)")
    axes[0].set_ylabel("Change in correlation with each listener")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Scrambling the story: how well each listener is predicted, by other people vs by TRIBE")
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def paired_isc_change(data: dict, roi_names: list[str], rng: np.random.Generator, n_boot: int = 5000) -> pd.DataFrame:
    """Scrambled minus intact ISC, paired within the same people, bootstrap over people."""
    rows = []
    for g, task in GROUPS.items():
        A, B = data[(g, "notthefallintact")], data[(g, task)]
        for k, roi in enumerate(roi_names):
            d = loo_isc(B[:, :, k]) - loo_isc(A[:, :, k])
            boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)]
            lo, hi = np.percentile(boots, [2.5, 97.5])
            rows.append({"group": g, "roi": roi, "isc_change": float(d.mean()), "ci_lo": float(lo), "ci_hi": float(hi)})
    return pd.DataFrame(rows)


def plot_narratives(summary: pd.DataFrame, paired: pd.DataFrame, rois: list[str], out_png: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), layout="constrained")
    x = np.arange(len(rois))
    colors = {"short": "#8e44ad", "long": "#16a085"}
    for j, (g, sub) in enumerate(paired.groupby("group", sort=False)):
        sub = sub.set_index("roi").loc[rois]
        off = (j - 0.5) * 0.3
        axes[0].bar(x + off, sub.isc_change, 0.28, color=colors[g], label=f"{g}-scrambled minus intact")
        axes[0].errorbar(x + off, sub.isc_change, yerr=[sub.isc_change - sub.ci_lo, sub.ci_hi - sub.isc_change],
                         fmt="none", ecolor="k", capsize=2, lw=1)
    axes[0].axhline(0, color="k", lw=0.8)
    axes[0].set_xticks(x, rois)
    axes[0].set(ylabel="Change in inter-subject correlation",
                title="Real brains: scrambling the story, same people")
    axes[0].legend(frameon=False, fontsize=8)
    for j, version in enumerate(("intact", "shortscram", "longscram")):
        sub = summary[summary.version == version].groupby("roi").tribe_r_normalised.mean().reindex(rois)
        axes[1].plot(x, sub, marker="o", lw=2, label=version)
    axes[1].axhline(0, color="k", lw=0.8)
    axes[1].set_xticks(x, rois)
    axes[1].set(ylabel="TRIBE r / noise ceiling",
                title="TRIBE v2 prediction accuracy (held-out dataset)")
    axes[1].legend(frameon=False, fontsize=8)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
