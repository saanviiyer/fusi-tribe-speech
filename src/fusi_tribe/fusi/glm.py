"""First-level GLM for one fUSI run (replaces the original analysis notebook).

QC -> censoring -> rigid registration (cached) -> FD censoring -> temporal
smoothing -> Friston-24 confounds -> nilearn FirstLevelModel with sample masks
-> per-condition and pairwise z maps.
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import image, plotting
from nilearn.glm.first_level import FirstLevelModel, make_first_level_design_matrix

from ..config import Config
from ..io import Run, load_run, save_nifti
from . import preprocess as pp


def sanitize(label: str) -> str:
    out = "".join(c if c.isalnum() else "_" for c in str(label).strip().lower())
    return "_".join(p for p in out.split("_") if p) or "contrast"


def condition_contrasts(design: pd.DataFrame, conditions: list[str]) -> dict[str, np.ndarray]:
    eye = np.eye(design.shape[1])
    idx = {c: i for i, c in enumerate(design.columns)}
    conds = [c for c in conditions if c in idx]
    out = {sanitize(c): eye[idx[c]] for c in conds}
    for a, b in combinations(conds, 2):
        out[f"{sanitize(a)}_minus_{sanitize(b)}"] = eye[idx[a]] - eye[idx[b]]
        out[f"{sanitize(b)}_minus_{sanitize(a)}"] = eye[idx[b]] - eye[idx[a]]
    return out


def _registration(run: Run, skip: list[int], mask: np.ndarray, cache: Path) -> pp.Registration:
    if cache.exists():
        z = np.load(cache)
        if np.array_equal(z["skip"], np.array(sorted(skip))):
            print(f"Using cached registration: {cache}")
            return pp.Registration(data=z["data"], params=z["params"],
                                   metric=z["metric"], kept=z["kept"])
    print(f"Registering {run.n_frames - len(skip)} frames (several minutes)...")
    reg = pp.register_volumes(run.data, spacing=run.voxel_size_mm, skip=skip, mask=mask,
                              progress=True)
    np.savez(cache, data=reg.data, params=reg.params, metric=reg.metric, kept=reg.kept,
             skip=np.array(sorted(skip)))
    return reg


def _qc_figure(frame_times, gs, gs_bad, fd, fd_thresh, dv, params, out_png: Path) -> None:
    fig, ax = plt.subplots(4, 1, figsize=(10, 9), sharex=True, layout="constrained")
    ax[0].plot(frame_times, gs, color="k", lw=1)
    for i in gs_bad:
        ax[0].axvline(frame_times[i], color="#c0392b", alpha=0.4, lw=1)
    ax[0].set(ylabel="Global signal", title="Quality control (red: censored by global-signal spike test)")
    ax[1].plot(frame_times, params[:3].T, lw=1)
    ax[1].legend(["x", "y", "z"], frameon=False, ncol=3, fontsize=8)
    ax[1].set(ylabel="Translation (mm)")
    ax[2].plot(frame_times, fd, color="#2e86c1", lw=1)
    ax[2].axhline(fd_thresh, color="grey", ls="--", lw=1)
    ax[2].set(ylabel="FD (mm)")
    ax[3].plot(frame_times, dv, color="#7d3c98", lw=1)
    ax[3].set(ylabel="DVARS (%)", xlabel="Time (s)")
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def run_glm(cfg: Config) -> dict:
    f = cfg.fusi
    out = cfg.output_dir / cfg.run_name
    res = cfg.results_dir / "fusi"
    out.mkdir(parents=True, exist_ok=True)
    res.mkdir(parents=True, exist_ok=True)

    run = load_run(cfg.data_dir, f.event_col)
    n = run.n_frames
    mask = pp.nan_mask(run.data)
    save_nifti(mask.astype(np.uint8), run.affine, out / "mask.nii.gz")

    # 1. Censoring before registration: initial frames + global-signal spikes.
    gs = pp.global_signal(run.data)
    skip = set(range(min(f.initial_censor_frames, n)))
    pre_kept = np.array([i for i in range(n) if i not in skip])
    gs_bad = pre_kept[pp.global_signal_outliers(gs[pre_kept], run.tr, f.global_signal_k)]
    skip |= {int(i) for i in gs_bad}

    # 2. Rigid registration (cached), then FD censoring.
    reg = _registration(run, sorted(skip), mask, out / "registration.npz")
    fd_kept = pp.framewise_displacement(reg.params, f.fd_radius_mm)
    fd = np.full(n, np.nan)
    fd[reg.kept] = fd_kept
    fd_bad = np.flatnonzero(fd > f.fd_thresh_mm)
    keep = np.array([i for i in reg.kept if i not in set(fd_bad)], dtype=int)
    data_full, params_full = pp.to_full_length(reg, n, keep)

    dv = np.full(n, np.nan)
    dv[reg.kept] = pp.dvars(reg.data, mask)
    _qc_figure(run.frame_times, gs, gs_bad, fd, f.fd_thresh_mm, dv, params_full, res / "qc.png")

    # 3. Temporal smoothing and confounds.
    if f.smooth_method == "boxcar":
        smoothed = pp.boxcar_smooth(data_full, f.smooth_window)
    elif f.smooth_method == "savgol":
        smoothed = pp.savgol_smooth(data_full, f.smooth_window)
    elif f.smooth_method == "none":
        smoothed = data_full
    else:
        raise ValueError(f"Unknown smooth_method {f.smooth_method!r}")
    motion, motion_names = pp.fsl24(params_full)
    pd.DataFrame(motion, columns=motion_names).to_csv(out / "confounds.tsv", sep="\t", index=False)
    save_nifti(data_full, run.affine, out / "registered.nii.gz")   # unsmoothed, for timing analyses
    np.save(out / "keep_frames.npy", keep)
    np.save(out / "frame_times.npy", run.frame_times)

    # 4. GLM.
    events = run.events.copy()
    events["onset"] = events["onset"] + f.onset_shift_s
    design = make_first_level_design_matrix(
        run.frame_times, events.rename(columns={f.event_col: "trial_type"}),
        hrf_model=f.hrf_model, drift_model=f.drift_model, high_pass=f.high_pass_hz,
        add_regs=motion, add_reg_names=motion_names)
    clean_img = nib.Nifti1Image(np.nan_to_num(smoothed).astype(np.float32), run.affine)
    glm = FirstLevelModel(t_r=run.tr, noise_model=f.noise_model, standardize=True,
                          smoothing_fwhm=f.spatial_fwhm_mm,
                          mask_img=nib.Nifti1Image(mask.astype(np.uint8), run.affine))
    glm.fit(clean_img, design_matrices=[design], sample_masks=[keep])
    design.to_csv(out / "design_matrix.tsv", sep="\t", index=False)

    conditions = [str(c) for c in pd.unique(run.events[f.event_col])]
    mean_img = image.mean_img(clean_img, copy_header=True)
    vox_mm3 = float(np.prod(run.voxel_size_mm))
    contrasts = {}
    for name, vec in condition_contrasts(glm.design_matrices_[0], conditions).items():
        z = glm.compute_contrast(vec, output_type="z_score")
        z.to_filename(out / f"{name}_z.nii.gz")
        zd = z.get_fdata()
        contrasts[name] = {"peak_z": float(np.nanmax(zd)),
                           "volume_mm3_above_threshold": float((zd > f.z_threshold).sum() * vox_mm3)}
        if "_minus_" in name and not name.startswith("speech"):
            continue  # skip the mirror-image plot; its map is still saved
        disp = plotting.plot_stat_map(
            z, bg_img=mean_img, threshold=f.z_threshold, display_mode="ortho", black_bg=True,
            cut_coords=plotting.find_xyz_cut_coords(z, activation_threshold=f.z_threshold),
            title=name.replace("_", " "))
        disp.savefig(res / f"{name}_z.png", dpi=200)
        disp.close()

    summary = {
        "run": cfg.run_name,
        "n_frames": n,
        "tr_s": run.tr,
        "censored_global_signal": [int(i) for i in sorted(gs_bad)],
        "censored_fd": [int(i) for i in fd_bad],
        "n_frames_in_glm": int(keep.size),
        "max_fd_mm": float(np.nanmax(fd)),
        "onset_shift_s": f.onset_shift_s,
        "z_threshold": f.z_threshold,
        "contrasts": contrasts,
    }
    (res / "glm_summary.json").write_text(json.dumps(summary, indent=2))
    return summary
