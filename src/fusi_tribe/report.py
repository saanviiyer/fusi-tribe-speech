"""Cross-modality figures: response timing (fUSI vs TRIBE vs Glover), stimuli, and
TRIBE audio-only vs full (text) mode."""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
from nilearn.glm.first_level.hemodynamic_models import glover_hrf
from scipy.signal import spectrogram

from .config import Config
from .fusi.timing import DT, gamma_hrf


def timing_comparison(cfg: Config, mode: str = "audio_only") -> None:
    fusi = json.loads((cfg.results_dir / "fusi" / "timing_summary.json").read_text())
    tribe_path = cfg.results_dir / "insilico" / mode / "tribe_timing.json"
    t = np.arange(0, 30, DT)
    glover = glover_hrf(DT, oversampling=1, time_length=30.0)[: t.size]
    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    ax.plot(t, glover / glover.max(), "k--", lw=1.5,
            label=f"Glover HRF (fMRI convention): centroid {fusi['glover']['centroid_s']:.1f} s")
    if tribe_path.exists():
        tr = json.loads(tribe_path.read_text())
        ax.plot(t, gamma_hrf(tr["peak_s"], tr["fwhm_s"], t), color="#16a085", lw=2.5,
                label=f"TRIBE v2 prediction ({tr['roi']}): centroid {tr['centroid_s']:.1f} s")
    g = fusi["gamma_fit"]["A"]
    ax.plot(t, gamma_hrf(g["peak_s"], g["fwhm_s"], t), color="#c0392b", lw=2.5,
            label=f"fUSI measured: centroid {g['centroid_s']:.1f} ± {g['centroid_se_s']:.1f} s")
    ax.set(xlabel="Time after logged stimulus onset (s)", ylabel="Normalised impulse response",
           title="Same paradigm, three response models")
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(cfg.results_dir / "timing_comparison.png", dpi=200)
    plt.close(fig)


def stimulus_spectrograms(cfg: Config, show_s: float = 4.0) -> None:
    stim = cfg.output_dir / "stimuli"
    manifest = json.loads((stim / "manifest.json").read_text())
    conds = [c for c in ("intact", "lowpass", "rotated", "vocoded", "reversed")
             if c in manifest["conditions"]]
    block = next(b for b in manifest["conditions"][conds[0]]["blocks"] if b["trial_type"] != "Speech")
    fig, axes = plt.subplots(1, len(conds), figsize=(3.4 * len(conds), 3.2), sharey=True,
                             layout="constrained")
    for ax, cond in zip(np.atleast_1d(axes), conds):
        x, sr = sf.read(stim / manifest["conditions"][cond]["path"])
        i0 = int((block["onset"] + 2.0) * sr)
        f, tt, S = spectrogram(x[i0:i0 + int(show_s * sr)], sr, nperseg=512, noverlap=448)
        ax.pcolormesh(tt, f / 1000, 10 * np.log10(S + 1e-12), shading="auto", cmap="magma",
                      vmin=-110, vmax=-40)
        ax.set(title=cond, xlabel="Time (s)")
    np.atleast_1d(axes)[0].set_ylabel("Frequency (kHz)")
    (cfg.results_dir / "insilico").mkdir(parents=True, exist_ok=True)
    fig.savefig(cfg.results_dir / "insilico" / "stimulus_spectrograms.png", dpi=160)
    plt.close(fig)


def mode_comparison(cfg: Config) -> bool:
    """Retention per region in audio-only vs full mode, one panel per acoustic control.

    Returns False (and draws nothing) until both modes have been analysed.
    """
    paths = {m: cfg.results_dir / "insilico" / m / "retention.tsv" for m in ("audio_only", "full")}
    if not all(p.exists() for p in paths.values()):
        return False
    df = pd.concat([pd.read_csv(p, sep="\t") for p in paths.values()])
    # Only regions with a usable speech response in both modes (IFG has none).
    rois = [r for r in ["HG", "PT", "STG", "STS", "MTG", "IFG"] if df[df.roi == r].reliable.all()]
    conds = ["lowpass", "vocoded", "reversed", "rotated"]   # most to fewest words recovered
    words = {}
    tpath = cfg.results_dir / "insilico" / "transcription.tsv"
    if tpath.exists():
        t = pd.read_csv(tpath, sep="\t").set_index("condition")
        words = t["fraction_of_intact_words_recovered"].to_dict()
    styles = {"audio_only": ("#7f8c8d", "o", "audio pathway only"),
              "full": ("#14707c", "s", "audio + text (Llama 3.2)")}
    fig, axes = plt.subplots(1, len(conds), figsize=(4 * len(conds), 3.8), sharey=True,
                             layout="constrained")
    x = np.arange(len(rois))
    for ax, cond in zip(axes, conds):
        for k, (mode, (color, marker, label)) in enumerate(styles.items()):
            sub = df[(df["mode"] == mode) & (df.condition == cond)].set_index("roi").reindex(rois)
            sub = sub.where(sub.reliable.astype(bool))
            off = (k - 0.5) * 0.18
            ax.errorbar(x + off, sub.retention, yerr=[sub.retention - sub.ci_lo, sub.ci_hi - sub.retention],
                        color=color, marker=marker, lw=1.4, capsize=2, label=label)
        ax.axhline(1, color="grey", lw=0.8, ls="--")
        ax.axhline(0, color="grey", lw=0.8)
        ax.set_xticks(x, rois)
        ax.set_title(f"{cond}: {words[cond]:.0%} of words recovered" if cond in words else cond)
        ax.set_ylim(-0.3, 1.8)
    axes[0].set_ylabel("Retention (control / speech response)")
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("Adding the words: TRIBE v2 with the audio pathway only vs audio + text")
    fig.savefig(cfg.results_dir / "insilico" / "mode_comparison.png", dpi=200)
    plt.close(fig)
    return True
