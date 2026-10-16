"""Causal swap tests: pair each run's audio with edited words in the control blocks.

Full-mode TRIBE reads the transcript cached next to each audio file
(`<wav>.tsv`). A swap copies a condition's audio and writes a transcript that
differs from the one TRIBE used for that condition only inside the control
blocks. Speech blocks keep their own words, so each swap is compared, passage
by passage, with its reference run in full mode.

    swap                       audio      control-block words
    swap_vocoded_nowords       vocoded    none
    swap_rotated_truewords     rotated    the true words (from the intact run)
    swap_reversed_truewords    reversed   the true words
    swap_intact_nowords        intact     none
    swap_intact_shuffled       intact     the same words in shuffled order
    swap_silence_truewords     silence    the true words

The control blocks hold the same passages in every condition, so the intact
run's transcript gives the true words at the right times.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..config import Config

SWAPS = {
    "swap_vocoded_nowords": ("vocoded", "none"),
    "swap_rotated_truewords": ("rotated", "true"),
    "swap_reversed_truewords": ("reversed", "true"),
    "swap_intact_nowords": ("intact", "none"),
    "swap_intact_shuffled": ("intact", "shuffled"),
    "swap_silence_truewords": ("silence", "true"),
}
SEED = 0


def in_control_blocks(starts: pd.Series, blocks: list[dict]) -> np.ndarray:
    mask = np.zeros(len(starts), bool)
    for b in blocks:
        if b["trial_type"] != "Speech":
            mask |= ((starts >= b["onset"]) & (starts < b["onset"] + b["duration"])).to_numpy()
    return mask


def edit_transcript(own: pd.DataFrame, true_words: pd.DataFrame, blocks: list[dict], words: str,
                    rng: np.random.Generator | None = None) -> pd.DataFrame:
    """Replace the control-block words of `own`; speech-block words are untouched.

    words: "none" removes them, "true" inserts the control-block words of
    `true_words`, "shuffled" permutes the control-block words of `own` within
    each block (timestamps kept; each block becomes one sentence of the shuffled words).
    """
    rng = rng or np.random.default_rng(SEED)
    keep = own.loc[~in_control_blocks(own.start, blocks)].copy()
    next_id = int(own.sequence_id.max()) + 1 if len(own) else 0
    if words == "none":
        added = own.iloc[0:0]
    elif words == "true":
        added = true_words.loc[in_control_blocks(true_words.start, blocks)].copy()
        added["sequence_id"] = added["sequence_id"] - added["sequence_id"].min() + next_id
    elif words == "shuffled":
        parts = []
        for b in blocks:
            if b["trial_type"] == "Speech":
                continue
            blk = own.loc[(own.start >= b["onset"]) & (own.start < b["onset"] + b["duration"])].copy()
            if not len(blk):
                continue
            blk["text"] = rng.permutation(blk.text.to_numpy())
            blk["sequence_id"] = next_id
            blk["sentence"] = " " + " ".join(blk.text)
            next_id += 1
            parts.append(blk)
        added = pd.concat(parts) if parts else own.iloc[0:0]
    else:
        raise ValueError(f"unknown words mode {words!r}")
    return pd.concat([keep, added]).sort_values("start").reset_index(drop=True)


def make_swaps(cfg: Config) -> list[str]:
    """Write swap audio copies, transcripts and manifest entries next to the stimuli.

    Needs the transcripts TRIBE cached in outputs/stimuli/<condition>.tsv during
    the full-mode runs, so the swaps change only the words under test.
    """
    from .stimuli import md5

    stim = cfg.output_dir / "stimuli"
    manifest = json.loads((stim / "manifest.json").read_text())
    true_words = pd.read_csv(stim / "intact.tsv", sep="\t")
    rng = np.random.default_rng(SEED)
    made = []
    for name, (source, words) in SWAPS.items():
        src_info = manifest["conditions"][source]
        own_path = stim / f"{source}.tsv"
        if not own_path.exists():
            raise FileNotFoundError(f"{own_path} missing: run full mode for {source} first")
        own = pd.read_csv(own_path, sep="\t")
        edited = edit_transcript(own, true_words, src_info["blocks"], words, rng)
        wav = stim / f"{name}.wav"
        shutil.copyfile(stim / src_info["path"], wav)
        edited.to_csv(stim / f"{name}.tsv", sep="\t", index=False)
        manifest["conditions"][name] = {"path": wav.name, "md5": md5(wav), "blocks": src_info["blocks"],
                                        "swap": {"audio": source, "control_words": words}}
        n_ctrl = int(in_control_blocks(edited.start, src_info["blocks"]).sum())
        print(f"{name}: {source} audio, control-block words = {words} ({n_ctrl} words)")
        made.append(name)
    (stim / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return made


def analyse_swaps(cfg: Config) -> pd.DataFrame:
    """Paired change in retention, swap minus its reference run (both full mode)."""
    from .analysis import PAIRED_ROIS, block_amplitudes, bootstrap_ci, load_predictions, roi_masks

    manifest = json.loads((cfg.output_dir / "stimuli" / "manifest.json").read_text())
    pred_dir = cfg.output_dir / "predictions" / "full"
    ref = pd.read_csv(cfg.results_dir / "insilico" / "full" / "per_block_retention.tsv", sep="\t")
    masks = roi_masks(cfg.output_dir / "atlas")
    rng = np.random.default_rng(0)
    rows = []
    for name, (source, words) in SWAPS.items():
        path = pred_dir / f"{name}.npz"
        if not path.exists():
            print(f"  no predictions for {name}, skipping")
            continue
        preds, starts = load_predictions(path, manifest["total_s"])
        blocks = manifest["conditions"][name]["blocks"]
        speech = np.array([b["trial_type"] == "Speech" for b in blocks])
        roi_names = list(masks)
        ts = np.column_stack([preds[:, masks[r]].mean(axis=1) for r in roi_names])
        amp = block_amplitudes(ts, blocks, starts + cfg.insilico.tribe_offset_s)
        swap_ret = amp[~speech] / amp[speech].mean(axis=0)
        for roi in PAIRED_ROIS:
            k = roi_names.index(roi)
            base = ref[(ref.condition == source) & (ref.roi == roi)].sort_values("passage_index").retention.to_numpy()
            d = swap_ret[:, k] - base
            lo, hi = bootstrap_ci(np.mean, [d], cfg.insilico.n_boot, rng)
            rows.append({"swap": name, "audio": source, "control_words": words, "roi": roi,
                         "retention_reference": float(base.mean()), "retention_swap": float(swap_ret[:, k].mean()),
                         "change": float(d.mean()), "ci_lo": lo, "ci_hi": hi})
    out = pd.DataFrame(rows).round(3)
    res = cfg.results_dir / "insilico"
    out.to_csv(res / "swaps.tsv", sep="\t", index=False)
    if len(out):
        _figure(out, res / "swaps.png")
    return out


def _figure(df: pd.DataFrame, out_png: Path) -> None:
    swaps = list(dict.fromkeys(df.swap))
    rois = list(dict.fromkeys(df.roi))
    fig, ax = plt.subplots(figsize=(12, 4.2), layout="constrained")
    x = np.arange(len(swaps))
    width = 0.8 / len(rois)
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(rois)))
    for j, roi in enumerate(rois):
        sub = df[df.roi == roi].set_index("swap").reindex(swaps)
        pos = x + (j - (len(rois) - 1) / 2) * width
        sig = (sub.ci_lo > 0) | (sub.ci_hi < 0)
        ax.bar(pos, sub["change"], width * 0.9, color=colors[j], alpha=np.where(sig, 1.0, 0.35), label=roi)
        ax.errorbar(pos, sub["change"], yerr=[sub["change"] - sub.ci_lo, sub.ci_hi - sub["change"]],
                    fmt="none", ecolor="k", lw=0.8, capsize=1.5)
    ax.axhline(0, color="k", lw=0.8)
    labels = [f"{s.removeprefix('swap_').replace('_', ' ')}" for s in swaps]
    ax.set_xticks(x, labels)
    ax.set(ylabel="Change in retention vs reference run",
           title="Causal swaps: same audio, edited words in the control blocks (TRIBE v2, full mode)")
    ax.legend(frameon=False, ncol=len(rois), fontsize=8)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
