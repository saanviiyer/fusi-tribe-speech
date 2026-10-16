"""Where do acoustic and linguistic information separate inside TRIBE's encoders?

TRIBE v2 feeds layer groups at 50%, 75% and 100% depth of two open models into
its brain model: Wav2Vec-BERT 2.0 (audio) and Llama-3.2-3B (text). For every
control condition, this compares each layer's representation of a control
passage with the same passage spoken intact:

* Audio: frame-level linear CKA between the condition's and the intact
  passage's hidden states (reversed speech is flipped back into register),
  pooled over the seven control passages. Floor: CKA between two *different*
  intact passages.
* Text: each block's words, embedded with the preceding words of its run as
  context (as TRIBE does), mean-pooled per block; centred cosine similarity to
  the intact block. Floor: similarity between different intact passages.

Needs a GPU for reasonable runtime. Writes results/insilico/layers_{audio,text}.tsv
and layers.png.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf

from ..config import Config

AUDIO_MODEL = "facebook/w2v-bert-2.0"
TEXT_MODEL = "meta-llama/Llama-3.2-3B"
AUDIO_CONDS = ("lowpass", "vocoded", "rotated", "reversed")
TEXT_CONDS = ("lowpass", "vocoded", "rotated", "reversed", "swap_intact_shuffled")
TRIBE_DEPTHS = (0.5, 0.75, 1.0)
MAX_CONTEXT_WORDS = 1024


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear CKA between two (n_samples, features) matrices with matched rows."""
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)
    xy = np.linalg.norm(Y.T @ X) ** 2
    return float(xy / (np.linalg.norm(X.T @ X) * np.linalg.norm(Y.T @ Y)))


def _control_blocks(manifest: dict, cond: str) -> list[dict]:
    return [b for b in manifest["conditions"][cond]["blocks"] if b["trial_type"] != "Speech"]


# ----------------------------------------------------------------------- audio


def audio_layers(cfg: Config, device: str) -> pd.DataFrame:
    import torch
    from transformers import AutoFeatureExtractor, Wav2Vec2BertModel

    stim = cfg.output_dir / "stimuli"
    manifest = json.loads((stim / "manifest.json").read_text())
    fe = AutoFeatureExtractor.from_pretrained(AUDIO_MODEL)
    model = Wav2Vec2BertModel.from_pretrained(AUDIO_MODEL).to(device).eval()

    def states(cond: str) -> list[list[np.ndarray]]:
        """Per control passage: list over layers of (frames, dims)."""
        x, sr = sf.read(stim / manifest["conditions"][cond]["path"], dtype="float32")
        out = []
        for b in _control_blocks(manifest, cond):
            seg = x[int(b["onset"] * sr): int((b["onset"] + b["duration"]) * sr)]
            inp = fe(seg, sampling_rate=sr, return_tensors="pt").to(device)
            with torch.no_grad():
                hs = model(**inp, output_hidden_states=True).hidden_states
            out.append([h[0].float().cpu().numpy() for h in hs])
        return out

    ref = states("intact")
    n_layers = len(ref[0])
    rows = []

    def pooled_cka(a: list[list[np.ndarray]], b: list[list[np.ndarray]], flip: bool) -> list[float]:
        res = []
        for layer in range(n_layers):
            xs, ys = [], []
            for pa, pb in zip(a, b):
                fa, fb = pa[layer], pb[layer]
                if flip:
                    fa = fa[::-1]
                n = min(len(fa), len(fb))
                xs.append(fa[:n]); ys.append(fb[:n])
            res.append(linear_cka(np.concatenate(xs), np.concatenate(ys)))
        return res

    floor = pooled_cka(ref[1:] + ref[:1], ref, flip=False)   # different passages
    for layer, v in enumerate(floor):
        rows.append({"condition": "different passage (floor)", "layer": layer, "depth": layer / (n_layers - 1), "cka": v})
    for cond in AUDIO_CONDS:
        for layer, v in enumerate(pooled_cka(states(cond), ref, flip=(cond == "reversed"))):
            rows.append({"condition": cond, "layer": layer, "depth": layer / (n_layers - 1), "cka": v})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------ text


def text_layers(cfg: Config, device: str) -> pd.DataFrame:
    import torch
    from transformers import AutoModel, AutoTokenizer

    stim = cfg.output_dir / "stimuli"
    manifest = json.loads((stim / "manifest.json").read_text())
    tok = AutoTokenizer.from_pretrained(TEXT_MODEL)
    model = AutoModel.from_pretrained(TEXT_MODEL).to(device).eval()

    def block_vectors(cond: str) -> np.ndarray:
        """(n_blocks, n_layers, dims): mean hidden state of each control block's words, in context."""
        words = pd.read_csv((stim / manifest["conditions"][cond]["path"]).with_suffix(".tsv"), sep="\t")
        words = words.sort_values("start").reset_index(drop=True)
        vecs = []
        for b in _control_blocks(manifest, cond):
            inside = ((words.start >= b["onset"]) & (words.start < b["onset"] + b["duration"])).to_numpy()
            if not inside.any():
                vecs.append(None)
                continue
            last = int(np.flatnonzero(inside).max())
            first_ctx = max(0, last + 1 - MAX_CONTEXT_WORDS)
            ctx = [str(w).strip() for w in words.text.iloc[first_ctx:last + 1]]
            block_mask = inside[first_ctx:last + 1]
            text, spans, pos = "", [], 0
            for w in ctx:
                start = len(text) + (1 if text else 0)
                text = f"{text} {w}" if text else w
                spans.append((start, len(text)))
            enc = tok(text, return_offsets_mapping=True, return_tensors="pt", add_special_tokens=False)
            offsets = enc.pop("offset_mapping")[0].numpy()
            in_block = np.zeros(len(offsets), bool)
            for (s, e), m in zip(spans, block_mask):
                if m:
                    in_block |= (offsets[:, 0] < e) & (offsets[:, 1] > s)
            with torch.no_grad():
                hs = model(**enc.to(device), output_hidden_states=True).hidden_states
            vecs.append(np.stack([h[0, in_block].float().mean(0).cpu().numpy() for h in hs]))
        return vecs

    conds = ("intact",) + TEXT_CONDS
    V = {c: block_vectors(c) for c in conds}
    n_layers = next(v for v in V["intact"] if v is not None).shape[0]
    all_vecs = np.stack([v for c in conds for v in V[c] if v is not None])    # (N, L, D)
    mean = all_vecs.mean(axis=0)                                              # remove the common direction

    def sim(a, b, layer):
        x, y = a[layer] - mean[layer], b[layer] - mean[layer]
        return float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y)))

    rows = []
    ref = V["intact"]
    n = len(ref)
    for layer in range(n_layers):
        depth = layer / (n_layers - 1)
        floor = [sim(ref[i], ref[j], layer) for i in range(n) for j in range(n) if i != j]
        rows.append({"condition": "different passage (floor)", "layer": layer, "depth": depth,
                     "similarity": float(np.mean(floor)), "n_blocks": n})
        for cond in TEXT_CONDS:
            pairs = [(v, r) for v, r in zip(V[cond], ref) if v is not None and r is not None]
            rows.append({"condition": cond, "layer": layer, "depth": depth,
                         "similarity": float(np.mean([sim(v, r, layer) for v, r in pairs])),
                         "n_blocks": len(pairs)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------- driver


def run_layers(cfg: Config, device: str | None = None) -> None:
    import torch

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    res = cfg.results_dir / "insilico"
    res.mkdir(parents=True, exist_ok=True)
    audio = audio_layers(cfg, device)
    audio.round(4).to_csv(res / "layers_audio.tsv", sep="\t", index=False)
    print("audio done", flush=True)
    text = text_layers(cfg, device)
    text.round(4).to_csv(res / "layers_text.tsv", sep="\t", index=False)
    print("text done", flush=True)
    plot_layers(audio, text, res / "layers.png")


def plot_layers(audio: pd.DataFrame, text: pd.DataFrame, out_png: Path) -> None:
    styles = {"lowpass": "#c0392b", "vocoded": "#d35400", "rotated": "#8e44ad", "reversed": "#16a085",
              "swap_intact_shuffled": "#2c3e50", "different passage (floor)": "#95a5a6"}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), layout="constrained")
    for ax, df, col, title in ((axes[0], audio, "cka", "Wav2Vec-BERT 2.0 (audio): CKA with intact passage"),
                               (axes[1], text, "similarity", "Llama 3.2-3B (text): similarity to intact passage")):
        for d in TRIBE_DEPTHS:
            ax.axvline(d, color="#bdc3c7", lw=6, alpha=0.4, zorder=0)
        for cond, sub in df.groupby("condition", sort=False):
            ls = "--" if "floor" in cond else "-"
            label = cond.replace("swap_intact_shuffled", "shuffled word order")
            ax.plot(sub.depth, sub[col], ls, color=styles.get(cond, "k"), lw=2, label=label)
        ax.set(xlabel="Relative layer depth (bands: layers TRIBE uses)", ylabel=col, title=title, xlim=(0, 1))
        ax.legend(frameon=False, fontsize=8)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
