"""In-silico replicas of the fUSI experiment: same block timing, controlled audio.

"Speech" blocks always hold intact synthetic speech. The condition decides what
fills the "Audio only" blocks:

    intact    intact speech (other passages)  null check: retention ~ 1
    lowpass   intact speech low-passed at ~4 kHz   band-matched reference for `rotated`
    rotated   spectrally rotated speech (Blesser 1972)   unintelligible, same band as `lowpass`
    vocoded   4-channel noise-vocoded speech (Shannon et al. 1995)   envelope kept, fine structure gone
    reversed  time-reversed speech             same long-term spectrum, unintelligible
    silence   nothing                          retention ~ 0

The Audio-only blocks use one fixed set of passages in every condition, so
conditions differ only in the manipulation (comparisons are paired over passages).

Reproducibility: macOS `say` output is not bit-identical across runs, so the
synthesised passages are cached in outputs/stimuli/passages/ on first use and
reused afterwards (`force=True` re-synthesises). Checksums of every stimulus
are written to the manifest; `run-tribe` records which ones it saw.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import butter, hilbert, resample_poly, sosfiltfilt

from ..config import Config
from .passages import PASSAGES

TARGET_RMS = 0.05
FADE_S = 0.05
SEED = 0


# ----------------------------------------------------------------- manipulations


def _bandpass(lo: float, hi: float, sr: int):
    return butter(4, [lo, hi], btype="band", fs=sr, output="sos")


def _rotation_lowpass(sr: int, pivot: float = 2000.0):
    return butter(8, 2 * pivot * 0.98, btype="low", fs=sr, output="sos")


def lowpass(x: np.ndarray, sr: int) -> np.ndarray:
    """Same band limit as `spectrally_rotate`, applied the same way (twice)."""
    lp = _rotation_lowpass(sr)
    return sosfiltfilt(lp, sosfiltfilt(lp, x))


def spectrally_rotate(x: np.ndarray, sr: int, pivot: float = 2000.0) -> np.ndarray:
    """Invert the spectrum around `pivot` Hz: f -> 2*pivot - f, within 0..2*pivot."""
    lp = _rotation_lowpass(sr, pivot)
    t = np.arange(x.size) / sr
    return sosfiltfilt(lp, sosfiltfilt(lp, x) * np.cos(2 * np.pi * 2 * pivot * t))


def noise_vocode(x: np.ndarray, sr: int, n_channels: int = 4, lo: float = 70.0,
                 hi: float = 7000.0, env_cutoff: float = 30.0, seed: int = SEED) -> np.ndarray:
    """Band envelopes (log-spaced channels) modulate band-limited noise."""
    rng = np.random.default_rng(seed)
    edges = np.geomspace(lo, hi, n_channels + 1)
    env_lp = butter(4, env_cutoff, btype="low", fs=sr, output="sos")
    noise = rng.standard_normal(x.size)
    out = np.zeros_like(x)
    for f0, f1 in zip(edges[:-1], edges[1:]):
        band = _bandpass(f0, f1, sr)
        env = np.clip(sosfiltfilt(env_lp, np.abs(hilbert(sosfiltfilt(band, x)))), 0, None)
        out += env * sosfiltfilt(band, noise)
    return out


def manipulate(x: np.ndarray, condition: str, sr: int) -> np.ndarray:
    ops = {
        "intact": lambda: x,
        "lowpass": lambda: lowpass(x, sr),
        "rotated": lambda: spectrally_rotate(x, sr),
        "vocoded": lambda: noise_vocode(x, sr),
        "reversed": lambda: x[::-1].copy(),
        "silence": lambda: np.zeros_like(x),
    }
    if condition not in ops:
        raise ValueError(f"Unknown condition {condition!r}")
    return ops[condition]()


# --------------------------------------------------------------------- helpers


def tts(text: str, voice: str, rate_wpm: int, sr: int) -> np.ndarray:
    """Synthesise speech with macOS `say` and resample to `sr`."""
    with tempfile.TemporaryDirectory() as tmp:
        aiff = Path(tmp) / "out.aiff"
        subprocess.run(["say", "-v", voice, "-r", str(rate_wpm), "-o", str(aiff), text], check=True)
        audio, native_sr = sf.read(aiff, dtype="float64")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if native_sr != sr:
        g = np.gcd(int(native_sr), sr)
        audio = resample_poly(audio, sr // g, int(native_sr) // g)
    return audio


def fit_to_duration(x: np.ndarray, dur_s: float, sr: int) -> np.ndarray:
    n = int(round(dur_s * sr))
    if x.size < n:
        raise ValueError(f"passage too short: {x.size / sr:.1f} s < {dur_s:.1f} s")
    y = x[:n].copy()
    ramp = np.linspace(0, 1, int(FADE_S * sr))
    y[: ramp.size] *= ramp
    y[-ramp.size:] *= ramp[::-1]
    return y


def set_rms(x: np.ndarray, target: float = TARGET_RMS) -> np.ndarray:
    rms = np.sqrt(np.mean(x ** 2))
    return x if rms == 0 else x * (target / rms)


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def cached_passages(ids, out_dir: Path, voice: str, rate_wpm: int, sr: int,
                    force: bool = False) -> dict[int, np.ndarray]:
    cache = out_dir / "passages"
    cache.mkdir(parents=True, exist_ok=True)
    audio = {}
    for p in sorted(ids):
        path = cache / f"passage_{p:02d}.wav"
        if force or not path.exists():
            sf.write(path, tts(PASSAGES[p], voice, rate_wpm, sr), sr, subtype="FLOAT")
        audio[p], file_sr = sf.read(path, dtype="float64")
        if file_sr != sr:
            raise ValueError(f"{path} is {file_sr} Hz, config says {sr} Hz; use force=True")
    return audio


def assign_passages(events: pd.DataFrame) -> dict[int, int]:
    """Speech blocks get passages 0..n_speech-1; Audio-only blocks get the rest."""
    speech = events.index[events.trial_type == "Speech"].tolist()
    other = events.index[events.trial_type != "Speech"].tolist()
    if len(speech) + len(other) > len(PASSAGES):
        raise ValueError(f"{len(events)} blocks but only {len(PASSAGES)} passages")
    return {r: i for i, r in enumerate(speech)} | {r: len(speech) + i for i, r in enumerate(other)}


# ------------------------------------------------------------------------ main


def make_stimuli(cfg: Config, only: list[str] | None = None, force: bool = False) -> Path:
    ic = cfg.insilico
    sr = ic.sample_rate
    out_dir = cfg.output_dir / "stimuli"
    out_dir.mkdir(parents=True, exist_ok=True)
    events = pd.read_csv(cfg.data_dir / "task_events.tsv", sep="\t").rename(
        columns={cfg.fusi.event_col: "trial_type"})
    passage_for_row = assign_passages(events)
    manifest_path = out_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"conditions": {}}
    todo = [c for c in (only or ic.conditions)
            if force or not (out_dir / f"{c}.wav").exists() or c not in manifest["conditions"]]
    raw = (cached_passages(set(passage_for_row.values()), out_dir, ic.voice, ic.rate_wpm, sr, force)
           if todo else {})
    total_s = float((events.onset + events.duration).max()) + ic.tail_s

    manifest |= {"sr": sr, "voice": ic.voice, "rate_wpm": ic.rate_wpm, "total_s": total_s}

    for cond in only or ic.conditions:
        if (out_dir / f"{cond}.wav").exists() and cond in manifest["conditions"] and not force:
            print(f"{cond}.wav exists, keeping it (use --force to rebuild)")
            continue
        track = np.zeros(int(round(total_s * sr)))
        blocks = []
        for r, row in events.iterrows():
            p = passage_for_row[r]
            seg = fit_to_duration(raw[p], row.duration, sr)
            if row.trial_type != "Speech":
                seg = manipulate(seg, cond, sr)
            seg = set_rms(seg)
            i0 = int(round(row.onset * sr))
            track[i0:i0 + seg.size] = seg
            blocks.append({"trial_type": row.trial_type, "onset": float(row.onset),
                           "duration": float(row.duration), "passage": p})
        peak = np.abs(track).max()
        if peak > 0.99:
            track *= 0.99 / peak
        sf.write(out_dir / f"{cond}.wav", track.astype(np.float32), sr)
        manifest["conditions"][cond] = {"path": f"{cond}.wav", "md5": md5(out_dir / f"{cond}.wav"),
                                        "blocks": blocks}
        print(f"wrote {cond}.wav ({total_s:.0f} s)")

    manifest["conditions"] = {c: manifest["conditions"][c] for c in ic.conditions
                              if c in manifest["conditions"]}
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return out_dir
