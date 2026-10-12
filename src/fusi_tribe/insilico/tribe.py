"""Run TRIBE v2 on the simulated runs.

Modes
    audio_only  audio pathway only (no transcription, no language model)
    full        audio + WhisperX transcript + Llama-3.2-3B text features
                (requires accepting the gated meta-llama/Llama-3.2-3B licence)

Writes outputs/predictions/<mode>/<condition>.npz with `preds` (n_seconds,
20484 fsaverage5 vertices) and `starts` (s); in full mode also the transcript
TRIBE saw, which shows how much of each acoustic control the text pathway heard.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Config
from .stimuli import md5

MODES = ("audio_only", "full")


def _device() -> str:
    import torch
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def transcribe(wav: Path, language: str = "english", device: str | None = None) -> pd.DataFrame:
    """WhisperX word transcript, as TRIBE's ExtractWordsFromAudio produces it.

    TRIBE runs `uvx whisperx ... --compute_type float16`, which the CPU backend
    rejects. This is the same call with int8 off-CUDA, using the `uvx` installed
    next to this Python.
    """
    import os
    import subprocess
    import sys
    import tempfile

    import torch

    codes = {"english": "en", "french": "fr", "spanish": "es", "dutch": "nl", "chinese": "zh"}
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    uvx = Path(sys.executable).with_name("uvx")
    with tempfile.TemporaryDirectory() as out:
        cmd = [str(uvx if uvx.exists() else "uvx"), "whisperx", str(wav), "--model", "large-v3",
               "--language", codes[language], "--device", device,
               "--compute_type", "float16" if device == "cuda" else "int8", "--batch_size", "16",
               "--output_dir", out, "--output_format", "json"]
        if language == "english":
            cmd += ["--align_model", "WAV2VEC2_ASR_LARGE_LV60K_960H"]
        env = {k: v for k, v in os.environ.items() if k != "MPLBACKEND"}
        res = subprocess.run(cmd, capture_output=True, text=True, env=env)
        if res.returncode != 0:
            raise RuntimeError(f"whisperx failed:\n{res.stderr[-3000:]}")
        transcript = json.loads((Path(out) / f"{wav.stem}.json").read_text())
    words = [{"text": w["word"].replace('"', ""), "start": w["start"], "duration": w["end"] - w["start"],
              "sequence_id": i, "sentence": seg["text"].replace('"', "")}
             for i, seg in enumerate(transcript["segments"]) for w in seg["words"] if "start" in w]
    return pd.DataFrame(words)


def _use_cpu_friendly_transcription() -> None:
    from tribev2.eventstransforms import ExtractWordsFromAudio
    ExtractWordsFromAudio._get_transcript_from_audio = staticmethod(
        lambda wav_filename, language: transcribe(Path(wav_filename), language))


def load_tribe(cache: Path, device: str | None = None):
    from tribev2.demo_utils import TribeModel

    device = device or _device()
    # The released config hard-codes CUDA for every feature extractor, and the
    # extractors only accept auto/cpu/cuda, so off-CUDA they run on the CPU.
    ext = "cuda" if device == "cuda" else "cpu"
    overrides = {f"data.{k}": ext for k in (
        "text_feature.device", "audio_feature.device",
        "image_feature.image.device", "video_feature.image.device")}
    model = TribeModel.from_pretrained("facebook/tribev2", cache_folder=cache, device=device,
                                       config_update=overrides)
    model.remove_empty_segments = False   # keep silent seconds: same time axis for every run
    return model, device


def run_tribe(cfg: Config, mode: str, conditions: list[str] | None = None) -> Path:
    from tribev2.demo_utils import get_audio_and_text_events

    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    stim_dir = cfg.output_dir / "stimuli"
    manifest = json.loads((stim_dir / "manifest.json").read_text())
    out_dir = cfg.output_dir / "predictions" / mode
    out_dir.mkdir(parents=True, exist_ok=True)
    model, device = load_tribe(cfg.output_dir / "tribe_cache")
    if mode == "full" and device != "cuda":
        _use_cpu_friendly_transcription()

    for cond in conditions or list(manifest["conditions"]):
        t0 = time.time()
        wav = stim_dir / manifest["conditions"][cond]["path"]
        events = get_audio_and_text_events(
            pd.DataFrame([{"type": "Audio", "filepath": str(wav), "start": 0,
                           "timeline": cond, "subject": "default"}]),
            audio_only=(mode == "audio_only"))
        preds, segments = model.predict(events=events)
        starts = np.array([s.start for s in segments], dtype=float)
        order = np.argsort(starts)
        np.savez_compressed(out_dir / f"{cond}.npz", preds=preds[order].astype(np.float32),
                            starts=starts[order], stimulus_md5=md5(wav))
        words = events[events.type == "Word"]
        if len(words):
            words[["start", "duration", "text"]].to_csv(out_dir / f"{cond}_words.tsv",
                                                        sep="\t", index=False)
        print(f"[{mode}] {cond}: {preds.shape[0]} s predicted, {len(words)} words transcribed, "
              f"{time.time() - t0:.0f} s on {device}", flush=True)
    return out_dir
