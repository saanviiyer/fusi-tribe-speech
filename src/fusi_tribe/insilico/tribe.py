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


def _text_model_on_mps() -> None:
    """Run the Llama text extractor on Apple's GPU in float16.

    On CPU, Llama-3.2-3B word embeddings take minutes per batch (hundreds of
    hours per run on a 24 GB Mac). neuralset's own "accelerate" mode loads the
    model in float16 but then sends inputs to the CPU off CUDA, so instead the
    extractor stays configured as "cpu", the loaded model is moved to MPS in
    float16 (the precision of neuralset's accelerate path), and a pre-forward
    hook moves each batch's inputs to the model's device.
    """
    import torch
    from neuralset.extractors.text import HuggingFaceText

    original = HuggingFaceText._load_model
    if getattr(original, "_fusi_tribe_mps", False):
        return

    def to_device(obj, device):
        if torch.is_tensor(obj):
            return obj.to(device)
        if isinstance(obj, dict):
            return {k: to_device(v, device) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return type(obj)(to_device(v, device) for v in obj)
        return obj

    def load_on_mps(self, **kwargs):
        model = original(self, **kwargs).to(device="mps", dtype=torch.float16)
        model.register_forward_pre_hook(
            lambda _m, args, kw: (to_device(args, "mps"), to_device(kw, "mps")), with_kwargs=True)
        return model

    load_on_mps._fusi_tribe_mps = True
    HuggingFaceText._load_model = load_on_mps


def load_tribe(cache: Path, device: str | None = None):
    from tribev2.demo_utils import TribeModel

    device = device or _device()
    # The released config hard-codes CUDA for every feature extractor, and the
    # extractors only accept auto/cpu/cuda, so off-CUDA they run on the CPU.
    ext = "cuda" if device == "cuda" else "cpu"
    overrides = {f"data.{k}": ext for k in (
        "text_feature.device", "audio_feature.device",
        "image_feature.image.device", "video_feature.image.device")}
    if device == "cuda":
        import torch
        if torch.cuda.get_device_properties(0).total_memory < 20e9:
            # Llama-3.2-3B in float32 (12.8 GB) plus the audio model does not fit a
            # 16 GB GPU (e.g. a Colab T4); neuralset's "accelerate" mode loads it in float16.
            overrides["data.text_feature.device"] = "accelerate"
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
        if device == "mps":
            _text_model_on_mps()

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


def transcription_summary(cfg: Config) -> pd.DataFrame:
    """What the text pathway would hear: WhisperX words per block type and condition.

    `matched` counts control-block words that occur in the passage actually
    spoken there; intact speech is the reference for how many words exist.
    """
    import re

    from .passages import PASSAGES

    stim = cfg.output_dir / "stimuli"
    out = cfg.output_dir / "transcripts"
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((stim / "manifest.json").read_text())
    norm = lambda w: re.sub(r"[^a-z']", "", str(w).lower())
    rows = []
    for cond, info in manifest["conditions"].items():
        path = out / f"{cond}.tsv"
        if path.exists():
            words = pd.read_csv(path, sep="\t")
        else:
            words = transcribe(stim / info["path"])
            words.to_csv(path, sep="\t", index=False)
        n_speech = n_control = matched = 0
        for b in info["blocks"]:
            w = words[(words.start >= b["onset"]) & (words.start < b["onset"] + b["duration"])]
            if b["trial_type"] == "Speech":
                n_speech += len(w)
            else:
                truth = {norm(x) for x in PASSAGES[b["passage"]].split()}
                n_control += len(w)
                matched += sum(1 for x in map(norm, w.text) if x and x in truth)
        rows.append({"condition": cond, "speech_block_words": n_speech,
                     "control_block_words": n_control, "control_words_matching_passage": matched})
    df = pd.DataFrame(rows)
    ref = df.loc[df.condition == "intact", "control_words_matching_passage"]
    if len(ref):
        df["fraction_of_intact_words_recovered"] = (df.control_words_matching_passage / ref.iloc[0]).round(3)
    (cfg.results_dir / "insilico").mkdir(parents=True, exist_ok=True)
    df.to_csv(cfg.results_dir / "insilico" / "transcription.tsv", sep="\t", index=False)
    return df
