"""Typed configuration loaded from YAML (see configs/default.yaml)."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"


@dataclass
class FusiConfig:
    subject: str = "Forest001"
    session: str = "20260414"
    run_label: str = "audio_no_hair"
    event_col: str = "trial_type"
    initial_censor_frames: int = 0
    global_signal_k: float = 4.0
    fd_thresh_mm: float = 0.5
    fd_radius_mm: float = 50.0
    smooth_method: str = "boxcar"
    smooth_window: int = 3
    spatial_fwhm_mm: float = 2.0
    hrf_model: str = "glover + derivative"
    drift_model: str = "cosine"
    high_pass_hz: float = 0.01
    noise_model: str = "ar1"
    onset_shift_s: float = 0.0
    z_threshold: float = 3.1


@dataclass
class TimingConfig:
    n_roi_voxels: int = 500
    fir_lags: int = 15
    shift_min_s: float = -6.0
    shift_max_s: float = 6.0
    shift_step_s: float = 0.2


@dataclass
class InsilicoConfig:
    voice: str = "Samantha"
    rate_wpm: int = 140
    sample_rate: int = 16000
    tail_s: float = 20.0
    conditions: list[str] = field(default_factory=lambda: [
        "intact", "lowpass", "rotated", "vocoded", "reversed", "silence"])
    tribe_offset_s: float = 5.0
    n_boot: int = 5000


@dataclass
class Config:
    data_dir: Path
    output_dir: Path
    results_dir: Path
    fusi: FusiConfig
    timing: TimingConfig
    insilico: InsilicoConfig

    @property
    def run_name(self) -> str:
        f = self.fusi
        return f"sub-{f.subject}_ses-{f.session}_{f.run_label}"


def _build(cls, values: dict | None):
    values = values or {}
    known = {f.name for f in fields(cls)}
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} keys: {sorted(unknown)}")
    return cls(**values)


def load_config(path: str | Path | None = None) -> Config:
    path = Path(path) if path else DEFAULT_CONFIG
    raw = yaml.safe_load(path.read_text()) or {}

    def resolve(p: str) -> Path:
        p = Path(p).expanduser()
        return p if p.is_absolute() else REPO_ROOT / p

    return Config(
        data_dir=resolve(raw.get("data_dir", "data")),
        output_dir=resolve(raw.get("output_dir", "outputs")),
        results_dir=resolve(raw.get("results_dir", "results")),
        fusi=_build(FusiConfig, raw.get("fusi")),
        timing=_build(TimingConfig, raw.get("timing")),
        insilico=_build(InsilicoConfig, raw.get("insilico")),
    )
