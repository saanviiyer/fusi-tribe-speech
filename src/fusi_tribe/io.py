"""Loading one fUSI run: Doppler volume, frame timestamps, task events."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

REQUIRED_EVENT_COLS = ("onset", "duration")


@dataclass
class Run:
    data: np.ndarray          # (x, y, z, t) power Doppler
    affine: np.ndarray
    frame_times: np.ndarray   # seconds from the first volume
    tr: float                 # median frame interval, seconds
    events: pd.DataFrame
    logs: pd.DataFrame

    @property
    def n_frames(self) -> int:
        return self.data.shape[-1]

    @property
    def voxel_size_mm(self) -> np.ndarray:
        return np.sqrt((self.affine[:3, :3] ** 2).sum(axis=0))


def resolve_pwd_path(data_dir: Path) -> Path:
    for name in ("pwd.nii.gz", "pwd.nii"):
        if (data_dir / name).exists():
            return data_dir / name
    raise FileNotFoundError(f"No pwd.nii or pwd.nii.gz in {data_dir} (see data/README.md)")


def frame_times_from_logs(logs: pd.DataFrame) -> tuple[np.ndarray, float]:
    """Frame times (s, first frame = 0) and median TR from per-volume timestamps.

    The NIfTI header's TR is not trusted: in the example run it says 1.0 s while
    the logged frame interval is 1.602 s.
    """
    if "time_ns" not in logs.columns:
        raise ValueError("logs.parquet is missing required column 'time_ns'")
    t = np.sort(logs["time_ns"].to_numpy(dtype=np.int64))
    if t.size < 2:
        raise ValueError("logs.parquet must contain at least two volume timestamps")
    frame_times = (t - t[0]) / 1e9
    return frame_times, float(np.median(np.diff(frame_times)))


def load_events(path: Path, event_col: str = "trial_type") -> pd.DataFrame:
    events = pd.read_csv(path, sep="\t", comment="#")
    missing = set(REQUIRED_EVENT_COLS + (event_col,)) - set(events.columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")
    return events


def load_run(data_dir: Path, event_col: str = "trial_type") -> Run:
    img = nib.load(resolve_pwd_path(data_dir))
    data = np.asarray(img.dataobj, dtype=np.float32)
    logs = pd.read_parquet(data_dir / "logs.parquet")
    frame_times, tr = frame_times_from_logs(logs)
    if frame_times.size != data.shape[-1]:
        raise ValueError(f"{frame_times.size} timestamps in logs.parquet but "
                         f"{data.shape[-1]} volumes in the NIfTI")
    events = load_events(data_dir / "task_events.tsv", event_col)
    return Run(data=data, affine=img.affine, frame_times=frame_times, tr=tr,
               events=events, logs=logs)


def save_nifti(arr: np.ndarray, affine: np.ndarray, path: Path) -> Path:
    img = nib.Nifti1Image(np.asarray(arr, dtype=np.float32), affine)
    img.set_data_dtype(np.float32)
    nib.save(img, path)
    return path
