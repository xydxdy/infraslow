"""Read-only loaders for preprocessing.py's saved per-subject artifacts.

Every function here does exactly one `np.load`/`pd.read_csv` against the
layout documented in `src/scripts/preprocessing.py`'s module docstring --
none of them touch an EDF, a `BioserenityPSGLoader`, or re-run any
detection. `subject_dir` throughout is `<data-dir>/<subject_id>` (the
directory containing one or more channel subdirectories), matching
`preprocessing.py`'s `output_dir / DATA_DIRNAME / subject_id`.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from ..constants import DEFAULT_METADATA, DEFAULT_METADATA2
from ..io.metadata import combine_bioserenity_metadata, load_bioserenity_metadata


def discover_subjects(data_dir: Path) -> List[str]:
    """Every subject id -- one subdirectory name directly under `data_dir`, sorted."""
    with os.scandir(data_dir) as it:
        return sorted(entry.name for entry in it if entry.is_dir())


def discover_channels(data_dir: Path, subject_id: str) -> List[str]:
    """Every channel processed for one subject -- subdirectory names under
    `<data_dir>/<subject_id>`, sorted."""
    with os.scandir(Path(data_dir) / subject_id) as it:
        return sorted(entry.name for entry in it if entry.is_dir())


def load_envelope(subject_dir: Path, channel: str, band: str) -> Tuple[np.ndarray, np.ndarray]:
    """`(t_env, power)` from `<subject_dir>/<channel>/envelope/<band>.npz`."""
    path = Path(subject_dir) / channel / "envelope" / f"{band}.npz"
    with np.load(path) as npz:
        return npz["t_env"], npz["power"]


def load_temporal_isfs(subject_dir: Path, channel: str, band: str) -> Tuple[np.ndarray, np.ndarray]:
    """`(t_env, filtered)` from `<subject_dir>/<channel>/temporal_ISFS/<band>.npz`."""
    path = Path(subject_dir) / channel / "temporal_ISFS" / f"{band}.npz"
    with np.load(path) as npz:
        return npz["t_env"], npz["power"]


def load_stage_bouts(subject_dir: Path, channel: str, stage: str) -> Dict[str, np.ndarray]:
    """`{"all": (n,2), "spindle": (m,2)}` from `<subject_dir>/<channel>/<stage>/bouts.npz`."""
    path = Path(subject_dir) / channel / stage / "bouts.npz"
    with np.load(path) as npz:
        return {"all": npz["all"], "spindle": npz["spindle"]}


def load_stage_sw_bouts(subject_dir: Path, channel: str, stage: str) -> Dict[str, np.ndarray]:
    """`{"all": (n,2), "sw": (m,2)}` from `<subject_dir>/<channel>/<stage>/sw_bouts.npz`."""
    path = Path(subject_dir) / channel / stage / "sw_bouts.npz"
    with np.load(path) as npz:
        return {"all": npz["all"], "sw": npz["sw"]}


def load_stage_isfs_spectra(
    subject_dir: Path, channel: str, stage: str, band: str,
) -> Dict[str, np.ndarray]:
    """`{"freqs": (f,), "psds": (n,f), "bout_start": (n,)}` from
    `<subject_dir>/<channel>/<stage>/ISFS/<band>.npz` -- one PSD per bout in
    that stage's `bouts.npz["all"]`, same order (both come from the same
    `all_bouts` list inside `preprocess_channel`)."""
    path = Path(subject_dir) / channel / stage / "ISFS" / f"{band}.npz"
    with np.load(path) as npz:
        return {"freqs": npz["freqs"], "psds": npz["psds"], "bout_start": npz["bout_start"]}


def load_spindle_summary(subject_dir: Path, channel: str, stage: str) -> pd.DataFrame:
    """YASA spindle summary from `<subject_dir>/<channel>/<stage>/spindel_yasa.csv`."""
    path = Path(subject_dir) / channel / stage / "spindel_yasa.csv"
    return pd.read_csv(path)


def load_sw_summary(subject_dir: Path, channel: str, stage: str) -> pd.DataFrame:
    """YASA slow-wave summary from `<subject_dir>/<channel>/<stage>/sw_yasa.csv`."""
    path = Path(subject_dir) / channel / stage / "sw_yasa.csv"
    return pd.read_csv(path)


def load_demographics(
    metadata_path: str = DEFAULT_METADATA, metadata2_path: str = DEFAULT_METADATA2,
) -> pd.DataFrame:
    """Combined `["ID", "Age", "Gender", "BMI"]` demographics (union of both
    metadata CSVs by ID) -- same source `sleep_statistics.py`'s cohort
    discovery and `preprocessing.py` both use."""
    return combine_bioserenity_metadata(
        load_bioserenity_metadata(Path(metadata_path)),
        load_bioserenity_metadata(Path(metadata2_path)),
    )


def load_sleep_statistics(path: Path) -> pd.DataFrame:
    """Whole-night sleep statistics from `sleep_statistics.py`'s output CSV.

    Never raises: `sleep_statistics.py` may still be running (partially
    written file), may not have started yet (missing file), or the file may
    be transiently unreadable (being written to concurrently) -- any of
    these degrade to an empty DataFrame with just an `"id"` column, so a
    left join against it always works and just leaves every other column NaN.
    """
    try:
        df = pd.read_csv(path)
    except (FileNotFoundError, OSError, ValueError, pd.errors.ParserError, pd.errors.EmptyDataError):
        return pd.DataFrame({"id": pd.Series(dtype=str)})
    if "id" not in df.columns:
        return pd.DataFrame({"id": pd.Series(dtype=str)})
    return df


__all__ = [
    "discover_subjects",
    "discover_channels",
    "load_envelope",
    "load_temporal_isfs",
    "load_stage_bouts",
    "load_stage_sw_bouts",
    "load_stage_isfs_spectra",
    "load_spindle_summary",
    "load_sw_summary",
    "load_demographics",
    "load_sleep_statistics",
]
