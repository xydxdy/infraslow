"""Reduce a hypnodensity CSV (per-epoch stage probabilities) to a hypnogram.

A *hypnodensity* gives, for every epoch, a probability for each sleep stage
(e.g. ``Wake, N1, N2, N3, REM``). The discrete stage annotation for an epoch is
simply the **argmax** stage. This module reads such a CSV and returns one
``(timestamp, stage)`` row per epoch -- the winning stage label per epoch.

Expected CSV layout (header row required)::

    Timestamp,Wake,N1,N2,N3,REM
    2012-01-01 21:52:52+00:00,0.9997,0.00017,7.3e-05,1.8e-05,3.0e-05
    ...
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, List, Mapping, Optional, Union

import pandas as pd

DEFAULT_TIMESTAMP_COLUMN = "Timestamp"
DEFAULT_STAGING_DIRNAME = "Sleep_Staging"
DEFAULT_HYPNODENSITY_SUFFIX = "_Hypnodensity.csv"


def hypnodensity_to_annotations(
    path: Union[str, Path],
    *,
    timestamp_column: str = DEFAULT_TIMESTAMP_COLUMN,
    stage_columns: Optional[List[str]] = None,
    stage_map: Optional[Mapping[str, str]] = None,
    parse_timestamps: bool = True,
) -> pd.DataFrame:
    """Read a hypnodensity CSV and reduce it to one stage per epoch via argmax.

    Args:
        path: Path to the hypnodensity CSV (header: a timestamp column plus one
            probability column per stage).
        timestamp_column: Name of the timestamp column (default ``"Timestamp"``).
        stage_columns: Probability columns to argmax over, in priority order
            (ties go to the earliest). Defaults to every column except the
            timestamp column, in file order.
        stage_map: Optional remap applied to the winning column name, e.g.
            ``{"Wake": "W", "REM": "R"}``. Unmapped labels are kept as-is.
        parse_timestamps: If True (default), parse the timestamp column to
            timezone-aware pandas datetimes.

    Returns:
        A DataFrame with columns ``["timestamp", "stage"]`` -- one row per epoch,
        ``stage`` being the argmax stage label.

    Raises:
        FileNotFoundError: if ``path`` does not exist.
        KeyError: if the timestamp or a requested stage column is absent.
        ValueError: if there are no stage columns to argmax over.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Hypnodensity file does not exist: {path}")

    df = pd.read_csv(path)
    if timestamp_column not in df.columns:
        raise KeyError(
            f"Timestamp column '{timestamp_column}' not found. "
            f"Columns: {list(df.columns)}"
        )

    if stage_columns is None:
        stage_columns = [c for c in df.columns if c != timestamp_column]
    if not stage_columns:
        raise ValueError("No stage probability columns found to argmax over.")
    missing = [c for c in stage_columns if c not in df.columns]
    if missing:
        raise KeyError(
            f"Stage column(s) not found: {missing}. Columns: {list(df.columns)}"
        )

    # Coerce probabilities to numeric (bad cells -> NaN) and take the per-epoch
    # argmax. ``idxmax`` returns the winning column's *name*, i.e. the stage.
    probs = df[stage_columns].apply(pd.to_numeric, errors="coerce")
    stage = probs.idxmax(axis=1)
    if stage_map:
        stage = stage.map(lambda s: stage_map.get(s, s))

    timestamps = df[timestamp_column]
    if parse_timestamps:
        timestamps = pd.to_datetime(timestamps, errors="coerce", utc=True)

    return pd.DataFrame({"timestamp": timestamps.to_numpy(), "stage": stage.to_numpy()})


def make_hypnodensity_annotation_loader(
    *,
    staging_dir: Optional[Union[str, Path]] = None,
    staging_dirname: str = DEFAULT_STAGING_DIRNAME,
    suffix: str = DEFAULT_HYPNODENSITY_SUFFIX,
    required: bool = True,
    **kwargs: Any,
) -> Callable[[Any, Path], Optional[pd.DataFrame]]:
    """Build an ``annotation_loader`` for :class:`BioserenityPSGLoader`.

    The returned callable matches the loader's ``annotation_loader(inst, edf_path)``
    contract: after the loader opens an EDF, it locates the matching hypnodensity
    CSV, reduces it with :func:`hypnodensity_to_annotations`, and the
    ``(timestamp, stage)`` DataFrame becomes available as ``loader.annotations``.

    The subject id is taken from the EDF file stem, so for
    ``.../Bioserenity/edf/{id}.edf`` the default locates
    ``.../Bioserenity/Sleep_Staging/{id}_Hypnodensity.csv``.

    Args:
        staging_dir: Directory holding the hypnodensity CSVs. If ``None`` (default),
            derived per-recording as ``<edf_path>/../../<staging_dirname>`` (the
            staging directory sits beside the ``edf`` directory).
        staging_dirname: Sibling directory name used when ``staging_dir`` is None.
        suffix: Filename suffix appended to the subject id.
        required: If ``True`` (default), a missing hypnodensity file raises
            :class:`FileNotFoundError` (which the loader surfaces as
            ``AnnotationLoadError``). If ``False``, a missing file yields ``None``.
        **kwargs: Forwarded to :func:`hypnodensity_to_annotations` (e.g.
            ``stage_map``, ``stage_columns``, ``parse_timestamps``).

    Usage::

        from infraslow import BioserenityPSGLoader, BIOSERENITY_ALIAS_MAP
        from infraslow.io import make_hypnodensity_annotation_loader

        loader = BioserenityPSGLoader(
            subject_id="{00008D23-3AB0-499C-BA8F-A6A4FC3C3154}",
            alias_map=BIOSERENITY_ALIAS_MAP,
            annotation_loader=make_hypnodensity_annotation_loader(),
        ).load()
        loader.annotations  # (timestamp, stage) DataFrame
    """
    base_override = Path(staging_dir) if staging_dir is not None else None

    def _annotation_loader(inst: Any, edf_path: Path) -> Optional[pd.DataFrame]:
        edf_path = Path(edf_path)
        subject_id = edf_path.stem
        base = base_override if base_override is not None else edf_path.parent.parent / staging_dirname
        csv_path = base / f"{subject_id}{suffix}"
        if not csv_path.is_file():
            if required:
                raise FileNotFoundError(f"Hypnodensity file not found: {csv_path}")
            return None
        return hypnodensity_to_annotations(csv_path, **kwargs)

    return _annotation_loader
