"""Load 1-second U-Sleep hypnodensities and reduce them to coarser hypnograms.

U-Sleep scores sleep-stage probabilities once per second, one ``.npy`` file per
Bioserenity/U-Sleep channel name, under
``infraslow.constants.DEFAULT_USLEEP_HYPNODENSITY_DIR``::

    {DEFAULT_USLEEP_HYPNODENSITY_DIR}/{subject_id}/{channel_file}.npy

Each array is ``(n_seconds, 5)`` float, columns in
:data:`~infraslow.constants.USLEEP_STAGE_ORDER` order (``Wake, N1, N2, N3, REM``), each row
summing to ~1. This mirrors the stage order :mod:`infraslow.io.hypnodensity` already uses
for the (30-s epoch) Bioserenity hypnodensity CSVs, but it is a *different* data source
(1-s U-Sleep model output, not the scored 30-s CSV) -- nothing here reads or depends on
that module.

Channel filenames are not consistent across the cohort: some subjects use the raw analysis
channel name (``C3.npy``), others the referenced Bioserenity/U-Sleep name
(``EEG C3-A2.npy``). :func:`resolve_usleep_channel_path` tries every alias for a channel
(in :data:`~infraslow.constants.BIOSERENITY_ALIAS_MAP` priority order) and returns the
first one that exists on disk.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..constants import (
    BIOSERENITY_ALIAS_MAP,
    DEFAULT_HYPNOGRAM_EPOCH_SEC,
    DEFAULT_USLEEP_EPOCH_SEC,
    DEFAULT_USLEEP_HYPNODENSITY_DIR,
    USLEEP_STAGE_ORDER,
)
from ..processing.infraslow import _nearest_sample_index


def resolve_usleep_channel_path(
    subject_dir: Path, channel: str, *,
    alias_map: Mapping[str, Sequence[str]] = BIOSERENITY_ALIAS_MAP,
) -> Optional[Path]:
    """First ``<subject_dir>/<alias>.npy`` that exists, trying every alias for ``channel``
    in ``alias_map`` priority order. ``None`` if no alias file exists.

    Raises:
        KeyError: if ``channel`` is not a key of ``alias_map``.
    """
    subject_dir = Path(subject_dir)
    for alias in alias_map[channel]:
        candidate = subject_dir / f"{alias}.npy"
        if candidate.is_file():
            return candidate
    return None


def load_usleep_hypnodensity(
    subject_id: str, channel: str, *,
    base_dir: str = DEFAULT_USLEEP_HYPNODENSITY_DIR,
    alias_map: Mapping[str, Sequence[str]] = BIOSERENITY_ALIAS_MAP,
    epoch_sec: float = DEFAULT_USLEEP_EPOCH_SEC,
) -> Tuple[np.ndarray, np.ndarray]:
    """``(t_hyp, probs)`` for one subject/channel's 1-s U-Sleep hypnodensity.

    ``t_hyp`` is ``(n,)``, bin-centered seconds from the recording start
    (``epoch_sec * (i + 0.5)`` for row ``i``) -- the same sample-centered time convention
    :func:`infraslow.pipeline.io.load_temporal_isfs`'s ``t_env`` uses, so the two can be
    compared/aligned directly. ``probs`` is ``(n, 5)``, columns in
    :data:`~infraslow.constants.USLEEP_STAGE_ORDER` order.

    Raises:
        FileNotFoundError: if ``<base_dir>/<subject_id>`` does not exist, or no alias file
            for ``channel`` exists inside it.
        ValueError: if the loaded array's shape doesn't match ``(n, len(USLEEP_STAGE_ORDER))``.
    """
    subject_dir = Path(os.path.expandvars(str(base_dir))) / subject_id
    if not subject_dir.is_dir():
        raise FileNotFoundError(f"No U-Sleep hypnodensity directory for subject: {subject_dir}")

    path = resolve_usleep_channel_path(subject_dir, channel, alias_map=alias_map)
    if path is None:
        raise FileNotFoundError(
            f"No U-Sleep hypnodensity file for channel '{channel}' under {subject_dir} "
            f"(tried aliases: {list(alias_map[channel])})"
        )

    probs = np.load(path).astype(np.float64)
    if probs.ndim != 2 or probs.shape[1] != len(USLEEP_STAGE_ORDER):
        raise ValueError(
            f"Unexpected U-Sleep hypnodensity shape {probs.shape} in {path}; "
            f"expected (n_seconds, {len(USLEEP_STAGE_ORDER)})"
        )
    t_hyp = epoch_sec * (np.arange(probs.shape[0], dtype=np.float64) + 0.5)
    return t_hyp, probs


def hypnodensity_to_epoch_hypnogram(
    t_hyp: np.ndarray, probs: np.ndarray, *,
    epoch_sec: float = DEFAULT_HYPNOGRAM_EPOCH_SEC,
    src_epoch_sec: float = DEFAULT_USLEEP_EPOCH_SEC,
    stage_order: Sequence[str] = USLEEP_STAGE_ORDER,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reduce a 1-s hypnodensity to a coarser (default 3-s) hypnogram.

    Averages every consecutive ``round(epoch_sec / src_epoch_sec)`` rows of ``probs``
    first, *then* takes the argmax stage per window -- never the other way around
    (hard-labeling each 1-s row before averaging would throw away exactly the soft
    information the averaging step is for). A trailing partial window (fewer than the full
    number of source rows) is dropped.

    Returns:
        ``(t_epoch, probs_epoch, stage_epoch)``: bin-centered epoch times ``(m,)``,
        averaged probabilities ``(m, n_stages)``, and the argmax stage label per epoch
        ``(m,)`` (values from ``stage_order``).

    Raises:
        ValueError: if ``epoch_sec`` is not a positive integer multiple of
            ``src_epoch_sec``, or ``t_hyp``/``probs`` row counts disagree.
    """
    if t_hyp.shape[0] != probs.shape[0]:
        raise ValueError(f"t_hyp/probs length mismatch: {t_hyp.shape[0]} vs {probs.shape[0]}")
    ratio = epoch_sec / src_epoch_sec
    n_per_window = round(ratio)
    if n_per_window < 1 or not np.isclose(ratio, n_per_window):
        raise ValueError(
            f"epoch_sec ({epoch_sec}) must be a positive integer multiple of "
            f"src_epoch_sec ({src_epoch_sec})"
        )

    n_windows = probs.shape[0] // n_per_window
    if n_windows == 0:
        return (np.empty(0), np.empty((0, probs.shape[1])), np.empty(0, dtype=object))

    trimmed = probs[: n_windows * n_per_window]
    probs_epoch = trimmed.reshape(n_windows, n_per_window, probs.shape[1]).mean(axis=1)
    stage_idx = probs_epoch.argmax(axis=1)
    stage_epoch = np.asarray(stage_order, dtype=object)[stage_idx]

    t_trimmed = t_hyp[: n_windows * n_per_window].reshape(n_windows, n_per_window)
    t_epoch = t_trimmed.mean(axis=1)
    return t_epoch, probs_epoch, stage_epoch


def hypnodensity_probs_at_times(
    query_t: np.ndarray, t_hyp: np.ndarray, probs: np.ndarray,
) -> np.ndarray:
    """``probs`` row nearest each ``query_t`` sample -- ``(len(query_t), n_stages)``.

    A thin nearest-neighbour lookup (reusing
    :func:`infraslow.processing.infraslow._nearest_sample_index`, the same helper
    :mod:`infraslow.pipeline.phase` uses to assign event times to phase-bin samples) so a
    phase time series (recording-relative seconds -- see
    :func:`infraslow.pipeline.phase.build_subject_phase_timeseries`) can be matched up with
    the hypnodensity probability in effect at each of those times.
    """
    idx = _nearest_sample_index(np.asarray(t_hyp), np.asarray(query_t))
    return probs[idx]


def make_usleep_annotation_loader(
    channel: str,
    *,
    base_dir: str = DEFAULT_USLEEP_HYPNODENSITY_DIR,
    alias_map: Mapping[str, Sequence[str]] = BIOSERENITY_ALIAS_MAP,
    epoch_sec: float = DEFAULT_HYPNOGRAM_EPOCH_SEC,
    src_epoch_sec: float = DEFAULT_USLEEP_EPOCH_SEC,
    stage_order: Sequence[str] = USLEEP_STAGE_ORDER,
    required: bool = True,
) -> Callable[[Any, Path], Optional[pd.DataFrame]]:
    """Build an ``annotation_loader`` for :class:`BioserenityPSGLoader` from U-Sleep.

    The returned callable matches the loader's ``annotation_loader(inst, edf_path)``
    contract: after the loader opens an EDF, it reads ``channel``'s 1-s U-Sleep
    hypnodensity (:func:`load_usleep_hypnodensity`) and reduces it to an
    ``epoch_sec`` hypnogram (:func:`hypnodensity_to_epoch_hypnogram`); the result
    becomes available as ``loader.annotations`` -- a DataFrame with one row per
    epoch and columns ``["t", *stage_order, "stage"]`` (bin-centered epoch time,
    one probability column per stage in ``stage_order``, then the argmax
    ``stage`` label) -- both the per-epoch *hypnodensity* (the probability
    columns) and the *hypnogram* (``stage``) in one table, the same ``stage``
    column shape :func:`infraslow.processing.spindle.spindles_detect` already
    expects from ``loader.annotations['stage']``. This is
    :class:`BioserenityPSGLoader`'s default annotation source (see its
    ``usleep_channel``/``annotation_loader`` docs); build one directly only to
    override the channel, epoching, or U-Sleep directory.

    The subject id is taken from the EDF file stem, matching
    :func:`~infraslow.io.hypnodensity.make_hypnodensity_annotation_loader`'s own
    convention.

    Args:
        channel: Canonical channel name (an ``alias_map`` key) whose U-Sleep
            hypnodensity to load, e.g. ``"C3"``.
        base_dir: Root directory of the per-subject U-Sleep ``.npy`` files.
        alias_map: Canonical-channel -> physical-alias map used to resolve the
            on-disk filename (see :func:`resolve_usleep_channel_path`).
        epoch_sec: Width (s) of the reduced hypnogram's epochs (default 3 s).
        src_epoch_sec: Sampling interval (s) of the raw hypnodensity rows
            (default 1 s, matching U-Sleep's native output).
        stage_order: Column order of the raw hypnodensity (defaults to
            :data:`~infraslow.constants.USLEEP_STAGE_ORDER`).
        required: If ``True`` (default), a missing U-Sleep file raises
            :class:`FileNotFoundError` (surfaced by the loader as
            ``AnnotationLoadError``). If ``False``, a missing file yields ``None``.

    Usage::

        from infraslow import BioserenityPSGLoader
        from infraslow.io.usleep_hypnodensity import make_usleep_annotation_loader

        loader = BioserenityPSGLoader(
            subject_id="318562",
            requested_channels=["C3"],
            annotation_loader=make_usleep_annotation_loader("C3"),
        ).load()
        loader.annotations  # ["t", "Wake", "N1", "N2", "N3", "REM", "stage"] DataFrame,
                            # one row per epoch_sec epoch
    """

    def _annotation_loader(inst: Any, edf_path: Path) -> Optional[pd.DataFrame]:
        subject_id = Path(edf_path).stem
        try:
            t_hyp, probs = load_usleep_hypnodensity(
                subject_id, channel, base_dir=base_dir, alias_map=alias_map,
                epoch_sec=src_epoch_sec,
            )
        except FileNotFoundError:
            if required:
                raise
            return None
        t_epoch, probs_epoch, stage_epoch = hypnodensity_to_epoch_hypnogram(
            t_hyp, probs, epoch_sec=epoch_sec, src_epoch_sec=src_epoch_sec,
            stage_order=stage_order,
        )
        annotations = pd.DataFrame(probs_epoch, columns=list(stage_order))
        annotations.insert(0, "t", t_epoch)
        annotations["stage"] = stage_epoch
        return annotations

    return _annotation_loader


__all__ = [
    "resolve_usleep_channel_path",
    "load_usleep_hypnodensity",
    "hypnodensity_to_epoch_hypnogram",
    "hypnodensity_probs_at_times",
    "make_usleep_annotation_loader",
]
