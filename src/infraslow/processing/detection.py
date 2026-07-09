"""Sleep-event detection on loaded PSG signals using YASA.

This module wraps `yasa.spindles_detect
<https://raphaelvallat.com/yasa/>`_ so it composes with the rest of the
pipeline: it accepts the ``(n_channels, n_samples)`` array produced by
:class:`~infraslow.io.psg_loader.BioserenityPSGLoader` and the per-epoch
``(timestamp, stage)`` hypnogram produced by
:func:`~infraslow.io.hypnodensity.hypnodensity_to_annotations`, and restricts
detection to NREM sleep -- the workflow of YASA's
``03_spindles_detection_NREM_only`` notebook.

Detection itself is delegated entirely to YASA; this module only adapts the
repo's data shapes (canonical-named channels, string stage labels) to YASA's
expected inputs (sample-resolution integer hypnogram) and back.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# YASA's integer sleep-stage convention (see ``yasa.hypno_str_to_int``):
# -2=Unscored, -1=Artefact/Movement, 0=Wake, 1=N1, 2=N2, 3=N3, 4=REM.
# NREM sleep is N1+N2+N3; sleep spindles are a hallmark of N2 (and present in
# N3), so the NREM-only default mirrors the YASA notebook's ``include=(2, 3)``.
NREM_STAGES: Tuple[int, ...] = (2, 3)

# Default seconds per scored epoch. Bioserenity hypnodensity epochs are 30 s.
DEFAULT_EPOCH_SEC = 30.0

# Map the canonical stage labels this repo emits (Wake/N1/N2/N3/REM, see
# ``infraslow.io.hypnodensity``) onto YASA's integer codes. Lookups are
# case-insensitive; ``yasa.hypno_str_to_int`` covers the lowercase spellings,
# but pinning the mapping here keeps the contract explicit and stable.
DEFAULT_STAGE_MAP: Mapping[str, int] = {
    "wake": 0,
    "w": 0,
    "n1": 1,
    "n2": 2,
    "n3": 3,
    "rem": 4,
    "r": 4,
    "art": -1,
    "uns": -2,
}

# A YASA detection result, or ``None`` when no spindle is found.
SpindlesResult = Optional[Any]


def _stages_to_int(
    stages: Sequence[Any], stage_map: Mapping[str, int]
) -> np.ndarray:
    """Coerce a per-epoch stage sequence to YASA integer codes.

    Accepts labels that are already integers (passed through) or strings such as
    ``"N2"``/``"Wake"`` (mapped case-insensitively via ``stage_map``). Anything
    unrecognised raises, rather than being silently scored as Wake/Unscored.
    """
    out: List[int] = []
    for i, s in enumerate(stages):
        if isinstance(s, (int, np.integer)):
            out.append(int(s))
            continue
        key = str(s).strip().lower()
        if key not in stage_map:
            raise ValueError(
                f"Unrecognised sleep stage {s!r} at epoch {i}. Known labels: "
                f"{sorted(set(stage_map))}. Pass a stage_map to extend them."
            )
        out.append(stage_map[key])
    return np.asarray(out, dtype=int)


def _extract_epoch_stages(
    hypno: Union[pd.DataFrame, pd.Series, Sequence[Any]],
    *,
    stage_column: str,
) -> Sequence[Any]:
    """Pull the per-epoch stage labels out of the several shapes a caller may pass.

    A :class:`~pandas.DataFrame` (e.g. ``loader.annotations``) contributes its
    ``stage_column``; a :class:`~pandas.Series` or any other sequence is used as
    the stage labels directly.
    """
    if isinstance(hypno, pd.DataFrame):
        if stage_column not in hypno.columns:
            raise KeyError(
                f"Hypnogram DataFrame has no '{stage_column}' column; "
                f"columns: {list(hypno.columns)}."
            )
        return hypno[stage_column].tolist()
    if isinstance(hypno, pd.Series):
        return hypno.tolist()
    return list(hypno)


def _build_sample_hypno(
    hypno: Union[pd.DataFrame, pd.Series, Sequence[Any]],
    *,
    data: np.ndarray,
    sf: float,
    epoch_sec: float,
    stage_map: Mapping[str, int],
    stage_column: str,
) -> np.ndarray:
    """Turn a per-epoch hypnogram into a sample-resolution integer array for YASA.

    YASA wants one stage value per *sample*, aligned with ``data``. We map the
    per-epoch labels to integers and upsample with
    :func:`yasa.hypno_upsample_to_data`, which repeats each epoch's value across
    its samples and crops/pads to exactly match ``data`` so the two stay aligned.
    """
    import yasa

    stages = _extract_epoch_stages(hypno, stage_column=stage_column)
    hypno_int = _stages_to_int(stages, stage_map)
    # Per-epoch hypnogram is sampled at one value every ``epoch_sec`` seconds.
    sf_hypno = 1.0 / float(epoch_sec)
    return yasa.hypno_upsample_to_data(
        hypno_int, sf_hypno=sf_hypno, data=data, sf_data=sf, verbose=False
    )


def spindles_detect(
    loader: Any,
    *,
    ch_names: Optional[Union[str, Sequence[str]]] = None,
    include: Iterable[int] = NREM_STAGES,
    epoch_sec: float = DEFAULT_EPOCH_SEC,
    stage_map: Mapping[str, int] = DEFAULT_STAGE_MAP,
    stage_column: str = "stage",
    freq_sp: Tuple[float, float] = (12, 15),
    freq_broad: Tuple[float, float] = (1, 30),
    duration: Tuple[float, float] = (0.5, 2),
    min_distance: float = 500,
    thresh: Optional[Mapping[str, float]] = None,
    multi_only: bool = False,
    remove_outliers: bool = False,
    verbose: bool = False,
) -> SpindlesResult:
    """Detect sleep spindles on NREM epochs with YASA, from a loaded recording.

    A thin adapter over :func:`yasa.spindles_detect` that reads everything it
    needs off a :class:`~infraslow.io.psg_loader.BioserenityPSGLoader`: the EEG
    signal, its sampling rate, and the per-epoch ``(timestamp, stage)``
    hypnogram. When the loader carries a hypnogram, detection is restricted to
    NREM sleep (the ``03_spindles_detection_NREM_only`` workflow).

    Args:
        loader: A loaded (or loadable)
            :class:`~infraslow.io.psg_loader.BioserenityPSGLoader` -- anything
            exposing ``data``/``get_channel``, ``channel_names``, ``sf``,
            ``annotations`` and ``is_loaded``/``load()``. Loaded in place if not
            already.
        ch_names: Channel(s) to detect on. ``None`` (default) runs on every loaded
            channel (``loader.data``); a single canonical name (e.g. ``"C3"``)
            detects on that one channel; a list/sequence of names (e.g.
            ``["C3", "C4"]``) detects on just those, stacked into
            ``(n_channels, n_samples)``.
        include: Sleep stages (YASA integer codes) to detect within. Defaults to
            NREM ``(2, 3)`` = N2+N3. Ignored when the loader has no hypnogram.
        epoch_sec: Seconds per scored epoch, used to upsample the hypnogram to the
            sample rate of the data. Defaults to 30 s.
        stage_map: Case-insensitive label -> YASA-integer map for the string
            hypnogram. Defaults to this repo's Wake/N1/N2/N3/REM labels.
        stage_column: Stage column name in the loader's ``annotations`` DataFrame.
        freq_sp, freq_broad, duration, min_distance, thresh, multi_only,
        remove_outliers, verbose: Passed straight through to
            :func:`yasa.spindles_detect` (``thresh=None`` uses YASA's defaults).

    Returns:
        The :class:`yasa.SpindlesResults` object (call ``.summary()`` for the
        per-event table, ``.summary(grp_chan=True)`` for per-channel stats), or
        ``None`` when no spindle is detected -- matching YASA's own contract.

    Raises:
        ImportError: if YASA is not installed.
        ValueError: if the loader has no sampling rate or no data, or a stage
            label is unrecognised.
    """
    try:
        import yasa
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "spindles_detect requires the 'yasa' package. Install it with "
            "`pip install yasa` (in a Slurm job or interactive node, not the "
            "login node)."
        ) from exc

    # Load on demand so a freshly-built loader can be passed straight in.
    if hasattr(loader, "is_loaded") and not loader.is_loaded:
        loader.load()

    sf = getattr(loader, "sf", None)
    if sf is None:
        raise ValueError(
            "spindles_detect requires the loader's sampling frequency 'sf' (Hz); "
            "load the recording first."
        )

    # Select the channel(s): all loaded channels (None), one named channel (str),
    # or a given subset (list/sequence of names). A str is itself a Sequence, so
    # it must be checked before the general-sequence branch.
    ch_list: Optional[List[str]]
    if ch_names is None:
        arr = np.asarray(loader.data, dtype=float)
        loaded = getattr(loader, "channel_names", None)
        ch_list = list(loaded) if loaded else None
    elif isinstance(ch_names, str):
        arr = np.asarray(loader.get_channel(ch_names), dtype=float)
        ch_list = [ch_names]
    else:
        ch_list = list(ch_names)
        if not ch_list:
            raise ValueError("spindles_detect: `ch_names` list is empty.")
        arr = np.vstack(
            [np.asarray(loader.get_channel(c), dtype=float) for c in ch_list]
        )
    if arr.size == 0:
        raise ValueError("spindles_detect received empty data from the loader.")

    hypno = getattr(loader, "annotations", None)
    sample_hypno: Optional[np.ndarray] = None
    if hypno is not None:
        sample_hypno = _build_sample_hypno(
            hypno,
            data=arr,
            sf=sf,
            epoch_sec=epoch_sec,
            stage_map=stage_map,
            stage_column=stage_column,
        )

    # YASA only consults ``include``/stage restriction when a hypnogram exists.
    detect_kwargs: dict = dict(
        data=arr,
        sf=sf,
        ch_names=ch_list,
        hypno=sample_hypno,
        freq_sp=freq_sp,
        freq_broad=freq_broad,
        duration=duration,
        min_distance=min_distance,
        multi_only=multi_only,
        remove_outliers=remove_outliers,
        verbose=verbose,
    )
    if sample_hypno is not None:
        detect_kwargs["include"] = tuple(include)
    if thresh is not None:
        detect_kwargs["thresh"] = dict(thresh)

    result = yasa.spindles_detect(**detect_kwargs)
    if result is None:
        logger.info("No spindles detected for the requested stages %s.", tuple(include))
    return result


# --------------------------------------------------------------------------- #
# Multi-subject detection, aggregation, and plotting
# --------------------------------------------------------------------------- #
# Suggested EEG channels to load for spindle work when constructing the loaders
# passed to ``detect_subjects_spindles`` (detection then runs on one of them).
DEFAULT_EEG_CHANNELS: Tuple[str, ...] = ("F3", "F4", "C3", "C4", "O1", "O2")


def detect_subjects_spindles(
    loaders: Iterable[Any],
    *,
    channel: str = "C3",
    include: Iterable[int] = NREM_STAGES,
    skip_errors: bool = True,
    verbose: bool = True,
    **detect_kwargs: Any,
) -> "dict[str, SpindlesResult]":
    """Detect spindles on one EEG channel for each pre-built subject loader.

    Runner for the multi-subject workflow: given already-constructed
    :class:`~infraslow.io.psg_loader.BioserenityPSGLoader` objects (one per
    subject), it loads any that have not been loaded yet, then runs
    :func:`spindles_detect` on ``channel`` restricted to ``include`` stages
    (default NREM). Constructing the loaders -- choice of sampling rate, requested
    channels, alias map and annotation loader -- is the caller's responsibility,
    which keeps detection decoupled from loading (and this module free of any
    top-level dependency on the io layer).

    Returns an ordered ``{subject_id: SpindlesResults-or-None}`` mapping -- the
    value is ``None`` when no spindle is found, or when the subject errored and
    ``skip_errors`` is True (a warning is logged).

    Args:
        loaders: Iterable of ``BioserenityPSGLoader`` objects (or any object
            exposing ``subject_id``, ``is_loaded``/``load()``, ``get_channel``,
            ``sf`` and ``annotations``). Each is loaded in place if not already.
        channel: Canonical EEG channel to detect on (must have been resolved by
            each loader, i.e. included in its ``requested_channels``).
        include: Sleep stages (YASA integer codes) to detect within.
        skip_errors: If True, a failing subject is logged and stored as ``None``
            instead of aborting the whole batch.
        verbose: Print a one-line per-subject detection count.
        **detect_kwargs: Forwarded to :func:`spindles_detect`.
    """
    results: "dict[str, SpindlesResult]" = {}
    for loader in loaders:
        sid = str(getattr(loader, "subject_id", loader))
        try:
            res = spindles_detect(
                loader, ch_names=channel, include=include, **detect_kwargs
            )
        except Exception as exc:  # noqa: BLE001 - one bad subject must not abort the batch
            if not skip_errors:
                raise
            logger.warning("Subject %s failed; skipping: %s", sid, exc)
            res = None
        if verbose:
            n = 0 if res is None else len(res.summary())
            print(f"Subject {sid}: {n} spindle(s) detected on {channel}.")
        results[sid] = res
    return results


def _per_subject_event_means(
    results: Mapping[str, "SpindlesResult"], aggfunc: str
) -> "dict[str, pd.Series]":
    """Collapse each subject's events to one mean row of numeric features.

    Uses YASA's grouped ``summary`` (which adds ``Count`` and, when staged,
    ``Density``) and averages across any channel/stage groups so each subject
    contributes a single row. Falls back to stage-agnostic grouping for results
    detected without a hypnogram.
    """
    per_subject: "dict[str, pd.Series]" = {}
    for sid, res in results.items():
        if res is None:
            continue
        try:
            grouped = res.summary(grp_chan=True, grp_stage=True, aggfunc=aggfunc)
        except (KeyError, ValueError):
            grouped = res.summary(grp_chan=True, grp_stage=False, aggfunc=aggfunc)
        per_subject[sid] = grouped.select_dtypes("number").mean()
    return per_subject


def aggregate_spindle_summaries(
    results: Mapping[str, "SpindlesResult"],
    *,
    aggfunc: str = "mean",
) -> Tuple["pd.DataFrame", "pd.Series"]:
    """Per-subject event averages, plus their mean across subjects.

    Implements "average of all events detected in each subject, then mean those
    together across subjects": each subject's detected events are averaged into a
    single row (``Count``, ``Density``, ``Duration``, ``Amplitude``, ``Frequency``,
    ...), and the grand summary is the mean of those rows -- every subject weighted
    equally. Subjects with no spindles (``None``) are skipped.

    Returns:
        ``(per_subject_df, grand_mean)`` -- ``per_subject_df`` has one row per
        subject (index ``subject_id``); ``grand_mean`` is its column-wise mean
        across subjects.

    Raises:
        ValueError: if no subject had any detected spindles.
    """
    per_subject = _per_subject_event_means(results, aggfunc)
    if not per_subject:
        raise ValueError("No subject had any detected spindles to aggregate.")
    per_subject_df = pd.DataFrame(per_subject).T
    per_subject_df.index.name = "subject_id"
    grand_mean = per_subject_df.mean()
    grand_mean.name = f"mean_of_{len(per_subject_df)}_subjects"
    return per_subject_df, grand_mean


__all__ = [
    "spindles_detect",
    "detect_subjects_spindles",
    "aggregate_spindle_summaries",
    "NREM_STAGES",
    "DEFAULT_EPOCH_SEC",
    "DEFAULT_STAGE_MAP",
    "DEFAULT_EEG_CHANNELS",
]
