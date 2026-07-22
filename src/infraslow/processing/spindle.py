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


# Suggested EEG channels to load for spindle work (used by the subject_pipeline
# module; see infraslow.processing.subject_pipeline.CHANNELS).
DEFAULT_EEG_CHANNELS: Tuple[str, ...] = ("F3", "F4", "C3", "C4", "O1", "O2")


def spindle_rate_per_min(npz, channel: str) -> Tuple[float, float]:
    """``(rate, sem)`` spindles/min for ONE subject/channel, from
    ``{channel}__bouts__n_spindles``/``__start``/``__stop`` (as written by
    :func:`~infraslow.processing.subject_pipeline.calculate_channel_events`), or
    ``(NaN, NaN)`` if there are no bouts for this channel.

    ``rate`` pools all of this subject's bouts into a single rate: total
    spindles over total bout duration. This collapses all of a subject's bouts
    into one per-subject rate -- callers must average these per-subject rates
    across subjects (not pool raw per-bout counts across subjects), otherwise
    subjects with more/longer bouts would be over-weighted relative to subjects
    with fewer bouts.

    ``sem`` is the SEM, across this subject's own bouts, of each bout's own
    spindles/min rate: ``0.0`` with exactly one bout, ``NaN`` with none.
    """
    n_key, start_key, stop_key = f"{channel}__bouts__n_spindles", f"{channel}__bouts__start", f"{channel}__bouts__stop"
    if n_key not in npz.files or start_key not in npz.files or stop_key not in npz.files:
        return np.nan, np.nan
    n_spindles, start, stop = npz[n_key], npz[start_key], npz[stop_key]
    if n_spindles.size == 0:
        return np.nan, np.nan
    duration_sec = stop - start
    total_sec = float(duration_sec.sum())
    if total_sec <= 0:
        return np.nan, np.nan
    rate = float(n_spindles.sum()) / (total_sec / 60.0)

    duration_min = duration_sec / 60.0
    valid = duration_min > 0
    if not valid.any():
        return rate, np.nan
    per_bout_rate = n_spindles[valid] / duration_min[valid]
    sem = float(per_bout_rate.std(ddof=1) / np.sqrt(per_bout_rate.size)) if per_bout_rate.size > 1 else 0.0
    return rate, sem


__all__ = [
    "spindles_detect",
    "NREM_STAGES",
    "DEFAULT_EPOCH_SEC",
    "DEFAULT_STAGE_MAP",
    "DEFAULT_EEG_CHANNELS",
    "spindle_rate_per_min",
]
