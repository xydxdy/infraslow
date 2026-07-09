"""Sleep-spindle detection on loaded PSG signals using Luna (``lunapi``).

A drop-in alternative to :func:`infraslow.processing.detection.spindles_detect`
(which wraps YASA), using the wavelet-based ``SPINDLES`` command of the Luna
toolset via its Python interface ``lunapi``. Like the YASA adapter it reads
everything it needs off a
:class:`~infraslow.io.psg_loader.BioserenityPSGLoader` -- the EEG signal, its
sampling rate, and the per-epoch ``(timestamp, stage)`` hypnogram -- and
restricts detection to NREM sleep.

The two detectors are intentionally not identical: YASA uses a relative-power /
moving-correlation pipeline over a fixed ``freq_sp`` band, whereas Luna detects
with complex Morlet wavelets centred on one or more target frequencies
(``fc``). This module adapts the repo's data shapes to Luna's in-memory EDF API
and hands the actual detection to Luna; see
https://zzz.bwh.harvard.edu/luna/ref/spindles-so/ for the command reference.

Detection is delegated entirely to Luna; this module only

1. builds an in-memory EDF from the loader's ``(n_channels, n_samples)`` array,
2. turns the per-epoch hypnogram into a NREM annotation and masks to it, and
3. runs ``SPINDLES`` and collects the result tables into a small wrapper whose
   ``.summary()`` mirrors the parts of the YASA contract this repo relies on.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

# Reuse the YASA adapter's stage plumbing so both detectors share one stage
# vocabulary (labels -> integer codes) and one hypnogram-extraction contract.
from .detection import (
    DEFAULT_EPOCH_SEC,
    DEFAULT_STAGE_MAP,
    NREM_STAGES,
    _extract_epoch_stages,
    _stages_to_int,
)

logger = logging.getLogger(__name__)

# Defaults chosen to mirror ``infraslow.processing.detection.spindles_detect``
# (the YASA adapter) so the two functions are as interchangeable as possible.
# Each YASA-named parameter is translated to the matching Luna ``SPINDLES``
# option (see ``_build_spindles_command``); these constants document that
# correspondence and keep the two signatures aligned.
DEFAULT_FC: Tuple[float, float] = (13.5,) #FC value(s), e.g. (11, 15) to find 11 and 15 Hz spindles (default 13.5 Hz)
DEFAULT_FREQ_SP: Tuple[float, float] = (12.0, 15.0)  # sigma band -> Luna fc-lower/fc-upper sweep
DEFAULT_FC_STEP = 1  # Hz; wavelet-frequency increment across the band (Luna ``fc-step``)
DEFAULT_FREQ_BROAD: Tuple[float, float] = (1.0, 30.0)  # parity only; unused by Luna (see docstring)
DEFAULT_DURATION: Tuple[float, float] = (0.5, 2.0)  # (min, max) whole-spindle sec -> Luna min/max
DEFAULT_MIN_DISTANCE = 500.0  # ms; merge spindles closer than this -> Luna merge (sec)
DEFAULT_CYCLES = 7  # Morlet wavelet bandwidth, in cycles (Luna ``cycles``)
DEFAULT_TH = 4.5  # core spindle threshold (Luna ``th``)
DEFAULT_TH2 = 2.0  # flanking-region threshold (Luna ``th2``)
# Luna threshold options that may be passed via the ``thresh`` mapping.
_THRESH_KEYS = {"th": "th", "th2": "th2", "th_max": "th-max", "q": "q"}


# Annotation class used to carry the NREM mask into Luna.
_NREM_ANNOT = "SP_NREM"


# Luna per-event column -> YASA ``summary()`` column. Mapping these lets the
# YASA-oriented ``infraslow.viz`` helpers consume a Luna result unchanged.
_EVENT_COLMAP = {
    "START": "Start",
    "PEAK": "Peak",
    "STOP": "End",
    "DUR": "Duration",
    "AMP": "Amplitude",
    "FRQ": "Frequency",
    "NOSC": "Oscillations",
    "SYMM": "Symmetry",
    "CH": "Channel",
}

# Broadband range the signal is filtered to before time-locking spindles in
# :meth:`LunaSpindlesResult.get_sync_events` (matches the YASA tutorial's
# ``data_broad`` and :mod:`infraslow.viz.spindle_power`).
DEFAULT_FREQ_BROAD_SYNC: Tuple[float, float] = (1.0, 30.0)


class LunaSpindlesResult:
    """Holder for Luna ``SPINDLES`` output that mimics YASA's ``SpindlesResults``.

    Carries the raw Luna tables *and* the signal they were detected from, so it
    exposes the slice of YASA's :class:`~yasa.SpindlesResults` API that the rest
    of this repo (notably :mod:`infraslow.viz`) relies on -- without any YASA
    dependency or a separate adapter object:

    * :meth:`summary` -- the per-event table with YASA column names by default
      (so ``len(res.summary())`` is the spindle count and ``res.summary()`` plugs
      straight into the plotting helpers), or the raw Luna per-channel summary
      when ``grp_chan=True``.
    * :meth:`get_sync_events` -- spindle-locked waveforms in YASA's long format.

    The unmodified Luna tables remain on :attr:`events`, :attr:`per_channel` and
    :attr:`tables` for callers that want the native, frequency-stratified output.

    Attributes:
        events: One row per detected spindle (Luna ``CH``/``F``/``SPINDLE``
            strata, native column names), or ``None`` if Luna emitted none.
        per_channel: Per-channel summary (Luna ``CH``/``CH_F`` strata) -- counts,
            density, mean amplitude/duration/frequency, etc.
        tables: ``{strata_string: DataFrame}`` for every ``SPINDLES`` strata.
    """

    def __init__(
        self,
        events: Optional[pd.DataFrame],
        per_channel: Optional[pd.DataFrame],
        tables: "dict[str, pd.DataFrame]",
        *,
        data: Optional[np.ndarray] = None,
        sf: Optional[float] = None,
        ch_names: Optional[Sequence[str]] = None,
        loader_ch_names: Optional[Sequence[str]] = None,
        hypno: Optional[np.ndarray] = None,
        freq_broad: Tuple[float, float] = DEFAULT_FREQ_BROAD_SYNC,
    ) -> None:
        self.events = events
        self.per_channel = per_channel
        self.tables = tables
        # Signal context for waveform time-locking (set by ``spindles_detect_luna``).
        # ``_ch_names`` indexes the *detection* array ``_data`` (the channel subset
        # detection ran on); ``_loader_ch_names`` is the loader's full channel list,
        # used to report ``IdxChannel`` relative to the loader -- the two differ
        # whenever detection ran on a subset of the loaded channels.
        self._data = None if data is None else np.atleast_2d(np.asarray(data, dtype=float))
        self._sf = None if sf is None else float(sf)
        self._ch_names = list(ch_names) if ch_names is not None else None
        self._loader_ch_names = list(loader_ch_names) if loader_ch_names is not None else None
        self._hypno = None if hypno is None else np.asarray(hypno)
        self._freq_broad = freq_broad
        self._broad_cache: "dict[int, np.ndarray]" = {}

    # ------------------------------------------------------------------ #
    def _event_stage_codes(self) -> Optional[np.ndarray]:
        """Integer sleep-stage code at each spindle's start, or ``None``.

        Reads the per-sample hypnogram passed in by :func:`spindles_detect_luna`
        at the sample each spindle starts (Luna ``START``, in original elapsed
        seconds), matching YASA's per-event ``Stage`` (the stage at spindle onset).
        ``None`` when no hypnogram was available or the events lack a start time.
        """
        if (
            self._hypno is None
            or self._sf is None
            or self.events is None
            or "START" not in self.events.columns
        ):
            return None
        starts = pd.to_numeric(self.events["START"], errors="coerce").to_numpy(dtype=float)
        idx = np.clip(np.round(starts * float(self._sf)).astype(int), 0, self._hypno.size - 1)
        return self._hypno[idx]

    # ------------------------------------------------------------------ #
    def summary(self, grp_chan: bool = False, **_: Any) -> Optional[pd.DataFrame]:
        """Per-event table with YASA column names, or the per-channel summary.

        With ``grp_chan=False`` (default) the per-event table is relabelled from
        Luna's columns to YASA's (``START`` -> ``Start``, ``AMP`` -> ``Amplitude``,
        ``CH`` -> ``Channel``, ...) so YASA-oriented code finds the columns it
        expects; a ``Stage`` column (integer code at each spindle's onset) is
        appended when a hypnogram was available at detection time. With
        ``grp_chan=True`` the raw Luna per-channel summary is returned unchanged.
        Extra keyword arguments (``grp_stage``, ``aggfunc``, ...) are accepted and
        ignored for signature-compatibility with YASA.
        """
        if grp_chan:
            return self.per_channel
        if self.events is None:
            return None
        out: "dict[str, Any]" = {}
        for src, dst in _EVENT_COLMAP.items():
            if src in self.events.columns:
                col = self.events[src]
                out[dst] = (
                    col.astype(str)
                    if dst == "Channel"
                    else pd.to_numeric(col, errors="coerce")
                )
        df = pd.DataFrame(out)
        stages = self._event_stage_codes()
        if stages is not None and len(stages) == len(df):
            df["Stage"] = stages
        return df

    # ------------------------------------------------------------------ #
    def _broadband(self, idx: int) -> np.ndarray:
        """Broadband-filtered signal for channel ``idx``, cached across spindles."""
        cached = self._broad_cache.get(idx)
        if cached is None:
            from .signal import ButterFilter  # noqa: PLC0415 - lazy, avoids import cost

            cached = ButterFilter(self._sf, list(self._freq_broad), mode="band")(
                self._data[idx]
            )
            self._broad_cache[idx] = cached
        return cached

    def _lock_index(self, data_broad: np.ndarray, a: int, b: int, distance: int) -> int:
        """Index of YASA's "Peak" in ``[a, b)``: most prominent broadband peak.

        The detrended broadband signal's most prominent positive peak inside the
        spindle (falling back to the segment maximum). This is a signal-phase
        landmark -- unlike Luna's ``PEAK`` (the wavelet-power envelope maximum) --
        so events time-locked to it stay phase-aligned and average into a
        spindle-shaped waveform rather than a smooth blob.
        """
        from scipy.signal import find_peaks  # noqa: PLC0415 - lazy, optional dependency

        seg = data_broad[a:b]
        x = np.arange(seg.size, dtype=float)
        seg_det = seg - np.polyval(np.polyfit(x, seg, 1), x)
        pk, props = find_peaks(seg_det, distance=distance, prominence=(None, None))
        if pk.size == 0:
            return a + int(np.argmax(seg_det))
        return a + int(pk[props["prominences"].argmax()])

    # ------------------------------------------------------------------ #
    def get_sync_events(
        self,
        center: str = "Peak",
        time_before: float = 1.5,
        time_after: float = 1.5,
    ) -> pd.DataFrame:
        """Spindle-locked waveforms in YASA's long format.

        Time-locks each detected spindle and stacks the windows into one
        long-format table -- exactly what :func:`infraslow.viz.plot_spindles`
        groups over. ``center`` is accepted for signature-compatibility with
        YASA but each spindle is always locked to its dominant broadband peak
        (see :meth:`_lock_index`), which is what phase-aligns the oscillations.

        Returns a :class:`~pandas.DataFrame` with columns:

        * ``Event`` -- event number (one per spindle row in :attr:`events`).
        * ``Time`` -- time relative to the lock point, in seconds.
        * ``Amplitude`` -- broadband-filtered signal for the event, in µV.
        * ``Channel`` -- channel the spindle was detected on.
        * ``IdxChannel`` -- index of that channel in the **loader**'s
          ``channel_names`` (falls back to its position in the detection array
          when the loader's channel list is unavailable).
        * ``Stage`` -- sleep stage (integer code) the spindle occurred in, taken
          from ``summary()``'s ``Stage`` column; *only present when a hypnogram
          was available at detection time*.

        Requires the signal context populated by
        :func:`spindles_detect_luna`; returns an empty (correctly-columned)
        frame when there are no spindles or no signal is attached.
        """
        summary = self.summary()
        has_stage = summary is not None and "Stage" in summary.columns
        cols = ["Event", "Time", "Amplitude", "Channel", "IdxChannel"]
        if has_stage:
            cols.append("Stage")

        if summary is None or len(summary) == 0 or self._data is None:
            return pd.DataFrame(columns=cols)

        sf = float(self._sf)
        starts = summary["Start"].to_numpy(dtype=float)
        ends = summary["End"].to_numpy(dtype=float)
        stages = summary["Stage"].to_numpy() if has_stage else None
        channels = (
            summary["Channel"].astype(str).to_numpy()
            if "Channel" in summary.columns
            else None
        )
        ch_names = self._ch_names or []
        # Row of the channel in the detection array ``_data`` (for signal lookup)...
        data_index = {str(c): i for i, c in enumerate(ch_names)}

        n_before, n_after = int(round(time_before * sf)), int(round(time_after * sf))
        rel_t = np.arange(-n_before, n_after + 1) / sf
        distance = max(1, int(round(60 * sf / 1000)))  # YASA: ~60 ms min peak spacing

        frames = []
        for ev, (s, e) in enumerate(zip(starts, ends)):
            ch = channels[ev] if channels is not None else (ch_names[0] if ch_names else "0")
            idx = data_index.get(str(ch), 0)  # row in _data (for broadband/lock)
            data_broad = self._broadband(idx)
            a, b = int(round(s * sf)), int(round(e * sf))
            a, b = max(a, 0), min(b, data_broad.size)
            if b - a < 3:
                continue
            lock = self._lock_index(data_broad, a, b, distance)
            lo, hi = lock - n_before, lock + n_after + 1
            if lo < 0 or hi > data_broad.size:
                continue
            row = {
                "Event": ev,
                "Time": rel_t,
                "Amplitude": data_broad[lo:hi],
                "Channel": ch,
            }
            if has_stage:
                row["Stage"] = int(stages[ev])
            frames.append(pd.DataFrame(row))
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=cols)

    def __repr__(self) -> str:  # pragma: no cover - display only
        n = 0 if self.events is None else len(self.events)
        return f"<LunaSpindlesResult: {n} spindle(s)>"


# A Luna detection result, or ``None`` when no spindle is found.
SpindlesResult = Optional[LunaSpindlesResult]


def _record_layout(sf: float) -> Tuple[int, int]:
    """Choose an EDF (record-duration-sec, samples-per-record) pair for ``sf``.

    Luna stores signals in fixed-size records; an inserted signal must contain
    exactly ``n_records * record_sec * sr`` samples, and ``record_sec * sr`` must
    be a whole number of samples. We pick the smallest integer record length
    (1-10 s) for which ``sf`` yields an integer sample count, falling back to a
    1 s record at the rounded rate when ``sf`` is too irregular.
    """
    for rs in range(1, 11):
        spr = sf * rs
        if abs(spr - round(spr)) < 1e-6:
            return rs, int(round(spr))
    return 1, int(round(sf))


def _select_channels(
    loader: Any, ch_names: Optional[Union[str, Sequence[str]]]
) -> Tuple[np.ndarray, List[str]]:
    """Resolve ``ch_names`` to a ``(n_channels, n_samples)`` array and label list.

    ``None`` uses every loaded channel, a ``str`` picks one named channel, and a
    sequence stacks the named subset -- mirroring the YASA adapter's selection.
    Luna requires explicit channel labels, so a label list is always returned.
    """
    if ch_names is None:
        arr = np.atleast_2d(np.asarray(loader.data, dtype=float))
        loaded = getattr(loader, "channel_names", None)
        if not loaded:
            raise ValueError(
                "spindles_detect_luna needs channel labels; the loader exposes "
                "no 'channel_names'. Pass ch_names explicitly."
            )
        ch_list = list(loaded)
    elif isinstance(ch_names, str):
        arr = np.atleast_2d(np.asarray(loader.get_channel(ch_names), dtype=float))
        ch_list = [ch_names]
    else:
        ch_list = list(ch_names)
        if not ch_list:
            raise ValueError("spindles_detect_luna: `ch_names` list is empty.")
        arr = np.vstack(
            [np.asarray(loader.get_channel(c), dtype=float) for c in ch_list]
        )
    if arr.size == 0:
        raise ValueError("spindles_detect_luna received empty data from the loader.")
    if arr.shape[0] != len(ch_list):
        raise ValueError(
            f"Channel count mismatch: {arr.shape[0]} signal row(s) vs "
            f"{len(ch_list)} label(s) {ch_list}."
        )
    return arr, ch_list


def _nrem_intervals(
    hypno: Any,
    *,
    include_codes: "set[int]",
    epoch_sec: float,
    total_sec: float,
    stage_map,
    stage_column: str,
) -> List[List[float]]:
    """Build merged ``[start_sec, stop_sec]`` intervals for the kept stages.

    Each epoch ``i`` spans ``[i*epoch_sec, (i+1)*epoch_sec)``; epochs whose
    integer stage code is in ``include_codes`` are kept, contiguous kept epochs
    are merged into one interval, and everything is clipped to ``total_sec`` (the
    in-memory EDF's true length) so no annotation runs off the end of the data.
    """
    stages = _extract_epoch_stages(hypno, stage_column=stage_column)
    codes = _stages_to_int(stages, stage_map)

    intervals: List[List[float]] = []
    for i, code in enumerate(codes):
        if int(code) not in include_codes:
            continue
        start = i * epoch_sec
        stop = min((i + 1) * epoch_sec, total_sec)
        if stop <= start:
            continue
        # Extend the previous interval when this epoch abuts it, else open a new one.
        if intervals and abs(intervals[-1][1] - start) < 1e-6:
            intervals[-1][1] = stop
        else:
            intervals.append([start, stop])
    return intervals


def _sample_stage_codes(
    hypno: Any,
    *,
    sf: float,
    n_samples: int,
    epoch_sec: float,
    stage_map,
    stage_column: str,
) -> np.ndarray:
    """Upsample a per-epoch hypnogram to one integer stage code per sample.

    Repeats each epoch's stage code across its samples and crops/pads to exactly
    ``n_samples`` so it stays aligned with the signal array (used to label the
    ``Stage`` column of :meth:`LunaSpindlesResult.get_sync_events`).
    """
    codes = _stages_to_int(
        _extract_epoch_stages(hypno, stage_column=stage_column), stage_map
    )
    samples_per_epoch = max(1, int(round(epoch_sec * sf)))
    upsampled = np.repeat(np.asarray(codes, dtype=int), samples_per_epoch)
    if upsampled.size < n_samples:
        pad_val = int(upsampled[-1]) if upsampled.size else -2  # -2 = Unscored
        upsampled = np.concatenate(
            [upsampled, np.full(n_samples - upsampled.size, pad_val, dtype=int)]
        )
    return upsampled[:n_samples]


def _include_to_codes(include: Iterable[Any], stage_map) -> "set[int]":
    """Normalise an ``include`` spec to a set of integer stage codes.

    Accepts YASA-style integer codes (``2, 3``) and/or string labels
    (``"N2", "N3"``), so the same call works whether a caller thinks in Luna
    labels or YASA codes.
    """
    codes: "set[int]" = set()
    for s in include:
        if isinstance(s, (int, np.integer)):
            codes.add(int(s))
            continue
        key = str(s).strip().lower()
        if key not in stage_map:
            raise ValueError(
                f"Unrecognised include stage {s!r}. Known labels: "
                f"{sorted(set(stage_map))}, or pass YASA integer codes."
            )
        codes.add(stage_map[key])
    return codes


def _fc_command_tokens(
    freq_sp: Union[float, Sequence[float]],
    fc_step: float,
    fc: Optional[Union[float, Sequence[float]]] = None,
) -> List[str]:
    """Build Luna's wavelet-frequency options, ``fc`` taking precedence.

    When ``fc`` is given it wins: one explicit target frequency (Luna ``fc``), or
    a comma-separated list of them, and ``freq_sp``/``fc_step`` are ignored.
    Otherwise the sigma *band* ``freq_sp=(low, high)`` is swept with a family of
    Morlet wavelets via ``fc-lower``/``fc-upper``/``fc-step`` -- detection runs at
    each target frequency, so the result tables are stratified by ``F`` (one
    spindle set per wavelet). A scalar ``freq_sp`` pins a single ``fc``. Per
    wavelet bandwidth is set by ``cycles``.
    """
    # `fc` overrides the band sweep entirely.
    if fc is not None:
        fcs = [fc] if isinstance(fc, (int, float)) else list(fc)
        if not fcs:
            raise ValueError("fc must be a frequency or a non-empty list of frequencies.")
        return ["fc=" + ",".join(str(float(f)) for f in fcs)]
    if isinstance(freq_sp, (int, float)):
        return [f"fc={float(freq_sp)}"]
    vals = list(freq_sp)
    if len(vals) != 2:
        raise ValueError(
            f"freq_sp must be a (low, high) pair or a single frequency; got {freq_sp!r}."
        )
    low, high = float(vals[0]), float(vals[1])
    if high < low:
        raise ValueError(f"freq_sp lower bound {low} exceeds upper bound {high}.")
    return [f"fc-lower={low}", f"fc-upper={high}", f"fc-step={float(fc_step)}"]


def _resolve_thresholds(thresh: Optional[Any]) -> "dict[str, float]":
    """Merge a YASA-style ``thresh`` mapping onto Luna's threshold defaults.

    Mirrors ``spindles_detect``'s ``thresh`` argument shape (``None`` or a
    mapping). For Luna the recognised keys are its wavelet thresholds --
    ``th`` (core), ``th2`` (flanking), ``th_max`` (core ceiling) and ``q``
    (quality criterion) -- rather than YASA's ``rel_pow``/``corr``/``rms``.
    """
    resolved: "dict[str, float]" = {"th": DEFAULT_TH, "th2": DEFAULT_TH2}
    if thresh:
        unknown = set(thresh) - set(_THRESH_KEYS)
        if unknown:
            raise ValueError(
                f"Unknown thresh key(s) {sorted(unknown)} for the Luna detector; "
                f"recognised keys are {sorted(_THRESH_KEYS)}."
            )
        resolved.update(thresh)
    return resolved


def _build_spindles_command(
    ch_list: Sequence[str],
    *,
    fc_tokens: Sequence[str],
    cycles: int,
    thresholds: "dict[str, float]",
    duration: Tuple[float, float],
    merge_sec: float,
    extra: str,
) -> str:
    """Assemble the Luna ``SPINDLES`` command string from translated parameters."""
    parts = [
        "SPINDLES",
        "sig=" + ",".join(ch_list),
        *fc_tokens,  # fc-lower/fc-upper/fc-step sweep, or a single fc
        f"cycles={cycles}",
        f"min={duration[0]}",
        f"max={duration[1]}",
        f"merge={merge_sec}",
        "per-spindle",  # emit the per-event (CH,F,SPINDLE) table
    ]
    # Threshold options, using Luna's exact option spelling (e.g. ``th-max``).
    for key, luna_opt in _THRESH_KEYS.items():
        if key in thresholds and thresholds[key] is not None:
            parts.append(f"{luna_opt}={thresholds[key]}")
    if extra:
        parts.append(extra)
    return " ".join(parts)


def _collect_tables(inst: Any, strata_df: Optional[pd.DataFrame]) -> "dict[str, pd.DataFrame]":
    """Pull every ``SPINDLES`` strata table out of the instance's result set."""
    tables: "dict[str, pd.DataFrame]" = {}
    if strata_df is None:
        return tables
    sp = strata_df[strata_df["Command"] == "SPINDLES"]
    for strata in sp["Strata"].tolist():
        tbl = inst.table("SPINDLES", strata)
        if tbl is not None:
            tables[strata] = tbl
    return tables


def _split_event_and_channel(
    tables: "dict[str, pd.DataFrame]",
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Pick the per-event and per-channel tables out of the collected strata.

    The per-event table is the strata whose factors include ``SPINDLE`` (e.g.
    ``CH_F_SPINDLE``); the per-channel summary is the non-event strata with the
    fewest factors that still includes ``CH`` -- ``CH`` itself if present, else
    the per-channel-per-frequency ``CH_F`` that Luna always emits when ``fc`` is
    set. Matching by factor content (rather than a hard-coded strata string)
    keeps this robust to how lunapi joins strata factor names into a key.
    """
    events: Optional[pd.DataFrame] = None
    summary_strata: Optional[str] = None
    for strata, tbl in tables.items():
        factors = set(strata.split("_"))
        if "SPINDLE" in factors:
            if events is None:
                events = tbl
            continue
        if "CH" not in factors:
            continue
        # Prefer the per-channel table with the fewest extra factors.
        if summary_strata is None or len(factors) < len(summary_strata.split("_")):
            summary_strata = strata
    per_channel = tables.get(summary_strata) if summary_strata else None
    return events, per_channel


def spindles_detect_luna(
    loader: Any,
    *,
    ch_names: Optional[Union[str, Sequence[str]]] = None,
    include: Iterable[Any] = NREM_STAGES,
    epoch_sec: float = DEFAULT_EPOCH_SEC,
    stage_map=DEFAULT_STAGE_MAP,
    stage_column: str = "stage",
    freq_sp: Union[float, Tuple[float, float]] = DEFAULT_FREQ_SP,
    freq_broad: Tuple[float, float] = DEFAULT_FREQ_BROAD,
    duration: Tuple[float, float] = DEFAULT_DURATION,
    min_distance: float = DEFAULT_MIN_DISTANCE,
    thresh: Optional[Any] = None,
    multi_only: bool = False,
    remove_outliers: bool = False,
    verbose: bool = False,
    fc: Optional[Union[float, Sequence[float]]] = DEFAULT_FC,
    cycles: int = DEFAULT_CYCLES,
    fc_step: float = DEFAULT_FC_STEP,
    extra_args: str = "",
    inst_id: str = "infraslow",
) -> SpindlesResult:
    """Detect sleep spindles on NREM epochs with Luna, from a loaded recording.

    The Luna-backed counterpart of
    :func:`infraslow.processing.detection.spindles_detect`, with a deliberately
    parallel signature so the two are as interchangeable as possible. It reads
    the EEG signal, sampling rate, and per-epoch ``(timestamp, stage)`` hypnogram
    off the loader, builds an in-memory EDF, masks to NREM sleep (when a
    hypnogram is present), and runs Luna's wavelet ``SPINDLES`` detector.

    The shared (YASA-named) arguments are translated to the matching Luna
    ``SPINDLES`` options; because the two algorithms differ (YASA: relative-power
    / moving-correlation over a band; Luna: Morlet wavelet centred on ``fc``),
    a few YASA options have no Luna equivalent and are noted below.

    Args:
        loader: A loaded (or loadable)
            :class:`~infraslow.io.psg_loader.BioserenityPSGLoader` -- anything
            exposing ``data``/``get_channel``, ``channel_names``, ``sf``,
            ``annotations`` and ``is_loaded``/``load()``. Loaded in place if not
            already.
        ch_names: Channel(s) to detect on. ``None`` runs on every loaded channel;
            a single canonical name (e.g. ``"C3"``) detects on that one; a
            sequence detects on just those.
        include: Sleep stages to detect within, as YASA integer codes (default
            NREM ``(2, 3)`` = N2+N3) and/or string labels (``"N2"``). Ignored
            when the loader carries no hypnogram.
        epoch_sec: Seconds per scored epoch, used both to set Luna's epoch length
            and to place the NREM annotation. Defaults to 30 s.
        stage_map: Case-insensitive label -> integer-code map for the string
            hypnogram. Defaults to this repo's Wake/N1/N2/N3/REM labels.
        stage_column: Stage column name in the loader's ``annotations`` DataFrame.
        freq_sp: Sigma band ``(low, high)`` in Hz (default ``(12, 15)``, as in
            ``spindles_detect``). Realised as a Morlet-wavelet sweep across the
            band via Luna ``fc-lower``/``fc-upper`` (step ``fc_step``); detection
            runs at each target frequency, so the result tables are stratified by
            frequency ``F`` (a spindle near a band edge may appear under more than
            one ``F``). Pass a scalar to pin a single wavelet (Luna ``fc``).
            Per-wavelet bandwidth is governed by ``cycles``. **Ignored when
            ``fc`` is given.**
        freq_broad: Broadband range ``(low, high)`` in Hz. Accepted for parity
            with ``spindles_detect`` but **not used**: Luna's wavelet thresholds
            against the signal's own coefficient distribution and has no separate
            broadband-normalisation stage.
        duration: ``(min, max)`` whole-spindle duration in seconds, mapped to
            Luna ``min``/``max`` (default ``(0.5, 2)``, matching ``spindles_detect``).
        min_distance: Merge two spindles closer than this many **milliseconds**
            (default 500), mapped to Luna ``merge`` in seconds.
        thresh: ``None`` (Luna defaults) or a mapping of Luna wavelet thresholds:
            ``th`` (core, default 4.5), ``th2`` (flanking, default 2),
            ``th_max`` (core ceiling) and ``q`` (quality criterion). Mirrors the
            ``thresh``-is-a-mapping shape of ``spindles_detect`` (whose keys --
            ``rel_pow``/``corr``/``rms`` -- are YASA-specific and do not apply).
        multi_only: Present for signature parity. Luna's ``SPINDLES`` detects each
            channel independently, so ``True`` raises :class:`NotImplementedError`
            (use Luna's merged-spindle/MSPINDLES workflow instead).
        remove_outliers: Present for signature parity (YASA uses an isolation
            forest). Luna has no such step, so ``True`` raises
            :class:`NotImplementedError` -- use ``thresh={"q": ...}`` to drop
            low-quality spindles instead.
        verbose: When False (default), silence Luna's console log.
        fc: Luna-specific. Explicit wavelet target frequency in Hz, or a list of
            them. When set it **takes precedence over ``freq_sp``**: detection
            uses exactly these centre frequencies (Luna ``fc``) and the
            ``freq_sp``/``fc_step`` band sweep is skipped. ``None`` (default)
            falls back to the ``freq_sp`` sweep.
        cycles: Luna-specific. Morlet wavelet bandwidth in cycles (default 7);
            more cycles -> sharper frequency, coarser time resolution.
        fc_step: Luna-specific. Wavelet-frequency increment (Hz) across the
            ``freq_sp`` band (Luna ``fc-step``, default 0.5); ignored when
            ``freq_sp`` is a scalar.
        extra_args: Luna-specific. Raw text appended to the ``SPINDLES`` command
            for any option not surfaced here (e.g. ``"so mag=2"`` for SO coupling).
        inst_id: Luna-specific. Identifier for the in-memory Luna instance.

    Returns:
        A :class:`LunaSpindlesResult` whose ``.summary()`` is the per-event table
        (YASA column names), ``.summary(grp_chan=True)`` the per-channel summary,
        and ``.get_sync_events()`` the spindle-locked waveforms -- or ``None``
        when no spindle is detected (matching ``spindles_detect``'s contract).

    Raises:
        ImportError: if ``lunapi`` is not installed.
        NotImplementedError: if ``multi_only`` or ``remove_outliers`` is True.
        ValueError: if the loader has no sampling rate or no data, a stage label
            is unrecognised, or ``thresh`` carries an unknown key.
    """
    if multi_only:
        raise NotImplementedError(
            "spindles_detect_luna: `multi_only` is unsupported -- Luna's SPINDLES "
            "detects each channel independently. Run per channel and intersect the "
            "events, or use Luna's MSPINDLES workflow via `extra_args`."
        )
    if remove_outliers:
        raise NotImplementedError(
            "spindles_detect_luna: `remove_outliers` is unsupported -- Luna has no "
            "isolation-forest step. Drop low-quality spindles with a quality "
            "threshold instead, e.g. thresh={'q': 0.0}."
        )
    try:
        import lunapi as lp
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "spindles_detect_luna requires the 'lunapi' package. Install it with "
            "`pip install lunapi` (in a Slurm job or interactive node, not the "
            "login node)."
        ) from exc

    # Load on demand so a freshly-built loader can be passed straight in.
    if hasattr(loader, "is_loaded") and not loader.is_loaded:
        loader.load()

    sf = getattr(loader, "sf", None)
    if sf is None:
        raise ValueError(
            "spindles_detect_luna requires the loader's sampling frequency 'sf' "
            "(Hz); load the recording first."
        )
    sf = float(sf)

    arr, ch_list = _select_channels(loader, ch_names)

    # Translate the YASA-named arguments to Luna's units/options.
    # `fc`, when given, overrides the `freq_sp` band sweep.
    fc_tokens = _fc_command_tokens(freq_sp, fc_step, fc=fc)
    thresholds = _resolve_thresholds(thresh)
    merge_sec = float(min_distance) / 1000.0  # YASA min_distance is in ms; Luna merge in s
    # `freq_broad` is not used by Luna's wavelet *detector* (it thresholds against
    # the signal's own coefficient distribution); it is the broadband range the
    # result's `get_sync_events` filters to before time-locking spindles.

    # Lay the data out into whole EDF records and trim the (sub-record) tail so
    # every channel has exactly n_records * samples_per_record samples.
    rs, spr = _record_layout(sf)
    n_records = arr.shape[1] // spr
    if n_records < 1:
        raise ValueError(
            f"spindles_detect_luna: only {arr.shape[1]} sample(s) at {sf} Hz -- "
            f"too short for one {rs}s record."
        )
    n_keep = n_records * spr
    arr = arr[:, :n_keep]
    total_sec = float(n_records * rs)

    # Build the in-memory EDF and insert each selected channel.
    proj = lp.proj(verbose=verbose)
    proj.silence(not verbose)
    inst = proj.empty_inst(inst_id, n_records, rs)
    for label, row in zip(ch_list, arr):
        inst.insert_signal(label, np.ascontiguousarray(row, dtype=np.float64), spr)

    # Epoch, then restrict to NREM via an annotation mask when staging exists.
    inst.epoch(f"len={epoch_sec}")
    hypno = getattr(loader, "annotations", None)
    sample_hypno: Optional[np.ndarray] = None
    if hypno is not None:
        include_codes = _include_to_codes(include, stage_map)
        intervals = _nrem_intervals(
            hypno,
            include_codes=include_codes,
            epoch_sec=epoch_sec,
            total_sec=total_sec,
            stage_map=stage_map,
            stage_column=stage_column,
        )
        if not intervals:
            logger.info(
                "No epochs in stages %s; nothing to detect on.", sorted(include_codes)
            )
            return None
        inst.insert_annot(_NREM_ANNOT, intervals)
        inst.mask(f"ifnot={_NREM_ANNOT}")  # MASK ifnot=... ; RE (restructure)
        # Per-sample stage codes for get_sync_events' Stage column (original time
        # base -- Luna reports spindle times in original elapsed seconds).
        sample_hypno = _sample_stage_codes(
            hypno,
            sf=sf,
            n_samples=arr.shape[1],
            epoch_sec=epoch_sec,
            stage_map=stage_map,
            stage_column=stage_column,
        )

    # Run Luna's wavelet spindle detector and collect its result tables.
    cmd = _build_spindles_command(
        ch_list,
        fc_tokens=fc_tokens,
        cycles=cycles,
        thresholds=thresholds,
        duration=duration,
        merge_sec=merge_sec,
        extra=extra_args,
    )
    strata_df = inst.eval(cmd)
    tables = _collect_tables(inst, strata_df)
    events, per_channel = _split_event_and_channel(tables)

    if events is None or len(events) == 0:
        logger.info("No spindles detected for stages %s.", list(include))
        return None
    return LunaSpindlesResult(
        events=events,
        per_channel=per_channel,
        tables=tables,
        data=arr,
        sf=sf,
        ch_names=ch_list,
        loader_ch_names=getattr(loader, "channel_names", None),
        hypno=sample_hypno,
        freq_broad=freq_broad,
    )


__all__ = [
    "spindles_detect_luna",
    "LunaSpindlesResult",
    "DEFAULT_FREQ_SP",
    "DEFAULT_FC_STEP",
    "DEFAULT_FREQ_BROAD",
    "DEFAULT_DURATION",
    "DEFAULT_MIN_DISTANCE",
    "DEFAULT_CYCLES",
    "DEFAULT_TH",
    "DEFAULT_TH2",
]
