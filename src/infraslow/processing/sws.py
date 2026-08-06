"""Slow-wave detection on loaded PSG signals using YASA.

This module wraps :func:`yasa.sw_detect <https://raphaelvallat.com/yasa/>`_ the
same way :mod:`~infraslow.processing.spindle` wraps ``yasa.spindles_detect``: it
accepts the ``(n_channels, n_samples)`` array produced by
:class:`~infraslow.io.psg_loader.BioserenityPSGLoader` and the per-epoch
``(timestamp, stage)`` hypnogram produced by
:func:`~infraslow.io.hypnodensity.hypnodensity_to_annotations`, and restricts
detection to NREM sleep by default. Hypnogram handling
(``_build_sample_hypno`` and friends) is shared with
:mod:`~infraslow.processing.spindle` rather than duplicated here.

Unlike ``spindle.py``, the detection algorithm itself is **not** delegated to
``yasa.sw_detect`` directly: :func:`_sw_detect_vendored` below is a vendored
copy of it (reproduced from
`yasa==0.7.0's yasa/detection.py:1442-1980
<https://github.com/raphaelvallat/yasa/blob/master/src/yasa/detection.py#L1458-L1995>`_,
the range for this version's public docs). The only change from upstream is
that the bandpass filter's transition bandwidth -- hardcoded to 0.2 Hz in
``yasa.sw_detect`` and not exposed as a parameter -- is threaded through as
``l_trans_bandwidth``/``h_trans_bandwidth``. YASA's own hardcoded 0.2 Hz means
any ``freq_sw`` lower edge below ~0.2 Hz pushes the filter's stop-band edge
negative and MNE raises ("Filter specification invalid: Lower stop frequency
negative"), which previously forced this module to round this repo's stated
protocol band (0.1-4 Hz) up to (0.2, 4.0). With the transition bandwidth now
narrowed automatically for low ``freq_sw`` edges (see
``_auto_l_trans_bandwidth``), ``DEFAULT_FREQ_SW`` can be the protocol's actual
(0.1, 4.0). Everything else -- peak finding, zero-crossing duration checks,
amplitude thresholds, coupling, outlier removal -- is unchanged from upstream
and still calls YASA's own private helpers (``_check_data_hypno``,
``SWResults``, ``_zerocrossings``, ``get_centered_indices``) so it stays in
sync with however YASA itself preprocesses data and builds results.

.. warning::
    Because this vendors YASA internals rather than calling its public API,
    upgrading the installed ``yasa`` version can silently drift this module
    out of sync (e.g. if a future YASA release changes ``sw_detect``'s
    algorithm or its private helpers' signatures). Re-diff
    :func:`_sw_detect_vendored` against the installed ``yasa.detection.sw_detect``
    after any ``yasa`` upgrade.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from typing import Any, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from infraslow.processing.spindle import (
    DEFAULT_EPOCH_SEC,
    DEFAULT_STAGE_MAP,
    NREM_STAGES,
    _build_sample_hypno,
)

from ..constants import (
    DEFAULT_AMP_NEG,
    DEFAULT_AMP_POS,
    DEFAULT_AMP_PTP,
    DEFAULT_DUR_NEG,
    DEFAULT_DUR_POS,
    DEFAULT_FREQ_SW,
    DEFAULT_H_TRANS_BANDWIDTH,
)

logger = logging.getLogger(__name__)

# A YASA detection result, or ``None`` when no slow wave is found.
SWResult = Optional[Any]


def _auto_l_trans_bandwidth(freq_lo: float) -> float:
    """Safe lower transition bandwidth for a given ``freq_sw[0]``.

    Reproduces YASA's own hardcoded 0.2 Hz whenever that stays safe (stop-band
    edge ``freq_lo - l_trans_bandwidth >= 0.2 * freq_lo``), and shrinks
    proportionally below that so the stop-band edge never goes negative --
    e.g. 0.08 Hz at this module's own ``DEFAULT_FREQ_SW[0]`` of 0.1 Hz (stop
    edge 0.02 Hz).
    """
    return min(0.2, freq_lo * 0.8)


def _sw_detect_vendored(
    data: np.ndarray,
    sf: float,
    ch_names: Optional[Sequence[str]] = None,
    hypno: Optional[np.ndarray] = None,
    include: Any = (2, 3),
    freq_sw: Tuple[float, float] = (0.3, 1.5),
    l_trans_bandwidth: Optional[float] = None,
    h_trans_bandwidth: float = DEFAULT_H_TRANS_BANDWIDTH,
    dur_neg: Tuple[float, float] = (0.3, 1.5),
    dur_pos: Tuple[float, float] = (0.1, 1.0),
    amp_neg: Tuple[float, float] = (40.0, 200.0),
    amp_pos: Tuple[float, float] = (10.0, 150.0),
    amp_ptp: Tuple[float, float] = (75.0, 350.0),
    coupling: bool = False,
    coupling_params: Mapping[str, Any] = {"freq_sp": (12, 16), "time": 1, "p": 0.05},
    remove_outliers: bool = False,
    verbose: bool = False,
) -> SWResult:
    """Vendored ``yasa.detection.sw_detect`` (yasa==0.7.0), parameterized filter.

    A line-for-line reproduction of YASA's slow-wave detection algorithm --
    see the module docstring for why this exists instead of calling
    ``yasa.sw_detect`` directly. ``data``/``sf``/``ch_names``/``hypno`` are
    already-validated array_like inputs (this module's public :func:`sw_detect`
    does the loader adaptation); everything downstream, including docstrings
    for the detection parameters, matches ``yasa.sw_detect`` exactly.
    """
    try:
        from mne.filter import filter_data
        from scipy import signal
        from scipy.fftpack import next_fast_len
        from sklearn.ensemble import IsolationForest
        from yasa.detection import SWResults, _check_data_hypno
        from yasa.io import is_tensorpac_installed, set_log_level
        from yasa.others import _zerocrossings, get_centered_indices
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "sw_detect requires the 'yasa' package (and its own dependencies, "
            "incl. mne, scipy, scikit-learn). Install it with `pip install "
            "yasa` (in a Slurm job or interactive node, not the login node)."
        ) from exc

    if l_trans_bandwidth is None:
        l_trans_bandwidth = _auto_l_trans_bandwidth(freq_sw[0])

    set_log_level(verbose)

    (data, sf, ch_names, hypno, include, mask, n_chan, n_samples, bad_chan) = _check_data_hypno(
        data, sf, ch_names, hypno, include, verbose=verbose
    )

    # If all channels are bad
    if sum(bad_chan) == n_chan:
        logger.warning("All channels have bad amplitude. Returning None.")
        return None

    # Define time vector
    times = np.arange(data.size) / sf
    idx_mask = np.where(mask)[0]

    # Bandpass filter
    nfast = next_fast_len(n_samples)
    data_filt = filter_data(
        data,
        sf,
        freq_sw[0],
        freq_sw[1],
        method="fir",
        verbose=0,
        l_trans_bandwidth=l_trans_bandwidth,
        h_trans_bandwidth=h_trans_bandwidth,
    )

    # Extract the spindles-related sigma signal for coupling
    if coupling:
        is_tensorpac_installed()
        import tensorpac.methods as tpm

        # The width of the transition band is set to 1.5 Hz on each side,
        # meaning that for freq_sp = (12, 15 Hz), the -6 dB points are located
        # at 11.25 and 15.75 Hz. The frequency band for the amplitude signal
        # must be large enough to fit the sidebands caused by the assumed
        # modulating lower frequency band (Aru et al. 2015).
        # https://doi.org/10.1016/j.conb.2014.08.002
        assert isinstance(coupling_params, dict)
        assert "freq_sp" in coupling_params.keys()
        assert "time" in coupling_params.keys()
        assert "p" in coupling_params.keys()
        freq_sp = coupling_params["freq_sp"]
        data_sp = filter_data(
            data,
            sf,
            freq_sp[0],
            freq_sp[1],
            method="fir",
            l_trans_bandwidth=1.5,
            h_trans_bandwidth=1.5,
            verbose=0,
        )
        # Now extract the instantaneous phase/amplitude using Hilbert transform
        sw_pha = np.angle(signal.hilbert(data_filt, N=nfast)[:, :n_samples])
        sp_amp = np.abs(signal.hilbert(data_sp, N=nfast)[:, :n_samples])

    # Initialize empty output dataframe
    df = pd.DataFrame()

    for i in range(n_chan):
        # ####################################################################
        # START SINGLE CHANNEL DETECTION
        # ####################################################################
        # First, skip channels with bad data amplitude
        if bad_chan[i]:
            continue

        # Find peaks in data
        # Negative peaks with value comprised between -40 to -300 uV
        idx_neg_peaks, _ = signal.find_peaks(-1 * data_filt[i, :], height=amp_neg)
        # Positive peaks with values comprised between 10 to 200 uV
        idx_pos_peaks, _ = signal.find_peaks(data_filt[i, :], height=amp_pos)
        # Intersect with sleep stage vector
        idx_neg_peaks = np.intersect1d(idx_neg_peaks, idx_mask, assume_unique=True)
        idx_pos_peaks = np.intersect1d(idx_pos_peaks, idx_mask, assume_unique=True)

        # If no peaks are detected, return None
        if len(idx_neg_peaks) == 0 or len(idx_pos_peaks) == 0:
            logger.warning("No SW were found in channel %s.", ch_names[i])
            continue

        # Make sure that the last detected peak is a positive one
        if idx_pos_peaks[-1] < idx_neg_peaks[-1]:
            # If not, append a fake positive peak one sample after the last neg
            idx_pos_peaks = np.append(idx_pos_peaks, idx_neg_peaks[-1] + 1)

        # For each negative peak, we find the closest following positive peak
        pk_sorted = np.searchsorted(idx_pos_peaks, idx_neg_peaks)
        closest_pos_peaks = idx_pos_peaks[pk_sorted] - idx_neg_peaks
        closest_pos_peaks = closest_pos_peaks[np.nonzero(closest_pos_peaks)]
        idx_pos_peaks = idx_neg_peaks + closest_pos_peaks

        # Now we compute the PTP amplitude and keep only the good peaks
        sw_ptp = np.abs(data_filt[i, idx_neg_peaks]) + data_filt[i, idx_pos_peaks]
        sw_pt = np.abs(data_filt[i, idx_neg_peaks])
        good_ptp = np.logical_and(sw_ptp > amp_ptp[0], sw_ptp < amp_ptp[1])

        # If good_ptp is all False
        if all(~good_ptp):
            logger.warning("No SW were found in channel %s.", ch_names[i])
            continue

        sw_ptp = sw_ptp[good_ptp]
        sw_pt = sw_pt[good_ptp]
        idx_neg_peaks = idx_neg_peaks[good_ptp]
        idx_pos_peaks = idx_pos_peaks[good_ptp]

        # Now we need to check the negative and positive phase duration
        # For that we need to compute the zero crossings of the filtered signal
        zero_crossings = _zerocrossings(data_filt[i, :])
        # Make sure that there is a zero-crossing after the last detected peak
        if zero_crossings[-1] < max(idx_pos_peaks[-1], idx_neg_peaks[-1]):
            # If not, append the index of the last peak
            zero_crossings = np.append(zero_crossings, max(idx_pos_peaks[-1], idx_neg_peaks[-1]))

        # Find distance to previous and following zc
        neg_sorted = np.searchsorted(zero_crossings, idx_neg_peaks)
        previous_neg_zc = zero_crossings[neg_sorted - 1] - idx_neg_peaks
        following_neg_zc = zero_crossings[neg_sorted] - idx_neg_peaks

        # Distance between the positive peaks and the previous and
        # following zero-crossings
        pos_sorted = np.searchsorted(zero_crossings, idx_pos_peaks)
        previous_pos_zc = zero_crossings[pos_sorted - 1] - idx_pos_peaks
        following_pos_zc = zero_crossings[pos_sorted] - idx_pos_peaks

        # Duration of the negative and positive phases, in seconds
        neg_phase_dur = (np.abs(previous_neg_zc) + following_neg_zc) / sf
        pos_phase_dur = (np.abs(previous_pos_zc) + following_pos_zc) / sf

        # We now compute a set of metrics
        sw_start = times[idx_neg_peaks + previous_neg_zc]
        sw_end = times[idx_pos_peaks + following_pos_zc]
        # This should be the same as `sw_dur = pos_phase_dur + neg_phase_dur`
        # We round to avoid floating point errr (e.g. 1.9000000002)
        sw_dur = (sw_end - sw_start).round(4)
        sw_dur_both_phase = (pos_phase_dur + neg_phase_dur).round(4)
        sw_midcrossing = times[idx_neg_peaks + following_neg_zc]
        sw_idx_neg = times[idx_neg_peaks]  # Location of negative peak
        sw_idx_pos = times[idx_pos_peaks]  # Location of positive peak
        # Slope between peak trough and midcrossing.
        sw_slope = sw_pt / (sw_midcrossing - sw_idx_neg)
        # Hypnogram
        if hypno is not None:
            sw_sta = hypno[idx_neg_peaks]
        else:
            sw_sta = np.zeros(sw_dur.shape)

        # And we apply a set of thresholds to remove bad slow waves
        good_sw = np.logical_and.reduce(
            (
                # Data edges
                previous_neg_zc != 0,
                following_neg_zc != 0,
                previous_pos_zc != 0,
                following_pos_zc != 0,
                # Duration criteria
                sw_dur == sw_dur_both_phase,  # dur = negative + positive
                sw_dur <= dur_neg[1] + dur_pos[1],  # dur < max(neg) + max(pos)
                sw_dur >= dur_neg[0] + dur_pos[0],  # dur > min(neg) + min(pos)
                neg_phase_dur > dur_neg[0],
                neg_phase_dur < dur_neg[1],
                pos_phase_dur > dur_pos[0],
                pos_phase_dur < dur_pos[1],
                # Sanity checks
                sw_midcrossing > sw_start,
                sw_midcrossing < sw_end,
                sw_slope > 0,
            )
        )

        if all(~good_sw):
            logger.warning("No SW were found in channel %s.", ch_names[i])
            continue

        # Filter good events
        idx_neg_peaks = idx_neg_peaks[good_sw]
        idx_pos_peaks = idx_pos_peaks[good_sw]
        sw_start = sw_start[good_sw]
        sw_idx_neg = sw_idx_neg[good_sw]
        sw_midcrossing = sw_midcrossing[good_sw]
        sw_idx_pos = sw_idx_pos[good_sw]
        sw_end = sw_end[good_sw]
        sw_dur = sw_dur[good_sw]
        sw_ptp = sw_ptp[good_sw]
        sw_slope = sw_slope[good_sw]
        sw_sta = sw_sta[good_sw]

        # Create a dictionnary
        sw_params = OrderedDict(
            {
                "Start": sw_start,
                "NegPeak": sw_idx_neg,
                "MidCrossing": sw_midcrossing,
                "PosPeak": sw_idx_pos,
                "End": sw_end,
                "Duration": sw_dur,
                "ValNegPeak": data_filt[i, idx_neg_peaks],
                "ValPosPeak": data_filt[i, idx_pos_peaks],
                "PTP": sw_ptp,
                "Slope": sw_slope,
                "Frequency": 1 / sw_dur,
                "Stage": sw_sta,
            }
        )

        # Add phase (in radians) of slow-oscillation signal at maximum
        # spindles-related sigma amplitude within a XX-seconds centered epochs.
        if coupling:
            # Get phase and amplitude for each centered epoch
            time_before = time_after = coupling_params["time"]
            assert float(sf * time_before).is_integer(), (
                "Invalid time parameter for coupling. Must be a whole number of samples."
            )
            bef = int(sf * time_before)
            aft = int(sf * time_after)
            # Center of each epoch is defined as the negative peak of the SW
            n_peaks = idx_neg_peaks.shape[0]
            # idx.shape = (len(idx_valid), bef + aft + 1)
            idx, idx_valid = get_centered_indices(data[i, :], idx_neg_peaks, bef, aft)
            sw_pha_ev = sw_pha[i, idx]
            sp_amp_ev = sp_amp[i, idx]
            # 1) Find location of max sigma amplitude in epoch
            idx_max_amp = sp_amp_ev.argmax(axis=1)
            # Now we need to append it back to the original unmasked shape
            # to avoid error when idx.shape[0] != idx_valid.shape, i.e.
            # some epochs were out of data bounds.
            sw_params["SigmaPeak"] = np.ones(n_peaks) * np.nan
            # Timestamp at sigma peak, expressed in seconds from negative peak
            # e.g. -0.39, 0.5, 1, 2 -- limits are [time_before, time_after]
            time_sigpk = (idx_max_amp - bef) / sf
            # convert to absolute time from beginning of the recording
            # time_sigpk only includes valid epoch
            time_sigpk_abs = sw_idx_neg[idx_valid] + time_sigpk
            sw_params["SigmaPeak"][idx_valid] = time_sigpk_abs
            # 2) PhaseAtSigmaPeak
            # Find SW phase at max sigma amplitude in epoch
            pha_at_max = np.squeeze(np.take_along_axis(sw_pha_ev, idx_max_amp[..., None], axis=1))
            sw_params["PhaseAtSigmaPeak"] = np.ones(n_peaks) * np.nan
            sw_params["PhaseAtSigmaPeak"][idx_valid] = pha_at_max
            # 3) Normalized Direct PAC, with thresholding
            # Unreliable values are set to 0
            ndp = np.squeeze(
                tpm.norm_direct_pac(
                    sw_pha_ev[None, ...], sp_amp_ev[None, ...], p=coupling_params["p"]
                )
            )
            sw_params["ndPAC"] = np.ones(n_peaks) * np.nan
            sw_params["ndPAC"][idx_valid] = ndp
            # Make sure that Stage is the last column of the dataframe
            sw_params.move_to_end("Stage")

        # Convert to dataframe, keeping only good events
        df_chan = pd.DataFrame(sw_params)

        # Remove all duplicates
        df_chan = df_chan.drop_duplicates(subset=["Start"], keep=False)
        df_chan = df_chan.drop_duplicates(subset=["End"], keep=False)

        # We need at least 50 detected slow waves to apply the Isolation Forest
        if remove_outliers and df_chan.shape[0] >= 50:
            col_keep = ["Duration", "ValNegPeak", "ValPosPeak", "PTP", "Slope", "Frequency"]
            ilf = IsolationForest(
                contamination="auto", max_samples="auto", verbose=0, random_state=42
            )
            good = ilf.fit_predict(df_chan[col_keep])
            good[good == -1] = 0
            logger.info(
                "%i outliers were removed in channel %s." % ((good == 0).sum(), ch_names[i])
            )
            # Remove outliers from DataFrame
            df_chan = df_chan[good.astype(bool)]
            logger.info("%i slow-waves were found in channel %s." % (df_chan.shape[0], ch_names[i]))

        # ####################################################################
        # END SINGLE CHANNEL DETECTION
        # ####################################################################

        df_chan["Channel"] = ch_names[i]
        df_chan["IdxChannel"] = i
        df = pd.concat([df, df_chan], axis=0, ignore_index=True)

    # If no SW were detected, return None
    if df.empty:
        logger.warning("No SW were found in data. Returning None.")
        return None

    if hypno is None:
        df = df.drop(columns=["Stage"])
    else:
        df["Stage"] = df["Stage"].astype(int)

    return SWResults(
        events=df, data=data, sf=sf, ch_names=ch_names, hypno=hypno, data_filt=data_filt
    )


def sw_detect(
    loader: Any,
    *,
    ch_names: Optional[Union[str, Sequence[str]]] = None,
    include: Iterable[int] = NREM_STAGES,
    epoch_sec: float = DEFAULT_EPOCH_SEC,
    stage_map: Mapping[str, int] = DEFAULT_STAGE_MAP,
    stage_column: str = "stage",
    freq_sw: Tuple[float, float] = DEFAULT_FREQ_SW,
    l_trans_bandwidth: Optional[float] = None,
    h_trans_bandwidth: float = DEFAULT_H_TRANS_BANDWIDTH,
    dur_neg: Tuple[float, float] = DEFAULT_DUR_NEG,
    dur_pos: Tuple[float, float] = DEFAULT_DUR_POS,
    amp_neg: Tuple[float, float] = DEFAULT_AMP_NEG,
    amp_pos: Tuple[float, float] = DEFAULT_AMP_POS,
    amp_ptp: Tuple[float, float] = DEFAULT_AMP_PTP,
    coupling: bool = False,
    coupling_params: Optional[Mapping[str, Any]] = None,
    remove_outliers: bool = False,
    verbose: bool = False,
) -> SWResult:
    """Detect slow waves on NREM epochs with YASA, from a loaded recording.

    A thin adapter over :func:`_sw_detect_vendored` (this module's own copy of
    ``yasa.sw_detect``'s algorithm -- see the module docstring for why) that
    reads everything it needs off a
    :class:`~infraslow.io.psg_loader.BioserenityPSGLoader`: the EEG signal, its
    sampling rate, and the per-epoch ``(timestamp, stage)`` hypnogram. When the
    loader carries a hypnogram, detection is restricted to NREM sleep by
    default (same ``include=(2, 3)`` convention as
    :func:`~infraslow.processing.spindle.spindles_detect`).

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
        freq_sw: Slow-wave bandpass range, in Hz. Defaults to this module's
            ``DEFAULT_FREQ_SW`` (0.1-4 Hz, this repo's protocol).
        l_trans_bandwidth: FIR filter transition bandwidth below ``freq_sw[0]``,
            in Hz. ``None`` (default) picks a value that keeps the stop-band
            edge (``freq_sw[0] - l_trans_bandwidth``) non-negative, matching
            YASA's own hardcoded 0.2 Hz whenever that stays safe -- see
            :func:`_auto_l_trans_bandwidth`. YASA itself does not expose this
            as a parameter; it is only adjustable here because ``sw_detect``'s
            algorithm is vendored (see the module docstring).
        h_trans_bandwidth: FIR filter transition bandwidth above ``freq_sw[1]``,
            in Hz. Defaults to YASA's own hardcoded 0.2 Hz.
        dur_neg, dur_pos, amp_neg, amp_pos, amp_ptp, coupling, coupling_params,
        remove_outliers, verbose: Passed straight through to
            :func:`_sw_detect_vendored` (``coupling_params=None`` uses its own
            default). All default to this module's ``DEFAULT_*`` constants,
            which match YASA's own defaults.

    Returns:
        The :class:`yasa.SWResults` object (call ``.summary()`` for the
        per-event table, ``.summary(grp_chan=True, grp_stage=True)`` for
        per-channel/stage stats), or ``None`` when no slow wave is detected --
        matching YASA's own contract.

    Raises:
        ImportError: if YASA (or one of its own dependencies) is not installed.
        ValueError: if the loader has no sampling rate or no data, or a stage
            label is unrecognised.
    """
    # Load on demand so a freshly-built loader can be passed straight in.
    if hasattr(loader, "is_loaded") and not loader.is_loaded:
        loader.load()

    sf = getattr(loader, "sf", None)
    if sf is None:
        raise ValueError(
            "sw_detect requires the loader's sampling frequency 'sf' (Hz); "
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
            raise ValueError("sw_detect: `ch_names` list is empty.")
        arr = np.vstack(
            [np.asarray(loader.get_channel(c), dtype=float) for c in ch_list]
        )
    if arr.size == 0:
        raise ValueError("sw_detect received empty data from the loader.")

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

    # Only restrict by stage when a hypnogram exists (matches yasa.sw_detect's
    # own behavior: ``include`` is otherwise ignored).
    detect_kwargs: dict = dict(
        data=arr,
        sf=sf,
        ch_names=ch_list,
        hypno=sample_hypno,
        freq_sw=freq_sw,
        l_trans_bandwidth=l_trans_bandwidth,
        h_trans_bandwidth=h_trans_bandwidth,
        dur_neg=dur_neg,
        dur_pos=dur_pos,
        amp_neg=amp_neg,
        amp_pos=amp_pos,
        amp_ptp=amp_ptp,
        coupling=coupling,
        remove_outliers=remove_outliers,
        verbose=verbose,
    )
    if sample_hypno is not None:
        detect_kwargs["include"] = tuple(include)
    if coupling_params is not None:
        detect_kwargs["coupling_params"] = dict(coupling_params)

    result = _sw_detect_vendored(**detect_kwargs)
    if result is None:
        logger.info("No slow waves detected for the requested stages %s.", tuple(include))
    return result


__all__ = [
    "sw_detect",
    "SWResult",
    "DEFAULT_FREQ_SW",
    "DEFAULT_DUR_NEG",
    "DEFAULT_DUR_POS",
    "DEFAULT_AMP_NEG",
    "DEFAULT_AMP_POS",
    "DEFAULT_AMP_PTP",
    "DEFAULT_H_TRANS_BANDWIDTH",
]
