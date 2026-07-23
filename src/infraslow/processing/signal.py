from __future__ import annotations

import logging
import types
from typing import Any, Callable, List, Mapping, Optional, Sequence

import mne
import numpy as np
import scipy.signal as signal

from .utils import is_nan

logger = logging.getLogger(__name__)

# Default common sampling rate (Hz) for resampling heterogeneous-rate channels.
DEFAULT_TARGET_SFREQ = 128.0

# Largest post-resample length mismatch (samples) tolerated across channels that
# should share one recording duration, before a warning is logged.
_LENGTH_MISMATCH_WARN = 4

class Pipeline:
    """Pipeline of signal processing
    Sequentially apply a list of signal transforms. Intermediate steps of the pipeline must be ‘transforms’, 
    that is, they must implement __call__ method. 

    Args:
        steps (List of signal transforms): implementing __call__ that are chained in sequential order. 

    Example:
            signal.Pipeline([
                NotchFilter(...),
                ButterFilter(...),
            ])
    """

    def __init__(self, steps):
        self.steps = steps

    def __call__(self, data):
        for s in self.steps:
            data = s(data)
        return data

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.steps:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string

class Resample:
    """Resample data to new sampling frequency in time or frequency domain using Fourier method along the given axis

    Args:
        orig_sfreq (int): original sampling frequency.
        new_sfreq (int): new sampling frequency.
        axis (int, optional): EEG time axis. Defaults to -1.
        domain (str, optional): A string indicating the domain of the input data. Defaults to "time": 
            `time` Consider the input data as time-domain, 
            `freq` Consider the input data as frequency-domain.
        verbose (bool, optional): If True, print information about resampling. Defaults to False.
    """
    def __init__(self, orig_sfreq, new_sfreq, axis=-1, domain="time", verbose=False):
        self.orig_sfreq = orig_sfreq
        self.new_sfreq = new_sfreq
        self.axis = axis
        self.domain = domain
        self.verbose = verbose

    def __call__(self, data):
        """built-in method to call Resample.

        Handles downsampling and upsampling (and is a no-op when the rates are
        equal): the Fourier method works in either direction, so a channel
        sampled below the target rate is interpolated up rather than rejected.

        Args:
            data (ndarray): EEG signal to be resampled
        """
        if self.orig_sfreq <= 0 or self.new_sfreq <= 0:
            raise ValueError(
                f"Sampling rates must be positive (orig={self.orig_sfreq}, "
                f"new={self.new_sfreq})."
            )
        if self.new_sfreq == self.orig_sfreq:
            return data
        if self.verbose:
            direction = "Downsample" if self.new_sfreq < self.orig_sfreq else "Upsample"
            print("{} data from {} Hz to {} Hz.".format(direction, self.orig_sfreq, self.new_sfreq))
        new_smp_point = int(round((data.shape[self.axis] / self.orig_sfreq) * self.new_sfreq))
        return signal.resample(data, new_smp_point, axis=self.axis, domain=self.domain)

class ButterFilter:
    """
    General Butterworth filter supporting low-pass, high-pass, and band-pass filtering.

    Parameters
    ----------
    sfreq : float
        Sampling frequency (Hz).
    cutoff : float or list of float
        Cutoff frequency/frequencies (Hz).
    mode : str
        'low', 'high', or 'band'.
    order : int, optional
        Filter order. Default = 2.
    axis : int, optional
        Axis along which to filter. Default = -1.
    filtering : str, optional
        Either 'filtfilt' (zero-phase) or 'lfilter' (causal). Default = 'filtfilt'.
    verbose : bool, optional
        If True, print information about filtering. Default = False.

    Example
    -------
    >>> low_sig  = ButterFilter(sfreq=256, cutoff=30, mode="low", order=4)(data)
    >>> high_sig = ButterFilter(sfreq=256, cutoff=0.5, mode="high", order=4)(data)
    >>> band_sig = ButterFilter(sfreq=256, cutoff=[8, 12], mode="band", order=4)(data)
    """

    def __init__(self, sfreq, cutoff, mode, order=2, axis=-1, filtering="filtfilt", verbose=False):
        self.sfreq = sfreq
        self.order = order
        self.cutoff = cutoff
        self.mode = mode
        self.axis = axis
        self.filtering = filtering
        self.verbose = verbose

    def _butter_coeff(self, cutoff, mode):
        nyq = 0.5 * self.sfreq
        if isinstance(cutoff, (int, float)):
            Wn = cutoff / nyq
        else:
            Wn = np.asarray(cutoff) / nyq

        if mode not in ("low", "high", "band"):
            raise ValueError("mode must be 'low', 'high', or 'band'")

        if mode == "band" and (not isinstance(Wn, (list, np.ndarray)) or len(Wn) != 2):
            raise ValueError("For band-pass, cutoff must be a sequence [low, high]")

        b, a = signal.butter(self.order, Wn, btype=mode)
        return b, a

    def _apply_filter(self, data, b, a):
        if isinstance(self.filtering, types.FunctionType):
          func = self.filtering
        else:
          func = getattr(signal, self.filtering)
        return func(b, a, data, axis=self.axis)

    def __call__(self, data):
        """
        Apply the Butterworth filter.

        Parameters
        ----------
        data : ndarray
            Input signal(s).

        Returns
        -------
        ndarray
            Filtered signal.
        """
        if self.verbose:
            if self.mode == "band":
                print(f"Applying {self.mode}-pass Butterworth filter (order={self.order}) with cutoff={self.cutoff} Hz at {self.sfreq} Hz sampling rate")
            else:
                print(f"Applying {self.mode}-pass Butterworth filter (order={self.order}) with cutoff={self.cutoff} Hz at {self.sfreq} Hz sampling rate")
        b, a = self._butter_coeff(self.cutoff, self.mode)
        return self._apply_filter(data, b, a)

class NotchFilter:
    """
    IIR Notch filter to remove a specific frequency (e.g., 50 Hz or 60 Hz powerline noise).

    Parameters
    ----------
    sfreq : float
        Sampling frequency (Hz).
    f0 : float
        Frequency to remove (Hz).
    Q : float, optional
        Quality factor. Higher Q means narrower notch. Default = 30.0.
    axis : int, optional
        Axis along which to filter. Default = -1.
    filtering : str, optional
        Either 'filtfilt' (zero-phase) or 'lfilter' (causal). Default = 'filtfilt'.
    verbose : bool, optional
        If True, print information about filtering. Default = False.

    Example
    -------
    >>> notch_sig = NotchFilter(sfreq=128, f0=50, Q=30)(data)
    """

    def __init__(self, sfreq, f0, Q=30.0, axis=-1, filtering="filtfilt", verbose=False):
        self.sfreq = sfreq
        self.f0 = f0
        self.Q = Q
        self.axis = axis
        self.filtering = filtering
        self.verbose = verbose

    def _notch_coeff(self):
        b, a = signal.iirnotch(self.f0, self.Q, self.sfreq)
        return b, a

    def _apply_filter(self, data, b, a):
        if isinstance(self.filtering, types.FunctionType):
            func = self.filtering
        else:
            func = getattr(signal, self.filtering)
        return func(b, a, data, axis=self.axis)

    def __call__(self, data):
        """
        Apply the notch filter.

        Parameters
        ----------
        data : ndarray
            Input signal(s).

        Returns
        -------
        ndarray
            Filtered signal.
        """
        if self.verbose:
            print(f"Applying notch filter at {self.f0} Hz (Q={self.Q}) at {self.sfreq} Hz sampling rate")
        b, a = self._notch_coeff()
        return self._apply_filter(data, b, a)


# --------------------------------------------------------------------------- #
# Resample a set of heterogeneous-rate channels onto one common rate
# --------------------------------------------------------------------------- #
def resample_channels(
    signals: Sequence[np.ndarray],
    source_rates: Sequence[float],
    new_sfreq: float = DEFAULT_TARGET_SFREQ,
) -> np.ndarray:
    """Resample each 1-D channel to ``new_sfreq`` and stack to ``(n_channels, n_samples)``.

    ``signals[i]`` is a 1-D array sampled at ``source_rates[i]``; channels may
    have different native rates. Each is resampled with the :class:`Resample`
    transform, then trimmed to the shortest resulting length so the output is
    rectangular. All channels of one recording share a duration, so the resampled
    lengths match up to rounding -- a larger gap is logged as a warning. Returns
    an empty ``(0, 0)`` array when given no channels.
    """
    if len(signals) != len(source_rates):
        raise ValueError(
            f"signals ({len(signals)}) and source_rates ({len(source_rates)}) "
            "must have the same length."
        )
    if not signals:
        return np.empty((0, 0), dtype=float)

    resampled: List[np.ndarray] = []
    for sig, rate in zip(signals, source_rates):
        x = np.asarray(sig, dtype=float)
        if x.ndim != 1:
            raise ValueError(f"Each channel must be 1-D; got shape {x.shape}.")
        resampled.append(np.asarray(Resample(rate, new_sfreq)(x), dtype=float))

    lengths = [r.shape[0] for r in resampled]
    n = min(lengths)
    if max(lengths) - n > _LENGTH_MISMATCH_WARN:
        logger.warning(
            "Channels differ in length after resampling to %g Hz (%s); trimming all "
            "to the shortest (%d). Check the channels share one recording duration.",
            new_sfreq,
            lengths,
            n,
        )
    return np.vstack([r[:n] for r in resampled])


# --------------------------------------------------------------------------- #
# Loader integration: a resampling signal_reader for BioserenityPSGLoader
# --------------------------------------------------------------------------- #
# Loader-compatible reader (mirrors io.psg_loader.SignalReader).
SignalReader = Callable[[Any, List[str]], np.ndarray]
# Read ONE channel's raw 1-D samples from an attached instance.
PerChannelReader = Callable[[Any, str], np.ndarray]
# Return a {physical_channel_label: native_sampling_rate_hz} mapping.
RateLookup = Callable[[Any], Mapping[str, float]]


def _default_per_channel_reader(inst: Any, label: str) -> np.ndarray:
    """Read one channel's raw 1-D samples via lunapi's ``inst.data``.

    Reuses the loader's payload reducer so lunapi's ``(labels, ndarray)`` return
    shape is handled identically here.
    """
    from ..io.psg_loader import _to_1d  # local import avoids an import cycle

    return _to_1d(inst.data([label]), label)


def _default_rate_lookup(inst: Any) -> Mapping[str, float]:
    """Build ``{channel_label: sampling_rate_hz}`` from lunapi's ``inst.headers()``.

    ``headers()`` runs Luna's ``HEADERS`` command and returns a per-channel
    DataFrame: the channel name is in the ``CH`` column (or the index) and the
    native rate in the ``SR`` column. Lookups are case-insensitive to tolerate
    minor build differences.
    """
    df = inst.headers()
    if df is None or len(df) == 0:
        return {}

    cols = {str(c).lower(): c for c in df.columns}
    sr_col = cols.get("sr")
    if sr_col is None:
        raise KeyError(
            f"inst.headers() has no 'SR' column; available: {list(df.columns)}"
        )
    ch_col = cols.get("ch")
    labels = df[ch_col] if ch_col is not None else df.index

    return {
        str(ch): float(sr)
        for ch, sr in zip(labels, df[sr_col])
        if sr is not None and str(sr) != "" and not is_nan(sr)
    }


def make_resampling_signal_reader(
    new_sfreq: float = DEFAULT_TARGET_SFREQ,
    *,
    per_channel_reader: Optional[PerChannelReader] = None,
    rate_lookup: Optional[RateLookup] = None,
) -> SignalReader:
    """Build a loader ``signal_reader`` that resamples every channel to ``new_sfreq``.

    The returned callable matches the loader's
    ``signal_reader(inst, physical_channels) -> (n_channels, n_samples)`` contract
    and yields a uniform array even when the source channels have different native
    sampling rates -- letting
    :class:`~infraslow.io.psg_loader.BioserenityPSGLoader` load mixed-rate
    recordings (which it otherwise refuses to stack)::

        from infraslow import BioserenityPSGLoader, BIOSERENITY_ALIAS_MAP
        from infraslow.processing.signal import make_resampling_signal_reader

        loader = BioserenityPSGLoader(
            subject_id="{...}",
            alias_map=BIOSERENITY_ALIAS_MAP,
            signal_reader=make_resampling_signal_reader(128.0),
        ).load()
        loader.data  # (n_channels, n_samples), every channel at 128 Hz

    Args:
        new_sfreq: Common output rate in Hz (default 128).
        per_channel_reader: Reads one channel's raw 1-D samples; defaults to a
            lunapi ``inst.data`` reader. Injectable for testing.
        rate_lookup: Returns ``{channel: native_hz}``; defaults to reading lunapi
            ``inst.headers()``. Injectable for testing.

    Raises:
        KeyError: if a requested channel's native sampling rate is unavailable.
    """
    read_one = per_channel_reader or _default_per_channel_reader
    lookup = rate_lookup or _default_rate_lookup

    def _reader(inst: Any, physical_channels: List[str]) -> np.ndarray:
        if not physical_channels:
            return np.empty((0, 0), dtype=float)

        rates = lookup(inst)
        signals: List[np.ndarray] = []
        source_rates: List[float] = []
        for label in physical_channels:
            if label not in rates:
                raise KeyError(
                    f"No sampling rate found for channel '{label}'. "
                    f"Known channels: {sorted(rates)}"
                )
            signals.append(np.asarray(read_one(inst, label), dtype=float))
            source_rates.append(rates[label])

        return resample_channels(signals, source_rates, new_sfreq)

    return _reader


__all__ = [
    "Pipeline",
    "Resample",
    "ButterFilter",
    "NotchFilter",
    "DEFAULT_TARGET_SFREQ",
    "resample_channels",
    "make_resampling_signal_reader",
]
