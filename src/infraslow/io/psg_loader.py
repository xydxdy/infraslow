"""Deterministic, alias-aware PSG/EDF loader built on LunaAPI (``lunapi``).

This module provides :class:`BioserenityPSGLoader`, a dataclass that loads a single
subject's polysomnography (PSG) recording from an EDF file and maps the
recording's *physical* channel labels (which vary wildly between montages and
acquisition systems) onto a fixed set of *canonical* channel names through a
caller-supplied alias map.

Scope is deliberately narrow. The loader **only** locates, validates, opens and
extracts raw signals. There are two optional exceptions, both convenience
selections of readers/filters from :mod:`infraslow.processing.signal`, not
general preprocessing:

* **Resampling**: when ``sf`` is set, every channel is resampled to that common
  rate *as it is read* into :attr:`~BioserenityPSGLoader.data` (each channel's
  native rate comes from lunapi ``inst.headers()``; resampling is scipy Fourier
  resampling). That is what lets channels with differing native rates pack into
  one array.
* **Powerline notch filtering**: by default (``notch_freq=60.0``), every channel
  of :attr:`~BioserenityPSGLoader.data` is notch-filtered at ``notch_freq``
  *after* resampling, via an MNE zero-phase FIR notch filter. Set
  ``notch_freq=None`` to skip it.

It still performs no other preprocessing -- no band-pass filtering,
re-referencing, scaling, or epoching. Those belong in downstream analysis code,
not in a loader.

LunaAPI touchpoints (confirmed against lunapi; see zzz-luna.org/luna/lunapi)
---------------------------------------------------------------------------
``lunapi`` is not importable in every environment, so every call into it is
funnelled through small, injectable callables. The three touchpoints below are
the entire confirmed-vs-assumed boundary:

1. ``proj = lunapi.proj()`` then ``inst = proj.inst(<id>)`` then
   ``inst.attach_edf(<path>)`` -- the project/instance construction + EDF attach
   sequence. (See :func:`_default_proj_factory` / :meth:`BioserenityPSGLoader._attach_edf`.)

2. ``inst.channels()`` -- the channel-listing method. CONFIRMED to return a
   **one-column pandas DataFrame** whose rows are the channel labels (column
   ``"Channels"``; ``inst.chs()`` is an alias). ``_coerce_label_list`` reads the
   labels from that column. (See :func:`_default_channel_lister`.)

3. ``inst.data(<channels>)`` -- the raw-signal extraction call. CONFIRMED to
   return ``(list[str] labels, ndarray of shape (n_samples, n_signals))`` --
   samples are rows, signals are columns. The default reader reads one channel
   at a time and reduces each ``(labels, array)`` payload to a 1-D vector.
   (See :func:`_default_signal_reader`.)

For testing, and to keep the confirmed-vs-assumed boundary in exactly one place,
all three can be overridden via the ``proj_factory``, ``channel_lister`` and
``signal_reader`` constructor arguments, and the pure resolution logic
(:meth:`BioserenityPSGLoader.resolve_channels`) can be driven directly with an injected
list of channel names -- no ``lunapi`` required.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..constants import BIOSERENITY_ALIAS_MAP, DEFAULT_EDF_DIR

logger = logging.getLogger(__name__)

# Type aliases for the injectable LunaAPI touchpoints. Keeping them named
# documents the expected shapes in one place.
ProjFactory = Callable[[], Any]
ChannelLister = Callable[[Any], Sequence[str]]
SignalReader = Callable[[Any, List[str]], np.ndarray]
AnnotationLoader = Callable[[Any, Path], Any]
# Returns {physical_channel_label: native_sampling_rate_hz} for an attached inst.
RateLookup = Callable[[Any], Mapping[str, float]]


# --------------------------------------------------------------------------- #
# Typed exception hierarchy
# --------------------------------------------------------------------------- #
class LunaPSGError(Exception):
    """Base class for every error raised by :class:`BioserenityPSGLoader`."""


class EnvironmentVariableError(LunaPSGError):
    """The required storage environment variable (e.g. ``$OAK``) is unset/empty."""


class DirectoryNotFoundError(LunaPSGError):
    """An expected directory (the storage root or the data subdirectory) is missing."""


class InvalidSubjectIDError(LunaPSGError):
    """The supplied subject ID is empty or unsafe (path separators, ``..``, etc.)."""


class EDFFileNotFoundError(LunaPSGError):
    """The constructed ``<subject_id>.edf`` path does not point at a real file."""


class ChannelResolutionError(LunaPSGError):
    """A requested channel could not be resolved (and strict mode is enabled),
    or a requested canonical name has no entry in the alias map."""


class DataLoadError(LunaPSGError):
    """LunaAPI failed to open the EDF or return usable signal data."""


class AnnotationLoadError(LunaPSGError):
    """The optional annotation-loading callback raised an error."""


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ChannelConflict:
    """One occurrence of an alias matching a physical channel already claimed.

    Recorded whenever a canonical channel's alias resolves to a physical EDF
    channel that an *earlier* (higher-priority) canonical channel already owns.
    This is how the loader prevents the same physical channel being assigned to
    two canonical names.
    """

    canonical: str          # the canonical name that wanted the channel
    alias: str              # the alias (from alias_map) that matched
    physical: str           # the physical EDF channel label that matched
    claimed_by: str         # the canonical name that already owns ``physical``


def _default_normalizer(label: str, *, case_insensitive: bool) -> str:
    """Normalize a channel label for matching: strip whitespace, optional casefold.

    Physical EDF labels are notoriously inconsistent in surrounding whitespace
    and capitalisation, so a light normalization makes alias matching robust
    without altering the stored physical label.
    """
    normalized = label.strip()
    if case_insensitive:
        normalized = normalized.casefold()
    return normalized


# --------------------------------------------------------------------------- #
# Default LunaAPI touchpoints (the "must confirm" surface)
# --------------------------------------------------------------------------- #
def _default_proj_factory() -> Any:
    """Create a LunaAPI project. Imports ``lunapi`` lazily so the rest of this
    module (and its tests) work without ``lunapi`` installed.

    CONFIRM: ``lunapi.proj()`` is the documented entry point.
    """
    try:
        import lunapi as lp  # noqa: PLC0415 - intentional lazy import
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise DataLoadError(
            "lunapi is not importable; install LunaAPI or inject a proj_factory."
        ) from exc
    return lp.proj()


def _default_channel_lister(inst: Any) -> Sequence[str]:
    """Return the physical channel labels of an attached EDF instance.

    CONFIRMED against lunapi (zzz-luna.org/luna/lunapi): ``inst.channels()``
    returns a one-column pandas DataFrame whose rows are the channel labels
    (``inst.chs()`` is an alias for it). ``_coerce_label_list`` extracts the
    labels from that column (and still tolerates Series/ndarray/list builds).
    """
    return _coerce_label_list(inst.channels())


def _default_signal_reader(inst: Any, physical_channels: List[str]) -> np.ndarray:
    """Read raw signals for the given physical channels into ``(n_channels, n_samples)``.

    Reads one channel at a time and stacks, so the orientation of LunaAPI's
    return value only has to be reduced to a 1-D vector (see ``_to_1d``). This
    avoids depending on whether ``inst.data()`` returns samples as rows or
    columns for the multi-channel case.

    CONFIRM: ``inst.data([label])`` is the raw-signal accessor and that its
    payload reduces to one sample vector per requested channel.

    Because the loader performs no resampling, channels with differing sample
    counts cannot be packed into a rectangular array and raise
    :class:`DataLoadError` rather than being silently truncated/padded.
    """
    if not physical_channels:
        return np.empty((0, 0), dtype=float)

    columns: List[np.ndarray] = []
    for label in physical_channels:
        try:
            raw = inst.data([label])  # CONFIRM signature/return shape
        except Exception as exc:  # noqa: BLE001 - surface any LunaAPI/IO error
            raise DataLoadError(
                f"lunapi failed to read channel '{label}': {exc}"
            ) from exc
        columns.append(_to_1d(raw, label))

    lengths = {col.shape[0] for col in columns}
    if len(lengths) > 1:
        detail = ", ".join(
            f"{name}={col.shape[0]}" for name, col in zip(physical_channels, columns)
        )
        raise DataLoadError(
            "Resolved channels have differing sample counts and cannot be packed "
            "into an (n_channels, n_samples) array. Set sf to resample all channels "
            f"to a common rate. Sample counts: {detail}."
        )
    return np.vstack(columns)


def _coerce_label_list(obj: Any) -> List[str]:
    """Best-effort coercion of a LunaAPI channel-listing return value to ``List[str]``.

    ``lunapi``'s :meth:`inst.channels` returns a **one-column pandas DataFrame**
    whose *rows* are the channel labels (the column is named ``"Channels"``), so
    the labels must be read from that column -- iterating the DataFrame directly
    would yield its column name(s) instead. Older/other builds may return a
    pandas Series, a numpy array, or a plain list; all are reduced here to a flat
    list of label strings. Duck-typed (``.columns``/``.iloc``) so this module
    needs no hard pandas import.
    """
    # pandas DataFrame: take the first (only) column's values, not the columns.
    if hasattr(obj, "columns") and hasattr(obj, "iloc"):
        if len(obj) == 0 or obj.shape[1] == 0:
            return []
        obj = obj.iloc[:, 0].tolist()
    # pandas Series / numpy array expose ``.tolist()``; a DataFrame does not.
    elif hasattr(obj, "tolist"):
        obj = obj.tolist()
    return [str(label) for label in obj]


def _to_1d(raw: Any, label: str) -> np.ndarray:
    """Reduce a single-channel LunaAPI ``data()`` payload to a 1-D float vector.

    Handles the documented ``(header, ndarray)`` tuple form as well as a bare
    array. Anything that is not reducible to 1-D after squeezing is an error
    rather than a guess.
    """
    payload = raw
    # ``(header, data)`` style return -> keep the data half.
    if isinstance(payload, tuple) and len(payload) == 2:
        payload = payload[1]
    arr = np.asarray(payload, dtype=float)
    arr = np.squeeze(arr)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    if arr.ndim != 1:
        raise DataLoadError(
            f"Channel '{label}' returned data with shape {arr.shape}; "
            "expected a single 1-D sample vector."
        )
    return arr


# --------------------------------------------------------------------------- #
# The loader
# --------------------------------------------------------------------------- #
@dataclass
class BioserenityPSGLoader:
    """Load and canonicalize one subject's EDF recording via LunaAPI.

    Path convention::

        <edf_dir>/<subject_id>.edf

    which with the default ``edf_dir`` is ``$OAK/psg/Bioserenity/edf/<subject_id>.edf``.

    Parameters
    ----------
    subject_id:
        Subject identifier; becomes the EDF file stem. Must not contain path
        separators or ``..``.
    alias_map:
        Mapping of canonical channel name -> ordered list of physical aliases.
        **List order is priority order**: earlier aliases win. Defaults to
        :data:`BIOSERENITY_ALIAS_MAP` when left ``None``.
    requested_channels:
        Canonical names to resolve, in priority order. ``None`` (default) means
        every key of ``alias_map``, in mapping order.
    strict:
        If ``True``, any unresolved requested channel (missing or lost to a
        conflict) raises :class:`ChannelResolutionError` during :meth:`load`.
        If ``False`` (default), unresolved channels are recorded and logged.
    edf_dir:
        Directory containing ``<subject_id>.edf`` files. May reference
        environment variables (expanded via :func:`os.path.expandvars`).
        Defaults to :data:`~infraslow.constants.DEFAULT_EDF_DIR`
        (``"$OAK/psg/Bioserenity/edf"``).
    case_insensitive:
        Match aliases against physical labels ignoring case (default ``True``).
        Surrounding whitespace is always ignored.
    sf:
        Target sampling frequency in Hz, and -- after :meth:`load` -- the actual
        rate of :attr:`data`. When set at construction (not ``None``), every
        channel is resampled to this common rate while being read, so channels
        with differing native rates pack into one ``(n_channels, n_samples)``
        :attr:`data` array (all at ``sf`` Hz). This is a convenience that selects
        the resampling reader from :mod:`infraslow.processing.signal`; passing an
        explicit ``signal_reader`` instead overrides it. When left ``None``, no
        resampling happens and :meth:`load` **back-fills** ``sf`` with the native
        sampling rate of the first loaded channel (read from lunapi), so ``sf``
        always describes the rate of :attr:`data` afterwards. Must be > 0;
        mutually exclusive with ``signal_reader``.
    rate_lookup:
        Optional ``inst -> {channel_label: native_hz}`` callback used to read
        native sampling rates (for back-filling ``sf`` and for the resampling
        reader). Defaults to lunapi's ``inst.headers()``. Injectable for testing.
    notch_freq:
        Frequency in Hz to notch out (powerline noise) via a zero-phase FIR
        notch filter (``mne.filter.notch_filter`` with ``method="fir"``),
        applied to every channel of :attr:`data` after resampling (whether that
        resampling was explicit via ``sf`` or the data's native rate).
        Default ``60.0`` (US/domestic powerline; use ``50.0`` where mains is
        50 Hz). Set to ``None`` to skip notch filtering entirely.
    annotation_loader:
        Callback ``(inst, edf_path) -> Any`` invoked after signals load; its
        return value is exposed via :attr:`annotations`. When left ``None``
        (default), it resolves to
        :func:`~infraslow.io.usleep_hypnodensity.make_usleep_annotation_loader`
        keyed on :attr:`usleep_channel`, so a plain loader automatically attaches
        the subject/channel's own U-Sleep hypnodensity, reduced to a per-epoch
        ``stage`` hypnogram. Pass an explicit callable to parse a different
        annotation format (e.g.
        :func:`~infraslow.io.hypnodensity.make_hypnodensity_annotation_loader`
        for the older 30-s Bioserenity Hypnodensity CSVs),
        ``make_usleep_annotation_loader(channel, required=False)`` to tolerate a
        missing U-Sleep file (yielding ``None`` instead of raising), or a no-op
        ``lambda inst, edf_path: None`` to skip annotations entirely.
        Decoupling annotation parsing from the loader keeps this class free of
        format-specific logic.
    usleep_channel:
        Canonical channel name whose U-Sleep hypnodensity the *default*
        ``annotation_loader`` loads (ignored if ``annotation_loader`` is set
        explicitly). ``None`` (default) uses the first of :attr:`requested_channels`
        when given, else the first key of ``alias_map`` -- i.e. it "just works" for
        the common single-channel case (``requested_channels=[CHANNEL]``); pass this
        explicitly whenever the default guess would pick the wrong channel (e.g.
        loading several channels but scoring off just one of them).
    proj_factory / channel_lister / signal_reader:
        Injection points for the three LunaAPI touchpoints (see module
        docstring). Default to the ``lunapi``-backed implementations. When ``sf``
        is set and no ``signal_reader`` is given, a resampling reader is selected
        automatically.
    """

    subject_id: str
    alias_map: Optional[Mapping[str, Sequence[str]]] = None
    requested_channels: Optional[Sequence[str]] = None
    strict: bool = False
    edf_dir: str = DEFAULT_EDF_DIR
    case_insensitive: bool = True
    sf: Optional[float] = None
    rate_lookup: Optional[RateLookup] = None
    notch_freq: Optional[float] = 60.0
    annotation_loader: Optional[AnnotationLoader] = None
    usleep_channel: Optional[str] = None
    proj_factory: Optional[ProjFactory] = None
    channel_lister: Optional[ChannelLister] = None
    signal_reader: Optional[SignalReader] = None

    # ---- internal state (populated by load()) ---------------------------- #
    _edf_path: Optional[Path] = field(default=None, init=False, repr=False)
    _inst: Any = field(default=None, init=False, repr=False)
    _data: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _resolved: Dict[str, str] = field(default_factory=dict, init=False, repr=False)
    _missing: List[str] = field(default_factory=list, init=False, repr=False)
    _conflicts: List[ChannelConflict] = field(
        default_factory=list, init=False, repr=False
    )
    _channel_order: List[str] = field(default_factory=list, init=False, repr=False)
    _annotations: Any = field(default=None, init=False, repr=False)
    _loaded: bool = field(default=False, init=False, repr=False)

    # ------------------------------------------------------------------ #
    # Construction-time validation
    # ------------------------------------------------------------------ #
    def __post_init__(self) -> None:
        # Default to the Bioserenity alias map (defined at module scope, below)
        # when none is supplied.
        if self.alias_map is None:
            self.alias_map = BIOSERENITY_ALIAS_MAP
        if not isinstance(self.alias_map, Mapping) or not self.alias_map:
            raise ValueError("alias_map must be a non-empty mapping.")
        if self.sf is not None:
            if not isinstance(self.sf, (int, float)) or self.sf <= 0:
                raise ValueError(
                    f"sf must be a positive sampling frequency, got {self.sf!r}."
                )
            if self.signal_reader is not None:
                raise ValueError(
                    "Pass either sf (to auto-select the resampling reader) or an "
                    "explicit signal_reader, not both."
                )
        if self.notch_freq is not None:
            if not isinstance(self.notch_freq, (int, float)) or self.notch_freq <= 0:
                raise ValueError(
                    f"notch_freq must be a positive frequency or None, got {self.notch_freq!r}."
                )
        self._validate_subject_id()
        # Fail fast on a requested canonical name that the alias map cannot ever
        # resolve -- this is a configuration error independent of strict mode.
        if self.requested_channels is not None:
            unknown = [c for c in self.requested_channels if c not in self.alias_map]
            if unknown:
                raise ChannelResolutionError(
                    f"requested_channels not present in alias_map: {', '.join(unknown)}"
                )
        # Default the annotation loader to the U-Sleep reducer so a plain loader
        # still attaches a per-epoch stage hypnogram, keyed on usleep_channel (or
        # the first requested/alias-map channel when unset -- see usleep_channel's
        # docstring). Imported lazily; usleep_hypnodensity carries no dependency
        # back on this module, so there is no import cycle. Only the factory is
        # built here -- no file is read until load() invokes it.
        if self.annotation_loader is None:
            from .usleep_hypnodensity import (  # noqa: PLC0415 - lazy, keeps construction cheap
                make_usleep_annotation_loader,
            )

            channel = self.usleep_channel or self.canonical_channels[0]
            self.annotation_loader = make_usleep_annotation_loader(channel, alias_map=self.alias_map)

    # ------------------------------------------------------------------ #
    # Public, pure-ish helpers
    # ------------------------------------------------------------------ #
    @property
    def canonical_channels(self) -> List[str]:
        """The canonical names to resolve, in deterministic priority order."""
        if self.requested_channels is not None:
            return list(self.requested_channels)
        return list(self.alias_map.keys())

    def _normalize(self, label: str) -> str:
        return _default_normalizer(label, case_insensitive=self.case_insensitive)

    def resolve_channels(
        self, available_channels: Sequence[str]
    ) -> Tuple[Dict[str, str], List[str], List[ChannelConflict]]:
        """Resolve canonical names against ``available_channels`` (physical labels).

        Pure with respect to LunaAPI: pass any list of physical labels (e.g. in a
        test) and it populates and returns ``(resolved, missing, conflicts)``.

        Algorithm (deterministic):

        * Canonical names are processed in :attr:`canonical_channels` order.
        * For each, its aliases are tried in list order (priority).
        * The first alias that is present in the EDF **and not already claimed**
          wins; the physical channel is marked claimed so it can never be
          assigned again.
        * An alias that is present but already claimed is recorded as a
          :class:`ChannelConflict` and skipped.
        * A canonical with no present alias at all is recorded as *missing*.
        """
        self._resolved = {}
        self._missing = []
        self._conflicts = []
        self._channel_order = []

        # Normalized -> actual physical label (first occurrence wins).
        norm_lookup: Dict[str, str] = {}
        for phys in available_channels:
            key = self._normalize(str(phys))
            norm_lookup.setdefault(key, str(phys))

        claimed: Dict[str, str] = {}  # physical label -> canonical that owns it

        for canonical in self.canonical_channels:
            aliases = self.alias_map.get(canonical)
            if aliases is None:
                # Guarded against in __post_init__ for requested_channels, but a
                # custom canonical list could still slip through.
                raise ChannelResolutionError(
                    f"No alias entry for requested channel '{canonical}'."
                )

            resolved_phys: Optional[str] = None
            saw_present_alias = False
            for alias in aliases:
                phys = norm_lookup.get(self._normalize(alias))
                if phys is None:
                    continue  # this alias is not in the EDF
                saw_present_alias = True
                if phys in claimed:
                    self._conflicts.append(
                        ChannelConflict(
                            canonical=canonical,
                            alias=alias,
                            physical=phys,
                            claimed_by=claimed[phys],
                        )
                    )
                    continue  # already taken; try the next alias
                resolved_phys = phys
                break

            if resolved_phys is not None:
                self._resolved[canonical] = resolved_phys
                claimed[resolved_phys] = canonical
                self._channel_order.append(canonical)
                logger.debug("Resolved %s -> %s", canonical, resolved_phys)
            elif not saw_present_alias:
                self._missing.append(canonical)
                logger.warning("Channel %s missing: no alias present in EDF.", canonical)
            else:
                # Present in the EDF but every match was already claimed.
                logger.warning(
                    "Channel %s unresolved: all matching physical channels already "
                    "claimed by higher-priority channels.",
                    canonical,
                )

        return self._resolved, self._missing, list(self._conflicts)

    # ------------------------------------------------------------------ #
    # The full load pipeline
    # ------------------------------------------------------------------ #
    def load(self) -> "BioserenityPSGLoader":
        """Validate, open, resolve and extract. Returns ``self`` for chaining."""
        data_dir = self._resolve_edf_dir()
        self._edf_path = self._build_and_validate_edf_path(data_dir)
        logger.info("Loading subject %s from %s", self.subject_id, self._edf_path)

        self._inst = self._attach_edf(self._edf_path)
        available = self._list_channels(self._inst)
        logger.info("EDF exposes %d physical channels.", len(available))

        self.resolve_channels(available)
        if self.strict and self.unresolved_channels:
            raise ChannelResolutionError(
                "strict mode: could not resolve channel(s): "
                + ", ".join(self.unresolved_channels)
            )

        self._data = self._read_signals(self._inst)

        # When the caller did not request a target rate, no resampling happened,
        # so report the data's real rate: back-fill sf with the native sampling
        # frequency of the first loaded channel (read from lunapi).
        if self.sf is None:
            self.sf = self._detect_native_sf(self._inst)
            if self.sf is not None:
                logger.info("Native sampling frequency of data: %g Hz.", self.sf)

        if self.notch_freq is not None and self.n_channels > 0:
            self._data = self._apply_notch(self._data)

        if self.annotation_loader is not None:
            self._annotations = self._load_annotations(self._inst, self._edf_path)

        self._loaded = True
        logger.info(
            "Loaded %d channel(s), %d sample(s); %d missing, %d conflict record(s).",
            self.n_channels,
            self.n_samples,
            len(self._missing),
            len(self._conflicts),
        )
        return self

    # ------------------------------------------------------------------ #
    # Validation steps
    # ------------------------------------------------------------------ #
    def _validate_subject_id(self) -> None:
        sid = self.subject_id
        if not isinstance(sid, str) or not sid.strip():
            raise InvalidSubjectIDError("subject_id must be a non-empty string.")
        if sid != sid.strip():
            raise InvalidSubjectIDError(
                f"subject_id has surrounding whitespace: {sid!r}"
            )
        bad = ("/", "\\", "..", "\x00")
        if any(token in sid for token in bad) or os.sep in sid:
            raise InvalidSubjectIDError(
                f"subject_id {sid!r} contains path separators or unsafe tokens."
            )

    def _resolve_edf_dir(self) -> Path:
        expanded = os.path.expandvars(self.edf_dir)
        if "$" in expanded:
            raise EnvironmentVariableError(
                f"edf_dir references an unset environment variable: {self.edf_dir!r}"
            )
        data_dir = Path(expanded)
        if not data_dir.is_dir():
            raise DirectoryNotFoundError(f"Data directory does not exist: {data_dir}")
        return data_dir

    def _build_and_validate_edf_path(self, data_dir: Path) -> Path:
        edf_path = data_dir / f"{self.subject_id}.edf"
        if not edf_path.is_file():
            raise EDFFileNotFoundError(f"EDF file does not exist: {edf_path}")
        return edf_path

    # ------------------------------------------------------------------ #
    # LunaAPI touchpoints (delegated to injectable callables)
    # ------------------------------------------------------------------ #
    def _attach_edf(self, edf_path: Path) -> Any:
        factory = self.proj_factory or _default_proj_factory
        try:
            proj = factory()
            inst = proj.inst(self.subject_id)  # CONFIRM: proj.inst(id)
            inst.attach_edf(str(edf_path))      # CONFIRM: inst.attach_edf(path)
        except LunaPSGError:
            raise
        except Exception as exc:  # noqa: BLE001 - wrap any LunaAPI/IO error
            raise DataLoadError(
                f"lunapi failed to attach EDF {edf_path}: {exc}"
            ) from exc
        return inst

    def _list_channels(self, inst: Any) -> List[str]:
        lister = self.channel_lister or _default_channel_lister
        try:
            labels = lister(inst)
        except Exception as exc:  # noqa: BLE001
            raise DataLoadError(f"lunapi failed to list channels: {exc}") from exc
        return _coerce_label_list(labels)

    def _resolve_signal_reader(self) -> SignalReader:
        """Pick the signal reader: explicit > sf-driven resampling > default.

        ``sf`` is a convenience for the resampling reader in
        :mod:`infraslow.processing.signal`; the import is local to avoid a
        package-level import cycle (that module imports helpers from this one).
        """
        if self.signal_reader is not None:
            return self.signal_reader
        if self.sf is not None:
            from ..processing.signal import (  # noqa: PLC0415 - lazy, avoids cycle
                make_resampling_signal_reader,
            )

            return make_resampling_signal_reader(
                float(self.sf), rate_lookup=self.rate_lookup
            )
        return _default_signal_reader

    def _detect_native_sf(self, inst: Any) -> Optional[float]:
        """Native sampling rate (Hz) of the first loaded channel, or ``None``.

        Reads ``{channel: native_hz}`` via :attr:`rate_lookup` (lunapi
        ``inst.headers()`` by default) and returns the rate of the first resolved
        channel. Returns ``None`` (rather than raising) when there are no resolved
        channels or the rate cannot be read, so it never breaks a successful load.
        """
        if not self._channel_order:
            return None
        first_physical = self._resolved[self._channel_order[0]]
        lookup = self.rate_lookup
        if lookup is None:
            from ..processing.signal import (  # noqa: PLC0415 - lazy, avoids cycle
                _default_rate_lookup,
            )

            lookup = _default_rate_lookup
        try:
            rate = lookup(inst).get(first_physical)
        except Exception as exc:  # noqa: BLE001 - reporting only; never fatal
            logger.warning("Could not read native sampling rate: %s", exc)
            return None
        return float(rate) if rate is not None else None

    def _read_signals(self, inst: Any) -> np.ndarray:
        reader = self._resolve_signal_reader()
        physical = [self._resolved[c] for c in self._channel_order]
        data = np.asarray(reader(inst, physical))
        # Enforce the (n_channels, n_samples) contract.
        if physical:
            if data.ndim != 2 or data.shape[0] != len(physical):
                raise DataLoadError(
                    f"signal_reader returned shape {data.shape}; expected "
                    f"({len(physical)}, n_samples)."
                )
        return data

    def _apply_notch(self, data: np.ndarray) -> np.ndarray:
        """Notch-filter ``data`` (already resampled) at :attr:`notch_freq`.

        Calls ``mne.filter.notch_filter`` directly (``method="fir"``, zero-phase);
        ``mne`` is imported lazily, matching the lazy ``lunapi``/resampling-reader
        imports elsewhere in this module.
        """
        if self.sf is None:
            raise DataLoadError(
                "notch_freq is set but sf could not be determined; cannot "
                "notch-filter without a known sampling rate."
            )
        import mne  # noqa: PLC0415 - lazy, keeps mne optional at import time

        logger.info("Applying %g Hz notch filter (FIR) at %g Hz sampling rate.", self.notch_freq, self.sf)
        return mne.filter.notch_filter(
            np.asarray(data, dtype=float), Fs=self.sf, freqs=self.notch_freq, method="fir", verbose=False
        )

    def _load_annotations(self, inst: Any, edf_path: Path) -> Any:
        assert self.annotation_loader is not None
        try:
            return self.annotation_loader(inst, edf_path)
        except Exception as exc:  # noqa: BLE001
            raise AnnotationLoadError(
                f"annotation_loader failed for {edf_path}: {exc}"
            ) from exc

    # ------------------------------------------------------------------ #
    # Accessors
    # ------------------------------------------------------------------ #
    def _require_loaded(self) -> None:
        if not self._loaded:
            raise LunaPSGError("Loader has not run; call load() first.")

    def get_channel(self, canonical: str) -> np.ndarray:
        """Return the 1-D sample vector for a resolved canonical channel name."""
        self._require_loaded()
        if canonical not in self._resolved:
            raise ChannelResolutionError(
                f"Channel '{canonical}' was not resolved for subject "
                f"{self.subject_id}."
            )
        idx = self._channel_order.index(canonical)
        return self._data[idx]

    @property
    def data(self) -> np.ndarray:
        """Signals as ``(n_channels, n_samples)`` in :attr:`channel_names` order."""
        self._require_loaded()
        assert self._data is not None
        return self._data

    @property
    def channel_names(self) -> List[str]:
        """Resolved canonical names, ordered to match :attr:`data` rows."""
        return list(self._channel_order)

    @property
    def resolved_channels(self) -> Dict[str, str]:
        """Mapping canonical name -> physical EDF label that was assigned."""
        return dict(self._resolved)

    @property
    def missing_channels(self) -> List[str]:
        """Canonical names with no alias present in the EDF at all."""
        return list(self._missing)

    @property
    def conflicts(self) -> List[ChannelConflict]:
        """Recorded alias-vs-already-claimed-channel conflicts."""
        return list(self._conflicts)

    @property
    def unresolved_channels(self) -> List[str]:
        """Requested canonical names that ended up without an assignment."""
        return [c for c in self.canonical_channels if c not in self._resolved]

    def usleep_hypnodensity(self, channel: Optional[str] = None, **kwargs: Any) -> Tuple[np.ndarray, np.ndarray]:
        """This loader's own raw (pre-reduction) 1-s U-Sleep hypnodensity.

        Delegates to
        :func:`~infraslow.io.usleep_hypnodensity.load_usleep_hypnodensity` for
        :attr:`subject_id` -- the exact call the default ``annotation_loader``
        makes internally (see :attr:`usleep_channel`) -- exposed directly so
        callers needing the un-reduced per-second data (e.g. to illustrate the
        1-s -> N-s reduction) don't have to duplicate the channel-resolution/
        alias-map logic themselves. Does not require :meth:`load` to have run
        first (only :attr:`subject_id`/:attr:`alias_map` are used).

        Args:
            channel: Canonical channel name. Defaults to :attr:`usleep_channel`,
                then the first of :attr:`canonical_channels`.
            **kwargs: Forwarded to :func:`load_usleep_hypnodensity` (e.g.
                ``base_dir``, ``epoch_sec``).

        Returns:
            ``(t_hyp, probs)`` -- see :func:`load_usleep_hypnodensity`.
        """
        from .usleep_hypnodensity import load_usleep_hypnodensity  # noqa: PLC0415 - lazy

        resolved_channel = channel or self.usleep_channel or self.canonical_channels[0]
        return load_usleep_hypnodensity(
            self.subject_id, resolved_channel, alias_map=self.alias_map, **kwargs
        )

    @property
    def annotations(self) -> Any:
        """Whatever the ``annotation_loader`` callback returned (or ``None``)."""
        return self._annotations

    @property
    def edf_path(self) -> Optional[Path]:
        """The validated EDF path (``None`` before :meth:`load`)."""
        return self._edf_path

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def n_channels(self) -> int:
        if self._data is None:
            return 0
        return int(self._data.shape[0]) if self._data.ndim == 2 else 0

    @property
    def n_samples(self) -> int:
        if self._data is None or self._data.ndim != 2 or self._data.shape[0] == 0:
            return 0
        return int(self._data.shape[1])

