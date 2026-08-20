"""Every constant used across the :mod:`infraslow` package, in one place.

Grouped by the concern each constant serves (paths, I/O, per-stage signal
processing, viz) rather than by which module originally defined it.
Each owning submodule re-exports its constants from here (``from ..constants
import X``) so existing call sites (``infraslow.config.DEFAULT_METADATA``,
``infraslow.processing.infraslow.DEFAULT_SIGMA_BAND``, ...) keep working
unchanged -- this module is the single place to look up or change a value,
not a new import path everything must switch to.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Mapping, Tuple

# --------------------------------------------------------------------------- #
# Paths -- locating Bioserenity pipeline inputs (see infraslow.config).
#
# Metadata CSVs are small and checked into the repo (``infraslow/metadata/``),
# so they no longer depend on ``$OAK`` being mounted. EDF/hypnodensity data is
# far too large to check in and stays on ``$OAK``.
# --------------------------------------------------------------------------- #
METADATA_DIR = Path(__file__).resolve().parent / "metadata"
REPO_ROOT = Path(__file__).resolve().parent.parent.parent

DEFAULT_METADATA = str(METADATA_DIR / "Morpheus_Data_All5.csv")
DEFAULT_METADATA2 = str(METADATA_DIR / "bioserenity_metadata3.csv")
DEFAULT_DRUG_METADATA = str(METADATA_DIR / "drug_usage_metadata.csv")
DEFAULT_EDF_DIR = "$OAK/psg/Bioserenity/edf"
DEFAULT_HYPNO_DIR = "$OAK/psg/Bioserenity/Sleep_Staging"
#: 1-second U-Sleep hypnodensities: one ``{channel_file}.npy`` per subject/channel,
#: shape ``(n_seconds, 5)``, columns in ``USLEEP_STAGE_ORDER`` order. A different data
#: source from ``DEFAULT_HYPNO_DIR``'s 30-s Bioserenity Hypnodensity CSVs (see
#: ``infraslow.io.usleep_hypnodensity``).
DEFAULT_USLEEP_HYPNODENSITY_DIR = "$OAK/AISleepScientist/data/bioserenity/usleep_hypnodensities"
DEFAULT_DRUG_EXCLUDE_LIST = str(REPO_ROOT / "drug" / "drug_exclude.csv")


# --------------------------------------------------------------------------- #
# I/O -- hypnodensity CSV parsing (see infraslow.io.hypnodensity).
# --------------------------------------------------------------------------- #
DEFAULT_TIMESTAMP_COLUMN = "Timestamp"
DEFAULT_STAGING_DIRNAME = "Sleep_Staging"
DEFAULT_HYPNODENSITY_SUFFIX = "_Hypnodensity.csv"


# --------------------------------------------------------------------------- #
# I/O -- 1-s U-Sleep hypnodensity (see infraslow.io.usleep_hypnodensity).
# --------------------------------------------------------------------------- #
#: Column order of the 5 stage-probability columns in each U-Sleep hypnodensity
#: ``.npy`` file -- matches the stage order ``infraslow.io.hypnodensity`` already uses
#: for the (30-s epoch) Bioserenity hypnodensity CSVs. Not independently verifiable from
#: the ``.npy`` files themselves (no header); documented, not asserted, as an assumption.
USLEEP_STAGE_ORDER: Tuple[str, ...] = ("Wake", "N1", "N2", "N3", "REM")

#: Seconds per row of the raw U-Sleep hypnodensity files (1-second resolution).
DEFAULT_USLEEP_EPOCH_SEC = 1.0

#: Default window (s) averaged when reducing 1-s U-Sleep hypnodensity to a coarser
#: hypnogram (see ``hypnodensity_to_epoch_hypnogram``).
DEFAULT_HYPNOGRAM_EPOCH_SEC = 3.0

#: Max allowed |duration mismatch| (s) between a subject's U-Sleep hypnodensity length
#: and its ``temporal_ISFS`` envelope length before the two are treated as misaligned
#: recordings.
DEFAULT_USLEEP_ALIGN_TOLERANCE_SEC = 120.0


# --------------------------------------------------------------------------- #
# I/O -- concurrency (see infraslow.io.utils).
#
# Opening/decoding a .npz (or reading any per-subject file) is dominated by
# Lustre I/O latency, not CPU, and a blocking read releases the GIL, so more
# threads than cores still overlaps more latency -- matters once a caller is
# loading a whole cohort (potentially 70k+ subject files).
# --------------------------------------------------------------------------- #
N_IO_WORKERS = min(32, len(os.sched_getaffinity(0)) * 4)


# --------------------------------------------------------------------------- #
# I/O -- PSG channel aliasing (see infraslow.io.psg_loader).
# --------------------------------------------------------------------------- #
#: Canonical alias map for the Bioserenity dataset. List order is priority order.
BIOSERENITY_ALIAS_MAP: Dict[str, List[str]] = {
    "F3": ["F3M2", "F3A2", "F3-M2", "EEG F3-A2", "FZM2", "FZA2", "FP1M2", "FP1A2", "F7M2", "F7A2", "F3:M2", "F3"],
    "F4": ["F4M1", "F4A1", "F4-M1", "EEG F4-A1", "FZM2", "FZA2", "FP2M1", "FP2A1", "F8M1", "F8A1", "F4:M1", "F4"],
    "C3": ["C3M2", "C3A2", "C3-M2", "EEG C3-A2", "C3M1", "CZM2", "C3:M2", "C3"],
    "C4": ["C4M1", "C4A1", "C4-M1", "EEG C4-A1", "C4M2", "CZM2", "C4:M1", "C4"],
    "O1": ["O1M2", "O1A2", "O1-M2", "EEG O1-A2", "O1M1", "O1:M2", "O1"],
    "O2": ["O2M1", "O2A1", "O2-M1", "EEG O2-A1", "O2M2", "O2:M1", "O2"],
    "A1A2": ["A1A2", "M1M2", "EEG A1-A2", "EEG M1-M2"],
    "LEOG": ["LOC", "LEOG", "E1-M2", "EOG LOC-A2", "EOG1:M2", "E1"],
    "REOG": ["ROC", "REOG", "E2-M2", "EOG ROC-A1", "EOG ROC-A2", "EOG2:M1", "E2"],
    "Chin": ["Chin", "CHIN", "chin", "emg_Chin", "EMG", "CHINEMG", "ChinEMG", "EMG Chin", "Chin1-Chin2", "Chin 1-Chin 2", "ChinL", "Chin-L", "ChinR", "Chin-R"],
    "ECG": ["ECG", "EKG", "ECG1-ECG2"],
    "LLeg": ["LLeg", "LLEG", "emg_LLeg", "LEMG", "L EMG", "LLEGEMG", "LEG/L", "Left Leg", "L-Leg 1-L-Leg 2", "Leg-L", "Leg 1", "LAT", "Leg/L", "LEG1"],
    "RLeg": ["RLeg", "RLEG", "emg_RLeg", "R EMG", "RLEGEMG", "LEG/R", "Right Leg", "R-Leg1-R-Leg2", "Leg-R", "Leg 2", "RAT", "Leg/R", "LEG2", "LEMG"],
    "LArm": ["L-Arm", "ARMLeft", "LArm"],
    "RArm": ["R-Arm", "ARMRight", "RArm"],
    "PFlow": ["PFlo", "Pflo", "PFLO", "flow_PFlo", "PTAF", "Nasal Pressure", "PAP Flow", "Pflow", "PFlow", "Ptaf", "Flow Patient"],
    "TFlow": ["TFlo", "Tflo", "TFLO", "flow_TFlo", "Flow", "Thermistor", "FLOW", "Thermist", "Airflow", "Therm", "Flow Patient"],
    "CFlow": ["CFlo", "Cflo", "CFLO", "flow_CFlo", "VFLOW", "CFlow", "PAP Flow", "CPAP Flow", "Flow Patient"],
    "Thorax": ["Tho", "THO", "Thorax", "Thor", "THOR", "Effort THO"],
    "Abdomen": ["Abd", "ABD", "Abdomen", "Abdo", "ABDM", "Effort ABD"],
    "SpO2": ["SpO2", "SAO2"],
    "Snoring": ["SNOR", "Snore", "PSNO", "SNORE", "MICR", "Snoring", "Micr", "Micro", "Snoring Sensor"],
    "Position": ["POS", "Body", "BODY", "Manual Pos", "ManPosition", "Body Position"],
    "CPAP": ["CPAP", "VPAP", "CPress", "PAP Press", "CPAP Pressure", "PressCheck"],
    "IPAP": ["IPAP", "xPAP IPAP", "CPAP IPAP"],
    "EPAP": ["EPAP", "xPAP EPAP", "CPAP EPAP"],
    "Leak": ["Leak", "PAP Leak", "Leak Total", "CPAP Leak", "LEAK"],
    "PPG": ["PPG", "Pleth", "Plethysmogram"],
    "Pulse": ["Pulse", "PulseRate", "PulseR", "PULSE", "HR"],
    "RR": ["RR", "rr"],
    "IntercostalEMG": ["ICOSEMG", "INT 1", "InterEMG", "EMG1", "INT"],
    "Impedance": ["imp"],
    "CO2": ["CO2", "pCo2", "EtCO2", "ECO2", "EtCO", "CO2_Flow", "tCO2", "mmHG", "mmHg"],
}


# --------------------------------------------------------------------------- #
# Processing -- signal resampling (see infraslow.processing.signal).
# --------------------------------------------------------------------------- #
#: Default common sampling rate (Hz) for resampling heterogeneous-rate channels.
DEFAULT_TARGET_SFREQ = 128.0


# --------------------------------------------------------------------------- #
# Processing -- sleep-stage epochs & spindle detection (see
# infraslow.processing.spindle).
# --------------------------------------------------------------------------- #
#: YASA's integer sleep-stage convention (see ``yasa.hypno_str_to_int``):
#: -2=Unscored, -1=Artefact/Movement, 0=Wake, 1=N1, 2=N2, 3=N3, 4=REM.
#: NREM sleep is N1+N2+N3; sleep spindles are a hallmark of N2 (and present in
#: N3), so the NREM-only default mirrors the YASA notebook's ``include=(2, 3)``.
NREM_STAGES: Tuple[int, ...] = (2, 3)

#: Default seconds per scored epoch. Bioserenity hypnodensity epochs are 30 s.
DEFAULT_EPOCH_SEC = 30.0

#: Map the canonical stage labels this repo emits (Wake/N1/N2/N3/REM, see
#: ``infraslow.io.hypnodensity``) onto YASA's integer codes. Lookups are
#: case-insensitive; ``yasa.hypno_str_to_int`` covers the lowercase spellings,
#: but pinning the mapping here keeps the contract explicit and stable.
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

#: EEG channels suggested for spindle work (used by preprocessing.py).
DEFAULT_EEG_CHANNELS: Tuple[str, ...] = ("F3", "F4", "C3", "C4", "O1", "O2")


# --------------------------------------------------------------------------- #
# Processing -- slow-wave detection (see infraslow.processing.sws).
#
# Detection thresholds passed to the vendored ``sw_detect``. Every value here
# matches YASA's own default except ``DEFAULT_FREQ_SW``, which is this repo's
# stated protocol ("EEG signals were bandpass filtered between 0.1 and 4 Hz
# [...]") rather than YASA's own default of (0.3, 1.5) Hz.
# --------------------------------------------------------------------------- #
DEFAULT_FREQ_SW: Tuple[float, float] = (0.1, 4.0)
DEFAULT_DUR_NEG: Tuple[float, float] = (0.3, 1.5)
DEFAULT_DUR_POS: Tuple[float, float] = (0.1, 1.0)
DEFAULT_AMP_NEG: Tuple[float, float] = (40.0, 200.0)
DEFAULT_AMP_POS: Tuple[float, float] = (10.0, 150.0)
DEFAULT_AMP_PTP: Tuple[float, float] = (75.0, 350.0)

#: YASA hardcodes both transition bandwidths to 0.2 Hz. That stays the default
#: for the upper edge (4 Hz has plenty of headroom below Nyquist), but the
#: lower edge needs to shrink automatically once ``freq_sw[0]`` drops much
#: below 0.25 Hz, or MNE's stop-band computation (``freq_sw[0] -
#: l_trans_bandwidth``) goes negative and raises -- see
#: ``sws._auto_l_trans_bandwidth``.
DEFAULT_H_TRANS_BANDWIDTH: float = 0.2


# --------------------------------------------------------------------------- #
# Processing -- infraslow (~0.02 Hz) oscillation analysis (see
# infraslow.processing.infraslow).
# --------------------------------------------------------------------------- #
#: Sigma (spindle) band the ISO sigma-power series is integrated over (Hz).
#: Matches the reference ``get_iso`` (Liu et al.), which uses 11-16 Hz for the
#: multitaper sigma-power estimate -- deliberately wider than the detector's
#: ``detection.spindles_detect`` ``freq_sp`` (12-15 Hz), as this is a power
#: integration, not a detection band.
DEFAULT_SIGMA_BAND: Tuple[float, float] = (11.0, 16.0)
DEFAULT_DELTA_BAND: Tuple[float, float] = (0.1, 4.0)
#: Infraslow band the sigma-power oscillation is expected to peak in (Hz).
#: Widened to 0.01-0.1 Hz (~10-100 s) around the canonical ~0.02 Hz (50 s)
#: sigma-power rhythm, so peak-finding and band power stay anchored to it while
#: capturing slower and faster infraslow components.
DEFAULT_INFRASLOW_BAND: Tuple[float, float] = (0.01, 0.1)
#: (Hz). 1 Hz is far above the infraslow Nyquist (>0.1*2) and keeps the series compact.
DEFAULT_SF_ENV: float = 1.0
#: Frequency-grid-spacing target (seconds) for the infraslow spectrum -- long enough to
#: resolve the 0.01 Hz lower edge of DEFAULT_INFRASLOW_BAND (0.01 Hz natural spacing at
#: 100 s; see infraslow_spectrum).
DEFAULT_WINDOW_SEC: float = 100.0
#: Band (Hz) the bi-Gaussian ISFS fit (:func:`~infraslow.processing.infraslow.fit_isfs`)
#: uses to estimate the noise floor for its detection threshold and chromatogram baseline.
DEFAULT_BASELINE_BAND: Tuple[float, float] = (0.06, 0.1)
#: -3 dB low-pass cutoff (Hz), Butterworth order, and Tukey taper fraction the
#: reference ``get_iso`` uses on the sigma-power course before its time-domain
#: ISFS check (:func:`~infraslow.processing.infraslow.isfs_lowpass`) -- peak/trough/
#: zero-crossing scanning for a 25-100 s cycle, independent of ``fit_isfs``'s
#: frequency-domain estimate.
DEFAULT_ISFS_LOWPASS_HZ: float = 0.04
DEFAULT_ISFS_FILTER_ORDER: int = 5
DEFAULT_ISFS_TUKEY_ALPHA: float = 0.1
#: Valid ISFS cycle period (s), matching the reference get_iso: two negative
#: zero-crossings 25-100 s apart (0.01-0.04 Hz), else "Not ISFS".
DEFAULT_ISFS_PERIOD: Tuple[float, float] = (25.0, 100.0)
#: Minimum total event count (spindles, microarousals, slow waves, or slow-wave-
#: spindle coupling) the reference get_iso requires before reporting a
#: participant's phase-bin distribution (``isfs_event_phase_distribution``);
#: below this the denominator is too small to be meaningful.
DEFAULT_ISFS_MIN_EVENTS: int = 5
#: Mirrors ``src/scripts/preprocessing.py``'s ``MIN_BOUT_SEC`` -- the minimum
#: consecutive-stage bout length (s) preprocessing.py requires before it will
#: ever write a bout to ``bouts.npz``. ``pipeline.py`` re-applies this as a
#: defensive filter rather than blindly trusting that every upstream artifact
#: satisfies the invariant.
DEFAULT_MIN_BOUT_SEC: float = 200.0


# --------------------------------------------------------------------------- #
# Viz -- shared visual identity across infraslow.viz.
# --------------------------------------------------------------------------- #
SEABORN_CONTEXT = "talk"
SEABORN_STYLE = "whitegrid"
SEABORN_PALETTE = "deep"


__all__ = [
    # Paths
    "METADATA_DIR", "REPO_ROOT", "DEFAULT_METADATA", "DEFAULT_METADATA2",
    "DEFAULT_DRUG_METADATA", "DEFAULT_EDF_DIR", "DEFAULT_HYPNO_DIR",
    "DEFAULT_USLEEP_HYPNODENSITY_DIR", "DEFAULT_DRUG_EXCLUDE_LIST",
    # I/O
    "DEFAULT_TIMESTAMP_COLUMN", "DEFAULT_STAGING_DIRNAME",
    "DEFAULT_HYPNODENSITY_SUFFIX", "USLEEP_STAGE_ORDER", "DEFAULT_USLEEP_EPOCH_SEC",
    "DEFAULT_HYPNOGRAM_EPOCH_SEC", "DEFAULT_USLEEP_ALIGN_TOLERANCE_SEC",
    "N_IO_WORKERS", "BIOSERENITY_ALIAS_MAP",
    # Processing
    "DEFAULT_TARGET_SFREQ", "NREM_STAGES", "DEFAULT_EPOCH_SEC",
    "DEFAULT_STAGE_MAP", "DEFAULT_EEG_CHANNELS", "DEFAULT_FREQ_SW",
    "DEFAULT_DUR_NEG", "DEFAULT_DUR_POS", "DEFAULT_AMP_NEG", "DEFAULT_AMP_POS",
    "DEFAULT_AMP_PTP", "DEFAULT_H_TRANS_BANDWIDTH", "DEFAULT_SIGMA_BAND",
    "DEFAULT_DELTA_BAND", "DEFAULT_INFRASLOW_BAND", "DEFAULT_SF_ENV",
    "DEFAULT_WINDOW_SEC", "DEFAULT_BASELINE_BAND", "DEFAULT_ISFS_LOWPASS_HZ",
    "DEFAULT_ISFS_FILTER_ORDER", "DEFAULT_ISFS_TUKEY_ALPHA",
    "DEFAULT_ISFS_PERIOD", "DEFAULT_ISFS_MIN_EVENTS", "DEFAULT_MIN_BOUT_SEC",
    # Viz
    "SEABORN_CONTEXT", "SEABORN_STYLE", "SEABORN_PALETTE",
]
