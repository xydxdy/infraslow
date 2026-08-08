# src/infraslow/pipeline/pipeline.py
"""Orchestration: per-subject/state/channel feature extraction, assembled
into the CSVs/figures the spec's "Suggested Output Structure" describes.

Every step here is a thin composition of Tasks 2-7's pure functions -- this
module itself does no signal processing and never re-runs detection.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from ..constants import DEFAULT_METADATA, DEFAULT_METADATA2, DEFAULT_MIN_BOUT_SEC
from . import features as pft
from . import figures as pfg
from . import io as pio
from . import phase as pph
from . import reporting as prep
from . import spectrum as psp

logger = logging.getLogger(__name__)

_BANDS = ("sigma", "delta")
_SPECTRUM_VALUE_COLS = [
    "peak_freq_hz", "peak_period_s", "bandwidth_hz", "auc", "chromatogram_peak_area",
    "bi_gaussian_amp", "bi_gaussian_mu", "bi_gaussian_sd_l", "bi_gaussian_sd_r",
]
_PHASE_VALUE_COLS = ["event_count", "mean_phase", "resultant_length"]
_FEATURE_VALUE_COLS = [
    "spindle_count", "spindle_density_per_min", "slow_wave_count", "slow_wave_density_per_min",
    "sigma_power_db", "delta_power_db",
]


@dataclass
class PipelineConfig:
    input_dir: Path
    output_dir: Path
    sleep_states: Tuple[str, ...] = ("N2", "N3")
    channels: Optional[Tuple[str, ...]] = None
    metadata_path: str = DEFAULT_METADATA
    metadata2_path: str = DEFAULT_METADATA2
    sleep_statistics_path: Optional[Path] = None

    def __post_init__(self) -> None:
        self.input_dir = Path(self.input_dir)
        self.output_dir = Path(self.output_dir)
        if self.sleep_statistics_path is None:
            self.sleep_statistics_path = self.input_dir / "sleep_statistics.csv"
        else:
            self.sleep_statistics_path = Path(self.sleep_statistics_path)

    @property
    def data_dir(self) -> Path:
        return self.input_dir / "data"


def run_subject_state_channel(
    data_dir: Path, subject_id: str, channel: str, state: str, *, return_curves: bool = False,
) -> Union[
    Tuple[Optional[dict], List[dict], Optional[dict]],
    Tuple[Optional[dict], List[dict], Optional[dict], Optional[dict]],
]:
    """`(subject_state_record_or_None, bout_records, failure_or_None)` for
    one `(subject, channel, state)`. A `None` record with a non-`None`
    failure means either an unreadable artifact (`stage` names which loader
    failed) or -- not an error -- zero spindle-containing bouts for this
    state (`stage="bouts"`), matching the spec's "keep N2, report N3
    unavailable" requirement rather than writing a NaN-filled row.

    When `return_curves` is set, a 4th element (the `freqs`/`rel`/`corrected`
    intermediate spectrum arrays, or `None` when no record was produced) is
    appended to the returned tuple; the default 3-tuple stays unchanged for
    every existing caller."""
    subject_dir = Path(data_dir) / subject_id
    try:
        bouts = pio.load_stage_bouts(subject_dir, channel, state)
    except (FileNotFoundError, OSError) as exc:
        failure = dict(subject_id=subject_id, sleep_state=state, channel=channel,
                        stage="bouts", error=str(exc))
        return (None, [], failure, None) if return_curves else (None, [], failure)

    # Defensive: preprocessing.py's MIN_BOUT_SEC (see src/scripts/preprocessing.py) already
    # keeps bouts.npz's "all"/"spindle" bouts >= this length in a real run, but pipeline.py
    # should not blindly trust that every upstream artifact satisfies that invariant.
    if bouts["spindle"].shape[0]:
        durations = bouts["spindle"][:, 1] - bouts["spindle"][:, 0]
        bouts["spindle"] = bouts["spindle"][durations >= DEFAULT_MIN_BOUT_SEC]

    if bouts["spindle"].shape[0] == 0:
        failure = dict(subject_id=subject_id, sleep_state=state, channel=channel,
                        stage="bouts", error="no spindle-containing bouts for this state")
        return (None, [], failure, None) if return_curves else (None, [], failure)

    try:
        isfs = pio.load_stage_isfs_spectra(subject_dir, channel, state, "sigma")
        selected = psp.select_spindle_bouts(bouts, isfs)
        spectrum_feats, curves = psp.compute_subject_spectrum_features(
            selected["freqs"], selected["psds"], return_curves=True,
        )
        bout_peak_freqs = psp.compute_bout_peak_freqs(selected["freqs"], selected["psds"])

        t_env, filtered = pio.load_temporal_isfs(subject_dir, channel, "sigma")
        spindle_summary = pio.load_spindle_summary(subject_dir, channel, state)
        event_times = spindle_summary["Peak"].to_numpy() if "Peak" in spindle_summary.columns else np.empty(0)
        bouts_data = pph.build_bout_phase_data(t_env, filtered, bouts["spindle"], event_times)
        phase_feats = pph.compute_subject_phase_features(bouts_data)

        sw_bouts = pio.load_stage_sw_bouts(subject_dir, channel, state)
        sw_summary = pio.load_sw_summary(subject_dir, channel, state)
        spindle_feats = pft.compute_spindle_features(spindle_summary, bouts["all"])
        sw_feats = pft.compute_slow_wave_features(sw_summary, sw_bouts["all"])

        band_feats: Dict[str, float] = {}
        for band in _BANDS:
            b_t_env, b_power = pio.load_envelope(subject_dir, channel, band)
            band_feats.update(pft.compute_band_power_feature(b_t_env, b_power, bouts["all"], name=band))
    except (FileNotFoundError, OSError, KeyError, ValueError) as exc:
        logger.exception("subject=%s channel=%s state=%s failed", subject_id, channel, state)
        failure = dict(subject_id=subject_id, sleep_state=state, channel=channel,
                        stage="features", error=str(exc))
        return (None, [], failure, None) if return_curves else (None, [], failure)

    record = dict(subject_id=subject_id, sleep_state=state, channel=channel)
    record.update(spectrum_feats)
    record.update(spindle_feats)
    record.update(sw_feats)
    record.update(band_feats)
    if phase_feats is not None:
        record.update(phase_feats)
    else:
        record.update({k: np.nan for k in ("event_count", "n_in_isfs", "preferred_phase",
                                            "mean_phase", "resultant_length")})
        record["phase_bin_counts"] = None
        record["phase_bin_rates"] = None

    # bout_start (from `selected`) only carries each bout's start time; build (start, stop)
    # bout records straight from bouts["spindle"] instead -- exact same values (same source
    # array `select_spindle_bouts` matched against), no lookup needed.
    bout_records = []
    for i, (a, b) in enumerate(bouts["spindle"]):
        bout_records.append(dict(
            subject_id=subject_id, sleep_state=state, channel=channel, bout_id=i,
            bout_start=float(a), bout_stop=float(b), bout_duration=float(b - a),
        ))

    return (record, bout_records, None, curves) if return_curves else (record, bout_records, None)


def _discover_channels(config: PipelineConfig, subject_id: str) -> Tuple[str, ...]:
    if config.channels is not None:
        return config.channels
    return tuple(pio.discover_channels(config.data_dir, subject_id))


def run_pipeline(config: PipelineConfig) -> None:
    """Full run: every subject x channel x sleep_state, then aggregate/write."""
    subjects = pio.discover_subjects(config.data_dir)
    logger.info("%d subject(s) found under %s", len(subjects), config.data_dir)

    subject_state_records: List[dict] = []
    bout_records: List[dict] = []
    failures: List[dict] = []
    phase_dists_by_state: Dict[str, List[dict]] = {s: [] for s in config.sleep_states}
    bout_peaks_by_state: Dict[str, List[float]] = {s: [] for s in config.sleep_states}
    example_spectrum_by_state: Dict[str, Tuple[dict, dict]] = {}

    for subject_id in subjects:
        for channel in _discover_channels(config, subject_id):
            for state in config.sleep_states:
                record, brecs, failure, curves = run_subject_state_channel(
                    config.data_dir, subject_id, channel, state, return_curves=True,
                )
                if failure is not None:
                    failures.append(failure)
                if record is not None:
                    subject_state_records.append(record)
                    bout_records.extend(brecs)
                    if record.get("phase_bin_counts") is not None:
                        phase_dists_by_state[state].append(dict(
                            event_count=record["event_count"], n_in_isfs=record["n_in_isfs"],
                            phase_bin_counts=record["phase_bin_counts"],
                            phase_bin_rates=record["phase_bin_rates"],
                        ))
                    if state not in example_spectrum_by_state and not np.isnan(record.get("peak_freq_hz", np.nan)):
                        example_spectrum_by_state[state] = (record, curves)

    subject_state_df = prep.build_subject_state_features_df(subject_state_records)
    demographics = pio.load_demographics(config.metadata_path, config.metadata2_path)
    sleep_stats = pio.load_sleep_statistics(config.sleep_statistics_path)
    subject_state_df = prep.join_demographics_and_sleep_stats(subject_state_df, demographics, sleep_stats)
    prep.write_csv(subject_state_df, config.output_dir / "subject_state_features.csv")

    bout_df = prep.build_bout_features_df(bout_records)
    prep.write_csv(bout_df, config.output_dir / "spectrum" / "bout_spectrum_features.csv")

    value_cols = [c for c in (_SPECTRUM_VALUE_COLS + _PHASE_VALUE_COLS + _FEATURE_VALUE_COLS)
                  if c in subject_state_df.columns]
    if len(subject_state_df) and value_cols:
        summary_df = prep.cohort_summary(subject_state_df, value_cols=value_cols)
    else:
        summary_df = prep.cohort_summary(
            subject_state_df.assign(**{c: np.nan for c in value_cols}) if len(subject_state_df) else
            subject_state_df, value_cols=value_cols,
        ) if value_cols else prep.build_bout_features_df([])
    prep.write_csv(summary_df, config.output_dir / "cohort_summary.csv")

    prep.write_csv(prep.build_bout_features_df(failures), config.output_dir / "failures.csv")

    for state in config.sleep_states:
        peaks = np.asarray(bout_peaks_by_state.get(state, []))
        pfg.plot_peak_frequency_distribution(
            peaks, state=state, out_path=config.output_dir / "figures" / f"{state}_peak_frequency_distribution.png",
        )
        if phase_dists_by_state[state]:
            pooled = pph.pool_phase_distributions(phase_dists_by_state[state])
        else:
            pooled = dict(event_count=0, n_in_isfs=0, phase_bin_counts=[0] * 8, phase_bin_rates=[0.0] * 8)
        pfg.plot_spindle_phase_distribution(
            pooled, state=state, out_path=config.output_dir / "figures" / f"{state}_spindle_phase_distribution.png",
        )
        if state in example_spectrum_by_state:
            rec, curves = example_spectrum_by_state[state]
            popt = (rec.get("bi_gaussian_amp", np.nan), rec.get("bi_gaussian_mu", np.nan),
                    rec.get("bi_gaussian_sd_l", np.nan), rec.get("bi_gaussian_sd_r", np.nan))
            pfg.plot_relative_spectrum_bigaussian(
                curves["freqs"], curves["rel"], curves["corrected"], rec, popt, state=state,
                out_path=config.output_dir / "figures" / f"{state}_relative_spectral_power_bigaussian.png",
            )
        else:
            nan_freqs = np.linspace(0.0025, 0.1, 40)
            nan_curve = np.full_like(nan_freqs, np.nan)
            pfg.plot_relative_spectrum_bigaussian(
                nan_freqs, nan_curve, nan_curve, {}, (np.nan,) * 4, state=state,
                out_path=config.output_dir / "figures" / f"{state}_relative_spectral_power_bigaussian.png",
            )

    logger.info("wrote %d subject-state record(s), %d bout record(s), %d failure(s) -> %s",
                len(subject_state_records), len(bout_records), len(failures), config.output_dir)


__all__ = ["PipelineConfig", "run_subject_state_channel", "run_pipeline"]
