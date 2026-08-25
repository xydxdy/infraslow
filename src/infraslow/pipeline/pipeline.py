# src/infraslow/pipeline/pipeline.py
"""Orchestration: per-subject/state/channel feature extraction, assembled
into the CSVs/figures the spec's "Suggested Output Structure" describes.

Every step here is a thin composition of Tasks 2-7's pure functions -- this
module itself does no signal processing and never re-runs detection.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from ..constants import DEFAULT_METADATA, DEFAULT_METADATA2, DEFAULT_MIN_BOUT_SEC
from . import features as pft
from . import figures as pfg
from . import io as pio
from . import phase as pph
from . import reporting as prep
from . import spectrum as psp

logger = logging.getLogger(__name__)

_BANDS = ("sigma", "delta")

# Which artifacts drive the ISFS spectrum + phase-locking analysis in
# `run_subject_state_channel`, per `event`. "spindle"/sigma is the original (and
# default) pathway; "sw"/delta is the slow-wave analog -- `preprocessing.py` writes
# both symmetrically (`bouts.npz["spindle"]` + `spindel_yasa.csv["Peak"]` vs.
# `sw_bouts.npz["sw"]` + `sw_yasa.csv["NegPeak"]`, both against the same underlying
# `all_bouts`), so this is purely a which-artifacts-to-read switch, not new science.
_EVENT_SPECS = {
    "spindle": dict(band="sigma", bouts_key="spindle",
                     load_bouts=pio.load_stage_bouts, load_summary=pio.load_spindle_summary,
                     peak_col="Peak"),
    "sw": dict(band="delta", bouts_key="sw",
                load_bouts=pio.load_stage_sw_bouts, load_summary=pio.load_sw_summary,
                peak_col="NegPeak"),
}
_SPECTRUM_VALUE_COLS = [
    "peak_freq_hz", "peak_period_s", "bandwidth_hz", "auc", "chromatogram_peak_area",
    "bi_gaussian_amp", "bi_gaussian_mu", "bi_gaussian_sd_l", "bi_gaussian_sd_r",
]
_PHASE_VALUE_COLS = ["event_count", "resultant_length"]
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
    data_dir: Path, subject_id: str, channel: str, state: str, *,
    event: str = "spindle", return_curves: bool = False,
) -> Union[
    Tuple[Optional[dict], List[dict], Optional[dict]],
    Tuple[Optional[dict], List[dict], Optional[dict], Optional[dict]],
]:
    """`(subject_state_record_or_None, bout_records, failure_or_None)` for
    one `(subject, channel, state)`. A `None` record with a non-`None`
    failure means either an unreadable artifact (`stage` names which loader
    failed) or -- not an error -- zero `event`-containing bouts for this
    state (`stage="bouts"`), matching the spec's "keep N2, report N3
    unavailable" requirement rather than writing a NaN-filled row.

    `event` selects which artifact pair drives the ISFS spectrum + phase-locking
    analysis (see `_EVENT_SPECS`): `"spindle"` (default, unchanged behavior) pairs
    sigma-band ISFS spectra with `bouts.npz["spindle"]`/`spindel_yasa.csv["Peak"]`;
    `"sw"` is the slow-wave analog, pairing delta-band ISFS spectra with
    `sw_bouts.npz["sw"]`/`sw_yasa.csv["NegPeak"]`. Either way, `spindle_count`/
    `slow_wave_count`/`sigma_power_db`/`delta_power_db` are always computed for
    *both* event types and bands in the returned record -- `event` only picks which
    one the spectrum/phase-bin/peak-frequency fields (and `bout_records`) describe.

    When `return_curves` is set, a 4th element (the `freqs`/`rel`/`corrected`/
    `bout_peak_freqs` intermediate spectrum arrays, or `None` when no record
    was produced) is appended to the returned tuple; the default 3-tuple
    stays unchanged for every existing caller."""
    spec = _EVENT_SPECS[event]
    band, bouts_key = spec["band"], spec["bouts_key"]
    subject_dir = Path(data_dir) / subject_id
    try:
        event_bouts = spec["load_bouts"](subject_dir, channel, state)
    except (FileNotFoundError, OSError) as exc:
        failure = dict(subject_id=subject_id, sleep_state=state, channel=channel,
                        stage="bouts", error=str(exc))
        return (None, [], failure, None) if return_curves else (None, [], failure)

    # Defensive: preprocessing.py's MIN_BOUT_SEC (see src/scripts/preprocessing.py) already
    # keeps bouts.npz's/sw_bouts.npz's "all"/event bouts >= this length in a real run, but
    # pipeline.py should not blindly trust that every upstream artifact satisfies that
    # invariant.
    if event_bouts[bouts_key].shape[0]:
        durations = event_bouts[bouts_key][:, 1] - event_bouts[bouts_key][:, 0]
        event_bouts[bouts_key] = event_bouts[bouts_key][durations >= DEFAULT_MIN_BOUT_SEC]

    if event_bouts[bouts_key].shape[0] == 0:
        failure = dict(subject_id=subject_id, sleep_state=state, channel=channel,
                        stage="bouts", error=f"no {event}-containing bouts for this state")
        return (None, [], failure, None) if return_curves else (None, [], failure)

    try:
        isfs = pio.load_stage_isfs_spectra(subject_dir, channel, state, band)
        selected = psp.select_event_bouts(event_bouts[bouts_key], isfs)
        spectrum_feats, curves = psp.compute_subject_spectrum_features(
            selected["freqs"], selected["psds"], return_curves=True,
        )
        bout_peak_freqs = psp.compute_bout_peak_freqs(selected["freqs"], selected["psds"])

        t_env, filtered = pio.load_temporal_isfs(subject_dir, channel, band)
        event_summary = spec["load_summary"](subject_dir, channel, state)
        peak_col = spec["peak_col"]
        event_times = event_summary[peak_col].to_numpy() if peak_col in event_summary.columns else np.empty(0)
        bouts_data = pph.build_bout_phase_data(t_env, filtered, event_bouts[bouts_key], event_times)
        phase_feats = pph.compute_subject_phase_features(bouts_data)

        # spindle_count/slow_wave_count/sigma_power_db/delta_power_db are always computed
        # for both event types and both bands, regardless of `event` -- reuse event_bouts/
        # event_summary instead of a redundant re-load when they already *are* the type
        # being reused here.
        bouts = event_bouts if event == "spindle" else pio.load_stage_bouts(subject_dir, channel, state)
        sw_bouts = event_bouts if event == "sw" else pio.load_stage_sw_bouts(subject_dir, channel, state)
        spindle_summary = event_summary if event == "spindle" else pio.load_spindle_summary(
            subject_dir, channel, state)
        sw_summary = event_summary if event == "sw" else pio.load_sw_summary(subject_dir, channel, state)
        spindle_feats = pft.compute_spindle_features(spindle_summary, bouts["all"])
        sw_feats = pft.compute_slow_wave_features(sw_summary, sw_bouts["all"])

        band_feats: Dict[str, float] = {}
        for b in _BANDS:
            b_t_env, b_power = pio.load_envelope(subject_dir, channel, b)
            band_feats.update(pft.compute_band_power_feature(b_t_env, b_power, bouts["all"], name=b))
    except Exception as exc:  # noqa: BLE001 - one bad subject/channel/state must not sink the whole run
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
    # bout records straight from event_bouts[bouts_key] instead -- exact same values (same
    # source array `select_event_bouts` matched against), no lookup needed.
    bout_records = []
    for i, (a, b) in enumerate(event_bouts[bouts_key]):
        bout_records.append(dict(
            subject_id=subject_id, sleep_state=state, channel=channel, bout_id=i,
            bout_start=float(a), bout_stop=float(b), bout_duration=float(b - a),
        ))

    curves["bout_peak_freqs"] = bout_peak_freqs
    return (record, bout_records, None, curves) if return_curves else (record, bout_records, None)


def find_paired_subject_data(
    data_dir: Path, channel: str, states: Sequence[str] = ("N2", "N3"), *,
    limit: Optional[int] = None, event: str = "spindle",
) -> Dict[str, Dict[str, Tuple[dict, List[dict], dict]]]:
    """The first `limit` subjects (sorted by id, via `pio.discover_subjects`) with a
    valid (non-`None`) record for *every* one of `states` on `channel` -- every eligible
    subject if `limit` is `None`.

    `event` is forwarded to `run_subject_state_channel` (see its docstring):
    `"spindle"` (default) for the sigma/spindle ISFS + phase-locking pathway, `"sw"`
    for the delta/slow-wave analog.

    Returns `{subject_id: {state: (record, bout_records, curves)}}`, reusing
    `run_subject_state_channel(..., return_curves=True)` per subject/state so the same
    spectrum/phase computation is not repeated later for analysis -- a caller that only
    needs the subject id list can do `list(result)`, already in sorted order. A subject
    failing on any one state (e.g. no `event`-containing bouts) is dropped entirely rather
    than included with a missing state, so every returned subject has a complete pair.

    Computes every requested state for a candidate subject even when an earlier state in
    `states` turns out to be the one that's actually valid and a later one disqualifies the
    subject -- deliberately simple (no bout-count pre-check) since this targets small demo
    selections (e.g. the first 10 of a cohort), not a full-cohort scan.
    """
    out: Dict[str, Dict[str, Tuple[dict, List[dict], dict]]] = {}
    for subject_id in pio.discover_subjects(data_dir):
        per_state: Dict[str, Tuple[dict, List[dict], dict]] = {}
        for state in states:
            record, bout_records, _failure, curves = run_subject_state_channel(
                data_dir, subject_id, channel, state, event=event, return_curves=True,
            )
            if record is None:
                break
            per_state[state] = (record, bout_records, curves)
        if len(per_state) == len(states):
            out[subject_id] = per_state
            if limit is not None and len(out) >= limit:
                break
    return out


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
        try:
            channels = _discover_channels(config, subject_id)
        except (FileNotFoundError, NotADirectoryError, OSError) as exc:
            logger.exception("subject=%s channel discovery failed", subject_id)
            failures.append(dict(subject_id=subject_id, sleep_state=None, channel=None,
                                  stage="channels", error=str(exc)))
            continue
        for channel in channels:
            for state in config.sleep_states:
                record, brecs, failure, curves = run_subject_state_channel(
                    config.data_dir, subject_id, channel, state, return_curves=True,
                )
                if failure is not None:
                    failures.append(failure)
                if record is not None:
                    subject_state_records.append(record)
                    bout_records.extend(brecs)
                    bout_peaks_by_state[state].extend(curves["bout_peak_freqs"].tolist())
                    if record.get("phase_bin_counts") is not None:
                        phase_dists_by_state[state].append(dict(
                            event_count=record["event_count"], n_in_isfs=record["n_in_isfs"],
                            phase_bin_counts=record["phase_bin_counts"],
                            phase_bin_rates=record["phase_bin_rates"],
                        ))
                    if state not in example_spectrum_by_state and not np.isnan(record.get("peak_freq_hz", np.nan)):
                        example_spectrum_by_state[state] = (record, curves)

    # Pool each state's per-subject phase distributions once here (not per
    # figure) so the circular cohort-mean-phase row below and the Figure 3
    # phase-bin bar chart later both use the exact same pooled counts.
    pooled_by_state: Dict[str, dict] = {}
    for state in config.sleep_states:
        if phase_dists_by_state[state]:
            pooled_by_state[state] = pph.pool_phase_distributions(phase_dists_by_state[state])
        else:
            pooled_by_state[state] = dict(event_count=0, n_in_isfs=0, phase_bin_counts=[0] * 8,
                                           phase_bin_rates=[0.0] * 8)

    subject_state_df = prep.build_subject_state_features_df(subject_state_records)
    demographics = pio.load_demographics(config.metadata_path, config.metadata2_path)
    sleep_stats = pio.load_sleep_statistics(config.sleep_statistics_path)
    subject_state_df = prep.join_demographics_and_sleep_stats(subject_state_df, demographics, sleep_stats)
    prep.write_csv(subject_state_df, config.output_dir / "subject_state_features.csv")

    bout_df = prep.build_bout_features_df(bout_records)
    prep.write_csv(bout_df, config.output_dir / "spectrum" / "bout_spectrum_features.csv")

    value_cols = [c for c in (_SPECTRUM_VALUE_COLS + _PHASE_VALUE_COLS + _FEATURE_VALUE_COLS)
                  if c in subject_state_df.columns]
    summary_df = prep.cohort_summary(subject_state_df, group_col=["sleep_state", "channel"], value_cols=value_cols)

    # mean_phase/preferred_phase are circular quantities (angles in (-pi, pi]) -- a
    # linear mean/std across subjects (what cohort_summary computes for every other
    # value_col) is not meaningful for them and is deliberately excluded from
    # _PHASE_VALUE_COLS above. Instead, add one circular-mean-phase row per state,
    # computed from the pooled (cross-subject, cross-channel) 8-bin phase-bin counts
    # via the standard mean-resultant-vector formula: weighted resultant
    # C = sum(c_i * cos(angle_i)), S = sum(c_i * sin(angle_i)), circular mean =
    # atan2(S, C). channel="ALL" marks this as a cross-channel pooled statistic
    # (phase distributions are already pooled across channels by design elsewhere
    # in this module); states with zero pooled in-cycle events are skipped.
    bin_angles = pph.bin_center_angle(np.arange(1, 9))
    circular_rows = []
    for state in config.sleep_states:
        pooled = pooled_by_state[state]
        n_in_isfs = int(pooled["n_in_isfs"])
        if n_in_isfs == 0:
            continue
        counts = np.asarray(pooled["phase_bin_counts"], dtype=float)
        c = float((counts * np.cos(bin_angles)).sum())
        s = float((counts * np.sin(bin_angles)).sum())
        circular_rows.append(dict(
            sleep_state=state, channel="ALL", variable="cohort_mean_phase_circular",
            n=n_in_isfs, mean=float(np.arctan2(s, c)),
            std=np.nan, sem=np.nan, median=np.nan, q25=np.nan, q75=np.nan, min=np.nan, max=np.nan,
        ))
    if circular_rows:
        summary_df = pd.concat([summary_df, pd.DataFrame(circular_rows)], ignore_index=True)

    prep.write_csv(summary_df, config.output_dir / "cohort_summary.csv")

    prep.write_csv(prep.build_bout_features_df(failures), config.output_dir / "failures.csv")

    for state in config.sleep_states:
        peaks = np.asarray(bout_peaks_by_state.get(state, []))
        pfg.plot_peak_frequency_distribution(
            peaks, state=state, out_path=config.output_dir / "figures" / f"{state}_peak_frequency_distribution.png",
        )
        pooled = pooled_by_state[state]
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


__all__ = ["PipelineConfig", "run_subject_state_channel", "find_paired_subject_data", "run_pipeline"]
