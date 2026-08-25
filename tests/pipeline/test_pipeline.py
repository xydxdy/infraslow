# tests/pipeline/test_pipeline.py
from __future__ import annotations

import numpy as np
import pandas as pd

from infraslow.pipeline import pipeline as ppl
from infraslow.pipeline import spectrum as psp


def test_run_subject_state_channel_n2_has_spindle_bouts(fake_subject_tree):
    record, bout_records, failure = ppl.run_subject_state_channel(
        fake_subject_tree, "SUBJ001", "C3", "N2",
    )
    assert failure is None
    assert record is not None
    assert record["subject_id"] == "SUBJ001"
    assert record["sleep_state"] == "N2"
    assert record["channel"] == "C3"
    assert "peak_freq_hz" in record
    assert "spindle_count" in record
    assert "sigma_power_db" in record
    assert len(bout_records) == 3
    for br in bout_records:
        assert br["sleep_state"] == "N2"
        assert br["channel"] == "C3"


def test_run_subject_state_channel_return_curves_includes_bout_peak_freqs(fake_subject_tree):
    # Figure 2 (N2/N3_peak_frequency_distribution.png) is built from bout_peaks_by_state,
    # which run_pipeline populates from curves["bout_peak_freqs"] -- confirm
    # run_subject_state_channel actually threads that key through the 4th tuple element
    # when return_curves=True, matching the fixture's 3 N2 spindle bouts.
    record, bout_records, failure, curves = ppl.run_subject_state_channel(
        fake_subject_tree, "SUBJ001", "C3", "N2", return_curves=True,
    )
    assert failure is None
    assert record is not None
    assert len(bout_records) == 3
    assert "bout_peak_freqs" in curves
    bout_peak_freqs = np.asarray(curves["bout_peak_freqs"])
    assert bout_peak_freqs.shape == (3,)


def test_run_subject_state_channel_n3_has_no_valid_bouts_returns_none_record(fake_subject_tree):
    # The fixture's N3 bout is 45s, below the 200s min_dur -- preprocessing.py itself
    # would never have written it to bouts.npz["all"] in a real run, but bouts.npz IS
    # still present (io.py doesn't filter by duration -- see test_io.py's note), so
    # this exercises the "state has bouts.npz but zero SPINDLE bouts -> unavailable"
    # path, not a missing-file path.
    record, bout_records, failure = ppl.run_subject_state_channel(
        fake_subject_tree, "SUBJ001", "C3", "N3",
    )
    assert record is None
    assert bout_records == []
    assert failure is not None
    assert failure["subject_id"] == "SUBJ001"
    assert failure["sleep_state"] == "N3"


def test_run_subject_state_channel_missing_subject_returns_failure(fake_subject_tree):
    record, bout_records, failure = ppl.run_subject_state_channel(
        fake_subject_tree, "DOES_NOT_EXIST", "C3", "N2",
    )
    assert record is None
    assert failure is not None
    assert "stage" in failure


def test_run_pipeline_end_to_end_writes_expected_files(fake_subject_tree, tmp_path):
    output_dir = tmp_path / "analysis_output"
    config = ppl.PipelineConfig(
        input_dir=fake_subject_tree.parent, output_dir=output_dir,
        sleep_statistics_path=tmp_path / "no_such_sleep_stats.csv",
    )
    ppl.run_pipeline(config)

    assert (output_dir / "subject_state_features.csv").is_file()
    assert (output_dir / "cohort_summary.csv").is_file()
    assert (output_dir / "failures.csv").is_file()
    assert (output_dir / "spectrum" / "bout_spectrum_features.csv").is_file()
    assert (output_dir / "figures" / "N2_relative_spectral_power_bigaussian.png").is_file()
    assert (output_dir / "figures" / "N2_peak_frequency_distribution.png").is_file()
    assert (output_dir / "figures" / "N2_spindle_phase_distribution.png").is_file()

    subject_state = pd.read_csv(output_dir / "subject_state_features.csv")
    # Only N2 has valid (spindle-containing) bouts in the fixture -- N3 is absent,
    # not a NaN row, per the "keep N2, report N3 unavailable" spec requirement.
    assert list(subject_state["sleep_state"]) == ["N2"]

    failures = pd.read_csv(output_dir / "failures.csv")
    assert (failures["sleep_state"] == "N3").any()


def test_run_subject_state_channel_catches_runtime_error_not_in_old_narrow_tuple(fake_subject_tree, monkeypatch):
    # scipy.optimize.curve_fit (inside spectrum.py's fit_isfs) raises RuntimeError on
    # non-convergence -- a real possibility on a degenerate real-world spectrum, but
    # NOT one of the exception types the old narrow `except (FileNotFoundError, OSError,
    # KeyError, ValueError)` clause caught. Confirm the broadened `except Exception`
    # catches it and returns a failure dict instead of propagating.
    def _raise_runtime_error(*args, **kwargs):
        raise RuntimeError("curve_fit did not converge")

    monkeypatch.setattr(ppl.psp, "compute_subject_spectrum_features", _raise_runtime_error)

    record, bout_records, failure = ppl.run_subject_state_channel(
        fake_subject_tree, "SUBJ001", "C3", "N2",
    )
    assert record is None
    assert bout_records == []
    assert failure is not None
    assert failure["stage"] == "features"
    assert "curve_fit" in failure["error"]


def test_run_pipeline_survives_one_subject_state_raising_runtime_error(fake_subject_tree, tmp_path, monkeypatch):
    # Two subjects (duplicate the fixture's one real subject under a second id) so the
    # main loop has more than one subject to iterate. The first call into
    # compute_subject_spectrum_features (SUBJ001's N2, subjects are processed in sorted
    # order) raises RuntimeError; the second (SUBJ002's N2) must still succeed and be
    # written out -- proving one bad subject no longer kills the whole run.
    import shutil

    shutil.copytree(fake_subject_tree / "SUBJ001", fake_subject_tree / "SUBJ002")

    original_fn = psp.compute_subject_spectrum_features
    calls = {"n": 0}

    def _flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("simulated curve_fit non-convergence")
        return original_fn(*args, **kwargs)

    monkeypatch.setattr(ppl.psp, "compute_subject_spectrum_features", _flaky)

    output_dir = tmp_path / "analysis_output"
    config = ppl.PipelineConfig(
        input_dir=fake_subject_tree.parent, output_dir=output_dir,
        sleep_statistics_path=tmp_path / "no_such_sleep_stats.csv",
    )
    ppl.run_pipeline(config)  # must not raise despite one subject/state failing

    subject_state = pd.read_csv(output_dir / "subject_state_features.csv")
    assert len(subject_state) == 1
    assert subject_state["subject_id"].iloc[0] == "SUBJ002"

    failures = pd.read_csv(output_dir / "failures.csv")
    features_failures = failures[failures["stage"] == "features"]
    assert len(features_failures) == 1
    assert features_failures["subject_id"].iloc[0] == "SUBJ001"


def test_run_pipeline_zero_records_for_selected_state_does_not_crash(fake_subject_tree, tmp_path):
    # sleep_states=("N3",) only -- the fixture's N3 bout is below the min_dur threshold,
    # so run_subject_state_channel never produces a record for any subject/channel in
    # this run: subject_state_records stays empty for the whole pipeline. Before the
    # fix, build_subject_state_features_df([]) returned a 0-column frame and the
    # downstream merge/groupby calls raised KeyError.
    output_dir = tmp_path / "analysis_output"
    config = ppl.PipelineConfig(
        input_dir=fake_subject_tree.parent, output_dir=output_dir,
        sleep_states=("N3",),
        sleep_statistics_path=tmp_path / "no_such_sleep_stats.csv",
    )
    ppl.run_pipeline(config)  # must not raise

    assert (output_dir / "subject_state_features.csv").is_file()
    assert (output_dir / "cohort_summary.csv").is_file()
    assert (output_dir / "failures.csv").is_file()

    subject_state = pd.read_csv(output_dir / "subject_state_features.csv")
    assert len(subject_state) == 0

    failures = pd.read_csv(output_dir / "failures.csv")
    assert (failures["sleep_state"] == "N3").any()


def test_find_paired_subject_data_selects_first_n_with_both_states(tmp_path):
    from .conftest import _write_envelope, _write_temporal_isfs, _write_stage

    data_dir = tmp_path / "data"
    bouts = [(50.0, 350.0), (450.0, 750.0)]

    # SUBJ001: valid N2 AND N3 (paired) -- included.
    ch1 = data_dir / "SUBJ001" / "C3"
    _write_envelope(ch1, "sigma", 1200, seed=1)
    _write_envelope(ch1, "delta", 1200, seed=2)
    _write_temporal_isfs(ch1, "sigma", 1200, seed=3)
    _write_temporal_isfs(ch1, "delta", 1200, seed=4)
    _write_stage(ch1, "N2", bouts, seed=10)
    _write_stage(ch1, "N3", bouts, seed=20)

    # SUBJ002: N2 only -- N3's only bout (45s) is below the 200s min_dur, matching
    # fake_subject_tree's own pattern for an "N3 unavailable" subject -- excluded.
    ch2 = data_dir / "SUBJ002" / "C3"
    _write_envelope(ch2, "sigma", 1200, seed=5)
    _write_envelope(ch2, "delta", 1200, seed=6)
    _write_temporal_isfs(ch2, "sigma", 1200, seed=7)
    _write_temporal_isfs(ch2, "delta", 1200, seed=8)
    _write_stage(ch2, "N2", bouts, seed=30)
    _write_stage(ch2, "N3", [(0.0, 45.0)], seed=40)

    # SUBJ003: valid N2 AND N3 -- included (2nd pick once SUBJ002 is skipped).
    ch3 = data_dir / "SUBJ003" / "C3"
    _write_envelope(ch3, "sigma", 1200, seed=9)
    _write_envelope(ch3, "delta", 1200, seed=11)
    _write_temporal_isfs(ch3, "sigma", 1200, seed=12)
    _write_temporal_isfs(ch3, "delta", 1200, seed=13)
    _write_stage(ch3, "N2", bouts, seed=50)
    _write_stage(ch3, "N3", bouts, seed=60)

    result = ppl.find_paired_subject_data(data_dir, "C3", ("N2", "N3"), limit=2)

    assert list(result) == ["SUBJ001", "SUBJ003"]
    for sid in result:
        for state in ("N2", "N3"):
            record, bout_records, curves = result[sid][state]
            assert record["subject_id"] == sid
            assert record["sleep_state"] == state
            assert isinstance(bout_records, list)
            assert {"freqs", "rel", "corrected"}.issubset(curves)


def test_find_paired_subject_data_no_limit_returns_every_eligible_subject(tmp_path):
    from .conftest import _write_envelope, _write_temporal_isfs, _write_stage

    data_dir = tmp_path / "data"
    bouts = [(50.0, 350.0)]
    for sid, seed in (("SUBJ_A", 1), ("SUBJ_B", 2)):
        ch = data_dir / sid / "C3"
        _write_envelope(ch, "sigma", 1200, seed=seed)
        _write_envelope(ch, "delta", 1200, seed=seed + 1)
        _write_temporal_isfs(ch, "sigma", 1200, seed=seed + 2)
        _write_temporal_isfs(ch, "delta", 1200, seed=seed + 3)
        _write_stage(ch, "N2", bouts, seed=seed + 10)
        _write_stage(ch, "N3", bouts, seed=seed + 20)

    result = ppl.find_paired_subject_data(data_dir, "C3", ("N2", "N3"))
    assert list(result) == ["SUBJ_A", "SUBJ_B"]


def test_find_paired_subject_data_excludes_subject_missing_from_every_state(fake_subject_tree):
    # fake_subject_tree's only subject (SUBJ001) has valid N2 but no valid N3 bout --
    # zero fully-paired subjects, not a crash or a partial entry.
    result = ppl.find_paired_subject_data(fake_subject_tree, "C3", ("N2", "N3"))
    assert result == {}


def test_cohort_summary_csv_has_circular_mean_phase_not_linear_mean_phase(fake_subject_tree, tmp_path, monkeypatch):
    # The fixture's N2 only has 3 spindle events, below DEFAULT_ISFS_MIN_EVENTS=5, so
    # compute_subject_phase_features would normally return None (no phase feats, no
    # pooled phase distribution) -- same override (min_events=1) test_phase.py's own
    # test_compute_subject_phase_features_real_data uses to get a real result out of
    # this fixture's 3 events.
    original_fn = ppl.pph.compute_subject_phase_features

    def _force_min_events_1(bouts_data, **kwargs):
        return original_fn(bouts_data, min_events=1)

    monkeypatch.setattr(ppl.pph, "compute_subject_phase_features", _force_min_events_1)

    output_dir = tmp_path / "analysis_output"
    config = ppl.PipelineConfig(
        input_dir=fake_subject_tree.parent, output_dir=output_dir,
        sleep_statistics_path=tmp_path / "no_such_sleep_stats.csv",
    )
    ppl.run_pipeline(config)

    summary = pd.read_csv(output_dir / "cohort_summary.csv")
    # mean_phase is a circular quantity (angle in (-pi, pi]) -- averaged linearly it is
    # not interpretable, so it must no longer appear as a plain cohort_summary variable.
    assert not (summary["variable"] == "mean_phase").any()

    circular = summary[summary["variable"] == "cohort_mean_phase_circular"]
    assert len(circular) == 1
    row = circular.iloc[0]
    assert row["sleep_state"] == "N2"
    assert row["channel"] == "ALL"
    assert -np.pi <= row["mean"] <= np.pi
    assert row["n"] > 0
    assert np.isnan(row["std"])
