# tests/pipeline/test_pipeline.py
from __future__ import annotations

import numpy as np
import pandas as pd

from infraslow.pipeline import pipeline as ppl


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
