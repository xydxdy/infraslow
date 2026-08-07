# tests/pipeline/test_spectrum.py
from __future__ import annotations

import numpy as np
import pytest

from infraslow.pipeline import io as pio
from infraslow.pipeline import spectrum as psp


def test_select_spindle_bouts_matches_all_when_every_bout_has_spindle(fake_subject_tree):
    subject_dir = fake_subject_tree / "SUBJ001"
    bouts = pio.load_stage_bouts(subject_dir, "C3", "N2")
    isfs = pio.load_stage_isfs_spectra(subject_dir, "C3", "N2", "sigma")
    selected = psp.select_spindle_bouts(bouts, isfs)
    assert selected["psds"].shape == (3, 40)
    assert selected["bout_start"].shape == (3,)
    np.testing.assert_array_equal(selected["freqs"], isfs["freqs"])


def test_select_spindle_bouts_empty_when_no_spindle_bouts():
    isfs = {
        "freqs": np.linspace(0.0025, 0.1, 10),
        "psds": np.ones((2, 10)),
        "bout_start": np.array([0.0, 300.0]),
    }
    bouts = {"all": np.array([[0.0, 250.0], [300.0, 550.0]]), "spindle": np.empty((0, 2))}
    selected = psp.select_spindle_bouts(bouts, isfs)
    assert selected["psds"].shape == (0, 10)
    assert selected["bout_start"].shape == (0,)


def test_compute_subject_spectrum_features_real_data(fake_subject_tree):
    subject_dir = fake_subject_tree / "SUBJ001"
    bouts = pio.load_stage_bouts(subject_dir, "C3", "N2")
    isfs = pio.load_stage_isfs_spectra(subject_dir, "C3", "N2", "sigma")
    selected = psp.select_spindle_bouts(bouts, isfs)
    feats = psp.compute_subject_spectrum_features(selected["freqs"], selected["psds"])
    for key in (
        "peak_freq_hz", "peak_period_s", "bandwidth_hz", "auc", "chromatogram_peak_area",
        "bi_gaussian_amp", "bi_gaussian_mu", "bi_gaussian_sd_l", "bi_gaussian_sd_r",
        "isfs_detected", "isfs_threshold", "real_peak_freq_hz", "real_peak_period_s",
    ):
        assert key in feats
    # The fixture's synthetic PSDs have a bump at 0.02 Hz -- the fit should land near there.
    assert 0.01 < feats["peak_freq_hz"] < 0.03
    assert feats["peak_period_s"] == pytest.approx(1.0 / feats["peak_freq_hz"])
    assert feats["auc"] > 0


def test_compute_subject_spectrum_features_empty_psds_returns_nan_dict():
    freqs = np.linspace(0.0025, 0.1, 10)
    psds = np.empty((0, 10))
    feats = psp.compute_subject_spectrum_features(freqs, psds)
    assert feats["isfs_detected"] is False
    assert np.isnan(feats["peak_freq_hz"])
    assert np.isnan(feats["chromatogram_peak_area"])


def test_compute_bout_peak_freqs_shape(fake_subject_tree):
    subject_dir = fake_subject_tree / "SUBJ001"
    isfs = pio.load_stage_isfs_spectra(subject_dir, "C3", "N2", "sigma")
    peaks = psp.compute_bout_peak_freqs(isfs["freqs"], isfs["psds"])
    assert peaks.shape == (3,)
    assert np.all((peaks >= 0.01) & (peaks <= 0.1))
