from __future__ import annotations

import numpy as np

from infraslow.pipeline import figures as pfg


def test_plot_relative_spectrum_bigaussian_writes_file(tmp_path):
    freqs = np.linspace(0.0025, 0.1, 40)
    rel = np.exp(-0.5 * ((freqs - 0.02) / 0.01) ** 2) + 0.01
    corrected = rel - rel[-5:].mean()
    fit_features = dict(peak_freq_hz=0.02, peak_period_s=50.0, bandwidth_hz=0.02,
                         auc=0.4, chromatogram_peak_area=0.3)
    popt = (1.0, 0.02, 0.01, 0.015)
    out_path = tmp_path / "figures" / "N2_relative_spectral_power_bigaussian.png"
    pfg.plot_relative_spectrum_bigaussian(freqs, rel, corrected, fit_features, popt, state="N2", out_path=out_path)
    assert out_path.is_file()
    assert out_path.stat().st_size > 0


def test_plot_peak_frequency_distribution_writes_file(tmp_path):
    peaks = np.random.default_rng(0).uniform(0.01, 0.1, 30)
    out_path = tmp_path / "figures" / "N3_peak_frequency_distribution.png"
    pfg.plot_peak_frequency_distribution(peaks, state="N3", out_path=out_path)
    assert out_path.is_file()


def test_plot_peak_frequency_distribution_empty_input_does_not_raise(tmp_path):
    out_path = tmp_path / "figures" / "N3_peak_frequency_distribution_empty.png"
    pfg.plot_peak_frequency_distribution(np.empty(0), state="N3", out_path=out_path)
    assert out_path.is_file()


def test_plot_spindle_phase_distribution_writes_file(tmp_path):
    pooled = dict(event_count=10, n_in_isfs=8,
                  phase_bin_counts=[1, 2, 0, 1, 2, 1, 0, 1],
                  phase_bin_rates=[10, 20, 0, 10, 20, 10, 0, 10])
    out_path = tmp_path / "figures" / "N2_spindle_phase_distribution.png"
    pfg.plot_spindle_phase_distribution(pooled, state="N2", out_path=out_path)
    assert out_path.is_file()
