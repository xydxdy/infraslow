"""Shared fixtures for pipeline tests: build a fake preprocessing.py output
tree so pipeline/io.py, spectrum.py, phase.py, features.py can be tested
without real Sherlock data.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SF_ENV = 1.0  # matches DEFAULT_SF_ENV


def _write_envelope(ch_dir: Path, band: str, n_samples: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    t_env = np.arange(n_samples, dtype=float) / SF_ENV
    # A slow ~50s-period oscillation (the ISFS rhythm) plus noise, in "dB" scale,
    # so infraslow_spectrum has a real peak near 0.02 Hz to fit against.
    power = 10.0 * np.sin(2 * np.pi * t_env / 50.0) + rng.normal(0, 1.0, n_samples)
    (ch_dir / "envelope").mkdir(parents=True, exist_ok=True)
    np.savez(ch_dir / "envelope" / f"{band}.npz", t_env=t_env, power=power)


def _write_temporal_isfs(ch_dir: Path, band: str, n_samples: int, seed: int) -> None:
    rng = np.random.default_rng(seed + 1)
    t_env = np.arange(n_samples, dtype=float) / SF_ENV
    filtered = 5.0 * np.sin(2 * np.pi * t_env / 50.0) + rng.normal(0, 0.2, n_samples)
    (ch_dir / "temporal_ISFS").mkdir(parents=True, exist_ok=True)
    np.savez(ch_dir / "temporal_ISFS" / f"{band}.npz", t_env=t_env, power=filtered)


def _write_stage(ch_dir: Path, stage: str, bouts: list, seed: int) -> None:
    """bouts: list of (start, stop) tuples, all >= 200s, e.g. one every ~300s."""
    stage_dir = ch_dir / stage
    (stage_dir / "ISFS").mkdir(parents=True, exist_ok=True)

    all_arr = np.asarray(bouts, dtype=np.float64)
    # Every bout "contains a spindle" in this fixture (simplest case); tests
    # that need a bout WITHOUT a spindle build their own tree by hand.
    spindle_arr = all_arr.copy()
    np.savez(stage_dir / "bouts.npz", all=all_arr, spindle=spindle_arr)

    sw_arr = all_arr.copy()
    np.savez(stage_dir / "sw_bouts.npz", all=all_arr, sw=sw_arr)

    rng = np.random.default_rng(seed + 2)
    rows = []
    for i, (a, b) in enumerate(bouts):
        peak = a + (b - a) / 2.0
        rows.append({
            "Start": peak - 1.0, "Peak": peak, "End": peak + 1.0,
            "Duration": 2.0, "Amplitude": 20.0 + rng.normal(0, 1),
            "Frequency": 13.0 + rng.normal(0, 0.2), "Stage": 2 if stage == "N2" else 3,
        })
    pd.DataFrame(rows).to_csv(stage_dir / "spindel_yasa.csv", index=False)

    sw_rows = []
    for i, (a, b) in enumerate(bouts):
        neg = a + (b - a) / 3.0
        sw_rows.append({
            "Start": neg - 0.5, "NegPeak": neg, "PosPeak": neg + 0.6, "End": neg + 1.0,
            "Duration": 1.5, "PTP": 120.0 + rng.normal(0, 5),
            "Frequency": 0.8 + rng.normal(0, 0.05), "Stage": 2 if stage == "N2" else 3,
        })
    pd.DataFrame(sw_rows).to_csv(stage_dir / "sw_yasa.csv", index=False)

    freqs = np.linspace(0.0025, 0.1, 40)
    n_bouts = len(bouts)
    rng2 = np.random.default_rng(seed + 3)
    # A spectral bump near 0.02 Hz for every bout, so fit_isfs has something to find.
    base = np.exp(-0.5 * ((freqs - 0.02) / 0.01) ** 2)
    psds = np.stack([base * (1.0 + rng2.normal(0, 0.05, freqs.size)) + 0.01 for _ in range(n_bouts)])
    bout_start = np.asarray([a for a, _ in bouts], dtype=np.float64)
    for band in ("sigma", "delta"):
        np.savez(stage_dir / "ISFS" / f"{band}.npz", freqs=freqs, psds=psds, bout_start=bout_start)


@pytest.fixture
def fake_subject_tree(tmp_path: Path) -> Path:
    """<tmp_path>/data/SUBJ001/C3/{envelope,temporal_ISFS,N2,N3}/... -- one full fake subject."""
    data_dir = tmp_path / "data"
    ch_dir = data_dir / "SUBJ001" / "C3"

    n_samples = 1200  # 20 min at 1 Hz -- long enough for several ISFS cycles
    _write_envelope(ch_dir, "sigma", n_samples, seed=1)
    _write_envelope(ch_dir, "delta", n_samples, seed=2)
    _write_temporal_isfs(ch_dir, "sigma", n_samples, seed=3)
    _write_temporal_isfs(ch_dir, "delta", n_samples, seed=4)

    n2_bouts = [(50.0, 350.0), (450.0, 750.0), (850.0, 1150.0)]
    n3_bouts = [(0.0, 45.0)]  # deliberately < 200s min_dur -- N3 has NO valid bouts for this subject
    _write_stage(ch_dir, "N2", n2_bouts, seed=10)
    _write_stage(ch_dir, "N3", n3_bouts, seed=20)

    return data_dir
