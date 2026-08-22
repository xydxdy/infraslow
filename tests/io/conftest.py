"""Shared fixtures for infraslow.io tests."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def fake_usleep_tree(tmp_path: Path) -> Path:
    """<tmp_path>/usleep/{SUBJ001,SUBJ002}/... -- two fake 1-s U-Sleep hypnodensity
    subjects. SUBJ001 uses the raw channel-name convention (``C3.npy``); SUBJ002 uses the
    referenced Bioserenity/U-Sleep name (``EEG C3-A2.npy``) -- the same two filename
    conventions observed in the real data (see this plan's Global Constraints)."""
    root = tmp_path / "usleep"
    rng = np.random.default_rng(0)

    def _make_probs(n: int) -> np.ndarray:
        raw = rng.random((n, 5))
        return (raw / raw.sum(axis=1, keepdims=True)).astype(np.float32)

    d1 = root / "SUBJ001"
    d1.mkdir(parents=True)
    np.save(d1 / "C3.npy", _make_probs(30))

    d2 = root / "SUBJ002"
    d2.mkdir(parents=True)
    np.save(d2 / "EEG C3-A2.npy", _make_probs(30))

    return root


@pytest.fixture
def fake_hypno_dir(tmp_path: Path) -> Path:
    """<tmp_path>/hypno/SUBJ001_Hypnodensity.csv -- a fake raw Bioserenity Hypnodensity
    CSV, 6 epochs (30s each): N2, N2, N2, N3, N3, Wake -- exercises a same-stage epoch
    (index 1), a transition epoch (index 3, N2->N3), and a trailing stage with no
    further epoch after it (index 5, Wake, last row)."""
    root = tmp_path / "hypno"
    root.mkdir(parents=True)
    rows = [
        ("2012-01-01 21:52:52+00:00", 0.02, 0.02, 0.90, 0.03, 0.03),  # N2
        ("2012-01-01 21:53:22+00:00", 0.02, 0.02, 0.90, 0.03, 0.03),  # N2
        ("2012-01-01 21:53:52+00:00", 0.02, 0.02, 0.90, 0.03, 0.03),  # N2
        ("2012-01-01 21:54:22+00:00", 0.02, 0.02, 0.03, 0.90, 0.03),  # N3
        ("2012-01-01 21:54:52+00:00", 0.02, 0.02, 0.03, 0.90, 0.03),  # N3
        ("2012-01-01 21:55:22+00:00", 0.90, 0.02, 0.03, 0.02, 0.03),  # Wake
    ]
    csv_path = root / "SUBJ001_Hypnodensity.csv"
    with csv_path.open("w") as f:
        f.write("Timestamp,Wake,N1,N2,N3,REM\n")
        for ts, wake, n1, n2, n3, rem in rows:
            f.write(f"{ts},{wake},{n1},{n2},{n3},{rem}\n")
    return root
