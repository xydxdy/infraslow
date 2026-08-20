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
