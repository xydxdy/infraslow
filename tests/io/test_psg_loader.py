from __future__ import annotations

from pathlib import Path

from infraslow.io.psg_loader import BioserenityPSGLoader
from infraslow.io.usleep_hypnodensity import make_usleep_annotation_loader


def test_usleep_hypnodensity_defaults_to_first_requested_channel(fake_usleep_tree):
    loader = BioserenityPSGLoader(subject_id="SUBJ001", requested_channels=["C3"])
    t_hyp, probs = loader.usleep_hypnodensity(base_dir=str(fake_usleep_tree))
    assert t_hyp.shape == (30,)
    assert probs.shape == (30, 5)


def test_usleep_hypnodensity_honors_explicit_usleep_channel(fake_usleep_tree):
    # requested_channels lists C4 first, but usleep_channel should win.
    loader = BioserenityPSGLoader(
        subject_id="SUBJ002", requested_channels=["C4", "C3"], usleep_channel="C3",
    )
    t_hyp, probs = loader.usleep_hypnodensity(base_dir=str(fake_usleep_tree))
    assert t_hyp.shape == (30,)


def test_usleep_hypnodensity_call_arg_overrides_usleep_channel(fake_usleep_tree):
    loader = BioserenityPSGLoader(subject_id="SUBJ001", usleep_channel="O1")
    t_hyp, probs = loader.usleep_hypnodensity("C3", base_dir=str(fake_usleep_tree))
    assert t_hyp.shape == (30,)


def test_construction_builds_a_default_usleep_annotation_loader():
    # No .load()/lunapi needed: __post_init__ builds annotation_loader eagerly.
    loader = BioserenityPSGLoader(subject_id="SUBJ001", requested_channels=["C3"])
    assert callable(loader.annotation_loader)


def test_explicit_annotation_loader_is_not_overridden(fake_usleep_tree):
    explicit = make_usleep_annotation_loader("C3", base_dir=str(fake_usleep_tree))
    loader = BioserenityPSGLoader(
        subject_id="SUBJ001", requested_channels=["C3"], annotation_loader=explicit,
    )
    assert loader.annotation_loader is explicit
    annotations = loader.annotation_loader(None, Path("/whatever/SUBJ001.edf"))
    assert list(annotations.columns) == ["t", "Wake", "N1", "N2", "N3", "REM", "stage"]
