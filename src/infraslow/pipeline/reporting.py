"""DataFrame builders, the demographics/sleep-statistics left join, cohort
summary stats, and CSV writing -- everything that turns per-subject/state
records (Python dicts, produced by `pipeline.pipeline`) into the files
listed in the pipeline spec's "Suggested Output Structure".
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


def build_bout_features_df(records: List[dict]) -> pd.DataFrame:
    """One row per `(subject_id, sleep_state, channel, bout_id)` record."""
    return pd.DataFrame(records)


def build_subject_state_features_df(records: List[dict]) -> pd.DataFrame:
    """One row per `(subject_id, sleep_state[, channel])` record."""
    return pd.DataFrame(records)


def join_demographics_and_sleep_stats(
    subject_state_df: pd.DataFrame, demographics: pd.DataFrame, sleep_stats: pd.DataFrame,
) -> pd.DataFrame:
    """Left join demographics (`ID`) then sleep statistics (`id`) onto
    `subject_state_df` (`subject_id`) -- every row of `subject_state_df` is
    preserved regardless of whether either side table has a matching row
    (or is entirely empty), with unmatched columns left `NaN`. Demographics/
    sleep-stats values are naturally repeated across a subject's N2 and N3
    rows (both are whole-night, not state-specific)."""
    demo = demographics.rename(columns={"ID": "subject_id"})
    out = subject_state_df.merge(demo, on="subject_id", how="left")

    stats = sleep_stats.rename(columns={"id": "subject_id"})
    out = out.merge(stats, on="subject_id", how="left")
    return out


def cohort_summary(df: pd.DataFrame, *, group_col: str = "sleep_state", value_cols: List[str]) -> pd.DataFrame:
    """Long-form `sleep_state | variable | n | mean | std | sem | median |
    q25 | q75 | min | max`, one row per `(sleep_state, variable)`. `n` counts
    only non-NaN values for that variable within that state -- N2 and N3 are
    summarized independently (grouped by `group_col`), never pooled."""
    rows = []
    for state, group in df.groupby(group_col):
        for col in value_cols:
            values = pd.to_numeric(group[col], errors="coerce").dropna()
            n = int(values.size)
            rows.append(dict(
                sleep_state=state, variable=col, n=n,
                mean=float(values.mean()) if n else np.nan,
                std=float(values.std(ddof=1)) if n > 1 else (0.0 if n == 1 else np.nan),
                sem=float(values.std(ddof=1) / np.sqrt(n)) if n > 1 else (0.0 if n == 1 else np.nan),
                median=float(values.median()) if n else np.nan,
                q25=float(values.quantile(0.25)) if n else np.nan,
                q75=float(values.quantile(0.75)) if n else np.nan,
                min=float(values.min()) if n else np.nan,
                max=float(values.max()) if n else np.nan,
            ))
    return pd.DataFrame(rows)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    """`df.to_csv(path, index=False)`, creating `path`'s parent directories first."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


__all__ = [
    "build_bout_features_df",
    "build_subject_state_features_df",
    "join_demographics_and_sleep_stats",
    "cohort_summary",
    "write_csv",
]
