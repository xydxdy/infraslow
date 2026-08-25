"""DataFrame builders, the demographics/sleep-statistics left join, cohort
summary stats, and CSV writing -- everything that turns per-subject/state
records (Python dicts, produced by `pipeline.pipeline`) into the files
listed in the pipeline spec's "Suggested Output Structure".
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Union

import numpy as np
import pandas as pd


def build_bout_features_df(records: List[dict]) -> pd.DataFrame:
    """One row per `(subject_id, sleep_state, channel, bout_id)` record."""
    return pd.DataFrame(records)


def build_subject_state_features_df(records: List[dict]) -> pd.DataFrame:
    """One row per `(subject_id, sleep_state[, channel])` record. An empty
    `records` list still yields a `subject_id`/`sleep_state`/`channel`-columned
    (0-row) frame rather than a bare 0-column `pd.DataFrame([])`, so downstream
    `merge`/`groupby` calls (`join_demographics_and_sleep_stats`, `cohort_summary`)
    have the columns they need to operate on and correctly produce an empty
    result instead of raising `KeyError`."""
    if not records:
        return pd.DataFrame(columns=["subject_id", "sleep_state", "channel"])
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


def cohort_summary(
    df: pd.DataFrame, *, group_col: Union[str, List[str]] = "sleep_state", value_cols: List[str],
) -> pd.DataFrame:
    """Long-form `<group_col...> | variable | n | mean | std | sem | median |
    q25 | q75 | min | max`, one row per `(<group_col...>, variable)`. `n`
    counts only non-NaN values for that variable within that group -- groups
    are summarized independently (grouped by `group_col`), never pooled.

    `group_col` may be a single column name (the default, `"sleep_state"`) or
    a list of column names (e.g. `["sleep_state", "channel"]`) -- grouping by
    `sleep_state` alone pools every channel's rows together for a given
    subject/state as if they were independent observations, which inflates
    `n` and understates `sem`; pass a list to keep those genuinely distinct
    measurements (e.g. different EEG channels) separate."""
    group_cols = [group_col] if isinstance(group_col, str) else list(group_col)
    rows = []
    for key, group in df.groupby(group_col):
        key_tuple = key if isinstance(key, tuple) else (key,)
        key_dict = dict(zip(group_cols, key_tuple))
        for col in value_cols:
            values = pd.to_numeric(group[col], errors="coerce").dropna()
            n = int(values.size)
            rows.append(dict(
                **key_dict, variable=col, n=n,
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
