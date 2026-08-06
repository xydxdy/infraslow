"""Same as N2_C3_002peak_drug_comparisons_combined.png, plus a 5th column:
spindle_per_hr itself (does spindle rate differ by drug use, in the same
0.02 Hz / power-outlier-removed cohort and the same with/without drug
groups)? N2-C3, dominant_freq_hz==0.02, power outliers removed via 3x IQR;
with/without Z_Drugs, Benzodiazepine, both, and any-of-43 "All". Not part of
the pipeline; safe to delete after use.
"""
import sys
sys.path.insert(0, "src")

from pathlib import Path

import numpy as np
import pandas as pd

import group_analysis as ga
from infraslow.constants import OUTLIER_IQR_MULTIPLIER
from infraslow.stats.group_comparison import compare_parameters
from infraslow.viz.group_analysis import group_palette

VALIDATED_CSV = (
    "/scratch/users/chaisaen/infraslow_outputs/group_analysis_before_remove_outlier/"
    "N2_C3/All/validated_N2_C3_subject_results.csv"
)
OUT_DIR = Path("/scratch/users/chaisaen/tmp")
OUT_DIR.mkdir(parents=True, exist_ok=True)
RATE_COL = "spindle_per_hr"
PARAMS = ["power", "bandwidth_hz", "auc", "chromatogram_peak_area", RATE_COL]
WITH_LABEL, WITHOUT_LABEL = "with_drug", "without_drug"

# --- Same flow as before: dominant_freq_hz==0.02, power outliers removed ---
df = pd.read_csv(
    VALIDATED_CSV, usecols=["subject_id", "dominant_freq_hz", *PARAMS], dtype={"subject_id": str},
)
df = df[df["dominant_freq_hz"].round(2) == 0.02].copy()
power = df["power"].to_numpy(dtype=float)
q1, q3 = np.percentile(power, [25, 75])
iqr = q3 - q1
lo, hi = q1 - OUTLIER_IQR_MULTIPLIER * iqr, q3 + OUTLIER_IQR_MULTIPLIER * iqr
df_clean = df[(df["power"] >= lo) & (df["power"] <= hi)].reset_index(drop=True)
print(f"power-outlier-removed (0.02 Hz only): {len(df)} -> {len(df_clean)}")

# --- Drug flags (same production function --exclude-drug-users uses) -------
z_drug_ids = ga.load_drug_excluded_subject_ids(["Z_Drugs"])
benzo_ids = ga.load_drug_excluded_subject_ids(["Benzodiazepine"])
both_ids = z_drug_ids & benzo_ids
all_drug_names = ga.load_drug_exclude_names(Path(ga.DEFAULT_DRUG_EXCLUDE_LIST))
all_ids = ga.load_drug_excluded_subject_ids(all_drug_names)

ROWS = [
    ("Z_Drugs", z_drug_ids),
    ("Benzodiazepine", benzo_ids),
    ("Z_Drugs & Benzodiazepine (both)", both_ids),
    ("Any of 43 drugs (All)", all_ids),
]

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

colors = group_palette(2)  # [without_drug, with_drug]
rng = np.random.default_rng(0)

fig, axes = plt.subplots(len(ROWS), len(PARAMS), figsize=(5.0 * len(PARAMS), 4.0 * len(ROWS)))

summary_rows = []
for row_i, (row_label, flagged_ids) in enumerate(ROWS):
    d = df_clean.copy()
    d["drug_group"] = np.where(d["subject_id"].isin(flagged_ids), WITH_LABEL, WITHOUT_LABEL)
    n_with = int((d["drug_group"] == WITH_LABEL).sum())
    n_without = int((d["drug_group"] == WITHOUT_LABEL).sum())
    comparison = compare_parameters(d, "drug_group", PARAMS, low_label=WITHOUT_LABEL, high_label=WITH_LABEL)
    comparison.insert(0, "drug_comparison", row_label)
    summary_rows.append(comparison)
    q_by_param = comparison.set_index("parameter")["q_value"].to_dict()
    print(f"\n=== {row_label}: with={n_with} without={n_without} ===")
    print(comparison[["parameter", "low_mean", "high_mean", "p_value", "q_value", "significant_fdr"]]
          .to_string(index=False))

    for col_i, param in enumerate(PARAMS):
        ax = axes[row_i, col_i]
        without_vals = d.loc[d["drug_group"] == WITHOUT_LABEL, param].dropna().to_numpy(dtype=float)
        with_vals = d.loc[d["drug_group"] == WITH_LABEL, param].dropna().to_numpy(dtype=float)

        parts = ax.violinplot([without_vals, with_vals], showmedians=True)
        for body, color in zip(parts["bodies"], colors):
            body.set_facecolor(color)
            body.set_alpha(0.4)
        for key in ("cmedians", "cmins", "cmaxes", "cbars"):
            if key in parts:
                parts[key].set_color("0.2")

        for position, (values, color) in enumerate(((without_vals, colors[0]), (with_vals, colors[1])), start=1):
            show = values if values.size <= 300 else rng.choice(values, size=300, replace=False)
            jitter = rng.normal(0, 0.04, size=show.size)
            ax.scatter(np.full(show.size, position) + jitter, show, color=color, s=5, alpha=0.35, zorder=3,
                       edgecolor="none")

        q_value = q_by_param.get(param, np.nan)
        title = param if not np.isfinite(q_value) else f"{param} (q={q_value:.2g})"
        ax.set_xticks([1, 2])
        ax.set_xticklabels([f"without\n(n={n_without})", f"with\n(n={n_with})"], fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if col_i == 0:
            ax.set_ylabel(row_label, fontsize=10, fontweight="semibold")

fig.suptitle(
    "N2, C3, dominant_freq_hz=0.02 Hz (power outliers removed): with vs without drug use",
    fontsize=15, fontweight="bold", y=1.0,
)
fig.tight_layout()

out_png = OUT_DIR / "N2_C3_002peak_drug_comparisons_combined_with_rate.png"
out_pdf = OUT_DIR / "N2_C3_002peak_drug_comparisons_combined_with_rate.pdf"
fig.savefig(out_png, dpi=150, bbox_inches="tight")
fig.savefig(out_pdf, bbox_inches="tight")
pd.concat(summary_rows, ignore_index=True).to_csv(
    OUT_DIR / "N2_C3_002peak_drug_comparisons_with_rate.csv", index=False
)
print(f"\nsaved {out_png}")
print(f"saved {out_pdf}")
print("saved N2_C3_002peak_drug_comparisons_with_rate.csv")
