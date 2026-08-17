#!/bin/bash
#SBATCH --job-name=n2_vs_n3_compare_delta
#SBATCH --partition=normal
#SBATCH --output=/scratch/users/chaisaen/logs/n2_vs_n3_compare_delta_%j.out
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# Runs plot_n2_vs_n3_compare_delta.py for every eligible subject (N_SUBJECTS=none,
# CANDIDATE_LIMIT=none below -- no cap, scans the whole cohort under DATA_DIR
# and uses every subject with a finite, usable N2/N3 spectrum + slow-wave-phase
# pairing on CHANNEL, delta band). This is the delta/slow-wave counterpart of
# run_n2_vs_n3_compare.sbatch (sigma/spindle) -- see that file for the sigma version.
# This is a single group-level comparison job, not one task per subject -- the
# script itself loops over the whole cohort.
#
# Saves feature tables (n2/n3_features.csv), comparison result tables
# (phase_bin_results.csv, isfs_metric_results.csv, peak_freq_summary.csv),
# every figure, summary.txt, and progress.log to $SCRATCH/outputs/n2_vs_n3_compare_delta/.
#
# All knobs below are sent to the script explicitly as CLI flags -- edit the
# values directly in this file to change them.
# Run `python3 plot_n2_vs_n3_compare_delta.py --help` for what each flag does.
#
# Logs go to $SCRATCH (not $HOME/logs) -- tail -f the progress log to watch a
# running job:
#   tail -f $SCRATCH/outputs/n2_vs_n3_compare_delta/progress.log

set -euo pipefail

mkdir -p /scratch/users/chaisaen/logs

# --- Parameters (edit values here) ---
DATA_DIR=$SCRATCH/processed_data/data
CHANNEL=C3
N_SUBJECTS=none                        # none = every eligible subject found
CANDIDATE_LIMIT=none                   # none = scan the whole cohort, no cap
WORKERS=8                              # match --cpus-per-task above
OUTPUT_DIR=$SCRATCH/outputs/n2_vs_n3_compare_delta

# Initialize conda for non-interactive Slurm shell
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate infraslow-py310

cd /home/users/chaisaen/infraslow/src/scripts
export PYTHONPATH=/home/users/chaisaen/infraslow/src

ARGS=(--data-dir "$DATA_DIR" --channel "$CHANNEL" --output-dir "$OUTPUT_DIR" --workers "$WORKERS")
[[ "$N_SUBJECTS" != "none" ]] && ARGS+=(--n-subjects "$N_SUBJECTS")
[[ "$CANDIDATE_LIMIT" != "none" ]] && ARGS+=(--candidate-limit "$CANDIDATE_LIMIT")

python3 plot_n2_vs_n3_compare_delta.py "${ARGS[@]}"
