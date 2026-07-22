#!/bin/bash
#SBATCH --job-name=infraslow_compare
#SBATCH --partition=normal
#SBATCH --output=/scratch/users/chaisaen/logs/infraslow_compare_%j.out
#SBATCH --time=02:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# Runs demo_infraslow_yasa_compare.py (by default reads EVERY subject npz per
# stage from $SCRATCH/data/npz/{STAGE} -- ~77k+ files/stage as of writing,
# loaded via a thread pool since it's I/O-bound), saving one PNG per stage plus
# summary.csv and a progress.log to $SCRATCH/infraslow_outputs/infraslow/.
#
# All 4 knobs below are sent to the script explicitly as CLI flags -- edit the
# values directly in this file to change them.
# Run `python3 demo_infraslow_yasa_compare.py --help` for what each flag does.
#
# Logs go to $SCRATCH (not $HOME/logs like run_extract_featuress.sbatch) since
# this can be resubmitted often during iteration and $HOME's 15 GB quota is easy
# to fill with accumulated run logs. Loading progress (per-stage, logged every
# 1000 npz files) is written to {OUTPUT_DIR}/progress.log as well as this file --
# tail -f it to watch a running job:
#   tail -f $SCRATCH/infraslow_outputs/infraslow/progress.log

set -euo pipefail

mkdir -p /scratch/users/chaisaen/logs

# --- Parameters (edit values here) ---
SHOW_INDIVIDUAL=true
N_SUBJECTS=none                        # none = every subject npz found for a stage
RESULTS_DIR=$SCRATCH/data/npz
OUTPUT_DIR=$SCRATCH/infraslow_outputs/infraslow-show_individual

# Initialize conda for non-interactive Slurm shell
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate infraslow-py310

cd /home/users/chaisaen/infraslow/src

ARGS=(--results-dir "$RESULTS_DIR" --output-dir "$OUTPUT_DIR")
[[ "$N_SUBJECTS" != "none" ]] && ARGS+=(--n-subjects "$N_SUBJECTS")
[[ "$SHOW_INDIVIDUAL" == "true" ]] && ARGS+=(--show-individual)

python3 infraslow_yasa_compare.py.py "${ARGS[@]}"
