#!/bin/bash
#SBATCH --job-name=preprocessing
#SBATCH --partition=normal
#SBATCH --output=/scratch/users/chaisaen/logs/preprocessing_%A_%a.out
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G

# Runs src/preprocessing.py over the whole valid Bioserenity cohort (every
# subject with both an EDF and a Hypnodensity CSV), channel C3 only, saving per-subject
# envelope/temporal_ISFS/bouts/ISFS-spectra trees to
# ${OUTPUT_ROOT}/{subject_id}/C3/ (see preprocessing.py's module docstring for
# the exact layout).
#
# Submitted as a 1000-way job array so preprocessing.py's own --num-shards/
# --shard-index (defaulting from $SLURM_ARRAY_TASK_COUNT/$SLURM_ARRAY_TASK_ID
# when no --subject is given) split the cohort into 1000 disjoint pieces, one
# per array task -- no NUM_SHARDS/SHARD_INDEX exports needed for this:
#
#   sbatch --array=0-999 run_preprocessing.sbatch
#
# *** --mem/--time above are PLACEHOLDERS, not yet verified against a real
# run. Check actual per-subject memory/time first with a single-task probe,
# e.g.:
#   sbatch --array=0-0 --time=02:00:00 run_preprocessing.sbatch
#   seff <jobid>          # once it finishes (or squeue --me while running)
# and adjust --mem/--time above before submitting the full 1000-way array. ***
#
# Each shard writes its own progress_shard{N}.log under $OUTPUT_ROOT (see
# preprocessing.py's main()) -- shards never share a log file.
#
# Override at submit time, e.g.:
#   sbatch --array=0-999 --export=ALL,CHANNELS=C3,OUTPUT_ROOT=/scratch/users/chaisaen/data_v3 \
#       run_preprocessing.sbatch
#   sbatch --array=0-9 --export=ALL,LIMIT=2 run_preprocessing.sbatch   # quick test: 2 subjects/shard

set -euo pipefail

OUTPUT_ROOT="${OUTPUT_ROOT:-/scratch/users/chaisaen/data_v3}"
CHANNELS="${CHANNELS:-C3}"
LIMIT="${LIMIT:-}"

mkdir -p /scratch/users/chaisaen/logs
mkdir -p "${OUTPUT_ROOT}"

# Initialize conda for non-interactive Slurm shell
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate infraslow-py310

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd /home/users/chaisaen/infraslow/src

python3 preprocessing.py \
    --channels ${CHANNELS} \
    --output-dir "${OUTPUT_ROOT}" \
    ${LIMIT:+--limit "${LIMIT}"}
