#!/bin/bash
#SBATCH --job-name=preprocessing
#SBATCH --partition=normal
#SBATCH --output=/scratch/users/chaisaen/logs/preprocessing_%A_%a.out
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G

# Runs src/scripts/preprocessing.py over the whole valid cohort (every subject
# with both an EDF and a U-Sleep hypnodensity folder), channel C3 only, saving
# per-subject envelope/temporal_ISFS/bouts/ISFS-spectra/usleep trees to
# ${OUTPUT_ROOT}/${HYPNO_EPOCH_SEC}s/data/{subject_id}/C3/ (see
# preprocessing.py's module docstring for the exact layout). The native 1-s
# U-Sleep hypnodensity is reduced to
# ${HYPNO_EPOCH_SEC}-s epochs (preprocessing.py's own default: 3) before
# bout-finding/event detection -- see --hypno-epoch-sec below.
#
# Submitted as a 1000-way job array so preprocessing.py's --num-shards/
# --shard-index split the cohort into 1000 disjoint pieces, one per array
# task. --num-shards defaults to $SLURM_ARRAY_TASK_COUNT (the size of *this*
# array), which only equals the original shard count for a contiguous
# 0-999 array -- resubmitting a subset of indices (e.g. run_unfinished.sh)
# must export NUM_SHARDS=1000 explicitly so shard indices still map back to
# the original 1000-way split instead of the resubmitted task count:
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
# Each shard writes its own progress_shard{N}.log under $OUTPUT_ROOT/logs (see
# preprocessing.py's main()) -- shards never share a log file. Per-subject
# output trees, and the events_*/sleep_stats_* CSVs, go under
# $OUTPUT_ROOT/${HYPNO_EPOCH_SEC}s/ (default epoch width: 3s).
#
# Override at submit time, e.g.:
#   sbatch --array=0-999 --export=ALL,CHANNELS=C3,OUTPUT_ROOT=/scratch/users/chaisaen/processed_data \
#       run_preprocessing.sbatch
#   sbatch --array=0-9 --export=ALL,LIMIT=2 run_preprocessing.sbatch   # quick test: 2 subjects/shard
#   sbatch --array=0-9 --export=ALL,NUM_SHARDS=1000,LIMIT=50,HYPNO_EPOCH_SEC=3,OUTPUT_ROOT=/scratch/users/chaisaen/usleep_dryrun \
#       run_preprocessing.sbatch   # dry run: ~500 subjects (10 tasks x 50/task), 3-s U-Sleep epoch
#
# Submitted plain (no --array) instead, it's a single unsharded task -- NUM_SHARDS/
# --shard-index are omitted entirely (see NUM_SHARDS/SHARD_INDEX below), so
# preprocessing.py processes the whole --limit'd cohort itself, sequentially:
#   sbatch --export=ALL,LIMIT=100,OUTPUT_ROOT=/scratch/users/chaisaen/usleep_dryrun_100 \
#       run_preprocessing.sbatch   # dry run: 100 subjects, one task, no sharding

set -euo pipefail

OUTPUT_ROOT="${OUTPUT_ROOT:-/scratch/users/chaisaen/processed_data}"
CHANNELS="${CHANNELS:-C3}"
LIMIT="${LIMIT:-}"
HYPNO_EPOCH_SEC="${HYPNO_EPOCH_SEC:-}"
# Unset when submitted without --array (a single-task, unsharded run) --
# NUM_SHARDS/--shard-index are then omitted below entirely, so
# preprocessing.py's own resolve_shard() defaults to num_shards=1,
# shard_index=0 (the whole -- --limit'd -- cohort, no sharding).
NUM_SHARDS="${NUM_SHARDS:-${SLURM_ARRAY_TASK_COUNT:-}}"
SHARD_INDEX="${SLURM_ARRAY_TASK_ID:-}"

mkdir -p /scratch/users/chaisaen/logs
mkdir -p "${OUTPUT_ROOT}"

# Initialize conda for non-interactive Slurm shell
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate infraslow-py310

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd /home/users/chaisaen/infraslow/src/scripts
export PYTHONPATH=/home/users/chaisaen/infraslow/src

python3 preprocessing.py \
    --channels ${CHANNELS} \
    ${NUM_SHARDS:+--num-shards "${NUM_SHARDS}"} \
    ${SHARD_INDEX:+--shard-index "${SHARD_INDEX}"} \
    --output-dir "${OUTPUT_ROOT}" \
    ${LIMIT:+--limit "${LIMIT}"} \
    ${HYPNO_EPOCH_SEC:+--hypno-epoch-sec "${HYPNO_EPOCH_SEC}"}
