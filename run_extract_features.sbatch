#!/bin/bash
#SBATCH --job-name=all_metrics
#SBATCH --partition=mignot
#SBATCH --output=logs/all_metrics_%A_%a.log
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=30G

# Splits the subject cohort into NUM_SHARDS disjoint pieces; this job processes
# SHARD_INDEX. Two ways to parallelize, mix and match:
#
#   1. Uniform array on one partition -- add --array at submit time and leave
#      NUM_SHARDS/SHARD_INDEX unset; they default from Slurm's own
#      $SLURM_ARRAY_TASK_COUNT/_ID:
#        sbatch --array=0-9 extract_features.sbatch          # 10 shards, mignot
#
#   2. Mixed partitions/resources (e.g. 1 job on mignot + 2 on normal) -- submit
#      each shard as its own job, no --array, with NUM_SHARDS/SHARD_INDEX passed
#      explicitly (they take priority over any Slurm array vars):
#        sbatch --partition=mignot --cpus-per-task=10 --mem=30G \
#          --export=ALL,NUM_SHARDS=3,SHARD_INDEX=0 extract_features.sbatch
#        sbatch --partition=normal --cpus-per-task=10 --mem=30G --time=2-00:00:00 \
#          --export=ALL,NUM_SHARDS=3,SHARD_INDEX=1 extract_features.sbatch
#        sbatch --partition=normal --cpus-per-task=10 --mem=30G --time=2-00:00:00 \
#          --export=ALL,NUM_SHARDS=3,SHARD_INDEX=2 extract_features.sbatch
#      (normal's ceiling is 2 days -- override --time or add --qos=long for
#      longer, since the 72:00:00 default above exceeds it.)
#
# Every shard writes its own metadata/errors CSV (a shared path would have
# shards clobber each other's checkpoints); merge them once every shard job
# finishes, e.g.:
#   python -c "import pandas as pd, glob; \
#     pd.concat([pd.read_csv(f, dtype={'ID': str}) for f in \
#     glob.glob('\$SCRATCH/data/metadata_shard*.csv')]).to_csv( \
#     '\$SCRATCH/data/metadata.csv', index=False)"

set -euo pipefail

OUTPUT_ROOT="$SCRATCH/data"
NUM_SHARDS="${NUM_SHARDS:-${SLURM_ARRAY_TASK_COUNT:-1}}"
SHARD_INDEX="${SHARD_INDEX:-${SLURM_ARRAY_TASK_ID:-0}}"
# Stage group(s) to compute/write (see extract_features.py --stages); only
# these stages' npz files are touched. Override with --export=ALL,STAGES="N2 NREM".
STAGES="${STAGES:-N2}"

mkdir -p logs

# Initialize conda for non-interactive Slurm shell
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate infraslow-py310

for stage in ${STAGES}; do
    mkdir -p "${OUTPUT_ROOT}/npz/${stage}"
done

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export SLURM_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-10}"
export METADATA="$OAK/psg/Bioserenity/Excel/Morpheus_Data_All5.csv"
export METADATA2="$OAK/psg/Bioserenity/Excel/bioserenity_metadata3.csv"
export OUTPUT="${OUTPUT_ROOT}/metadata_shard${SHARD_INDEX}.csv"
export ERROR_OUTPUT="${OUTPUT_ROOT}/errors_shard${SHARD_INDEX}.csv"
export NPZ_DIR="${OUTPUT_ROOT}/npz"

python -u src/extract_features.py \
    --workers "${SLURM_CPUS_PER_TASK}" \
    --num-shards "${NUM_SHARDS}" \
    --shard-index "${SHARD_INDEX}" \
    --stages ${STAGES} \
    --metadata "${METADATA}" \
    --metadata2 "${METADATA2}" \
    --output "${OUTPUT}" \
    --error-output "${ERROR_OUTPUT}" \
    --npz-dir "${NPZ_DIR}"
