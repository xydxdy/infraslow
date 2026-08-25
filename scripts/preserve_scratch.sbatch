#!/bin/bash
#SBATCH --job-name=preserve_scratch
#SBATCH --partition=normal
#SBATCH --output=/scratch/users/chaisaen/logs/preserve_scratch_%j.out
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#
# Runs preserve_scratch.sh through Slurm instead of the login node, since the
# directory walk + per-file rewrite is real I/O against Lustre and can be
# heavy on a large TARGET tree.
#
# *** --time is a PLACEHOLDER, not yet verified. Run with DRY_RUN=1 first to
# see how many files qualify (files older than AGE_DAYS), then size --time
# from that before running for real. Check with `seff <jobid>` afterward. ***
#
# Usage:
#   sbatch --export=ALL,TARGET=/scratch/users/chaisaen/TEST preserve_scratch.sbatch
#   sbatch --export=ALL,TARGET=/scratch/users/chaisaen/TEST,DRY_RUN=1,AGE_DAYS=2 preserve_scratch.sbatch
#   sbatch --export=ALL,TARGET=/scratch/users/chaisaen/TEST,AGE_DAYS=2 preserve_scratch.sbatch
#
# To keep this running indefinitely without a login-node cron (Sherlock login
# nodes don't offer persistent user crontabs), resubmit this same job with a
# delayed start right before it exits, e.g. add near the end of this script:
#   sbatch --begin=now+60days --export=ALL,TARGET="$TARGET" "$0"
# Only do that once you've confirmed a normal run succeeds -- a delayed
# resubmit ties up a queued slot for two months.

set -euo pipefail

: "${TARGET:?Set TARGET=/path/under/scratch via --export=ALL,TARGET=...}"
AGE_DAYS="${AGE_DAYS:-2}"
JOBS="${JOBS:-${SLURM_CPUS_PER_TASK:-4}}"

DRY_RUN_FLAG=""
[[ "${DRY_RUN:-0}" == "1" ]] && DRY_RUN_FLAG="--dry-run"

mkdir -p /scratch/users/chaisaen/logs

cd /home/users/chaisaen/infraslow
./preserve_scratch.sh $DRY_RUN_FLAG --age-days "$AGE_DAYS" --jobs "$JOBS" "$TARGET"
