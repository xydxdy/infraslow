#!/bin/bash
#SBATCH --job-name=touch_all
#SBATCH --partition=normal
#SBATCH --output=/scratch/users/chaisaen/logs/touch_all_%j.out
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#
# Runs touch_all.sh through Slurm instead of the login node, since a `find`
# walk over a large TARGET tree is real I/O against Lustre.
#
# *** --time is a PLACEHOLDER, not yet verified. Size it from the tree's
# depth/file count (or a prior `find | wc -l`), then check with
# `seff <jobid>` afterward. ***
#
# Usage:
#   sbatch --export=ALL,TARGET=/scratch/users/chaisaen/processed_data touch_all.sbatch

set -euo pipefail

: "${TARGET:?Set TARGET=/path/under/scratch via --export=ALL,TARGET=...}"

mkdir -p /scratch/users/chaisaen/logs

cd /home/users/chaisaen/infraslow/scripts
TARGET="$TARGET" ./touch_all.sh
