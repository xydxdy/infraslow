#!/bin/bash
#SBATCH --job-name=isfs_multi_subject
#SBATCH --partition=mignot
#SBATCH --output=/scratch/users/chaisaen/logs/isfs_multi_subject_%j.out
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# Executes src/ISFS_multi_subject_analysis.ipynb headlessly (jupyter nbconvert
# --execute --inplace), saving all outputs (group plots, feature summary
# table) back into the notebook itself. Processes the fixed subject list in
# the notebook's SUBJECT_IDS -- no caching, so a re-run reprocesses every
# subject from scratch.

set -euo pipefail

mkdir -p /scratch/users/chaisaen/logs

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate infraslow-py310

cd /home/users/chaisaen/infraslow/src
export PYTHONPATH=/home/users/chaisaen/infraslow/src

jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.kernel_name=python3 \
    --ExecutePreprocessor.timeout=-1 \
    ISFS_multi_subject_analysis.ipynb
