#!/bin/bash
# Resubmits the preprocessing.py shards that never finished in the original
# 1000-shard run_preprocessing.sbatch array (progress_shard{N}.log exists but
# has no "finished:" line -- the task was killed mid-subject, not a channel-
# level error). Indices come from /scratch/users/chaisaen/unfinished_shards.txt.
#
# NUM_SHARDS=1000 is exported explicitly (matching the original run) so each
# resubmitted task's $SLURM_ARRAY_TASK_ID maps back to the same shard split --
# a prior retry (job 37370347) instead exported NUM_SHARDS=159, so every
# index >= 159 died immediately with "--shard-index N out of range for
# --num-shards 159" before processing a single subject.
#
# Usage: ./run_unfinished.sh

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

SHARD_LIST="/scratch/users/chaisaen/unfinished_shards.txt"
ARRAY_SPEC="$(paste -sd, "$SHARD_LIST")"

echo "resubmitting ${SHARD_LIST}: $(wc -l < "$SHARD_LIST") shards -> --array=${ARRAY_SPEC}"

sbatch --array="$ARRAY_SPEC" --export=ALL,NUM_SHARDS=1000 run_preprocessing.sbatch
