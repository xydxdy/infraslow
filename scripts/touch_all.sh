#!/usr/bin/env bash
# Touch every file and directory nested under TARGET.
# Usage: TARGET=/scratch/users/chaisaen/TEST ./touch_all.sh

set -euo pipefail

: "${TARGET:?Set TARGET to the directory to touch, e.g. TARGET=/scratch/users/chaisaen/TEST}"

if [[ ! -d "$TARGET" ]]; then
    echo "TARGET does not exist or is not a directory: $TARGET" >&2
    exit 1
fi

find "$TARGET" -type d -exec touch {} +

echo "Touched all paths under $TARGET"
