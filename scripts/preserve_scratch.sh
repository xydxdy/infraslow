#!/bin/bash
# Resets Lustre's 90-day scratch purge clock on files under a directory by
# rewriting their content in place. touch/mv/chmod do NOT reset the purge
# timer on Sherlock scratch -- only a genuine write to the file's data does,
# so this uses `dd conv=notrunc` to read+rewrite each file's bytes in place
# (no extra disk space, no double I/O of a copy+rename).
#
# Usage:
#   ./preserve_scratch.sh [--dry-run] [--age-days N] [--jobs N] <path-under-scratch>
#
# <path-under-scratch> must resolve under $SCRATCH or $GROUP_SCRATCH.
# --age-days (default 75) only rewrites files already close to the 90-day
# purge window, so re-running this periodically doesn't churn every file
# on every pass.

set -euo pipefail

AGE_DAYS=75
JOBS=8
DRY_RUN=0
TARGET=""

usage() {
    echo "Usage: $0 [--dry-run] [--age-days N] [--jobs N] <path-under-scratch>" >&2
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=1; shift ;;
        --age-days) AGE_DAYS="$2"; shift 2 ;;
        --jobs) JOBS="$2"; shift 2 ;;
        -h|--help) usage ;;
        *) TARGET="$1"; shift ;;
    esac
done

[[ -n "$TARGET" ]] || usage
[[ -d "$TARGET" ]] || { echo "Not a directory: $TARGET" >&2; exit 1; }

TARGET_REAL=$(readlink -f "$TARGET")

IN_SCRATCH=0
for root in "${SCRATCH:-}" "${GROUP_SCRATCH:-}"; do
    [[ -n "$root" ]] || continue
    root_real=$(readlink -f "$root")
    if [[ "$TARGET_REAL" == "$root_real" || "$TARGET_REAL" == "$root_real"/* ]]; then
        IN_SCRATCH=1
    fi
done
if [[ "$IN_SCRATCH" -ne 1 ]]; then
    echo "Refusing to run: $TARGET_REAL is not under \$SCRATCH or \$GROUP_SCRATCH." >&2
    exit 1
fi

echo "Target:     $TARGET_REAL"
echo "Age filter: files unmodified for > ${AGE_DAYS} days"
echo "Dry run:    $([[ $DRY_RUN -eq 1 ]] && echo yes || echo no)"
echo "Jobs:       $JOBS"

rewrite_one() {
    f="$1"
    if [[ ! -s "$f" ]]; then
        # Zero-byte file: no data to rewrite, so there's no "real write" to
        # perform. Left as-is rather than faked with a touch.
        echo "SKIP(empty): $f"
        return 0
    fi
    if dd if="$f" of="$f" bs=4M conv=notrunc,fsync status=none; then
        echo "OK: $f"
    else
        echo "FAIL(check write permission?): $f" >&2
    fi
}
export -f rewrite_one

if [[ "$DRY_RUN" -eq 1 ]]; then
    find "$TARGET_REAL" -type f -mtime +"$AGE_DAYS" -print
    exit 0
fi

find "$TARGET_REAL" -type f -mtime +"$AGE_DAYS" -print0 \
    | xargs -0 -P "$JOBS" -I{} bash -c 'rewrite_one "$@"' _ {}
