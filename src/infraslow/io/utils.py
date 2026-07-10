"""Generic utilities for the :mod:`infraslow.io` layer (reading PSG/CSV inputs).

Small, reusable, io-specific building blocks.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Iterable, Iterator


def list_dir_filenames(directory: Path) -> set:
    """Return the set of entry names directly under ``directory`` (one readdir).

    Listing the directory once and testing membership in memory turns a bulk
    file-existence check from ``N`` per-file ``stat`` calls -- slow and hard on
    the Lustre metadata server for ``$OAK`` -- into a single ``readdir``.
    ``os.scandir`` is used so we never ``stat`` each entry.
    """
    with os.scandir(directory) as it:
        return {entry.name for entry in it}


def progress_iter(
    items: Iterable[Any],
    total: int,
    *,
    enabled: bool,
    desc: str,
) -> Iterator[Any]:
    """Yield ``items`` while showing a percentage progress indicator.

    Uses ``tqdm`` when available; otherwise falls back to a dependency-free
    percentage line printed to stderr (updated only when the integer percent
    changes, so it stays cheap). Disabled or empty input passes straight through.
    """
    if not enabled or total == 0:
        yield from items
        return
    try:
        from tqdm import tqdm  # noqa: PLC0415 - optional, imported lazily
    except ImportError:
        last_pct = -1
        for i, item in enumerate(items, 1):
            pct = i * 100 // total
            if pct != last_pct:
                last_pct = pct
                print(
                    f"\r{desc}: {pct:3d}% ({i}/{total})",
                    end="",
                    file=sys.stderr,
                    flush=True,
                )
            yield item
        print(file=sys.stderr)  # finish the line
        return
    yield from tqdm(items, total=total, desc=desc, unit="subj")


__all__ = ["list_dir_filenames", "progress_iter"]
