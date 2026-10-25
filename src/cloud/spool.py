"""The feed: a spool directory of arrival files, and a publisher that
replays positions into it by arrival time.

The publisher writes one Parquet file per window of arrival time, named by
the window's start so that name order is arrival order, and writes an
``END`` marker after the last. The processor reads the files waiting,
and moves them to ``done/`` only after its checkpoint records them as
consumed. In a cloud deployment the spool is a queue or a bucket prefix;
the processor sees the same three operations (list, read, mark).
"""

import os
import time
from collections.abc import Callable
from datetime import timedelta
from pathlib import Path

import polars as pl

from cloud.files import write_atomic, write_parquet
from vessels.stream.micro import batches

END = "END"
DONE = "done"
# UTC basic format: sorts as time and is safe in file names.
STAMP = "%Y%m%dT%H%M%SZ"


class SpoolSource:
    """Arrival files waiting in a directory.

    Attributes:
        root: The spool directory.
    """

    def __init__(self, root: Path):
        """Use a spool directory, creating it if absent.

        Args:
            root: The spool directory.
        """
        self.root = root
        (root / DONE).mkdir(parents=True, exist_ok=True)

    def ended(self) -> bool:
        """Whether the publisher has written its last file.

        Check this before ``pending``: files listed after the marker is
        seen are then the whole feed.

        Returns:
            bool: True once ``END`` exists.
        """
        return (self.root / END).exists()

    def pending(self) -> list[str]:
        """Names of complete files not yet marked read, in arrival order.

        Returns:
            list[str]: File names.
        """
        return sorted(p.name for p in self.root.glob("*.parquet"))

    def read(self, names: list[str]) -> pl.DataFrame | None:
        """Positions in the named files.

        Args:
            names: File names from ``pending``.

        Returns:
            pl.DataFrame | None: Their rows in file order, or None for no
            files.
        """
        if not names:
            return None
        return pl.concat([pl.read_parquet(self.root / name) for name in names])

    def mark(self, names: list[str]) -> None:
        """Move files to ``done/``; a file already moved is skipped, so
        marking twice is harmless.

        Args:
            names: File names.
        """
        for name in names:
            path = self.root / name
            if path.exists():
                os.replace(path, self.root / DONE / name)


def publish(
    arrived: pl.DataFrame,
    root: Path,
    every: timedelta,
    speed: float | None = None,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> int:
    """Replay positions into a spool, one file per window of arrival time.

    Args:
        arrived: Positions with an ``arrival`` column (``sim.arrival``).
        root: The spool directory.
        every: Window length.
        speed: Arrival time per unit of wall time (3600 plays an hour a
            second); None writes as fast as it can.
        sleep: Waits a number of seconds (replaced in tests).
        clock: Seconds on a monotonic clock (replaced in tests).

    Returns:
        int: Files written.
    """
    root.mkdir(parents=True, exist_ok=True)
    started, first, written = clock(), None, 0
    for end, part in batches(arrived, every):
        first = end if first is None else first
        if speed:
            # A window's positions are all available at its end.
            wait = (end - first).total_seconds() / speed - (clock() - started)
            if wait > 0:
                sleep(wait)
        write_parquet(root / f"{(end - every).strftime(STAMP)}.parquet", part)
        written += 1
    write_atomic(root / END, lambda path: path.write_text("", encoding="utf-8"))
    return written
