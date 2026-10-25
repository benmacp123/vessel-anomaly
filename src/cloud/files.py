"""Atomic file writes: a reader, or a process restarted after a crash, sees
a file whole or not at all.

Each write goes to a temporary name beside the target, is flushed to disk,
then renamed over the target; the rename is atomic on POSIX file systems.
The directory is flushed too, so the rename itself survives a power cut.
"""

import json
import os
from collections.abc import Callable
from pathlib import Path

import polars as pl

# Suffix of a file still being written; readers skip it.
PARTIAL = ".partial"


def sync(path: Path) -> None:
    """Flush a file or directory to disk.

    Args:
        path: The file or directory.
    """
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_atomic(path: Path, write: Callable[[Path], None]) -> None:
    """Write a file so that it appears complete or not at all.

    Args:
        path: The target file.
        write: Writes the content to the path it is given.
    """
    partial = path.with_name(path.name + PARTIAL)
    write(partial)
    sync(partial)
    os.replace(partial, path)
    sync(path.parent)


def write_parquet(path: Path, frame: pl.DataFrame) -> None:
    """Write a frame as Parquet, atomically.

    Args:
        path: The target file.
        frame: The frame.
    """
    write_atomic(path, frame.write_parquet)


def write_json(path: Path, data) -> None:
    """Write JSON, atomically, with sorted keys so equal data gives equal
    bytes.

    Args:
        path: The target file.
        data: JSON-ready data.
    """
    text = json.dumps(data, sort_keys=True, indent=1) + "\n"
    write_atomic(path, lambda target: target.write_text(text, encoding="utf-8"))


def read_json(path: Path):
    """Read a JSON file.

    Args:
        path: The file.

    Returns:
        The parsed data.
    """
    return json.loads(path.read_text(encoding="utf-8"))
