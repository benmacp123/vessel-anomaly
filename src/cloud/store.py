"""State store: stream checkpoints in a directory, one generation per tick,
committed by an atomic pointer.

Layout::

    LOCK                 held by the one processor using the store
    CURRENT              number of the committed tick
    ticks/<tick>/        the checkpoint's frames (Parquet), the stream's
                         dict (stream.json) and the tick's own record
                         (tick.json: files consumed, clock, finished,
                         repeat-key parts)
    seen/<tick>.parquet  repeat keys first saved at that tick

A tick is written in full under its own number, then ``CURRENT`` is
replaced to point at it: a crash before that leaves the previous tick
committed and the new one ignored. Older generations are removed after the
commit.

Repeat keys are one per moving position ever heard and never shrink
(measured: 6.8 MB of a 12.3 MB checkpoint after a month), so rewriting
them every tick would cost more each tick. Each tick writes only the keys
it added; after ``compact_after`` parts they are rewritten as one.
"""

import fcntl
import shutil
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from cloud.files import read_json, sync, write_atomic, write_json, write_parquet

SEEN = "seen"
KEY = ["vessel_id", "key"]


class StoreLocked(RuntimeError):
    """Another processor holds the store."""


@dataclass
class Saved:
    """One committed tick.

    Attributes:
        tick: Tick number, from 1.
        frames: ``checkpoint.snapshot`` frames.
        meta: ``checkpoint.snapshot`` dict.
        consumed: Spool files the tick read.
        clock: Latest arrival time read so far, as ISO 8601 text, or None.
        finished: Whether the feed ended and the stream was flushed.
    """

    tick: int
    frames: dict[str, pl.DataFrame]
    meta: dict
    consumed: list[str]
    clock: str | None
    finished: bool


def _name(tick: int) -> str:
    """A tick's directory or file stem, padded so names sort as numbers.

    Args:
        tick: Tick number.

    Returns:
        str: The name.
    """
    return f"{tick:08d}"


class DirectoryStore:
    """Checkpoints in a local directory.

    Attributes:
        root: The store directory.
        compact_after: Repeat-key parts kept before they are merged.
    """

    def __init__(self, root: Path, compact_after: int = 64):
        """Open a store, creating its directories if absent.

        Args:
            root: The store directory.
            compact_after: Repeat-key parts kept before they are merged.
        """
        self.root = root
        self.compact_after = compact_after
        (root / "ticks").mkdir(parents=True, exist_ok=True)
        (root / SEEN).mkdir(exist_ok=True)
        self._lock = None
        # Repeat keys as last committed, and the parts holding them.
        self._seen = pl.DataFrame(schema={"vessel_id": pl.String, "key": pl.String})
        self._parts: list[str] = []

    def lock(self) -> None:
        """Take the store for this process until it exits.

        Two processors on one store would each commit ticks the other
        never sees.

        Raises:
            StoreLocked: If another process holds it.
        """
        handle = (self.root / "LOCK").open("a", encoding="utf-8")
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            handle.close()
            raise StoreLocked(f"{self.root} is in use by another processor") from error
        self._lock = handle

    def unlock(self) -> None:
        """Release the store."""
        if self._lock is not None:
            self._lock.close()
            self._lock = None

    def committed(self) -> int:
        """The committed tick's number.

        Returns:
            int: Tick number, or 0 before the first commit.
        """
        path = self.root / "CURRENT"
        return int(path.read_text(encoding="utf-8")) if path.exists() else 0

    def load(self) -> Saved | None:
        """Read the committed tick.

        Returns:
            Saved | None: The tick, or None before the first commit.
        """
        tick = self.committed()
        if not tick:
            self._tidy(tick)
            return None
        folder = self.root / "ticks" / _name(tick)
        record = read_json(folder / "tick.json")
        # Read the parts list before tidying, which keeps only those parts.
        self._parts = record["seen_parts"]
        self._tidy(tick)
        frames = {p.stem: pl.read_parquet(p) for p in sorted(folder.glob("*.parquet"))}
        self._seen = pl.concat(
            [self._seen.clear()]
            + [pl.read_parquet(self.root / SEEN / f"{p}.parquet") for p in self._parts]
        )
        frames[SEEN] = self._seen.sort(KEY)
        return Saved(
            tick,
            frames,
            read_json(folder / "stream.json"),
            record["consumed"],
            record["clock"],
            record["finished"],
        )

    def save(self, saved: Saved) -> None:
        """Write a tick and commit it.

        Args:
            saved: The tick; its number must follow the committed one.

        Raises:
            ValueError: If the tick does not follow the committed one.
        """
        if saved.tick != self.committed() + 1:
            raise ValueError(f"tick {saved.tick} after {self.committed()}")
        seen, parts = self._write_seen(saved.tick, saved.frames[SEEN])
        folder = self.root / "ticks" / _name(saved.tick)
        partial = folder.with_name(folder.name + ".partial")
        for leftover in (folder, partial):
            shutil.rmtree(leftover, ignore_errors=True)
        partial.mkdir()
        for name, frame in saved.frames.items():
            if name != SEEN:
                write_parquet(partial / f"{name}.parquet", frame)
        write_json(partial / "stream.json", saved.meta)
        record = {
            "consumed": saved.consumed,
            "clock": saved.clock,
            "finished": saved.finished,
            "seen_parts": parts,
        }
        write_json(partial / "tick.json", record)
        partial.rename(folder)
        sync(folder.parent)
        self._commit(saved.tick)
        self._seen, self._parts = seen, parts
        self._tidy(saved.tick)

    def _write_seen(self, tick: int, seen: pl.DataFrame) -> tuple[pl.DataFrame, list]:
        """Write the repeat keys added since the last commit.

        Args:
            tick: The tick being written.
            seen: Every repeat key the stream holds.

        Returns:
            tuple: The keys and the part names the tick will record.
        """
        added = seen.join(self._seen, on=KEY, how="anti")
        parts = list(self._parts)
        # Keys only ever grow; if some went missing, append-only parts
        # would bring them back, so rewrite the lot.
        lost = seen.height != self._seen.height + added.height
        if lost or len(parts) >= self.compact_after:
            added, parts = seen, []
        if added.height:
            write_parquet(self.root / SEEN / f"{_name(tick)}.parquet", added)
            parts.append(_name(tick))
        return seen, parts

    def _commit(self, tick: int) -> None:
        """Point ``CURRENT`` at a written tick: the commit.

        Args:
            tick: Tick number.
        """
        text = f"{tick}\n"
        write_atomic(
            self.root / "CURRENT",
            lambda path: path.write_text(text, encoding="utf-8"),
        )

    def _tidy(self, tick: int) -> None:
        """Remove generations and repeat-key parts the committed tick does
        not use, including any left by a crash.

        Args:
            tick: The committed tick.
        """
        for folder in (self.root / "ticks").iterdir():
            if folder.name != _name(tick):
                shutil.rmtree(folder)
        for part in (self.root / SEEN).iterdir():
            if part.stem not in self._parts or part.suffix != ".parquet":
                part.unlink()
