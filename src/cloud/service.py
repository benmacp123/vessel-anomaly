"""The processor: reads arrival files, runs them through the checkpointed
micro-batch stream and writes settled alerts.

One ``tick`` takes the files waiting (all of them, or the oldest
``max_files`` so a restart after downtime catches up in bounded steps) and
then:

1. pushes their positions through the stream (and flushes it once the
   feed has ended);
2. writes the alerts it settled to the sink under the tick's number;
3. saves the stream's checkpoint with the names of the files consumed and
   commits it: the commit point;
4. rewrites the map's GeoJSON and marks the files read.

A crash before the commit leaves the previous tick committed: on restart
the uncommitted alerts are removed and the files, still waiting, are read
again. A crash after it leaves files waiting that the checkpoint already
records as consumed: on restart they are marked read without being
pushed. Either way no position is pushed twice and no alert is lost or
written twice, so a restarted run gives the same events and counts as one
that never stopped (``tests/test_service.py``).

Ports are duck-typed (``spool.SpoolSource``, ``store.DirectoryStore``,
``sink.FileSink``); a cloud deployment would swap them for a queue, an
object store and a table without touching this module.
"""

import time
from collections.abc import Callable
from datetime import datetime, timedelta

from cloud.sink import FileSink
from cloud.spool import SpoolSource
from cloud.store import DirectoryStore, Saved
from vessels.stream.checkpoint import restore, snapshot
from vessels.stream.micro import Inputs, MicroBatchStream


class Processor:
    """Runs the stream over a spool, one tick at a time.

    Attributes:
        source: Where arrival files wait.
        store: Where checkpoints are committed.
        sink: Where alerts go.
        tick_number: The committed tick.
        finished: Whether the feed ended and the stream was flushed.
        max_files: Most files read per tick, or None for all.
    """

    def __init__(
        self,
        source: SpoolSource,
        store: DirectoryStore,
        sink: FileSink,
        inputs: Inputs,
        hold: timedelta,
        max_files: int | None = None,
    ):  # pylint: disable=too-many-arguments,too-many-positional-arguments
        """Resume from the committed tick, or start empty.

        Takes the store's lock, removes alerts of uncommitted ticks and
        marks read any file the committed tick consumed.

        Args:
            source: Where arrival files wait.
            store: Where checkpoints are committed.
            sink: Where alerts go.
            inputs: Reference layers and thresholds, the same on every
                start.
            hold: Reorder hold, used only when starting empty.
            max_files: Most files read per tick, or None for all.
        """
        self.source, self.store, self.sink = source, store, sink
        self.max_files = max_files
        store.lock()
        saved = store.load()
        if saved is None:
            self.stream = MicroBatchStream(inputs, hold)
            self.tick_number, self.finished, self._clock = 0, False, None
        else:
            self.stream = restore(saved.frames, saved.meta, inputs)
            self.tick_number, self.finished = saved.tick, saved.finished
            self._clock = (
                None if saved.clock is None else datetime.fromisoformat(saved.clock)
            )
            source.mark(saved.consumed)
        sink.discard_after(self.tick_number)
        sink.publish(self.tick_number)

    def tick(self) -> bool:
        """Process the files waiting, up to ``max_files``.

        Returns:
            bool: Whether there was anything to do.
        """
        if self.finished:
            return False
        # Checked before listing, so that once ended the list is complete.
        ended = self.source.ended()
        names = self.source.pending()
        if self.max_files is not None and len(names) > self.max_files:
            names, ended = names[: self.max_files], False
        if not names and not ended:
            return False
        rows = self.source.read(names)
        if rows is not None and rows.height:
            latest = rows["arrival"].max()
            self._clock = latest if self._clock is None else max(self._clock, latest)
            self.stream.push(rows, now=self._clock)
        if ended:
            self.stream.flush(now=self._clock)
        events, incidents = self.stream.drain()
        tick = self.tick_number + 1
        self.sink.write(tick, events, incidents)
        frames, meta = snapshot(self.stream)
        clock = None if self._clock is None else self._clock.isoformat()
        self.store.save(Saved(tick, frames, meta, names, clock, ended))
        self.tick_number, self.finished = tick, ended
        self.sink.publish(tick)
        self.source.mark(names)
        return True

    def run(
        self, poll: float = 1.0, sleep: Callable[[float], None] = time.sleep
    ) -> int:
        """Tick until the feed has ended and been flushed.

        Args:
            poll: Seconds to wait when nothing is waiting.
            sleep: Waits a number of seconds (replaced in tests).

        Returns:
            int: The committed tick.
        """
        while not self.finished:
            if not self.tick():
                sleep(poll)
        return self.tick_number

    def close(self) -> None:
        """Release the store for another processor."""
        self.store.unlock()
