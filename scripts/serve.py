"""Run the processor over a spool until the feed ends.

Resumes from the store's committed tick, so it can be stopped (or killed)
and started again. Alerts go to ``--alerts`` (Parquet parts per tick and
``alerts.geojson``). The reference layers are the block's, as in the
evaluation; the reorder hold is the evaluation's arrival scenario's.
"""

import argparse
import time
from pathlib import Path

from cloud.service import Processor
from cloud.sink import FileSink
from cloud.spool import SpoolSource
from cloud.store import DirectoryStore
from vessels.evaluation import injection
from vessels.evaluation.inputs import inputs
from vessels.sim.arrival import SCENARIOS


def main() -> None:
    """Parse arguments and run until the feed ends."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--block", choices=("tuning", "heldout"), default="tuning")
    parser.add_argument("--spool", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--alerts", type=Path, required=True)
    parser.add_argument("--poll", type=float, default=1.0, help="seconds")
    parser.add_argument("--max-files", type=int, default=None)
    args = parser.parse_args()
    processor = Processor(
        SpoolSource(args.spool),
        DirectoryStore(args.state),
        FileSink(args.alerts),
        inputs(args.block),
        SCENARIOS[injection.ARRIVAL].hold,
        args.max_files,
    )
    print(f"resuming after tick {processor.tick_number}", flush=True)
    while not processor.finished:
        clock = time.monotonic()
        if processor.tick():
            counts = processor.stream.counts
            print(
                f"tick {processor.tick_number}: {counts['received']} received,"
                f" {time.monotonic() - clock:.2f} s",
                flush=True,
            )
        else:
            time.sleep(args.poll)
    processor.close()
    print("feed ended; stream flushed", flush=True)


if __name__ == "__main__":
    main()
