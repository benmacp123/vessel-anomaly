"""Check a finished service run against a batch run over the same days.

Passes when the feed was consumed in full, every arrival (the simulated
feed's duplicates included) was received exactly once, every committed
event equals the batch detectors' output and no event ID repeats. Reads the run's directories without changing them, so
it can run beside the map. Exits non-zero on the first failure.
"""

import argparse
import json
import sys
from pathlib import Path

import polars as pl
from polars.testing import assert_frame_equal

from cloud.sink import FileSink
from cloud.spool import DONE, END
from vessels.evaluation import injection
from vessels.evaluation.inputs import EXTRACT, inputs
from vessels.io.marinecadastre import read_extract
from vessels.sim.arrival import SCENARIOS, arrivals
from vessels.stream.micro import run_batch


def fail(message: str) -> None:
    """Report a failed check and exit.

    Args:
        message: What failed.
    """
    print(f"FAIL: {message}")
    sys.exit(1)


def main() -> None:
    """Parse arguments and run the checks."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--block", choices=("tuning", "heldout"), default="tuning")
    parser.add_argument("--days", type=int, default=3, help="first days of the block")
    parser.add_argument("--spool", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--alerts", type=Path, required=True)
    args = parser.parse_args()

    tick = int((args.state / "CURRENT").read_text(encoding="utf-8"))
    folder = args.state / "ticks" / f"{tick:08d}"
    if not json.loads((folder / "tick.json").read_text(encoding="utf-8"))["finished"]:
        fail(f"tick {tick} committed but the feed has not been flushed")
    if not (args.spool / END).exists() or list(args.spool.glob("*.parquet")):
        fail("spool still has files waiting")

    days = sorted((EXTRACT / args.block).glob("*.parquet"))[: args.days]
    positions = read_extract(days).collect()
    # The feed delivers some positions twice, as the publisher simulated.
    arrived = arrivals(positions, SCENARIOS[injection.ARRIVAL], injection.ARRIVAL_SEED)
    published = sum(
        pl.scan_parquet(p).select(pl.len()).collect().item()
        for p in (args.spool / DONE).glob("*.parquet")
    )
    counts = json.loads((folder / "stream.json").read_text(encoding="utf-8"))["counts"]
    if not arrived.height == published == counts["received"]:
        fail(
            f"{arrived.height} arrivals, {published} published,"
            f" {counts['received']} received"
        )
    print(f"tick {tick}: all {published} arrivals received once; {counts}")

    batch = run_batch(positions, inputs(args.block))
    events, incidents = FileSink(args.alerts).read(tick)
    for name, expected in batch.items():
        try:
            assert_frame_equal(events[name], expected, rel_tol=1e-12, abs_tol=1e-12)
        except AssertionError as error:
            fail(f"{name} differs from batch: {error}")
        print(f"{name}: {expected.height} events, equal to batch")
    ids = [i for frame in events.values() for i in frame["event_id"]]
    if len(ids) != len(set(ids)):
        fail(f"{len(ids) - len(set(ids))} repeated event IDs")
    print(f"{len(ids)} event IDs, all unique; {len(incidents)} incident transitions")
    print("PASS")


if __name__ == "__main__":
    main()
