"""Replay a block's first days into a spool directory, as a timed feed.

Positions get simulated arrival times (the evaluation's arrival scenario
and seed) and are written one file per window of arrival time; ``--speed``
sets how much arrival time passes per second (3600: an hour a second;
omitted: as fast as possible). Run ``scripts/serve.py`` on the same spool.
"""

import argparse
from datetime import timedelta
from pathlib import Path

from cloud.spool import publish
from vessels.evaluation import injection
from vessels.evaluation.inputs import EXTRACT
from vessels.io.marinecadastre import read_extract
from vessels.sim.arrival import SCENARIOS, arrivals


def main() -> None:
    """Parse arguments and replay."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--block", choices=("tuning", "heldout"), default="tuning")
    parser.add_argument("--days", type=int, default=3, help="first days of the block")
    parser.add_argument("--spool", type=Path, required=True)
    parser.add_argument("--every", type=int, default=10, help="window, minutes")
    parser.add_argument("--speed", type=float, default=None)
    args = parser.parse_args()
    days = sorted((EXTRACT / args.block).glob("*.parquet"))[: args.days]
    scene = SCENARIOS[injection.ARRIVAL]
    arrived = arrivals(read_extract(days).collect(), scene, injection.ARRIVAL_SEED)
    written = publish(arrived, args.spool, timedelta(minutes=args.every), args.speed)
    print(f"{arrived.height} positions in {written} files to {args.spool}")


if __name__ == "__main__":
    main()
