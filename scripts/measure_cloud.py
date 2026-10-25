"""Measure the work the cost model prices, on this machine.

Runs the first days of the tuning block four ways and writes
``results/cloud/measured.json``:

- **batch**: the detectors once over the whole span;
- **hourly**: the stream fed hour by hour (no I/O);
- **service**: the processor over a spool of 10-minute files, as in the
  demo, with its checkpoint and sink on disk. Per tick: wall time, files
  written (each an object-store PUT) and bytes, and, every few ticks, the
  time to load and restore the committed checkpoint, which a stateless
  function would pay on every tick;
- **single row**: the stream warmed on the first day, then fed the next
  positions one at a time, as a per-message design would. The time is the
  stream's in-memory work only; loading and saving the vessel's state per
  message would come on top.

Times are wall-clock and CPU seconds on this machine, not on any cloud
service, so the file records the machine. ``make measure`` limits polars
to one thread, so wall time stands for one vCPU's work; with more threads
wall time falls below the CPU time a one-vCPU function would need. Re-running gives different times; the
cost report reads the committed file, so it does not change until this is
run again.
"""

import argparse
import json
import platform
import statistics
import tempfile
import time
from datetime import date
from pathlib import Path

import polars as pl

from cloud.service import Processor
from cloud.sink import FileSink
from cloud.spool import DONE, SpoolSource, publish
from cloud.store import DirectoryStore
from vessels.evaluation import injection
from vessels.evaluation.inputs import EXTRACT, inputs
from vessels.io.marinecadastre import read_extract
from vessels.sim.arrival import SCENARIOS, arrivals
from vessels.stream.checkpoint import restore
from vessels.stream.micro import DAY, HOUR, MicroBatchStream, batches, replay
from vessels.stream.micro import run_batch

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "cloud" / "measured.json"
WINDOW_MINUTES = 10
SINGLE_ROWS = 300
LOAD_EVERY = 12


def cpu_model() -> str:
    """The processor's model name, where the system reports one.

    Returns:
        str: Such as ``AMD Ryzen 7 7840U``, or the platform's own name.
    """
    info = Path("/proc/cpuinfo")
    if info.exists():
        for line in info.read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or platform.machine()


def spread(values: list[float]) -> dict:
    """Summary of a sample of times.

    Args:
        values: The sample, non-empty.

    Returns:
        dict: Count, mean, median, 95th percentile and maximum.
    """
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "mean": statistics.fmean(ordered),
        "median": statistics.median(ordered),
        "p95": ordered[min(len(ordered) - 1, round(0.95 * (len(ordered) - 1)))],
        "max": ordered[-1],
    }


def files(*roots: Path) -> dict[Path, tuple[int, int]]:
    """Every file under some directories, with its size and change time.

    Args:
        *roots: The directories.

    Returns:
        dict: Size in bytes and modification time (ns) by path.
    """
    found = {}
    for root in roots:
        for path in root.rglob("*"):
            if path.is_file():
                stat = path.stat()
                found[path] = (stat.st_size, stat.st_mtime_ns)
    return found


def size(root: Path) -> int:
    """Bytes in a directory's files.

    Args:
        root: The directory.

    Returns:
        int: Total size.
    """
    return sum(p.stat().st_size for p in root.rglob("*") if p.is_file())


def measure_service(arrived: pl.DataFrame, block: str, scratch: Path) -> dict:
    """Run the processor over a spool of the arrivals and time each tick.

    Args:
        arrived: Positions with arrival times.
        block: Block whose reference layers the processor uses.
        scratch: An empty directory for the spool, state and alerts.

    Returns:
        dict: Ticks, positions, per-tick time, writes and bytes, files
        read, checkpoint sizes and load-and-restore times.
    """
    spool, state, alerts = scratch / "spool", scratch / "state", scratch / "alerts"
    published = publish(arrived, spool, WINDOW_MINUTES * HOUR / 60)
    rows_per_file = [
        pl.scan_parquet(p).select(pl.len()).collect().item()
        for p in sorted(spool.glob("*.parquet"))
    ]
    spool_bytes = sum(p.stat().st_size for p in spool.glob("*.parquet"))
    layers = inputs(block)
    processor = Processor(
        SpoolSource(spool),
        DirectoryStore(state),
        FileSink(alerts),
        layers,
        SCENARIOS[injection.ARRIVAL].hold,
        max_files=1,
    )
    seconds, writes, written, checkpoint, loads = [], [], [], [], []
    cpu = time.process_time()
    while not processor.finished:
        before = files(state, alerts)
        clock = time.perf_counter()
        processor.tick()
        seconds.append(time.perf_counter() - clock)
        after = files(state, alerts)
        changed = [p for p, stamp in after.items() if before.get(p) != stamp]
        writes.append(len(changed))
        written.append(sum(after[p][0] for p in changed))
        checkpoint.append(size(state))
        if processor.tick_number % LOAD_EVERY == 0:
            clock = time.perf_counter()
            saved = DirectoryStore(state).load()
            restore(saved.frames, saved.meta, layers)
            loads.append(
                {
                    "seconds": time.perf_counter() - clock,
                    "files": sum(1 for p in state.rglob("*.parquet")) + 2,
                }
            )
    # Includes the bookkeeping above; small beside the ticks.
    cpu = time.process_time() - cpu
    processor.close()
    return {
        "window_minutes": WINDOW_MINUTES,
        "files": published,
        "files_read": len(list((spool / DONE).glob("*.parquet"))),
        "positions": sum(rows_per_file),
        "spool_bytes": spool_bytes,
        "ticks": len(seconds),
        "seconds": sum(seconds),
        "cpu_seconds": cpu,
        "tick_seconds": spread(seconds),
        "puts_per_tick": statistics.fmean(writes),
        "bytes_written_per_tick": statistics.fmean(written),
        "checkpoint_bytes": {
            "mean": statistics.fmean(checkpoint),
            "max": max(checkpoint),
        },
        "load": {
            "seconds": spread([load["seconds"] for load in loads]),
            "files": statistics.fmean(load["files"] for load in loads),
        },
        "counts": processor.stream.counts,
    }


def measure_single(arrived: pl.DataFrame, block: str) -> dict:
    """Warm a stream on the first day of arrivals, then push rows one at a
    time.

    Args:
        arrived: Positions with arrival times.
        block: Block whose reference layers the stream uses.

    Returns:
        dict: Warm-up span, rows pushed singly and their push times.
    """
    start = arrived["arrival"].min()
    warm = arrived.filter(pl.col("arrival") < start + DAY)
    rest = arrived.filter(pl.col("arrival") >= start + DAY).sort(
        "arrival", maintain_order=True
    )
    stream = MicroBatchStream(inputs(block), SCENARIOS[injection.ARRIVAL].hold)
    for now, part in batches(warm, WINDOW_MINUTES * HOUR / 60):
        stream.push(part, now)
    seconds, advanced = [], 0
    for index in range(min(SINGLE_ROWS, rest.height)):
        row = rest.slice(index, 1)
        released = stream._released  # pylint: disable=protected-access
        clock = time.perf_counter()
        stream.push(row, row["arrival"].item())
        seconds.append(time.perf_counter() - clock)
        # Most rows leave the watermark where it was and return at once;
        # the rest run every detector over the tails.
        advanced += stream._released != released  # pylint: disable=protected-access
    return {
        "warm_hours": DAY / HOUR,
        "rows": len(seconds),
        "advanced": advanced,
        "seconds": spread(seconds),
    }


def timed(work) -> dict:
    """Run some work and time it.

    Args:
        work: A function of no arguments.

    Returns:
        dict: Wall and CPU seconds.
    """
    wall, cpu = time.perf_counter(), time.process_time()
    work()
    return {
        "seconds": time.perf_counter() - wall,
        "cpu_seconds": time.process_time() - cpu,
    }


def main() -> None:
    """Parse arguments, measure and write the results."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--days", type=int, default=3, help="first days of tuning")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    block = "tuning"
    days = sorted((EXTRACT / block).glob("*.parquet"))[: args.days]
    positions = read_extract(days).collect()
    scene = SCENARIOS[injection.ARRIVAL]
    arrived = arrivals(positions, scene, injection.ARRIVAL_SEED)
    layers = inputs(block)

    batch = timed(lambda: run_batch(positions, layers))
    print(f"batch: {batch['seconds']:.1f} s", flush=True)
    hourly = timed(lambda: replay(arrived, layers, scene.hold, HOUR))
    print(f"hourly: {hourly['seconds']:.1f} s", flush=True)

    with tempfile.TemporaryDirectory() as scratch:
        service = measure_service(arrived, block, Path(scratch))
    print(f"service: {service['seconds']:.1f} s in {service['ticks']} ticks")

    single = measure_single(arrived, block)
    print(f"single row: median {single['seconds']['median'] * 1000:.0f} ms")

    result = {
        "measured": date.today().isoformat(),
        "machine": {
            "cpu": cpu_model(),
            "system": platform.system(),
            "python": platform.python_version(),
            "polars": pl.__version__,
            "polars_threads": pl.thread_pool_size(),
        },
        "block": block,
        "days": [p.stem for p in days],
        "positions": positions.height,
        "vessels": positions["vessel_id"].n_unique(),
        "arrivals": arrived.height,
        "json_bytes_per_position": len(positions.write_ndjson().encode("utf-8"))
        / positions.height,
        "batch": batch,
        "hourly": hourly,
        "service": service,
        "single_row": single,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
