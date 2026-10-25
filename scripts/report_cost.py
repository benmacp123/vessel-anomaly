"""Write the cost note from committed files: ``make cost``.

Reads the measured work (``results/cloud/measured.json``, from ``make
measure``), the price snapshot (``data/prices/aws-us-east-1.json``, from
``make prices``) and the extract's daily all-traffic counts, and writes
``results/cloud/cost.json`` and ``results/cloud/COST.md``. Needs no
network or account, and rebuilds byte for byte from the same inputs.
"""

import argparse
import json
from pathlib import Path

import polars as pl

from cloud import cost

ROOT = Path(__file__).resolve().parent.parent
MEASURED = ROOT / "results" / "cloud" / "measured.json"
PRICES = ROOT / "data" / "prices" / "aws-us-east-1.json"
COUNTS = ROOT / "data" / "extract" / "us" / "counts.csv"
OUT = ROOT / "results" / "cloud"

# Global volumes from the literature. GFW's 2016 figure was read on the
# page; the later one comes from a search result whose page was not
# reachable (2026-10-25), so it is unverified.
GLOBAL = [
    {
        "name": "Global, GFW 2016",
        "positions_per_day": 20_000_000,
        "label": "literature",
        "source": "Global Fishing Watch, 'What does an AIS message look like"
        " anyway?' (8 November 2016): 'more than 20 million AIS messages a"
        " day', https://globalfishingwatch.org/?p=1318",
    },
    {
        "name": "Global, GFW later",
        "positions_per_day": 110_000_000,
        "label": "literature, unverified",
        "source": "Global Fishing Watch datasets page as quoted in search"
        " results ('over 110 million AIS messages each day'); undated, page"
        " not reachable on 2026-10-25",
    },
]
KEEPS_UP = {
    "service_lambda": cost.lambda_keeps_up,
    "service_fargate": cost.fargate_keeps_up,
}
DESIGNS = {
    "per_message": "Per message (collar pattern)",
    "service_lambda": "Service on Lambda",
    "service_fargate": "Service on Fargate",
}


def volumes(counts: pl.DataFrame) -> list[dict]:
    """The volumes priced: the box's tuning days, then global figures.

    Args:
        counts: ``counts.csv``.

    Returns:
        list[dict]: Name, positions a day, label and source of each.
    """
    tuning = counts.filter(pl.col("block") == "tuning")["all_traffic_rows"]
    source = (
        "all traffic in the study box, tuning block (data/extract/us/counts.csv);"
        " MarineCadastre as published, filtered to a one-minute rate"
    )
    return [
        {
            "name": "Regional, median day",
            "positions_per_day": float(tuning.median()),
            "label": "observed",
            "source": source,
        },
        {
            "name": "Regional, busiest day",
            "positions_per_day": float(tuning.max()),
            "label": "observed",
            "source": source,
        },
    ] + GLOBAL


def compute(measured: dict, snapshot: dict, counts: pl.DataFrame) -> dict:
    """Price every design and transport at every volume.

    Args:
        measured: The measured work.
        snapshot: The price snapshot.
        counts: The extract's daily counts.

    Returns:
        dict: Work, prices and, per volume, each design's bill, the
        service's duty and slow-tick time, and each transport's bill.
    """
    work = cost.Work.from_measured(measured)
    prices = cost.unit_prices(snapshot)
    out = []
    for volume in volumes(counts):
        rate = volume["positions_per_day"]
        out.append(
            volume
            | {
                "scale": cost.scale(work, rate),
                "duty_vcpu": cost.duty(work, rate),
                "lambda_tick_seconds_p95": cost.tick_seconds(work, rate, load=True),
                "checkpoint_mb": work.checkpoint_bytes * cost.scale(work, rate) / 2**20,
                "keeps_up": {
                    name: check(work, rate) for name, check in KEEPS_UP.items()
                },
                "designs": {
                    name: cost.bill(getattr(cost, name)(work, rate), prices)
                    for name in DESIGNS
                },
                "transports": {
                    name: cost.bill(lines, prices)
                    for name, lines in cost.transports(work, rate).items()
                },
            }
        )
    return {
        "measured": measured["measured"],
        "machine": measured["machine"],
        "prices_retrieved": snapshot["retrieved"],
        "region": snapshot["region"],
        "work": vars(work),
        "volumes": out,
    }


def queue_and_state(priced: dict) -> float:
    """Share of a per-message bill spent on the queue and state table.

    Args:
        priced: The per-message design's ``cost.bill``.

    Returns:
        float: The share, 0 to 1.
    """
    spent = sum(
        line["usd"]
        for line in priced["lines"]
        if line["name"].startswith(("SQS", "DynamoDB"))
    )
    return spent / priced["total_usd"]


def usd(value: float) -> str:
    """Dollars for a table: cents below $100, whole dollars above.

    Args:
        value: USD.

    Returns:
        str: Such as ``$0.42`` or ``$1,234``.
    """
    return f"${value:,.2f}" if value < 100 else f"${value:,.0f}"


def count(value: float) -> str:
    """A large quantity, to three significant figures.

    Args:
        value: The quantity.

    Returns:
        str: Such as ``420k`` or ``1.23M``.
    """
    for size, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "k")):
        if abs(value) >= size:
            return f"{value / size:.3g}{suffix}"
    return f"{value:.3g}"


def table(header: list[str], rows: list[list[str]]) -> list[str]:
    """A Markdown table.

    Args:
        header: Column names.
        rows: Cells.

    Returns:
        list[str]: Its lines.
    """
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return lines + ["| " + " | ".join(row) + " |" for row in rows]


def markdown(result: dict, measured: dict) -> str:
    """The cost note.

    Args:
        result: ``compute``.
        measured: The measured work.

    Returns:
        str: Markdown.
    """
    work, service = result["work"], measured["service"]
    machine = result["machine"]
    lines = [
        "# Cost note",
        "",
        "Generated by `make cost` from committed files; do not edit by hand."
        f" Prices: AWS {result['region']} public Price List, retrieved"
        f" {result['prices_retrieved']}, list rates before free tiers."
        f" Work: measured {result['measured']} on {machine['cpu']}"
        f" (Python {machine['python']}, polars {machine['polars']},"
        f" {machine['polars_threads']} thread). **Projected, not billed:**"
        " nothing was deployed.",
        "",
        "## Measured work",
        "",
        f"The first {len(measured['days'])} tuning days"
        f" ({measured['positions']:,} positions from {measured['vessels']}"
        f" vessels; {measured['arrivals']:,} arrivals with the simulated"
        " feed's duplicates), run four ways. CPU time per position:",
        "",
    ]
    lines += table(
        ["Path", "CPU per position", "Against batch"],
        [
            [
                label,
                f"{seconds * 1000:.3g} ms",
                f"{seconds / work['batch_cpu_per_position']:,.0f}×",
            ]
            for label, seconds in (
                ("Batch: detectors once over the span", work["batch_cpu_per_position"]),
                ("Stream fed hourly, in memory", work["hourly_cpu_per_position"]),
                (
                    f"Service, {work['tick_minutes']}-minute ticks with checkpoint and"
                    " sink I/O",
                    work["tick_cpu_per_position"],
                ),
                (
                    "One position at a time, in memory (mean)",
                    work["single_row_seconds"],
                ),
            )
        ],
    )
    single = measured["single_row"]
    lines += [
        "",
        "Lower latency costs more compute: each push that moves the"
        " watermark runs every detector over the vessels' tails, whatever"
        " it carries. Of the"
        f" {single['rows']} positions pushed alone after a"
        f" {single['warm_hours']:.0f}-hour warm-up, {single['advanced']}"
        " moved it (median"
        f" {single['seconds']['median'] * 1000:.2f} ms, 95th percentile"
        f" {single['seconds']['p95'] * 1000:.0f} ms).",
        "",
        f"Service ticks: {service['ticks']} ticks, 95th percentile"
        f" {service['tick_seconds']['p95']:.2f} s;"
        f" {service['puts_per_tick']:.1f} objects"
        f" ({service['bytes_written_per_tick'] / 2**20:.2f} MB) written per tick;"
        f" checkpoint mean {service['checkpoint_bytes']['mean'] / 2**20:.2f} MB"
        f" (largest {service['checkpoint_bytes']['max'] / 2**20:.2f} MB);"
        f" loading it back takes {service['load']['seconds']['mean'] * 1000:.0f} ms"
        f" from {service['load']['files']:.0f} files. One position is"
        f" {work['json_bytes']:.0f} bytes as JSON.",
        "",
        "## Monthly cost by volume",
        "",
    ]
    rows = []
    for volume in result["volumes"]:
        designs = volume["designs"]
        rows.append(
            [
                volume["name"],
                f"{count(volume['positions_per_day'])} ({volume['label']})",
                f"{volume['scale']:,.0f}×",
            ]
            + [
                usd(designs[name]["total_usd"])
                + ("" if volume["keeps_up"].get(name, True) else " †")
                for name in DESIGNS
            ]
        )
    lines += table(
        ["Volume", "Positions a day", "Of measured"] + list(DESIGNS.values()),
        rows,
    )
    lines += [
        "",
        "† Does not keep up as built: see the next table.",
        "",
        "### Does one process keep up?",
        "",
    ]
    lines += table(
        [
            "Volume",
            "Service vCPUs busy",
            "Slow Lambda tick",
            "Checkpoint",
            "Lambda keeps up",
            "Fargate keeps up",
        ],
        [
            [
                v["name"],
                f"{v['duty_vcpu']:.2f}",
                f"{v['lambda_tick_seconds_p95']:,.0f} s",
                f"{v['checkpoint_mb']:,.0f} MB",
            ]
            + ["yes" if v["keeps_up"][n] else "no" for n in KEEPS_UP]
            for v in result["volumes"]
        ],
    )
    lines += [
        "",
        "vCPUs busy is the average share of one vCPU the ticks need. A"
        " Lambda tick must end within its"
        f" {work['tick_minutes'] * 60} s window and Lambda's"
        f" {cost.LAMBDA_MAX_SECONDS} s limit; the slow tick is the measured"
        " 95th percentile, checkpoint load included, scaled with traffic."
        f" The Fargate task needs vCPUs busy below 1 and its"
        f" {cost.FARGATE_GB} GB to hold at least the checkpoint. Where"
        " neither keeps up, the stream would have to be split between"
        " processes by vessel, with encounters exchanged between them. That"
        " is not built, so those rows price compute the design does not yet"
        " have.",
        "",
        "## Worked example: regional median day",
        "",
    ]
    example = result["volumes"][0]
    for name, title in DESIGNS.items():
        priced = example["designs"][name]
        lines += [f"**{title}:** {usd(priced['total_usd'])} a month.", ""]
        lines += table(
            ["Line", "Quantity", "Unit price", "Monthly", "Basis"],
            [
                [
                    line["name"],
                    count(line["quantity"]),
                    f"${line['unit_usd']:.3g}",
                    usd(line["usd"]),
                    line["basis"],
                ]
                for line in priced["lines"]
            ],
        )
        lines.append("")
    lines += ["## Transport for the feed", ""]
    names = list(example["transports"])
    lines += table(
        ["Volume"] + names,
        [
            [v["name"]] + [usd(v["transports"][n]["total_usd"]) for n in names]
            for v in result["volumes"]
        ],
    )
    lines += [
        "",
        "Packing positions into messages, records or window files is what"
        " makes transport cheap. In the collar pattern, the per-message"
        " queue and state transactions are"
        f" {queue_and_state(example['designs']['per_message']):.0%} of the"
        " bill.",
        "",
        "## Volumes",
        "",
    ]
    lines += [
        f"- **{v['name']}** ({v['label']}): {v['positions_per_day']:,.0f} a"
        f" day; {v['source']}."
        for v in result["volumes"]
    ]
    lines += [
        "",
        "## Assumptions and limits",
        "",
        "- Work is scaled from the extract in proportion to positions"
        " (provisional). The extract holds one group's fishing vessels only"
        " (7% of the box's rows); a tick's work follows the state the stream"
        " carries, assumed to grow with traffic.",
        "- Laptop CPU seconds stand for Arm vCPU seconds (provisional);"
        " Lambda gives one vCPU at 1,769 MB, so CPU-bound work is priced at"
        " that size.",
        "- The per-message design's compute is this stream fed one row at a"
        " time, in memory: a lower bound, since a stateless function would"
        " also load each vessel's state. Its DynamoDB item size (1 KB) is"
        " assumed (provisional).",
        "- Regional volumes are MarineCadastre's published rows, already"
        " filtered to one a minute; a raw receiver feed is larger.",
        "- List prices before free tiers, one region, no data transfer,"
        " logs or alert delivery.",
        "",
        "## The same design on Google Cloud",
        "",
        "The service maps onto Apache Beam with little change. The spool"
        " becomes a Pub/Sub subscription; the reorder hold and watermark"
        " become Beam's event-time watermark with allowed lateness; the"
        " per-vessel tails, segmenter and episode lifecycle become a"
        " stateful DoFn keyed by vessel, with timers settling episodes; the"
        " checkpoint is Dataflow's own state, so the store disappears;"
        " encounters re-key resampled positions by H3 cell, as in GFW's"
        " pipelines, which run on Beam and BigQuery; and the alert sink"
        " becomes a BigQuery table. No GCP prices were researched, so this"
        " note does not price it.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    """Parse arguments and write the cost note."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--measured", type=Path, default=MEASURED)
    parser.add_argument("--prices", type=Path, default=PRICES)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    measured = json.loads(args.measured.read_text(encoding="utf-8"))
    snapshot = json.loads(args.prices.read_text(encoding="utf-8"))
    result = compute(measured, snapshot, pl.read_csv(COUNTS))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "cost.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    (args.out / "COST.md").write_text(markdown(result, measured), encoding="utf-8")
    print(f"wrote {args.out / 'COST.md'}")


if __name__ == "__main__":
    main()
