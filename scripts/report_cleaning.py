"""Report what cleaning and segmentation do to one block of the US extract.

Runs from committed files only: ``make report-cleaning`` (tuning block).
Writes, under ``results/cleaning/``:

- ``<block>_rules.csv``: rows or values affected per rule, with the
  MMSI rules applied in the extract build taken from ``counts.csv``;
- ``<block>_segments.csv``: segment counts, point statuses and vessels
  with overlapping segments (likely shared MMSIs);
- ``<block>_overlaps.csv``: those vessels and their overlap counts.

The held-out block is reported only at the one-shot scoring in Phase 5.
"""

import argparse
import csv
from pathlib import Path

import polars as pl

from vessels.core.cleaning import clean
from vessels.core.segment import overlapping_vessels, segment, segment_summary
from vessels.io.marinecadastre import read_extract

ROOT = Path(__file__).resolve().parent.parent
EXTRACT = ROOT / "data" / "extract" / "us"
OUT = ROOT / "results" / "cleaning"


def mmsi_rule_rows(block: str) -> list[dict]:
    """Rows the extract build dropped per MMSI rule for a block.

    Args:
        block: ``tuning`` or ``heldout``.

    Returns:
        list[dict]: ``stage``, ``rule``, ``kind`` and ``count`` per rule.
    """
    with (EXTRACT / "counts.csv").open(encoding="utf-8") as handle:
        days = [row for row in csv.DictReader(handle) if row["block"] == block]
    rules = [c for c in days[0] if c.startswith("mmsi_") and c.endswith("_rows")]
    return [
        {
            "stage": "extract build",
            "rule": rule.removesuffix("_rows"),
            "kind": "rows dropped",
            "count": sum(int(day[rule]) for day in days),
        }
        for rule in rules
    ]


def report(block: str) -> tuple[list[dict], dict, pl.DataFrame]:
    """Clean and segment a block and count what each step did.

    Args:
        block: ``tuning`` or ``heldout``.

    Returns:
        tuple[list[dict], dict, pl.DataFrame]: Per-rule counts, segment
        statistics, and the vessels with overlapping segments.
    """
    raw = read_extract(sorted((EXTRACT / block).glob("*.parquet"))).collect()
    result = clean(raw)
    segmented = segment(result.frame)
    summary = segment_summary(segmented)
    overlaps = overlapping_vessels(summary)
    statuses = dict(segmented.group_by("point_status").len().iter_rows())
    rules = mmsi_rule_rows(block)
    rules += [
        {"stage": "clean", "rule": r, "kind": "rows dropped", "count": n}
        for r, n in result.dropped.items()
    ]
    rules += [
        {"stage": "clean", "rule": r, "kind": "values nulled", "count": n}
        for r, n in result.nulled.items()
    ]
    rules += [
        {"stage": "segment", "rule": s, "kind": "rows without segment", "count": n}
        for s, n in sorted(statuses.items())
        if s != "position"
    ]
    stats = {
        "block": block,
        "extract_rows": raw.height,
        "clean_rows": result.frame.height,
        "segmented_rows": statuses.get("position", 0),
        "vessels": summary["vessel_id"].n_unique(),
        "segments": summary.height,
        "short_segments": int(summary["short"].sum()),
        "rows_in_short_segments": int(summary.filter("short")["points"].sum()),
        "median_segments_per_vessel": summary.group_by("vessel_id")
        .len()["len"]
        .median(),
        "vessels_with_overlapping_segments": overlaps.height,
    }
    return rules, stats, overlaps


def main():
    """Write the cleaning report for one block."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--block", choices=["tuning", "heldout"], default="tuning")
    args = parser.parse_args()
    rules, stats, overlaps = report(args.block)
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / f"{args.block}_rules.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rules[0]))
        writer.writeheader()
        writer.writerows(rules)
    pl.DataFrame([stats]).write_csv(OUT / f"{args.block}_segments.csv")
    overlaps.write_csv(OUT / f"{args.block}_overlaps.csv")
    for row in rules:
        print(
            f"{row['stage']:>13}  {row['rule']:<16} {row['kind']:<21} {row['count']:>8,}"
        )
    for key, value in stats.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
