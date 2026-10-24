"""Run the per-vessel detectors over the tuning block and summarise them.

Run ``make detectors`` (after ``make coverage`` and ``make distance``).
Only counts are written, never event rows: a hashed ID with exact times and
positions would let a reader rebuild a vessel's track. Writes, under
``results/detectors/``:

- ``tuning_closures.csv``: per closure and reason, apparent fishing events,
  vessels, hours, the largest share held by one vessel, and the closure's
  status under the headline rule;
- ``tuning_gaps.csv``: per AIS class, gaps over 6 hours, how many pass
  each check, and the unexplained and long ones;
- ``tuning_loitering.csv``: per duration threshold and AIS class, events,
  vessels and events per 1,000 vessel-days.

The held-out block is not read.
"""

from pathlib import Path

import polars as pl

from vessels.core.loitering import LoiterRules, loiter_events
from vessels.core.cleaning import clean
from vessels.core.segment import segment
from vessels.core.closure_fishing import closure_events
from vessels.core.gaps import gap_events
from vessels.evaluation.inputs import load

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "detectors"
# Decided 2026-10-24 (PLAN.md "Scored closures"): the five headline
# candidates, kept as headline only with at least MIN_VESSELS vessels and
# MIN_EVENTS events on the tuning block; the rule was fixed before any
# held-out read.
HEADLINE = (
    "wgom_hma",
    "eastern_maine_hma",
    "gsc_hma",
    "stellwagen_dhra",
    "jeffreys_bank_hma",
)
MIN_VESSELS = 3
MIN_EVENTS = 5
CHECKS = (
    "covered",
    "away_from_port",
    "away_from_mooring",
    "dense_before",
    "cells_heard",
)


def closure_table(events: pl.DataFrame) -> pl.DataFrame:
    """Apparent fishing per closure and reason, with headline status.

    Args:
        events: ``closure_events`` output.

    Returns:
        pl.DataFrame: One row per closure and reason.
    """
    per_vessel = events.group_by("closure_id", "reason", "vessel_id").len()
    top = per_vessel.group_by("closure_id", "reason").agg(
        top_vessel_events=pl.col("len").max()
    )
    table = (
        events.group_by("closure_id", "reason")
        .agg(
            events=pl.len(),
            vessels=pl.col("vessel_id").n_unique(),
            may=(pl.col("start").dt.month() == 5).sum(),
            september=(pl.col("start").dt.month() == 9).sum(),
            hours=pl.col("hours").sum().round(1),
            median_hours=pl.col("hours").median().round(2),
        )
        .join(top, on=["closure_id", "reason"])
        .with_columns(
            top_vessel_share=(pl.col("top_vessel_events") / pl.col("events")).round(2)
        )
    )
    headline = (
        pl.col("closure_id").is_in(list(HEADLINE))
        & (pl.col("reason") == "closure")
        & (pl.col("vessels") >= MIN_VESSELS)
        & (pl.col("events") >= MIN_EVENTS)
    )
    return (
        table.with_columns(
            status=pl.when(pl.col("reason") != "closure")
            .then(pl.lit("reported separately"))
            .when(headline)
            .then(pl.lit("headline"))
            .otherwise(pl.lit("report only"))
        )
        .drop("top_vessel_events")
        .sort("closure_id", "reason")
    )


def gap_table(gaps: pl.DataFrame) -> pl.DataFrame:
    """Gaps per AIS class and how many pass each check.

    Args:
        gaps: ``gap_events`` output.

    Returns:
        pl.DataFrame: One row per class.
    """
    return (
        gaps.group_by("transceiver")
        .agg(
            pl.len().alias("gaps"),
            *[pl.col(c).sum().alias(c) for c in CHECKS],
            pl.col("unexplained").sum().alias("unexplained"),
            pl.col("long").sum().alias("long"),
            pl.col("vessel_id")
            .filter("unexplained")
            .n_unique()
            .alias("vessels_unexplained"),
            pl.col("hours")
            .filter("unexplained")
            .median()
            .round(1)
            .alias("median_unexplained_hours"),
        )
        .sort("transceiver")
    )


def loiter_table(segmented: pl.DataFrame, inputs) -> pl.DataFrame:
    """Loitering at the headline and sensitivity durations, per class.

    Args:
        segmented: The block, cleaned and segmented.
        inputs: Detector inputs.

    Returns:
        pl.DataFrame: One row per duration and class.
    """
    days = (
        segmented.select("vessel_id", "transceiver", day=pl.col("timestamp").dt.date())
        .unique()
        .group_by("transceiver")
        .agg(vessel_days=pl.len())
    )
    rows = []
    for hours in (8.0, 4.0):
        found = loiter_events(segmented, inputs.shore, LoiterRules(min_hours=hours))
        rows.append(
            found.group_by("transceiver")
            .agg(
                events=pl.len(),
                vessels=pl.col("vessel_id").n_unique(),
                median_hours=pl.col("hours").median().round(1),
                median_shore_nm=pl.col("avg_shore_nm").median().round(1),
            )
            .with_columns(min_hours=pl.lit(hours))
        )
    return (
        pl.concat(rows)
        .join(days, on="transceiver")
        .with_columns(
            per_1000_vessel_days=(
                1000 * pl.col("events") / pl.col("vessel_days")
            ).round(1)
        )
        .select(
            "min_hours",
            "transceiver",
            "events",
            "vessels",
            "vessel_days",
            "per_1000_vessel_days",
            "median_hours",
            "median_shore_nm",
        )
        .sort("min_hours", "transceiver", descending=[True, False])
    )


def main():
    """Write the detector summaries for the tuning block."""
    positions, inputs = load("tuning")
    segmented = segment(clean(positions).frame)
    closures = closure_table(closure_events(segmented, inputs.closures))
    gaps = gap_table(
        gap_events(segmented, inputs.grid, inputs.ports, inputs.heard, inputs.gap_rules)
    )
    loitering = loiter_table(segmented, inputs)
    OUT.mkdir(parents=True, exist_ok=True)
    closures.write_csv(OUT / "tuning_closures.csv")
    gaps.write_csv(OUT / "tuning_gaps.csv")
    loitering.write_csv(OUT / "tuning_loitering.csv")
    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=200):
        print(closures)
        print(gaps)
        print(loitering)


if __name__ == "__main__":
    main()
