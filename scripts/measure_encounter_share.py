"""Measure how many encounters the committed extract cannot see.

The extract holds one vessel group's fishing vessels per block, so an
encounter is visible in it only when both vessels are group-A fishing
vessels. This script runs the encounter detector on every vessel in the
raw tuning days and counts the encounters that involve a group-A fishing
vessel, by who the partner is: another group-A fishing vessel (visible), a
group-B fishing vessel, or a non-fishing vessel by AIS type.

Owner only: it needs the raw MarineCadastre days (``scripts/fetch_us.py
--blocks``) and the salt (``data/mmsi.salt``) to tell the groups apart.
Run it with ``PYTHONPATH=src python scripts/measure_encounter_share.py``.
Pairs without a group-A fishing vessel (group B among themselves, or two
non-fishing vessels) are dropped as soon as the adjacency search finds
them and never reach a run or a count, so the held-out fleet is seen only
as partners of tuning vessels, and only on tuning days. Writes counts
only, never event rows, to ``results/encounters/tuning_share.csv``.
"""

import argparse
from pathlib import Path

import polars as pl

from build_us_extract import RAW, SALT, load_raw_day
from vessels.core.cleaning import clean, mmsi_rule
from vessels.core.encounters import EncounterRules, adjacent, merge, raw_encounters
from vessels.core.encounters import resample
from vessels.core.segment import segment
from vessels.evaluation.inputs import load_layers
from vessels.io.blocks import BLOCKS
from vessels.io.marinecadastre import EXTRACT_SCHEMA, FISHING_TYPES, pseudonymise

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "encounters"
# AIS ship-type groups for non-fishing partners (ITU-R M.1371 type codes).
TYPE_GROUPS = (
    ("tug or towing", (31, 32, 52)),
    ("pleasure or sailing", (36, 37)),
    ("passenger", tuple(range(60, 70))),
    ("cargo", tuple(range(70, 80))),
    ("tanker", tuple(range(80, 90))),
)


def vessel_classes(frame: pl.DataFrame) -> pl.DataFrame:
    """Each vessel's class: its fishing group, or its AIS type group.

    A vessel counts as fishing if any of its positions carries a fishing
    type, as the extract keeps it; otherwise its most common type decides.

    Args:
        frame: ``pseudonymise`` output.

    Returns:
        pl.DataFrame: ``vessel_id`` and ``class``.
    """
    fishing = pl.col("vessel_type").is_in(FISHING_TYPES)
    common = pl.col("vessel_type").drop_nulls().mode().sort().first()
    per = frame.group_by("vessel_id").agg(
        fishing=fishing.any(), group=pl.col("group").first(), common=common
    )
    kind = pl.lit("other or unknown")
    for name, codes in reversed(TYPE_GROUPS):
        kind = pl.when(pl.col("common").is_in(codes)).then(pl.lit(name)).otherwise(kind)
    return per.select(
        "vessel_id",
        pl.when("fishing")
        .then(pl.concat_str(pl.lit("fishing "), "group"))
        .otherwise(kind)
        .alias("class"),
    )


def share_table(merged: pl.DataFrame, classes: pl.DataFrame) -> pl.DataFrame:
    """Encounters with a group-A fishing vessel, by partner class.

    Args:
        merged: ``merge`` output for pairs with a group-A fishing vessel.
        classes: ``vessel_classes`` output.

    Returns:
        pl.DataFrame: Per partner class: passing encounters, their hours
        and group-A vessels, and every offshore candidate (merged, passing
        or not).
    """
    side = classes.rename({"vessel_id": "id"})
    found = merged.join(
        side.rename({"id": "vessel_id", "class": "class_a"}), on="vessel_id"
    ).join(side.rename({"id": "partner_id", "class": "class_b"}), on="partner_id")
    tuning = "fishing A"
    found = found.with_columns(
        partner=pl.when(pl.col("class_a") == tuning)
        .then("class_b")
        .otherwise("class_a"),
        tuning_vessel=pl.when(pl.col("class_a") == tuning)
        .then("vessel_id")
        .otherwise("partner_id"),
    )
    table = found.group_by("partner").agg(
        encounters=pl.col("passes").sum(),
        hours=pl.col("hours").filter("passes").sum().round(1),
        tuning_vessels=pl.col("tuning_vessel").filter("passes").n_unique(),
        offshore_candidates=pl.len(),
    )
    total = table["encounters"].sum()
    return table.with_columns(
        share=(pl.col("encounters") / total).round(3) if total else pl.lit(None)
    ).sort("encounters", "partner", descending=[True, False])


def main():
    """Count the tuning block's encounters by partner class."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--salt", type=Path, default=SALT)
    args = parser.parse_args()
    salt = args.salt.read_text(encoding="utf-8").strip().encode()
    days = [
        load_raw_day(RAW / f"ais-{day.isoformat()}.parquet") for day in BLOCKS["tuning"]
    ]
    raw = pl.concat(days)
    frame = pseudonymise(raw.filter(mmsi_rule(pl.col("mmsi")).is_null()), salt)
    classes = vessel_classes(frame)
    positions = frame.select(
        pl.col(name).cast(dtype) for name, dtype in EXTRACT_SCHEMA.items()
    )
    segmented = segment(clean(positions).frame)
    rules = EncounterRules()
    ports, shore = load_layers()
    pairs = adjacent(resample(segmented, rules), ports.projection, rules)
    tuning = classes.filter(pl.col("class") == "fishing A")["vessel_id"].implode()
    pairs = pairs.filter(
        pl.col("vessel_id_a").is_in(tuning) | pl.col("vessel_id_b").is_in(tuning)
    )
    runs = raw_encounters(pairs, ports, shore, rules)
    merged = merge(runs.filter(pl.col("near_port").not_()), rules)
    table = share_table(merged, classes)
    OUT.mkdir(parents=True, exist_ok=True)
    table.write_csv(OUT / "tuning_share.csv")
    print(f"positions {positions.height:,}, vessels {classes.height:,}")
    print(f"raw runs {runs.height:,}, near port {int(runs['near_port'].sum()):,}")
    with pl.Config(tbl_rows=-1, tbl_cols=-1):
        print(table)


if __name__ == "__main__":
    main()
