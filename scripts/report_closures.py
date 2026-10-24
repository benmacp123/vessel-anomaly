"""Count the tuning block's apparent towing inside closures, from the
committed geofence table and coverage grid.

Run ``make closures`` (after ``make coverage``). Towing is a fishing-type
position at 2.5-5.5 kn (de Souza et al. 2016 trawl band); AIS has no gear,
so every towing position is assumed to be mobile bottom-tending gear and
exemptions for any such gear lift the ban. No persistence rule is applied
here: that belongs to the closure detector. Writes, under
``results/closures/``:

- ``tuning_summary.csv``: per closure, reason and month, the towing
  positions, how many fall in cells covered for their AIS class, and the
  vessels;
- ``tuning_cells.csv``: per closure and AIS class, the H3 cells the
  closure overlaps and the share covered (a cell with no reception counts
  as not covered).
"""

from pathlib import Path

import h3
import polars as pl
import shapely
from shapely.geometry import Polygon

from vessels.core.geofence import closure_hits
from vessels.io.marinecadastre import read_extract
from vessels.io.reference import read_closures

ROOT = Path(__file__).resolve().parent.parent
TUNING = ROOT / "data" / "extract" / "us" / "tuning"
GRID = ROOT / "results" / "coverage" / "tuning_coverage.parquet"
OUT = ROOT / "results" / "closures"
TOWING_KN = (2.5, 5.5)


def towing_hits(rules, grid: pl.DataFrame) -> pl.DataFrame:
    """Find towing positions inside closures and mark their coverage.

    Args:
        rules: The geofence table.
        grid: The coverage grid (``h3_cell``, ``transceiver``,
            ``covered``).

    Returns:
        pl.DataFrame: ``closure_hits`` output plus ``month`` and
        ``covered``.
    """
    towing = (
        read_extract(sorted(TUNING.glob("*.parquet")))
        .filter(pl.col("sog").is_between(*TOWING_KN))
        .select("vessel_id", "timestamp", "lon", "lat", "transceiver")
        .collect()
    )
    hits = closure_hits(towing, rules)
    resolution = h3.get_resolution(grid["h3_cell"][0])
    cells = [
        h3.latlng_to_cell(lat, lon, resolution)
        for lat, lon in zip(hits["lat"], hits["lon"])
    ]
    return (
        hits.with_columns(
            h3_cell=pl.Series(cells, dtype=pl.String),
            month=pl.col("timestamp").dt.month(),
        )
        .join(
            grid.select("h3_cell", "transceiver", "covered"),
            on=["h3_cell", "transceiver"],
            how="left",
        )
        .with_columns(pl.col("covered").fill_null(False))
    )


def summary_table(hits: pl.DataFrame) -> pl.DataFrame:
    """Towing positions and vessels per closure, reason and month.

    Args:
        hits: ``towing_hits`` output.

    Returns:
        pl.DataFrame: One row per closure, reason and month.
    """
    return (
        hits.group_by("closure_id", "reason", "month")
        .agg(
            towing_positions=pl.len(),
            covered_positions=pl.col("covered").sum(),
            vessels=pl.col("vessel_id").n_unique(),
            covered_vessels=pl.col("vessel_id").filter("covered").n_unique(),
        )
        .with_columns(
            covered_share=(
                pl.col("covered_positions") / pl.col("towing_positions")
            ).round(3)
        )
        .sort("closure_id", "reason", "month")
    )


def cell_table(rules, grid: pl.DataFrame) -> pl.DataFrame:
    """The share of each closure's overlapping cells that is covered.

    Args:
        rules: The geofence table.
        grid: The coverage grid.

    Returns:
        pl.DataFrame: ``closure_id``, ``transceiver``, ``cells`` and
        ``covered_share``.
    """
    cells = grid["h3_cell"].unique().sort().to_list()
    shapes = [
        Polygon([(lon, lat) for lat, lon in h3.cell_to_boundary(c)]) for c in cells
    ]
    tree = shapely.STRtree(shapes)
    rows = []
    for rule in rules:
        if rule.kind != "closure":
            continue
        for index in tree.query(rule.geometry, predicate="intersects"):
            rows.append({"closure_id": rule.area_id, "h3_cell": cells[index]})
    overlap = pl.DataFrame(rows)
    classes = grid.select("transceiver").unique()
    return (
        overlap.join(classes, how="cross")
        .join(
            grid.select("h3_cell", "transceiver", "covered"),
            on=["h3_cell", "transceiver"],
            how="left",
        )
        .group_by("closure_id", "transceiver")
        .agg(
            cells=pl.len(),
            covered_share=pl.col("covered").fill_null(False).mean().round(3),
        )
        .sort("closure_id", "transceiver")
    )


def main():
    """Write the closure summaries for the tuning block."""
    rules = read_closures()
    grid = pl.read_parquet(GRID)
    summary = summary_table(towing_hits(rules, grid))
    cells = cell_table(rules, grid)
    OUT.mkdir(parents=True, exist_ok=True)
    summary.write_csv(OUT / "tuning_summary.csv")
    cells.write_csv(OUT / "tuning_cells.csv")
    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=200):
        print(summary)
        print(cells)


if __name__ == "__main__":
    main()
