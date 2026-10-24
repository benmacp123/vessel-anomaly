"""Build the US coverage grid from the tuning block's reception counts.

Runs from committed files only: ``make coverage``. The baseline comes
from the tuning block alone; the held-out reception files stay unread
until scoring. Writes, under ``results/coverage/``:

- ``tuning_coverage.parquet``: per H3 resolution-6 cell and AIS class,
  the tuning totals, ``covered``, ``status`` and ``edge_km`` (see
  ``vessels.core.coverage``);
- ``tuning_summary.csv``: cells, moving intervals and the tuning block's
  fishing positions per class and status;
- ``tuning_coverage.png``: the map, for checking by eye.
"""

from pathlib import Path

import h3
import polars as pl

from vessels.core.coverage import MAX_LONG_SHARE, coverage, summarise
from vessels.evaluation.maps import plot_coverage
from vessels.io.blocks import BOX
from vessels.io.marinecadastre import read_extract, read_reception
from vessels.io.reference import read_land

ROOT = Path(__file__).resolve().parent.parent
TUNING = ROOT / "data" / "extract" / "us" / "tuning"
OUT = ROOT / "results" / "coverage"


def fishing_by_cell(resolution: int) -> pl.DataFrame:
    """Count the tuning block's fishing positions per cell and class.

    Args:
        resolution: H3 resolution of the coverage grid.

    Returns:
        pl.DataFrame: ``h3_cell``, ``transceiver`` and
        ``fishing_positions``.
    """
    fishing = (
        read_extract(sorted(TUNING.glob("*.parquet")))
        .select("lat", "lon", "transceiver")
        .collect()
    )
    cells = [
        h3.latlng_to_cell(lat, lon, resolution)
        for lat, lon in zip(fishing["lat"], fishing["lon"])
    ]
    return (
        fishing.with_columns(h3_cell=pl.Series(cells, dtype=pl.String))
        .group_by("h3_cell", "transceiver")
        .agg(fishing_positions=pl.len().cast(pl.Int64))
    )


def summary_table(result: pl.DataFrame, fishing: pl.DataFrame) -> pl.DataFrame:
    """Cells, intervals and fishing positions per class and status.

    Fishing positions in cells with no reception row at all (none, since
    the grid counts all traffic) would show as status ``no reception``.

    Args:
        result: ``coverage`` output.
        fishing: ``fishing_by_cell`` output.

    Returns:
        pl.DataFrame: One row per class and status, with each count's
        share of its class total.
    """
    joined = result.join(
        fishing, on=["h3_cell", "transceiver"], how="full", coalesce=True
    ).with_columns(
        pl.col("status").fill_null("no reception"),
        pl.col("fishing_positions").fill_null(0),
        pl.col("moving_intervals").fill_null(0),
    )
    table = (
        joined.group_by("transceiver", "status")
        .agg(
            cells=pl.len(),
            moving_intervals=pl.col("moving_intervals").sum(),
            fishing_positions=pl.col("fishing_positions").sum(),
        )
        .sort("transceiver", "status")
    )
    return table.with_columns(
        (pl.col(c) / pl.col(c).sum().over("transceiver")).round(4).alias(f"{c}_share")
        for c in ("cells", "moving_intervals", "fishing_positions")
    )


def main():
    """Write the coverage grid, its summary and its map."""
    grid = read_reception(sorted((TUNING / "reception").glob("*.parquet"))).collect()
    result = coverage(summarise(grid), BOX)
    resolution = h3.get_resolution(result["h3_cell"][0])
    table = summary_table(result, fishing_by_cell(resolution))
    OUT.mkdir(parents=True, exist_ok=True)
    result.write_parquet(OUT / "tuning_coverage.parquet")
    table.write_csv(OUT / "tuning_summary.csv")
    plot_coverage(
        result,
        read_land(),
        BOX,
        OUT / "tuning_coverage.png",
        "Receiver coverage, tuning block (28 days, May and September 2024)",
        f"Covered: 95% Wilson upper bound of the share of moving intervals over "
        f"10 min at most {MAX_LONG_SHARE:g}. White: no reception. Edge distance "
        "runs to the nearest uncovered cell of the same class.",
    )
    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=200):
        print(table)


if __name__ == "__main__":
    main()
