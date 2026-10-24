"""Find US anchorages in the tuning block and report distance to shore and
port.

Runs from committed files only: ``make distance``. Anchorages come from
the tuning block alone and are then frozen for the held-out block. Writes:

- ``data/reference/us/anchorages.csv``: the anchorage cells (see
  ``vessels.core.distance``), and ``SHA256SUMS`` for every layer;
- under ``results/distance/``:

  - ``tuning_candidates.csv``: every cell with enough stationary vessels,
    including those rejected as too far offshore;
  - ``tuning_summary.csv``: the share of positions per AIS class in each
    band of distance to shore and to port;
  - ``tuning_distance.png``: the 20 nm and 50 nm contours and the
    anchorages, for checking by eye.
"""

from pathlib import Path

import numpy as np
import polars as pl

from vessels.core.distance import (
    MAX_SHORE_KM,
    MIN_HOURS,
    MIN_VESSELS,
    RADIUS_KM,
    PortDistance,
    ShoreDistance,
    anchorage_cells,
    stationary_runs,
)
from vessels.core.geo import KM_PER_NM, LocalProjection
from vessels.evaluation.maps import plot_distance
from vessels.io.blocks import BOX
from vessels.io.marinecadastre import read_extract
from vessels.io.reference import ANCHORAGES_US, read_land, write_sums

ROOT = Path(__file__).resolve().parent.parent
TUNING = ROOT / "data" / "extract" / "us" / "tuning"
OUT = ROOT / "results" / "distance"
# Thresholds the detectors use: loitering at 20 nm (Miller et al. 2018),
# GFW's intentional-gap rule at 50 nm, the encounter port mask at 10 km.
SHORE_BANDS_KM = (0.0, MAX_SHORE_KM, 20 * KM_PER_NM, 50 * KM_PER_NM)
SHORE_LABELS = ("on land", "0-5 km", "5 km-20 nm", "20-50 nm", "50 nm or more")
PORT_BANDS_KM = (3.0, 10.0)
PORT_LABELS = ("under 3 km", "3-10 km", "10 km or more")
# Grid step for the contour map, degrees; display only.
GRID_STEP = 0.02


def band(values: pl.Series, edges: tuple, labels: tuple) -> pl.Series:
    """Label each value with the band it falls in.

    Args:
        values: Distances, kilometres.
        edges: Band edges, ascending; a value on an edge goes up.
        labels: One more label than edges.

    Returns:
        pl.Series: Band labels.
    """
    index = np.searchsorted(np.asarray(edges), values.to_numpy(), side="right")
    return pl.Series(np.asarray(labels)[index], dtype=pl.String)


def summary_table(positions: pl.DataFrame) -> pl.DataFrame:
    """Share of each class's positions per shore band and per port band.

    Args:
        positions: ``transceiver``, ``shore_km``, ``port_km``.

    Returns:
        pl.DataFrame: ``measure``, ``band``, ``transceiver``,
        ``positions`` and ``share`` of that class's positions.
    """
    banded = positions.with_columns(
        shore=band(positions["shore_km"], SHORE_BANDS_KM, SHORE_LABELS),
        port=band(positions["port_km"], PORT_BANDS_KM, PORT_LABELS),
    )
    order = {label: i for i, label in enumerate(SHORE_LABELS + PORT_LABELS)}
    parts = [
        banded.group_by("transceiver", column)
        .agg(positions=pl.len())
        .rename({column: "band"})
        .with_columns(measure=pl.lit(column))
        for column in ("shore", "port")
    ]
    return (
        pl.concat(parts)
        .with_columns(
            share=(
                pl.col("positions")
                / pl.col("positions").sum().over("measure", "transceiver")
            ).round(4),
            order=pl.col("band").replace_strict(order),
        )
        .sort("measure", "transceiver", "order", descending=[True, False, False])
        .select("measure", "band", "transceiver", "positions", "share")
    )


def shore_grid(shore: ShoreDistance) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Distance to shore on a regular grid over the box, for the map.

    Args:
        shore: Distance to shore.

    Returns:
        tuple: Longitudes, latitudes and distances (latitude by
        longitude), kilometres.
    """
    min_lon, min_lat, max_lon, max_lat = BOX
    lon = np.arange(min_lon, max_lon + GRID_STEP / 2, GRID_STEP)
    lat = np.arange(min_lat, max_lat + GRID_STEP / 2, GRID_STEP)
    grid_lon, grid_lat = np.meshgrid(lon, lat)
    km = shore.km(grid_lon.ravel(), grid_lat.ravel()).reshape(grid_lon.shape)
    return lon, lat, km


def main():
    """Write the anchorages, their checksums and the distance report."""
    projection = LocalProjection((BOX[0] + BOX[2]) / 2, (BOX[1] + BOX[3]) / 2)
    land = read_land()
    shore = ShoreDistance(land, projection)
    positions = (
        read_extract(sorted(TUNING.glob("*.parquet")))
        .select("vessel_id", "timestamp", "lon", "lat", "transceiver")
        .collect()
    )
    runs = stationary_runs(positions, projection)
    cells = anchorage_cells(runs, shore)
    anchorages = cells.filter(pl.col("anchorage")).drop("anchorage")
    anchorages.with_columns(
        pl.col("lon", "lat").round(6), pl.col("shore_km").round(3)
    ).write_csv(ANCHORAGES_US)
    write_sums()

    ports = PortDistance(anchorages["lon"], anchorages["lat"], projection)
    lon, lat = positions["lon"].to_numpy(), positions["lat"].to_numpy()
    measured = positions.with_columns(
        shore_km=pl.Series(shore.km(lon, lat)),
        port_km=pl.Series(ports.km(lon, lat)),
    )
    summary = summary_table(measured)
    OUT.mkdir(parents=True, exist_ok=True)
    cells.with_columns(
        pl.col("lon", "lat").round(6), pl.col("shore_km").round(3)
    ).write_csv(OUT / "tuning_candidates.csv")
    summary.write_csv(OUT / "tuning_summary.csv")
    plot_distance(
        shore_grid(shore),
        {"20 nm": 20 * KM_PER_NM, "50 nm": 50 * KM_PER_NM},
        cells,
        land,
        BOX,
        OUT / "tuning_distance.png",
        "Distance to shore and anchorages, tuning block (28 days, May and "
        "September 2024)",
        f"Anchorage: res-8 cell, {MIN_VESSELS}+ vessels within {RADIUS_KM:g} km "
        f"for {MIN_HOURS:g} h+, centre {MAX_SHORE_KM:g} km or less from land. "
        "Contours: distance to Natural Earth 10m land.",
    )
    print(f"{runs.height:,} stationary runs, {runs['vessel_id'].n_unique()} vessels")
    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=200):
        print(cells.filter(~pl.col("anchorage")))
        print(summary)
    print(f"{ANCHORAGES_US}: {anchorages.height} anchorages")


if __name__ == "__main__":
    main()
