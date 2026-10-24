"""Build the committed US extract from the raw MarineCadastre days.

Owner only: it needs the salt (``data/mmsi.salt``, outside git), and the
raw days fetched by ``scripts/fetch_us.py --blocks``. Reviewers use the
committed output. Run it with
``PYTHONPATH=src python scripts/build_us_extract.py``.

For each block day it writes, under ``data/extract/us/<block>/``:

- ``<day>.parquet``: that block's vessel group's fishing-type positions in
  the box, pseudonymised and trimmed to the extract schema;
- ``reception/<day>.parquet``: positions and distinct vessels from all
  traffic per H3 resolution-6 cell and AIS class, for the coverage mask.

Then it writes ``SHA256SUMS`` and ``counts.csv`` (rows and vessels per
file) under ``data/extract/us/``.
"""

import argparse
import csv
from pathlib import Path

import duckdb
import h3
import polars as pl

from vessels.io.blocks import BLOCK_GROUP, BLOCKS
from vessels.io.fetch import sha256_of
from vessels.io.marinecadastre import fishing_extract, pseudonymise

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "marinecadastre"
OUT = ROOT / "data" / "extract" / "us"
SALT = ROOT / "data" / "mmsi.salt"
# Gulf of Maine and Georges Bank.
BOX = (-71.5, 40.0, -65.0, 45.0)
H3_RESOLUTION = 6


def load_raw_day(path: Path) -> pl.DataFrame:
    """Read every position in the box from one raw MarineCadastre day.

    DuckDB 2.0 without the spatial extension can print the geometry as
    WKT but not take coordinates from it, so they are parsed from the text.

    Args:
        path: A raw daily GeoParquet file.

    Returns:
        pl.DataFrame: ``mmsi``, ``timestamp`` (UTC), ``lon``, ``lat`` and
        the other extract columns, all vessel types.
    """
    min_lon, min_lat, max_lon, max_lat = BOX
    sql = f"""
        with points as (
            select mmsi, base_date_time as timestamp, sog, cog, heading,
                   status, vessel_type, length, transceiver,
                   regexp_extract(st_astext(geometry),
                       'POINT \\(([-0-9.e]+) ([-0-9.e]+)\\)', ['lon', 'lat'])
                       as xy
            from read_parquet('{path}')
        )
        select * exclude (xy), xy.lon::double as lon, xy.lat::double as lat
        from points
        where xy.lon::double between {min_lon} and {max_lon}
          and xy.lat::double between {min_lat} and {max_lat}
    """
    return pl.DataFrame(duckdb.sql(sql)).with_columns(
        pl.col("mmsi").cast(pl.Int64),
        pl.col("timestamp").dt.replace_time_zone("UTC"),
    )


def reception_grid(frame: pl.DataFrame) -> pl.DataFrame:
    """Count all-traffic positions and vessels per H3 cell and AIS class.

    Args:
        frame: ``pseudonymise`` output for every vessel type.

    Returns:
        pl.DataFrame: ``h3_cell`` (hex string), ``transceiver``,
        ``positions`` and ``vessels``, sorted by cell and class.
    """
    cells = [
        h3.latlng_to_cell(lat, lon, H3_RESOLUTION)
        for lat, lon in zip(frame["lat"], frame["lon"])
    ]
    return (
        frame.with_columns(h3_cell=pl.Series(cells, dtype=pl.String))
        .group_by("h3_cell", "transceiver")
        .agg(
            positions=pl.len().cast(pl.Int64),
            vessels=pl.col("vessel_id").n_unique().cast(pl.Int64),
        )
        .sort("h3_cell", "transceiver", nulls_last=True)
    )


def write_sums(out: Path) -> None:
    """Write ``SHA256SUMS`` for every Parquet file under the extract.

    Args:
        out: The extract directory.
    """
    lines = [
        f"{sha256_of(path)}  {path.relative_to(out).as_posix()}"
        for path in sorted(out.rglob("*.parquet"))
    ]
    (out / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    """Build every block day whose raw file is present."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--salt", type=Path, default=SALT)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    salt = args.salt.read_text(encoding="utf-8").strip().encode()
    counts = []
    for block, days in BLOCKS.items():
        for day in days:
            raw = RAW / f"ais-{day.isoformat()}.parquet"
            if not raw.exists():
                print(f"{block} {day}: raw file missing; skipped")
                continue
            frame = pseudonymise(load_raw_day(raw), salt)
            fishing = fishing_extract(frame, BLOCK_GROUP[block])
            grid = reception_grid(frame)
            (args.out / block / "reception").mkdir(parents=True, exist_ok=True)
            fishing.write_parquet(args.out / block / f"{day}.parquet")
            grid.write_parquet(args.out / block / "reception" / f"{day}.parquet")
            counts.append(
                {
                    "block": block,
                    "day": day.isoformat(),
                    "rows": fishing.height,
                    "vessels": fishing["vessel_id"].n_unique(),
                    "grid_cells": grid["h3_cell"].n_unique(),
                    "all_traffic_rows": frame.height,
                }
            )
            print(", ".join(f"{k} {v}" for k, v in counts[-1].items()))
    with (args.out / "counts.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(counts[0]))
        writer.writeheader()
        writer.writerows(counts)
    write_sums(args.out)


if __name__ == "__main__":
    main()
