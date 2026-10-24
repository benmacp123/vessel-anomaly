"""The committed US extract: pseudonymised, column-trimmed fishing-vessel
positions from MarineCadastre 2024, one zstd Parquet file per day.

Raw MarineCadastre files carry a Parquet GEOMETRY column that polars 2.0
cannot read, so the extract builder script reads them with DuckDB and
hands this module plain columns. Everything after that (pseudonymising,
selecting a block's vessels, the extract schema) lives here so it is
tested once and used by both the builder and the pipeline.
"""

from pathlib import Path

import polars as pl

from vessels.io.blocks import vessel_group, vessel_id

FISHING_TYPES = (30, 1001, 1002)
# Column name and type in the committed extract, in file order. Names,
# IMO numbers and call signs are dropped; MMSI becomes ``vessel_id``.
EXTRACT_SCHEMA = {
    "vessel_id": pl.String,
    "timestamp": pl.Datetime("us", "UTC"),
    "lon": pl.Float64,
    "lat": pl.Float64,
    "sog": pl.Float32,
    "cog": pl.Float32,
    "heading": pl.Int32,
    "status": pl.Int32,
    "vessel_type": pl.Int32,
    "length": pl.Float32,
    "transceiver": pl.String,
}
# The daily reception grid from all traffic, in file order (see
# ``scripts/build_us_extract.py``). Each file holds one day.
RECEPTION_SCHEMA = {
    "h3_cell": pl.String,
    "transceiver": pl.String,
    "positions": pl.Int64,
    "vessels": pl.Int64,
    "moving_intervals": pl.Int64,
    "long_intervals": pl.Int64,
}


def pseudonymise(frame: pl.DataFrame, salt: bytes) -> pl.DataFrame:
    """Replace ``mmsi`` with ``vessel_id`` and add the vessel ``group``.

    Args:
        frame: Positions with an integer ``mmsi`` column.
        salt: The secret key, kept out of git.

    Returns:
        pl.DataFrame: The same rows without ``mmsi``, with ``vessel_id``
        first and ``group`` (``A`` or ``B``) last.
    """
    ids = pl.DataFrame({"mmsi": frame["mmsi"].unique()}).with_columns(
        vessel_id=pl.col("mmsi").map_elements(
            lambda m: vessel_id(m, salt), return_dtype=pl.String
        )
    )
    ids = ids.with_columns(
        group=pl.col("vessel_id").map_elements(vessel_group, return_dtype=pl.String)
    )
    joined = frame.join(ids, on="mmsi", how="left")
    return joined.select("vessel_id", pl.exclude("vessel_id", "mmsi"))


def fishing_extract(frame: pl.DataFrame, group: str) -> pl.DataFrame:
    """One block's fishing-vessel rows in the extract schema and order.

    Rows are sorted on every column, not just vessel and time, so that
    reports sharing a timestamp land in the same order on every rebuild
    and the file is byte for byte reproducible.

    Args:
        frame: ``pseudonymise`` output carrying the extract columns.
        group: The vessel group the block keeps.

    Returns:
        pl.DataFrame: Fishing-type rows of that group, cast to
        ``EXTRACT_SCHEMA``.
    """
    return (
        frame.filter(
            pl.col("vessel_type").is_in(FISHING_TYPES), pl.col("group") == group
        )
        .select(pl.col(name).cast(dtype) for name, dtype in EXTRACT_SCHEMA.items())
        .sort(list(EXTRACT_SCHEMA), nulls_last=True)
    )


def read_extract(paths: list[Path]) -> pl.LazyFrame:
    """Scan committed extract files.

    Args:
        paths: Daily extract files.

    Returns:
        pl.LazyFrame: Rows in the extract schema.

    Raises:
        ValueError: If a file's schema differs from ``EXTRACT_SCHEMA``.
    """
    for path in paths:
        schema = dict(pl.read_parquet_schema(path))
        if schema != EXTRACT_SCHEMA:
            raise ValueError(f"{path}: unexpected extract schema {schema}")
    return pl.scan_parquet(paths)


def read_reception(paths: list[Path]) -> pl.LazyFrame:
    """Scan committed daily reception grid files.

    Args:
        paths: Daily ``reception/<day>.parquet`` files.

    Returns:
        pl.LazyFrame: Rows in the reception schema, one per cell, class
        and day.

    Raises:
        ValueError: If a file's schema differs from ``RECEPTION_SCHEMA``.
    """
    for path in paths:
        schema = dict(pl.read_parquet_schema(path))
        if schema != RECEPTION_SCHEMA:
            raise ValueError(f"{path}: unexpected reception schema {schema}")
    return pl.scan_parquet(paths)
