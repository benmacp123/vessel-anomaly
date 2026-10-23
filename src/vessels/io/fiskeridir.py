"""Readers for Fiskeridirektoratet open data (VMS and ERS).

The VMS archive is one semicolon-separated CSV of about 3.75 GB a year,
too large to extract or load whole, so the reader streams it out of the
zip and keeps only the lines for the requested dates before parsing.
"""

import io
import zipfile
from datetime import date
from pathlib import Path

import polars as pl

# Source column, internal name and type, in file order. Every column is
# typed explicitly so a year whose first rows are blank cannot change the
# inferred schema.
VMS_COLUMNS = {
    "MeldingID": ("message_id", pl.Int64),
    "Meldingstype(kode)": ("message_type", pl.String),
    "Meldingstype": ("message_type_name", pl.String),
    "Tidspunkt(UTC)": ("timestamp", pl.String),
    "Breddegrad": ("lat", pl.Float64),
    "Lengdegrad": ("lon", pl.Float64),
    "Kurs": ("course", pl.Int64),
    # Documented as tenths of a knot, but the 2024 values are knots
    # (median 0, 99th percentile 12.4, maximum 31 in 27 May-2 June):
    # observed, not converted.
    "Fart": ("speed_kn", pl.Float64),
    "Radiokallesignal": ("call_sign", pl.String),
    "Registreringsmerke": ("registration", pl.String),
    "Fartøynavn": ("vessel_name", pl.String),
    "Størstelengde": ("length_m", pl.Float64),
    "Bruttotonnasje": ("gross_tonnage", pl.Int64),
    "Motorkraft": ("engine_kw", pl.Int64),
    "Fartøytype": ("vessel_type", pl.String),
}

# Timestamps look like 01-JAN-24 15.40.00.000000000 (UTC). The month is
# matched by hand so line filtering does not depend on the locale.
_MONTHS = "JAN FEB MAR APR MAY JUN JUL AUG SEP OCT NOV DEC".split()
_TIMESTAMP_FORMAT = "%d-%b-%y %H.%M.%S%.f"


def vms_date_token(day: date) -> bytes:
    """The byte pattern that marks a VMS line as belonging to one day.

    The timestamp is the only unquoted field that starts with a date, so
    the leading separator and trailing space make the match unambiguous.

    Args:
        day: Calendar day (UTC).

    Returns:
        bytes: For example ``b";27-MAY-24 "``.
    """
    return f";{day.day:02d}-{_MONTHS[day.month - 1]}-{day.year % 100:02d} ".encode()


def read_vms_zip(path: Path, days: list[date]) -> pl.DataFrame:
    """Read the VMS rows for some days from a yearly archive.

    Args:
        path: The yearly zip, holding a single CSV.
        days: UTC days to keep.

    Returns:
        pl.DataFrame: One row per position report, columns renamed as in
        ``VMS_COLUMNS`` and ``timestamp`` parsed to a UTC datetime. Rows
        are kept as published: duplicates are not removed.

    Raises:
        ValueError: If the archive does not hold exactly one file, or its
            header differs from ``VMS_COLUMNS``.
    """
    tokens = [vms_date_token(day) for day in days]
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != 1:
            raise ValueError(f"{path}: expected one CSV, found {names}")
        with archive.open(names[0]) as handle:
            header = handle.readline()
            lines = [line for line in handle if any(t in line for t in tokens)]
    columns = [c.strip('"') for c in header.decode("utf-8").strip().split(";")]
    if columns != list(VMS_COLUMNS):
        raise ValueError(f"{path}: unexpected VMS header {columns}")
    frame = pl.read_csv(
        io.BytesIO(header + b"".join(lines)),
        separator=";",
        schema={source: dtype for source, (_, dtype) in VMS_COLUMNS.items()},
    )
    return frame.rename(
        {source: name for source, (name, _) in VMS_COLUMNS.items()}
    ).with_columns(
        pl.col("timestamp").str.to_datetime(_TIMESTAMP_FORMAT, time_zone="UTC")
    )
