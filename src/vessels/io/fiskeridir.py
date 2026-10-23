"""Readers for Fiskeridirektoratet open data (VMS, ERS and coral-reef
closures).

The VMS archive is one semicolon-separated CSV of about 3.75 GB a year,
too large to extract or load whole, so the reader streams it out of the
zip and keeps only the lines for the requested dates before parsing. The
ERS catch table (DCA) is about 309 MB and is read whole.
"""

import io
import xml.etree.ElementTree as ET
import zipfile
from datetime import date, datetime
from pathlib import Path

import polars as pl
from shapely.geometry import Polygon

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


# DCA source column, internal name and type. The file has 114 columns;
# these are the ones the haul analysis reads. Numbers use a decimal comma,
# so every column is read as text and cast after the comma is replaced.
DCA_COLUMNS = {
    "Melding ID": ("message_id", pl.Int64),
    "Meldingsversjon": ("message_version", pl.Int64),
    "Aktivitet (kode)": ("activity", pl.String),
    "Radiokallesignal (ERS)": ("call_sign", pl.String),
    "Starttidspunkt": ("start", pl.String),
    "Stopptidspunkt": ("stop", pl.String),
    "Startposisjon bredde": ("start_lat", pl.Float64),
    "Startposisjon lengde": ("start_lon", pl.Float64),
    "Stopposisjon bredde": ("stop_lat", pl.Float64),
    "Stopposisjon lengde": ("stop_lon", pl.Float64),
    "Varighet": ("duration_min", pl.Int64),
    "Redskap FAO (kode)": ("gear_fao", pl.String),
    "Art FAO (kode)": ("species_fao", pl.String),
    "Rundvekt": ("round_weight_kg", pl.Float64),
}
_DCA_TIME_FORMAT = "%d.%m.%Y %H:%M:%S"


def read_dca_zip(path: Path, start: datetime, end: datetime) -> pl.DataFrame:
    """Read the ERS catch and activity rows (DCA) that start in a period.

    The header names no time zone. Times are parsed as UTC, which the VMS
    file states for its own times; for ERS it is provisional until checked
    against AIS.

    Args:
        path: The yearly ERS zip, holding one ``...-dca.csv`` among others.
        start: First instant to keep (UTC, inclusive).
        end: Last instant to keep (UTC, exclusive).

    Returns:
        pl.DataFrame: One row per published DCA line (one per species and
        haul, so a haul repeats), columns renamed as in ``DCA_COLUMNS`` and
        ``start``/``stop`` parsed to UTC datetimes. Rows without a start
        time (such as steaming reports) are dropped.

    Raises:
        ValueError: If the archive holds no single DCA file or the file
            lacks a column in ``DCA_COLUMNS``.
    """
    with zipfile.ZipFile(path) as archive:
        names = [n for n in archive.namelist() if n.endswith("-dca.csv")]
        if len(names) != 1:
            raise ValueError(f"{path}: expected one DCA CSV, found {names}")
        raw = archive.read(names[0])
    frame = pl.read_csv(io.BytesIO(raw), separator=";", infer_schema=False)
    missing = set(DCA_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"{path}: DCA columns missing: {sorted(missing)}")
    frame = frame.select(
        pl.col(source).str.replace(",", ".").cast(dtype).alias(name)
        for source, (name, dtype) in DCA_COLUMNS.items()
    ).with_columns(
        pl.col("start", "stop").str.to_datetime(_DCA_TIME_FORMAT, time_zone="UTC")
    )
    return frame.filter(pl.col("start") >= start, pl.col("start") < end)


_GML = "{http://www.opengis.net/gml/3.2}"
_CORAL = (
    "{http://skjema.geonorge.no/SOSI/produktspesifikasjon/KorallrevForbud/20170601}"
)


def read_coral_gml(path: Path) -> list[dict]:
    """Read the coral-reef closures from a WFS GetFeature response.

    Args:
        path: GML 3.2 file in EPSG:4326, whose axis order is latitude then
            longitude.

    Returns:
        list[dict]: One dict per closure: ``name``, ``paragraph`` (section
        and letter of the regulation, such as ``3d``), ``ban`` (the
        regulation's description of the banned gear), ``in_force`` (date)
        and ``polygon`` (a shapely Polygon in longitude, latitude order).

    Raises:
        ValueError: If a closure has interior rings or no exterior ring,
            which this reader does not handle.
    """
    closures = []
    for feature in ET.parse(path).getroot().iter(f"{_CORAL}Korallrev"):
        rings = feature.findall(f".//{_GML}LinearRing")
        exterior = feature.find(f".//{_GML}exterior/{_GML}LinearRing")
        if exterior is None or len(rings) != 1:
            raise ValueError(f"{path}: closure without a single exterior ring")
        values = [float(v) for v in exterior.findtext(f"{_GML}posList").split()]
        closures.append(
            {
                "name": feature.findtext(f"{_CORAL}områdenavn"),
                "paragraph": feature.findtext(f".//{_CORAL}forskriftsParagraf")
                + feature.findtext(f".//{_CORAL}forskriftsBokstav"),
                "ban": feature.findtext(f".//{_CORAL}forskriftsParagrafinfo"),
                "in_force": date.fromisoformat(
                    feature.findtext(f".//{_CORAL}forskriftsIkrafttredelsesdato")
                ),
                "polygon": Polygon(zip(values[1::2], values[0::2])),
            }
        )
    return closures
