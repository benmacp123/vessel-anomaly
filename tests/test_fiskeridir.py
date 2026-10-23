"""Tests for the Fiskeridirektoratet VMS reader, on a small archive built
in the test with the real header and row format."""

import zipfile
from datetime import date, datetime, timezone

import polars as pl
import pytest

from vessels.io.fiskeridir import VMS_COLUMNS, read_vms_zip, vms_date_token

HEADER = ";".join(f'"{c}"' for c in VMS_COLUMNS)
ROWS = [
    '1;"POS";"Posisjonsrapport";26-MAY-24 23.59.00.000000000;64.1;8.0;90;'
    '3.5;"LAAA";"M 0001A";"ÆRLIG";20.5;;650;"Fiskefartøy"',
    '2;"POS";"Posisjonsrapport";27-MAY-24 00.10.00.000000000;64.2;8.1;0;'
    '0.1;"LAAA";"M 0001A";"ÆRLIG";20.5;;650;"Fiskefartøy"',
    '3;"POS";"Posisjonsrapport";27-MAY-24 13.40.00.000000000;67.3;9.0;360;'
    '12.4;"LBBB";;"BØEN";75;6870;7242;"Fiskefartøy"',
]


def write_zip(path, header=HEADER, rows=ROWS):
    """Write a one-file VMS archive.

    Args:
        path: Zip file to create.
        header: CSV header line.
        rows: CSV data lines.

    Returns:
        Path: The zip written.
    """
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("2024-VMS.csv", "\n".join([header, *rows]) + "\n")
    return path


def test_date_token_is_locale_free():
    """The day pattern uses upper-case English months and two-digit years."""
    assert vms_date_token(date(2024, 6, 2)) == b";02-JUN-24 "


def test_reader_keeps_only_requested_days(tmp_path):
    """Only rows on the requested days survive, with their message IDs."""
    frame = read_vms_zip(write_zip(tmp_path / "v.zip"), [date(2024, 5, 27)])
    assert frame["message_id"].to_list() == [2, 3]


def test_reader_types_and_utc_times(tmp_path):
    """Columns are renamed and typed, and times are parsed as UTC."""
    frame = read_vms_zip(write_zip(tmp_path / "v.zip"), [date(2024, 5, 27)])
    assert frame.columns == [name for name, _ in VMS_COLUMNS.values()]
    assert frame["timestamp"].to_list()[1] == datetime(
        2024, 5, 27, 13, 40, tzinfo=timezone.utc
    )
    assert frame["speed_kn"].to_list() == [0.1, 12.4]
    assert frame["registration"].to_list() == ["M 0001A", None]
    assert frame["vessel_name"].to_list() == ["ÆRLIG", "BØEN"]


def test_reader_rejects_changed_header(tmp_path):
    """A renamed column fails loudly instead of shifting fields."""
    path = write_zip(tmp_path / "v.zip", header=HEADER.replace("Fart", "Speed"))
    with pytest.raises(ValueError, match="unexpected VMS header"):
        read_vms_zip(path, [date(2024, 5, 27)])


def test_reader_empty_selection(tmp_path):
    """A day with no reports gives an empty, correctly typed frame."""
    frame = read_vms_zip(write_zip(tmp_path / "v.zip"), [date(2024, 7, 1)])
    assert frame.height == 0
    assert frame.schema["timestamp"] == pl.Datetime("us", "UTC")
