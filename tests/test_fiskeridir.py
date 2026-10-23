"""Tests for the Fiskeridirektoratet readers (VMS, ERS DCA and coral-reef
closures), on small files built in the test with the real headers and row
formats."""

import zipfile
from datetime import date, datetime, timezone

import polars as pl
import pytest

from vessels.io.fiskeridir import (
    DCA_COLUMNS,
    VMS_COLUMNS,
    read_coral_gml,
    read_dca_zip,
    read_vms_zip,
    vms_date_token,
)

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


DCA_HEADER = ";".join(DCA_COLUMNS)
DCA_ROWS = [
    "3130313;1;FIS;LAAA;27.05.2024 06:05:00;27.05.2024 12:09:00;"
    "67,35;9,1;67,4;9,25;364;OTB;POK;800",
    "3130313;1;FIS;LAAA;27.05.2024 06:05:00;27.05.2024 12:09:00;"
    "67,35;9,1;67,4;9,25;364;OTB;COD;120,5",
    "3130400;1;STE;LBBB;;;;;;;;;;",
    "3130500;2;FIS;LBBB;03.06.2024 00:00:00;03.06.2024 02:00:00;"
    "64,0;8,0;64,1;8,1;120;SSC;HAD;10",
]


def write_ers_zip(path, header=DCA_HEADER, rows=DCA_ROWS):
    """Write an ERS archive with a DCA file and one unrelated table.

    Args:
        path: Zip file to create.
        header: DCA header line.
        rows: DCA data lines.

    Returns:
        Path: The zip written.
    """
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("ers-2024-dca.csv", "\n".join([header, *rows]) + "\n")
        archive.writestr("ers-2024-por.csv", "a;b\n1;2\n")
    return path


WEEK_START = datetime(2024, 5, 27, tzinfo=timezone.utc)
WEEK_END = datetime(2024, 6, 3, tzinfo=timezone.utc)


def test_dca_reader_parses_commas_and_times(tmp_path):
    """Decimal commas become floats and times are UTC; the week end is excluded."""
    frame = read_dca_zip(write_ers_zip(tmp_path / "e.zip"), WEEK_START, WEEK_END)
    assert frame["round_weight_kg"].to_list() == [800.0, 120.5]
    assert frame["start_lat"].to_list() == [67.35, 67.35]
    assert frame["stop"][0] == datetime(2024, 5, 27, 12, 9, tzinfo=timezone.utc)


def test_dca_reader_rejects_missing_column(tmp_path):
    """A dropped column fails loudly."""
    header = DCA_HEADER.replace("Rundvekt", "Vekt")
    path = write_ers_zip(tmp_path / "e.zip", header=header)
    with pytest.raises(ValueError, match="DCA columns missing"):
        read_dca_zip(path, WEEK_START, WEEK_END)


CORAL_GML = """<?xml version='1.0' encoding='UTF-8'?>
<wfs:FeatureCollection xmlns:wfs="http://www.opengis.net/wfs/2.0"
    xmlns:gml="http://www.opengis.net/gml/3.2"
    xmlns:app="http://skjema.geonorge.no/SOSI/produktspesifikasjon/KorallrevForbud/20170601">
  <wfs:member>
    <app:Korallrev gml:id="korallrev.1">
      <app:område>
        <gml:Polygon gml:id="p1" srsName="urn:ogc:def:crs:EPSG::4326">
          <gml:exterior><gml:LinearRing>
            <gml:posList>67.3 8.8 67.3 9.7 67.6 9.7 67.3 8.8</gml:posList>
          </gml:LinearRing></gml:exterior>
        </gml:Polygon>
      </app:område>
      <app:områdenavn>Røstrevet</app:områdenavn>
      <app:forskrift><app:ForskriftsReferanse>
        <app:forskriftsParagraf>3</app:forskriftsParagraf>
        <app:forskriftsBokstav>c</app:forskriftsBokstav>
        <app:forskriftsParagrafinfo>Forbud mot bruk av redskap som slepes under fiske</app:forskriftsParagrafinfo>
        <app:forskriftsIkrafttredelsesdato>2016-04-15</app:forskriftsIkrafttredelsesdato>
      </app:ForskriftsReferanse></app:forskrift>
    </app:Korallrev>
  </wfs:member>
</wfs:FeatureCollection>
"""


def test_coral_reader_swaps_to_lon_lat(tmp_path):
    """Closures come back with their fields and polygons in lon, lat order."""
    path = tmp_path / "k.gml"
    path.write_text(CORAL_GML, encoding="utf-8")
    (closure,) = read_coral_gml(path)
    assert (closure["name"], closure["paragraph"]) == ("Røstrevet", "3c")
    assert closure["in_force"] == date(2016, 4, 15)
    assert closure["polygon"].bounds == (8.8, 67.3, 9.7, 67.6)
