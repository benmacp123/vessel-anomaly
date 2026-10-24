"""Tests for the CFR closure reader and the spike (a) counts, on small
inputs built in the test in the eCFR's own markup."""

import gzip
import importlib.util
from pathlib import Path

import polars as pl
import pytest
from shapely.geometry import Polygon

from vessels.io.us_closures import cfr_polygon, cfr_table, cfr_text

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "spike_us_closures.py"
TABLE = """<P>(b) Stellwagen DHRA.</P><TABLE><TBODY>
<TR><TD>SDHRA1 <sup>1</sup></TD><TD>42°15.0′ N</TD><TD>70°00.0′ W</TD></TR>
<TR><TD>SDHRA2</TD><TD>42°15.0′ N</TD><TD>70°15.0′ W</TD></TR>
<TR><TD>SDHRA3</TD><TD>42°45′ N</TD><TD>70° W</TD></TR>
<TR><TD>SDHRA1</TD><TD>42°15.0′ N</TD><TD>70°00.0′ W</TD></TR>
</TBODY></TABLE>"""

# The 2024 McBlair table as eCFR prints it: plain numbers, longitude first.
LON_FIRST = """<P>(A) The McBlair Dredge Exemption Area is defined by the following
points:</P><TABLE><TBODY><TR><TH>Point</TH><TH>Longitude</TH><TH>Latitude</TH></TR>
<TR><TD>1</TD><TD>69°49.255′ W</TD><TD>41°25.878′ N</TD></TR>
<TR><TD>2</TD><TD>69°46.951′ W</TD><TD>41°25.878′ N</TD></TR>
<TR><TD>3</TD><TD>69°46.951′ W</TD><TD>41°19.34′ N</TD></TR>
<TR><TD>1</TD><TD>69°49.255′ W</TD><TD>41°25.878′ N</TD></TR>
</TBODY></TABLE><P>(B) Open year-round. (iii) Next area 41°7′ N 69°45′ W</P>"""


@pytest.fixture(name="spike")
def fixture_spike():
    """Load the spike script as a module, since ``scripts/`` is not a package.

    Returns:
        module: The loaded spike script.
    """
    spec = importlib.util.spec_from_file_location("spike_us_closures", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_gz(path, text):
    """Write text gzipped, as the eCFR fetch stores it.

    Args:
        path: File to create.
        text: Content.

    Returns:
        Path: The file written.
    """
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(text)
    return path


def test_cfr_polygon_reads_points_in_order(tmp_path):
    """Footnotes and minute-less degrees parse; the closing repeat is dropped."""
    polygon = cfr_polygon(cfr_text(write_gz(tmp_path / "s.xml.gz", TABLE)), "SDHRA")
    assert list(polygon.exterior.coords)[:-1] == [
        (-70.0, 42.25),
        (-70.25, 42.25),
        (-70.0, 42.75),
    ]


def test_cfr_polygon_rejects_gaps(tmp_path):
    """A missing point fails loudly instead of cutting a corner."""
    text = cfr_text(write_gz(tmp_path / "s.xml.gz", TABLE.replace("SDHRA2", "SDHRA9")))
    with pytest.raises(ValueError, match="unexpected point table"):
        cfr_polygon(text, "SDHRA")


def test_cfr_table_reads_longitude_first(tmp_path):
    """A lon-first table parses and stops at its closing repeat."""
    text = cfr_text(write_gz(tmp_path / "s.xml.gz", LON_FIRST))
    polygon = cfr_table(text, "McBlair Dredge Exemption Area is defined")
    assert list(polygon.exterior.coords)[:-1] == pytest.approx(
        [
            (-(69 + 49.255 / 60), 41 + 25.878 / 60),
            (-(69 + 46.951 / 60), 41 + 25.878 / 60),
            (-(69 + 46.951 / 60), 41 + 19.34 / 60),
        ]
    )


def test_cfr_table_reads_mixed_labels_latitude_first(tmp_path):
    """Labels are ignored, so a table mixing prefixes still parses."""
    text = cfr_text(write_gz(tmp_path / "s.xml.gz", TABLE.replace("SDHRA2", "MBTG2")))
    assert len(cfr_table(text, "Stellwagen DHRA").exterior.coords) == 4


def test_cfr_table_rejects_unclosed_table(tmp_path):
    """Without the closing repeat it fails rather than run into the next table."""
    unclosed = LON_FIRST.replace(
        "<TR><TD>1</TD><TD>69°49.255′ W</TD><TD>41°25.878′ N</TD></TR>\n</TBODY>",
        "</TBODY>",
    )
    text = cfr_text(write_gz(tmp_path / "s.xml.gz", unclosed))
    with pytest.raises(ValueError, match="no closed table"):
        cfr_table(text, "McBlair Dredge Exemption Area is defined")


def test_cfr_table_rejects_missing_heading(tmp_path):
    """A heading that is not in the text fails loudly."""
    text = cfr_text(write_gz(tmp_path / "s.xml.gz", LON_FIRST))
    with pytest.raises(ValueError, match="found 0 times"):
        cfr_table(text, "Fishing Rip Dredge Exemption Area is defined")


def test_count_inside_splits_speed_and_union(spike):
    """Counts separate towing speed, and overlapping closures count once."""
    big = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
    small = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    points = pl.DataFrame(
        {
            "mmsi": [1, 1, 2, 3],
            "sog": [3.0, 9.0, 4.0, 3.0],
            "lon": [0.5, 0.5, 1.5, 5.0],
            "lat": [0.5, 0.5, 1.5, 5.0],
        }
    )
    counts = spike.count_inside(points, {"big": big, "small": small})
    rows = {r["closure"]: r for r in counts.iter_rows(named=True)}
    assert (rows["small"]["positions"], rows["small"]["towing_positions"]) == (2, 1)
    assert (rows["union"]["positions"], rows["union"]["towing_mmsis"]) == (3, 2)


def test_near_zero_rule(spike):
    """Near-zero needs both 3 MMSIs and 100 positions to be avoided."""
    assert spike.near_zero(99, 10)
    assert spike.near_zero(500, 2)
    assert not spike.near_zero(100, 3)
