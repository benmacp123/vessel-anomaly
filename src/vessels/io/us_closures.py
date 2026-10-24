"""Readers for US closure boundaries printed in the Code of Federal
Regulations.

The CFR, not GARFO's shapefiles, is the legal definition: GARFO calls its
GIS files approximate. Areas the CFR defines by straight lines between
listed points are read from the eCFR XML here; areas with an edge that
follows another boundary (such as the outer limit of state waters) need
the shapefile instead.
"""

import gzip
import re
from pathlib import Path

from shapely.geometry import Polygon

# A point row as it reads once tags are stripped, such as
# "SDHRA3 42°45.2′ N 70°15.0′ W" or "SEA1 43°15′ N 70° W"; a footnote
# marker ("EMH1 ^1") may follow the label.
_POINT = (
    r"{prefix}(\d+)(?:\s*\^\d+)?\s+"
    r"(\d+)°(?:([\d.]+)′)?\s*N\s+(\d+)°(?:([\d.]+)′)?\s*W"
)
# One coordinate of either axis, such as "69°49.255′ W" or "41°7′ N".
_COORD = r"(\d+)°(?:([\d.]+)′)?\s*([NW])"


def cfr_text(path: Path) -> str:
    """Read an eCFR XML section as plain text.

    Footnote markers are kept as ``^n`` so they do not run into the
    numbers beside them.

    Args:
        path: A gzipped eCFR XML response.

    Returns:
        str: The section's text with tags removed and whitespace collapsed.
    """
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        text = handle.read()
    text = re.sub(r"<sup>(.*?)</sup>", r" ^\1 ", text)
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", text))


def cfr_polygon(text: str, prefix: str) -> Polygon:
    """Build the polygon a CFR point table defines by straight lines.

    Args:
        text: ``cfr_text`` output.
        prefix: The point labels' prefix, such as ``SDHRA``.

    Returns:
        Polygon: Longitude, latitude (west negative), in the listed order.
        The table's closing repeat of the first point is dropped.

    Raises:
        ValueError: If the table has fewer than three points or its labels
            are not numbered 1, 2, 3 and so on.
    """
    rows = re.findall(_POINT.format(prefix=re.escape(prefix)), text)
    points, labels = [], []
    for label, lat_deg, lat_min, lon_deg, lon_min in rows:
        if labels and label == labels[0]:
            break
        labels.append(label)
        points.append(
            (
                -(int(lon_deg) + float(lon_min or 0) / 60),
                int(lat_deg) + float(lat_min or 0) / 60,
            )
        )
    if len(points) < 3 or labels != [str(i) for i in range(1, len(labels) + 1)]:
        raise ValueError(f"{prefix}: unexpected point table {labels}")
    return Polygon(points)


def cfr_table(text: str, heading: str) -> Polygon:
    """Build the polygon of the first point table after a heading.

    Some tables have no common label prefix (the 2018 Great South Channel
    table runs GSC1, MBTG2, MBTG3) or list longitude before latitude (the
    dredge exemption areas), so this reader ignores labels and pairs the
    coordinates in either order. It stops at the closing repeat of the
    first point, which must be there.

    Args:
        text: ``cfr_text`` output.
        heading: Text that appears once just before the table, such as
            ``"McBlair Dredge Exemption Area is defined"``.

    Returns:
        Polygon: Longitude, latitude (west negative), in the listed order,
        without the closing repeat.

    Raises:
        ValueError: If the heading is missing or ambiguous, a pair is not
            one latitude and one longitude, or the table does not close on
            its first point after at least three points.
    """
    if text.count(heading) != 1:
        raise ValueError(f"{heading!r}: found {text.count(heading)} times")
    coords = re.findall(_COORD, text[text.index(heading) :])
    points = []
    for first, second in zip(coords[0::2], coords[1::2]):
        if {first[2], second[2]} != {"N", "W"}:
            break
        lat, lon = (first, second) if first[2] == "N" else (second, first)
        point = (
            -(int(lon[0]) + float(lon[1] or 0) / 60),
            int(lat[0]) + float(lat[1] or 0) / 60,
        )
        if points and point == points[0]:
            if len(points) < 3:
                break
            return Polygon(points)
        points.append(point)
    raise ValueError(f"{heading!r}: no closed table of three or more points")
