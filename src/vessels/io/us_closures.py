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
