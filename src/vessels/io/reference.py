"""Committed reference layers: read them, and write their checksums.

Polygon layers are stored as GeoJSON in longitude and latitude so a
reviewer can open them in any GIS; point layers (anchorages) as CSV.
Reading needs only ``json``, polars and shapely, keeping GeoPandas out of
the pipeline.
"""

import json
from datetime import date
from pathlib import Path

import polars as pl
import shapely
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry

from vessels.core.geofence import Rule, check_table
from vessels.io.fetch import sha256_of

REFERENCE_US = Path(__file__).resolve().parents[3] / "data" / "reference" / "us"
LAND_US = REFERENCE_US / "land.geojson"
CLOSURES_US = REFERENCE_US / "closures.geojson"
ANCHORAGES_US = REFERENCE_US / "anchorages.csv"
# Every committed layer; SHA256SUMS lists exactly these.
LAYER_PATTERNS = ("*.geojson", "*.csv")


def read_land(path: Path = LAND_US) -> BaseGeometry:
    """Read a land layer as one geometry.

    Args:
        path: GeoJSON FeatureCollection of land polygons.

    Returns:
        BaseGeometry: The union of every feature, in degrees.

    Raises:
        ValueError: If the file holds no features.
    """
    features = json.loads(path.read_text(encoding="utf-8"))["features"]
    if not features:
        raise ValueError(f"{path}: no land features")
    return shapely.union_all([shape(f["geometry"]) for f in features])


def read_closures(path: Path = CLOSURES_US) -> list[Rule]:
    """Read a geofence table into rules.

    Properties that only document a row (name, citation, sources, status)
    stay in the file; the rules carry what the join needs.

    Args:
        path: GeoJSON FeatureCollection, one feature per rule, with the
            ``Rule`` fields as properties (dates in ISO format).

    Returns:
        list[Rule]: The table in file order.

    Raises:
        ValueError: If a row is malformed or the table's parents do not fit
            together.
    """
    rules = []
    for feature in json.loads(path.read_text(encoding="utf-8"))["features"]:
        props = feature["properties"]
        rules.append(
            Rule(
                area_id=props["area_id"],
                kind=props["kind"],
                gear=tuple(props["gear"]),
                valid_from=date.fromisoformat(props["valid_from"]),
                valid_to=(
                    date.fromisoformat(props["valid_to"]) if props["valid_to"] else None
                ),
                geometry=shape(feature["geometry"]),
                parent=props["parent"],
                season=tuple(props["season"]) if props["season"] else None,
                timezone=props["timezone"],
            )
        )
    check_table(rules)
    return rules


def read_anchorages(path: Path = ANCHORAGES_US) -> pl.DataFrame:
    """Read the anchorage cells.

    Args:
        path: CSV with ``h3_cell``, ``lon``, ``lat``, ``vessels``,
            ``runs`` and ``shore_km``.

    Returns:
        pl.DataFrame: One row per anchorage cell.

    Raises:
        ValueError: If the file holds no anchorages.
    """
    frame = pl.read_csv(path, schema_overrides={"h3_cell": pl.String})
    if frame.is_empty():
        raise ValueError(f"{path}: no anchorages")
    return frame


def write_sums(folder: Path = REFERENCE_US) -> Path:
    """Write ``SHA256SUMS`` for every committed layer in a folder.

    Args:
        folder: The reference folder.

    Returns:
        Path: The checksum file.
    """
    paths = sorted(p for pattern in LAYER_PATTERNS for p in folder.glob(pattern))
    lines = [f"{sha256_of(path)}  {path.name}" for path in sorted(paths)]
    sums = folder / "SHA256SUMS"
    sums.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return sums
