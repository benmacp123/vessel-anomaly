"""Committed reference layers: read them into shapely geometries.

Layers are stored as GeoJSON in longitude and latitude so a reviewer can
open them in any GIS. Reading needs only ``json`` and shapely, keeping
GeoPandas out of the pipeline.
"""

import json
from datetime import date
from pathlib import Path

import shapely
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry

from vessels.core.geofence import Rule, check_table

REFERENCE_US = Path(__file__).resolve().parents[3] / "data" / "reference" / "us"
LAND_US = REFERENCE_US / "land.geojson"
CLOSURES_US = REFERENCE_US / "closures.geojson"


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
