"""Committed reference layers: read them into shapely geometries.

Layers are stored as GeoJSON in longitude and latitude so a reviewer can
open them in any GIS. Reading needs only ``json`` and shapely, keeping
GeoPandas out of the pipeline.
"""

import json
from pathlib import Path

import shapely
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry

REFERENCE_US = Path(__file__).resolve().parents[3] / "data" / "reference" / "us"
LAND_US = REFERENCE_US / "land.geojson"


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
