"""Build the committed US reference layers from raw downloads.

Owner only, though nothing in it is secret: run
``PYTHONPATH=src python scripts/fetch_us.py`` first, then
``PYTHONPATH=src python scripts/build_us_reference.py``. It writes, under
``data/reference/us/``:

- ``land.geojson``: Natural Earth 10m land clipped to the study box plus a
  margin, so the clip edge does not act as a coastline for the land
  mask's inland buffer. Coordinates are rounded to 1e-6 degrees (about
  0.1 m) to keep the file small and stable;
- ``SHA256SUMS`` for every layer.
"""

import json
from pathlib import Path

import geopandas as gpd
import shapely

from vessels.io.blocks import BOX
from vessels.io.fetch import sha256_of
from vessels.io.reference import REFERENCE_US

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "reference"
# Degrees added on every side of the box; far more than any land buffer.
MARGIN = 0.5


def land_geojson(zip_path: Path) -> dict:
    """Clip Natural Earth land to the padded study box.

    Args:
        zip_path: The downloaded ``ne_10m_land.zip``.

    Returns:
        dict: A GeoJSON FeatureCollection with one MultiPolygon feature.
    """
    min_lon, min_lat, max_lon, max_lat = BOX
    padded = (min_lon - MARGIN, min_lat - MARGIN, max_lon + MARGIN, max_lat + MARGIN)
    land = gpd.read_file(f"zip://{zip_path}").clip(padded)
    geom = shapely.set_precision(shapely.union_all(land.geometry.values), 1e-6)
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"source": "Natural Earth 10m land"},
                "geometry": json.loads(shapely.to_geojson(geom)),
            }
        ],
    }


def main():
    """Write every reference layer and its checksums."""
    REFERENCE_US.mkdir(parents=True, exist_ok=True)
    land = REFERENCE_US / "land.geojson"
    land.write_text(
        json.dumps(land_geojson(RAW / "ne_10m_land.zip")) + "\n", encoding="utf-8"
    )
    lines = [
        f"{sha256_of(path)}  {path.name}"
        for path in sorted(REFERENCE_US.glob("*.geojson"))
    ]
    (REFERENCE_US / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{land}: {land.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
