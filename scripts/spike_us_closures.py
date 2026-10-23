"""Spike (a): is there towing-speed fishing inside the US candidate closures?

Local only: it reads the git-ignored MarineCadastre days and closure files
fetched by ``scripts/fetch_us.py``. The days, polygons, speed band and the
near-zero rule were fixed in the project plan before any day was
downloaded. Run it with ``PYTHONPATH=src python scripts/spike_us_closures.py``.

For each day it keeps fishing-type positions (vessel types 30, 1001 and
1002) inside the closures' bounding box, then counts positions and MMSIs
inside each closure, at any speed and at towing speed (2.5-5.5 kn). The
verdict is taken over all days and the union of the closures.
"""

from pathlib import Path

import duckdb
import geopandas as gpd
import polars as pl
import shapely
from shapely.geometry import Polygon

from vessels.io.us_closures import cfr_polygon, cfr_text

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
CLOSURES = RAW / "us_closures"
HMA_SHAPEFILE = (
    "zip://"
    + str(CLOSURES / "habitat-management-areas-20180409.zip")
    + "!Habitat_Management_Areas/Habitat_Management_Areas.shp"
)
FISHING_TYPES = (30, 1001, 1002)
TOWING_KN = (2.5, 5.5)
# Near-zero: fewer than this many distinct MMSIs or positions at towing
# speed, over all days and the union of the closures.
MIN_MMSIS = 3
MIN_POSITIONS = 100


def load_closures() -> dict[str, Polygon]:
    """The three candidate closures, as fixed in the plan.

    Western GOM and Stellwagen come from the CFR point tables, which are
    straight lines and legal. Eastern Maine's north-west edge follows the
    state-waters limit, which the CFR does not list, so it comes from the
    GARFO shapefile.

    Returns:
        dict: Name to polygon in longitude, latitude (NAD83, treated as
        WGS84: the two differ by about a metre here).
    """
    section_370 = cfr_text(CLOSURES / "cfr_648_370_2024-07-01.xml.gz")
    section_371 = cfr_text(CLOSURES / "cfr_648_371_2024-07-01.xml.gz")
    hma = gpd.read_file(HMA_SHAPEFILE).set_index("COMMNAME").geometry
    return {
        "western_gom_hma_ex_shrimp": cfr_polygon(section_370, "WGMH").difference(
            cfr_polygon(section_370, "SEA")
        ),
        "stellwagen_dhra": cfr_polygon(section_371, "SDHRA"),
        "eastern_maine_hma": hma["Eastern ME HMA"],
    }


def load_day(path: Path, bounds: tuple) -> pl.DataFrame:
    """Read one day's fishing-type positions inside a bounding box.

    DuckDB 2.0 without the spatial extension can print the geometry as
    WKT but not take coordinates from it, so they are parsed from the text.

    Args:
        path: A MarineCadastre daily GeoParquet file.
        bounds: (min lon, min lat, max lon, max lat).

    Returns:
        pl.DataFrame: ``mmsi``, ``timestamp``, ``sog``, ``vessel_type``,
        ``lon`` and ``lat``.
    """
    min_lon, min_lat, max_lon, max_lat = bounds
    types = ", ".join(str(t) for t in FISHING_TYPES)
    sql = f"""
        with points as (
            select mmsi, base_date_time as timestamp, sog, vessel_type,
                   regexp_extract(st_astext(geometry),
                       'POINT \\(([-0-9.e]+) ([-0-9.e]+)\\)', ['lon', 'lat'])
                       as xy
            from read_parquet('{path}')
            where vessel_type in ({types})
        )
        select mmsi, timestamp, sog::double as sog, vessel_type,
               xy.lon::double as lon, xy.lat::double as lat
        from points
        where xy.lon::double between {min_lon} and {max_lon}
          and xy.lat::double between {min_lat} and {max_lat}
    """
    return pl.DataFrame(duckdb.sql(sql))


def count_inside(points: pl.DataFrame, closures: dict[str, Polygon]) -> pl.DataFrame:
    """Count positions and MMSIs inside each closure and their union.

    Args:
        points: Positions with ``mmsi``, ``sog``, ``lon`` and ``lat``.
        closures: Name to polygon.

    Returns:
        pl.DataFrame: One row per closure plus ``union``: ``positions``,
        ``mmsis``, ``towing_positions`` and ``towing_mmsis``.
    """
    shapes = {**closures, "union": shapely.union_all(list(closures.values()))}
    lon, lat = points["lon"].to_numpy(), points["lat"].to_numpy()
    rows = []
    for name, shape in shapes.items():
        inside = points.filter(pl.Series(shapely.contains_xy(shape, lon, lat)))
        towing = inside.filter(pl.col("sog").is_between(*TOWING_KN))
        rows.append(
            {
                "closure": name,
                "positions": inside.height,
                "mmsis": inside["mmsi"].n_unique(),
                "towing_positions": towing.height,
                "towing_mmsis": towing["mmsi"].n_unique(),
            }
        )
    return pl.DataFrame(rows)


def near_zero(towing_positions: int, towing_mmsis: int) -> bool:
    """The plan's near-zero rule for spike (a).

    Args:
        towing_positions: Towing-speed positions in the union, all days.
        towing_mmsis: Distinct MMSIs among them.

    Returns:
        bool: True if the closure detector has nothing real to score.
    """
    return towing_mmsis < MIN_MMSIS or towing_positions < MIN_POSITIONS


def main():
    """Count each spike day and print the verdict."""
    closures = load_closures()
    bounds = shapely.union_all(list(closures.values())).bounds
    days = sorted((RAW / "marinecadastre").glob("ais-2024-*.parquet"))
    points = []
    for path in days:
        day = load_day(path, bounds).with_columns(day=pl.lit(path.stem[4:]))
        points.append(day)
        print(f"{path.stem}: {day.height:,} fishing-type positions in the box")
        with pl.Config(tbl_rows=-1):
            print(count_inside(day, closures))
    total = count_inside(pl.concat(points), closures)
    with pl.Config(tbl_rows=-1):
        print("All days:")
        print(total)
    union = total.filter(pl.col("closure") == "union").row(0, named=True)
    verdict = near_zero(union["towing_positions"], union["towing_mmsis"])
    print(
        f"Union, {len(days)} days: {union['towing_positions']:,} towing-speed "
        f"positions from {union['towing_mmsis']} MMSIs; "
        f"{'near-zero' if verdict else 'not near-zero'}"
    )


if __name__ == "__main__":
    main()
