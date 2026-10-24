"""Distance to shore and to port, and the anchorages that define ports.

Distance to shore is signed: positive at sea, negative on land, in
kilometres along the great circle to the nearest coastline point. The
nearest segment is found in a local flat projection and the distance to
the point found is then measured exactly, so the projection's few-percent
scale error moves which coastline point is chosen, not the distance.

Anchorages follow Global Fishing Watch's method: a vessel is stationary
when it stays within 0.5 km for at least 12 hours, and a cell where
enough different vessels were stationary is an anchorage. GFW asks for 20
vessels over years of global data; four weeks of mostly fishing traffic
reaches that only at New Bedford, so the vessel count here is lower, and
cells far from shore are excluded because a loose threshold would
otherwise call an offshore meeting spot a port and hide the encounters
there. Distance to port is the distance to the nearest anchorage cell's
centre.
"""

import h3
import numpy as np
import polars as pl
import shapely
from scipy.spatial import cKDTree
from shapely.geometry.base import BaseGeometry

from vessels.core.geo import LocalProjection, haversine_km

# GFW's stationary rule: within 0.5 km for at least 12 hours.
RADIUS_KM = 0.5
MIN_HOURS = 12.0
# H3 resolution 8 cells average 0.74 km², the nearest H3 size to GFW's
# S2 level-14 cells (about 0.3 km²) that still pools a harbour's berths.
RESOLUTION = 8
# Provisional, set on the tuning block (2026-10-24): GFW's 20 vessels
# finds only New Bedford there; 3 misses small harbours (14% of positions
# under 0.5 kn lay within 2 km of shore but 10 km from an anchorage), 2
# leaves 1% and still puts no candidate offshore.
MIN_VESSELS = 2
# Provisional: harbours drawn as land by Natural Earth read as negative,
# outer anchorages a few kilometres out.
MAX_SHORE_KM = 5.0
# Anchorages checked exactly for each position, nearest by projection
# first, so the projection's scale error cannot pick the wrong one.
NEAREST_PORTS = 4


class ShoreDistance:
    """Signed great-circle distance to the nearest coastline.

    Attributes:
        projection: Flat projection used to find the nearest segment.
    """

    def __init__(self, land: BaseGeometry, projection: LocalProjection):
        """Index the coastline segment by segment.

        Args:
            land: Land polygons in degrees.
            projection: Projection about the study area.
        """
        self.projection = projection
        self._land = projection.geometry(land)
        shapely.prepare(self._land)
        rings = shapely.get_parts(shapely.boundary(self._land))
        coords, ring = shapely.get_coordinates(rings, return_index=True)
        # One short segment per tree entry keeps nearest queries tight
        # where a long ring's box would cover most of the area.
        same = ring[:-1] == ring[1:]
        self._segments = shapely.linestrings(
            np.stack([coords[:-1][same], coords[1:][same]], axis=1)
        )
        self._tree = shapely.STRtree(self._segments)

    def km(self, lon, lat) -> np.ndarray:
        """Distance from each position to the nearest coastline.

        Args:
            lon: Longitudes, degrees.
            lat: Latitudes, degrees.

        Returns:
            np.ndarray: Kilometres, negative on land, 0 on the coastline.
        """
        lon = np.asarray(lon, dtype=float)
        lat = np.asarray(lat, dtype=float)
        if lon.size == 0:
            return np.empty(0)
        x, y = self.projection.forward(lon, lat)
        points = shapely.points(x, y)
        found, segment = self._tree.query_nearest(points, all_matches=False)
        lines = self._segments[segment]
        nearest = shapely.line_interpolate_point(
            lines, shapely.line_locate_point(lines, points[found])
        )
        near_lon, near_lat = self.projection.inverse(
            *shapely.get_coordinates(nearest).T
        )
        km = np.empty(lon.size)
        km[found] = haversine_km(lon[found], lat[found], near_lon, near_lat)
        return np.where(shapely.contains_xy(self._land, x, y), -km, km)


def stationary_runs(
    positions: pl.DataFrame,
    projection: LocalProjection,
    radius_km: float = RADIUS_KM,
    min_hours: float = MIN_HOURS,
) -> pl.DataFrame:
    """Find where each vessel stayed within a radius for long enough.

    A run starts at a position and takes every later position of the same
    vessel until one lies beyond ``radius_km`` of that first position,
    which starts the next run. A reception gap inside a run does not end
    it, as in GFW's rule.

    Args:
        positions: ``vessel_id``, ``timestamp``, ``lon``, ``lat``, in any
            order.
        projection: Projection about the study area.
        radius_km: How far a vessel may move from the run's first
            position.
        min_hours: Shortest run kept, first to last position.

    Returns:
        pl.DataFrame: ``vessel_id``, ``start``, ``end``, ``hours`` and the
        run's mean ``lon`` and ``lat``, one row per run kept.
    """
    frame = positions.sort("vessel_id", "timestamp")
    x, y = projection.forward(frame["lon"].to_numpy(), frame["lat"].to_numpy())
    vessels = frame["vessel_id"].to_list()
    x, y = x.tolist(), y.tolist()
    limit = radius_km**2
    run = [0] * frame.height
    start = 0
    # A plain loop: each run's anchor depends on where the last one ended.
    for i in range(1, frame.height):
        if (
            vessels[i] != vessels[start]
            or (x[i] - x[start]) ** 2 + (y[i] - y[start]) ** 2 > limit
        ):
            start = i
        run[i] = start
    return (
        frame.with_columns(run=pl.Series(run, dtype=pl.Int64))
        .group_by("run")
        .agg(
            pl.col("vessel_id").first(),
            start=pl.col("timestamp").min(),
            end=pl.col("timestamp").max(),
            lon=pl.col("lon").mean(),
            lat=pl.col("lat").mean(),
        )
        .with_columns(hours=(pl.col("end") - pl.col("start")).dt.total_seconds() / 3600)
        .filter(pl.col("hours") >= min_hours)
        .sort("vessel_id", "start")
        .select("vessel_id", "start", "end", "hours", "lon", "lat")
    )


def anchorage_cells(
    runs: pl.DataFrame,
    shore: ShoreDistance,
    resolution: int = RESOLUTION,
    min_vessels: int = MIN_VESSELS,
    max_shore_km: float = MAX_SHORE_KM,
) -> pl.DataFrame:
    """Cells where enough different vessels were stationary.

    Args:
        runs: ``stationary_runs`` output.
        shore: Distance to shore, for the near-shore rule.
        resolution: H3 resolution of the cells.
        min_vessels: Fewest different vessels for a candidate cell.
        max_shore_km: Furthest a candidate's centre may lie from shore
            and still be an anchorage.

    Returns:
        pl.DataFrame: ``h3_cell``, centre ``lon`` and ``lat``, ``vessels``,
        ``runs``, ``shore_km`` and ``anchorage`` (false for candidates
        rejected as too far offshore), sorted by cell.
    """
    cells = [
        h3.latlng_to_cell(lat, lon, resolution)
        for lat, lon in zip(runs["lat"], runs["lon"])
    ]
    found = (
        runs.with_columns(h3_cell=pl.Series(cells, dtype=pl.String))
        .group_by("h3_cell")
        .agg(vessels=pl.col("vessel_id").n_unique(), runs=pl.len())
        .filter(pl.col("vessels") >= min_vessels)
        .sort("h3_cell")
    )
    centres = [h3.cell_to_latlng(cell) for cell in found["h3_cell"]]
    lat = np.array([c[0] for c in centres], dtype=float)
    lon = np.array([c[1] for c in centres], dtype=float)
    shore_km = shore.km(lon, lat)
    return found.with_columns(
        lon=pl.Series(lon, dtype=pl.Float64),
        lat=pl.Series(lat, dtype=pl.Float64),
        shore_km=pl.Series(shore_km, dtype=pl.Float64),
        anchorage=pl.Series(shore_km <= max_shore_km, dtype=pl.Boolean),
    ).select("h3_cell", "lon", "lat", "vessels", "runs", "shore_km", "anchorage")


class PortDistance:
    """Great-circle distance to the nearest anchorage.

    Attributes:
        projection: Flat projection used to shortlist anchorages.
    """

    def __init__(self, lon, lat, projection: LocalProjection):
        """Index the anchorages.

        Args:
            lon: Anchorage longitudes, degrees.
            lat: Anchorage latitudes, degrees.
            projection: Projection about the study area.
        """
        self.projection = projection
        self._lon = np.asarray(lon, dtype=float)
        self._lat = np.asarray(lat, dtype=float)
        self._tree = (
            cKDTree(np.column_stack(projection.forward(self._lon, self._lat)))
            if self._lon.size
            else None
        )

    def km(self, lon, lat) -> np.ndarray:
        """Distance from each position to the nearest anchorage.

        Args:
            lon: Longitudes, degrees.
            lat: Latitudes, degrees.

        Returns:
            np.ndarray: Kilometres; infinite when there are no anchorages.
        """
        lon = np.asarray(lon, dtype=float)
        lat = np.asarray(lat, dtype=float)
        if self._tree is None:
            return np.full(lon.size, np.inf)
        if lon.size == 0:
            return np.empty(0)
        k = min(NEAREST_PORTS, self._lon.size)
        _, index = self._tree.query(
            np.column_stack(self.projection.forward(lon, lat)), k=k
        )
        index = index.reshape(lon.size, k)
        km = haversine_km(
            lon[:, None], lat[:, None], self._lon[index], self._lat[index]
        )
        return km.min(axis=1)
