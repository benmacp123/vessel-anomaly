"""Small-area geometry: a local flat projection and great-circle distance.

The study regions are a few hundred kilometres across, so a local
equirectangular projection about a fixed reference latitude is accurate
enough for buffers and neighbour searches: over the US box (40-45N about
42.5N) east-west distances are off by at most about 4%.
"""

import numpy as np
import shapely
from shapely.geometry.base import BaseGeometry

EARTH_RADIUS_KM = 6371.0088
KM_PER_NM = 1.852


class LocalProjection:
    """Equirectangular projection to kilometres about a reference point.

    Attributes:
        lon0: Reference longitude, degrees.
        lat0: Reference latitude, degrees; east-west scale is exact here.
    """

    def __init__(self, lon0: float, lat0: float):
        """Fix the reference point.

        Args:
            lon0: Reference longitude, degrees.
            lat0: Reference latitude, degrees.
        """
        self.lon0 = lon0
        self.lat0 = lat0
        self._km_per_deg_lat = np.radians(EARTH_RADIUS_KM)
        self._km_per_deg_lon = self._km_per_deg_lat * np.cos(np.radians(lat0))

    def forward(self, lon, lat) -> tuple[np.ndarray, np.ndarray]:
        """Project longitudes and latitudes to kilometres east and north.

        Args:
            lon: Longitudes, degrees (scalar or array).
            lat: Latitudes, degrees (scalar or array).

        Returns:
            tuple[np.ndarray, np.ndarray]: x and y in kilometres.
        """
        x = (np.asarray(lon, dtype=float) - self.lon0) * self._km_per_deg_lon
        y = (np.asarray(lat, dtype=float) - self.lat0) * self._km_per_deg_lat
        return x, y

    def inverse(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        """Turn kilometres east and north back into longitudes and latitudes.

        Args:
            x: Kilometres east of the reference point (scalar or array).
            y: Kilometres north of the reference point (scalar or array).

        Returns:
            tuple[np.ndarray, np.ndarray]: Longitudes and latitudes, degrees.
        """
        lon = np.asarray(x, dtype=float) / self._km_per_deg_lon + self.lon0
        lat = np.asarray(y, dtype=float) / self._km_per_deg_lat + self.lat0
        return lon, lat

    def geometry(self, geom: BaseGeometry) -> BaseGeometry:
        """Project a shapely geometry given in longitude and latitude.

        Args:
            geom: Geometry in degrees.

        Returns:
            BaseGeometry: The same geometry in kilometres.
        """
        return shapely.transform(geom, lambda xy: np.column_stack(self.forward(*xy.T)))


def haversine_km(lon1, lat1, lon2, lat2) -> np.ndarray:
    """Great-circle distance between point pairs.

    Args:
        lon1: Start longitudes, degrees.
        lat1: Start latitudes, degrees.
        lon2: End longitudes, degrees.
        lat2: End latitudes, degrees.

    Returns:
        np.ndarray: Distances in kilometres.
    """
    lon1, lat1, lon2, lat2 = (
        np.radians(np.asarray(v, dtype=float)) for v in (lon1, lat1, lon2, lat2)
    )
    a = (
        np.sin((lat2 - lat1) / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    )
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))
