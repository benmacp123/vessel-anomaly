"""Tests for distance to shore and port and for anchorage finding, on
synthetic land and tracks."""

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest
from shapely.geometry import box

from vessels.core.distance import (
    PortDistance,
    ShoreDistance,
    anchorage_cells,
    stationary_runs,
)
from vessels.core.geo import LocalProjection, haversine_km

PROJECTION = LocalProjection(-68.0, 42.5)
# Land west of 70W between 41N and 45N.
LAND = box(-72.0, 41.0, -70.0, 45.0)
START = datetime(2024, 5, 6)


@pytest.fixture(name="shore", scope="module")
def fixture_shore():
    """Distance to the synthetic coast.

    Returns:
        ShoreDistance: Built once for the module.
    """
    return ShoreDistance(LAND, PROJECTION)


def track(vessel, points, step_minutes=60):
    """A vessel's positions at a fixed interval from ``START``.

    Args:
        vessel: Vessel id.
        points: ``(lon, lat)`` pairs in time order.
        step_minutes: Minutes between positions.

    Returns:
        pl.DataFrame: ``vessel_id``, ``timestamp``, ``lon``, ``lat``.
    """
    return pl.DataFrame(
        {
            "vessel_id": [vessel] * len(points),
            "timestamp": [
                START + timedelta(minutes=step_minutes * i) for i in range(len(points))
            ],
            "lon": [p[0] for p in points],
            "lat": [p[1] for p in points],
        }
    )


def test_projection_inverse_round_trips():
    """Inverse projection returns the original coordinates."""
    lon, lat = np.array([-71.4, -65.1]), np.array([40.2, 44.9])
    back = PROJECTION.inverse(*PROJECTION.forward(lon, lat))
    np.testing.assert_allclose(back, (lon, lat), atol=1e-12)


def test_shore_distance_is_great_circle_far_from_projection_centre(shore):
    """East of a meridian coast the distance is the exact great circle,
    though the projection's scale is 2% off at 44N."""
    km = shore.km([-69.0], [44.0])
    expected = haversine_km(-69.0, 44.0, -70.0, 44.0)
    assert km[0] == pytest.approx(expected, rel=1e-6)


def test_shore_distance_is_negative_on_land_and_zero_on_coast(shore):
    """Land reads negative, the coastline zero, the sea positive."""
    km = shore.km([-70.5, -70.0, -69.5], [43.0, 43.0, 43.0])
    assert km[0] < 0 and km[1] == pytest.approx(0, abs=1e-9) and km[2] > 0
    assert -km[0] == pytest.approx(km[2], rel=0.01)


def test_shore_distance_takes_nearest_edge(shore):
    """South of the land's corner the nearest coast is the south edge."""
    km = shore.km([-70.5], [40.8])
    assert km[0] == pytest.approx(haversine_km(-70.5, 40.8, -70.5, 41.0), rel=1e-6)


def test_shore_distance_handles_no_positions(shore):
    """An empty input gives an empty result."""
    assert shore.km([], []).size == 0


def test_stationary_run_needs_twelve_hours():
    """13 hours within 0.4 km is a run; 11 hours is not."""
    still = [(-69.0, 42.0 + 0.003 * (i % 2)) for i in range(14)]
    runs = stationary_runs(track("a", still), PROJECTION)
    assert runs["hours"].to_list() == [13.0]
    assert stationary_runs(track("a", still[:12]), PROJECTION).is_empty()


def test_drift_beyond_radius_splits_runs():
    """A move of 0.6 km ends one run and anchors the next at the new spot."""
    here = [(-69.0, 42.0)] * 13
    there = [(-69.0, 42.0 + 0.6 / 111.2)] * 13
    runs = stationary_runs(track("a", here + there), PROJECTION)
    assert runs.height == 2
    assert runs["start"][1] == START + timedelta(hours=13)


def test_runs_never_join_vessels_and_ignore_input_order():
    """Two vessels at one spot give one run each, in any row order."""
    spot = [(-69.0, 42.0)] * 13
    both = pl.concat([track("a", spot), track("b", spot)])
    runs = stationary_runs(both.reverse(), PROJECTION)
    assert runs["vessel_id"].to_list() == ["a", "b"]


def test_anchorage_counts_vessels_not_runs(shore):
    """Many runs by one vessel do not make an anchorage; two vessels do."""
    runs = pl.DataFrame(
        {
            "vessel_id": ["a", "a", "a", "b"],
            "lon": [-69.99] * 4,
            "lat": [43.0] * 4,
        }
    )
    cells = anchorage_cells(runs, shore, min_vessels=2)
    assert cells["vessels"].to_list() == [2] and cells["runs"].to_list() == [4]
    assert anchorage_cells(runs.head(3), shore, min_vessels=2).is_empty()


def test_offshore_candidate_is_not_an_anchorage(shore):
    """A busy stationary cell 40 km out is kept as a candidate only."""
    runs = pl.DataFrame(
        {"vessel_id": ["a", "b", "c"], "lon": [-69.5] * 3, "lat": [43.0] * 3}
    )
    cells = anchorage_cells(runs, shore)
    assert cells["shore_km"][0] > 30 and not cells["anchorage"][0]


def test_port_distance_is_exact_to_nearest_anchorage():
    """The nearest of several anchorages is measured on the great circle."""
    ports = PortDistance([-70.0, -66.0, -69.0], [43.0, 44.0, 41.0], PROJECTION)
    km = ports.km([-69.5, -66.1], [43.1, 44.0])
    np.testing.assert_allclose(
        km,
        [
            haversine_km(-69.5, 43.1, -70.0, 43.0),
            haversine_km(-66.1, 44.0, -66.0, 44.0),
        ],
    )


def test_port_distance_corrects_projection_order():
    """At 44N the projection stretches east-west by 2.6%, ranking an
    anchorage 10.2 km north before one truly 10.0 km east; the exact
    check still finds the east one."""
    east = -66.0 + 10.0 / haversine_km(-66.0, 44.0, -65.0, 44.0)
    north = 44.0 + 10.2 / haversine_km(-66.0, 44.0, -66.0, 45.0)
    ports = PortDistance([east, -66.0], [44.0, north], PROJECTION)
    km = ports.km([-66.0], [44.0])
    assert km[0] == pytest.approx(haversine_km(-66.0, 44.0, east, 44.0), rel=1e-9)
    assert km[0] < 10.1


def test_port_distance_without_anchorages_is_infinite():
    """No anchorages means no position is near a port."""
    assert np.isinf(PortDistance([], [], PROJECTION).km([-69.0], [42.0])).all()
