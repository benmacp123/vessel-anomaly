"""Tests for the encounter detector: GFW's resampling, adjacency, runs,
merge and filters, near-miss negatives and symmetry between the vessels."""

import math
from datetime import timedelta

import polars as pl
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from shapely.geometry import box

from synthetic import T0, Track, frame
from vessels.core.cleaning import clean
from vessels.core.distance import PortDistance, ShoreDistance
from vessels.core.encounters import (
    ENCOUNTER_SCHEMA,
    EncounterRules,
    adjacent,
    encounter_events,
    encounter_groups,
    resample,
)
from vessels.core.geo import LocalProjection
from vessels.core.segment import segment

PROJECTION = LocalProjection(-69.0, 42.0)
# Land west of -70.5 and one anchorage far from the meeting spot.
SHORE = ShoreDistance(box(-72, 40, -70.5, 44), PROJECTION)
PORTS = PortDistance([-71.0], [41.0], PROJECTION)
SPOT = (-67.5, 42.5)
KM_PER_DEG_LAT = 111.195
RULES = EncounterRules()


def segmented(*tracks) -> pl.DataFrame:
    """Clean and segment tracks.

    Args:
        *tracks: ``Track`` objects.

    Returns:
        pl.DataFrame: ``segment.segment`` output.
    """
    return segment(clean(frame(*tracks)).frame)


def groups(*tracks, ports=PORTS) -> pl.DataFrame:
    """Every merged encounter among the tracks.

    Args:
        *tracks: ``Track`` objects.
        ports: Anchorages.

    Returns:
        pl.DataFrame: ``encounter_groups`` output.
    """
    return encounter_groups(segmented(*tracks), ports, SHORE)


def anchored(hours=8.0, name="a", start=T0) -> Track:
    """A vessel stopped at ``SPOT``, reporting every five minutes.

    Args:
        hours: How long it reports.
        name: Vessel ID.
        start: First report.

    Returns:
        Track: The vessel.
    """
    return Track(name, *SPOT, start=start).leg(hours * 60, 0.0, step=5)


def visitor(visits, km=0.3, name="b", start=T0 + timedelta(hours=1)) -> Track:
    """A vessel that waits 5 km north of ``SPOT``, runs in to ``km`` north
    of it at 10 kn, stops, and runs out again, once per visit.

    Args:
        visits: ``(minutes_stopped, minutes_away_after)`` per visit.
        km: Distance north of ``SPOT`` while stopped.
        name: Vessel ID.
        start: First report.

    Returns:
        Track: The vessel.
    """
    run_kn = (5 - km) / 1.852 / 0.25
    track = Track(name, SPOT[0], SPOT[1] + 5 / KM_PER_DEG_LAT, start=start)
    track.leg(10, 0.0, step=5)
    for stopped, away in visits:
        track.leg(15, run_kn, cog=180, step=1).leg(stopped, 0.0, step=5)
        track.leg(15, run_kn, cog=0, step=1).leg(away, 0.0, step=5)
    return track


def test_meeting_is_an_encounter():
    """Two vessels stopped 300 m apart for three hours meet once."""
    found = encounter_events(segmented(anchored(), visitor([(180, 60)])), PORTS, SHORE)
    assert found.schema == ENCOUNTER_SCHEMA
    assert found.height == 1
    row = found.row(0, named=True)
    assert (row["vessel_id"], row["partner_id"]) == ("a", "b")
    assert row["hours"] == pytest.approx(3.0, abs=0.35)
    assert row["median_km"] == pytest.approx(0.3, abs=0.01)
    assert row["median_kn"] == pytest.approx(0.0, abs=1e-6)
    assert row["runs"] == 1


def test_near_miss_600_m():
    """Stopped 600 m apart for 2.5 hours is never adjacent."""
    assert groups(anchored(), visitor([(150, 60)], km=0.6)).is_empty()


def test_near_miss_short():
    """400 m apart for 1.5 hours is a raw encounter but too short."""
    found = groups(anchored(), visitor([(90, 60)], km=0.4))
    assert found.height == 1
    assert found["hours"][0] < 2.0
    assert not found["passes"][0]


def test_two_hours_is_not_enough():
    """An encounter must last more than two hours, not exactly two."""
    exact = groups(anchored(), visitor([(130, 60)]))
    assert exact["hours"].to_list() == [pytest.approx(2.0)]
    assert not exact["passes"][0]
    longer = groups(anchored(), visitor([(140, 60)]))
    assert longer["passes"].to_list() == [True]


def test_near_port():
    """A meeting within 10 km of an anchorage is dropped."""
    port = PortDistance([SPOT[0]], [SPOT[1] - 0.05], PROJECTION)
    assert groups(anchored(), visitor([(180, 60)]), ports=port).is_empty()


def test_on_land():
    """A meeting whose mean position is on land is dropped."""
    shore = ShoreDistance(box(-68, 42, -67, 43), PROJECTION)
    found = encounter_groups(segmented(anchored(), visitor([(180, 60)])), PORTS, shore)
    assert found.is_empty()


def test_pair_trawling_too_fast():
    """Two vessels towing side by side at 3 kn are close but not slow."""
    a = Track("a", *SPOT).leg(4 * 60, 3.0, step=3)
    b = Track("b", SPOT[0], SPOT[1] + 0.3 / KM_PER_DEG_LAT).leg(4 * 60, 3.0, step=4)
    found = groups(a, b)
    assert found.height == 1
    assert found["median_kn"][0] == pytest.approx(3.0, rel=0.01)
    assert not found["passes"][0]


def test_positions_are_midpoints():
    """An encounter's start and end positions lie midway between the
    vessels."""
    east = 0.3 / KM_PER_DEG_LAT / math.cos(math.radians(SPOT[1]))
    a = Track("a", *SPOT).leg(4 * 60, 0.5, cog=0, step=5)
    b = Track("b", SPOT[0] + east, SPOT[1]).leg(4 * 60, 0.5, cog=0, step=5)
    row = groups(a, b).row(0, named=True)
    assert row["start_lon"] == pytest.approx(SPOT[0] + east / 2, abs=1e-9)
    assert row["end_lon"] == pytest.approx(SPOT[0] + east / 2, abs=1e-9)
    assert row["mean_lon"] == pytest.approx(SPOT[0] + east / 2, abs=1e-9)


def test_point_counts_follow_each_vessel():
    """Each side's point count reflects its own reporting rate."""
    a = Track("a", *SPOT).leg(4 * 60, 0.5, step=5)
    b = Track("b", SPOT[0], SPOT[1] + 0.2 / KM_PER_DEG_LAT).leg(4 * 60, 0.5, step=20)
    row = groups(a, b).row(0, named=True)
    assert row["vessel_points"] > 1.5 * row["partner_points"] > 0


@pytest.mark.parametrize("away, expected", [(150, 1), (300, 2)])
def test_merge_within_four_hours(away, expected):
    """Meetings under four hours apart merge; further apart they do not."""
    found = groups(anchored(14), visitor([(150, away), (150, 60)]))
    assert found.height == expected
    assert found["runs"].sum() == 2


def test_midnight_splits_raw_runs():
    """A meeting across midnight is two raw runs, merged into one."""
    found = groups(
        anchored(30), visitor([(240, 60)], start=T0 + timedelta(hours=21, minutes=50))
    )
    assert found.height == 1
    assert found["runs"][0] == 2
    assert found["start"][0] < T0 + timedelta(days=1) < found["end"][0]


def test_short_part_before_midnight_is_lost():
    """GFW drops a raw run under an hour, so 40 minutes before midnight do
    not count."""
    found = groups(
        anchored(30), visitor([(200, 60)], start=T0 + timedelta(hours=22, minutes=55))
    )
    assert found["runs"].to_list() == [1]
    assert found["start"][0] == T0 + timedelta(days=1)


def test_same_vessel_segments_not_paired():
    """Two segments of one vessel ID side by side are never a pair."""
    slices = pl.DataFrame(
        {
            "segment": ["s1", "s2"],
            "vessel_id": ["a", "a"],
            "transceiver": ["A", "A"],
            "slice": [0, 0],
            "lon": [SPOT[0], SPOT[0]],
            "lat": [SPOT[1], SPOT[1] + 0.001],
            "density": [1.0, 1.0],
        }
    )
    assert adjacent(slices, PROJECTION, RULES).is_empty()
    other = slices.with_columns(vessel_id=pl.Series(["a", "b"]))
    assert adjacent(other, PROJECTION, RULES).height == 1


def minutes(slices: pl.DataFrame, start=T0) -> list[int]:
    """Grid times as minutes after ``start``.

    Args:
        slices: ``resample`` output.
        start: Reference time.

    Returns:
        list[int]: Minutes.
    """
    zero = int(start.timestamp() * 1_000_000)
    return [(s - zero) // 60_000_000 for s in slices["slice"]]


def test_resample_interpolates_and_extends():
    """Positions every 30 minutes fill the grid between them, and the track
    is dead-reckoned an hour past its last position."""
    track = Track("v", *SPOT).leg(60, 6.0, step=30)
    slices = resample(segmented(track), RULES)
    assert minutes(slices) == list(range(0, 130, 10))
    lon = slices["lon"].to_list()
    steps = [b - a for a, b in zip(lon, lon[1:])]
    assert steps == pytest.approx([steps[0]] * len(steps), rel=1e-6)
    # A third of a point per grid time between reports, then thinning
    # with distance from the last one.
    assert slices["density"].to_list() == [0.33] * 7 + [1.0, 0.5, 0.33, 0.25, 0.2, 0.17]


def test_resample_skips_hour_gaps():
    """A gap of exactly an hour is not interpolated, not even at the
    position that ends it."""
    track = Track("v", *SPOT).leg(60, 6.0, step=60).leg(30, 6.0, step=30)
    slices = resample(segmented(track), RULES)
    assert minutes(slices) == [70, 80, 90] + list(range(100, 160, 10))


def test_resample_stops_at_midnight():
    """Dead-reckoning before the first position and after the last stays
    inside the UTC day."""
    start = T0 + timedelta(hours=22)
    track = Track("v", *SPOT, start=start).leg(90, 6.0, step=30)
    slices = resample(segmented(track), RULES)
    assert minutes(slices, start) == list(range(-60, 120, 10))


def test_resample_needs_speed_and_course():
    """An end position with no speed is not dead-reckoned."""
    data = segmented(Track("v", *SPOT).leg(60, 6.0, step=30))
    slices = resample(data.with_columns(sog=pl.lit(None, pl.Float64)), RULES)
    assert minutes(slices) == list(range(0, 70, 10))


def renamed(found: pl.DataFrame) -> pl.DataFrame:
    """Encounters with the vessels' roles swapped back after renaming.

    Args:
        found: ``encounter_groups`` output for renamed vessels.

    Returns:
        pl.DataFrame: The same encounters keyed as before renaming.
    """
    names = {"a": "b", "b": "a"}
    swapped = found.rename(
        {
            "vessel_id": "partner_id",
            "partner_id": "vessel_id",
            "transceiver": "partner_transceiver",
            "partner_transceiver": "transceiver",
            "vessel_points": "partner_points",
            "partner_points": "vessel_points",
        }
    )
    return swapped.with_columns(
        pl.col("vessel_id", "partner_id").replace_strict(names)
    ).select(found.columns)


@settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    km=st.floats(0.0, 0.7),
    east=st.floats(-0.4, 0.4),
    kn=st.tuples(st.floats(0.0, 3.0), st.floats(0.0, 3.0)),
    steps=st.tuples(st.integers(1, 15), st.integers(1, 15)),
    hours=st.tuples(st.floats(1.0, 6.0), st.floats(1.0, 6.0)),
    delay=st.integers(0, 180),
    cog=st.floats(0.0, 360.0),
    transceivers=st.sampled_from(["AA", "AB", "BA"]),
)
def test_symmetric(km, east, kn, steps, hours, delay, cog, transceivers):
    """Swapping which vessel is which changes nothing but the roles."""

    def pair(first, second):
        """The two tracks under the given names.

        Args:
            first: Name of the vessel at ``SPOT``.
            second: Name of the vessel offset north and east.

        Returns:
            pl.DataFrame: Encounter groups.
        """
        a = Track(first, *SPOT, transceiver=transceivers[0])
        a.leg(hours[0] * 60, kn[0], cog=cog, step=steps[0])
        b = Track(
            second,
            SPOT[0] + east / KM_PER_DEG_LAT / math.cos(math.radians(SPOT[1])),
            SPOT[1] + km / KM_PER_DEG_LAT,
            start=T0 + timedelta(minutes=delay),
            transceiver=transceivers[1],
        )
        b.leg(hours[1] * 60, kn[1], cog=cog, step=steps[1])
        return groups(a, b).drop("event_id")

    forward = pair("a", "b")
    backward = renamed(pair("b", "a"))
    assert _close(forward, backward)


def _close(left: pl.DataFrame, right: pl.DataFrame) -> bool:
    """Whether two frames match up to the last bits of floats.

    Args:
        left: A frame.
        right: Another with the same columns.

    Returns:
        bool: True if every column matches.
    """
    if left.shape != right.shape:
        return False
    for name in left.columns:
        a, b = left[name].to_list(), right[name].to_list()
        if left.schema[name] == pl.Float64:
            if a != pytest.approx(b, rel=1e-12, abs=1e-12):
                return False
        elif a != b:
            return False
    return True
