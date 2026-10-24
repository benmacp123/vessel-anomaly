"""Tests for the loitering detector: slow hours far from shore, grouped as
GFW groups them."""

from datetime import timedelta

import pytest
from shapely.geometry import box

from synthetic import T0, Track, frame
from vessels.core.cleaning import clean
from vessels.core.distance import ShoreDistance
from vessels.core.geo import LocalProjection
from vessels.core.loitering import LOITER_SCHEMA, LoiterRules, loiter_events
from vessels.core.segment import segment

# Land west of -70.5; a vessel at -68 is about 110 nm offshore.
SHORE = ShoreDistance(box(-72, 40, -70.5, 44), LocalProjection(-69.0, 42.0))
OFFSHORE = (-68.0, 42.0)


def detect(*tracks, rules=LoiterRules()):
    """Clean, segment and detect loitering.

    Args:
        *tracks: ``Track`` objects.
        rules: Thresholds.

    Returns:
        pl.DataFrame: ``loiter_events`` output.
    """
    return loiter_events(segment(clean(frame(*tracks)).frame), SHORE, rules)


def test_drifting_far_offshore():
    """Nine hours drifting at 0.5 kn far offshore is one event."""
    found = detect(Track("v", *OFFSHORE).leg(9 * 60, 0.5))
    assert found.height == 1
    row = found.row(0, named=True)
    assert row["hours"] == pytest.approx(9.0)
    assert row["avg_kn"] == pytest.approx(0.5, rel=1e-3)
    assert row["avg_shore_nm"] > 100
    assert found.schema == LOITER_SCHEMA


def test_too_short():
    """Seven hours slow is under the 8-hour headline but passes at 4."""
    track = Track("v", *OFFSHORE).leg(7 * 60, 0.5)
    assert detect(track).is_empty()
    assert detect(track, rules=LoiterRules(min_hours=4)).height == 1


def test_near_shore():
    """Drifting about 9 nm from shore is not loitering."""
    assert detect(Track("v", -70.3, 42.0).leg(9 * 60, 0.5)).is_empty()


def test_transit():
    """Steaming at 10 kn is never slow."""
    assert detect(Track("v", *OFFSHORE).leg(9 * 60, 10.0)).is_empty()


def test_silent_hour_does_not_break():
    """Two hours without positions inside a slow run keep one event."""
    track = Track("v", *OFFSHORE).leg(5 * 60, 0.5).silence(2, 0.5).leg(5 * 60, 0.5)
    found = detect(track)
    assert found.height == 1
    assert found["hours"][0] == pytest.approx(12.0)


def test_fast_hours_break():
    """Steaming for 90 minutes between two slow spells leaves two events."""
    track = Track("v", *OFFSHORE).leg(5 * 60, 0.5).leg(90, 10.0).leg(5 * 60, 0.5)
    assert detect(track).is_empty()
    assert detect(track, rules=LoiterRules(min_hours=4)).height == 2


def test_starts_at_lead_in_position():
    """The event starts at the last position of the hour before the first
    slow hour."""
    track = Track("v", *OFFSHORE).leg(120, 10.0).leg(9 * 60, 0.5)
    found = detect(track)
    assert found["start"].to_list() == [T0 + timedelta(minutes=118)]
    # Duration runs from that position, without the step leading to it.
    assert found["hours"][0] == pytest.approx(9 + 2 / 60)


def test_runs_stay_in_their_segment():
    """Two vessels drifting one after the other give one event each."""
    first = Track("a", *OFFSHORE).leg(9 * 60, 0.5)
    second = Track("b", -67.0, 43.0, first.time + timedelta(minutes=30))
    assert detect(first, second.leg(9 * 60, 0.5)).height == 2
