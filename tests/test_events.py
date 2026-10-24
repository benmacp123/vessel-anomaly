"""Tests for GFW's event rules: bursts, splits, merges and the filters that
keep lasting events."""

from datetime import timedelta

import polars as pl
import pytest

from synthetic import Track, frame
from vessels.core.events import EventRules, flag_events, steps

LON, LAT = -69.0, 42.0


def events(*tracks, flag=pl.col("sog").is_between(2.5, 5.5)):
    """Run the event rules with each vessel as one track.

    Args:
        *tracks: ``Track`` objects.
        flag: Expression for the flagged state; towing speed by default.

    Returns:
        pl.DataFrame: ``flag_events`` output.
    """
    data = frame(*tracks).pipe(steps, "vessel_id").with_columns(_flag=flag)
    return flag_events(data, "_flag", "vessel_id")


def test_one_tow_is_one_event():
    """A steady hour of towing is one event of the right length."""
    found = events(Track("v", LON, LAT).leg(60, 3.0))
    assert found.height == 1
    assert found["positions"][0] == 31
    assert found["hours"][0] == pytest.approx(1.0)
    assert found["distance_km"][0] == pytest.approx(3.0 * 1.852, rel=1e-3)


def test_short_tow_is_dropped():
    """A 20-minute tow is not longer than the minimum and is dropped."""
    assert events(Track("v", LON, LAT).leg(20, 3.0)).is_empty()
    assert events(Track("v", LON, LAT).leg(22, 3.0)).height == 1


def test_brief_stop_merges():
    """A 30-minute stop between two tows joins them into one event, and
    the stopped positions are not counted."""
    track = Track("v", LON, LAT).leg(40, 3.0).leg(30, 0.5).leg(40, 3.0)
    found = events(track)
    assert found.height == 1
    assert found["positions"][0] == 21 + 20
    assert found["hours"][0] == pytest.approx(110 / 60)


def test_long_stop_splits():
    """A stop longer than an hour leaves two events."""
    track = Track("v", LON, LAT).leg(40, 3.0).leg(70, 0.5).leg(40, 3.0)
    assert events(track).height == 2


def test_merge_distance_is_from_the_last_burst_end():
    """A two-hour tow, a 10-minute pause and more towing merge, although
    the new burst starts 11 km from where the first began."""
    track = Track("v", LON, LAT).leg(120, 3.0).leg(10, 0.5).leg(40, 3.0)
    assert events(track).height == 1


def test_steaming_away_splits():
    """Steaming 6 km between tows, under an hour, keeps two events."""
    track = Track("v", LON, LAT).leg(40, 3.0).leg(20, 18.0).leg(40, 3.0)
    assert events(track).height == 2


def test_silence_over_two_hours_splits():
    """Flagged positions three hours apart are not one burst."""
    track = Track("v", LON, LAT).leg(40, 3.0).silence(3, sog=0.1).leg(40, 3.0)
    assert events(track).height == 2


def test_few_positions_dropped():
    """Five positions over half an hour are too few; six are enough."""
    assert events(Track("v", LON, LAT).leg(30, 3.0, step=7.5)).is_empty()
    assert events(Track("v", LON, LAT).leg(30, 3.0, step=6.0)).height == 1


def test_short_distance_dropped():
    """A flagged half hour covering under 0.5 km is dropped."""
    slow = events(Track("v", LON, LAT).leg(30, 0.5), flag=pl.col("sog") > 0)
    assert slow.is_empty()


def test_fast_average_dropped():
    """A flagged run averaging 10 kn or more is dropped."""
    fast = events(Track("v", LON, LAT).leg(40, 12.0), flag=pl.col("sog") > 0)
    assert fast.is_empty()


def test_tracks_never_merge():
    """A vessel starting to tow where and just after another stopped gets
    its own event."""
    first = Track("a", LON, LAT).leg(40, 3.0)
    second = Track("b", first.lon, first.lat, first.time + timedelta(minutes=10))
    found = events(first, second.leg(40, 3.0))
    assert sorted(found["vessel_id"]) == ["a", "b"]
    assert found["event"].n_unique() == 2


def test_lead_in_step_not_counted():
    """Distance starts at the event's first position, not the step that
    led into it."""
    found = events(Track("v", LON, LAT).leg(20, 8.0).leg(60, 3.0))
    assert found["distance_km"][0] == pytest.approx(29 * 0.1 * 1.852, rel=1e-3)


def test_custom_rules():
    """Thresholds come from the rules passed in."""
    track = Track("v", LON, LAT).leg(40, 3.0).leg(30, 0.5).leg(40, 3.0)
    data = frame(track).pipe(steps, "vessel_id")
    data = data.with_columns(_flag=pl.col("sog") > 1)
    strict = EventRules(merge_hours=0.25)
    assert flag_events(data, "_flag", "vessel_id", strict).height == 2
