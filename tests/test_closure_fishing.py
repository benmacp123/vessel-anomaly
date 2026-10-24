"""Tests for the closure-fishing detector: towing runs inside a closure in
force become events; transits, other dates, exemptions and brief
incursions do not."""

from datetime import date

from shapely.geometry import box

from synthetic import Track, frame
from vessels.core.cleaning import clean
from vessels.core.closure_fishing import EVENT_SCHEMA, closure_events
from vessels.core.geofence import Rule
from vessels.core.segment import segment

# A closure 0.2 degrees square (about 16 x 22 km), west edge at -69.0.
CLOSURE = Rule(
    "hma",
    "closure",
    ("mobile_bottom",),
    date(2018, 4, 9),
    None,
    box(-69, 42, -68.8, 42.2),
)
INSIDE = (-68.95, 42.1)


def detect(*tracks, rules=(CLOSURE,)):
    """Clean, segment and run the detector.

    Args:
        *tracks: ``Track`` objects.
        rules: The geofence table.

    Returns:
        pl.DataFrame: ``closure_events`` output.
    """
    return closure_events(segment(clean(frame(*tracks)).frame), list(rules))


def test_tow_inside_is_an_event():
    """An hour's towing inside the closure is one apparent event."""
    found = detect(Track("v", *INSIDE).leg(60, 3.5))
    assert found.height == 1
    row = found.row(0, named=True)
    assert (row["closure_id"], row["reason"], row["vessel_id"]) == (
        "hma",
        "closure",
        "v",
    )
    assert row["positions"] == 31
    assert found.schema == EVENT_SCHEMA


def test_transit_is_not_towing():
    """Steaming through the closure at 9 kn raises nothing."""
    assert detect(Track("v", -69.05, 42.1).leg(120, 9.0)).is_empty()


def test_tow_outside_raises_nothing():
    """Towing just west of the closure raises nothing."""
    assert detect(Track("v", -69.2, 42.1).leg(60, 3.5)).is_empty()


def test_tow_before_closure_raises_nothing():
    """The same tow before the closure's first day raises nothing."""
    later = Rule(
        "hma", "closure", ("mobile_bottom",), date(2025, 1, 1), None, CLOSURE.geometry
    )
    assert detect(Track("v", *INSIDE).leg(60, 3.5), rules=(later,)).is_empty()


def test_open_exemption_lifts_the_ban():
    """Towing inside an open dredge exemption is not an event."""
    exemption = Rule(
        "clam",
        "exemption",
        ("hydraulic_clam_dredge",),
        date(2018, 4, 9),
        None,
        box(-69, 42, -68.9, 42.2),
        parent="hma",
    )
    assert detect(
        Track("v", *INSIDE).leg(60, 3.5), rules=(CLOSURE, exemption)
    ).is_empty()


def test_brief_incursion_is_dropped():
    """Towing across the closure edge for about 13 minutes is too short."""
    # Starts 1.5 km west of the edge, tows east 3 km at 3.5 kn (about 28
    # minutes), so about 13 minutes fall inside.
    found = detect(Track("v", -69.018, 42.1).leg(28, 3.5))
    assert found.is_empty()


def test_brief_exit_merges():
    """Towing out of the closure for about 14 minutes and back is one event."""
    # Crosses the west edge after about 23 minutes, so about 7 minutes
    # out and 7 back.
    track = Track("v", -68.97, 42.1).leg(30, 3.5, cog=270).leg(40, 3.5, cog=90)
    found = detect(track)
    assert found.height == 1
    assert found["hours"][0] > 1


def test_event_ids_are_stable():
    """Running twice gives the same event IDs."""
    track = Track("v", *INSIDE).leg(60, 3.5)
    assert detect(track)["event_id"].to_list() == detect(track)["event_id"].to_list()


def test_no_hits_gives_empty_schema():
    """A block with no hits returns an empty frame in the event schema."""
    found = detect(Track("v", -70, 41).leg(60, 3.5))
    assert found.is_empty() and found.schema == EVENT_SCHEMA
