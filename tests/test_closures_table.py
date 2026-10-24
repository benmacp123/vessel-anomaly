"""Tests for the committed US geofence table at the real dates and places
where its rules change."""

import hashlib
import json
from datetime import date, datetime, timezone

import polars as pl
import pytest

from vessels.core.geofence import closure_hits
from vessels.io.blocks import BOX
from vessels.io.reference import CLOSURES_US, REFERENCE_US, read_closures


@pytest.fixture(name="rules", scope="module")
def fixture_rules():
    """Read the committed table once.

    Returns:
        list[Rule]: The table.
    """
    return read_closures()


def inside(rules, area_id):
    """A point well inside one rule's polygon.

    Args:
        rules: The table.
        area_id: The rule to sample.

    Returns:
        tuple: Longitude and latitude.
    """
    point = next(r for r in rules if r.area_id == area_id).geometry
    point = point.representative_point()
    return point.x, point.y


def hits(rules, area_id, when, gear="mobile_bottom"):
    """The closures and reasons for a position inside one rule's polygon.

    Args:
        rules: The table.
        area_id: Rule whose polygon the position is placed in.
        when: A date (taken at noon UTC) or a UTC datetime.
        gear: Gear family assumed.

    Returns:
        list[tuple]: ``(closure_id, reason)`` pairs, sorted.
    """
    if not isinstance(when, datetime):
        when = datetime(when.year, when.month, when.day, 12)
    lon, lat = inside(rules, area_id)
    frame = pl.DataFrame(
        {
            "timestamp": [when.replace(tzinfo=timezone.utc)],
            "lon": [lon],
            "lat": [lat],
        }
    )
    return sorted(
        closure_hits(frame, rules, gear).select("closure_id", "reason").iter_rows()
    )


GSC = [("gsc_hma", "closure")]


def test_checksum_matches_committed_sums():
    """The committed table is the one ``SHA256SUMS`` records."""
    sums = dict(
        reversed(line.split("  "))
        for line in (REFERENCE_US / "SHA256SUMS").read_text().splitlines()
    )
    digest = hashlib.sha256(CLOSURES_US.read_bytes()).hexdigest()
    assert sums["closures.geojson"] == digest


def test_every_row_is_cited_and_in_the_box(rules):
    """Each row names its source and status and lies in the study box."""
    features = json.loads(CLOSURES_US.read_text())["features"]
    for feature in features:
        props = feature["properties"]
        assert props["citation"] and props["status"] in ("observed", "provisional")
    min_lon, min_lat, max_lon, max_lat = BOX
    for rule in rules:
        x0, y0, x1, y1 = rule.geometry.bounds
        assert min_lon <= x0 and x1 <= max_lon and min_lat <= y0 and y1 <= max_lat


def test_closures_open_with_amendment_2(rules):
    """Every HMA and DHRA bans mobile gear from 2018-04-09, not before."""
    # The GSC HMA's own sample point lay in the 2018 clam exemption; its
    # never-exempt corner is checked separately.
    for area_id in ("wgom_hma", "eastern_maine_hma", "georges_bank_dhra"):
        assert not hits(rules, area_id, date(2018, 4, 8))
        assert hits(rules, area_id, date(2018, 4, 9))


def test_2018_clam_exemption_ends_after_9_april_2019(rules):
    """Fishing Rip lay in the 2018 exemption, which lapsed after 9 April
    2019, and reopened with the 2020 framework."""
    assert not hits(rules, "gsc_fishing_rip", date(2019, 4, 9))
    assert hits(rules, "gsc_fishing_rip", date(2019, 4, 10)) == GSC
    assert hits(rules, "gsc_fishing_rip", date(2020, 6, 17)) == GSC
    assert not hits(rules, "gsc_fishing_rip", date(2020, 6, 18))


def test_mbtg_corner_never_exempt(rules):
    """The 2018 corner (north-east of the HMA) closed on 2018-04-09 and
    was never exempt."""
    frame = pl.DataFrame(
        {
            "timestamp": [
                datetime(day.year, day.month, day.day, 12, tzinfo=timezone.utc)
                for day in (date(2018, 4, 8), date(2018, 4, 9), date(2018, 6, 1))
            ],
            "lon": [-69.6] * 3,
            "lat": [41.45] * 3,
        }
    )
    found = closure_hits(frame, rules)
    assert found["timestamp"].dt.date().to_list() == [
        date(2018, 4, 9),
        date(2018, 6, 1),
    ]
    assert found["reason"].to_list() == ["closure", "closure"]


def test_old_south_season(rules):
    """Old South is open 1 May to 31 October and closed otherwise."""
    assert hits(rules, "gsc_old_south", date(2024, 4, 30)) == GSC
    assert not hits(rules, "gsc_old_south", date(2024, 5, 1))
    assert not hits(rules, "gsc_old_south", date(2024, 10, 31))
    assert hits(rules, "gsc_old_south", date(2024, 11, 1)) == GSC


def test_old_south_opens_on_local_day(rules):
    """03:00 UTC on 1 May 2024 is still 30 April in Massachusetts."""
    assert hits(rules, "gsc_old_south", datetime(2024, 5, 1, 3)) == GSC
    assert not hits(rules, "gsc_old_south", datetime(2024, 5, 1, 4))


def test_efp_window(rules):
    """The permit area is exempt from 1 August to 5 November 2024."""
    assert hits(rules, "gsc_efp_davis_bank_east", date(2024, 7, 31)) == GSC
    assert not hits(rules, "gsc_efp_davis_bank_east", date(2024, 8, 1))
    assert not hits(rules, "gsc_efp_davis_bank_east", date(2024, 11, 5))
    assert hits(rules, "gsc_efp_davis_bank_east", date(2024, 11, 6)) == GSC


def test_shrimp_area_under_moratorium(rules):
    """The shrimp exemption is open in 2018 and suspended 2019-2024."""
    assert not hits(rules, "wgom_shrimp_ea", date(2018, 6, 1))
    suspended = [("wgom_hma", "suspended_exemption")]
    assert hits(rules, "wgom_shrimp_ea", date(2019, 1, 1)) == suspended
    assert hits(rules, "wgom_shrimp_ea", date(2024, 9, 1)) == suspended


def test_stellwagen_bans_fixed_gear_inside_wgom(rules):
    """Stellwagen sits inside the Western GOM HMA: towed gear breaks both,
    sink gillnet only Stellwagen."""
    day = date(2024, 5, 10)
    assert hits(rules, "stellwagen_dhra", day) == [
        ("stellwagen_dhra", "closure"),
        ("wgom_hma", "closure"),
    ]
    assert hits(rules, "stellwagen_dhra", day, "sink_gillnet") == [
        ("stellwagen_dhra", "closure")
    ]
