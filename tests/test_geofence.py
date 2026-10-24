"""Tests for dated, gear-specific geofences on small synthetic tables."""

from datetime import date, datetime, timezone

import polars as pl
import pytest
from shapely.geometry import box

from vessels.core.geofence import Rule, check_table, closure_hits, in_force

OPEN = date(2018, 4, 9)


def square(x0, y0, x1, y1):
    """A box in degrees, for short rule definitions.

    Args:
        x0: West edge.
        y0: South edge.
        x1: East edge.
        y1: North edge.

    Returns:
        Polygon: The box.
    """
    return box(x0, y0, x1, y1)


@pytest.fixture(name="table")
def fixture_table():
    """A closure with a permanent, a dated, a seasonal and a suspended
    exemption, plus a second closure that bans fixed gear.

    Returns:
        list[Rule]: The table.
    """
    return [
        Rule("hma", "closure", ("mobile_bottom",), OPEN, None, square(0, 0, 10, 10)),
        Rule(
            "shrimp",
            "exemption",
            ("shrimp_trawl",),
            OPEN,
            None,
            square(0, 0, 2, 2),
            parent="hma",
        ),
        Rule(
            "moratorium",
            "suspension",
            ("shrimp_trawl",),
            date(2019, 1, 1),
            date(2022, 1, 1),
            square(0, 0, 2, 2),
            parent="shrimp",
        ),
        Rule(
            "season",
            "exemption",
            ("hydraulic_clam_dredge",),
            date(2020, 6, 18),
            None,
            square(4, 4, 6, 6),
            parent="hma",
            season=("05-01", "10-31"),
        ),
        Rule(
            "temporary",
            "exemption",
            ("hydraulic_clam_dredge",),
            OPEN,
            date(2019, 4, 10),
            square(7, 7, 9, 9),
            parent="hma",
        ),
        Rule(
            "dhra",
            "closure",
            ("mobile_bottom", "sink_gillnet"),
            OPEN,
            None,
            square(0, 0, 1, 1),
        ),
    ]


def at(lon, lat, when):
    """A one-position frame.

    Args:
        lon: Longitude.
        lat: Latitude.
        when: UTC datetime, or a date taken at noon UTC.

    Returns:
        pl.DataFrame: ``vessel_id``, ``timestamp`` (UTC), ``lon``, ``lat``.
    """
    if isinstance(when, date) and not isinstance(when, datetime):
        when = datetime(when.year, when.month, when.day, 12)
    return pl.DataFrame(
        {
            "vessel_id": ["v"],
            "timestamp": [when.replace(tzinfo=timezone.utc)],
            "lon": [lon],
            "lat": [lat],
        }
    )


def reasons(table, lon, lat, when, gear="mobile_bottom"):
    """The closures and reasons one position breaks.

    Args:
        table: The rules.
        lon: Longitude.
        lat: Latitude.
        when: As for ``at``.
        gear: Gear family assumed.

    Returns:
        list[tuple]: ``(closure_id, reason)`` pairs, sorted.
    """
    hits = closure_hits(at(lon, lat, when), table, gear)
    return sorted(hits.select("closure_id", "reason").iter_rows())


def test_in_force_either_side_of_dates(table):
    """valid_from is included and valid_to excluded."""
    temporary = table[4]
    assert not in_force(temporary, date(2018, 4, 8))
    assert in_force(temporary, date(2018, 4, 9))
    assert in_force(temporary, date(2019, 4, 9))
    assert not in_force(temporary, date(2019, 4, 10))


def test_in_force_season_edges(table):
    """A May-October season includes both ends and nothing outside."""
    season = table[3]
    assert not in_force(season, date(2024, 4, 30))
    assert in_force(season, date(2024, 5, 1))
    assert in_force(season, date(2024, 10, 31))
    assert not in_force(season, date(2024, 11, 1))


def test_in_force_season_wrapping_new_year(table):
    """A November-April season wraps the year end."""
    wrapped = Rule(
        "w",
        "exemption",
        ("mussel_dredge",),
        OPEN,
        None,
        square(0, 0, 1, 1),
        parent="hma",
        season=("11-01", "04-30"),
    )
    assert in_force(wrapped, date(2024, 1, 15))
    assert in_force(wrapped, date(2024, 11, 1))
    assert not in_force(wrapped, date(2024, 7, 1))
    assert table  # the fixture's parent closure exists for this rule


def test_outside_or_before_closure_is_not_hit(table):
    """Nothing outside the polygon or before the closure opens is hit."""
    assert not reasons(table, 11, 11, date(2024, 5, 10))
    assert not reasons(table, 3, 3, date(2018, 4, 8))
    assert reasons(table, 3, 3, date(2018, 4, 9)) == [("hma", "closure")]


def test_boundary_counts_as_inside(table):
    """A position exactly on a closure line is inside it."""
    assert reasons(table, 10, 3, date(2024, 5, 10)) == [("hma", "closure")]


def test_dated_exemption_ends(table):
    """The temporary exemption lifts the ban through its last day only."""
    assert not reasons(table, 8, 8, date(2019, 4, 9))
    assert reasons(table, 8, 8, date(2019, 4, 10)) == [("hma", "closure")]


def test_seasonal_exemption_starts_on_its_date(table):
    """The seasonal exemption does nothing before its valid_from."""
    assert reasons(table, 5, 5, date(2020, 6, 17)) == [("hma", "closure")]
    assert not reasons(table, 5, 5, date(2020, 6, 18))


def test_seasonal_exemption_uses_local_day(table):
    """03:00 UTC on 1 May is still 30 April in New York, so the season is
    shut; the same instant read as UTC falls in the season."""
    when = datetime(2024, 5, 1, 3)
    assert reasons(table, 5, 5, when) == [("hma", "closure")]
    utc_table = [
        Rule(**{**r.__dict__, "timezone": "UTC"}) if r.area_id == "season" else r
        for r in table
    ]
    assert not reasons(utc_table, 5, 5, when)


def test_naive_timestamps_are_read_as_utc(table):
    """A frame without a time zone gives the same answer as UTC."""
    frame = at(5, 5, datetime(2024, 5, 1, 3)).with_columns(
        pl.col("timestamp").dt.replace_time_zone(None)
    )
    assert closure_hits(frame, table)["closure_id"].to_list() == ["hma"]


def test_suspension_switches_exemption_off(table):
    """Inside a suspended exemption the ban returns, with its own reason."""
    assert not reasons(table, 1.5, 1.5, date(2018, 12, 31))
    assert reasons(table, 1.5, 1.5, date(2019, 1, 1)) == [
        ("hma", "suspended_exemption")
    ]
    assert not reasons(table, 1.5, 1.5, date(2022, 1, 1))


def test_overlapping_closures_each_reported(table):
    """A position in two closures that both ban the gear appears twice;
    the exemption of one closure does not lift the other."""
    assert reasons(table, 0.5, 0.5, date(2018, 6, 1)) == [("dhra", "closure")]
    assert reasons(table, 0.5, 0.5, date(2020, 6, 1)) == [
        ("dhra", "closure"),
        ("hma", "suspended_exemption"),
    ]


def test_gear_family_selects_closures(table):
    """Only closures banning the asked family hit, and a trawl exemption
    does not lift a gillnet ban."""
    assert reasons(table, 0.5, 0.5, date(2018, 6, 1), "sink_gillnet") == [
        ("dhra", "closure")
    ]
    assert not reasons(table, 3, 3, date(2024, 6, 1), "sink_gillnet")


def test_specific_gear_is_not_a_family(table):
    """Asking about a specific gear rather than a family fails loudly."""
    with pytest.raises(ValueError, match="not a gear family"):
        closure_hits(at(3, 3, date(2024, 6, 1)), table, "shrimp_trawl")


def test_hits_keep_position_columns(table):
    """Each hit carries the position's own columns."""
    hits = closure_hits(at(3, 3, date(2024, 6, 1)), table)
    assert hits.columns == [
        "vessel_id",
        "timestamp",
        "lon",
        "lat",
        "closure_id",
        "reason",
    ]
    assert closure_hits(at(3, 3, date(2024, 6, 1)).clear(), table).is_empty()


@pytest.mark.parametrize(
    "rule, message",
    [
        (
            dict(area_id="x", kind="zone", parent=None),
            "unknown kind",
        ),
        (dict(area_id="x", kind="closure", gear=("trap",)), "unknown gear"),
        (dict(area_id="x", kind="exemption", parent=None), "only closures"),
        (dict(area_id="x", valid_to=OPEN), "not after"),
    ],
)
def test_rule_rejects_bad_fields(rule, message):
    """Bad kinds, gear, parents and date order fail on construction."""
    fields = dict(
        area_id="x",
        kind="closure",
        gear=("mobile_bottom",),
        valid_from=OPEN,
        valid_to=None,
        geometry=square(0, 0, 1, 1),
    )
    with pytest.raises(ValueError, match=message):
        Rule(**{**fields, **rule})


def test_check_table_rejects_bad_parents(table):
    """Repeated ids and parents of the wrong kind fail."""
    with pytest.raises(ValueError, match="repeated"):
        check_table(table + [table[0]])
    orphan = Rule(
        "orphan",
        "suspension",
        ("shrimp_trawl",),
        OPEN,
        None,
        square(0, 0, 1, 1),
        parent="hma",
    )
    with pytest.raises(ValueError, match="is not of kind exemption"):
        check_table(table + [orphan])
