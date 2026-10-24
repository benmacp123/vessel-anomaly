"""Tests for the gap detector: which silences are gaps, and each check that
can explain one away."""

import json
from datetime import timedelta

import h3
import polars as pl
import pytest

from synthetic import T0, Track, frame
from vessels.core.cleaning import clean
from vessels.core.distance import PortDistance
from vessels.core.gaps import (
    GAP_SCHEMA,
    GapRules,
    Moorings,
    gap_candidates,
    gap_events,
    gap_rows,
)
from vessels.core.geo import LocalProjection
from vessels.core.segment import segment

LON, LAT = -69.0, 42.0
RES = 6
PROJECTION = LocalProjection(LON, LAT)
# One anchorage well away from the test tracks unless a test moves it.
FAR_PORT = PortDistance([-71.0], [41.0], PROJECTION)


def run(*tracks, uncovered=(), silent=(), ports=FAR_PORT, rules=GapRules()):
    """Clean, segment and detect gaps, with every cell the tracks touch
    covered and heard on every day unless listed.

    Args:
        *tracks: ``Track`` objects.
        uncovered: Cells to mark not covered.
        silent: ``(cell, date)`` pairs with no traffic heard.
        ports: Anchorage distance.
        rules: Gap thresholds.

    Returns:
        pl.DataFrame: ``gap_events`` output.
    """
    data = segment(clean(frame(*tracks)).frame)
    cells = sorted(
        {h3.latlng_to_cell(y, x, RES) for x, y in zip(data["lon"], data["lat"])}
    )
    grid = pl.DataFrame({"h3_cell": cells}).join(
        pl.DataFrame({"transceiver": ["A", "B"]}), how="cross"
    )
    grid = grid.with_columns(covered=pl.col("h3_cell").is_in(list(uncovered)).not_())
    days = data["timestamp"].dt.date().unique()
    heard = pl.DataFrame({"h3_cell": cells}).join(
        pl.DataFrame({"day": days}), how="cross"
    )
    if silent:
        gone = pl.DataFrame(list(silent), schema=["h3_cell", "day"], orient="row")
        heard = heard.join(gone, on=["h3_cell", "day"], how="anti")
    return gap_events(data, grid, ports, heard, rules)


def at_sea(hours, before=12 * 60, after=60, sog=8.0):
    """A vessel heard every 2 minutes, silent for ``hours``, then heard
    again.

    Args:
        hours: Gap length.
        before: Minutes heard before the gap.
        after: Minutes heard after it.
        sog: Speed throughout.

    Returns:
        Track: The track.
    """
    return Track("v", LON, LAT).leg(before, sog).silence(hours, sog).leg(after, sog)


def cell(lon, lat):
    """The test resolution's cell at a point.

    Args:
        lon: Longitude.
        lat: Latitude.

    Returns:
        str: Cell ID.
    """
    return h3.latlng_to_cell(lat, lon, RES)


def test_unexplained_gap():
    """An 8-hour silence at sea in good coverage is unexplained, not long."""
    found = run(at_sea(8))
    assert found.height == 1
    row = found.row(0, named=True)
    assert row["unexplained"] and not row["long"]
    assert row["hours"] == pytest.approx(8 + 2 / 60)
    assert row["prior_positions"] == 361
    assert found.schema == GAP_SCHEMA


def test_long_gap():
    """A 13-hour silence is long."""
    assert run(at_sea(13))["long"].to_list() == [True]


def test_short_silence_is_not_a_gap():
    """Five hours of silence is under the 6-hour minimum."""
    assert run(at_sea(5)).is_empty()


def test_gap_across_segment_break():
    """A 30-hour silence ends the segment but is still one gap."""
    track = at_sea(30)
    data = segment(clean(frame(track)).frame)
    assert data["segment"].n_unique() == 2
    found = run(track)
    assert found.height == 1 and found["long"][0]


def test_end_outside_coverage():
    """A gap ending in an uncovered cell is explained by coverage."""
    track = at_sea(8)
    end = track.rows[361]
    row = run(track, uncovered=[cell(end["lon"], end["lat"])]).row(0, named=True)
    assert not row["covered"] and not row["unexplained"]
    assert row["away_from_port"] and row["dense_before"] and row["cells_heard"]


@pytest.mark.parametrize("end, index", [("start", 360), ("end", 361)])
def test_end_near_port(end, index):
    """A gap starting or ending within 10 km of an anchorage is explained."""
    track = at_sea(8)
    near = track.rows[index]
    port = PortDistance([near["lon"]], [near["lat"] + 0.05], PROJECTION)
    row = run(track, ports=port).row(0, named=True)
    assert row[f"{end}_port_km"] == pytest.approx(5.56, abs=0.05)
    assert not row["away_from_port"] and not row["unexplained"]


def test_sparse_before():
    """Thirteen positions in the 12 hours before a gap are too few; 14
    are enough."""
    hourly = Track("v", LON, LAT).leg(12 * 60, 8.0, step=60).silence(8, 8.0)
    sparse = run(hourly.leg(60, 8.0))
    assert sparse["prior_positions"].to_list() == [13]
    assert not sparse["dense_before"][0]
    denser = Track("v", LON, LAT).leg(12 * 60, 8.0, step=50).silence(8, 8.0)
    assert run(denser.leg(60, 8.0))["dense_before"].to_list() == [True]


def test_cell_silent_one_day():
    """A gap whose start cell heard nothing on a day it spans is explained
    as a possible receiver outage."""
    track = at_sea(8)
    start = track.rows[360]
    day = (start["timestamp"] + timedelta(hours=1)).date()
    row = run(track, silent=[(cell(start["lon"], start["lat"]), day)]).row(
        0, named=True
    )
    assert not row["cells_heard"] and not row["unexplained"]


def test_repeat_does_not_fill_a_gap():
    """A relayed copy of an old position inside the silence is ignored."""
    track = at_sea(8)
    echo = dict(track.rows[100])
    echo["timestamp"] = track.rows[360]["timestamp"] + timedelta(hours=4)
    track.rows.append(echo)
    found = run(track)
    assert found["hours"].to_list() == [pytest.approx(8 + 2 / 60)]


def test_shared_id_gaps_per_segment():
    """When two transmitters share an ID, one going silent is a gap even
    though the other keeps reporting."""
    quiet = at_sea(8)
    other = Track("v", -66.0, 44.0).leg(12 * 60 + 8 * 60 + 60, 6.0, step=10)
    found = run(quiet, other)
    assert found.height == 1
    assert found["track"][0] != "v"


def test_threshold_from_rules():
    """The minimum gap length comes from the rules."""
    assert run(at_sea(5), rules=GapRules(min_hours=4)).height == 1


def moored(vessel="v", heard_hours=14, gap_hours=8):
    """A vessel heard stationary, then silent in the same spot, then heard
    again there.

    Args:
        vessel: Vessel ID.
        heard_hours: Hours heard before the gap.
        gap_hours: Gap length.

    Returns:
        Track: The track.
    """
    track = Track(vessel, LON, LAT).leg(heard_hours * 60, 0.0)
    return track.silence(gap_hours).leg(60, 0.0)


def test_gap_at_own_mooring():
    """A silence where the vessel was heard stationary for 14 hours is
    explained as a mooring."""
    row = run(moored()).row(0, named=True)
    assert not row["away_from_mooring"] and not row["unexplained"]
    assert row["covered"] and row["away_from_port"] and row["dense_before"]


def test_gap_does_not_explain_itself():
    """A 13-hour silence after two stationary hours is not a mooring,
    although the run spanning it lasts 16 hours."""
    row = run(moored(heard_hours=2, gap_hours=13)).row(0, named=True)
    assert row["away_from_mooring"] and row["unexplained"]


def test_other_vessels_mooring_does_not_count():
    """Another vessel's mooring at the same spot does not explain a gap."""
    other = Track("w", LON, LAT).leg(14 * 60, 0.0)
    later = T0 + timedelta(hours=15)
    gap = Track("v", LON, LAT, later).leg(2 * 60, 0.0).silence(8).leg(60, 0.0)
    rows = run(other, gap)
    assert rows.filter(pl.col("vessel_id") == "v")["away_from_mooring"].to_list() == [
        True
    ]


def test_leaving_the_mooring_dark():
    """A vessel that goes dark as it leaves its mooring and reappears far
    away is explained by the mooring at the start."""
    track = Track("v", LON, LAT).leg(14 * 60, 0.0).silence(8, sog=10.0)
    row = run(track.leg(60, 10.0)).row(0, named=True)
    assert row["distance_km"] > 100
    assert not row["away_from_mooring"]


def test_shared_only_at_the_time():
    """A second transmitter heard days earlier does not stop a later gap
    being measured across the vessel."""
    early = Track("v", -66.0, 44.0).leg(60, 6.0, step=10)
    later = Track("v", LON, LAT, T0 + timedelta(days=3))
    found = run(early, later.leg(12 * 60, 8.0).silence(30, 8.0).leg(60, 8.0))
    assert found.filter(pl.col("hours") < 31)["track"].to_list() == ["v"]


def test_later_mooring_does_not_explain():
    """A gap is not explained by a mooring the vessel first uses after it."""
    track = Track("v", LON, LAT).leg(11 * 60, 0.0).silence(8).leg(60, 0.0)
    track.silence(1).leg(14 * 60, 0.0)
    # Eleven hours stationary before the gap is not yet a mooring; the
    # stay that follows it, at the same spot, must not count.
    assert run(track)["away_from_mooring"].to_list() == [True]


def test_moorings_resume():
    """Saving and resuming mooring state midway gives the same flags as one
    pass."""
    track = moored(heard_hours=14, gap_hours=8)
    data = segment(clean(frame(track)).frame)
    rows = gap_rows(data)
    found = gap_candidates(data)
    whole = Moorings(PROJECTION).flags(rows, found, 1.0)
    first = Moorings(PROJECTION)
    head = first.flags(rows.head(300), found.clear(), 1.0)
    state = json.loads(json.dumps(first.state))
    tail = Moorings(PROJECTION, state).flags(rows.slice(300), found, 1.0)
    assert head.is_empty()
    assert tail.equals(whole)


def test_rows_without_segment_are_not_a_second_transmitter():
    """Speedless positions before the first segment do not make the
    vessel look shared."""
    track = at_sea(8)
    for row in track.rows[:5]:
        row["sog"] = None
    assert run(track)["track"].to_list() == ["v"]


def test_speedless_rows_do_not_extend_an_old_segment():
    """Speedless positions after a long silence do not make the segment
    heard before the silence look recent."""
    track = Track("v", LON, LAT).leg(60, 8.0).silence(30, 0.0)
    start = len(track.rows)
    track.leg(60, 8.0)
    for row in track.rows[start:]:
        row["sog"] = None
    track.leg(12 * 60, 8.0).silence(8, 8.0).leg(60, 8.0)
    found = run(track).filter(pl.col("hours") < 10)
    assert found["track"].to_list() == ["v"]
