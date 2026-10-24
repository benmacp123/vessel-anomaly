"""Tests for the planted-event generator on synthetic tracks: each planted
case has the shape it claims, and planting is repeatable."""

from datetime import timedelta

import numpy as np
import polars as pl
import pytest
from shapely.geometry import box

from synthetic import T0, Track, frame
from vessels.core.geo import LocalProjection, haversine_km
from vessels.sim.inject import InjectionRules, _hourly_kn, plant_closures
from vessels.sim.inject import plant_encounters, plant_gaps, plant_loitering

LON, LAT = -69.0, 42.0
PROJECTION = LocalProjection(LON, LAT)
NONE = {
    "gaps": 0,
    "gap_negatives": 0,
    "closures": 0,
    "closure_transits": 0,
    "encounters": 0,
    "near_misses": 0,
    "loiters": 0,
    "loiter_negatives": 0,
}


def rules(**counts) -> InjectionRules:
    """Generator settings planting only the cases asked for.

    Args:
        **counts: Case counts to set; every other count is zero.

    Returns:
        InjectionRules: The settings.
    """
    return InjectionRules(**(NONE | counts))


def annotated(
    data: pl.DataFrame, covered=True, port_km=50.0, shore_nm=40.0
) -> pl.DataFrame:
    """Add the planters' annotation columns with fixed values.

    Args:
        data: Positions.
        covered: Value of ``covered``.
        port_km: Value of ``port_km``.
        shore_nm: Value of ``shore_nm``.

    Returns:
        pl.DataFrame: ``data`` plus the annotations.
    """
    return data.with_columns(
        covered=pl.lit(covered), port_km=pl.lit(port_km), shore_nm=pl.lit(shore_nm)
    )


def steaming(
    vessel: str, hours: float, sog: float, step: float = 2.0, lat=LAT
) -> Track:
    """A track steaming east for some hours.

    Args:
        vessel: Vessel ID.
        hours: Duration.
        sog: Speed, knots.
        step: Minutes between positions.
        lat: Start latitude.

    Returns:
        Track: The track.
    """
    return Track(vessel, LON, lat).leg(hours * 60, sog, 90.0, step)


def test_planted_gaps_are_silent_inside():
    """Each planted gap has no positions inside it and keeps both ends."""
    data = annotated(frame(steaming("a", 96, 6.0), steaming("b", 96, 6.0, lat=42.5)))
    out, truth = plant_gaps(data, np.array([8.0]), rules(gaps=4, gap_negatives=2))
    assert truth.height == 6
    assert set(truth["label"]) == {"positive", "short"}
    for row in truth.iter_rows(named=True):
        mine = out.filter(pl.col("vessel_id") == row["vessel_id"])
        inside = mine.filter(
            pl.col("timestamp").is_between(row["start"], row["end"], "none")
        )
        assert inside.is_empty()
        assert (
            mine.filter(pl.col("timestamp").is_in([row["start"], row["end"]])).height
            == 2
        )
        hours = (row["end"] - row["start"]) / timedelta(hours=1)
        assert hours >= (8.0 if row["label"] == "positive" else 4.0)
    assert out.columns == data.columns[:11]


def test_planted_gaps_keep_apart_on_one_vessel():
    """Two cases on one vessel are at least the buffer apart."""
    data = annotated(frame(steaming("a", 120, 6.0)))
    _, truth = plant_gaps(data, np.array([7.0]), rules(gaps=3))
    spans = sorted(zip(truth["start"], truth["end"]))
    for (_, end), (start, _) in zip(spans, spans[1:]):
        assert start - end >= timedelta(hours=12)


def test_no_gaps_outside_coverage():
    """Positions in uncovered cells or near an anchorage take no gap."""
    data = frame(steaming("a", 96, 6.0))
    assert plant_gaps(annotated(data, covered=False), np.array([8.0]), rules(gaps=3))[
        1
    ].is_empty()
    assert plant_gaps(annotated(data, port_km=2.0), np.array([8.0]), rules(gaps=3))[
        1
    ].is_empty()


def test_no_gap_across_a_real_one():
    """A window that already holds a long silence is not used."""
    track = Track("a", LON, LAT).leg(600, 6.0).silence(5).leg(600, 6.0)
    _, truth = plant_gaps(annotated(frame(track)), np.array([8.0]), rules(gaps=5))
    gap_start, gap_end = T0 + timedelta(hours=10), T0 + timedelta(hours=15)
    for row in truth.iter_rows(named=True):
        assert row["end"] <= gap_start or row["start"] >= gap_end


def test_planting_is_repeatable():
    """The same seed plants the same cases."""
    data = annotated(frame(steaming("a", 96, 6.0)))
    first = plant_gaps(data, np.array([7.0, 9.0]), rules(gaps=3))
    second = plant_gaps(data, np.array([7.0, 9.0]), rules(gaps=3))
    assert first[0].equals(second[0]) and first[1].equals(second[1])


def test_planted_tows_land_inside_and_transits_cross():
    """Moved tows lie wholly inside the closure; transits cross its edge."""
    tows = [steaming(f"t{k}", 4, 3.5, step=1.0, lat=LAT + k * 0.01) for k in range(3)]
    # Each real run is moved once, so two transits need two tracks.
    transits = [
        steaming(f"x{k}", 2, 9.0, step=1.0, lat=41.0 + k * 0.1) for k in range(2)
    ]
    data = annotated(frame(*tows, *transits))
    area = box(-68.5, 42.6, -68.2, 42.9)
    out, truth = plant_closures(
        data, {"square": area}, PROJECTION, rules(closures=3, closure_transits=2)
    )
    assert truth.height == 5
    for row in truth.iter_rows(named=True):
        planted = out.filter(pl.col("vessel_id") == row["vessel_id"])
        assert row["vessel_id"].startswith("inj-")
        assert row["closure_id"] == "square"
        inside = planted["lon"].is_between(-68.5, -68.2) & planted["lat"].is_between(
            42.6, 42.9
        )
        if row["label"] == "positive":
            assert inside.all()
            assert planted["sog"].is_between(2.5, 5.5).all()
        else:
            assert inside.any() and not inside.all()
    assert out.filter(pl.col("vessel_id").str.starts_with("inj")).height < out.height


def test_planted_partner_closes_holds_and_leaves():
    """A planted partner is far, then within range for the meeting, then far."""
    host = Track("h", LON, LAT).leg(16 * 60, 0.8, 90.0, 2.0)
    donor = steaming("d", 16, 8.0, step=3.0, lat=41.0)
    data = annotated(frame(host, donor))
    out, truth = plant_encounters(data, PROJECTION, rules(encounters=1))
    assert truth.height == 1
    row = truth.row(0, named=True)
    partner = out.filter(pl.col("vessel_id") == row["partner_id"])
    hosted = out.filter(pl.col("vessel_id") == "h")
    t_host = hosted["timestamp"].dt.epoch("us").to_numpy()
    t = partner["timestamp"].dt.epoch("us").to_numpy()
    km = haversine_km(
        partner["lon"].to_numpy(),
        partner["lat"].to_numpy(),
        np.interp(t, t_host, hosted["lon"].to_numpy()),
        np.interp(t, t_host, hosted["lat"].to_numpy()),
    )
    start, end = (row[k].timestamp() * 1e6 for k in ("start", "end"))
    meeting = (t >= start) & (t <= end)
    far = (t < start - 1.8e9) | (t > end + 1.8e9)
    assert meeting.any() and far.any()
    assert km[meeting].max() < 0.45
    assert km[far].min() > 4.9
    # The partner keeps the donor's three-minute cadence and class.
    assert np.median(np.diff(t)) == pytest.approx(180e6)
    assert partner["transceiver"].unique().to_list() == ["A"]


def test_hourly_speed():
    """Clock-hour speeds match a constant-speed track; a step ending on the
    hour counts in the hour it ends in, as the loitering rule buckets it."""
    us = np.arange(0, 3 * 3_600_000_000 + 1, 600_000_000)
    x = np.linspace(0, 3 * 5 * 1.852, us.size)
    assert _hourly_kn(us, x, np.zeros_like(x)) == pytest.approx([5.0] * 4)


def varying(vessel: str, hours: int) -> Track:
    """A track alternating hours at 6 and 10 knots, so hourly speeds differ.

    Args:
        vessel: Vessel ID.
        hours: Duration in whole hours.

    Returns:
        Track: The track.
    """
    track = Track(vessel, LON, LAT)
    for hour in range(hours):
        track.leg(60, 6.0 if hour % 2 else 10.0, 90.0, 5.0)
    return track


def planted_hours(out: pl.DataFrame, row: dict) -> tuple[np.ndarray, np.ndarray]:
    """Hourly speeds inside a planted case, and every step's length.

    Args:
        out: Positions after planting.
        row: The case's truth row.

    Returns:
        tuple[np.ndarray, np.ndarray]: Knots per clock hour inside the
        case, and kilometres per step over the whole planted vessel.
    """
    planted = out.filter(pl.col("vessel_id") == row["vessel_id"])
    us = planted["timestamp"].dt.epoch("us").to_numpy()
    x, y = PROJECTION.forward(planted["lon"].to_numpy(), planted["lat"].to_numpy())
    inside = (us >= row["start"].timestamp() * 1e6) & (
        us <= row["end"].timestamp() * 1e6
    )
    return _hourly_kn(us[inside], x[inside], y[inside]), np.hypot(
        np.diff(x), np.diff(y)
    )


def test_planted_loitering_is_slow_inside_and_joins_up():
    """A slowed stretch is slow in every hour and rejoins without a jump."""
    data = annotated(frame(varying("a", 30)))
    out, truth = plant_loitering(data, PROJECTION, rules(loiters=1))
    assert truth.height == 1
    row = truth.row(0, named=True)
    hourly, steps_km = planted_hours(out, row)
    assert hourly.max() < 1.6
    # Five minutes at 10 kn is about 1.5 km; no step jumps further.
    assert steps_km.max() < 1.6
    hours = (row["end"] - row["start"]) / timedelta(hours=1)
    assert 9.0 - 0.1 <= hours <= 16.0


def test_loitering_negatives_are_short_or_never_slow():
    """Short negatives are slow but under 8 hours; fast ones are slow in no hour."""
    data = annotated(frame(varying("a", 30), varying("b", 30), varying("c", 30)))
    out, truth = plant_loitering(data, PROJECTION, rules(loiter_negatives=1))
    assert sorted(truth["label"]) == ["fast", "short"]
    for row in truth.iter_rows(named=True):
        hourly, _ = planted_hours(out, row)
        hours = (row["end"] - row["start"]) / timedelta(hours=1)
        if row["label"] == "short":
            assert hourly.max() < 1.6 and hours < 7.1
        else:
            assert hourly.min() > 2.4


def test_planted_gap_ends_inside_coverage():
    """A gap may not end in an uncovered cell, even when it starts in one."""
    data = annotated(frame(steaming("a", 96, 6.0)))
    # Most 10-hour windows starting in the first 14 hours end outside.
    edge = T0 + timedelta(hours=14)
    data = data.with_columns(covered=pl.col("timestamp") < edge)
    _, truth = plant_gaps(data, np.array([10.0]), rules(gaps=4))
    assert truth.height > 0
    assert (truth["end"] < edge).all()
