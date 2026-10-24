"""Tests for the evaluation helpers: delays, reception with planted
silences removed, closure areas and the alert-rate pieces."""

from datetime import date, datetime, timedelta, timezone

import polars as pl
import pytest
from shapely.geometry import box

from synthetic import Track, frame
from vessels.core.geo import LocalProjection
from vessels.core.geofence import Rule
from vessels.core.lifecycle import Incident
from vessels.evaluation.alerts import edge_strata, hours_near, vessel_days
from vessels.evaluation.injection import closure_areas, delays, recall_table, silenced

T0 = datetime(2024, 5, 20, tzinfo=timezone.utc)
UTC = pl.Datetime("us", "UTC")


def incident(vessel: str, detector: str, opened: float, raised: float) -> Incident:
    """An open incident at given hours after T0.

    Args:
        vessel: Vessel ID.
        detector: Detector name.
        opened: Evidence time, hours after T0.
        raised: Processing time, hours after T0.

    Returns:
        Incident: The incident.
    """
    return Incident(
        incident_id=f"{vessel}:{opened}",
        vessel_id=vessel,
        detector=detector,
        status="open",
        opened_at=T0 + timedelta(hours=opened),
        message="",
        raised_at=T0 + timedelta(hours=raised),
    )


def truth(rows: list[tuple]) -> pl.DataFrame:
    """A truth table from (id, label, vessel, partner, start h, end h).

    Args:
        rows: Case tuples.

    Returns:
        pl.DataFrame: The truth table.
    """
    return pl.DataFrame(
        [
            {
                "truth_id": r[0],
                "label": r[1],
                "vessel_id": r[2],
                "partner_id": r[3],
                "start": T0 + timedelta(hours=r[4]),
                "end": T0 + timedelta(hours=r[5]),
            }
            for r in rows
        ],
        schema={
            "truth_id": pl.String,
            "label": pl.String,
            "vessel_id": pl.String,
            "partner_id": pl.String,
            "start": UTC,
            "end": UTC,
        },
    )


def test_delay_counts_from_ready_time():
    """Delay is the first matching incident's raise time less start + ready."""
    cases = truth(
        [("g1", "positive", "a", None, 0, 10), ("g2", "positive", "b", None, 0, 10)]
    )
    raised = [
        incident("a", "gap", 6.0, 7.5),
        incident("a", "gap", 9.0, 7.0),
        incident("a", "closure_fishing", 6.0, 6.1),
        incident("b", "gap", 40.0, 41.0),
    ]
    result = delays(cases, raised, "gap", ready=6.0)
    assert result["truth_id"].to_list() == ["g1", "g2"]
    assert result["delay_hours"][0] == pytest.approx(1.0)
    # b's only incident opened over a day after its end: not this case.
    assert result["delay_hours"][1] is None


def test_delay_skips_negatives_and_uses_partner():
    """Negatives get no delay; an encounter's partner incident counts."""
    cases = truth(
        [("e1", "positive", "h", "p", 0, 3), ("e2", "near_miss_0", "h", "q", 20, 22)]
    )
    result = delays(
        cases, [incident("p", "encounter", 2.5, 4.0)], "encounter", ready=2.0
    )
    assert result["truth_id"].to_list() == ["e1"]
    assert result["delay_hours"][0] == pytest.approx(2.0)


def test_silenced_drops_cells_heard_only_by_deleted_positions():
    """A cell-day heard only through deleted positions is no longer heard."""
    day = date(2024, 5, 20)
    annotated = pl.DataFrame(
        {
            "vessel_id": ["a", "a", "a"],
            "timestamp": [T0, T0 + timedelta(hours=1), T0 + timedelta(hours=2)],
            "lon": [0.0, 0.1, 0.2],
            "lat": [0.0, 0.0, 0.0],
            "h3_cell": ["x", "y", "z"],
        },
        schema_overrides={"timestamp": UTC},
    )
    planted = annotated.filter(pl.col("h3_cell") != "y")
    reception = pl.DataFrame(
        {"h3_cell": ["x", "y", "z", "w"], "day": [day] * 4, "positions": [5, 1, 1, 3]}
    )
    kept = silenced(reception, annotated, planted)
    assert sorted(kept["h3_cell"]) == ["w", "x", "z"]


def test_closure_areas_cut_exemptions():
    """A scored closure loses its exemption areas; others are left out."""
    rules = [
        Rule(
            "c", "closure", ("mobile_bottom",), date(2018, 1, 1), None, box(0, 0, 2, 2)
        ),
        Rule(
            "e",
            "exemption",
            ("shrimp_trawl",),
            date(2018, 1, 1),
            None,
            box(0, 0, 1, 1),
            parent="c",
        ),
        Rule(
            "d", "closure", ("mobile_bottom",), date(2018, 1, 1), None, box(5, 5, 6, 6)
        ),
    ]
    areas = closure_areas(rules, ("c",))
    assert list(areas) == ["c"]
    assert areas["c"].area == pytest.approx(3.0)


def test_recall_table_intervals():
    """Recall per group carries its Wilson bounds."""
    cases = pl.DataFrame({"label": ["positive"] * 10, "hit": [True] * 8 + [False] * 2})
    table = recall_table(cases, ["label"])
    assert table.row(0, named=True) == {
        "label": "positive",
        "planted": 10,
        "hit": 8,
        "share": 0.8,
        "low": 0.49,
        "high": 0.943,
    }


def test_vessel_days_per_class():
    """Vessel-days count each vessel once per UTC day, per class and in all."""
    a = Track("a", -69.0, 42.0).leg(30 * 60, 5.0, 90.0, 60.0)
    b = Track("b", -69.0, 42.0, transceiver="B").leg(60, 5.0)
    assert vessel_days(frame(a, b)) == {"A": 2, "B": 1, "all": 3}


def test_hours_near_caps_steps_and_buffers():
    """Time near an area counts steps up to an hour, within the buffer."""
    projection = LocalProjection(-69.0, 42.0)
    track = (
        Track("a", -69.0, 42.0)
        .leg(120, 0.0, 90.0, 30.0)
        .silence(5)
        .leg(30, 0.0, 90.0, 30.0)
    )
    positions = frame(track)
    near = box(-68.95, 41.95, -68.9, 42.05)
    far = box(-67.0, 41.95, -66.9, 42.05)
    # Positions at 0, 0.5, 1, 1.5, 2 and 7.5 hours: four 30-minute steps,
    # the 5.5-hour step capped at an hour; the last position counts nothing.
    assert hours_near(positions, near, projection) == pytest.approx(4 * 0.5 + 1.0)
    assert hours_near(positions, far, projection) == 0.0


def test_edge_strata_bands():
    """Gaps are labelled uncovered or by distance to the coverage edge."""
    gaps = pl.DataFrame(
        {"start_cell": ["u", "n", "m", "f", "?"], "transceiver": ["A"] * 5}
    )
    grid = pl.DataFrame(
        {
            "h3_cell": ["u", "n", "m", "f"],
            "transceiver": ["A"] * 4,
            "covered": [False, True, True, True],
            "edge_km": [0.0, 5.0, 20.0, 50.0],
        }
    )
    assert edge_strata(gaps, grid)["stratum"].to_list() == [
        "uncovered",
        "edge under 10 km",
        "edge 10-30 km",
        "edge 30 km+",
        "uncovered",
    ]
