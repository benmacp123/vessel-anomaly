"""Tests for the cleaning rules: MMSI validity, invalid values, duplicates,
the land mask and the order- and repeat-invariance of the whole cleaner."""

from datetime import datetime, timedelta, timezone

import polars as pl
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from shapely.geometry import box

from vessels.core.cleaning import (
    DROP_RULES,
    NULL_RULES,
    clean,
    inland_area,
    mmsi_rule,
)
from vessels.core.geo import LocalProjection

T0 = datetime(2024, 5, 6, 12, 0, tzinfo=timezone.utc)
PROJECTION = LocalProjection(-68.0, 42.5)


def positions(rows):
    """Positions in the internal schema from short tuples.

    Args:
        rows: ``(vessel_id, minutes after T0, lon, lat, sog, cog, heading)``
            per row.

    Returns:
        pl.DataFrame: The rows with extract column types.
    """
    ids, minutes, lons, lats, sogs, cogs, headings = zip(*rows)
    return pl.DataFrame(
        {
            "vessel_id": list(ids),
            "timestamp": [T0 + timedelta(minutes=m) for m in minutes],
            "lon": list(lons),
            "lat": list(lats),
            "sog": list(sogs),
            "cog": list(cogs),
            "heading": list(headings),
        },
        schema_overrides={
            "sog": pl.Float32,
            "cog": pl.Float32,
            "heading": pl.Int32,
        },
    )


def test_mmsi_rule_names_each_problem():
    """Each invalid MMSI is named by the first rule it breaks."""
    frame = pl.DataFrame(
        {"mmsi": [367000001, 123456789, 1193046, 12345678, 1000000000, 199000001]}
    )
    rules = frame.select(rule=mmsi_rule(pl.col("mmsi")))["rule"].to_list()
    assert rules == [None, "placeholder", "placeholder", "length", "length", "mid"]


def test_mmsi_rule_accepts_mid_range_ends():
    """MIDs 201 and 775 are valid; 200 and 776 are not."""
    frame = pl.DataFrame({"mmsi": [201000000, 775999999, 200999999, 776000000]})
    rules = frame.select(rule=mmsi_rule(pl.col("mmsi")))["rule"].to_list()
    assert rules == [None, None, "mid", "mid"]


def test_bad_positions_drop_rows():
    """Out-of-range and 0,0 positions are dropped and counted."""
    frame = positions(
        [
            ("a", 0, -68.0, 42.0, 3.0, 90.0, 90),
            ("a", 1, 181.0, 42.0, 3.0, 90.0, 90),
            ("a", 2, -68.0, 91.0, 3.0, 90.0, 90),
            ("a", 3, 0.0, 0.0, 3.0, 90.0, 90),
        ]
    )
    result = clean(frame)
    assert result.frame.height == 1
    assert result.dropped["bad_position"] == 3


def test_invalid_values_become_null_and_keep_row():
    """Invalid speed, noise speed, course and heading are nulled, not dropped."""
    frame = positions(
        [
            ("a", 0, -68.0, 42.0, 102.3, 90.0, 90),
            ("a", 1, -68.0, 42.0, 51.2, 90.0, 90),
            ("a", 2, -68.0, 42.0, 3.0, 360.0, 90),
            ("a", 3, -68.0, 42.0, 3.0, 90.0, 511),
            ("a", 4, -68.0, 42.0, 3.0, 90.0, 90),
        ]
    )
    result = clean(frame)
    assert result.frame.height == 5
    assert result.nulled == {
        "sog_invalid": 1,
        "sog_noise": 1,
        "cog_invalid": 1,
        "heading_invalid": 1,
    }
    assert result.frame["sog"].null_count() == 2
    assert result.frame["cog"].null_count() == 1
    assert result.frame["heading"].null_count() == 1


def test_duplicate_position_dropped_but_moved_vessel_kept():
    """A repeated time and position is dropped; same time elsewhere is kept."""
    frame = positions(
        [
            ("a", 0, -68.0, 42.0, 3.0, 90.0, 90),
            ("a", 0, -68.0, 42.0, 3.1, 90.0, 90),
            ("a", 0, -68.1, 42.0, 3.0, 90.0, 90),
            ("b", 0, -68.0, 42.0, 3.0, 90.0, 90),
        ]
    )
    result = clean(frame)
    assert result.frame.height == 3
    assert result.dropped["duplicate"] == 1


def test_land_mask_drops_only_beyond_buffer():
    """Points well inland drop; points just inside the coastline stay."""
    # Land is everything west of 68W within the test area.
    land = box(-70.0, 41.0, -68.0, 44.0)
    inland = inland_area(land, 1.0, PROJECTION)
    km_per_deg_lon = 111.195 * 0.7373
    frame = positions(
        [
            ("a", 0, -68.0 - 0.5 / km_per_deg_lon, 42.5, 0.0, 0.0, 0),
            ("a", 1, -68.0 - 2.0 / km_per_deg_lon, 42.5, 0.0, 0.0, 0),
            ("a", 2, -67.9, 42.5, 0.0, 0.0, 0),
        ]
    )
    result = clean(frame, inland, PROJECTION)
    assert result.dropped["inland"] == 1
    assert result.frame["lon"].max() > -68.0 - 1.0 / km_per_deg_lon


def test_land_mask_needs_projection():
    """Passing a land area without its projection is refused."""
    frame = positions([("a", 0, -68.0, 42.0, 3.0, 90.0, 90)])
    with pytest.raises(ValueError):
        clean(frame, box(0, 0, 1, 1))


def test_counts_name_every_rule():
    """The result always carries a count for every rule, even when zero."""
    result = clean(positions([("a", 0, -68.0, 42.0, 3.0, 90.0, 90)]))
    assert list(result.dropped) == list(DROP_RULES)
    assert list(result.nulled) == list(NULL_RULES)


ROWS = st.lists(
    st.tuples(
        st.sampled_from(["a", "b", "c"]),
        st.integers(0, 30),
        st.sampled_from([-68.0, -67.5, 0.0, 181.0]),
        st.sampled_from([42.0, 42.5, 0.0]),
        st.sampled_from([0.0, 3.0, 51.2, 102.3]),
        st.sampled_from([0.0, 180.0, 360.0]),
        st.sampled_from([0, 90, 511]),
    ),
    min_size=1,
    max_size=40,
)


@settings(max_examples=200, deadline=None)
@given(ROWS)
def test_cleaning_twice_changes_nothing(rows):
    """Cleaning a cleaned frame returns it unchanged and drops nothing."""
    once = clean(positions(rows)).frame
    twice = clean(once)
    assert twice.frame.equals(once)
    assert sum(twice.dropped.values()) == 0


@settings(max_examples=200, deadline=None)
@given(ROWS, st.randoms(use_true_random=False))
def test_cleaning_ignores_order_and_repeats(rows, rng):
    """Shuffled rows with some repeated give the same cleaned frame."""
    noisy = list(rows) + rng.sample(list(rows), k=len(rows) // 2)
    rng.shuffle(noisy)
    assert clean(positions(noisy)).frame.equals(clean(positions(rows)).frame)
