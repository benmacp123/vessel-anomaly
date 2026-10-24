"""Tests for the multi-track segmenter: joining, splitting on time,
separating shared IDs, isolating outliers, point status, and invariants
over random tracks."""

import json
import math
from datetime import datetime, timedelta, timezone

import polars as pl
from hypothesis import given, settings
from hypothesis import strategies as st

from vessels.core.cleaning import clean
from vessels.core.segment import (
    MAX_HOURS,
    Segmenter,
    overlapping_vessels,
    segment,
    segment_summary,
)

T0 = datetime(2024, 5, 6, 0, 0, tzinfo=timezone.utc)


def track(vessel, start_min, n, step_min, lon, lat, sog=8.0, cog=90.0, cls="A"):
    """A straight track at constant speed, positions dead-reckoned exactly.

    Args:
        vessel: Vessel ID.
        start_min: Minutes after T0 of the first point.
        n: Number of points.
        step_min: Minutes between points.
        lon: Start longitude.
        lat: Start latitude.
        sog: Speed, knots.
        cog: Course, degrees.
        cls: AIS class.

    Returns:
        list[dict]: One row per point.
    """
    rows = []
    for i in range(n):
        nm = sog * i * step_min / 60
        dlat = nm * math.cos(math.radians(cog)) / 60
        dlon = nm * math.sin(math.radians(cog)) / (60 * math.cos(math.radians(lat)))
        rows.append(
            {
                "vessel_id": vessel,
                "timestamp": T0 + timedelta(minutes=start_min + i * step_min),
                "lon": lon + dlon,
                "lat": lat + dlat,
                "sog": sog,
                "cog": cog,
                "heading": None,
                "transceiver": cls,
            }
        )
    return rows


def frame(rows):
    """Clean rows into the segmenter's input.

    Args:
        rows: Dicts from ``track`` or hand-written.

    Returns:
        pl.DataFrame: Cleaned, sorted positions.
    """
    raw = pl.DataFrame(
        rows,
        schema={
            "vessel_id": pl.String,
            "timestamp": pl.Datetime("us", "UTC"),
            "lon": pl.Float64,
            "lat": pl.Float64,
            "sog": pl.Float32,
            "cog": pl.Float32,
            "heading": pl.Int32,
            "transceiver": pl.String,
        },
    )
    return clean(raw).frame


def test_straight_track_is_one_segment():
    """A steady track gives one segment holding every point."""
    out = segment(frame(track("v", 0, 50, 5, -68.0, 42.0)))
    assert out["segment"].n_unique() == 1
    assert out["point_status"].unique().to_list() == ["position"]


def test_twelve_hour_silence_stays_in_segment():
    """A 12 h gap stays inside one segment, so gap detectors can see it."""
    rows = track("v", 0, 10, 5, -68.0, 42.0, sog=0.0) + track(
        "v", 12 * 60 + 45, 10, 5, -68.0, 42.0, sog=0.0
    )
    assert segment(frame(rows))["segment"].n_unique() == 1


def test_silence_over_max_hours_splits():
    """A gap longer than MAX_HOURS starts a new segment."""
    later = int((MAX_HOURS + 1) * 60) + 45
    rows = track("v", 0, 10, 5, -68.0, 42.0, sog=0.0) + track(
        "v", later, 10, 5, -68.0, 42.0, sog=0.0
    )
    assert segment(frame(rows))["segment"].n_unique() == 2


def test_shared_id_gives_two_overlapping_segments():
    """Two transmitters 50 km apart under one ID become two segments."""
    rows = track("v", 0, 40, 6, -68.0, 42.0) + track("v", 3, 40, 6, -67.4, 42.0)
    out = segment(frame(rows))
    summary = segment_summary(out)
    assert summary.height == 2
    assert sorted(summary["points"].to_list()) == [40, 40]
    assert overlapping_vessels(summary)["vessel_id"].to_list() == ["v"]


def test_outlier_is_isolated_and_track_continues():
    """A single far-off position gets its own segment; the track stays whole."""
    rows = track("v", 0, 30, 5, -68.0, 42.0)
    rows[15] = {**rows[15], "lon": -67.0, "lat": 42.5}
    summary = segment_summary(segment(frame(rows)))
    assert sorted(summary["points"].to_list()) == [1, 29]


def test_bad_and_repeat_rows_get_no_segment():
    """No speed, or moving with no course, is bad; a moving repeat is skipped."""
    rows = track("v", 0, 5, 5, -68.0, 42.0)
    rows.append({**rows[0], "timestamp": T0 + timedelta(minutes=30), "sog": None})
    rows.append({**rows[0], "timestamp": T0 + timedelta(minutes=31), "cog": None})
    slow = {**rows[0], "timestamp": T0 + timedelta(minutes=32), "sog": 0.2}
    rows.append({**slow, "cog": None})
    rows.append({**rows[2], "timestamp": T0 + timedelta(minutes=33)})
    out = segment(frame(rows))
    status = dict(zip(out["timestamp"].dt.minute(), out["point_status"]))
    assert [status[m] for m in (30, 31, 32, 33)] == [
        "bad",
        "bad",
        "position",
        "repeat",
    ]
    assert out.filter(pl.col("point_status") != "position")["segment"].null_count() == 3


def test_vessels_are_segmented_independently():
    """Segmenting vessels together equals segmenting each alone."""
    a = frame(track("a", 0, 20, 5, -68.0, 42.0))
    b = frame(track("b", 2, 20, 5, -68.0, 42.0, cog=0.0))
    together = segment(pl.concat([a, b]).sort("vessel_id", "timestamp"))
    apart = pl.concat([segment(a), segment(b)])
    assert together.equals(apart)


def test_summary_flags_short_segments():
    """Segments under ten points are flagged short."""
    rows = track("v", 0, 9, 5, -68.0, 42.0) + track("w", 0, 10, 5, -68.0, 42.0)
    summary = segment_summary(segment(frame(rows)))
    assert dict(zip(summary["vessel_id"], summary["short"])) == {"v": True, "w": False}


LEGS = st.lists(
    st.tuples(
        st.integers(0, 2000),
        st.integers(1, 15),
        st.integers(1, 120),
        st.floats(-68.5, -67.5),
        st.floats(41.5, 42.5),
        st.sampled_from([0.0, 3.0, 9.0, 20.0]),
        st.sampled_from([0.0, 90.0, 225.0]),
        st.sampled_from(["A", "B"]),
    ),
    min_size=1,
    max_size=6,
)


def legs_to_rows(vessels):
    """Expand per-vessel lists of legs into rows.

    Args:
        vessels: Per vessel, a list of ``track`` argument tuples.

    Returns:
        list[dict]: Every leg's rows.
    """
    return [
        row
        for vessel, legs in zip("abc", vessels)
        for leg in legs
        for row in track(vessel, *leg)
    ]


@settings(max_examples=100, deadline=None)
@given(st.lists(LEGS, min_size=1, max_size=3))
def test_segments_respect_vessel_and_time_limits(vessels):
    """Every usable row gets one segment of its own vessel, gaps within limits."""
    out = segment(frame(legs_to_rows(vessels)))
    usable = out.filter(pl.col("point_status") == "position")
    assert usable["segment"].null_count() == 0
    assert out.filter(pl.col("point_status") != "position")["segment"].null_count() == (
        out.height - usable.height
    )
    assert usable.select(
        pl.col("segment").str.starts_with(pl.col("vessel_id") + "-").all()
    ).item()
    gaps = usable.sort("segment", "timestamp").select(
        pl.col("timestamp").diff().over("segment").dt.total_seconds().max()
    )
    assert (gaps.item() or 0) <= MAX_HOURS * 3600


@settings(max_examples=100, deadline=None)
@given(st.lists(LEGS, min_size=1, max_size=3), st.randoms(use_true_random=False))
def test_segments_ignore_input_order(vessels, rng):
    """Shuffled input rows give the same segments once cleaned."""
    rows = legs_to_rows(vessels)
    shuffled = rows[:]
    rng.shuffle(shuffled)
    assert segment(frame(shuffled)).equals(segment(frame(rows)))


def test_resume_from_saved_state_matches_one_pass():
    """Segmenting in two pieces, saving and restoring state as JSON in
    between, gives the same labels as one pass."""
    rows = (
        track("a", 0, 60, 2, -69.0, 42.0)
        + track("a", 10, 60, 2, -66.0, 44.0, cls="B")
        + track("b", 0, 40, 5, -68.0, 41.0)
    )
    data = clean(frame(rows)).frame
    whole = segment(data)
    cut = T0 + timedelta(minutes=50)
    head = data.filter(pl.col("timestamp") <= cut)
    tail = data.filter(pl.col("timestamp") > cut)
    first, seen = Segmenter(), {}
    part1 = segment(head, first, seen)
    saved = json.loads(json.dumps(first.to_dict()))
    sizes = dict(part1.group_by("segment").len().drop_nulls().iter_rows())
    assert {s["label"]: s["count"] for s in saved["a"]["open"]} == {
        k: v for k, v in sizes.items() if k.startswith("a-")
    }
    part2 = segment(tail, Segmenter.from_dict(saved), seen)
    # The two transmitters report at the same minutes, so ties need lon.
    order = ["vessel_id", "timestamp", "lon"]
    joined = pl.concat([part1, part2]).sort(order)
    assert joined["segment"].to_list() == whole.sort(order)["segment"].to_list()


def test_segments_started_in_one_second_get_distinct_labels():
    """Two transmitters first heard in the same second start two segments
    with different labels."""
    rows = track("a", 0, 5, 2, -69.0, 42.0) + track("a", 0, 5, 2, -66.0, 44.0)
    labels = segment(clean(frame(rows)).frame)["segment"].unique().to_list()
    assert sorted(labels) == ["a-20240506T000000-1", "a-20240506T000000-2"]
