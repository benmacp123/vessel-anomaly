"""Loitering: hours spent slow, far from shore.

Adapted from GFW's pipe-loitering (``calculate_hourly_stats.py``,
``group_loitering_ranges.py``, ``calculate_loitering_stats.py``) and the
filter in pipe-events ``loitering-events.sql.j2`` (Apache-2.0; read
2026-10-24). Along each segment:

1. Positions fall into UTC clock-hour buckets. A bucket is slow when its
   implied speed (the steps ending in it, distance over time) is under
   ``slow_kn``. A bucket with no step time is not slow.
2. Consecutive slow buckets form one event; an hour with no positions
   does not break it. The event also takes the previous bucket's last
   position, so it starts where the vessel slowed down.
3. An event's duration, distance and speed come from the steps after its
   first position. Its distance from shore is the mean over its positions
   weighted by each position's step time, as GFW computes it.
4. Events whose mean distance from shore is under ``min_shore_nm`` (20 nm; Miller et al. 2018,
   literature-sourced) or that last under ``min_hours`` are dropped. GFW's
   pipeline leaves duration to the caller; 8 hours (Miller et al. 2018) is
   the headline and 4 hours the sensitivity.

GFW also drops events of one position; none can arise here, since a slow
bucket holds a step and so the position before it. Anchorages are not
excluded separately: every US anchorage lies within
5 km of shore (``distance.MAX_SHORE_KM``), well inside 20 nm. Only fishing
types are in the extract and carriers are out of scope, so loitering here
mostly finds vessels drifting or hauling static gear; it is no evidence of
transshipment.
"""

from dataclasses import dataclass

import polars as pl

from vessels.core.distance import ShoreDistance
from vessels.core.events import steps
from vessels.core.geo import KM_PER_NM

DETECTOR = "loitering"


@dataclass(frozen=True)
class LoiterRules:
    """Thresholds for loitering.

    Attributes:
        slow_kn: An hour is slow under this implied speed (GFW default).
        min_shore_nm: Least mean distance from shore (Miller et al. 2018).
        min_hours: Least duration (Miller et al. 2018: 8 hours).
    """

    slow_kn: float = 2.0
    min_shore_nm: float = 20.0
    min_hours: float = 8.0


LOITER_SCHEMA = {
    "event_id": pl.String,
    "vessel_id": pl.String,
    "segment": pl.String,
    "transceiver": pl.String,
    "start": pl.Datetime("us", "UTC"),
    "end": pl.Datetime("us", "UTC"),
    "hours": pl.Float64,
    "positions": pl.UInt32,
    "distance_km": pl.Float64,
    "avg_kn": pl.Float64,
    "avg_shore_nm": pl.Float64,
    "start_lon": pl.Float64,
    "start_lat": pl.Float64,
    "end_lon": pl.Float64,
    "end_lat": pl.Float64,
}


def hourly_buckets(
    segmented: pl.DataFrame, slow_kn: float
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Each segment's positions and its clock-hour buckets.

    Args:
        segmented: ``segment.segment`` output.
        slow_kn: Slow-bucket speed threshold.

    Returns:
        tuple[pl.DataFrame, pl.DataFrame]: The positions with ``steps``
        columns, ``_row`` and ``bucket``; and one row per segment and
        bucket with ``hours``, ``km``, ``last_row``, ``last_time`` and
        ``slow``, in time order.
    """
    tracks = (
        segmented.filter(pl.col("segment").is_not_null())
        .sort("segment", "timestamp", maintain_order=True)
        .pipe(steps, "segment")
        .with_row_index("_row")
        .with_columns(bucket=pl.col("timestamp").dt.truncate("1h"))
    )
    buckets = (
        tracks.group_by("segment", "bucket", maintain_order=True)
        .agg(
            hours=pl.col("hours_to_prev").sum(),
            km=pl.col("km_to_prev").sum(),
            last_row=pl.col("_row").last(),
            last_time=pl.col("timestamp").last(),
        )
        .with_columns(
            slow=(pl.col("hours") > 0)
            & (pl.col("km") / KM_PER_NM / pl.col("hours") < slow_kn)
        )
    )
    return tracks, buckets


def slow_groups(tracks: pl.DataFrame, buckets: pl.DataFrame) -> pl.DataFrame:
    """Each slow run's positions, including the lead-in position.

    Args:
        tracks: Positions from ``hourly_buckets``.
        buckets: Buckets from ``hourly_buckets``.

    Returns:
        pl.DataFrame: Positions with segment, ``steps`` columns and
        ``group`` (one number per slow run), sorted by group and time.
    """
    buckets = buckets.with_columns(
        same=pl.col("segment").eq(pl.col("segment").shift(1)).fill_null(False),
        prev_last=pl.col("last_row").shift(1),
    )
    first = pl.col("slow") & (
        pl.col("slow").shift(1).fill_null(False).not_() | pl.col("same").not_()
    )
    buckets = buckets.with_columns(first=first, group=first.cum_sum()).filter("slow")
    members = tracks.join(
        buckets.select("segment", "bucket", "group"), on=["segment", "bucket"]
    )
    # The bucket before a run lends its last position as the run's start.
    lead = buckets.filter(pl.col("first") & pl.col("same")).select(
        _row="prev_last", group="group"
    )
    leads = tracks.join(lead, on="_row")
    return pl.concat([members, leads.select(members.columns)]).sort("group", "_row")


def loiter_events(
    segmented: pl.DataFrame, shore: ShoreDistance, rules: LoiterRules = LoiterRules()
) -> pl.DataFrame:
    """Find loitering events.

    Args:
        segmented: ``segment.segment`` output.
        shore: Signed distance to shore.
        rules: Thresholds.

    Returns:
        pl.DataFrame: The ``loiter_groups`` that pass, in ``LOITER_SCHEMA``.
    """
    groups, _ = loiter_groups(segmented, shore, rules)
    return groups.filter("passes").select(list(LOITER_SCHEMA))


def loiter_groups(
    segmented: pl.DataFrame, shore: ShoreDistance, rules: LoiterRules = LoiterRules()
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Every slow run, whether or not it passes, and the buckets behind it.

    Args:
        segmented: ``segment.segment`` output.
        shore: Signed distance to shore.
        rules: Thresholds.

    Returns:
        tuple[pl.DataFrame, pl.DataFrame]: One row per run in
        ``LOITER_SCHEMA`` plus ``passes``, ordered by vessel and start
        (``event_id`` is the segment and start time); and the
        ``hourly_buckets`` buckets, which a stream needs to tell when a run
        can no longer grow.
    """
    tracks, buckets = hourly_buckets(segmented, rules.slow_kn)
    schema = {**LOITER_SCHEMA, "passes": pl.Boolean}
    groups = slow_groups(tracks, buckets)
    if groups.is_empty():
        return pl.DataFrame(schema=schema), buckets
    groups = groups.with_columns(
        shore_km=pl.Series(shore.km(groups["lon"], groups["lat"])),
        after_first=pl.int_range(pl.len()).over("group") > 0,
    )
    # GFW weights by every position's step time, the first one's too; with
    # no step time at all it falls back to a plain mean.
    weight = pl.col("hours_to_prev").fill_null(0.0)
    weighted = (pl.col("shore_km") * weight).sum() / weight.sum()
    events = groups.group_by("group", maintain_order=True).agg(
        pl.col("segment").first(),
        pl.col("vessel_id").first(),
        pl.col("transceiver").first(),
        start=pl.col("timestamp").first(),
        end=pl.col("timestamp").last(),
        positions=pl.len(),
        hours=pl.col("hours_to_prev").filter("after_first").sum(),
        distance_km=pl.col("km_to_prev").filter("after_first").sum(),
        avg_shore_km=pl.when(weight.sum() > 0)
        .then(weighted)
        .otherwise(pl.col("shore_km").mean()),
        start_lon=pl.col("lon").first(),
        start_lat=pl.col("lat").first(),
        end_lon=pl.col("lon").last(),
        end_lat=pl.col("lat").last(),
    )
    return (
        events.with_columns(
            avg_kn=pl.col("distance_km") / KM_PER_NM / pl.col("hours"),
            avg_shore_nm=pl.col("avg_shore_km") / KM_PER_NM,
            event_id=pl.concat_str(
                "segment",
                pl.lit(DETECTOR),
                pl.col("start").dt.strftime("%Y%m%dT%H%M%S"),
                separator=":",
            ),
        )
        .with_columns(
            passes=(pl.col("avg_shore_nm") >= rules.min_shore_nm)
            & (pl.col("hours") >= rules.min_hours)
        )
        .select(pl.col(name).cast(dtype) for name, dtype in schema.items())
        .sort("vessel_id", "start", "segment")
    ), buckets
