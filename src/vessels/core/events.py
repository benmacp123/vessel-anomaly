"""Turn a per-position state into events, by GFW's fishing-event rules.

Adapted from GFW's pipe-events (``fishing-events-1-incremental.sql.j2``
and ``fishing-events-3-filter.sql.j2``, Apache-2.0; constants
literature-sourced (GFW), read 2026-10-24). GFW's state is the fishing
model's score; here it is any boolean column, such as "towing inside this
closure". Within each track, in time order:

1. **Bursts.** Consecutive flagged positions form a burst. A burst ends at
   any unflagged position, and also when a flagged position is more than
   ``split_km`` or ``split_hours`` from the position before it.
2. **Merge.** A burst starting within ``merge_hours`` and ``merge_km`` of
   the previous burst's end joins its event, whatever lies between (GFW:
   "even if there are null or non-fishing scores between events"). Only
   flagged positions belong to the event.
3. **Filter.** An event is kept when it lasts more than ``min_minutes``,
   has more than ``min_positions`` positions, averages under
   ``max_avg_sog`` knots and covers more than ``min_km``. Its distance is
   the sum of each position's step from the position before it in the
   track, leaving out the first position's step, which leads into the
   event.

One difference from pipe-events: GFW measures the merge distance from the
previous event's first position to this burst's first position. The
parameter is described as the distance "between two fishing bursts", and a
two-hour tow at 3 kn ends about 11 km from where it started, so a long tow
interrupted for a few minutes would never merge. Here it is measured from
the previous burst's last position.
"""

from dataclasses import dataclass

import numpy as np
import polars as pl

from vessels.core.geo import haversine_km


@dataclass(frozen=True)
class EventRules:  # pylint: disable=too-many-instance-attributes
    """Thresholds for building and keeping events.

    Defaults are pipe-events' (``DEFAULT_MAX_FISHING_EVENT_*`` and
    ``DEFAULT_MIN_EVENT_*``); literature-sourced (GFW).

    Attributes:
        split_km: Step length that ends a burst.
        split_hours: Step duration that ends a burst.
        merge_km: Largest distance between bursts that still merge.
        merge_hours: Largest time between bursts that still merge.
        min_minutes: An event must last longer than this.
        min_positions: An event must have more positions than this.
        max_avg_sog: An event's mean reported speed must be under this.
        min_km: An event must cover more than this.
    """

    split_km: float = 10.0
    split_hours: float = 2.0
    merge_km: float = 2.0
    merge_hours: float = 1.0
    min_minutes: float = 20.0
    min_positions: int = 5
    max_avg_sog: float = 10.0
    min_km: float = 0.5


def steps(frame: pl.DataFrame, track: str) -> pl.DataFrame:
    """Add each position's step from the position before it in its track.

    Args:
        frame: Positions sorted by ``track`` then ``timestamp``, with
            ``lon`` and ``lat``.
        track: Column naming the track (a segment, or a vessel).

    Returns:
        pl.DataFrame: ``frame`` plus ``km_to_prev`` and ``hours_to_prev``,
        null at each track's first position.
    """
    first = pl.col(track).ne(pl.col(track).shift(1)).fill_null(True)
    km = haversine_km(
        frame["lon"].shift(1).to_numpy(allow_copy=True),
        frame["lat"].shift(1).to_numpy(allow_copy=True),
        frame["lon"].to_numpy(),
        frame["lat"].to_numpy(),
    )
    return frame.with_columns(
        km_to_prev=pl.when(first).then(None).otherwise(pl.Series(km)),
        hours_to_prev=pl.when(first)
        .then(None)
        .otherwise(pl.col("timestamp").diff().dt.total_microseconds() / 3.6e9),
    )


def flag_events(
    frame: pl.DataFrame, flag: str, track: str, rules: EventRules = EventRules()
) -> pl.DataFrame:
    """Group a track's flagged positions into events and keep the lasting
    ones.

    Args:
        frame: ``steps`` output with a boolean ``flag`` column and ``sog``.
        flag: Column that is True where the state holds.
        track: The track column ``steps`` used.
        rules: Event thresholds.

    Returns:
        pl.DataFrame: ``chains`` output for the events that pass, without
        ``passes``.
    """
    found = chains(frame, flag, track, rules)
    return found.filter("passes").drop("passes")


def chains(
    frame: pl.DataFrame, flag: str, track: str, rules: EventRules = EventRules()
) -> pl.DataFrame:
    """Group a track's flagged positions into events, kept or not.

    A stream needs the events that do not pass yet, since they may still
    grow into ones that do.

    Args:
        frame: ``steps`` output with a boolean ``flag`` column and ``sog``.
        flag: Column that is True where the state holds.
        track: The track column ``steps`` used.
        rules: Event thresholds.

    Returns:
        pl.DataFrame: One row per event: ``track``, ``event`` (a number,
        unique within the frame), ``start``, ``end``, ``hours``,
        ``positions``, ``distance_km``, ``avg_sog``, ``start_lon``,
        ``start_lat``, ``end_lon``, ``end_lat`` and ``passes`` (meets every
        filter), ordered by track and start.
    """
    same_track = pl.col(track).eq(pl.col(track).shift(1)).fill_null(False)
    prev_flag = pl.col(flag).shift(1).fill_null(False) & same_track
    starts = pl.col(flag) & (
        prev_flag.not_()
        | pl.col("km_to_prev").gt(rules.split_km).fill_null(False)
        | pl.col("hours_to_prev").gt(rules.split_hours).fill_null(False)
    )
    flagged = frame.with_columns(burst=starts.cum_sum()).filter(pl.col(flag))
    bursts = flagged.group_by(track, "burst", maintain_order=True).agg(
        start=pl.col("timestamp").first(),
        end=pl.col("timestamp").last(),
        start_lon=pl.col("lon").first(),
        start_lat=pl.col("lat").first(),
        end_lon=pl.col("lon").last(),
        end_lat=pl.col("lat").last(),
    )
    gap_km = haversine_km(
        bursts["end_lon"].shift(1).to_numpy(allow_copy=True),
        bursts["end_lat"].shift(1).to_numpy(allow_copy=True),
        bursts["start_lon"].to_numpy(),
        bursts["start_lat"].to_numpy(),
    )
    same = pl.col(track).eq(pl.col(track).shift(1)).fill_null(False)
    gap_hours = (pl.col("start") - pl.col("end").shift(1)).dt.total_microseconds()
    merged = same & (gap_hours / 3.6e9).le(rules.merge_hours).fill_null(False)
    merged = merged & pl.Series(np.nan_to_num(gap_km, nan=np.inf)).le(rules.merge_km)
    events = bursts.with_columns(event=merged.not_().cum_sum()).select("burst", "event")
    # Each event's first burst starts with a lead-in step; later bursts'
    # first steps lie inside the event, after the gap they bridge.
    flagged = flagged.join(events, on="burst", maintain_order="left").with_columns(
        inside_km=pl.when(pl.col("event").is_first_distinct())
        .then(0.0)
        .otherwise(pl.col("km_to_prev").fill_null(0.0))
    )
    summary = flagged.group_by(track, "event", maintain_order=True).agg(
        start=pl.col("timestamp").first(),
        end=pl.col("timestamp").last(),
        positions=pl.len(),
        distance_km=pl.col("inside_km").sum(),
        avg_sog=pl.col("sog").mean(),
        start_lon=pl.col("lon").first(),
        start_lat=pl.col("lat").first(),
        end_lon=pl.col("lon").last(),
        end_lat=pl.col("lat").last(),
    )
    summary = summary.with_columns(
        hours=(pl.col("end") - pl.col("start")).dt.total_microseconds() / 3.6e9
    )
    passes = (
        pl.col("hours").gt(rules.min_minutes / 60)
        & pl.col("positions").gt(rules.min_positions)
        & pl.col("avg_sog").lt(rules.max_avg_sog)
        & pl.col("distance_km").gt(rules.min_km)
    )
    return summary.with_columns(passes=passes).select(
        track,
        "event",
        "start",
        "end",
        "hours",
        "positions",
        "distance_km",
        "avg_sog",
        "start_lon",
        "start_lat",
        "end_lon",
        "end_lat",
        "passes",
    )
