"""Encounters: two vessels close together, slow, away from port.

Adapted from GFW's pipe-encounters (commit c5cc27d, 2026-10-16:
``transforms/resample.py``, ``compute_adjacency.py``,
``compute_encounters.py``, ``merge_encounters.py``, ``merge_pipeline.py``)
and the filter in pipe-events ``encounter-events.sql.j2`` (commit 7976fc6,
2026-09-28); Apache-2.0, read 2026-10-24. Constants are literature-sourced
(GFW). Steps:

1. **Resample.** Each segment's positions within one UTC day are put on a
   10-minute grid aligned with midnight. A grid time between two positions
   under an hour apart is interpolated linearly; there is nothing across
   longer gaps. Before the day's first position and after its last, the
   track is dead-reckoned from that position's speed and course for up to
   an hour, never past the day's edges.
2. **Adjacency.** At each grid time, two segments of different vessels
   within 0.5 km are adjacent (a k-d tree over projected positions finds
   candidates; distance is then great-circle).
3. **Runs.** A pair's consecutive adjacent grid times within one UTC day
   form a run; one lasting an hour or more is a raw encounter. Its median
   implied speed pools both vessels' steps along the run.
4. **Port.** A raw encounter whose mean position is under 10 km from an
   anchorage, or on land, is dropped.
5. **Merge.** A pair of vessels' raw encounters less than 4 hours apart
   merge into one encounter (its speed and distance are medians of the
   raw ones' medians).
6. **Filter.** An encounter passes when it lasts more than 2 hours and
   its median speed is under 2 knots.

Differences from GFW, so that swapping the two vessels cannot change any
result: GFW takes the implied speed and start position from one vessel
of the pair and walks runs along that vessel's grid times, so a run can
bridge that vessel's own resampling gaps; here both vessels' steps are
pooled, positions are midpoints, and a run breaks at any grid time the
pair is not adjacent. GFW filters on the mean position's 0.01-degree cell;
here it is the mean position itself. A grid time is skipped for
dead-reckoning when its position has no speed or course. Pairs of
segments sharing one vessel ID are never paired.
"""

from dataclasses import dataclass
from datetime import timedelta

import numpy as np
import polars as pl
from scipy.spatial import cKDTree

from vessels.core.distance import PortDistance, ShoreDistance
from vessels.core.geo import LocalProjection, haversine_km

DETECTOR = "encounter"
MPS_TO_KNOTS = 1.94384
US = 1_000_000
DAY_US = 86_400 * US
# Grid times in different slices are kept this far apart in the k-d tree's
# third axis, so a neighbour search never pairs two slices.
SLICE_KM = 1e4
# Margin on the projected search radius: the flat projection is off by up
# to about 4% east-west over the box (``geo``).
SEARCH_MARGIN = 1.1


@dataclass(frozen=True)
class EncounterRules:  # pylint: disable=too-many-instance-attributes
    """Thresholds for encounters (GFW defaults).

    Attributes:
        max_km: Greatest distance between adjacent vessels.
        step_minutes: Resampling grid.
        max_gap_minutes: Longest gap interpolated across, and how far a
            track is dead-reckoned past its ends.
        min_raw_minutes: Least duration of a raw encounter.
        merge_hours: Raw encounters closer than this merge.
        min_hours: An encounter must last longer than this.
        max_median_kn: Its median speed must be under this.
        min_port_km: Least distance of a raw encounter from an anchorage.
    """

    max_km: float = 0.5
    step_minutes: int = 10
    max_gap_minutes: int = 60
    min_raw_minutes: float = 60.0
    merge_hours: float = 4.0
    min_hours: float = 2.0
    max_median_kn: float = 2.0
    min_port_km: float = 10.0


ENCOUNTER_SCHEMA = {
    "event_id": pl.String,
    "vessel_id": pl.String,
    "partner_id": pl.String,
    "transceiver": pl.String,
    "partner_transceiver": pl.String,
    "start": pl.Datetime("us", "UTC"),
    "end": pl.Datetime("us", "UTC"),
    "hours": pl.Float64,
    "runs": pl.UInt32,
    "median_km": pl.Float64,
    "median_kn": pl.Float64,
    "mean_lon": pl.Float64,
    "mean_lat": pl.Float64,
    "start_lon": pl.Float64,
    "start_lat": pl.Float64,
    "end_lon": pl.Float64,
    "end_lat": pl.Float64,
    "vessel_points": pl.Int64,
    "partner_points": pl.Int64,
}
SLICE_SCHEMA = {
    "segment": pl.String,
    "vessel_id": pl.String,
    "transceiver": pl.String,
    "slice": pl.Int64,
    "lon": pl.Float64,
    "lat": pl.Float64,
    "density": pl.Float64,
}


def _positions(segmented: pl.DataFrame) -> pl.DataFrame:
    """Segment positions in time order, one per segment and timestamp.

    Args:
        segmented: ``segment.segment`` output.

    Returns:
        pl.DataFrame: Positions with ``t`` (microseconds since the epoch),
        ``day`` (its UTC day start, same unit) and each segment's first
        ``transceiver``, sorted by segment and time.
    """
    rows = (
        segmented.filter(pl.col("segment").is_not_null())
        .sort("segment", "timestamp", maintain_order=True)
        .unique(["segment", "timestamp"], keep="first", maintain_order=True)
        .with_columns(t=pl.col("timestamp").dt.epoch("us"))
        .with_columns(
            day=pl.col("t") // DAY_US * DAY_US,
            transceiver=pl.col("transceiver").first().over("segment"),
        )
    )
    return rows.select(
        "segment", "vessel_id", "transceiver", "t", "day", "lon", "lat", "sog", "cog"
    )


def _interpolated(rows: pl.DataFrame, step: int, max_gap: int) -> pl.DataFrame:
    """Grid times between a segment-day's positions.

    A grid time belongs to the step ending at the first position at or
    after it, as GFW brackets it; the day's first grid time, if it falls on
    the first position, belongs to the first step.

    Args:
        rows: ``_positions`` output.
        step: Grid spacing, microseconds.
        max_gap: Steps this long or longer are not interpolated.

    Returns:
        pl.DataFrame: In ``SLICE_SCHEMA``.
    """
    group = ["segment", "day"]
    steps = rows.with_columns(
        t1=pl.col("t").shift(-1).over(group),
        lon1=pl.col("lon").shift(-1).over(group),
        lat1=pl.col("lat").shift(-1).over(group),
        first=pl.int_range(pl.len()).over(group) == 0,
    ).filter(pl.col("t1").is_not_null(), pl.col("t1") - pl.col("t") < max_gap)
    # Ceiling for the first step (its start is included), else the next
    # grid time strictly after the step's start.
    low = (
        pl.when("first")
        .then(-(-pl.col("t") // step))
        .otherwise(pl.col("t") // step + 1)
    )
    span = pl.col("t1") - pl.col("t")
    return (
        steps.with_columns(n=pl.int_ranges(low, pl.col("t1") // step + 1))
        .explode("n")
        .filter(pl.col("n").is_not_null())
        .with_columns(slice=pl.col("n") * step)
        .with_columns(mix=(pl.col("slice") - pl.col("t")) / span)
        .with_columns(
            lon=pl.col("lon1") * pl.col("mix") + pl.col("lon") * (1 - pl.col("mix")),
            lat=pl.col("lat1") * pl.col("mix") + pl.col("lat") * (1 - pl.col("mix")),
            density=pl.lit(float(step)) / pl.max_horizontal(span, pl.lit(step)),
        )
        .select(list(SLICE_SCHEMA))
    )


def _extrapolated(rows: pl.DataFrame, step: int, max_gap: int) -> pl.DataFrame:
    """Grid times dead-reckoned before a segment-day's first position and
    after its last.

    Args:
        rows: ``_positions`` output.
        step: Grid spacing, microseconds.
        max_gap: How far past each end, microseconds.

    Returns:
        pl.DataFrame: In ``SLICE_SCHEMA``.
    """
    group = ["segment", "day"]
    ends = rows.filter(pl.col("sog").is_not_null(), pl.col("cog").is_not_null())
    first = rows.group_by(group).agg(pl.all().first())
    last = rows.group_by(group).agg(pl.all().last())
    # Ends without speed or course cannot be dead-reckoned.
    first = first.join(ends.select("segment", "t"), on=["segment", "t"])
    last = last.join(ends.select("segment", "t"), on=["segment", "t"])
    before = first.with_columns(
        low=-(-pl.max_horizontal(pl.col("day"), pl.col("t") - max_gap) // step),
        high=-(-pl.col("t") // step) - 1,
    )
    after = last.with_columns(
        low=pl.col("t") // step + 1,
        high=pl.min_horizontal(pl.col("day") + DAY_US - 1, pl.col("t") + max_gap)
        // step,
    )
    both = pl.concat([before, after])
    if both.is_empty():
        return pl.DataFrame(schema=SLICE_SCHEMA)
    grid = (
        both.with_columns(n=pl.int_ranges("low", pl.col("high") + 1))
        .explode("n")
        .filter(pl.col("n").is_not_null())
        .with_columns(slice=pl.col("n") * step)
    )
    seconds = (grid["slice"] - grid["t"]).to_numpy() / US
    course = np.radians(grid["cog"].to_numpy())
    deg_per_s = grid["sog"].to_numpy() / 3600 / 60
    lat0 = grid["lat"].to_numpy()
    lat = lat0 + seconds * np.cos(course) * deg_per_s
    lon = grid["lon"].to_numpy() + seconds * np.sin(course) * deg_per_s / np.maximum(
        np.cos(np.radians(lat0)), 0.1
    )
    return grid.with_columns(
        lon=pl.Series((lon + 180) % 360 - 180),
        lat=pl.Series(lat),
        density=pl.lit(float(step))
        / pl.max_horizontal(pl.col("slice") - pl.col("t"), pl.lit(step)),
    ).select(list(SLICE_SCHEMA))


def resample(segmented: pl.DataFrame, rules: EncounterRules) -> pl.DataFrame:
    """Every segment's positions on the grid.

    Args:
        segmented: ``segment.segment`` output.
        rules: Thresholds.

    Returns:
        pl.DataFrame: One row per segment and grid time, in
        ``SLICE_SCHEMA`` (``slice`` in microseconds since the epoch;
        ``density`` is the grid spacing over the step, at most 1), sorted
        by segment and time.
    """
    step = rules.step_minutes * 60 * US
    max_gap = rules.max_gap_minutes * 60 * US
    rows = _positions(segmented)
    parts = [_interpolated(rows, step, max_gap), _extrapolated(rows, step, max_gap)]
    return (
        pl.concat(parts)
        .with_columns(density=pl.col("density").round(2))
        .sort("segment", "slice")
    )


def adjacent(
    slices: pl.DataFrame, projection: LocalProjection, rules: EncounterRules
) -> pl.DataFrame:
    """Pairs of segments of different vessels within ``max_km`` at the same
    grid time.

    Args:
        slices: ``resample`` output.
        projection: Flat projection about the study area.
        rules: Thresholds.

    Returns:
        pl.DataFrame: One row per pair and grid time, with ``slice``,
        ``km`` and, for each side (``_a``, ``_b``), segment, vessel,
        transceiver, lon, lat and density. Side ``a`` has the smaller
        vessel ID (then segment), so a pair is always the same way round.
    """
    sides = ["segment", "vessel_id", "transceiver", "lon", "lat", "density"]
    schema = {
        "slice": pl.Int64,
        "km": pl.Float64,
        **{f"{c}_{s}": slices.schema[c] for s in "ab" for c in sides},
    }
    if slices.height < 2:
        return pl.DataFrame(schema=schema)
    x, y = projection.forward(slices["lon"], slices["lat"])
    rank = slices["slice"].rank("dense").to_numpy().astype(float)
    tree = cKDTree(np.column_stack([x, y, rank * SLICE_KM]))
    pairs = tree.query_pairs(rules.max_km * SEARCH_MARGIN, output_type="ndarray")
    if pairs.size == 0:
        return pl.DataFrame(schema=schema)
    first, second = slices[pairs[:, 0]], slices[pairs[:, 1]]
    km = haversine_km(first["lon"], first["lat"], second["lon"], second["lat"])
    swap = (first["vessel_id"] > second["vessel_id"]) | (
        (first["vessel_id"] == second["vessel_id"])
        & (first["segment"] > second["segment"])
    )
    a = first.select(sides).with_columns(_swap=swap)
    b = second.select(sides)
    side_a = {c: pl.when("_swap").then(b[c]).otherwise(pl.col(c)) for c in sides}
    side_b = {c: pl.when("_swap").then(pl.col(c)).otherwise(b[c]) for c in sides}
    out = a.select(
        slice=first["slice"],
        km=pl.Series(km),
        **{f"{c}_a": e for c, e in side_a.items()},
        **{f"{c}_b": e for c, e in side_b.items()},
    )
    return (
        out.filter(
            pl.col("vessel_id_a") != pl.col("vessel_id_b"), pl.col("km") <= rules.max_km
        )
        .select(list(schema))
        .sort("segment_a", "segment_b", "slice")
    )


def raw_encounters(
    pairs: pl.DataFrame,
    ports: PortDistance,
    shore: ShoreDistance,
    rules: EncounterRules,
) -> pl.DataFrame:
    """Runs of adjacency an hour or longer, away from port.

    Args:
        pairs: ``adjacent`` output.
        ports: Distance to anchorages.
        shore: Signed distance to shore.
        rules: Thresholds.

    Returns:
        pl.DataFrame: One row per raw encounter: both sides' vessel,
        segment and transceiver, ``start`` and ``end`` (microseconds),
        median distance and speed, mean, start and end positions
        (midpoints), point counts, and ``near_port``.
    """
    step = rules.step_minutes * 60 * US
    key = ["segment_a", "segment_b"]
    mid_lon = (pl.col("lon_a") + pl.col("lon_b")) / 2
    mid_lat = (pl.col("lat_a") + pl.col("lat_b")) / 2
    pairs = pairs.with_columns(
        broke=(
            (pl.col("slice") - pl.col("slice").shift(1).over(key) != step)
            | (
                pl.col("slice") // DAY_US
                != (pl.col("slice").shift(1) // DAY_US).over(key)
            )
        ).fill_null(True),
        mid_lon=mid_lon,
        mid_lat=mid_lat,
    ).with_columns(run=pl.col("broke").cum_sum())

    # Implied speed of each side from the run's previous grid time, m/s.
    prev = {
        f"prev_{c}_{s}": pl.col(f"{c}_{s}").shift(1).over("run")
        for s in "ab"
        for c in ("lon", "lat")
    }
    pairs = pairs.with_columns(**prev)
    pairs = pairs.with_columns(
        **{
            f"speed_{s}": pl.Series(
                haversine_km(
                    pairs[f"prev_lon_{s}"],
                    pairs[f"prev_lat_{s}"],
                    pairs[f"lon_{s}"],
                    pairs[f"lat_{s}"],
                )
                * 1000
                / (step / US)
            ).fill_nan(None)
            for s in "ab"
        }
    )
    runs = pairs.group_by("run", maintain_order=True).agg(
        *[
            pl.col(f"{c}_{s}").first()
            for s in "ab"
            for c in ("vessel_id", "segment", "transceiver")
        ],
        start=pl.col("slice").first(),
        end=pl.col("slice").last(),
        slices=pl.len(),
        median_km=pl.col("km").median(),
        median_kn=pl.concat_list("speed_a", "speed_b").explode().drop_nulls().median()
        * MPS_TO_KNOTS,
        mean_lat=pl.col("mid_lat").mean(),
        cos_lon=pl.col("mid_lon").radians().cos().mean(),
        sin_lon=pl.col("mid_lon").radians().sin().mean(),
        start_lon=pl.col("mid_lon").first(),
        start_lat=pl.col("mid_lat").first(),
        end_lon=pl.col("mid_lon").last(),
        end_lat=pl.col("mid_lat").last(),
        points_a=pl.col("density_a").sum().round().cast(pl.Int64),
        points_b=pl.col("density_b").sum().round().cast(pl.Int64),
    )
    runs = runs.filter(
        pl.col("slices") >= 2,
        pl.col("end") - pl.col("start") >= rules.min_raw_minutes * 60 * US,
    ).with_columns(mean_lon=pl.arctan2("sin_lon", "cos_lon").degrees())
    port_km = ports.km(runs["mean_lon"], runs["mean_lat"])
    shore_km = shore.km(runs["mean_lon"], runs["mean_lat"])
    return runs.with_columns(
        near_port=pl.Series(
            (port_km < rules.min_port_km) | (shore_km <= 0), dtype=pl.Boolean
        )
    ).drop("run", "slices", "cos_lon", "sin_lon")


def merge(raw: pl.DataFrame, rules: EncounterRules) -> pl.DataFrame:
    """Merge a vessel pair's raw encounters under ``merge_hours`` apart.

    Args:
        raw: ``raw_encounters`` output without the ones near port.
        rules: Thresholds.

    Returns:
        pl.DataFrame: One row per encounter in ``ENCOUNTER_SCHEMA`` plus
        ``passes``, sorted by vessel, start and partner.
    """
    schema = {**ENCOUNTER_SCHEMA, "passes": pl.Boolean}
    if raw.is_empty():
        return pl.DataFrame(schema=schema)
    pair = ["vessel_id_a", "vessel_id_b"]
    raw = raw.sort(*pair, "start", "end", "segment_a", "segment_b").with_columns(
        weight=pl.max_horizontal(pl.col("end") - pl.col("start"), pl.lit(0)) / US
    )
    previous_end = pl.col("end").cum_max().shift(1).over(pair)
    raw = raw.with_columns(
        new=(pl.col("start") - previous_end >= rules.merge_hours * 3600 * US).fill_null(
            True
        )
    ).with_columns(group=pl.col("new").cum_sum())
    total = pl.max_horizontal(pl.col("weight").sum(), pl.lit(1.0))
    lon = pl.col("mean_lon").radians()
    merged = raw.group_by("group", maintain_order=True).agg(
        vessel_id=pl.col("vessel_id_a").first(),
        partner_id=pl.col("vessel_id_b").first(),
        transceiver=pl.col("transceiver_a").first(),
        partner_transceiver=pl.col("transceiver_b").first(),
        start=pl.col("start").min(),
        end=pl.col("end").max(),
        runs=pl.len(),
        median_km=pl.col("median_km").median(),
        median_kn=pl.col("median_kn").median(),
        mean_lat=(pl.col("weight") * pl.col("mean_lat")).sum() / total,
        cos_lon=(pl.col("weight") * lon.cos()).sum() / total,
        sin_lon=(pl.col("weight") * lon.sin()).sum() / total,
        start_lon=pl.col("start_lon").sort_by("start").first(),
        start_lat=pl.col("start_lat").sort_by("start").first(),
        end_lon=pl.col("end_lon").get(pl.col("end").arg_max()),
        end_lat=pl.col("end_lat").get(pl.col("end").arg_max()),
        vessel_points=pl.col("points_a").sum(),
        partner_points=pl.col("points_b").sum(),
    )
    merged = merged.with_columns(
        start=pl.from_epoch("start", "us").dt.replace_time_zone("UTC"),
        end=pl.from_epoch("end", "us").dt.replace_time_zone("UTC"),
        mean_lon=pl.arctan2("sin_lon", "cos_lon").degrees(),
    ).with_columns(
        hours=(pl.col("end") - pl.col("start")).dt.total_microseconds() / (3600 * US),
        event_id=pl.concat_str(
            "vessel_id",
            "partner_id",
            pl.lit(DETECTOR),
            pl.col("start").dt.strftime("%Y%m%dT%H%M%S"),
            separator=":",
        ),
    )
    return (
        merged.with_columns(
            passes=(pl.col("hours") > rules.min_hours)
            & (pl.col("median_kn") < rules.max_median_kn)
        )
        .select(pl.col(name).cast(dtype) for name, dtype in schema.items())
        .sort("vessel_id", "start", "partner_id")
    )


def encounter_groups(
    segmented: pl.DataFrame,
    ports: PortDistance,
    shore: ShoreDistance,
    rules: EncounterRules = EncounterRules(),
) -> pl.DataFrame:
    """Every merged encounter, whether or not it passes.

    Args:
        segmented: ``segment.segment`` output.
        ports: Distance to anchorages (its projection is reused).
        shore: Signed distance to shore.
        rules: Thresholds.

    Returns:
        pl.DataFrame: ``merge`` output.
    """
    pairs = adjacent(resample(segmented, rules), ports.projection, rules)
    raw = raw_encounters(pairs, ports, shore, rules)
    return merge(raw.filter(pl.col("near_port").not_()), rules)


def encounter_events(
    segmented: pl.DataFrame,
    ports: PortDistance,
    shore: ShoreDistance,
    rules: EncounterRules = EncounterRules(),
) -> pl.DataFrame:
    """Find encounters.

    Args:
        segmented: ``segment.segment`` output.
        ports: Distance to anchorages.
        shore: Signed distance to shore.
        rules: Thresholds.

    Returns:
        pl.DataFrame: The encounters that pass, in ``ENCOUNTER_SCHEMA``.
    """
    groups = encounter_groups(segmented, ports, shore, rules)
    return groups.filter("passes").select(list(ENCOUNTER_SCHEMA))


def settle_after(rules: EncounterRules) -> timedelta:
    """How long after an encounter's end no later position can change it.

    A raw encounter that starts within ``merge_hours`` of the end would
    still merge, but it exists only once it has lasted
    ``min_raw_minutes``, and a grid time is final only once the positions
    bracketing it, up to ``max_gap_minutes`` later, are released.
    Dead-reckoned grid times after a day's last position also vanish if a
    later position that day arrives, so the stream also waits for the
    following midnight.

    Args:
        rules: Thresholds.

    Returns:
        timedelta: The wait after the end.
    """
    return timedelta(
        hours=rules.merge_hours,
        minutes=rules.min_raw_minutes + rules.max_gap_minutes,
    )
